"""Temperature scaling, fitting, and the calibration file."""

import io
import json
import math
import os
import random
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from nex import calibration as calibration_module
from nex.calibration import (
    DEFAULT_PATH,
    T_MAX,
    T_MIN,
    Calibration,
    apply_temperature,
    at_bound,
    fit_temperature,
    negative_log_likelihood,
)


def without_override():
    """Patch os.environ for one test with NEX_CALIBRATION removed."""
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    os.environ.pop("NEX_CALIBRATION", None)
    return patcher


def softmax(logits):
    top = max(logits)
    weights = [math.exp(l - top) for l in logits]
    total = sum(weights)
    return [w / total for w in weights]


def sample(rng, probabilities):
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probabilities):
        acc += p
        if r < acc:
            return i
    return len(probabilities) - 1


def synthetic(true_temperature, n=1500, k=4, seed=7):
    """Model outputs whose truths are drawn from the tempered distribution, so
    the temperature that best explains them is ``true_temperature``."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        reported = softmax([rng.gauss(0, 2.0) for _ in range(k)])
        truth = sample(rng, apply_temperature(reported, true_temperature))
        samples.append((reported, truth))
    return samples


class ApplyTemperatureTest(unittest.TestCase):
    def test_identity_at_one(self):
        probs = [0.5, 0.3, 0.2]
        out = apply_temperature(probs, 1.0)
        self.assertEqual(out, probs)
        self.assertIsNot(out, probs)
        self.assertEqual(apply_temperature(probs, 1), probs)

    def test_known_value(self):
        # sqrt(0.8) : sqrt(0.2) is 2 : 1.
        out = apply_temperature([0.8, 0.2], 2.0)
        self.assertAlmostEqual(out[0], 2 / 3)
        self.assertAlmostEqual(out[1], 1 / 3)
        out = apply_temperature([2 / 3, 1 / 3], 0.5)
        self.assertAlmostEqual(out[0], 0.8)

    def test_high_temperature_flattens(self):
        probs = [0.7, 0.2, 0.1]
        out = apply_temperature(probs, 2.0)
        self.assertLess(out[0], probs[0])
        self.assertGreater(out[2], probs[2])
        flatter = apply_temperature(probs, 10.0)
        self.assertLess(flatter[0], out[0])
        self.assertLess(max(flatter) - min(flatter), max(out) - min(out))

    def test_low_temperature_sharpens(self):
        probs = [0.7, 0.2, 0.1]
        out = apply_temperature(probs, 0.5)
        self.assertGreater(out[0], probs[0])
        self.assertLess(out[2], probs[2])
        self.assertGreater(apply_temperature(probs, 0.1)[0], out[0])

    def test_sums_to_one_and_keeps_order(self):
        rng = random.Random(1)
        for _ in range(200):
            probs = softmax([rng.gauss(0, 3) for _ in range(rng.randint(2, 20))])
            t = math.exp(rng.uniform(math.log(T_MIN), math.log(T_MAX)))
            out = apply_temperature(probs, t)
            self.assertAlmostEqual(sum(out), 1.0)
            self.assertEqual(probs.index(max(probs)), out.index(max(out)))
            order = sorted(range(len(probs)), key=probs.__getitem__)
            for a, b in zip(order, order[1:]):
                self.assertLessEqual(out[a], out[b] + 1e-15)

    def test_zeros_handled(self):
        out = apply_temperature([1.0, 0.0], 2.0)
        self.assertAlmostEqual(sum(out), 1.0)
        self.assertGreater(out[0], 0.99)
        out = apply_temperature([0.0, 0.0, 0.0], 0.5)
        for p in out:
            self.assertAlmostEqual(p, 1 / 3)
        out = apply_temperature([0.0, 0.5, 0.5], 0.05)
        self.assertAlmostEqual(out[1], 0.5)
        self.assertLess(out[0], 1e-100)

    def test_does_not_mutate_input(self):
        probs = [0.6, 0.4]
        apply_temperature(probs, 3.0)
        self.assertEqual(probs, [0.6, 0.4])


class NegativeLogLikelihoodTest(unittest.TestCase):
    def test_mean_of_true_slot_log_probs(self):
        samples = [([0.8, 0.2], 0), ([0.6, 0.4], 1)]
        self.assertAlmostEqual(negative_log_likelihood(samples, 1.0), -(math.log(0.8) + math.log(0.4)) / 2)

    def test_uses_temperature(self):
        samples = [([0.8, 0.2], 1)]
        self.assertAlmostEqual(negative_log_likelihood(samples, 2.0), -math.log(1 / 3))

    def test_zero_probability_truth_is_finite(self):
        self.assertTrue(math.isfinite(negative_log_likelihood([([1.0, 0.0], 1)], 1.0)))


class FitTemperatureTest(unittest.TestCase):
    def assertRecovers(self, true_temperature, **kwargs):
        fitted = fit_temperature(synthetic(true_temperature, **kwargs))
        self.assertLess(abs(fitted - true_temperature) / true_temperature, 0.15, (true_temperature, fitted))

    def test_recovers_overconfident_model(self):
        self.assertRecovers(2.5)

    def test_recovers_underconfident_model(self):
        self.assertRecovers(0.5)

    def test_recovers_well_calibrated_model(self):
        self.assertRecovers(1.0, seed=11)

    def test_recovers_with_two_slots(self):
        self.assertRecovers(1.8, k=2, n=3000, seed=3)

    def test_fit_beats_neighbouring_temperatures(self):
        samples = synthetic(2.0)
        fitted = fit_temperature(samples)
        best = negative_log_likelihood(samples, fitted)
        for t in [fitted * 0.8, fitted * 1.25, 1.0]:
            self.assertLessEqual(best, negative_log_likelihood(samples, t) + 1e-9)

    def test_empty_input(self):
        self.assertEqual(fit_temperature([]), 1.0)

    def test_always_right_hits_lower_bound(self):
        samples = [([0.7, 0.3], 0), ([0.2, 0.8], 1)] * 20
        fitted = fit_temperature(samples)
        self.assertGreaterEqual(fitted, T_MIN)
        self.assertLess(fitted, T_MIN * 1.05)

    def test_always_right_with_high_confidence_hits_lower_bound(self):
        # NLL rounds to exactly 0 over a stretch of low T here, so the
        # search must not stop at the edge of that stretch.
        for raw in ([0.9, 0.1], [0.999, 0.001], [1.0, 0.0, 0.0]):
            fitted = fit_temperature([(raw, 0)] * 40)
            self.assertLess(fitted, T_MIN * 1.05, raw)
            self.assertTrue(at_bound(fitted), raw)

    def test_always_wrong_hits_upper_bound(self):
        samples = [([0.7, 0.3], 1), ([0.2, 0.8], 0)] * 20
        fitted = fit_temperature(samples)
        self.assertLessEqual(fitted, T_MAX)
        self.assertGreater(fitted, T_MAX * 0.95)

    def test_one_hot_samples_stay_in_bounds(self):
        fitted = fit_temperature([([1.0, 0.0, 0.0], 2), ([1.0, 0.0, 0.0], 0)])
        self.assertGreaterEqual(fitted, T_MIN)
        self.assertLessEqual(fitted, T_MAX)

    def test_at_bound(self):
        self.assertTrue(at_bound(fit_temperature([([0.7, 0.3], 0), ([0.2, 0.8], 1)] * 20)))
        self.assertTrue(at_bound(fit_temperature([([0.7, 0.3], 1), ([0.2, 0.8], 0)] * 20)))
        self.assertFalse(at_bound(fit_temperature(synthetic(2.5))))
        for t in (T_MIN, T_MIN * 1.04, T_MAX * 0.96, T_MAX):
            self.assertTrue(at_bound(t), t)
        for t in (T_MIN * 1.06, 1.0, T_MAX * 0.94):
            self.assertFalse(at_bound(t), t)


class CalibrationTest(unittest.TestCase):
    def test_default_temperature(self):
        c = Calibration()
        self.assertEqual(c.temperature("qwen3:4b-instruct", "choice"), 1.0)
        c.set("qwen3:4b-instruct", "choice", 1.7, n=40)
        self.assertEqual(c.temperature("qwen3:4b-instruct", "noul"), 1.0)
        self.assertEqual(c.temperature("other", "choice"), 1.0)

    def test_default_path(self):
        self.addCleanup(without_override().stop)
        self.assertEqual(Calibration().path, DEFAULT_PATH)
        self.assertEqual(DEFAULT_PATH.name, "calibration.json")
        self.assertEqual(DEFAULT_PATH.parent, Path(calibration_module.__file__).parent)
        os.environ["NEX_CALIBRATION"] = ""
        self.assertEqual(Calibration().path, DEFAULT_PATH)

    def test_nex_calibration_overrides_the_default(self):
        self.addCleanup(without_override().stop)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mine.json"
            Calibration({"m": {"noul": {"temperature": 2.5, "n": 40}}}, path).save()
            os.environ["NEX_CALIBRATION"] = str(path)
            self.assertEqual(Calibration().path, path)
            loaded = Calibration.load()
            self.assertEqual(loaded.path, path)
            self.assertEqual(loaded.temperature("m", "noul"), 2.5)

    def test_set_and_read(self):
        c = Calibration()
        c.set("m", "score", 1.234567, n=12)
        self.assertEqual(c.temperature("m", "score"), 1.2346)
        self.assertEqual(c.models, {"m": {"score": {"temperature": 1.2346, "n": 12}}})
        c.set("m", "score", 0.8, n=30)
        self.assertEqual(c.models["m"]["score"], {"temperature": 0.8, "n": 30})

    def test_set_rejects_unknown_types(self):
        c = Calibration()
        for kind in ["boolean", "Choice", "", None]:
            with self.assertRaises(ValueError):
                c.set("m", kind, 1.5, n=1)
        self.assertEqual(c.models, {})

    def test_constructor_models(self):
        c = Calibration({"m": {"noul": {"temperature": 2.0, "n": 5}}})
        self.assertEqual(c.temperature("m", "noul"), 2.0)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            c = Calibration(path=path)
            c.set("qwen3:4b-instruct", "choice", 1.5, n=100)
            c.set("qwen3:4b-instruct", "noul", 2.25, n=80)
            c.set("qwen3.5:9b", "score", 0.75, n=60)
            c.save()
            data = json.loads(path.read_text())
            self.assertEqual(set(data), {"models"})
            loaded = Calibration.load(path)
            self.assertEqual(loaded.models, c.models)
            self.assertEqual(loaded.path, path)
            self.assertEqual(loaded.temperature("qwen3:4b-instruct", "noul"), 2.25)
            self.assertEqual(loaded.temperature("qwen3.5:9b", "score"), 0.75)
            self.assertEqual(loaded.temperature("qwen3.5:9b", "choice"), 1.0)

    def test_save_to_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Calibration(path=Path(tmp) / "a.json")
            c.set("m", "score", 3.0, n=2)
            other = Path(tmp) / "b.json"
            c.save(str(other))
            self.assertFalse((Path(tmp) / "a.json").exists())
            self.assertEqual(Calibration.load(str(other)).temperature("m", "score"), 3.0)

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            with redirect_stderr(io.StringIO()) as err:
                c = Calibration.load(path)
            self.assertEqual(err.getvalue(), "")
            self.assertEqual(c.models, {})
            self.assertEqual(c.path, path)
            self.assertEqual(c.temperature("m", "choice"), 1.0)
            c.set("m", "choice", 1.1, n=3)
            c.save()
            self.assertTrue(path.exists())

    def test_missing_default_file_is_silent(self):
        self.addCleanup(without_override().stop)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            with mock.patch.object(calibration_module, "DEFAULT_PATH", path), redirect_stderr(io.StringIO()) as err:
                c = Calibration.load()
            self.assertEqual(err.getvalue(), "")
            self.assertEqual((c.models, c.path), ({}, path))

    def test_missing_override_file_warns_once(self):
        self.addCleanup(without_override().stop)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "typo.json"
            os.environ["NEX_CALIBRATION"] = str(path)
            with redirect_stderr(io.StringIO()) as err:
                first = Calibration.load()
                second = Calibration.load()
            self.assertEqual(first.models, {})
            self.assertEqual(second.temperature("m", "score"), 1.0)
            lines = err.getvalue().splitlines()
            self.assertEqual(len(lines), 1, lines)
            self.assertIn(str(path), lines[0])
            self.assertIn("running uncalibrated", lines[0])

    def test_malformed_file_names_the_file(self):
        bad = {
            "{not json": "invalid JSON",
            "[1, 2]": "JSON object",
            '"models"': "JSON object",
            '{"models": [1]}': "models must be an object",
            '{"models": {"m": 1.5}}': "'m'",
            '{"models": {"m": {"noul": 1.5}}}': "positive temperature",
            '{"models": {"m": {"noul": {"temperature": "2"}}}}': "positive temperature",
            '{"models": {"m": {"noul": {"temperature": 0}}}}': "positive temperature",
            '{"models": {"m": {"noul": {"temperature": true}}}}': "positive temperature",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            for text, fragment in bad.items():
                path.write_text(text)
                with self.assertRaises(ValueError, msg=text) as ctx:
                    Calibration.load(path)
                message = str(ctx.exception)
                self.assertTrue(message.startswith(f"calibration file {path}: "), message)
                self.assertIn(fragment, message)
            path.write_bytes(b"\xff\xfe")
            with self.assertRaises(ValueError):
                Calibration.load(path)
            path.write_text("{}")
            self.assertEqual(Calibration.load(path).models, {})


if __name__ == "__main__":
    unittest.main()
