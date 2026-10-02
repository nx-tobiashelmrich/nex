import json
import math
import tempfile
import unittest
import zlib
from pathlib import Path

from nex import Calibration, FakeBackend, Nex
from nex import evaluation
from nex.backend import BackendError
from nex.calibration import fit_temperature
from nex.evaluation import (
    SCORE_MAE_TOLERANCE,
    TypeFit,
    collect,
    cross_fit,
    fit_temperatures,
    fit_type,
    fold_of,
    load_cases,
    metrics,
    percentile,
    score_mae,
    truth_index,
)
from nex.primitives import Choice, Noul, Score

DATA = Path(__file__).resolve().parent.parent / "evals" / "data"


def choice_case(case_id, label="billing", **extra):
    case = {
        "id": case_id,
        "domain": "support",
        "difficulty": "easy",
        "state": "I was charged twice",
        "question": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": None, "technical": "Bugs and outages", "sales": None},
        },
        "label": label,
    }
    case.update(extra)
    return case


def score_case(case_id, label=1):
    return {
        "id": case_id,
        "domain": "reviews",
        "difficulty": "medium",
        "state": {"review": "It was fine"},
        "question": {"type": "score", "instructions": "How positive?", "criteria": ["bad", "ok", "great"]},
        "label": label,
    }


def noul_case(case_id, label=True):
    return {
        "id": case_id,
        "domain": "support",
        "difficulty": "hard",
        "state": ["line one", "line two"],
        "question": {"type": "noul", "instructions": "Is this about billing?"},
        "label": label,
    }


def write_jsonl(path, cases, blank_lines=False):
    lines = []
    for case in cases:
        lines.append(case if isinstance(case, str) else json.dumps(case))
        if blank_lines:
            lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def record(rtype, raw, truth, latency_ms=10.0, label_mass=1.0, case_id=None):
    return {
        "id": case_id or f"{rtype}-{raw}-{truth}",
        "type": rtype,
        "domain": "d",
        "difficulty": "easy",
        "raw": raw,
        "truth": truth,
        "label_mass": label_mass,
        "latency_ms": latency_ms,
        "prompt_tokens": 0,
        "cached_tokens": 0,
    }


class LoadCasesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assertLoadFails(self, cases, *fragments):
        path = self.dir / "bad.jsonl"
        write_jsonl(path, cases)
        with self.assertRaises(ValueError) as ctx:
            load_cases(path)
        message = str(ctx.exception)
        for fragment in ("bad.jsonl", *fragments):
            self.assertIn(fragment, message)
        return message

    def test_directory_loads_every_jsonl_in_sorted_order(self):
        write_jsonl(self.dir / "score.jsonl", [score_case("s1"), score_case("s2", 0)])
        write_jsonl(self.dir / "choice.jsonl", [choice_case("c1")], blank_lines=True)
        write_jsonl(self.dir / "noul.jsonl", [noul_case("n1", False)])
        (self.dir / "notes.txt").write_text("not data")
        cases = load_cases(self.dir)
        self.assertEqual([c["id"] for c in cases], ["c1", "n1", "s1", "s2"])
        self.assertEqual(cases[0]["question"]["type"], "choice")
        self.assertEqual(cases[1]["label"], False)

    def test_accepts_str_path_and_list_of_paths(self):
        write_jsonl(self.dir / "a.jsonl", [choice_case("c1")])
        write_jsonl(self.dir / "b.jsonl", [noul_case("n1")])
        self.assertEqual(len(load_cases(str(self.dir / "a.jsonl"))), 1)
        cases = load_cases([self.dir / "b.jsonl", self.dir / "a.jsonl"])
        self.assertEqual([c["id"] for c in cases], ["n1", "c1"])

    def test_invalid_json_names_file_and_line(self):
        self.assertLoadFails([choice_case("c1"), "{not json"], "bad.jsonl:2", "invalid JSON")

    def test_invalid_question_names_id(self):
        case = choice_case("c1")
        case["question"]["criteria"] = {"billing": None}
        self.assertLoadFails([case], ":1", "'c1'", "criteria")

    def test_choice_label_must_be_an_option(self):
        self.assertLoadFails([choice_case("c1", label="refunds")], "'c1'", "refunds")

    def test_score_label_must_be_an_int_level(self):
        self.assertLoadFails([score_case("s1", label=3)], "'s1'", "0 to 2")
        self.assertLoadFails([score_case("s1", label=-1)], "'s1'")
        self.assertLoadFails([score_case("s1", label="1")], "'s1'")
        self.assertLoadFails([score_case("s1", label=1.0)], "'s1'")
        self.assertLoadFails([score_case("s1", label=True)], "'s1'")

    def test_noul_label_must_be_bool(self):
        self.assertLoadFails([noul_case("n1", label="yes")], "'n1'", "true or false")
        self.assertLoadFails([noul_case("n1", label=1)], "'n1'")

    def test_missing_fields_and_bad_values(self):
        case = choice_case("c1")
        del case["label"]
        self.assertLoadFails([case], "missing label")
        self.assertLoadFails([choice_case("c1", difficulty="extreme")], "difficulty")
        self.assertLoadFails([choice_case("", label="billing")], "id")
        self.assertLoadFails([choice_case("c1", state=None)], "state")
        self.assertLoadFails([choice_case("c1", state=42)], "state")
        self.assertLoadFails(["[1, 2]"], "object")

    def test_duplicate_ids_fail_across_files(self):
        write_jsonl(self.dir / "a.jsonl", [choice_case("dup")])
        write_jsonl(self.dir / "b.jsonl", [noul_case("dup")])
        with self.assertRaises(ValueError) as ctx:
            load_cases(self.dir)
        message = str(ctx.exception)
        self.assertIn("duplicate", message)
        self.assertIn("b.jsonl:1", message)
        self.assertIn("a.jsonl:1", message)

    def test_missing_path_and_empty_directory(self):
        with self.assertRaises(ValueError):
            load_cases(self.dir / "nope.jsonl")
        with self.assertRaises(ValueError):
            load_cases(self.dir)

    def test_shipped_dataset(self):
        # Resolved from this file, so the test passes from any working directory.
        cases = load_cases(DATA)
        counts = {}
        for case in cases:
            counts[case["question"]["type"]] = counts.get(case["question"]["type"], 0) + 1
        self.assertEqual(counts, {"choice": 60, "score": 50, "noul": 60})


class TruthIndexTest(unittest.TestCase):
    def test_choice_uses_criteria_order(self):
        q = Choice("Which?", {"x": None, "y": None, "z": None})
        self.assertEqual(truth_index(q, "x"), 0)
        self.assertEqual(truth_index(q, "z"), 2)

    def test_score_is_the_level(self):
        self.assertEqual(truth_index(Score("How much?", ["a", "b", "c"]), 2), 2)

    def test_noul_yes_is_slot_zero(self):
        q = Noul("True?")
        self.assertEqual(truth_index(q, True), 0)
        self.assertEqual(truth_index(q, False), 1)

    def test_accepts_wire_format_dicts(self):
        self.assertEqual(truth_index({"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}, "b"), 1)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("noul", [0.8, 0.2], 0, latency_ms=10, label_mass=1.0),
            record("noul", [0.6, 0.4], 1, latency_ms=20, label_mass=0.9),
            record("choice", [0.5, 0.3, 0.2], 0, latency_ms=30, label_mass=0.8),
            record("choice", [0.1, 0.7, 0.2], 2, latency_ms=40, label_mass=1.0),
            record("score", [0.1, 0.2, 0.7], 2, latency_ms=50, label_mass=1.0),
            record("score", [0.25, 0.25, 0.5], 0, latency_ms=60, label_mass=0.7),
        ]

    def test_per_type_numbers(self):
        m = metrics(self.records)
        self.assertEqual(list(m), ["choice", "score", "noul", "all"])

        noul = m["noul"]
        self.assertEqual(noul["n"], 2)
        self.assertEqual(noul["temperature"], 1.0)
        self.assertAlmostEqual(noul["accuracy"], 0.5)
        self.assertAlmostEqual(noul["nll"], -math.log(0.8 * 0.4) / 2)
        # (0.04 + 0.04) and (0.36 + 0.36)
        self.assertAlmostEqual(noul["brier"], (0.08 + 0.72) / 2)
        # bin 8 holds a hit at 0.8, bin 6 a miss at 0.6
        self.assertAlmostEqual(noul["ece"], (0.2 + 0.6) / 2)
        self.assertAlmostEqual(noul["mean_confidence"], 0.7)
        self.assertAlmostEqual(noul["label_mass_mean"], 0.95)
        self.assertNotIn("mae", noul)

        choice = m["choice"]
        self.assertAlmostEqual(choice["accuracy"], 0.5)
        self.assertAlmostEqual(choice["nll"], -math.log(0.5 * 0.2) / 2)
        # (0.25 + 0.09 + 0.04) and (0.01 + 0.49 + 0.64)
        self.assertAlmostEqual(choice["brier"], (0.38 + 1.14) / 2)
        self.assertAlmostEqual(choice["ece"], (0.5 + 0.7) / 2)

        score = m["score"]
        self.assertAlmostEqual(score["accuracy"], 0.5)
        self.assertAlmostEqual(score["nll"], -math.log(0.7 * 0.25) / 2)
        # (0.01 + 0.04 + 0.09) and (0.5625 + 0.0625 + 0.25)
        self.assertAlmostEqual(score["brier"], (0.14 + 0.875) / 2)
        self.assertAlmostEqual(score["ece"], (0.3 + 0.5) / 2)
        # expected levels 1.6 and 1.25 against truths 2 and 0
        self.assertAlmostEqual(score["mae"], (0.4 + 1.25) / 2)

    def test_all_pools_records(self):
        a = metrics(self.records)["all"]
        self.assertEqual(a["n"], 6)
        self.assertNotIn("temperature", a)
        self.assertAlmostEqual(a["accuracy"], 0.5)
        self.assertAlmostEqual(a["nll"], -math.log(0.8 * 0.4 * 0.5 * 0.2 * 0.7 * 0.25) / 6)
        self.assertAlmostEqual(a["brier"], 3.335 / 6)
        # bin 5: hit + miss at 0.5 cancel, bin 6: miss at 0.6,
        # bin 7: hit + miss at 0.7 give |1 - 1.4|, bin 8: hit at 0.8
        self.assertAlmostEqual(a["ece"], (0.0 + 0.6 + 0.4 + 0.2) / 6)
        self.assertAlmostEqual(a["mean_confidence"], 3.8 / 6)
        self.assertAlmostEqual(a["latency_ms"]["p50"], 35.0)
        self.assertAlmostEqual(a["latency_ms"]["p95"], 57.5)

    def test_ece_compares_bin_means_not_single_records(self):
        records = [
            record("noul", [0.91, 0.09], 0),
            record("noul", [0.99, 0.01], 1),
            record("noul", [1.0, 0.0], 0),
        ]
        # All three land in the top bin (1.0 included): accuracy 2/3, mean
        # confidence 2.9/3. A per-record average would give 0.3667 instead.
        self.assertAlmostEqual(metrics(records)["noul"]["ece"], abs(2 - 2.9) / 3)

    def test_nll_is_clamped(self):
        m = metrics([record("noul", [1.0, 0.0], 1)])
        self.assertAlmostEqual(m["noul"]["nll"], -math.log(1e-12))
        self.assertAlmostEqual(m["noul"]["brier"], 2.0)

    def test_ties_go_to_the_first_index(self):
        self.assertEqual(metrics([record("noul", [0.5, 0.5], 0)])["all"]["accuracy"], 1.0)
        self.assertEqual(metrics([record("noul", [0.5, 0.5], 1)])["all"]["accuracy"], 0.0)

    def test_temperatures_apply_per_type(self):
        records = [record("noul", [0.8, 0.2], 0), record("noul", [0.9, 0.1], 1), record("choice", [0.6, 0.4], 0)]
        m = metrics(records, {"noul": 2.0})
        # T = 2 takes square roots: 0.8:0.2 becomes 2:1, 0.9:0.1 becomes 3:1.
        self.assertEqual(m["noul"]["temperature"], 2.0)
        self.assertAlmostEqual(m["noul"]["nll"], -math.log(2 / 3 * 1 / 4) / 2)
        self.assertAlmostEqual(m["noul"]["mean_confidence"], (2 / 3 + 3 / 4) / 2)
        self.assertEqual(m["choice"]["temperature"], 1.0)
        self.assertAlmostEqual(m["choice"]["nll"], -math.log(0.6))

    def test_empty_records(self):
        self.assertEqual(metrics([]), {"all": {"n": 0}})

    def test_percentile(self):
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([7], 0.95), 7)
        self.assertAlmostEqual(percentile([40, 10, 30, 20], 0.5), 25.0)


def overconfident_records(n=200):
    """The model puts 0.95 on yes but is right 70% of the time, and 0.9 on
    the first choice option but is right 60% of the time."""
    records = []
    for i in range(n):
        records.append(record("noul", [0.95, 0.05], 0 if i % 10 < 7 else 1, case_id=f"noul-{i}"))
        records.append(record("choice", [0.9, 0.04, 0.03, 0.03], 0 if i % 10 < 6 else 1 + i % 3, case_id=f"choice-{i}"))
    return records


class CrossFitTest(unittest.TestCase):
    def test_cross_fit_improves_overconfident_records(self):
        records = overconfident_records()
        raw = metrics(records)
        cv = cross_fit(records, folds=2)
        self.assertEqual(cv["folds"], 2)
        self.assertEqual(cv["metrics"]["all"]["n"], len(records))
        for key in ("noul", "choice", "all"):
            self.assertLess(cv["metrics"][key]["nll"], raw[key]["nll"])
            self.assertLess(cv["metrics"][key]["ece"], raw[key]["ece"])
            self.assertLess(cv["metrics"][key]["brier"], raw[key]["brier"])
        # Softening never changes the ranking.
        self.assertEqual(cv["metrics"]["all"]["accuracy"], raw["all"]["accuracy"])
        for qtype in ("noul", "choice"):
            self.assertGreater(cv["temperatures"][qtype], 1.0)
            self.assertEqual(len(cv["fold_temperatures"][qtype]), 2)

    def test_each_fold_uses_temperatures_fit_on_the_others(self):
        ids = [f"case-{i}" for i in range(60)]
        # Fold 0 is always right, fold 1 always wrong, at the same 0.9.
        records = [record("noul", [0.9, 0.1], 0 if fold_of(i, 2) == 0 else 1, case_id=i) for i in ids]
        cv = cross_fit(records, folds=2)
        t_for_fold0, t_for_fold1 = cv["fold_temperatures"]["noul"]
        # Fold 0 is scored with T fit on the always-wrong fold (as soft as
        # allowed), fold 1 with T fit on the always-right fold (as sharp).
        self.assertGreater(t_for_fold0, 10)
        self.assertLess(t_for_fold1, 0.1)

    def test_types_without_training_data_keep_t_one(self):
        cv = cross_fit([record("noul", [0.9, 0.1], 1, case_id="only")], folds=2)
        self.assertEqual(cv["temperatures"], {"noul": 1.0})
        self.assertAlmostEqual(cv["metrics"]["noul"]["nll"], -math.log(0.1))

    def test_needs_two_folds(self):
        with self.assertRaises(ValueError):
            cross_fit(overconfident_records(5), folds=1)

    def test_fit_temperatures_on_all_records(self):
        fitted = fit_temperatures(overconfident_records())
        self.assertEqual(set(fitted), {"noul", "choice"})
        t, n = fitted["noul"].temperature, fitted["noul"].n
        self.assertEqual(n, 200)
        self.assertIsNone(fitted["noul"].mae_raw)
        # Calibrated yes probability should come out near the 70% hit rate.
        self.assertAlmostEqual(0.95 ** (1 / t) / (0.95 ** (1 / t) + 0.05 ** (1 / t)), 0.7, places=3)

    def test_cross_fit_and_fit_temperatures_share_the_fit(self):
        records = [
            record("score", raw, truth, case_id=f"s{i}-{j}")
            for i, (raw, truth) in enumerate(OVERCONFIDENT_SCORE)
            for j in range(4)
        ]
        cv = cross_fit(records, folds=2)
        for k, t in enumerate(cv["fold_temperatures"]["score"]):
            train = [(r["raw"], r["truth"]) for r in records if fold_of(r["id"], 2) != k]
            self.assertEqual(t, fit_type("score", train).temperature)
        fit = fit_temperatures(records)["score"]
        self.assertTrue(fit.mae_worse)
        self.assertEqual(fit.n, len(records))


# Right at 0.96 eight times, wrong by four levels twice. NLL wants T near
# 1.65, which pulls every expected level toward the middle and costs MAE.
OVERCONFIDENT_SCORE = [([0.96, 0.01, 0.01, 0.01, 0.01], 0)] * 8 + [([0.96, 0.01, 0.01, 0.01, 0.01], 4)] * 2


class FitTypeTest(unittest.TestCase):
    def test_choice_and_noul_use_the_nll_optimum(self):
        samples = [([0.95, 0.05], 0)] * 7 + [([0.95, 0.05], 1)] * 3
        for qtype in ("choice", "noul"):
            fit = fit_type(qtype, samples)
            self.assertEqual(fit, TypeFit(fit_temperature(samples), 10))
            self.assertFalse(fit.mae_worse)

    def test_score_uses_the_nll_optimum_and_reports_mae(self):
        samples = OVERCONFIDENT_SCORE
        fit = fit_type("score", samples)
        self.assertEqual(fit.temperature, fit_temperature(samples))
        self.assertGreater(fit.temperature, 1.5)
        self.assertAlmostEqual(fit.mae_raw, score_mae(samples, 1.0))
        self.assertAlmostEqual(fit.mae_fitted, score_mae(samples, fit.temperature))
        self.assertGreater(fit.mae_fitted, fit.mae_raw + SCORE_MAE_TOLERANCE)
        self.assertTrue(fit.mae_worse)

    def test_score_mae_within_tolerance_is_not_flagged(self):
        samples = [([0.6, 0.3, 0.1], 0)] * 9 + [([0.6, 0.3, 0.1], 1)]
        fit = fit_type("score", samples)
        self.assertLess(fit.temperature, 1.0)
        self.assertLess(fit.mae_fitted, fit.mae_raw)
        self.assertFalse(fit.mae_worse)

    def test_empty_samples(self):
        self.assertEqual(fit_type("score", []), TypeFit(1.0, 0))


class FoldOfTest(unittest.TestCase):
    def test_stable_and_in_range(self):
        self.assertEqual(fold_of("abc", 5), 891568578 % 5)
        self.assertEqual(fold_of("abc", 5), zlib.crc32(b"abc") % 5)
        self.assertEqual(fold_of("abc", 2), fold_of("abc", 2))
        folds = [fold_of(f"case-{i}", 3) for i in range(900)]
        self.assertEqual(set(folds), {0, 1, 2})
        for k in range(3):
            self.assertGreater(folds.count(k), 200)


def fake_responder(messages):
    text = messages[-1]["content"]
    if "Yes or No" in text:
        return [("Yes", math.log(0.8)), ("No", math.log(0.2))]
    if "letter" in text:
        return [("B", math.log(0.7)), (" A", math.log(0.2)), ("C", math.log(0.1))]
    return [("1", math.log(0.6)), ("2", math.log(0.3)), ("0", math.log(0.1))]


class CollectTest(unittest.TestCase):
    def setUp(self):
        # A temperature for the fake model shows that records stay raw.
        calibration = Calibration({"fake": {"noul": {"temperature": 3.0, "n": 10}}})
        self.nex = Nex(backend=FakeBackend(fake_responder), calibration=calibration)
        self.cases = [choice_case("c1", label="technical"), score_case("s1", label=2), noul_case("n1", label=False)]

    def test_records(self):
        calls = []
        records = collect(self.nex, self.cases, progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual([r["id"] for r in records], ["c1", "s1", "n1"])
        self.assertEqual(
            set(records[0]),
            {"id", "type", "domain", "difficulty", "raw", "truth", "label_mass", "latency_ms",
             "prompt_tokens", "cached_tokens"},
        )
        choice, score, noul = records
        self.assertEqual((choice["type"], choice["truth"], choice["domain"]), ("choice", 1, "support"))
        self.assertEqual((score["type"], score["truth"], score["difficulty"]), ("score", 2, "medium"))
        self.assertEqual((noul["type"], noul["truth"]), ("noul", 1))
        for got, want in zip(choice["raw"], [0.2, 0.7, 0.1]):
            self.assertAlmostEqual(got, want)
        for got, want in zip(noul["raw"], [0.8, 0.2]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(noul["label_mass"], 1.0)
        self.assertGreater(noul["prompt_tokens"], 0)
        json.dumps(records)

    def test_workers_keep_case_order(self):
        cases = [noul_case(f"n{i}", label=i % 2 == 0) for i in range(12)]
        serial = collect(self.nex, cases)
        parallel = collect(self.nex, cases, workers=4)
        strip = lambda rs: [{k: v for k, v in r.items() if k != "latency_ms"} for r in rs]
        self.assertEqual(strip(parallel), strip(serial))

    def test_model_selects_backend(self):
        records = collect(self.nex, self.cases[:1], model="other")
        self.assertEqual(len(records), 1)
        self.assertEqual(len(self.nex.resolve_backend("other").calls), 1)
        self.assertEqual(self.nex.backend.calls, [])

    def test_backend_errors_name_the_case(self):
        def broken(messages):
            raise BackendError("cannot reach Ollama")

        nex = Nex(backend=FakeBackend(broken), calibration=Calibration())
        for workers in (1, 2):
            with self.assertRaises(BackendError) as ctx:
                collect(nex, self.cases, workers=workers)
            self.assertIn("c1", str(ctx.exception))
            self.assertIn("cannot reach Ollama", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
