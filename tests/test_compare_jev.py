"""evals/compare_jev.py with Jev stubbed out, so no test reaches the network."""

import copy
import email.message
import email.utils
import http.client
import importlib.util
import io
import json
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_jev", ROOT / "evals" / "compare_jev.py")
compare_jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_jev)

from nex import evaluation  # noqa: E402

CASES = [
    {"id": "c1", "domain": "d", "difficulty": "easy", "state": "s1",
     "question": {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None}}, "label": "b"},
    {"id": "s1", "domain": "d", "difficulty": "easy", "state": "s2",
     "question": {"type": "score", "instructions": "q", "criteria": ["lo", "mid", "hi"]}, "label": 2},
    {"id": "n1", "domain": "d", "difficulty": "easy", "state": "s3",
     "question": {"type": "noul", "instructions": "q"}, "label": False},
]

JEV_ANSWERS = {
    "c1": {"type": "choice", "choice": "b", "probabilities": {"b": 0.9, "a": 0.1}, "confidence": 0.8},
    "s1": {"type": "score", "score": 1.9, "legend": {}, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}, "confidence": 0.9},
    "n1": {"type": "noul", "noul": 0.2},
}

# Enough Noul cases in both folds for cross-fitted temperatures to differ from 1.
NOUL_CASES = [
    {"id": f"n{i:02d}", "domain": "d", "difficulty": "easy", "state": f"state {i}",
     "question": {"type": "noul", "instructions": "q"}, "label": i % 2 == 0}
    for i in range(16)
]


def noul_records():
    """Nex records for NOUL_CASES, confident and wrong on every fifth case."""
    out = []
    for i, case in enumerate(NOUL_CASES):
        truth = 0 if case["label"] else 1
        confidence = 0.6 + 0.025 * i
        top = truth if i % 5 else 1 - truth
        raw = [confidence, 1 - confidence] if top == 0 else [1 - confidence, confidence]
        out.append({"id": case["id"], "type": "noul", "raw": raw, "truth": truth, "label_mass": 1.0, "latency_ms": 2.0})
    return out


def fake_ask(case, key, model, retries=4):
    answer = JEV_ANSWERS.get(case["id"], {"type": "noul", "noul": 0.9})
    return {"model": "jev-test", "answers": {"q": answer}, "usage": {"input_tokens": 10}}, 5.0


def fake_ask_as(version):
    """fake_ask, answering as Jev ``version``."""
    def ask(case, key, model, retries=4):
        response, ms = fake_ask(case, key, model)
        return dict(response, model=version), ms
    return ask


def http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(compare_jev.JEV_URL, code, "error", headers, io.BytesIO(b"busy"))


def http_date(seconds_from_now):
    return email.utils.format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now), usegmt=True)


class CompareJevTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.cache = self.dir / "jev.json"
        self.results = self.dir / "nex.json"
        # Nex always picks the first slot, so it is right only on n1.
        records = [
            {"id": "c1", "type": "choice", "raw": [0.8, 0.2], "truth": 1, "label_mass": 1.0, "latency_ms": 1.0},
            {"id": "s1", "type": "score", "raw": [0.7, 0.2, 0.1], "truth": 2, "label_mass": 1.0, "latency_ms": 1.0},
            {"id": "n1", "type": "noul", "raw": [0.3, 0.7], "truth": 1, "label_mass": 1.0, "latency_ms": 1.0},
        ]
        self.write_eval(CASES, records)
        env = mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key", "NEX_CALIBRATION": str(self.dir / "none.json")})
        env.start()
        self.addCleanup(env.stop)

    def write_eval(self, cases, records):
        data = self.dir / "cases.jsonl"
        data.write_text("".join(json.dumps(c) + "\n" for c in cases))
        self.data = str(data)
        self.results.write_text(json.dumps({"fake-model": {"records": records}}))

    def write_cache_versions(self, versions):
        """A current cache for CASES where case ``id`` was answered by ``versions[id]``."""
        entries = {
            c["id"]: {"hash": compare_jev.case_hash(c), "model": versions[c["id"]], "answer": JEV_ANSWERS[c["id"]],
                      "latency_ms": 5.0, "input_tokens": 10, "fetched_at": None}
            for c in CASES
        }
        self.cache.write_text(json.dumps({"jev-latest": entries}))

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            compare_jev.main([str(self.results), "--data", self.data, "--jev-cache", str(self.cache), *extra])
        return out.getvalue(), err.getvalue()

    def test_jev_distribution_uses_nex_slot_order(self):
        self.assertEqual(compare_jev.jev_distribution(CASES[0]["question"], JEV_ANSWERS["c1"]), [0.1, 0.9])
        self.assertEqual(compare_jev.jev_distribution(CASES[1]["question"], JEV_ANSWERS["s1"]), [0.0, 0.1, 0.9])
        self.assertEqual(compare_jev.jev_distribution(CASES[2]["question"], JEV_ANSWERS["n1"]), [0.2, 0.8])

    def test_end_to_end_and_head_to_head(self):
        out_file = self.dir / "cmp.json"
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            out, err = self.run_main("--out", str(out_file))
        self.assertEqual(ask.call_count, 3)
        self.assertIn("asking Jev about 3 cases", err)
        self.assertIn("jev jev-test", out)
        self.assertIn("nex fake-model", out)
        result = json.loads(out_file.read_text())
        self.assertEqual(result["jev_model"], "jev-test")
        self.assertEqual(result["metrics"]["jev jev-test"]["all"]["accuracy"], 1.0)
        self.assertAlmostEqual(result["metrics"]["nex fake-model"]["all"]["accuracy"], 1 / 3)
        h2h = result["head_to_head"]["nex fake-model"]
        self.assertEqual(sorted(h2h["jev_only_right"]), ["c1", "s1"])
        self.assertEqual(h2h["nex_only_right"], [])
        self.assertEqual(h2h["both_wrong"], [])

    def test_cache_is_reused_and_refreshed_when_a_case_changes(self):
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask):
            self.run_main()
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            self.run_main()
        self.assertEqual(ask.call_count, 0)

        changed = [dict(c) for c in CASES]
        changed[2]["state"] = "a different state"
        Path(self.data).write_text("".join(json.dumps(c) + "\n" for c in changed))
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            self.run_main()
        self.assertEqual([c.args[0]["id"] for c in ask.call_args_list], ["n1"])

    def test_missing_key_stops_before_any_request(self):
        os.environ.pop("TYPESAFE_API_KEY")
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            with self.assertRaises(SystemExit) as ctx:
                self.run_main()
        self.assertIn("TYPESAFE_API_KEY", str(ctx.exception))
        self.assertEqual(ask.call_count, 0)

    def test_failed_request_keeps_the_answers_fetched_before_it(self):
        def flaky(case, key, model, retries=4):
            if case["id"] == "s1":
                raise RuntimeError("Jev returned HTTP 400 for s1: bad request")
            return fake_ask(case, key, model)

        with mock.patch.object(compare_jev, "ask_jev", side_effect=flaky):
            with self.assertRaises(SystemExit) as ctx:
                self.run_main("--workers", "1")
        self.assertIn("HTTP 400 for s1", str(ctx.exception))
        saved = json.loads(self.cache.read_text())["jev-latest"]
        self.assertIn("c1", saved)
        self.assertNotIn("s1", saved)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            self.run_main()
        asked = {c.args[0]["id"] for c in ask.call_args_list}
        self.assertIn("s1", asked)
        self.assertNotIn("c1", asked)

    def test_cache_file_is_replaced_atomically(self):
        self.cache.write_text('{"old": {}}\n')
        with mock.patch.object(compare_jev.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                compare_jev.save_cache(self.cache, {"new": {}})
        self.assertEqual(self.cache.read_text(), '{"old": {}}\n')
        compare_jev.save_cache(self.cache, {"new": {}})
        self.assertEqual(json.loads(self.cache.read_text()), {"new": {}})
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_stale_nex_records_stop_before_any_request(self):
        good = json.loads(self.results.read_text())
        changes = {
            "type": ("c1", {"type": "noul"}, "c1 is noul but the case is choice"),
            "label": ("s1", {"truth": 1}, "s1 has a different label"),
            "slots": ("c1", {"raw": [0.5, 0.3, 0.2]}, "c1 has 3 slots but the case has 2"),
            "missing": ("n1", None, "2 of 3 cases"),
        }
        for what, (case_id, change, message) in changes.items():
            with self.subTest(what):
                data = copy.deepcopy(good)
                records = data["fake-model"]["records"]
                if change is None:
                    records[:] = [r for r in records if r["id"] != case_id]
                else:
                    next(r for r in records if r["id"] == case_id).update(change)
                self.results.write_text(json.dumps(data))
                with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
                    with self.assertRaises(SystemExit) as ctx:
                        self.run_main()
                self.assertIn(message, str(ctx.exception))
                self.assertIn("rerun nex eval", str(ctx.exception).lower())
                self.assertEqual(ask.call_count, 0)

    def test_cross_fit_temperatures_match_nex_eval(self):
        records = noul_records() + json.loads(self.results.read_text())["fake-model"]["records"]
        by_id, means = compare_jev.cross_fit_temperatures(records, 2)
        held_out = evaluation.metrics(compare_jev.rescaled(records, by_id))
        want = evaluation.cross_fit(records, 2)
        self.assertEqual(means, want["temperatures"])
        for qtype, row in want["metrics"].items():
            for key in ("accuracy", "nll", "ece", "brier"):
                self.assertAlmostEqual(held_out[qtype][key], row[key], msg=f"{qtype} {key}")

    def test_nex_is_scored_held_out_by_default(self):
        records = noul_records()
        self.write_eval(NOUL_CASES, records)
        self.assertEqual({evaluation.fold_of(r["id"], 2) for r in records}, {0, 1})
        out_file = self.dir / "cmp.json"
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask):
            out, _ = self.run_main("--out", str(out_file))
        self.assertIn("Nex calibration: 2-fold cross-fit", out)
        self.assertIn("(held out)", out)
        result = json.loads(out_file.read_text())
        self.assertEqual(result["nex_calibration"], {"method": "cross-fit", "folds": 2})
        got = result["metrics"]["nex fake-model"]["all"]
        want = evaluation.cross_fit(records, 2)["metrics"]["all"]
        for key in ("accuracy", "nll", "ece", "brier"):
            self.assertAlmostEqual(got[key], want[key])
        fitted = evaluation.fit_temperatures(records)["noul"].temperature
        self.assertNotAlmostEqual(got["nll"], evaluation.metrics(records)["all"]["nll"])
        self.assertNotAlmostEqual(got["nll"], evaluation.metrics(records, {"noul": fitted})["all"]["nll"])

    def test_calibration_file_option(self):
        records = noul_records()
        self.write_eval(NOUL_CASES, records)
        calibration = self.dir / "calibration.json"
        calibration.write_text(json.dumps({"models": {"fake-model": {"noul": {"temperature": 2.5, "n": 16}}}}))
        os.environ["NEX_CALIBRATION"] = str(calibration)
        out_file = self.dir / "cmp.json"
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask):
            out, _ = self.run_main("--nex-calibration", "file", "--out", str(out_file))
        self.assertIn(f"Nex calibration: temperatures from {calibration}, in-sample", out)
        result = json.loads(out_file.read_text())
        self.assertEqual(result["nex_calibration"], {"method": "file", "path": str(calibration)})
        got = result["metrics"]["nex fake-model"]
        want = evaluation.metrics(records, {"noul": 2.5})
        self.assertEqual(got["noul"]["temperature"], 2.5)
        for key in ("accuracy", "nll", "ece", "brier"):
            self.assertAlmostEqual(got["all"][key], want["all"][key])

    def test_cache_is_kept_per_requested_model(self):
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            self.run_main("--jev-model", "jev-a")
            self.run_main("--jev-model", "jev-b")
            self.run_main("--jev-model", "jev-a")
        self.assertEqual(ask.call_count, 6)
        self.assertEqual({c.args[2] for c in ask.call_args_list}, {"jev-a", "jev-b"})
        cache = json.loads(self.cache.read_text())
        self.assertEqual(sorted(cache), ["jev-a", "jev-b"])
        entry = cache["jev-a"]["c1"]
        self.assertEqual(entry["model"], "jev-test")
        self.assertNotIn("model_asked", entry)
        self.assertIsNotNone(datetime.fromisoformat(entry["fetched_at"]).tzinfo)

    def test_flat_cache_is_migrated_without_refetching(self):
        flat = {
            c["id"]: {"hash": compare_jev.case_hash(c), "model_asked": "jev-latest", "model": "jev-1.13.0",
                      "answer": JEV_ANSWERS[c["id"]], "latency_ms": 5.0, "input_tokens": 10}
            for c in CASES
        }
        self.cache.write_text(json.dumps(flat))
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            out, _ = self.run_main()
        self.assertEqual(ask.call_count, 0)
        self.assertIn("jev jev-1.13.0", out)
        loaded = compare_jev.load_cache(self.cache)
        self.assertEqual(list(loaded), ["jev-latest"])
        self.assertEqual(loaded["jev-latest"]["c1"]["answer"], JEV_ANSWERS["c1"])
        self.assertNotIn("model_asked", loaded["jev-latest"]["c1"])

        # The next save writes the new format and keeps the old answers.
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask) as ask:
            self.run_main("--jev-model", "jev-other")
        self.assertEqual(ask.call_count, 3)
        saved = json.loads(self.cache.read_text())
        self.assertEqual(sorted(saved), ["jev-latest", "jev-other"])
        self.assertEqual(saved["jev-latest"]["c1"]["model"], "jev-1.13.0")

    def test_answers_from_an_older_jev_version_are_asked_again(self):
        # Compared by number, 1.13 is newer than 1.9.
        self.write_cache_versions({"c1": "jev-1.9.0", "s1": "jev-1.13.0", "n1": "jev-1.13.0"})
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask_as("jev-1.13.0")) as ask:
            out, err = self.run_main()
        self.assertEqual([c.args[0]["id"] for c in ask.call_args_list], ["c1"])
        self.assertIn("older than jev-1.13.0", err)
        self.assertIn("jev jev-1.13.0", out)
        self.assertNotIn("jev-1.9.0", out)

    def test_a_new_jev_release_refetches_every_case(self):
        self.write_cache_versions({"c1": "jev-1.13.0", "s1": "jev-1.13.0", "n1": "jev-1.13.0"})
        changed = [dict(c) for c in CASES]
        changed[2]["state"] = "a different state"
        Path(self.data).write_text("".join(json.dumps(c) + "\n" for c in changed))
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask_as("jev-1.14.0")) as ask:
            out, _ = self.run_main()
        self.assertEqual([c.args[0]["id"] for c in ask.call_args_list], ["n1", "c1", "s1"])
        self.assertIn("jev jev-1.14.0", out)
        self.assertNotIn("jev-1.13.0", out)
        versions = {e["model"] for e in json.loads(self.cache.read_text())["jev-latest"].values()}
        self.assertEqual(versions, {"jev-1.14.0"})

    def test_versions_that_keep_mixing_stop_with_a_message(self):
        self.write_cache_versions({"c1": "jev-1.13.0", "s1": "jev-1.13.0", "n1": "jev-1.14.0"})
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask_as("jev-1.13.0")) as ask:
            with self.assertRaises(SystemExit) as ctx:
                self.run_main()
        self.assertIn("several versions (jev-1.13.0, jev-1.14.0)", str(ctx.exception))
        self.assertEqual(ask.call_count, 2 * compare_jev.VERSION_ROUNDS)

    def test_table_has_a_p50_latency_column(self):
        with mock.patch.object(compare_jev, "ask_jev", side_effect=fake_ask):
            out, _ = self.run_main()
        lines = out.splitlines()
        header = next(line for line in lines if line.startswith("system"))
        self.assertTrue(header.endswith("p50 ms"))
        jev_all = next(line for line in lines if line.startswith("jev jev-test") and " all " in line)
        nex_all = next(line for line in lines if line.startswith("nex fake-model") and " all " in line)
        self.assertEqual(jev_all.split()[-1], "5")
        self.assertEqual(nex_all.split()[-1], "1")


class AskJevTest(unittest.TestCase):
    def test_retry_delay_reads_seconds_and_http_dates_and_is_capped(self):
        cap = compare_jev.MAX_RETRY_DELAY
        self.assertEqual(compare_jev.retry_delay("3", 0), 3.0)
        self.assertEqual(compare_jev.retry_delay("9999", 0), cap)
        self.assertEqual(compare_jev.retry_delay("-4", 0), 0.0)
        self.assertTrue(25 <= compare_jev.retry_delay(http_date(30), 0) <= 30)
        self.assertEqual(compare_jev.retry_delay(http_date(86400), 0), cap)
        self.assertEqual(compare_jev.retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 0), 0.0)
        self.assertEqual(compare_jev.retry_delay("soon", 2), 4.0)
        self.assertEqual(compare_jev.retry_delay("nan", 1), 2.0)
        self.assertEqual(compare_jev.retry_delay(None, 3), 8.0)
        self.assertEqual(compare_jev.retry_delay(None, 10), cap)

    def test_retries_rate_limits_and_dropped_connections(self):
        response = {"model": "jev-test", "answers": {"q": JEV_ANSWERS["n1"]}}
        replies = [
            http_error(429, http_date(30)),
            http.client.RemoteDisconnected("closed"),
            ConnectionResetError("reset"),
            http.client.IncompleteRead(b""),
            io.BytesIO(json.dumps(response).encode()),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=replies) as urlopen, \
                mock.patch.object(compare_jev.time, "sleep") as sleep:
            got, _ = compare_jev.ask_jev(CASES[2], "key", "jev-latest")
        self.assertEqual(got, response)
        self.assertEqual(urlopen.call_count, 5)
        delays = [c.args[0] for c in sleep.call_args_list]
        self.assertTrue(25 <= delays[0] <= 30)
        self.assertEqual(delays[1:], [2.0, 4.0, 8.0])

    def test_gives_up_after_the_last_retry(self):
        with mock.patch("urllib.request.urlopen", side_effect=ConnectionResetError("reset")), \
                mock.patch.object(compare_jev.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError) as ctx:
                compare_jev.ask_jev(CASES[2], "key", "jev-latest", retries=2)
        self.assertIn("cannot reach Jev for n1", str(ctx.exception))
        self.assertEqual(sleep.call_count, 2)

    def test_client_errors_are_not_retried(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(400)), \
                mock.patch.object(compare_jev.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError) as ctx:
                compare_jev.ask_jev(CASES[2], "key", "jev-latest")
        self.assertIn("HTTP 400 for n1: busy", str(ctx.exception))
        self.assertEqual(sleep.call_count, 0)


if __name__ == "__main__":
    unittest.main()
