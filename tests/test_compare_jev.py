"""evals/compare_jev.py with Jev stubbed out, so no test reaches the network."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_jev", ROOT / "evals" / "compare_jev.py")
compare_jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_jev)

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


def fake_ask(case, key, model, retries=4):
    return {"model": "jev-test", "answers": {"q": JEV_ANSWERS[case["id"]]}, "usage": {"input_tokens": 10}}, 5.0


class CompareJevTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        data = self.dir / "cases.jsonl"
        data.write_text("".join(json.dumps(c) + "\n" for c in CASES))
        self.data = str(data)
        self.cache = self.dir / "jev.json"
        # Nex always picks the first slot, so it is right only on n1.
        records = [
            {"id": "c1", "type": "choice", "raw": [0.8, 0.2], "truth": 1, "label_mass": 1.0, "latency_ms": 1.0},
            {"id": "s1", "type": "score", "raw": [0.7, 0.2, 0.1], "truth": 2, "label_mass": 1.0, "latency_ms": 1.0},
            {"id": "n1", "type": "noul", "raw": [0.3, 0.7], "truth": 1, "label_mass": 1.0, "latency_ms": 1.0},
        ]
        self.results = self.dir / "nex.json"
        self.results.write_text(json.dumps({"fake-model": {"records": records}}))
        env = mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key", "NEX_CALIBRATION": str(self.dir / "none.json")})
        env.start()
        self.addCleanup(env.stop)

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


if __name__ == "__main__":
    unittest.main()
