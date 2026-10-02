import io
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import nex.calibration
from nex import Calibration, FakeBackend, Nex
from nex.backend import BackendError
from nex.cli import main, read_value
from nex.evaluation import fit_temperatures

ROOT = Path(__file__).resolve().parent.parent
_guards = []
_guard_dir = None
GUARD_PATH = None


def setUpModule():
    # Point the default calibration file at a temp path, so nothing in these
    # tests can read or write the real nex/calibration.json.
    global _guard_dir, GUARD_PATH
    _guard_dir = tempfile.TemporaryDirectory()
    GUARD_PATH = Path(_guard_dir.name) / "default.json"
    _guards.append(mock.patch.object(nex.calibration, "DEFAULT_PATH", GUARD_PATH))
    _guards.append(mock.patch.dict(os.environ, {"NEX_CALIBRATION": str(GUARD_PATH)}))
    for guard in _guards:
        guard.start()


def tearDownModule():
    for guard in reversed(_guards):
        guard.stop()
    _guards.clear()
    _guard_dir.cleanup()


def responder(messages):
    """An overconfident fake model that always picks the first option."""
    text = messages[-1]["content"]
    if "Yes or No" in text:
        return [("Yes", math.log(0.9)), ("No", math.log(0.1))]
    if "letter" in text:
        return [("A", math.log(0.9)), ("B", math.log(0.06)), ("C", math.log(0.04))]
    return [("1", math.log(0.9)), ("2", math.log(0.06)), ("3", math.log(0.04))]


def run(argv, nex=None):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv, nex=nex)
    return code, out.getvalue(), err.getvalue()


def record(case_id, rtype, raw, truth):
    return {"id": case_id, "type": rtype, "raw": raw, "truth": truth, "label_mass": 1.0, "latency_ms": 5.0}


def write_dataset(directory):
    choice = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": None, "sales": None}}
    score = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}
    noul = {"type": "noul", "instructions": "Is this about billing?"}
    files = {
        "choice.jsonl": [(choice, label) for label in ["billing", "billing", "billing", "technical", "technical", "sales"]],
        "score.jsonl": [(score, label) for label in [0, 0, 0, 1, 2, 1]],
        "noul.jsonl": [(noul, label) for label in [True, True, True, True, False, False]],
    }
    for name, rows in files.items():
        lines = []
        for i, (question, label) in enumerate(rows):
            case = {
                "id": f"{name.split('.')[0]}-{i}",
                "domain": "support",
                "difficulty": "easy",
                "state": f"ticket number {i}",
                "question": question,
                "label": label,
            }
            lines.append(json.dumps(case))
        (Path(directory) / name).write_text("\n".join(lines) + "\n")


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.backend = FakeBackend(responder)
        self.nex = Nex(backend=self.backend, calibration=Calibration(path=self.dir / "unused.json"))

    def tearDown(self):
        self._tmp.cleanup()

    def ask(self, *argv):
        code, out, err = run(["ask", "--state", "I was charged twice", *argv], self.nex)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def assertFails(self, argv, *fragments):
        code, out, err = run(argv, self.nex)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("nex: error: "), err)
        self.assertEqual(err.count("\n"), 1, err)
        self.assertNotIn("Traceback", err)
        for fragment in fragments:
            self.assertIn(fragment, err)
        return err


class AskTest(CliTestCase):
    def test_noul(self):
        out = self.ask("--noul", "Is this about billing?")
        self.assertEqual(out["model"], "nex-0.1.0+fake")
        self.assertEqual(out["answers"], {"noul": {"type": "noul", "noul": 0.9}})
        self.assertEqual(out["usage"]["output_tokens"], 1)
        self.assertNotIn("diagnostics", out)

    def test_choice(self):
        out = self.ask("--choice", "Which team?", "--options", "billing, technical,sales")
        answer = out["answers"]["choice"]
        self.assertEqual(answer["choice"], "billing")
        self.assertEqual(list(answer["probabilities"]), ["billing", "technical", "sales"])
        self.assertIn("A) billing", self.backend.calls[-1][-1]["content"])

    def test_score(self):
        out = self.ask("--score", "How urgent?", "--levels", "low,mid,high")
        answer = out["answers"]["score"]
        self.assertEqual(answer["legend"], {"0": "low", "1": "mid", "2": "high"})
        self.assertAlmostEqual(answer["score"], 0.06 + 2 * 0.04, places=4)

    def test_quick_forms_combine(self):
        out = self.ask("--noul", "Billing?", "--choice", "Which team?", "--options", "billing,technical")
        self.assertEqual(set(out["answers"]), {"noul", "choice"})
        self.assertEqual(out["usage"]["output_tokens"], 2)

    def test_questions_inline_and_from_file(self):
        questions = {
            "billing": {"type": "noul", "instructions": "Is this about billing?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": "Bugs"}},
        }
        out = self.ask("--questions", json.dumps(questions))
        self.assertEqual(set(out["answers"]), {"billing", "team"})
        path = self.dir / "questions.json"
        path.write_text(json.dumps(questions))
        self.assertEqual(self.ask("--questions", f"@{path}")["answers"], out["answers"])

    def test_debug_includes_diagnostics(self):
        out = self.ask("--noul", "Billing?", "--debug")
        diag = out["diagnostics"]["noul"]
        self.assertEqual(diag["raw_probabilities"], [0.9, 0.1])
        self.assertAlmostEqual(diag["label_mass"], 1.0)

    def test_state_from_file_and_json(self):
        path = self.dir / "state.json"
        path.write_text(json.dumps({"ticket": {"subject": "Double charge"}}) + "\n")
        code, _, err = run(["ask", "--state", f"@{path}", "--noul", "Billing?"], self.nex)
        self.assertEqual(code, 0, err)
        self.assertIn('"ticket": {\n', self.backend.calls[-1][-1]["content"])

        text = self.dir / "state.txt"
        text.write_text("plain words\n")
        self.assertEqual(read_value(f"@{text}"), "plain words")
        self.assertEqual(read_value('["a", "b"]'), ["a", "b"])
        self.assertEqual(read_value('{"a": 1}'), {"a": 1})
        self.assertEqual(read_value("42"), "42")
        self.assertEqual(read_value('"quoted"'), '"quoted"')
        with mock.patch("sys.stdin", io.StringIO('{"from": "stdin"}\n')):
            self.assertEqual(read_value("@-"), {"from": "stdin"})

    def test_backend_model_option_before_or_after_subcommand(self):
        code, out, err = run(["ask", "--backend-model", "other", "--state", "x", "--noul", "y?"], self.nex)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["model"], "nex-0.1.0+other")
        code, out, err = run(["--backend-model", "second", "ask", "--state", "x", "--noul", "y?"], self.nex)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["model"], "nex-0.1.0+second")

    def test_errors_are_one_line(self):
        self.assertFails(["ask", "--state", "x"], "nothing to ask")
        self.assertFails(["ask", "--state", "x", "--choice", "Which?"], "--options")
        self.assertFails(["ask", "--state", "x", "--options", "a,b"], "--choice")
        self.assertFails(["ask", "--state", "x", "--score", "How?"], "--levels")
        self.assertFails(["ask", "--state", "x", "--levels", "a,b"], "--score")
        self.assertFails(["ask", "--state", "x", "--choice", "Which?", "--options", "a,a"], "twice")
        self.assertFails(["ask", "--state", "x", "--choice", "Which?", "--options", "only"], "questions.choice.criteria")
        self.assertFails(["ask", "--state", "x", "--questions", "{nope"], "not valid JSON")
        self.assertFails(["ask", "--state", "x", "--questions", "[]"], "questions")
        self.assertFails(["ask", "--state", "x", "--questions", "{}", "--noul", "y?"], "either")
        self.assertFails(["ask", "--state", f"@{self.dir / 'missing.txt'}", "--noul", "y?"], "missing.txt")
        self.assertEqual(self.backend.calls, [])

    def test_backend_error_exits_1(self):
        def unreachable(messages):
            raise BackendError("cannot reach Ollama at http://127.0.0.1:1: Connection refused")

        self.nex = Nex(backend=FakeBackend(unreachable), calibration=Calibration(path=self.dir / "c.json"))
        self.assertFails(["ask", "--state", "x", "--noul", "y?"], "cannot reach Ollama")

    def test_missing_state_is_a_usage_error(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            main(["ask", "--noul", "y?"], nex=self.nex)
        self.assertEqual(ctx.exception.code, 2)

    def test_bad_nex_port_does_not_affect_ask_or_help(self):
        with mock.patch.dict(os.environ, {"NEX_PORT": "not-a-port", "NEX_HOST": ""}):
            self.assertEqual(self.ask("--noul", "Billing?")["answers"]["noul"]["noul"], 0.9)
            for argv in (["--help"], ["serve", "--help"]):
                with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as ctx:
                    main(argv, nex=self.nex)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("usage: nex", out.getvalue())


class ServeTest(CliTestCase):
    def setUp(self):
        super().setUp()
        # The developer's own NEX_HOST and NEX_PORT must not leak in.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("NEX_HOST", None)
        os.environ.pop("NEX_PORT", None)
        self.calls = []
        fake_server = types.ModuleType("nex.server")
        fake_server.serve = lambda nex, host, port: self.calls.append((host, port))
        patcher = mock.patch.dict(sys.modules, {"nex.server": fake_server})
        patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, *argv, **env):
        os.environ.update(env)
        code, _, err = run(["serve", *argv], self.nex)
        self.assertEqual(code, 0, err)
        return self.calls.pop()

    def test_defaults_and_options(self):
        self.assertEqual(self.serve(), ("127.0.0.1", 8787))
        self.assertEqual(self.serve("--port", "9999"), ("127.0.0.1", 9999))
        self.assertEqual(self.serve("--host", "0.0.0.0"), ("0.0.0.0", 8787))
        self.assertEqual(self.serve("--port", "0"), ("127.0.0.1", 0))
        self.assertEqual(self.serve("--port", "65535"), ("127.0.0.1", 65535))

    def test_environment_and_option_precedence(self):
        self.assertEqual(self.serve(NEX_HOST="10.0.0.5", NEX_PORT="9000"), ("10.0.0.5", 9000))
        self.assertEqual(self.serve("--host", "::1", "--port", "9001"), ("::1", 9001))

    def test_empty_environment_counts_as_unset(self):
        # An empty host would bind every interface.
        self.assertEqual(self.serve(NEX_HOST="", NEX_PORT=""), ("127.0.0.1", 8787))
        self.assertEqual(self.serve(NEX_HOST="  ", NEX_PORT=" "), ("127.0.0.1", 8787))
        self.assertEqual(self.serve("--host", "", NEX_HOST=""), ("127.0.0.1", 8787))

    def test_bad_nex_port_is_a_one_line_error(self):
        for value in ("abc", "80.5", "70000", "-1"):
            os.environ["NEX_PORT"] = value
            self.assertFails(["serve"], "NEX_PORT", "0 to 65535", repr(value))
        os.environ["NEX_PORT"] = "abc"
        self.assertEqual(self.serve("--port", "9002"), ("127.0.0.1", 9002))
        self.assertEqual(self.calls, [])

    def test_port_option_is_range_checked(self):
        for value in ("70000", "-1", "http"):
            with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as ctx:
                main(["serve", "--port", value], nex=self.nex)
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("0 to 65535", err.getvalue())
        self.assertEqual(self.calls, [])


class EvalTest(CliTestCase):
    def setUp(self):
        super().setUp()
        self.data = self.dir / "data"
        self.data.mkdir()
        write_dataset(self.data)

    def test_eval_prints_table_and_writes_out(self):
        self.nex.calibration.set("fake", "noul", 2.0, 5)
        out_path = self.dir / "results" / "eval.json"
        code, out, err = run(["eval", "--data", str(self.data), "--out", str(out_path)], self.nex)
        self.assertEqual(code, 0, err)
        self.assertIn("fake: 18/18", err)
        self.assertIn("NLL raw -> cv", out)
        rows = {line.split()[0] for line in out.splitlines() if line.strip()}
        self.assertTrue({"choice", "score", "noul", "all"} <= rows, out)
        self.assertIn("noul T=2.00", out)
        self.assertIn("current calibration, in-sample if it was fitted on these cases", out)
        self.assertEqual(len(self.backend.calls), 18)

        result = json.loads(out_path.read_text())
        self.assertEqual(list(result), ["fake"])
        fake = result["fake"]
        self.assertEqual(set(fake), {"records", "raw", "cross_fit", "current"})
        self.assertEqual(len(fake["records"]), 18)
        self.assertEqual(fake["raw"]["all"]["n"], 18)
        self.assertAlmostEqual(fake["raw"]["noul"]["accuracy"], 4 / 6)
        self.assertEqual(fake["cross_fit"]["folds"], 2)
        self.assertEqual(fake["current"]["noul"]["temperature"], 2.0)
        self.assertEqual(fake["current"]["choice"]["temperature"], 1.0)
        self.assertNotEqual(fake["current"]["noul"]["nll"], fake["raw"]["noul"]["nll"])
        self.assertIn("mae", fake["raw"]["score"])

    def test_eval_several_models_and_options(self):
        out_path = self.dir / "eval.json"
        code, out, err = run(
            ["eval", "--data", str(self.data / "noul.jsonl"), "--model", "m1", "--model", "nex-0.1.0+m2",
             "--model", "m1", "--workers", "2", "--folds", "3", "--out", str(out_path)],
            self.nex,
        )
        self.assertEqual(code, 0, err)
        result = json.loads(out_path.read_text())
        self.assertEqual(list(result), ["m1", "m2"])
        self.assertEqual(result["m2"]["cross_fit"]["folds"], 3)
        self.assertEqual(len(result["m1"]["records"]), 6)
        self.assertIn("m1: 6 cases", out)
        self.assertIn("m2: 6 cases", out)
        self.assertNotIn("choice", out)

    def test_eval_errors(self):
        self.assertFails(["eval", "--data", str(self.dir / "nope")], "no such file")
        bad = self.dir / "bad.jsonl"
        bad.write_text(json.dumps({"id": "x1", "domain": "d", "difficulty": "easy", "state": "s",
                                   "question": {"type": "noul", "instructions": "q"}, "label": "yes"}) + "\n")
        self.assertFails(["eval", "--data", str(bad)], "bad.jsonl:1", "x1")
        self.assertFails(["eval", "--data", str(self.data), "--folds", "1"], "--folds")
        self.assertFails(["eval", "--data", str(self.data), "--workers", "0"], "--workers")
        self.assertEqual(self.backend.calls, [])


class CalibrateTest(CliTestCase):
    def setUp(self):
        super().setUp()
        self.data = self.dir / "data"
        self.data.mkdir()
        write_dataset(self.data)
        self.path = self.dir / "calibration.json"

    def test_calibrate_runs_model_and_saves(self):
        code, out, err = run(["calibrate", "--data", str(self.data), "--path", str(self.path)], self.nex)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.backend.calls), 18)
        saved = Calibration.load(self.path)
        for qtype in ("choice", "score", "noul"):
            self.assertEqual(saved.models["fake"][qtype]["n"], 6)
            # The fake model always says 0.9 for the first option but is
            # right half the time, so every fitted temperature softens.
            self.assertGreater(saved.temperature("fake", qtype), 1.0)
        self.assertIn(f"saved to {self.path}", out)
        self.assertIn("NLL before -> after", out)
        self.assertFalse(GUARD_PATH.exists())

    def test_calibrate_keeps_other_models(self):
        Calibration({"other": {"noul": {"temperature": 1.5, "n": 3}}}, self.path).save()
        code, _, err = run(["calibrate", "--data", str(self.data / "noul.jsonl"), "--path", str(self.path)], self.nex)
        self.assertEqual(code, 0, err)
        saved = Calibration.load(self.path)
        self.assertEqual(saved.temperature("other", "noul"), 1.5)
        self.assertEqual(set(saved.models["fake"]), {"noul"})

    def test_calibrate_from_eval_output(self):
        out_path = self.dir / "eval.json"
        code, _, err = run(["eval", "--data", str(self.data), "--out", str(out_path)], self.nex)
        self.assertEqual(code, 0, err)
        calls = len(self.backend.calls)

        code, out, err = run(["calibrate", "--from", str(out_path), "--path", str(self.path)], self.nex)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.backend.calls), calls)
        direct = self.dir / "direct.json"
        run(["calibrate", "--data", str(self.data), "--path", str(direct)], self.nex)
        self.assertEqual(Calibration.load(self.path).models, Calibration.load(direct).models)

        self.assertFails(["calibrate", "--from", str(out_path), "--model", "missing", "--path", str(self.path)],
                         "no records for missing")
        junk = self.dir / "junk.json"
        junk.write_text("[1, 2]")
        self.assertFails(["calibrate", "--from", str(junk), "--path", str(self.path)], "eval --out")

    def calibrate_from(self, records, *argv):
        source = self.dir / "records.json"
        source.write_text(json.dumps({"fake": {"records": records}}))
        return run(["calibrate", "--from", str(source), *argv], self.nex)

    def test_boundary_fit_is_not_saved(self):
        Calibration({"fake": {"noul": {"temperature": 1.5, "n": 40}}}, self.path).save()
        # Always right at 0.9 drives T to T_MIN, always wrong drives it to T_MAX.
        right = [record(f"r{i}", "noul", [0.9, 0.1], 0) for i in range(40)]
        wrong = [record(f"w{i}", "choice", [0.8, 0.1, 0.1], 1) for i in range(40)]
        normal = [record(f"s{i}", "score", [0.7, 0.2, 0.1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 2][i % 10]) for i in range(40)]
        code, out, err = self.calibrate_from(right + wrong + normal, "--path", str(self.path))
        self.assertEqual(code, 0, err)
        saved = Calibration.load(self.path)
        self.assertEqual(saved.models["fake"]["noul"], {"temperature": 1.5, "n": 40})
        self.assertNotIn("choice", saved.models["fake"])
        self.assertIn("score", saved.models["fake"])
        not_saved = [line for line in err.splitlines() if line.startswith("nex: warning: ") and "not saved" in line]
        self.assertEqual(len(not_saved), 2, err)
        self.assertIn("noul: not saved, kept T=1.50", err)
        self.assertIn("lower bound", err)
        self.assertIn("every case right", err)
        self.assertIn("choice: not saved, kept T=1.00", err)
        self.assertIn("upper bound", err)
        self.assertIn("every case wrong", err)

    def test_few_records_warn_but_save(self):
        code, out, err = run(["calibrate", "--data", str(self.data / "noul.jsonl"), "--path", str(self.path)], self.nex)
        self.assertEqual(code, 0, err)
        self.assertIn("nex: warning: noul: only 6 records", err)
        self.assertEqual(Calibration.load(self.path).models["fake"]["noul"]["n"], 6)

        normal = [record(f"s{i}", "noul", [0.7, 0.3], i % 3 // 2) for i in range(30)]
        code, out, err = self.calibrate_from(normal, "--path", str(self.path))
        self.assertEqual(code, 0, err)
        self.assertNotIn("warning", err)

    def test_score_mae_warning(self):
        raw = [0.96, 0.01, 0.01, 0.01, 0.01]
        records = [record(f"s{i}", "score", raw, 0 if i % 5 else 4) for i in range(40)]
        code, out, err = self.calibrate_from(records, "--path", str(self.path))
        self.assertEqual(code, 0, err)
        fit = fit_temperatures(records)["score"]
        self.assertTrue(fit.mae_worse)
        # The NLL optimum is saved, the MAE cost is only reported.
        self.assertEqual(Calibration.load(self.path).temperature("fake", "score"), round(fit.temperature, 4))
        self.assertIn(f"nex: warning: score: T={fit.temperature:.3f} makes probabilities and confidence honest", err)
        self.assertIn(f"MAE {fit.mae_raw:.3f} -> {fit.mae_fitted:.3f}", err)
        self.assertIn("MAE before -> after", out)

    def test_note_before_writing_the_packaged_file(self):
        packaged = self.dir / "packaged.json"
        with mock.patch.object(nex.calibration, "DEFAULT_PATH", packaged), mock.patch.dict(os.environ):
            os.environ.pop("NEX_CALIBRATION")
            code, out, err = run(["calibrate", "--data", str(self.data / "noul.jsonl")], self.nex)
            self.assertEqual(code, 0, err)
            self.assertIn(f"nex: note: this updates the packaged calibration file {packaged}. "
                          "Pass --path FILE or set NEX_CALIBRATION to write elsewhere.", err.splitlines())
            self.assertTrue(packaged.exists())

            os.environ["NEX_CALIBRATION"] = str(self.path)
            code, out, err = run(["calibrate", "--data", str(self.data / "noul.jsonl")], self.nex)
            self.assertEqual(code, 0, err)
            self.assertNotIn("nex: note:", err)
            # A NEX_CALIBRATION file that does not exist yet is created, not
            # reported as a typo.
            self.assertNotIn("running uncalibrated", err)
            self.assertIn(f"saved to {self.path}", out)
            self.assertTrue(self.path.exists())

            code, out, err = run(["calibrate", "--data", str(self.data / "noul.jsonl"),
                                  "--path", str(self.dir / "other.json")], self.nex)
            self.assertEqual(code, 0, err)
            self.assertNotIn("nex: note:", err)

    def test_malformed_calibration_file_fails_before_asking(self):
        self.path.write_text("{oops")
        self.assertFails(["calibrate", "--data", str(self.data), "--path", str(self.path)],
                         f"calibration file {self.path}")
        self.assertEqual(self.backend.calls, [])


class ModuleEntryTest(unittest.TestCase):
    def test_python_m_nex_help(self):
        # A bad NEX_PORT used to crash every subcommand, --help included.
        env = {**os.environ, "NEX_PORT": "abc", "NEX_HOST": ""}
        proc = subprocess.run([sys.executable, "-m", "nex", "--help"], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for command in ("serve", "ask", "eval", "calibrate"):
            self.assertIn(command, proc.stdout)


if __name__ == "__main__":
    unittest.main()
