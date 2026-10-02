"""Engine behavior with scripted backends. No Ollama needed."""

import _thread
import gc
import json
import math
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from nex import __version__
from nex.backend import ContextOverflowError, FakeBackend, ModelNotFoundError, NextToken
from nex.calibration import Calibration, apply_temperature
from nex.confidence import choice_confidence, score_confidence
from nex.engine import (
    MAX_CACHED_BACKENDS,
    Nex,
    build_answer,
    label_distribution,
    run_concurrently,
    validate_state,
)
from nex.primitives import (
    Choice,
    ChoiceAnswer,
    Diagnostics,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    ValidationError,
)

TEAMS = Choice("Which team?", {"billing": "Payments", "tech": "Bugs", "sales": "Deals"})
URGENCY = Score("How urgent?", ["not urgent", "somewhat urgent", "very urgent"])
BILLING = Noul("Is this about billing?")


def lp(p):
    return math.log(p)


def nested(depth, leaf="x"):
    """``leaf`` inside ``depth`` levels of alternating arrays and objects."""
    value = leaf
    for i in range(depth):
        value = [value] if i % 2 else {"k": value}
    return value


class ModelBackend(FakeBackend):
    """A FakeBackend that fails for the model names in ``missing``, like
    Ollama for a model that is not pulled. ``created`` records every
    with_model call, shared across the family."""

    def __init__(self, responder, model="fake", missing=(), created=None):
        super().__init__(responder, model)
        self.missing = set(missing)
        self.created = created if created is not None else []

    def with_model(self, model):
        self.created.append(model)
        return ModelBackend(self.responder, model, self.missing, self.created)

    def next_token(self, messages):
        if self.model in self.missing:
            raise ModelNotFoundError(f'model "{self.model}" not found, try pulling it first')
        return super().next_token(messages)


def question_text(messages):
    """The QUESTION block of a built prompt, used to script per-question replies."""
    user = messages[-1]["content"]
    return user.split("QUESTION:\n", 1)[1].split("\n\nOPTIONS:", 1)[0]


class LabelDistributionTest(unittest.TestCase):
    def test_folds_spellings_of_a_label(self):
        top = [("Yes", lp(0.5)), (" yes", lp(0.2)), ("YES", lp(0.05)), ("No", lp(0.1)), ("Maybe", lp(0.15))]
        probs, mass = label_distribution(top, BILLING, 2)
        self.assertAlmostEqual(mass, 0.85)
        self.assertAlmostEqual(probs[0], 0.75 / 0.85)
        self.assertAlmostEqual(probs[1], 0.10 / 0.85)

    def test_folds_choice_spellings(self):
        top = [("A", lp(0.4)), (" A", lp(0.1)), ("a", lp(0.05)), ("B)", lp(0.25)), ("C", lp(0.2))]
        probs, mass = label_distribution(top, TEAMS, 3)
        self.assertAlmostEqual(mass, 1.0)
        for got, want in zip(probs, [0.55, 0.25, 0.2]):
            self.assertAlmostEqual(got, want)

    def test_missing_label_gets_smallest_listed_probability(self):
        top = [("A", lp(0.6)), ("B", lp(0.3)), ("The", lp(0.05)), ("I", lp(0.02))]
        probs, mass = label_distribution(top, TEAMS, 3)
        self.assertAlmostEqual(mass, 0.9)
        total = 0.6 + 0.3 + 0.02
        for got, want in zip(probs, [0.6 / total, 0.3 / total, 0.02 / total]):
            self.assertAlmostEqual(got, want)
        self.assertGreater(probs[2], 0)

    def test_zero_label_mass_gives_uniform(self):
        top = [("The", lp(0.7)), ("I", lp(0.2)), ("Hmm", lp(0.1))]
        probs, mass = label_distribution(top, URGENCY, 3)
        self.assertEqual(mass, 0.0)
        for p in probs:
            self.assertAlmostEqual(p, 1 / 3)

    def test_empty_top_logprobs_gives_uniform(self):
        probs, mass = label_distribution([], BILLING, 2)
        self.assertEqual(mass, 0.0)
        self.assertEqual(probs, [0.5, 0.5])

    def test_underflowing_logprobs_do_not_crash(self):
        # exp(-1000) is 0.0 in floating point, so every slot would be zero.
        probs, mass = label_distribution([("The", -1000.0)], BILLING, 2)
        self.assertEqual(probs, [0.5, 0.5])
        probs, _ = label_distribution([("A", -1000.0), ("x", -1001.0)], TEAMS, 3)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_sums_to_one(self):
        top = [("0", lp(0.3)), ("1", lp(0.2)), (" 2", lp(0.1)), ("x", lp(0.01))]
        probs, _ = label_distribution(top, URGENCY, 3)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_pronoun_i_is_not_an_option(self):
        # With nine options the ninth label used to be I, so a reply starting
        # "I cannot" counted as that option with full label mass.
        q = Choice("x", {f"o{i}": None for i in range(9)})
        top = [("I", lp(0.9)), (" I", lp(0.04)), ("J", lp(0.05)), ("A", lp(0.01))]
        probs, mass = label_distribution(top, q, 9)
        self.assertAlmostEqual(mass, 0.06)
        self.assertEqual(max(range(9), key=probs.__getitem__), 8)
        self.assertAlmostEqual(probs[8], 0.05 / 0.13)


class BuildAnswerTest(unittest.TestCase):
    def test_choice(self):
        a = build_answer(TEAMS, [0.2, 0.7, 0.1])
        self.assertIsInstance(a, ChoiceAnswer)
        self.assertEqual(a.choice, "tech")
        self.assertEqual(a.probabilities, {"billing": 0.2, "tech": 0.7, "sales": 0.1})
        self.assertEqual(list(a.probabilities), ["billing", "tech", "sales"])
        self.assertAlmostEqual(a.confidence, choice_confidence([0.2, 0.7, 0.1]))

    def test_choice_tie_picks_first_option(self):
        a = build_answer(TEAMS, [0.4, 0.4, 0.2])
        self.assertEqual(a.choice, "billing")
        a = build_answer(TEAMS, [0.2, 0.4, 0.4])
        self.assertEqual(a.choice, "tech")
        a = build_answer(TEAMS, [1 / 3] * 3)
        self.assertEqual(a.choice, "billing")
        self.assertAlmostEqual(a.confidence, 0.0)

    def test_score(self):
        a = build_answer(URGENCY, [0.1, 0.2, 0.7])
        self.assertIsInstance(a, ScoreAnswer)
        self.assertAlmostEqual(a.score, 1.6)
        self.assertEqual(a.legend, {"0": "not urgent", "1": "somewhat urgent", "2": "very urgent"})
        self.assertEqual(a.probabilities, {"0": 0.1, "1": 0.2, "2": 0.7})
        self.assertAlmostEqual(a.confidence, score_confidence([0.1, 0.2, 0.7]))

    def test_score_can_land_between_levels(self):
        a = build_answer(URGENCY, [0.0, 0.5, 0.5])
        self.assertAlmostEqual(a.score, 1.5)
        self.assertAlmostEqual(a.confidence, 0.25)

    def test_score_legend_renders_object_levels(self):
        q = Score("x", [{"label": "low"}, "high"])
        a = build_answer(q, [0.5, 0.5])
        self.assertEqual(a.legend, {"0": json.dumps({"label": "low"}, indent=2), "1": "high"})

    def test_noul(self):
        a = build_answer(BILLING, [0.8, 0.2])
        self.assertIsInstance(a, NoulAnswer)
        self.assertEqual(a.noul, 0.8)

    def test_unsupported(self):
        with self.assertRaises(TypeError):
            build_answer("noul", [0.5, 0.5])


class ValidateStateTest(unittest.TestCase):
    def test_valid_states(self):
        for state in ["text", {"a": 1}, [1, 2], {}, []]:
            validate_state(state)

    def test_missing_state(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_state(None)
        self.assertEqual(ctx.exception.field, "state")
        self.assertIn("required", ctx.exception.message)

    def test_wrong_types(self):
        for state in [5, 1.5, True, ("a",), b"bytes"]:
            with self.assertRaises(ValidationError) as ctx:
                validate_state(state)
            self.assertEqual(ctx.exception.field, "state")

    def test_nesting_depth_limit(self):
        validate_state(nested(32))
        for depth in [33, 100_000]:
            with self.assertRaises(ValidationError) as ctx:
                validate_state(nested(depth))
            self.assertEqual(ctx.exception.field, "state")
            self.assertIn("32 levels", ctx.exception.message)


def routed_responder(messages):
    """Answer each question differently so mix-ups between ids show up."""
    q = question_text(messages)
    if q == "Which team?":
        return [("B", lp(0.7)), (" B", lp(0.1)), ("A", lp(0.15)), ("C", lp(0.05))]
    if q == "How urgent?":
        return NextToken([("3", lp(0.6)), ("2", lp(0.3)), ("1", lp(0.1))], prompt_tokens=100, cached_tokens=40)
    if q == "Is this about billing?":
        return NextToken([("No", lp(0.9)), ("Yes", lp(0.1))], prompt_tokens=70, cached_tokens=40)
    if q == "Is the customer angry?":
        return NextToken([("Yes", lp(0.6)), ("No", lp(0.4))], prompt_tokens=30, cached_tokens=0)
    raise AssertionError(f"unexpected question {q!r}")


class NexTest(unittest.TestCase):
    def nex(self, responder=routed_responder, calibration=None, **kwargs):
        backend = FakeBackend(responder)
        return Nex(backend, calibration if calibration is not None else Calibration(), **kwargs), backend

    def test_system_one_keys_answers_by_question(self):
        nex, backend = self.nex()
        questions = {"dept": TEAMS, "urgency": URGENCY, "billing": BILLING, "angry": Noul("Is the customer angry?")}
        r = nex.system_one("Your app crashes when I pay!", questions)
        self.assertIsInstance(r, SystemOneResponse)
        self.assertEqual(list(r.answers), ["dept", "urgency", "billing", "angry"])
        self.assertEqual(r.answers["dept"].choice, "tech")
        self.assertAlmostEqual(r.answers["dept"].probabilities["tech"], 0.8)
        self.assertAlmostEqual(r.answers["urgency"].score, 1.5)
        self.assertAlmostEqual(r.answers["billing"].noul, 0.1)
        self.assertAlmostEqual(r.answers["angry"].noul, 0.6)
        self.assertEqual(set(r.choices), {"dept"})
        self.assertEqual(set(r.scores), {"urgency"})
        self.assertEqual(set(r.nouls), {"billing", "angry"})
        self.assertEqual(len(backend.calls), 4)

    def test_questions_run_concurrently(self):
        # Every call blocks until all three are in flight, so a serial run
        # would break the barrier.
        barrier = threading.Barrier(3, timeout=5)

        def responder(messages):
            barrier.wait()
            return routed_responder(messages)

        nex, backend = self.nex(responder)
        r = nex.system_one("state", {"dept": TEAMS, "urgency": URGENCY, "billing": BILLING})
        self.assertEqual(r.answers["dept"].choice, "tech")
        self.assertEqual(len(backend.calls), 3)

    def test_questions_run_on_daemon_threads(self):
        # Non-daemon workers are joined at exit, which held up Ctrl-C.
        daemon = []

        def responder(messages):
            daemon.append(threading.current_thread().daemon)
            return routed_responder(messages)

        nex, _ = self.nex(responder)
        nex.system_one("state", {"dept": TEAMS, "urgency": URGENCY, "billing": BILLING})
        self.assertEqual(daemon, [True, True, True])

    def test_state_is_rendered_once_per_call(self):
        state = {"ticket": {"subject": "Charged twice", "lines": ["a", "b"]}}
        nex, backend = self.nex()
        with mock.patch("json.dumps", wraps=json.dumps) as dumps:
            nex.system_one(state, {"dept": TEAMS, "urgency": URGENCY, "billing": BILLING})
        self.assertEqual(sum(1 for c in dumps.call_args_list if c.args and c.args[0] is state), 1)
        prefix = "STATE:\n" + json.dumps(state, indent=2) + "\n\nQUESTION:\n"
        self.assertEqual(len(backend.calls), 3)
        for messages in backend.calls:
            self.assertTrue(messages[1]["content"].startswith(prefix))

    def test_deep_state_is_rejected_before_rendering(self):
        nex, backend = self.nex()
        with mock.patch("json.dumps", wraps=json.dumps) as dumps:
            with self.assertRaises(ValidationError) as ctx:
                nex.system_one(nested(33), {"billing": BILLING})
        self.assertEqual(ctx.exception.field, "state")
        dumps.assert_not_called()
        self.assertEqual(backend.calls, [])

    def test_single_worker_runs_serially(self):
        nex, _ = self.nex(max_workers=1)
        r = nex.system_one("state", {"dept": TEAMS, "billing": BILLING})
        self.assertEqual(r.answers["dept"].choice, "tech")
        self.assertAlmostEqual(r.answers["billing"].noul, 0.1)

    def test_usage_sums_prompt_tokens(self):
        nex, _ = self.nex()
        r = nex.system_one("state", {"urgency": URGENCY, "billing": BILLING, "angry": Noul("Is the customer angry?")})
        self.assertEqual(r.usage, {"input_tokens": 200, "output_tokens": 3})
        self.assertEqual(r.diagnostics["urgency"].prompt_tokens, 100)
        self.assertEqual(r.diagnostics["urgency"].cached_tokens, 40)

    def test_usage_with_fake_token_estimate(self):
        nex, backend = self.nex(lambda m: [("Yes", lp(0.9))])
        r = nex.system_one("state", {"a": BILLING, "b": Noul("Second?")})
        expected = sum(sum(len(m["content"]) // 4 for m in call) for call in backend.calls)
        self.assertEqual(r.usage["input_tokens"], expected)
        self.assertGreater(expected, 0)

    def test_response_model_and_dict(self):
        nex, _ = self.nex()
        r = nex.system_one("state", {"billing": BILLING})
        self.assertEqual(r.model, f"nex-{__version__}+fake")
        self.assertEqual(r.model, "nex-0.1.0+fake")
        d = r.to_dict()
        self.assertEqual(d, {"model": "nex-0.1.0+fake", "answers": {"billing": {"type": "noul", "noul": 0.1}},
                             "usage": {"input_tokens": 70, "output_tokens": 1}})
        json.dumps(r.to_dict(include_diagnostics=True))

    def test_wire_format_questions(self):
        nex, _ = self.nex()
        r = nex.system_one(
            {"ticket": "app crashes"},
            {
                "dept": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "Payments", "tech": "Bugs", "sales": "Deals"}},
                "urgency": {"type": "score", "instructions": "How urgent?",
                            "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
                "billing": {"type": "noul", "instructions": "Is this about billing?"},
                "angry": Noul("Is the customer angry?"),
            },
        )
        self.assertEqual(r.answers["dept"].choice, "tech")
        self.assertEqual(r.answers["urgency"].legend["2"], "very urgent")
        self.assertIsInstance(r.answers["billing"], NoulAnswer)
        self.assertIsInstance(r.answers["angry"], NoulAnswer)

    def test_validation_happens_before_any_backend_call(self):
        nex, backend = self.nex()
        with self.assertRaises(ValidationError) as ctx:
            nex.system_one(None, {"billing": BILLING})
        self.assertEqual(ctx.exception.field, "state")
        with self.assertRaises(ValidationError) as ctx:
            nex.system_one(42, {"billing": BILLING})
        self.assertEqual(ctx.exception.field, "state")
        with self.assertRaises(ValidationError) as ctx:
            nex.system_one("state", {})
        self.assertEqual(ctx.exception.field, "questions")
        with self.assertRaises(ValidationError) as ctx:
            nex.system_one("state", {"billing": BILLING, "dept": {"type": "choice", "instructions": "x",
                                                                  "criteria": {"only": None}}})
        self.assertEqual(ctx.exception.field, "questions.dept.criteria")
        self.assertEqual(backend.calls, [])

    def test_ask_returns_answer_and_diagnostics(self):
        nex, backend = self.nex()
        answer, diag = nex.ask("state", URGENCY)
        self.assertIsInstance(answer, ScoreAnswer)
        self.assertIsInstance(diag, Diagnostics)
        self.assertAlmostEqual(diag.label_mass, 1.0)
        self.assertEqual(diag.prompt_tokens, 100)
        self.assertGreaterEqual(diag.latency_ms, 0)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][0]["role"], "system")

    def test_ask_reports_label_mass(self):
        nex, _ = self.nex(lambda m: [("Yes", lp(0.3)), ("No", lp(0.1)), ("I", lp(0.6))])
        answer, diag = nex.ask("state", BILLING)
        self.assertAlmostEqual(diag.label_mass, 0.4)
        self.assertAlmostEqual(answer.noul, 0.75)

    def test_ask_with_explicit_backend(self):
        nex, default = self.nex()
        other = FakeBackend(lambda m: [("Yes", lp(0.99)), ("No", lp(0.01))], model="other")
        answer, _ = nex.ask("state", BILLING, backend=other)
        self.assertAlmostEqual(answer.noul, 0.99)
        self.assertEqual(default.calls, [])
        self.assertEqual(len(other.calls), 1)
        # A backend the caller built is used once, not cached under its name.
        self.assertIsNot(nex.resolve_backend("other"), other)

    def test_ask_accepts_a_raw_state(self):
        nex, backend = self.nex()
        state = {"ticket": "charged twice"}
        nex.ask(state, BILLING)
        self.assertTrue(backend.calls[0][1]["content"].startswith("STATE:\n" + json.dumps(state, indent=2) + "\n\n"))
        with self.assertRaises(ValidationError) as ctx:
            nex.ask(nested(33), BILLING)
        self.assertEqual(ctx.exception.field, "state")
        self.assertEqual(len(backend.calls), 1)


class ContextSizeTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend(lambda m: [("Yes", lp(0.9)), ("No", lp(0.1))])
        self.backend.num_ctx = 10
        self.nex = Nex(self.backend, Calibration())

    def test_state_longer_than_64_chars_per_token_is_rejected_without_a_call(self):
        self.nex.system_one("x" * 640, {"a": BILLING})
        self.assertEqual(len(self.backend.calls), 1)
        with self.assertRaises(ContextOverflowError) as ctx:
            self.nex.system_one("x" * 641, {"a": BILLING, "b": Noul("Second?")})
        self.assertIn("641 characters", str(ctx.exception))
        self.assertIn("10-token", str(ctx.exception))
        self.assertEqual(len(self.backend.calls), 1)

    def test_rendered_length_counts(self):
        # The raw text is short enough, but its JSON rendering is not.
        state = {"text": "x" * 630}
        with self.assertRaises(ContextOverflowError):
            self.nex.system_one(state, {"a": BILLING})
        self.assertEqual(self.backend.calls, [])

    def test_ask_checks_too(self):
        with self.assertRaises(ContextOverflowError):
            self.nex.ask("x" * 641, BILLING)
        self.assertEqual(self.backend.calls, [])

    def test_backend_without_num_ctx_is_not_checked(self):
        backend = FakeBackend(lambda m: [("Yes", lp(0.9))])
        Nex(backend, Calibration()).system_one("x" * 100_000, {"a": BILLING})
        self.assertEqual(len(backend.calls), 1)


class RunConcurrentlyTest(unittest.TestCase):
    def test_results_keep_input_order(self):
        def fn(i):
            time.sleep(0.01 * (5 - i))
            return i * 10

        self.assertEqual(run_concurrently(fn, range(6), 3), [0, 10, 20, 30, 40, 50])

    def test_empty_and_single_worker(self):
        self.assertEqual(run_concurrently(lambda x: x, [], 4), [])
        threads = []

        def fn(x):
            threads.append(threading.current_thread())
            return x + 1

        self.assertEqual(run_concurrently(fn, [1, 2, 3], 1), [2, 3, 4])
        self.assertEqual(threads, [threading.current_thread()] * 3)

    def test_daemon_threads_up_to_max_workers(self):
        lock = threading.Lock()
        active, peak, daemon = [0], [0], []

        def fn(i):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                daemon.append(threading.current_thread().daemon)
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return i

        self.assertEqual(run_concurrently(fn, range(8), 3), list(range(8)))
        self.assertEqual(peak[0], 3)
        self.assertEqual(daemon, [True] * 8)

    def test_raises_earliest_failure_after_running_items_finish(self):
        # Item 1 fails first, item 0 fails later. The caller sees item 0's
        # error, and only after every running item is done.
        barrier = threading.Barrier(3, timeout=5)
        finished = []

        def fn(i):
            barrier.wait()
            if i == 1:
                raise ValueError("early")
            time.sleep(0.05)
            finished.append(i)
            if i == 0:
                raise KeyError("late")
            return i

        with self.assertRaises(KeyError):
            run_concurrently(fn, range(3), 3)
        self.assertEqual(sorted(finished), [0, 2])

    def test_no_new_items_start_after_a_failure(self):
        barrier = threading.Barrier(2, timeout=5)
        calls = []

        def fn(i):
            calls.append(i)
            if i < 2:
                barrier.wait()
            if i == 0:
                raise ValueError("boom")
            time.sleep(0.1)
            return i

        with self.assertRaises(ValueError):
            run_concurrently(fn, range(6), 2)
        self.assertEqual(sorted(calls), [0, 1])

    def test_keyboard_interrupt_is_not_blocked_by_calls_in_flight(self):
        release = threading.Event()
        in_flight = threading.Semaphore(0)
        calls = []

        def fn(i):
            calls.append(i)
            in_flight.release()
            release.wait(10)
            return i

        def interrupt():
            for _ in range(2):
                in_flight.acquire(timeout=5)
            _thread.interrupt_main()

        # unittest's --catch mode swallows the first SIGINT, so use the default.
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        threading.Thread(target=interrupt, daemon=True).start()
        started = time.monotonic()
        try:
            with self.assertRaises(KeyboardInterrupt):
                run_concurrently(fn, range(5), 2)
            self.assertLess(time.monotonic() - started, 3)
        finally:
            release.set()
            signal.signal(signal.SIGINT, previous)
        for t in threading.enumerate():
            if t.name.startswith("nex-worker-"):
                t.join(5)
        # The interrupt also stops workers from starting new items.
        self.assertEqual(sorted(calls), [0, 1])


class CalibrationInEngineTest(unittest.TestCase):
    def setUp(self):
        self.responder = lambda m: [("A", lp(0.7)), ("B", lp(0.2)), ("C", lp(0.1))]

    def test_temperature_changes_probabilities_but_not_argmax(self):
        calibration = Calibration()
        calibration.set("fake", "choice", 2.0, n=50)
        nex = Nex(FakeBackend(self.responder), calibration)
        r = nex.system_one("state", {"dept": TEAMS})
        answer, diag = r.answers["dept"], r.diagnostics["dept"]
        raw = [0.7, 0.2, 0.1]
        for got, want in zip(diag.raw_probabilities, raw):
            self.assertAlmostEqual(got, want)
        expected = apply_temperature(raw, 2.0)
        for got, want in zip(answer.probabilities.values(), expected):
            self.assertAlmostEqual(got, want)
        self.assertEqual(answer.choice, "billing")
        self.assertLess(answer.probabilities["billing"], 0.7)
        self.assertLess(answer.confidence, choice_confidence(raw))
        self.assertAlmostEqual(sum(answer.probabilities.values()), 1.0)

    def test_sharpening_temperature(self):
        calibration = Calibration()
        calibration.set("fake", "choice", 0.5, n=50)
        answer, diag = Nex(FakeBackend(self.responder), calibration).ask("state", TEAMS)
        self.assertGreater(answer.probabilities["billing"], 0.7)
        self.assertAlmostEqual(diag.raw_probabilities[0], 0.7)

    def test_temperature_is_per_question_type(self):
        calibration = Calibration()
        calibration.set("fake", "noul", 3.0, n=50)
        nex = Nex(FakeBackend(self.responder), calibration)
        answer, _ = nex.ask("state", TEAMS)
        self.assertAlmostEqual(answer.probabilities["billing"], 0.7)
        noul, diag = Nex(FakeBackend(lambda m: [("Yes", lp(0.9)), ("No", lp(0.1))]), calibration).ask("s", BILLING)
        self.assertAlmostEqual(diag.raw_probabilities[0], 0.9)
        self.assertAlmostEqual(noul.noul, apply_temperature([0.9, 0.1], 3.0)[0])
        self.assertLess(noul.noul, 0.9)
        self.assertGreater(noul.noul, 0.5)

    def test_temperature_is_per_backend_model(self):
        calibration = Calibration()
        calibration.set("other", "choice", 4.0, n=50)
        nex = Nex(FakeBackend(self.responder), calibration)
        default = nex.system_one("state", {"dept": TEAMS})
        other = nex.system_one("state", {"dept": TEAMS}, model="other")
        self.assertAlmostEqual(default.answers["dept"].probabilities["billing"], 0.7)
        self.assertAlmostEqual(other.answers["dept"].probabilities["billing"], apply_temperature([0.7, 0.2, 0.1], 4.0)[0])
        self.assertAlmostEqual(other.diagnostics["dept"].raw_probabilities[0], 0.7)

    def test_explicit_empty_calibration_is_used(self):
        calibration = Calibration()
        nex = Nex(FakeBackend(self.responder), calibration)
        self.assertIs(nex.calibration, calibration)


class ModelResolutionTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend(lambda m: [("Yes", lp(0.8)), ("No", lp(0.2))])
        self.nex = Nex(self.backend, Calibration())

    def test_model_id(self):
        self.assertEqual(self.nex.model_id, "nex-0.1.0+fake")

    def test_default_aliases(self):
        for name in [None, "nex-latest", "jev-latest", "jev-preview", "nex-0.1.0+fake", "nex-0.2.0+fake", "fake"]:
            self.assertIs(self.nex.resolve_backend(name), self.backend, name)

    def test_jev_latest_answers_with_the_default_backend(self):
        r = self.nex.system_one("state", {"billing": BILLING}, model="jev-latest")
        self.assertEqual(r.model, "nex-0.1.0+fake")
        self.assertEqual(len(self.backend.calls), 1)

    def test_pinned_jev_ids_are_backend_model_names(self):
        for name in ["jev-1.13.0", "jev-1"]:
            b = self.nex.resolve_backend(name)
            self.assertIsNot(b, self.backend)
            self.assertEqual(b.model, name)

    def test_bare_name_uses_with_model_and_is_cached(self):
        b = self.nex.resolve_backend("qwen3:4b-instruct")
        self.assertIsNot(b, self.backend)
        self.assertIsInstance(b, FakeBackend)
        self.assertEqual(b.model, "qwen3:4b-instruct")
        self.assertIs(self.nex.resolve_backend("qwen3:4b-instruct"), b)
        self.assertIs(self.nex.resolve_backend("nex-0.1.0+qwen3:4b-instruct"), b)

    def test_successful_backend_is_kept_after_the_call(self):
        self.nex.system_one("state", {"billing": BILLING}, model="qwen3:4b-instruct")
        gc.collect()
        b = self.nex.resolve_backend("qwen3:4b-instruct")
        self.assertEqual(len(b.calls), 1)
        self.assertIs(self.nex.resolve_backend("qwen3:4b-instruct"), b)

    def test_nex_id_with_other_backend_model(self):
        b = self.nex.resolve_backend("nex-0.1.0+llama3.2:3b")
        self.assertEqual(b.model, "llama3.2:3b")
        self.assertIs(self.nex.resolve_backend("llama3.2:3b"), b)

    def test_system_one_with_model(self):
        r = self.nex.system_one("state", {"billing": BILLING}, model="qwen3:4b-instruct")
        self.assertEqual(r.model, "nex-0.1.0+qwen3:4b-instruct")
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(len(self.nex.resolve_backend("qwen3:4b-instruct").calls), 1)
        r = self.nex.system_one("state", {"billing": BILLING}, model="nex-latest")
        self.assertEqual(r.model, "nex-0.1.0+fake")
        self.assertEqual(len(self.backend.calls), 1)


class BackendCacheTest(unittest.TestCase):
    def setUp(self):
        self.created = []
        self.backend = ModelBackend(lambda m: [("Yes", lp(0.8)), ("No", lp(0.2))], missing={"typo"}, created=self.created)
        self.nex = Nex(self.backend, Calibration())

    def ask(self, model):
        return self.nex.system_one("state", {"billing": BILLING, "again": Noul("Second?")}, model=model)

    def test_failing_name_is_not_cached(self):
        for _ in range(3):
            with self.assertRaises(ModelNotFoundError):
                self.ask("typo")
            gc.collect()
        self.assertEqual(self.created, ["typo"] * 3)
        self.assertNotIn("typo", self.nex._backends)
        self.assertEqual(len(self.nex._untried), 0)

    def test_successful_name_is_created_once(self):
        for _ in range(3):
            self.ask("good")
            gc.collect()
        self.assertEqual(self.created, ["good"])
        self.assertEqual(len(self.nex.resolve_backend("good").calls), 6)

    def test_at_most_16_backends_least_recently_used_evicted(self):
        self.assertEqual(MAX_CACHED_BACKENDS, 16)
        for i in range(MAX_CACHED_BACKENDS):
            self.ask(f"m{i}")
        # Using m0 again makes m1 the least recently used.
        m0 = self.nex.resolve_backend("m0")
        self.ask(f"m{MAX_CACHED_BACKENDS}")
        gc.collect()
        self.assertEqual(len(self.nex._backends), MAX_CACHED_BACKENDS)
        self.assertNotIn("m1", self.nex._backends)
        self.assertIs(self.nex.resolve_backend("m0"), m0)
        created = len(self.created)
        self.nex.resolve_backend("m2")
        self.assertEqual(len(self.created), created)
        self.nex.resolve_backend("m1")
        self.assertEqual(len(self.created), created + 1)

    def test_default_backend_is_never_evicted(self):
        for i in range(MAX_CACHED_BACKENDS * 2):
            self.ask(f"m{i}")
        for name in [None, "nex-latest", "jev-latest", "fake", self.nex.model_id]:
            self.assertIs(self.nex.resolve_backend(name), self.backend)
        self.assertNotIn("fake", self.nex._backends)

    def test_concurrent_requests_for_a_new_name_share_one_backend(self):
        first = self.nex.resolve_backend("new")
        self.assertIs(self.nex.resolve_backend("new"), first)
        self.ask("new")
        self.assertIs(self.nex.resolve_backend("new"), first)
        self.assertEqual(self.created, ["new"])


class PackageTest(unittest.TestCase):
    def test_old_python_gets_a_clear_import_error(self):
        code = (
            "import sys\n"
            "sys.version_info = (3, 9, 6, 'final', 0)\n"
            "try:\n"
            "    import nex\n"
            "except ImportError as e:\n"
            "    print(e)\n"
        )
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "Nex needs Python 3.10 or newer, this is 3.9.6", out.stderr)


if __name__ == "__main__":
    unittest.main()
