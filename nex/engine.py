"""The Nex decision model: state + typed questions in, typed answers out.

For each question Nex builds a prompt whose next token is an option label,
reads the backend's next-token distribution, sums probability over every
spelling of each label, renormalizes over the labels, applies the calibrated
temperature, and derives the typed answer and its confidence. No text is
generated.
"""

import math
import threading
import time
import weakref
from collections import OrderedDict

from . import __version__
from .backend import ContextOverflowError, OllamaBackend
from .calibration import Calibration, apply_temperature
from .confidence import choice_confidence, score_confidence
from .primitives import (
    Choice,
    ChoiceAnswer,
    Diagnostics,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    ValidationError,
    check_depth,
    questions_from_dict,
)
from .prompts import build_prompt, render, token_to_slot

# Names that mean "the default backend". Jev clients send jev-latest unless
# told otherwise. Pinned Jev ids such as jev-1.13.0 are not mapped.
ALIASES = ("nex-latest", "jev-latest", "jev-preview")
# Backends for other model names kept besides the default one.
MAX_CACHED_BACKENDS = 16
# A rendered state longer than num_ctx * this many characters cannot fit the
# context window, so it is rejected without asking the backend.
MAX_CHARS_PER_TOKEN = 64


def run_concurrently(fn, items, max_workers):
    """Return ``[fn(item) for item in items]``, computed on up to
    ``max_workers`` threads.

    The workers are daemon threads and the caller waits with a timeout, so
    Ctrl-C in the calling thread is raised at once instead of waiting for
    calls in flight, and the process can exit without joining them. After
    a failure or an interrupt no new items start. Once the running ones
    finish, the exception of the earliest failed item is raised."""
    items = list(items)
    workers = max(1, min(len(items), max_workers))
    if workers == 1:
        return [fn(item) for item in items]
    results = [None] * len(items)
    errors = {}
    stop = threading.Event()
    lock = threading.Lock()
    pending = iter(range(len(items)))

    def work():
        while not stop.is_set():
            with lock:
                i = next(pending, None)
            if i is None:
                return
            try:
                results[i] = fn(items[i])
            except BaseException as e:
                with lock:
                    errors[i] = e
                stop.set()

    threads = [threading.Thread(target=work, name=f"nex-worker-{n}", daemon=True) for n in range(workers)]
    try:
        # Starting is inside the try too, because an interrupt can arrive
        # while start() waits for a thread that is already running items.
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                # A join without timeout cannot be interrupted on every platform.
                t.join(0.1)
    except BaseException:
        stop.set()
        raise
    if errors:
        raise errors[min(errors)]
    return results


def label_distribution(top_logprobs, question, n_slots):
    """Fold candidate tokens into one probability per answer slot.

    Returns ``(probabilities, label_mass)``. ``label_mass`` is how much of the
    next-token distribution landed on valid labels before renormalizing. A
    label missing from the top candidates gets the smallest listed
    probability, an upper bound on its true value, so it is never treated as
    impossible."""
    mass = [0.0] * n_slots
    for token, logprob in top_logprobs:
        slot = token_to_slot(token, question, n_slots)
        if slot is not None:
            mass[slot] += math.exp(logprob)
    label_mass = sum(mass)
    floor = math.exp(min(lp for _, lp in top_logprobs)) if top_logprobs else 1.0
    mass = [m if m > 0 else floor for m in mass]
    total = sum(mass)
    if total <= 0:
        # Every listed logprob underflowed exp(), so there is no signal.
        return [1 / n_slots] * n_slots, label_mass
    return [m / total for m in mass], label_mass


def build_answer(question, probabilities):
    if isinstance(question, Choice):
        options = list(question.criteria)
        best = max(range(len(options)), key=probabilities.__getitem__)
        return ChoiceAnswer(
            choice=options[best],
            probabilities=dict(zip(options, probabilities)),
            confidence=choice_confidence(probabilities),
        )
    if isinstance(question, Score):
        return ScoreAnswer(
            score=sum(i * p for i, p in enumerate(probabilities)),
            legend={str(i): render(level) for i, level in enumerate(question.criteria)},
            probabilities={str(i): p for i, p in enumerate(probabilities)},
            confidence=score_confidence(probabilities),
        )
    if isinstance(question, Noul):
        return NoulAnswer(noul=probabilities[0])
    raise TypeError(f"unsupported question type {type(question).__name__}")


def validate_state(state):
    if state is None:
        raise ValidationError("state", "is required")
    if not isinstance(state, (str, dict, list)):
        raise ValidationError("state", "must be a string, object, or array")
    check_depth(state, "state")


def check_fits(backend, text):
    """Raise ContextOverflowError when the rendered state is too long for the
    backend's context window, before any request is sent."""
    num_ctx = getattr(backend, "num_ctx", None)
    if num_ctx and len(text) > num_ctx * MAX_CHARS_PER_TOKEN:
        raise ContextOverflowError(
            f"state renders to {len(text)} characters, which cannot fit the {num_ctx}-token context window. "
            "Shorten the state or raise NEX_NUM_CTX."
        )


class Nex:
    """A System One-style decision model on top of a local language model.

    >>> nex = Nex()
    >>> r = nex.system_one(state="I was charged twice!",
    ...                    questions={"billing": Noul("Is this about billing?")})
    >>> r.nouls["billing"].noul
    """

    def __init__(self, backend=None, calibration=None, max_workers=8):
        self.backend = backend or OllamaBackend()
        self.calibration = calibration if calibration is not None else Calibration.load()
        self.max_workers = max_workers
        # Backends for other model names, least recently used first. A backend
        # enters only after a call through it worked, so names that fail
        # (typos, models that are not pulled) are not kept.
        self._backends = OrderedDict()
        # Backends handed out by resolve_backend that have not worked yet.
        # Held weakly, so one is gone as soon as its request is done.
        self._untried = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    @property
    def model_id(self):
        return f"nex-{__version__}+{self.backend.model}"

    def resolve_backend(self, model=None):
        """``None``, an alias such as ``nex-latest`` or ``jev-latest``, or a
        Nex model id use the default backend. Any other name is treated as a
        backend model name, e.g. ``qwen3:4b-instruct``. The backend for a new
        name is cached once a call through it succeeds."""
        if model is None or model in ALIASES or model == self.model_id:
            return self.backend
        if model.startswith("nex-") and "+" in model:
            model = model.split("+", 1)[1]
        if model == self.backend.model:
            return self.backend
        with self._lock:
            backend = self._backends.get(model)
            if backend is not None:
                self._backends.move_to_end(model)
                return backend
            backend = self._untried.get(model)
            if backend is None:
                backend = self.backend.with_model(model)
                self._untried[model] = backend
            return backend

    def _remember(self, backend):
        """Cache a backend from resolve_backend after a call through it worked."""
        if backend is self.backend:
            return
        model = backend.model
        with self._lock:
            if self._backends.get(model) is backend:
                return
            # A backend the caller built and passed in is not ours to keep.
            if self._untried.get(model) is not backend:
                return
            del self._untried[model]
            self._backends[model] = backend
            while len(self._backends) > MAX_CACHED_BACKENDS:
                self._backends.popitem(last=False)

    def ask(self, state, question, backend=None):
        """Evaluate one parsed question. Returns ``(answer, diagnostics)``.

        ``state`` may be raw or already rendered, since rendering a string
        returns it unchanged."""
        backend = backend or self.backend
        validate_state(state)
        text = render(state)
        check_fits(backend, text)
        messages, labels = build_prompt(text, question)
        started = time.perf_counter()
        next_token = backend.next_token(messages)
        latency_ms = (time.perf_counter() - started) * 1000
        self._remember(backend)
        raw, label_mass = label_distribution(next_token.top_logprobs, question, len(labels))
        temperature = self.calibration.temperature(backend.model, question.type)
        answer = build_answer(question, apply_temperature(raw, temperature))
        diagnostics = Diagnostics(
            label_mass=label_mass,
            raw_probabilities=raw,
            prompt_tokens=next_token.prompt_tokens,
            cached_tokens=next_token.cached_tokens,
            latency_ms=latency_ms,
        )
        return answer, diagnostics

    def system_one(self, state, questions, model=None):
        """Evaluate every question independently against the same state.

        ``questions`` maps your ids to ``Choice``/``Score``/``Noul`` objects or
        their wire-format dicts. Questions run concurrently and cannot see
        each other's answers."""
        validate_state(state)
        parsed = questions_from_dict(questions)
        backend = self.resolve_backend(model)
        # Rendered once here. ask() renders again, which returns a string as is.
        text = render(state)
        check_fits(backend, text)
        ids = list(parsed)
        results = run_concurrently(lambda qid: self.ask(text, parsed[qid], backend), ids, self.max_workers)
        answers = {qid: answer for qid, (answer, _) in zip(ids, results)}
        diagnostics = {qid: diag for qid, (_, diag) in zip(ids, results)}
        usage = {
            "input_tokens": sum(d.prompt_tokens for d in diagnostics.values()),
            # One decision token per question.
            "output_tokens": len(ids),
        }
        return SystemOneResponse(
            model=f"nex-{__version__}+{backend.model}", answers=answers, usage=usage, diagnostics=diagnostics
        )
