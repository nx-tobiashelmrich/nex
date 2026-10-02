"""Temperature scaling, so that probabilities match how often answers are right.

Nex cannot retrain the backbone, but it can fit one temperature ``T`` per backend
model and question type on labeled examples and rescale every distribution
as ``p_i ** (1 / T)`` (renormalized). ``T > 1`` softens an overconfident
model, ``T < 1`` sharpens an underconfident one. The ranking of options never
changes, only how much probability the top answer gets.

Fitted temperatures live in ``calibration.json`` next to this file, keyed by
backend model name, so each backbone gets its own. Set ``NEX_CALIBRATION``
to use a different file.
"""

import json
import math
import os
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("calibration.json")
QUESTION_TYPES = ("choice", "score", "noul")
T_MIN, T_MAX = 0.05, 20.0
BOUND_MARGIN = 0.05
_EPS = 1e-12
_warned_missing = set()


def apply_temperature(probabilities, temperature):
    """Rescale a distribution by ``1 / temperature`` in log space."""
    if temperature == 1.0:
        return list(probabilities)
    logits = [math.log(max(p, _EPS)) / temperature for p in probabilities]
    top = max(logits)
    weights = [math.exp(l - top) for l in logits]
    total = sum(weights)
    return [w / total for w in weights]


def negative_log_likelihood(samples, temperature):
    """Mean NLL of the true slots. ``samples`` is a list of
    ``(probabilities, true_index)``."""
    total = 0.0
    for probabilities, truth in samples:
        total -= math.log(max(apply_temperature(probabilities, temperature)[truth], _EPS))
    return total / len(samples)


def fit_temperature(samples, iterations=60):
    """Temperature in ``[T_MIN, T_MAX]`` minimizing NLL, found by
    golden-section search over ``log T`` (NLL is unimodal there)."""
    if not samples:
        return 1.0
    lo, hi = math.log(T_MIN), math.log(T_MAX)
    ratio = (math.sqrt(5) - 1) / 2

    def loss(log_t):
        return negative_log_likelihood(samples, math.exp(log_t))

    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(iterations):
        # Ties keep the lower part. When every case is right with high
        # confidence, NLL rounds to exactly 0 below some T, and the search
        # should end at T_MIN, not at the edge of that flat stretch.
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = loss(b)
    return math.exp((lo + hi) / 2)


def at_bound(temperature):
    """True when a fitted temperature lies within ``BOUND_MARGIN`` of T_MIN
    or T_MAX. The search then ran into the bound instead of finding an
    optimum, which happens when nearly every case is right (T_MIN) or wrong
    (T_MAX), and the value says more about the bounds than about the model."""
    return temperature < T_MIN * (1 + BOUND_MARGIN) or temperature > T_MAX * (1 - BOUND_MARGIN)


def default_path():
    """``NEX_CALIBRATION`` if set, else the calibration.json shipped with the package."""
    return Path(os.environ.get("NEX_CALIBRATION") or DEFAULT_PATH)


def _check_models(models):
    """The first problem in a file's ``models`` map, or None."""
    if not isinstance(models, dict):
        return "models must be an object"
    for model, types in models.items():
        if not isinstance(types, dict):
            return f"models[{model!r}] must be an object"
        for qtype, entry in types.items():
            t = entry.get("temperature") if isinstance(entry, dict) else None
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t < math.inf:
                return f"models[{model!r}][{qtype!r}] needs a positive temperature"
    return None


class Calibration:
    """Per-model, per-question-type temperatures. Unknown pairs use T = 1,
    which leaves the backbone's distribution unchanged."""

    def __init__(self, models=None, path=None):
        self.models = models or {}
        self.path = Path(path) if path else default_path()

    @classmethod
    def load(cls, path=None):
        """Read a calibration file, by default ``default_path()``. A missing
        file gives an empty calibration. When the file came from
        ``NEX_CALIBRATION`` that is likely a typo, so it is reported once on
        stderr. Raises ``ValueError`` naming the file when it is malformed."""
        explicit = bool(path)
        path = Path(path) if path else default_path()
        if not path.exists():
            if not explicit and os.environ.get("NEX_CALIBRATION") and path not in _warned_missing:
                _warned_missing.add(path)
                sys.stderr.write(f"nex: warning: NEX_CALIBRATION file {path} does not exist, running uncalibrated\n")
            return cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            # Covers JSONDecodeError and UnicodeDecodeError.
            raise ValueError(f"calibration file {path}: invalid JSON: {e}") from None
        if not isinstance(data, dict):
            raise ValueError(f"calibration file {path}: expected a JSON object with a models map")
        models = data.get("models", {})
        problem = _check_models(models)
        if problem:
            raise ValueError(f"calibration file {path}: {problem}")
        return cls(models, path)

    def save(self, path=None):
        path = Path(path) if path else self.path
        path.write_text(json.dumps({"models": self.models}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def temperature(self, model, question_type):
        entry = self.models.get(model, {}).get(question_type)
        return entry["temperature"] if entry else 1.0

    def set(self, model, question_type, temperature, n):
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"unknown question type {question_type!r}")
        self.models.setdefault(model, {})[question_type] = {"temperature": round(temperature, 4), "n": n}
