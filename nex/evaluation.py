"""Offline evaluation: run labeled cases through Nex and score the answers.

``collect`` asks the model once per case and keeps the uncalibrated
distribution, so temperatures can be fit and compared afterwards without
another pass over the model. ``metrics`` scores records under given
temperatures, ``fit_type`` fits one question type's temperature, and
``cross_fit`` estimates how fitted temperatures do on cases they were not fit
on.
"""

import json
import math
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from .backend import BackendError
from .calibration import QUESTION_TYPES, apply_temperature, fit_temperature
from .engine import validate_state
from .primitives import Choice, Noul, Score, question_from_dict

DIFFICULTIES = ("easy", "medium", "hard")
CASE_FIELDS = ("id", "domain", "difficulty", "state", "question", "label")
ECE_BINS = 10
SCORE_MAE_TOLERANCE = 0.05
_EPS = 1e-12


def load_cases(paths):
    """Read labeled cases from .jsonl files. A directory stands for every
    ``*.jsonl`` file inside it, in sorted order. Raises ``ValueError`` naming
    the file, line, and id of the first bad case."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            found = sorted(path.glob("*.jsonl"))
            if not found:
                raise ValueError(f"{path}: no .jsonl files")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"{path}: no such file or directory")

    cases, seen = [], {}
    for path in dict.fromkeys(files):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = f"{path}:{lineno}"
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{where}: invalid JSON at column {e.colno}: {e.msg}") from None
                case_id = case.get("id") if isinstance(case, dict) else None
                try:
                    check_case(case)
                except ValueError as e:
                    raise ValueError(f"{where} id {case_id!r}: {e}") from None
                if case_id in seen:
                    raise ValueError(f"{where} id {case_id!r}: duplicate id, first seen at {seen[case_id]}")
                seen[case_id] = where
                cases.append(case)
    return cases


def check_case(case):
    """Validate one case in the dataset format and return its parsed question."""
    if not isinstance(case, dict):
        raise ValueError("a case must be a JSON object")
    missing = [k for k in CASE_FIELDS if k not in case]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    if not isinstance(case["id"], str) or not case["id"].strip():
        raise ValueError("id must be a non-empty string")
    if not isinstance(case["domain"], str):
        raise ValueError("domain must be a string")
    if case["difficulty"] not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
    validate_state(case["state"])
    question = question_from_dict(case["question"])
    check_label(question, case["label"])
    return question


def check_label(question, label):
    if isinstance(question, Choice):
        if not isinstance(label, str) or label not in question.criteria:
            raise ValueError(f"label {label!r} is not one of the options {list(question.criteria)}")
    elif isinstance(question, Score):
        n = len(question.criteria)
        # bool is a subclass of int, but true/false is never a level.
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < n:
            raise ValueError(f"label {label!r} must be an integer level from 0 to {n - 1}")
    elif isinstance(question, Noul):
        if not isinstance(label, bool):
            raise ValueError(f"label {label!r} must be true or false")


def truth_index(question, label):
    """Answer slot of the correct label: the option's position for Choice,
    the level for Score, 0 (yes) or 1 (no) for Noul."""
    if isinstance(question, dict):
        question = question_from_dict(question)
    if isinstance(question, Choice):
        return list(question.criteria).index(label)
    if isinstance(question, Score):
        return int(label)
    if isinstance(question, Noul):
        return 0 if label else 1
    raise TypeError(f"unsupported question type {type(question).__name__}")


def collect(nex, cases, model=None, workers=1, progress=None):
    """Ask every case once and return one record per case, in case order.

    Records keep the raw (uncalibrated) distribution. ``workers`` defaults to
    1 because Ollama runs requests one at a time, and queued requests would
    inflate the latency numbers. ``progress(done, total)`` is called after
    each case."""
    backend = nex.resolve_backend(model)
    total = len(cases)
    lock = threading.Lock()
    done = 0

    def run(case):
        nonlocal done
        record = _run_case(nex, backend, case)
        if progress:
            with lock:
                done += 1
                progress(done, total)
        return record

    if workers <= 1:
        return [run(case) for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, case) for case in cases]
        try:
            return [f.result() for f in futures]
        except BaseException:
            # Without this the pool would still run every queued case before
            # the error reaches the caller.
            for f in futures:
                f.cancel()
            raise


def _run_case(nex, backend, case):
    question = question_from_dict(case["question"])
    try:
        _, diagnostics = nex.ask(case["state"], question, backend)
    except BackendError as e:
        raise type(e)(f"case {case['id']}: {e}") from None
    return {
        "id": case["id"],
        "type": question.type,
        "domain": case.get("domain"),
        "difficulty": case.get("difficulty"),
        "raw": list(diagnostics.raw_probabilities),
        "truth": truth_index(question, case["label"]),
        "label_mass": diagnostics.label_mass,
        "latency_ms": diagnostics.latency_ms,
        "prompt_tokens": diagnostics.prompt_tokens,
        "cached_tokens": diagnostics.cached_tokens,
    }


def temperatures_for(calibration, model):
    """The temperatures a ``Calibration`` holds for one backend model."""
    return {qtype: calibration.temperature(model, qtype) for qtype in QUESTION_TYPES}


def metrics(records, temperatures=None):
    """Score records after rescaling each with its question type's
    temperature (missing types use 1.0). Returns one entry per question type
    present plus ``"all"``."""
    temperatures = temperatures or {}
    pairs = [(r, apply_temperature(r["raw"], temperatures.get(r["type"], 1.0))) for r in records]
    out = _grouped(pairs)
    for qtype in QUESTION_TYPES:
        if qtype in out:
            out[qtype]["temperature"] = temperatures.get(qtype, 1.0)
    return out


def _grouped(pairs):
    out = {}
    for qtype in QUESTION_TYPES:
        group = [(r, p) for r, p in pairs if r["type"] == qtype]
        if not group:
            continue
        out[qtype] = _summarize(group)
        if qtype == "score":
            errors = [abs(sum(i * x for i, x in enumerate(p)) - r["truth"]) for r, p in group]
            out[qtype]["mae"] = sum(errors) / len(errors)
    out["all"] = _summarize(pairs)
    return out


def _argmax(probabilities):
    # max() keeps the first of equal values, so ties go to the lowest index.
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def _summarize(pairs):
    n = len(pairs)
    if not n:
        return {"n": 0}
    hits = 0
    nll = brier = confidence_total = mass = 0.0
    # Per ECE bin: [count, hits, summed confidence]. Bin k covers
    # [k/10, (k+1)/10), and a confidence of exactly 1 goes in the last bin.
    bins = [[0, 0, 0.0] for _ in range(ECE_BINS)]
    for record, probabilities in pairs:
        truth = record["truth"]
        top = _argmax(probabilities)
        confidence = probabilities[top]
        hit = top == truth
        hits += hit
        nll -= math.log(max(probabilities[truth], _EPS))
        brier += sum((p - (i == truth)) ** 2 for i, p in enumerate(probabilities))
        confidence_total += confidence
        mass += record["label_mass"]
        b = bins[min(int(confidence * ECE_BINS), ECE_BINS - 1)]
        b[0] += 1
        b[1] += hit
        b[2] += confidence
    latencies = [r["latency_ms"] for r, _ in pairs]
    return {
        "n": n,
        "accuracy": hits / n,
        "nll": nll / n,
        "brier": brier / n,
        # count/n * |accuracy - mean confidence| summed over bins.
        "ece": sum(abs(h - c) for _, h, c in bins) / n,
        "mean_confidence": confidence_total / n,
        "label_mass_mean": mass / n,
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
    }


def percentile(values, q):
    """Linearly interpolated percentile, ``q`` in [0, 1]."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo = math.floor(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def fold_of(case_id, folds):
    """Stable fold for a case id. crc32 does not change between runs or
    Python versions, unlike ``hash()``."""
    return zlib.crc32(case_id.encode("utf-8")) % folds


class TypeFit(NamedTuple):
    """One question type's fit: the NLL-optimal ``temperature`` and the sample
    count ``n``. For Score, ``mae_raw`` and ``mae_fitted`` are the error of
    the expected level (the ``score`` field) at T = 1 and at the fitted T,
    else None."""

    temperature: float
    n: int
    mae_raw: float | None = None
    mae_fitted: float | None = None

    @property
    def mae_worse(self):
        """True when the fitted Score temperature makes the ``score`` field
        worse than at T = 1 by more than ``SCORE_MAE_TOLERANCE``."""
        return self.mae_raw is not None and self.mae_fitted > self.mae_raw + SCORE_MAE_TOLERANCE


def score_mae(samples, temperature):
    """Mean absolute error of the expected level, the ``score`` field, under
    ``temperature``. ``samples`` is a list of ``(probabilities, true_level)``."""
    total = 0.0
    for probabilities, truth in samples:
        scaled = apply_temperature(probabilities, temperature)
        total += abs(sum(i * p for i, p in enumerate(scaled)) - truth)
    return total / len(samples)


def fit_type(qtype, samples):
    """Fit the temperature for one question type. ``cross_fit`` and
    ``fit_temperatures`` both use this, so the held-out estimate and the
    saved value come from the same procedure.

    The temperature is always the NLL optimum, because calibrated
    probabilities and confidence are what code gates on. For a model whose
    raw Score answers are near one-hot whether right or wrong, that optimum
    spreads probability onto every level and pulls the expected ``score``
    toward the middle of the scale. No temperature fixes both, so Score fits
    also report the MAE at T = 1 and at the fitted T, and the CLI warns."""
    t = fit_temperature(samples)
    if qtype != "score" or not samples:
        return TypeFit(t, len(samples))
    return TypeFit(t, len(samples), score_mae(samples, 1.0), score_mae(samples, t))


def _samples(records, qtype):
    return [(r["raw"], r["truth"]) for r in records if r["type"] == qtype]


def cross_fit(records, folds=2):
    """Held-out estimate of calibrated performance.

    Each fold is scored with temperatures fit by ``fit_type`` on the other
    folds only, so no record helps choose the temperature it is scored under.
    Returns the metrics over all held-out predictions and the mean fitted
    temperature per question type."""
    if folds < 2:
        raise ValueError("cross_fit needs at least 2 folds")
    assigned = [fold_of(r["id"], folds) for r in records]
    held_out = []
    fitted = {}
    for k in range(folds):
        test = [r for r, f in zip(records, assigned) if f == k]
        train = [r for r, f in zip(records, assigned) if f != k]
        for qtype in QUESTION_TYPES:
            group = [r for r in test if r["type"] == qtype]
            if not group:
                continue
            # fit_type returns 1.0 when there is nothing to fit on.
            t = fit_type(qtype, _samples(train, qtype)).temperature
            fitted.setdefault(qtype, []).append(t)
            held_out.extend((r, apply_temperature(r["raw"], t)) for r in group)
    return {
        "folds": folds,
        "metrics": _grouped(held_out),
        "temperatures": {q: sum(ts) / len(ts) for q, ts in fitted.items()},
        "fold_temperatures": fitted,
    }


def fit_temperatures(records):
    """``{qtype: TypeFit}`` from ``fit_type`` on all records of each type."""
    out = {}
    for qtype in QUESTION_TYPES:
        samples = _samples(records, qtype)
        if samples:
            out[qtype] = fit_type(qtype, samples)
    return out
