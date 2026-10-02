"""Typed questions (Choice, Score, Noul) and the typed answers they return.

The wire format is specified in docs/SPEC.md. It matches Jev's, so the same
request JSON works against Jev and Nex.
"""

from dataclasses import dataclass, field

# Ollama returns at most 20 top logprobs per position, and every option label
# must be visible in that list, so a Choice can have at most 20 options.
MAX_CHOICE_OPTIONS = 20
# Score levels are labelled with the single-token digits 0-9.
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
# Every question is one backend call, so a request must not fan out without bound.
MAX_QUESTIONS = 256
# Objects and arrays are rendered as indented JSON, which repeats the indent on
# every line, so the prompt grows with the square of the nesting depth.
MAX_DEPTH = 32


class ValidationError(ValueError):
    """A request or question is malformed. `field` points at the offending
    part, e.g. ``questions.department.criteria``."""

    def __init__(self, field, message):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def check_depth(value, where):
    """Reject objects and arrays nested more than MAX_DEPTH levels deep.

    Walks an explicit stack instead of recursing, so arbitrarily deep input
    cannot hit the recursion limit. A cycle counts as too deep."""
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        else:
            continue
        if depth > MAX_DEPTH:
            raise ValidationError(where, f"must not nest objects or arrays more than {MAX_DEPTH} levels deep")
        stack.extend((child, depth + 1) for child in children if isinstance(child, (dict, list, tuple)))


def _check_text(value, where):
    # Instructions and criteria may be a string, an object, or an array.
    if isinstance(value, str):
        if not value.strip():
            raise ValidationError(where, "must not be empty")
    elif isinstance(value, (dict, list)):
        if not value:
            raise ValidationError(where, "must not be empty")
        check_depth(value, where)
    else:
        raise ValidationError(where, "must be a string, object, or array")


@dataclass(frozen=True)
class Choice:
    """Pick one option from a set. ``criteria`` maps option -> description
    (or None when the option name says it all)."""

    instructions: object
    criteria: dict
    type = "choice"

    def validate(self, where="question"):
        _check_text(self.instructions, f"{where}.instructions")
        if not isinstance(self.criteria, dict):
            raise ValidationError(f"{where}.criteria", "must be an object mapping option -> description")
        if len(self.criteria) < 2:
            raise ValidationError(f"{where}.criteria", "needs at least 2 options")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValidationError(f"{where}.criteria", f"at most {MAX_CHOICE_OPTIONS} options are supported")
        for option, description in self.criteria.items():
            if not isinstance(option, str) or not option.strip():
                raise ValidationError(f"{where}.criteria", "option names must be non-empty strings")
            if description is not None:
                _check_text(description, f"{where}.criteria.{option}")

    def to_dict(self):
        return {"type": self.type, "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class Score:
    """Rate the state on ordered levels. ``criteria`` is a list of level
    descriptions, lowest first."""

    instructions: object
    criteria: list
    type = "score"

    def validate(self, where="question"):
        _check_text(self.instructions, f"{where}.instructions")
        if not isinstance(self.criteria, list):
            raise ValidationError(f"{where}.criteria", "must be an array of level descriptions")
        if not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
            raise ValidationError(
                f"{where}.criteria", f"needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels"
            )
        for i, level in enumerate(self.criteria):
            _check_text(level, f"{where}.criteria[{i}]")

    def to_dict(self):
        return {"type": self.type, "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class Noul:
    """Is this statement true? ``criteria`` may describe what yes ("true")
    and no ("false") mean."""

    instructions: object
    criteria: dict | None = None
    type = "noul"

    def validate(self, where="question"):
        _check_text(self.instructions, f"{where}.instructions")
        if self.criteria is None:
            return
        if not isinstance(self.criteria, dict):
            raise ValidationError(f"{where}.criteria", 'must be an object with optional "true" and "false" keys')
        unknown = set(self.criteria) - {"true", "false"}
        if unknown:
            raise ValidationError(f"{where}.criteria", f"unknown keys {sorted(unknown)}; use \"true\" and \"false\"")
        for key, description in self.criteria.items():
            if description is not None:
                _check_text(description, f"{where}.criteria.{key}")

    def to_dict(self):
        d = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


QUESTION_TYPES = {"choice": Choice, "score": Score, "noul": Noul}


def question_from_dict(data, where="question"):
    """Parse one wire-format question and validate it."""
    if isinstance(data, (Choice, Score, Noul)):
        data.validate(where)
        return data
    if not isinstance(data, dict):
        raise ValidationError(where, "must be an object")
    kind = data.get("type")
    # A list or object here would raise TypeError on the dict lookup.
    if not isinstance(kind, str) or kind not in QUESTION_TYPES:
        raise ValidationError(f"{where}.type", 'must be "choice", "score", or "noul"')
    if "instructions" not in data:
        raise ValidationError(f"{where}.instructions", "is required")
    if kind != "noul" and "criteria" not in data:
        raise ValidationError(f"{where}.criteria", "is required")
    unknown = set(data) - {"type", "instructions", "criteria"}
    if unknown:
        raise ValidationError(where, f"unknown fields {sorted(unknown)}")
    question = QUESTION_TYPES[kind](instructions=data["instructions"], criteria=data.get("criteria"))
    question.validate(where)
    return question


def questions_from_dict(data):
    """Parse the ``questions`` map of a request."""
    if not isinstance(data, dict) or not data:
        raise ValidationError("questions", "must be a non-empty object mapping question id -> question")
    if len(data) > MAX_QUESTIONS:
        raise ValidationError("questions", f"at most {MAX_QUESTIONS} questions per request, got {len(data)}")
    return {qid: question_from_dict(q, f"questions.{qid}") for qid, q in data.items()}


def _r(x):
    return round(x, 4)


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict
    confidence: float
    type = "choice"

    def to_dict(self):
        return {
            "type": self.type,
            "choice": self.choice,
            "probabilities": {k: _r(v) for k, v in self.probabilities.items()},
            "confidence": _r(self.confidence),
        }


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict
    probabilities: dict
    confidence: float
    type = "score"

    def to_dict(self):
        return {
            "type": self.type,
            "score": _r(self.score),
            "legend": self.legend,
            "probabilities": {k: _r(v) for k, v in self.probabilities.items()},
            "confidence": _r(self.confidence),
        }


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type = "noul"

    def to_dict(self):
        return {"type": self.type, "noul": _r(self.noul)}


@dataclass(frozen=True)
class Diagnostics:
    """Per-question details outside the answer format that help debugging: how much of the next-token mass landed on valid labels, the
    uncalibrated distribution, and token counts."""

    label_mass: float
    raw_probabilities: list
    prompt_tokens: int
    cached_tokens: int
    latency_ms: float

    def to_dict(self):
        return {
            "label_mass": _r(self.label_mass),
            "raw_probabilities": [_r(p) for p in self.raw_probabilities],
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class SystemOneResponse:
    model: str
    answers: dict
    usage: dict
    diagnostics: dict = field(default_factory=dict)

    @property
    def choices(self):
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def scores(self):
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}

    @property
    def nouls(self):
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    def to_dict(self, include_diagnostics=False):
        d = {
            "model": self.model,
            "answers": {k: a.to_dict() for k, a in self.answers.items()},
            "usage": self.usage,
        }
        if include_diagnostics:
            d["diagnostics"] = {k: v.to_dict() for k, v in self.diagnostics.items()}
        return d
