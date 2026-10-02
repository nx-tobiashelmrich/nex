"""Nex: a small System One-style decision model.

Send a state and typed questions (Choice, Score, Noul), get typed answers
with probabilities and confidence, from a local model through Ollama. The
contract is docs/SPEC.md.
"""

import sys

# The modules below use 3.10 syntax such as ``dict | None``, which fails on
# older versions with a TypeError that does not name the cause.
if sys.version_info < (3, 10):
    raise ImportError("Nex needs Python 3.10 or newer, this is %d.%d.%d" % tuple(sys.version_info[:3]))

__version__ = "0.1.0"

from .backend import BackendError, ContextOverflowError, FakeBackend, ModelNotFoundError, OllamaBackend
from .calibration import Calibration
from .engine import Nex
from .primitives import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    ValidationError,
)

__all__ = [
    "BackendError",
    "Calibration",
    "Choice",
    "ChoiceAnswer",
    "ContextOverflowError",
    "FakeBackend",
    "ModelNotFoundError",
    "Nex",
    "Noul",
    "NoulAnswer",
    "OllamaBackend",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "ValidationError",
]
