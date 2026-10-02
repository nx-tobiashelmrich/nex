"""Confidence statistics derived from answer distributions.

The formulas are specified in docs/SPEC.md, section 6. A confidence is 1
when all probability sits on one outcome and 0 when it is spread evenly.
"""


def noul_confidence(p):
    """Distance of a yes-probability from 0.5, on a 0-1 scale."""
    return abs(2 * p - 1)


def choice_confidence(probabilities):
    """How far the top option sits above an even 1/n split."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return (max(probabilities) - 1 / n) / (1 - 1 / n)


def score_confidence(probabilities):
    """Probability-weighted distance from the most likely level, relative to
    the same distance for an even spread. Neighbouring levels cost less than
    distant ones."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    m = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)
