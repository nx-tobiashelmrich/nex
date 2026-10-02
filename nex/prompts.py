"""Turn (state, question) into a chat prompt whose next token is the answer.

Every option gets a single-token label (the letters A to U without I for
Choice, the numbers 1 to n for Score, Yes/No for Noul). I is skipped because
a reply that starts with the pronoun "I" would otherwise count as that
option. Nex never samples: it reads the backend's next-token log-probabilities at the answer
position and maps tokens back to labels.

Each prompt starts with the same system message and state, so the backend
can reuse its prompt cache for that prefix and only process the question
part when several questions are asked about one state.
"""

import json

from .primitives import Choice, Noul, Score

SYSTEM_PROMPT = (
    "You are a decision model. You read a STATE and answer one QUESTION about it.\n"
    "Reply with exactly one option label from the OPTIONS list and nothing else: "
    "no explanation, no punctuation."
)

# 20 letters, one per possible option. No I, see the module docstring.
CHOICE_LABELS = "ABCDEFGHJKLMNOPQRSTU"
YES_TOKENS = {"yes", "y", "true"}
NO_TOKENS = {"no", "n", "false"}


def render(value):
    """Strings pass through. Objects and arrays become indented JSON so the
    model can follow backticked paths like `ticket.messages[0].text`."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _option_line(label, name, description):
    if description is None:
        return f"{label}) {name}"
    text = render(description)
    if "\n" in text:
        return f"{label}) {name}:\n{text}"
    return f"{label}) {name}: {text}"


def _letter_list(labels):
    # Spell out every valid letter. A range like A-K would invite the model
    # to answer with the I it skips.
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def score_labels(n):
    """Prompt labels for an n-level Score, lowest level first.

    Levels are shown from 1 because models read that more reliably. On the
    bundled eval set, labels from 0 made qwen3.5:9b answer one level too high
    in most of its Score errors. A 10-level Score does not fit the single
    digits 1 to 9, so it keeps 0 to 9 and the prompt says so."""
    if n <= 9:
        return [str(i + 1) for i in range(n)]
    return [str(i) for i in range(n)]


def build_prompt(state, question):
    """Return ``(messages, labels)``. ``labels[i]`` is the label of the i-th
    answer slot: option i for Choice, level i for Score, [yes, no] for Noul."""
    if isinstance(question, Choice):
        labels = list(CHOICE_LABELS[: len(question.criteria)])
        options = [_option_line(l, name, desc) for l, (name, desc) in zip(labels, question.criteria.items())]
        reply = f"Reply with only the letter ({_letter_list(labels)}) of the best option."
    elif isinstance(question, Score):
        labels = score_labels(len(question.criteria))
        options = [f"{l}) {render(level)}" for l, level in zip(labels, question.criteria)]
        note = "Levels are numbered from 0. " if labels[0] == "0" else ""
        reply = f"{note}Reply with only the number ({labels[0]}-{labels[-1]}) of the level that fits best."
    elif isinstance(question, Noul):
        labels = ["Yes", "No"]
        criteria = question.criteria or {}
        yes = criteria.get("true")
        no = criteria.get("false")
        options = [
            "Yes" + (f": {render(yes)}" if yes is not None else ""),
            "No" + (f": {render(no)}" if no is not None else ""),
        ]
        reply = "Reply with only Yes or No."
    else:
        raise TypeError(f"unsupported question type {type(question).__name__}")

    user = (
        f"STATE:\n{render(state)}\n\n"
        f"QUESTION:\n{render(question.instructions)}\n\n"
        "OPTIONS:\n" + "\n".join(options) + "\n\n" + reply
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return messages, labels


def token_to_slot(token, question, n_slots):
    """Map one candidate next token to an answer slot index, or None.

    Tokenizers emit the same answer in several spellings ("A", " A", "a",
    "A)"), so the token is normalized before matching and the caller sums
    probability over every spelling of a label."""
    t = token.strip().rstrip(").:").strip()
    if not t:
        return None
    if isinstance(question, Noul):
        low = t.lower()
        if low in YES_TOKENS:
            return 0
        if low in NO_TOKENS:
            return 1
        return None
    if isinstance(question, Score):
        # Compare against the ASCII labels. isdigit() alone also accepts "²"
        # and "₂".
        labels = score_labels(n_slots)
        return labels.index(t) if t in labels else None
    # ASCII only. Some letters upper-case to two (the ligature U+FB06 becomes
    # "ST"), which would then match as a substring of the label string.
    if len(t) == 1 and t.isascii() and t.upper() in CHOICE_LABELS[:n_slots]:
        return CHOICE_LABELS.index(t.upper())
    return None
