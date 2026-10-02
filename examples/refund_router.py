"""Refund routing: model judgments plus a deterministic check in code.

A refund request goes through three judgments and one rule. The state holds
the customer's message, recent transactions, and the refund policy. Three
independent Nouls judge the language. The refund amount and the limit check
are plain Python, because arithmetic over records should never be a guess.

    NEX_BACKEND_MODEL=qwen3:4b-instruct python3 examples/refund_router.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nex import Nex, Noul

AUTO_APPROVE_LIMIT_USD = 100
# Act on a judgment only when it is clearly yes or clearly no.
YES = 0.8
NO = 0.2

POLICY = (
    "Duplicate charges are refunded in full. Subscription renewals can be refunded within 14 days "
    "if the plan was not used after renewal. Hardware is refundable within 30 days if unopened."
)

QUESTIONS = {
    "refund_requested": Noul(
        instructions="Does `customer_message` ask for money back (a refund, credit, or reversed charge)?"
    ),
    "duplicate_charge": Noul(
        instructions="Do `customer_message` and `transactions` indicate a duplicate charge?",
        criteria={
            "true": "The same purchase was captured more than once for the same amount",
            "false": "Each captured charge is a separate purchase, or only one charge exists",
        },
    ),
    "policy_supports_refund": Noul(
        instructions="Does `refund_policy` support the refund requested in `customer_message`?"
    ),
}

SCENARIOS = {
    "double-charged coffee plan": {
        "customer_message": "I was charged twice for this month's coffee plan. Please refund the extra one.",
        "transactions": [
            {"id": "tx_901", "date": "2026-09-30", "description": "Coffee plan, monthly", "amount_usd": 18.0, "status": "captured"},
            {"id": "tx_902", "date": "2026-09-30", "description": "Coffee plan, monthly", "amount_usd": 18.0, "status": "captured"},
        ],
    },
    "double-charged espresso machine": {
        "customer_message": "Your checkout charged me twice for the espresso machine. I want one of them back.",
        "transactions": [
            {"id": "tx_711", "date": "2026-09-27", "description": "Espresso machine", "amount_usd": 649.0, "status": "captured"},
            {"id": "tx_712", "date": "2026-09-27", "description": "Espresso machine", "amount_usd": 649.0, "status": "captured"},
        ],
    },
    "receipt for expenses": {
        "customer_message": "Could you email me a receipt for last week's grinder order? I need it for my expense report.",
        "transactions": [
            {"id": "tx_655", "date": "2026-09-24", "description": "Burr grinder", "amount_usd": 129.0, "status": "captured"},
        ],
    },
}


def refund_amount(transactions):
    """Money at stake, from the records: the newest captured charge."""
    captured = [t for t in transactions if t["status"] == "captured"]
    return max(captured, key=lambda t: t["date"])["amount_usd"] if captured else 0.0


def decide(nouls, amount):
    """Return (decision, reason). Only clear duplicates under the limit are
    approved without a person. The model never sees the limit."""
    requested = nouls["refund_requested"].noul
    duplicate = nouls["duplicate_charge"].noul
    supported = nouls["policy_supports_refund"].noul
    if requested <= NO:
        return "no action", "no refund asked for"
    if requested < YES:
        return "human review", f"unclear whether a refund is asked for ({requested:.2f})"
    if duplicate < YES or supported < YES:
        return "human review", f"not a clear duplicate under policy ({duplicate:.2f}, {supported:.2f})"
    if amount > AUTO_APPROVE_LIMIT_USD:
        return "human review", f"${amount:.2f} is over the ${AUTO_APPROVE_LIMIT_USD} auto-approve limit"
    return "approve", f"duplicate charge of ${amount:.2f}, within policy and limit"


def main():
    nex = Nex()
    print(f"model {nex.model_id}\n")
    for name, case in SCENARIOS.items():
        state = {**case, "refund_policy": POLICY}
        nouls = nex.system_one(state=state, questions=QUESTIONS).nouls
        amount = refund_amount(case["transactions"])
        decision, reason = decide(nouls, amount)
        judged = "  ".join(f"{qid}={a.noul:.2f}" for qid, a in nouls.items())
        print(f"{name}\n  {judged}\n  amount ${amount:.2f} -> {decision.upper()}: {reason}\n")


if __name__ == "__main__":
    main()
