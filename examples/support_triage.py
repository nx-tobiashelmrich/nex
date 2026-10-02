"""Support triage: one Nex call per ticket, routing and priority in code.

Each ticket gets four independent questions in a single system_one call.
The model only judges. Routing and priority are plain Python over the typed
answers, so a policy change is a code change, not a new prompt.

    NEX_BACKEND_MODEL=qwen3:4b-instruct python3 examples/support_triage.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nex import Choice, Nex, Noul, Score

# Policy. Tune these and the queue re-sorts without another model call.
AUTO_ROUTE_CONFIDENCE = 0.6
URGENCY_WEIGHT = 0.8
CHURN_WEIGHT = 0.2
# A retention team ranks the same answers with churn first.
RETENTION_WEIGHTS = (0.3, 0.7)

TICKETS = {
    "T-101": "I was charged twice for my September invoice, two payments of $49 on the 3rd. "
    "Please refund the duplicate.",
    "T-102": "Since 9am every card payment at our three stores fails with error E-502. We are turning "
    "customers away at the till. If this is not fixed today we are switching providers.",
    "T-103": "Hello? I emailed last week and never heard back. Can someone get in touch?",
    "T-104": "The two-factor code never arrives by SMS, so I cannot log in. I have tried five times "
    "this morning and I need to close out yesterday's sales.",
    "T-105": "Your new pricing nearly doubles what we pay for our three stores. We have been "
    "customers for six years and are now comparing other providers.",
    "T-106": "Third time writing about this. The card reader I ordered three weeks ago still says "
    "'label created'. Honestly I am about to cancel and ask for my money back.",
}

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this support ticket?",
        criteria={
            "billing": "Money owed or paid: charges, invoices, refunds, subscription plans, payment methods",
            "technical": "The software misbehaves: error messages, outages, crashes, reports showing wrong numbers",
            "account": "Getting into the account: login, passwords, two-factor codes, staff access",
            "shipping": "A hardware order (card readers, printers) has not arrived, is late, or arrived damaged",
            "other": "Sales questions, feedback, or anything that fits none of the teams above",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this ticket, judged by what is broken for the customer right now?",
        criteria=[
            "Can wait: a question, request, or complaint. Nothing is broken",
            "Soon: something is broken or late, but the customer can keep working",
            "Today: a person is blocked from a core task, such as logging in or closing out sales",
            "Now: the business cannot take payments or serve customers at this moment",
        ],
    ),
    "refund_requested": Noul(
        instructions="Does the customer ask for money back (a refund, credit, or reversed charge)?"
    ),
    "churn_risk": Noul(
        instructions="Is the customer signalling that they might leave?",
        criteria={
            "true": "They threaten to cancel, mention switching providers, or say they are running out of patience",
            "false": "They ask for help without any sign that they might leave",
        },
    ),
}


def priority(answers, urgency_weight=URGENCY_WEIGHT, churn_weight=CHURN_WEIGHT):
    """0-100. Urgency is normalized to 0-1 by its top level."""
    urgency = answers["urgency"].score / (len(QUESTIONS["urgency"].criteria) - 1)
    churn = answers["churn_risk"].noul
    return round(100 * (urgency_weight * urgency + churn_weight * churn))


def route(answers):
    """Auto-route only a confident pick of a real team. A spread-out
    distribution or the explicit "other" option goes to a person."""
    department = answers["department"]
    if department.confidence >= AUTO_ROUTE_CONFIDENCE and department.choice != "other":
        return department.choice
    return "human triage"


def main():
    nex = Nex()
    print(f"model {nex.model_id}\n")
    results = {tid: nex.system_one(state=text, questions=QUESTIONS).answers for tid, text in TICKETS.items()}

    header = f"{'ticket':7} {'department':17} {'route':13} {'urgency':>7} {'refund':>6} {'churn':>5} {'prio':>4}"
    print(header)
    print("-" * len(header))
    for tid, a in sorted(results.items(), key=lambda kv: -priority(kv[1])):
        dept = f"{a['department'].choice} ({a['department'].confidence:.2f})"
        print(
            f"{tid:7} {dept:17} {route(a):13} {a['urgency'].score:7.2f} "
            f"{a['refund_requested'].noul:6.2f} {a['churn_risk'].noul:5.2f} {priority(a):4d}"
        )

    # Same answers, different policy, no new model calls.
    print(f"\nre-ranked with retention weights (urgency, churn) = {RETENTION_WEIGHTS}:")
    order = sorted(results, key=lambda tid: -priority(results[tid], *RETENTION_WEIGHTS))
    print("  " + "  ".join(f"{tid}={priority(results[tid], *RETENTION_WEIGHTS)}" for tid in order))

    queue = [tid for tid, a in results.items() if route(a) == "human triage"]
    refunds = [tid for tid, a in results.items() if a["refund_requested"].noul >= 0.5]
    print(f"\nhuman triage queue: {', '.join(queue) or 'empty'}")
    print(f"flag for the refund workflow: {', '.join(refunds) or 'none'}")


if __name__ == "__main__":
    main()
