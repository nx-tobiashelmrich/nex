"""Re-ranking: one Score per candidate passage, sorted in code.

A fast first stage (keyword search, embeddings) hands over a shortlist.
Nex scores every candidate against the query in one system_one call, one
question per candidate, and code sorts by score. Confidence says which
placements to trust.

    NEX_BACKEND_MODEL=qwen3:4b-instruct python3 examples/rerank.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nex import Nex, Score

QUERY = "How do I rotate an API key without downtime?"

# A shortlist in first-stage order. Several share the query's words but not its intent.
PASSAGES = [
    "API keys never expire on their own. We recommend rotating every key at least every 90 days.",
    "Our pricing has three tiers. Every tier includes API access and email support.",
    "Revoking a key takes effect immediately. Any request still using it fails with 401 Unauthorized.",
    "To rotate without downtime, create a second key, deploy it to all services, confirm traffic "
    "has moved in the request log, then revoke the old key. Both keys work until you revoke one.",
    "Webhook signing secrets are rotated from the Webhooks page. The old secret stays valid for 24 hours.",
    "Each project can hold up to five active API keys at once.",
    "To reset your dashboard password, click Forgot password on the sign-in page.",
    "Rate limits apply per key: 100 requests per second, with bursts up to 200.",
]

# Generic levels, so the same question works for any query.
LEVELS = [
    "Unrelated: shares no subject with the query",
    "Same subject: mentions what the query is about, but would not help answer it",
    "Helpful: a fact or step someone answering the query would use",
    "Answers it: the passage alone answers the query",
]


def main():
    nex = Nex()
    print(f"model {nex.model_id}\nquery: {QUERY}\n")
    state = {"query": QUERY, "passages": PASSAGES}
    # Speculative fan-out: every candidate is its own question, scored
    # independently, so none of them can bias another. All questions share
    # one state, so each prompt starts with the same prefix and the backend
    # reuses its prompt cache for it, processing only the short question tail.
    questions = {
        f"rel_{i}": Score(instructions=f"How well does `passages[{i}]` answer `query`?", criteria=LEVELS)
        for i in range(len(PASSAGES))
    }
    response = nex.system_one(state=state, questions=questions)

    ranked = sorted(response.scores.items(), key=lambda kv: -kv[1].score)
    print(f"{'rank':>4} {'was':>3} {'score':>5} {'conf':>4}  passage")
    for rank, (qid, answer) in enumerate(ranked, 1):
        i = int(qid.split("_")[1])
        text = PASSAGES[i] if len(PASSAGES[i]) <= 70 else PASSAGES[i][:67] + "..."
        print(f"{rank:4d} {i + 1:3d} {answer.score:5.2f} {answer.confidence:4.2f}  {text}")

    prompt = sum(d.prompt_tokens for d in response.diagnostics.values())
    cached = sum(d.cached_tokens for d in response.diagnostics.values())
    print(f"\n{len(questions)} questions, {prompt} prompt tokens, {cached} served from the prompt cache")


if __name__ == "__main__":
    main()
