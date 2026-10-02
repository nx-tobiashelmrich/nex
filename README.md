# Nex

A small decision model for software. Send a state and typed questions, get typed answers with probabilities and confidence back. No text generation, no parsing.

Nex runs on a local model through [Ollama](https://ollama.com). Its request and response format is specified in [docs/SPEC.md](docs/SPEC.md) and matches the format of TypeSafe's Jev, so clients written for Jev work against Nex.

```python
from nex import Nex, Choice, Noul, Score

nex = Nex()
response = nex.system_one(
    state={"document": "I was charged twice. Please fix this ASAP."},
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    },
)
response.nouls["billing"].noul       # 0.981
response.choices["tone"].choice      # "frustrated" (p 0.60, confidence 0.40)
response.scores["urgency"].score     # 1.89, on levels 0 = can wait ... 2 = today
```

## How it works

```
state + questions
   │  one prompt per question: system message, state, question, labeled options
   ▼
Ollama, one forward pass (num_predict = 1, no sampling)
   │  top 20 next-token logprobs at the answer position
   ▼
fold spellings ("A", " A", "a") into one probability per option
   │
   ▼
calibrate: p_i ** (1 / T), with T fitted per model and question type
   │
   ▼
typed answer: choice / score / noul, probabilities, confidence
```

- Every option gets a single-token label: letters for Choice (`A` to `U`, skipping `I` so a reply starting with the pronoun cannot count as an option), numbers from `1` for Score, `Yes`/`No` for Noul. The answer is read straight from the model's next-token distribution, so the probabilities are the model's own, not a number it was asked to write.
- Questions are evaluated independently and concurrently. None of them sees another's answer.
- Every prompt starts with the same system message and state, so Ollama reuses its prompt cache when several questions share a state. With `qwen3:4b-instruct`, each extra question about a 1,500-token state costs about 110 ms.
- `confidence` is 1 when all probability sits on one answer and 0 when it is spread evenly. The formulas are in [docs/SPEC.md](docs/SPEC.md#6-confidence).
- Calibration fits one temperature per model and question type on labeled examples, so that answers given 0.8 are right about 80% of the time. See [Calibration](#calibration).

## Requirements

- Python 3.10 or newer, no third-party packages. On macOS, `/usr/bin/python3` is 3.9, so use Python from python.org or Homebrew.
- Ollama, running, with the default model `qwen3.5:9b` pulled, or another model chosen with `NEX_BACKEND_MODEL`. Nex reads token logprobs, so it needs an Ollama version with logprobs support (tested with 0.35).

```sh
ollama pull qwen3.5:9b          # default, 6.6 GB
ollama pull qwen3:4b-instruct   # faster, 2.5 GB
```

## Quickstart

```sh
git clone https://github.com/nx-tobiashelmrich/nex.git && cd nex
python3 -m nex ask \
  --state "I was charged twice. Please fix this ASAP." \
  --noul "Is this about billing?"
```

This uses `qwen3.5:9b`. If you pulled only the 4B, add `--backend-model qwen3:4b-instruct` or set `NEX_BACKEND_MODEL=qwen3:4b-instruct`.

To get a `nex` command, install it into a virtual environment: `python3 -m venv .venv && . .venv/bin/activate && pip install -e .`. Everything below works with `nex` or `python3 -m nex`.

The quick forms are `--noul TEXT`, `--choice TEXT --options a,b,c`, and `--score TEXT --levels low,mid,high`. For full questions, pass a questions map with `--questions '{...}'` or `--questions @questions.json`. `--state @file.json` reads the state from a file, and `--debug` adds per-question diagnostics.

### HTTP server

```sh
nex serve                       # http://127.0.0.1:8787
```

```sh
curl -s http://127.0.0.1:8787/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "nex-latest",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      }
    }
  }'
```

| Route | Purpose |
| - | - |
| `POST /v1/systemone` | Evaluate questions. Add `?debug=1` for diagnostics. |
| `GET /v1/models` | `nex-latest` plus every model the Ollama server has pulled |
| `GET /health` | Liveness and the active model id |

Errors come back as `{"error": {"type", "message", "field"}}`. The most common are 422 `validation_error` (a malformed request), 422 `context_overflow` (the state does not fit the context window, it is never cut off silently), 422 `model_not_found` (Ollama has not pulled the model), and 502 `backend_error` (Ollama failed). The full list is in [docs/SPEC.md](docs/SPEC.md#9-http-api).

`model` accepts `nex-latest`, `jev-latest`, or `jev-preview` for the default model, a Nex model id such as `nex-0.1.0+qwen3:4b-instruct`, or any Ollama model name.

Security defaults:

- The server binds `127.0.0.1`. On a loopback address it refuses requests whose `Host` or `Origin` names another site, which blocks DNS rebinding and cross-site requests from web pages. curl and SDK clients are not affected.
- Set `NEX_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`. Without it, any `Authorization` header is accepted, so clients written for Jev work unchanged. `nex serve` warns when it binds a non-loopback address without a key.

## Questions and answers

| Type | Asks | `criteria` | Answer fields |
| - | - | - | - |
| `choice` | Which option fits? | map of option → description or `null`, 2 to 20 options | `choice`, `probabilities`, `confidence` |
| `score` | Which level fits? | ordered list of level descriptions, 2 to 10 levels | `score` (probability-weighted level), `legend`, `probabilities`, `confidence` |
| `noul` | Is this true? | optional `{"true": ..., "false": ...}` | `noul`, the probability of yes |

`instructions` and criteria descriptions can be strings, objects, or arrays. Objects and arrays are rendered as JSON, so a question can point at nested state with backticked paths like `` `order.items[0].sku` ``. A request takes at most 256 questions.

`usage.output_tokens` is always one per question, the decision token. With `--debug` or `?debug=1`, each answer also reports:

- `label_mass`: how much of the next-token probability landed on valid labels before renormalizing. Well below 1 means the model wanted to say something else, so treat that answer with suspicion.
- `raw_probabilities`: the distribution before calibration.
- `prompt_tokens`, `cached_tokens`, `latency_ms`.

Writing good questions:

- Ask one narrow judgment per question, and split independent factors into separate questions that code combines.
- Give options and levels concrete descriptions that stand on their own.
- Include a no-match option such as `"other"` when nothing may fit.
- Keep rules, lookups, and arithmetic in code, and put their results in the state.

## Examples

Each script runs against your local Ollama and honors `NEX_BACKEND_MODEL`.

- [`examples/support_triage.py`](examples/support_triage.py): four questions per ticket in one call. Code routes on confidence and builds a priority from weighted judgments, then re-ranks with different weights without calling the model again.
- [`examples/refund_router.py`](examples/refund_router.py): three Nouls plus a deterministic amount check in Python decide approve, human review, or no action.
- [`examples/rerank.py`](examples/rerank.py): one Score per candidate passage in a single call, sorted by relevance.

## Choosing a model

Measured with `nex eval` on the 170 labeled cases in [`evals/data`](evals/data) (60 Choice, 50 Score, 60 Noul) on an Apple Silicon Mac with 24 GB through Ollama's Metal backend. Calibrated numbers are held out: each half of the data is scored with temperatures fitted on the other half.

| | `qwen3.5:9b` (default) | `qwen3:4b-instruct` |
| - | - | - |
| Accuracy, all | **88.2%** | 79.4% |
| Choice / Score / Noul | 91.7% / 80.0% / 91.7% | 88.3% / 62.0% / 85.0% |
| Calibration error (ECE), raw → calibrated | 0.074 → 0.037 | 0.203 → 0.078 |
| Log loss, raw → calibrated | 0.313 → 0.325 | 2.319 → 0.476 |
| Latency per question, new state (p50) | 841 ms | **180 ms** |
| Each extra question on a 1,500-token state | about 1.3 s (partial cache reuse) | **about 110 ms** |
| Download | 6.6 GB | 2.5 GB |

How often each model is right when it is confident, after calibration:

| Top probability | `qwen3.5:9b` coverage, accuracy | `qwen3:4b-instruct` coverage, accuracy |
| - | - | - |
| ≥ 0.7 | 82%, 94.2% | 79%, 86.7% |
| ≥ 0.9 | 71%, 96.7% | 49%, 97.6% |

`qwen3.5:9b` is the default because it is more accurate and already close to calibrated. `qwen3:4b-instruct` is three to five times faster and makes full use of the prompt cache. Its raw probabilities sit at 0 or 1, so it relies on the shipped calibration, and it is weak on Score questions. Switch with `NEX_BACKEND_MODEL=qwen3:4b-instruct`.

Use the 9B for Score questions. The 4B gives a near-certain Score answer whether it is right or off by three levels, so its calibrated Score probabilities are spread over every level and its `score` values sit toward the middle of the scale. When code needs one level from the 4B, take the most likely entry of `probabilities` instead of rounding `score`.

Most errors on both models are arithmetic and multi-step logic: time zones, sums against a limit, date windows in a policy. TypeSafe's Jev answers most of these correctly, so this is a limit of these small open models rather than of one-token decisions. With Nex, compute such values in code and put the result in the state.

## Compared with Jev

The same 170 cases sent to TypeSafe's hosted Jev (`jev-1.13.0`) on 2026-10-02 with [`evals/compare_jev.py`](evals/compare_jev.py). Nex calibration numbers are held out, as above.

| | Jev | Nex `qwen3.5:9b` | Nex `qwen3:4b-instruct` |
| - | - | - | - |
| Accuracy, all | **97.1%** | 88.2% | 79.4% |
| Choice / Score / Noul | 95.0% / 98.0% / 98.3% | 91.7% / 80.0% / 91.7% | 88.3% / 62.0% / 85.0% |
| Calibration error (ECE) | 0.023 | 0.037 | 0.078 |
| Top probability ≥ 0.9: coverage, accuracy | 87%, 100% | 71%, 96.7% | 49%, 97.6% |
| Same top answer as Jev | | 91% | 81% |
| Latency (p50) | 345 ms per request, hosted, network included | 841 ms per question, local | 180 ms per question, local |

Head to head with the 9B, Jev is right and Nex wrong on 15 cases, Nex is right and Jev wrong on none, and both are wrong on the same 5: two time-zone questions, an expense-policy exception, a booking-eligibility rule, and an NDA clause. Jev's lead is largest on Score questions and on cases that need arithmetic. Nex runs offline on your machine and costs nothing per call, Jev is the more accurate decision model.

To rerun it, put your TypeSafe key in `TYPESAFE_API_KEY`:

```sh
nex eval --model qwen3.5:9b --out nex-results.json
python3 evals/compare_jev.py nex-results.json
```

Jev's answers are cached in `evals/results/jev-answers.json`, so a rerun only asks Jev about new or changed cases. The 170 cases were written for this project, so treat the numbers as a comparison on one small set, not as a general benchmark.

## Calibration

[`nex/calibration.json`](nex/calibration.json) holds one temperature per model and question type, fitted by minimizing log loss. `T > 1` softens an overconfident model and `T < 1` sharpens an underconfident one. The ranking of options never changes. Models without an entry run uncalibrated (`T = 1`).

The shipped temperatures cover `qwen3.5:9b` and `qwen3:4b-instruct` on the bundled eval set. For another model, a different quantization, or your own domain, measure and refit on cases like yours. Run these from the repository root, or point `--data` at your own cases:

```sh
nex eval --model qwen3:4b-instruct --out results.json   # accuracy, log loss, ECE, Brier, latency
nex calibrate --model qwen3:4b-instruct --from results.json --path my-calibration.json
export NEX_CALIBRATION=my-calibration.json
```

Without `--path`, `calibrate` updates the packaged `nex/calibration.json`. It skips a fit that runs into the temperature bounds and warns when a type has fewer than 30 cases or when a Score fit pulls `score` toward the middle.

Cases are JSON lines in `evals/data/*.jsonl`:

```json
{"id": "noul-001", "domain": "customer_support", "difficulty": "easy", "state": "...", "question": {"type": "noul", "instructions": "..."}, "label": true}
```

`label` is the option key for Choice, the level index for Score, and a boolean for Noul. The format and every metric are defined in [docs/SPEC.md](docs/SPEC.md#7-calibration).

## Configuration

| Variable | Default | Purpose |
| - | - | - |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama server, read the same way the Ollama CLI reads it. Loopback requests never go through `HTTP_PROXY`. |
| `NEX_BACKEND_MODEL` | `qwen3.5:9b` | Model behind `nex-latest` |
| `NEX_NUM_CTX` | `8192` | Context window sent to Ollama. Keep it fixed, because changing it reloads the model. |
| `NEX_CALIBRATION` | `nex/calibration.json` | Calibration file |
| `NEX_HOST`, `NEX_PORT` | `127.0.0.1`, `8787` | Bind address for `nex serve` |
| `NEX_API_KEY` | unset | Require a bearer token on `/v1/*` |

`--ollama-host` and `--backend-model` override the first two on any command.

## Limitations

- Ollama only. Nex needs next-token logprobs, which Ollama provides locally. Apart from the Ollama server, Nex makes no network requests.
- At most 20 Choice options, because Ollama returns at most 20 candidate tokens. At most 10 Score levels.
- Text only.
- A state that does not fit `NEX_NUM_CTX` is rejected with `context_overflow`, never truncated.
- The eval set is small and was written for this project. Treat the numbers above as a comparison between models, and choose thresholds on your own data before acting on them automatically.

## Development

```sh
python3 -m unittest discover -s tests -v
```

The tests use a scripted fake backend and local stub servers, so they do not need Ollama. [docs/SPEC.md](docs/SPEC.md) is the contract the tests pin.

## Credits and license

The idea, the primitive names Choice, Score, and Noul, and the confidence formulas come from [Jev by TypeSafe](https://docs.typesafe.ai). Nex is an independent reimplementation on open-weight models and is not affiliated with TypeSafe.

MIT licensed, see [LICENSE](LICENSE).
