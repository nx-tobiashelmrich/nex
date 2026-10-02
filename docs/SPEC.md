# Nex specification

This file is the contract for Nex: what it accepts, what it returns, and how every number in a response is computed. The code in `nex/` implements it, and the tests in `tests/` pin it. When the code and this file disagree, one of them has a bug.

The request and response format started as a copy of the public API of TypeSafe's Jev (version 1.13, October 2026), so clients written for Jev can point at Nex. Nex does not follow later changes to Jev. This file is the reference.

## 1. Request

A request has a state and one or more questions about it.

```json
{
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
}
```

| Field | Required | Meaning |
| - | - | - |
| `state` | yes | The content to judge. A string, an object, or an array. |
| `model` | no | Which model answers. See [Model names](#8-model-names). Default `nex-latest`. |
| `questions` | yes | Map of question id to question. Ids are chosen by the caller, are returned unchanged, and are never shown to the model. |

Every question sees the same state and is answered on its own. No question sees another question's answer.

## 2. Questions

Every question has a `type` and `instructions`. `instructions` is a non-empty string, or a non-empty object or array (for example `{"question": "...", "reference": {...}}`). Objects and arrays are shown to the model as JSON, so instructions can point at parts of the state with backticked paths such as `` `order.items[0].sku` ``.

### Choice: which option fits?

```json
{"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": null, "frustrated": null, "angry": "Insults, threats, or all caps"}}
```

`criteria` maps each option name to a description, or to `null` when the name says enough. 2 to 20 options. Option names are non-empty strings. Descriptions are `null` or non-empty text, objects, or arrays.

### Score: which level fits?

```json
{"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Can wait a week", "Needs an answer this week", "Needs an answer today"]}
```

`criteria` is an ordered list of level descriptions, lowest first. 2 to 10 levels. Level `i` is the `i`-th entry, starting at 0. Each level should describe a concrete situation that stands on its own.

### Noul: is this true?

```json
{"type": "noul", "instructions": "Does the customer ask for a refund?", "criteria": {"true": "Asks for money back", "false": "No refund request"}}
```

`criteria` is optional. When given, it may only have the keys `"true"` and `"false"`, each `null` or non-empty text.

## 3. Validation

A request that breaks any rule below is rejected before the model runs. Errors carry a `field` path such as `questions.department.criteria` or `questions.levels.criteria[2]`.

- `state` is present and is a string, object, or array.
- `questions` is a non-empty object with at most 256 questions.
- Each question is an object with a known `type`, the required fields, and no other fields.
- Text values are non-empty: an empty string, `{}`, or `[]` is rejected for instructions, descriptions, and levels.
- State, instructions, and criteria values nest at most 32 levels deep.
- Choice: 2 to 20 options. Score: 2 to 10 levels. Noul criteria: only `true` and `false`.
- The rendered state is at most `NEX_NUM_CTX × 64` characters, and the full prompt fits the context window. Otherwise the request fails with a context overflow (see [HTTP errors](#9-http-api)).

## 4. Response

```json
{
  "model": "nex-0.1.0+qwen3.5:9b",
  "answers": {
    "billing": {"type": "noul", "noul": 0.981},
    "tone": {
      "type": "choice",
      "choice": "frustrated",
      "probabilities": {"calm": 0.0536, "frustrated": 0.5977, "angry": 0.3486},
      "confidence": 0.3966
    },
    "urgency": {
      "type": "score",
      "score": 1.8916,
      "legend": {"0": "can wait", "1": "this week", "2": "today"},
      "probabilities": {"0": 0.0406, "1": 0.0271, "2": 0.9322},
      "confidence": 0.8374
    }
  },
  "usage": {"input_tokens": 346, "output_tokens": 3}
}
```

| Answer | Fields |
| - | - |
| Choice | `choice`: the most likely option (the first one on a tie). `probabilities`: every option, summing to 1. `confidence`. |
| Score | `score`: the probability-weighted level, `Σ i · p_i`, which can fall between levels. `legend`: level index (as a string) to its description. Object and array levels appear as JSON text. `probabilities`: level index (as a string) to probability. `confidence`. |
| Noul | `noul`: the probability that the statement is true. No `confidence` field (see [section 6](#6-confidence)). |

`model` is `nex-<version>+<backend model>`. `usage.input_tokens` is the sum of prompt tokens over all questions. `usage.output_tokens` is the number of questions, one decision token each. Numbers in JSON responses are rounded to 4 decimals.

### Diagnostics

With `?debug=1` on the HTTP API, `--debug` on the CLI, or `response.diagnostics` in Python, each answer also reports:

| Field | Meaning |
| - | - |
| `label_mass` | Share of the next-token probability that landed on valid labels, before renormalizing. Well below 1 means the model wanted to reply with something else. |
| `raw_probabilities` | The per-slot distribution before calibration. |
| `prompt_tokens`, `cached_tokens` | Tokens in the prompt, and how many the backend served from its prompt cache. |
| `latency_ms` | Time spent waiting for the backend. |

## 5. How an answer is computed

For each question, Nex:

1. **Builds a prompt.** A fixed system message tells the model to reply with exactly one option label. The user message is:

   ```
   STATE:
   <state>

   QUESTION:
   <instructions>

   OPTIONS:
   <one line per option>

   <reply line>
   ```

   Strings are inserted as they are. Objects and arrays are inserted as JSON with 2-space indentation and non-ASCII characters kept. The system message and the state come first, so every question about one state shares a prompt prefix and the backend can reuse its cache.

2. **Labels the options with single tokens.**
   - Choice: `A B C D E F G H J K L M N O P Q R S T U`. There is no `I`, because a reply that starts with the pronoun "I" would otherwise count as an option. Option lines read `A) name: description`, or `A) name` for a `null` description. The reply line lists the valid letters.
   - Score: `1` to `n`, one per level, lines `1) description`. On the bundled eval set, labels from 0 made qwen3.5:9b answer one level too high in 9 of its 14 Score errors, and labels from 1 removed most of those. A 10-level Score does not fit the digits 1 to 9, so it uses `0` to `9` and the reply line starts with "Levels are numbered from 0.". Either way, level `i` in the response (`legend`, `probabilities`) is the `i`-th level counted from 0.
   - Noul: `Yes` and `No`, followed by the true and false criteria when given.

3. **Reads one next-token distribution.** Nex asks the backend for a single forward pass and the 20 most likely first tokens of the reply with their log-probabilities. Nothing is sampled or generated. The Ollama request is `POST /api/chat` with `stream: false`, `logprobs: true`, `top_logprobs: 20`, `think: false`, `truncate: false`, `keep_alive: "30m"`, and options `num_predict: 1`, `temperature: 0`, `num_ctx: NEX_NUM_CTX`. `think` is dropped for models that refuse it.

4. **Folds spellings into labels.** Each candidate token is trimmed of whitespace and of trailing `)`, `.`, and `:`. Then:
   - Choice: a single ASCII letter, either case, that is one of the labels in use.
   - Score: a single ASCII digit that is one of the labels in use (`1` to `n`, or `0` to `9` for a 10-level Score). It maps to the level it labels, as in step 2.
   - Noul: `yes`, `y`, or `true` count as yes, and `no`, `n`, or `false` count as no, in any case.

   The probabilities of all spellings of a label are added up. Other tokens are ignored.

5. **Fills in missing labels and renormalizes.** A label absent from the 20 candidates gets the smallest listed probability, an upper bound on its true value, so it is never treated as impossible. The label probabilities are then scaled to sum to 1. If no candidate matches any label, every label gets the same probability.

6. **Calibrates.** With the temperature `T` stored for this backend model and question type (`T = 1` when none is stored):

   ```
   p_i ∝ p_i ^ (1 / T)          computed in log space, with p clamped to at least 1e-12
   ```

   `T > 1` softens an overconfident model and `T < 1` sharpens an underconfident one. The order of the options never changes.

7. **Builds the typed answer** from the calibrated distribution, as described in [section 4](#4-response).

## 6. Confidence

`confidence` summarizes how concentrated a distribution is, on a scale from 0 (spread evenly) to 1 (all probability on one answer). It is computed from the calibrated `probabilities` of the same answer.

**Choice**, with `n` options and `p_max` the probability of the chosen option:

```
confidence = (p_max - 1/n) / (1 - 1/n)
```

Only the top probability counts: `(0.6, 0.3, 0.1)` and `(0.6, 0.2, 0.2)` both give `0.4`.

**Score**, with `n` levels `0 … n-1`, `p_i` the probability of level `i`, and `m` the most likely level:

```
spread     = Σ p_i · |i - m|
even       = (1/n) · Σ |i - (n - 1)/2|
confidence = max(0, 1 - spread / even)
```

Probability on a neighbouring level costs less than the same probability further away. With 3 levels, `(0, 0.5, 0.5)` gives `0.25` and `(0.5, 0, 0.5)` gives `0`. `(0, 0.57, 0.43)` gives `1 - 0.43 / (2/3) ≈ 0.355`.

**Noul** answers have no `confidence` field, because `noul` already is the probability. A value near 0.5 means the model cannot tell. To gate Nouls and Choices with the same code, use `|2p - 1|`, which is the Choice formula for two options.

Confidence measures how sure the model is about one answer. It says nothing about whether a whole workflow is right, and how high it must be before code acts depends on what a wrong action costs.

## 7. Calibration

Nex cannot retrain its backbone. Instead it fits one temperature per backend model and question type on labeled cases, choosing the `T` that minimizes the mean negative log-likelihood of the true labels. That makes probabilities match observed accuracy across many predictions, for example answers given 0.8 are right about 80% of the time. It does not make any single answer right.

### File

`nex/calibration.json`, or the file named by `NEX_CALIBRATION`:

```json
{
  "models": {
    "qwen3.5:9b": {
      "choice": {"temperature": 1.0471, "n": 60},
      "noul": {"temperature": 1.0153, "n": 60},
      "score": {"temperature": 1.4046, "n": 50}
    }
  }
}
```

A model or type without an entry uses `T = 1`. Calibration is keyed by the exact Ollama model name, so a different tag or quantization needs its own fit.

### Fitting

`nex calibrate` runs every labeled case through the model once, keeps the raw distributions, and for each question type:

1. Finds the `T` in `[0.05, 20]` that minimizes mean NLL, by golden-section search on `log T`.
2. Does not save a `T` within 5% of either bound and keeps the previous value. A fit at the bound means nearly every case was right (lower bound) or wrong (upper bound), so the data cannot fix a temperature.
3. Warns when a type has fewer than 30 cases.
4. For Score, warns when the fitted `T` raises the mean absolute error of `score` by more than 0.05 levels. That happens when a model's raw Score answers are near one-hot whether right or wrong: the honest temperature then spreads probability onto every level and pulls `score` toward the middle of the scale. Nex keeps the honest probabilities, and the warning says so.

### Evaluation metrics

`nex eval` reports, per question type and overall:

| Metric | Definition |
| - | - |
| accuracy | Share of cases where the most likely slot is the true one (first slot on a tie). |
| NLL | Mean of `-ln p_true`, with `p` clamped to at least 1e-12. |
| Brier | Mean over cases of `Σ_i (p_i - [i is true])²`. |
| ECE | Cases are put in 10 equal-width bins by top probability (`[0, 0.1)` … `[0.9, 1.0]`). ECE is the case-weighted mean over bins of `|accuracy - mean top probability|`. |
| MAE (Score) | Mean of `|score - true level|`. |
| label mass | Mean `label_mass`. |
| latency | p50 and p95 of per-question backend time. |

Raw numbers use `T = 1`. Calibrated numbers are cross-fitted: case ids are split into folds by `crc32(id) mod k` (default `k = 2`), and each fold is scored with temperatures fitted on the other folds only. The `current` line uses the temperatures in the calibration file, which is in-sample if they were fitted on the same cases.

## 8. Model names

| `model` value | Answered by |
| - | - |
| missing, `nex-latest`, `jev-latest`, `jev-preview` | The default backend model (`NEX_BACKEND_MODEL`, default `qwen3.5:9b`) |
| `nex-<version>+<name>` | Ollama model `<name>` |
| anything else | Ollama model of that name |

A model Ollama has not pulled fails with `model_not_found`. Nex keeps a backend object for at most 16 non-default model names, and only for names that have answered at least once.

## 9. HTTP API

`nex serve` binds `NEX_HOST:NEX_PORT` (default `127.0.0.1:8787`).

| Route | Purpose |
| - | - |
| `POST /v1/systemone` | Evaluate a request ([section 1](#1-request)). `?debug=1` adds diagnostics. |
| `GET /v1/models` | `nex-latest` and every model the Ollama server has pulled |
| `GET /health` | `{"status": "ok", "model": "<model id>"}` |

Errors are JSON, `{"error": {"type": "...", "message": "...", "field": "..."}}`, where `field` appears when one part of the request is to blame.

| Status | `type` | When |
| - | - | - |
| 400 | `invalid_request` | Body is not JSON or not an object, bad or conflicting `Content-Length` |
| 401 | `unauthorized` | `NEX_API_KEY` is set and the bearer token is missing or wrong |
| 403 | `forbidden` | Bound to loopback and the `Host` or `Origin` header names another site |
| 404 | `not_found` | Unknown path |
| 405 | `method_not_allowed` | Wrong method for the path |
| 411 | `invalid_request` | Chunked body without `Content-Length` |
| 413 | `payload_too_large` | Body over 4 MB |
| 422 | `validation_error` | A rule in [section 3](#3-validation) is broken, an unknown top-level field, or `model` is not a string of at most 256 characters |
| 422 | `context_overflow` | The prompt does not fit the context window (`field: "state"`) |
| 422 | `model_not_found` | Ollama does not have the model (`field: "model"`) |
| 500 | `internal_error` | Unexpected error, details only in the server log |
| 502 | `backend_error` | Ollama failed or could not be reached |

**Authentication.** When `NEX_API_KEY` is set, `/v1/*` requires `Authorization: Bearer <key>`, compared in constant time. `/health` stays open. When it is unset, any `Authorization` header is accepted, so Jev clients work without changes.

**Local-only protection.** When bound to a loopback address, the server refuses requests whose `Host` is not `localhost`, a `*.localhost` name, a loopback IP, or the bound host, and requests whose `Origin` is not on localhost or a loopback IP. That blocks DNS rebinding and cross-site browser requests. Requests without `Origin`, such as from curl or an SDK, are not affected. On a non-loopback address these checks are off, and `nex serve` warns when no `NEX_API_KEY` is set.

## 10. Eval cases

Labeled cases are JSON lines, one case per line, in `evals/data/*.jsonl`:

```json
{"id": "noul-001", "domain": "customer_support", "difficulty": "easy", "state": "...", "question": {"type": "noul", "instructions": "..."}, "label": true}
```

| Field | Meaning |
| - | - |
| `id` | Unique across all files |
| `domain` | Free-form topic, used for slicing results |
| `difficulty` | `easy`, `medium`, or `hard` |
| `state`, `question` | As in a request, with a single question |
| `label` | Choice: the option name. Score: the level index. Noul: `true` or `false`. |

## 11. Configuration

| Variable | Default | Purpose |
| - | - | - |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama server, read the way the Ollama CLI reads it: without a scheme the port defaults to 11434, `0.0.0.0` and `::` mean this machine, quotes, credentials, query, and fragment are ignored. Requests to a loopback host never go through `HTTP_PROXY` or `HTTPS_PROXY`. |
| `NEX_BACKEND_MODEL` | `qwen3.5:9b` | Model behind `nex-latest` |
| `NEX_NUM_CTX` | `8192` | Context window requested from Ollama. Changing it makes Ollama reload the model. |
| `NEX_CALIBRATION` | `nex/calibration.json` | Calibration file. A set but missing file runs uncalibrated with a warning. |
| `NEX_HOST`, `NEX_PORT` | `127.0.0.1`, `8787` | Bind address for `nex serve`. Empty counts as unset. |
| `NEX_API_KEY` | unset | Bearer token required on `/v1/*` |

Nex needs Python 3.10 or newer and makes no network requests except to the Ollama server.
