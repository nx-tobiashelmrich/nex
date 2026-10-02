"""Compare Nex with TypeSafe's Jev on the labeled eval set.

Nex answers come from a file written by ``nex eval --out``, scored with the
temperatures in the current calibration file. Jev answers come from
TypeSafe's API, which needs ``TYPESAFE_API_KEY``. They are cached per case,
so a rerun only asks Jev about new or changed cases. Both systems are scored
with the same metrics as ``nex eval`` (see docs/SPEC.md, section 7).

    nex eval --model qwen3.5:9b --out nex-results.json
    TYPESAFE_API_KEY=... python3 evals/compare_jev.py nex-results.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nex.calibration import Calibration, apply_temperature  # noqa: E402
from nex.evaluation import load_cases, metrics, temperatures_for, truth_index  # noqa: E402

JEV_URL = "https://api.typesafe.ai/v1/systemone"
TYPES = ("choice", "score", "noul", "all")


def case_hash(case):
    """Changes whenever the state or question of a case changes."""
    text = json.dumps([case["state"], case["question"]], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ask_jev(case, key, model, retries=4):
    """One case, one request. Returns ``(response, latency_ms)``."""
    body = json.dumps({"state": case["state"], "model": model, "questions": {"q": case["question"]}}).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            JEV_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp), (time.perf_counter() - started) * 1000
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            e.close()
            # 429 and 529 are TypeSafe's rate-limit and overload replies.
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(float(e.headers.get("retry-after") or 2**attempt))
                continue
            raise RuntimeError(f"Jev returned HTTP {e.code} for {case['id']}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"cannot reach Jev: {getattr(e, 'reason', e)}") from None


def jev_distribution(question, answer):
    """A Jev answer as a probability list in Nex's slot order."""
    if question["type"] == "choice":
        return [answer["probabilities"].get(option, 0.0) for option in question["criteria"]]
    if question["type"] == "score":
        return [answer["probabilities"].get(str(i), 0.0) for i in range(len(question["criteria"]))]
    return [answer["noul"], 1 - answer["noul"]]


def fetch_jev(cases, cache_path, model, workers):
    """Jev answers for every case, from the cache where it is current."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c for c in cases if cache.get(c["id"], {}).get("hash") != case_hash(c) or cache[c["id"]]["model_asked"] != model]
    if todo:
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            sys.exit(f"{len(todo)} cases need Jev answers: set TYPESAFE_API_KEY")
        print(f"asking Jev about {len(todo)} cases", file=sys.stderr)
        with ThreadPoolExecutor(workers) as pool:
            for case, (response, ms) in zip(todo, pool.map(lambda c: ask_jev(c, key, model), todo)):
                cache[case["id"]] = {
                    "hash": case_hash(case), "model_asked": model, "model": response["model"],
                    "answer": response["answers"]["q"], "latency_ms": ms,
                    "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n")
    return cache


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def confident(rows, threshold=0.9):
    """Share of answers with top probability >= threshold, and their accuracy."""
    picked = [argmax(p) == truth for p, truth in rows if max(p) >= threshold]
    return len(picked) / len(rows), (sum(picked) / len(picked) if picked else None)


def report(systems):
    """systems: {name: (records, temperatures)}. Prints one table."""
    print(f"{'system':32s} {'type':7s} {'n':>4s} {'acc':>6s} {'NLL':>6s} {'ECE':>6s} {'Brier':>6s} "
          f"{'p>=0.9 share, acc':>18s} {'MAE':>6s}")
    summary = {}
    for name, (records, temps) in systems.items():
        m = metrics(records, temps)
        summary[name] = m
        for qtype in TYPES:
            if qtype not in m:
                continue
            group = [r for r in records if qtype == "all" or r["type"] == qtype]
            share, acc = confident([(apply_temperature(r["raw"], temps.get(r["type"], 1.0)), r["truth"]) for r in group])
            row = m[qtype]
            m[qtype]["confident_share"], m[qtype]["confident_accuracy"] = share, acc
            mae = f"{row['mae']:.3f}" if "mae" in row else "-"
            acc_text = f"{acc:.1%}" if acc is not None else "-"
            print(f"{name:32s} {qtype:7s} {row['n']:4d} {row['accuracy']:6.3f} {row['nll']:6.3f} {row['ece']:6.3f} "
                  f"{row['brier']:6.3f} {share:9.0%}, {acc_text:>6s} {mae:>6s}")
        print()
    return summary


def head_to_head(jev_records, name, records, temps):
    """Where the two disagree with the label, who is right."""
    jev = {r["id"]: r for r in jev_records}
    out = {"agree": {}, "jev_only_right": [], "nex_only_right": [], "both_wrong": []}
    for qtype in TYPES:
        group = [r for r in records if qtype == "all" or r["type"] == qtype]
        if group:
            same = sum(argmax(jev[r["id"]]["raw"]) == argmax(r["raw"]) for r in group)
            out["agree"][qtype] = same / len(group)
    for r in records:
        jev_right = argmax(jev[r["id"]]["raw"]) == r["truth"]
        nex_right = argmax(apply_temperature(r["raw"], temps.get(r["type"], 1.0))) == r["truth"]
        if jev_right and not nex_right:
            out["jev_only_right"].append(r["id"])
        elif nex_right and not jev_right:
            out["nex_only_right"].append(r["id"])
        elif not (jev_right or nex_right):
            out["both_wrong"].append(r["id"])
    agree = ", ".join(f"{q} {v:.0%}" for q, v in out["agree"].items())
    print(f"{name}: same top answer as Jev: {agree}")
    print(f"  Jev right, Nex wrong: {len(out['jev_only_right'])}  {' '.join(out['jev_only_right'])}")
    print(f"  Nex right, Jev wrong: {len(out['nex_only_right'])}  {' '.join(out['nex_only_right'])}")
    print(f"  both wrong:           {len(out['both_wrong'])}  {' '.join(out['both_wrong'])}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nex_results", help="file written by `nex eval --out`")
    parser.add_argument("--data", action="append", help="eval cases, file or directory (default: evals/data)")
    parser.add_argument("--jev-cache", default=str(ROOT / "evals" / "results" / "jev-answers.json"),
                        help="where Jev answers are kept between runs (default: evals/results/jev-answers.json)")
    parser.add_argument("--jev-model", default="jev-latest", help="model name sent to Jev (default: jev-latest)")
    parser.add_argument("--workers", type=int, default=8, help="parallel Jev requests (default: 8)")
    parser.add_argument("--out", help="write the comparison as JSON")
    args = parser.parse_args(argv)

    cases = load_cases(args.data or [str(ROOT / "evals" / "data")])
    nex_runs = json.loads(Path(args.nex_results).read_text())
    cache = fetch_jev(cases, Path(args.jev_cache), args.jev_model, args.workers)

    jev_records = []
    for case in cases:
        entry = cache[case["id"]]
        jev_records.append({
            "id": case["id"], "type": case["question"]["type"], "truth": truth_index(case["question"], case["label"]),
            "raw": jev_distribution(case["question"], entry["answer"]), "label_mass": 1.0,
            "latency_ms": entry["latency_ms"],
        })
    jev_name = "jev " + cache[cases[0]["id"]]["model"]
    calibration = Calibration.load()
    ids = {c["id"] for c in cases}
    systems = {jev_name: (jev_records, {})}
    for model, run in nex_runs.items():
        records = [r for r in run["records"] if r["id"] in ids]
        if len(records) != len(cases):
            sys.exit(f"{args.nex_results} has {len(records)} of {len(cases)} cases for {model}, rerun nex eval")
        systems[f"nex {model}"] = (records, temperatures_for(calibration, model))

    summary = report(systems)
    print("latency is per request for Jev (network included) and per question for Nex (local)\n")
    comparison = {}
    for name, (records, temps) in list(systems.items())[1:]:
        comparison[name] = head_to_head(jev_records, name, records, temps)
    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": summary, "head_to_head": comparison}, indent=1) + "\n")


if __name__ == "__main__":
    main()
