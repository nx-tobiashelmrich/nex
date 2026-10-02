"""Compare Nex with TypeSafe's Jev on the labeled eval set.

Nex answers come from a file written by ``nex eval --out``. By default they
are scored held out, the way ``nex eval`` reports calibrated numbers: case ids
are split into folds and each fold is scored with temperatures fit on the
other folds. ``--nex-calibration file`` uses the current calibration file
instead, which is in-sample if it was fitted on these cases.

Jev answers come from TypeSafe's API, which needs ``TYPESAFE_API_KEY``. They
are cached per requested model name and case, so a rerun only asks Jev about
new or changed cases. When the cached answers come from several Jev versions,
the ones not on the newest version are asked again, so every Jev score comes
from one version. Both systems are scored with the same metrics as
``nex eval`` (see docs/SPEC.md, section 7).

    nex eval --model qwen3.5:9b --out nex-results.json
    TYPESAFE_API_KEY=... python3 evals/compare_jev.py nex-results.json
"""

import argparse
import email.utils
import hashlib
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nex.calibration import QUESTION_TYPES, Calibration, apply_temperature  # noqa: E402
from nex.evaluation import fit_type, fold_of, load_cases, metrics, temperatures_for, truth_index  # noqa: E402

JEV_URL = "https://api.typesafe.ai/v1/systemone"
TYPES = ("choice", "score", "noul", "all")
# 429 and 529 are TypeSafe's rate-limit and overload replies.
RETRY_STATUS = (429, 500, 502, 503, 529)
MAX_RETRY_DELAY = 120.0
# Fetch rounds before giving up on answers that keep coming from mixed versions.
VERSION_ROUNDS = 3


def case_hash(case):
    """Changes whenever the state or question of a case changes."""
    text = json.dumps([case["state"], case["question"]], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def retry_delay(retry_after, attempt):
    """Seconds to wait before the next try. ``retry_after`` is the Retry-After
    header, in seconds or as an HTTP date. Without a usable one, back off
    exponentially. Never more than ``MAX_RETRY_DELAY``."""
    delay = None
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, OverflowError):
                when = None
            if when is not None:
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delay = (when - datetime.now(timezone.utc)).total_seconds()
    if delay is None or math.isnan(delay):
        delay = 2.0**attempt
    return min(max(delay, 0.0), MAX_RETRY_DELAY)


def ask_jev(case, key, model, retries=4):
    """One case, one request. Returns ``(response, latency_ms)``. Rate limits,
    server errors, and dropped connections are retried."""
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
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except (OSError, http.client.HTTPException):
                detail = ""
            e.close()
            if e.code in RETRY_STATUS and attempt < retries:
                time.sleep(retry_delay(e.headers.get("retry-after") if e.headers else None, attempt))
                continue
            raise RuntimeError(f"Jev returned HTTP {e.code} for {case['id']}: {detail}") from None
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(retry_delay(None, attempt))
                continue
            raise RuntimeError(f"cannot reach Jev for {case['id']}: {getattr(e, 'reason', None) or e!r}") from None


def jev_distribution(question, answer):
    """A Jev answer as a probability list in Nex's slot order."""
    if question["type"] == "choice":
        return [answer["probabilities"].get(option, 0.0) for option in question["criteria"]]
    if question["type"] == "score":
        return [answer["probabilities"].get(str(i), 0.0) for i in range(len(question["criteria"]))]
    return [answer["noul"], 1 - answer["noul"]]


def load_cache(path):
    """Cached Jev answers as ``{model_asked: {case_id: entry}}``. The older
    flat format, ``{case_id: entry}`` with ``model_asked`` in each entry, is
    converted on load and keeps every answer. The file is written in the new
    format the next time an answer is saved."""
    if not path.exists():
        return {}
    cache = {}
    for key, value in json.loads(path.read_text(encoding="utf-8")).items():
        if "hash" in value:
            # Flat format: the key is a case id.
            entry = {k: v for k, v in value.items() if k != "model_asked"}
            entry.setdefault("fetched_at", None)
            cache.setdefault(value["model_asked"], {})[key] = entry
        else:
            cache.setdefault(key, {}).update(value)
    return cache


def save_cache(path, cache):
    """Write the cache to a temporary file and move it into place, so an
    interrupted write never leaves a broken cache behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def version_key(name):
    """Orders Jev version names such as ``jev-1.13.0`` by their numbers."""
    return [int(n) for n in re.findall(r"\d+", name)], name


def stale_cases(cases, entries):
    """The cases to ask Jev about, and the newest Jev version among the
    current answers. A case is stale when it has no answer, has changed since,
    or was answered by an older version than the newest."""
    current = {
        c["id"]: entries[c["id"]]["model"]
        for c in cases if c["id"] in entries and entries[c["id"]]["hash"] == case_hash(c)
    }
    newest = max(set(current.values()), key=version_key, default=None)
    return [c for c in cases if current.get(c["id"]) != newest or c["id"] not in current], newest


def ask_all(todo, key, model, workers, cache, entries, cache_path):
    """Ask Jev about ``todo`` in parallel. Each answer goes into ``entries``
    and the cache file as it arrives, so a failed request keeps the answers
    fetched before it. On a failure, queued cases are dropped and the error is
    raised once running requests are done."""
    lock = threading.Lock()

    def ask(case):
        response, ms = ask_jev(case, key, model)
        try:
            entry = {
                "hash": case_hash(case), "model": response["model"], "answer": response["answers"]["q"],
                "latency_ms": ms, "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        except (KeyError, TypeError, AttributeError):
            raise RuntimeError(f"unexpected Jev response for {case['id']}: {json.dumps(response)[:200]}") from None
        with lock:
            entries[case["id"]] = entry
            save_cache(cache_path, cache)

    pool = ThreadPoolExecutor(workers)
    try:
        for future in as_completed([pool.submit(ask, case) for case in todo]):
            future.result()
    finally:
        pool.shutdown(cancel_futures=True)


def fetch_jev(cases, cache_path, model, workers):
    """Jev answers for every case as ``{case_id: entry}``, all from one Jev
    version. Only stale cases (see ``stale_cases``) are sent to Jev."""
    cache = load_cache(cache_path)
    entries = cache.setdefault(model, {})
    for _ in range(VERSION_ROUNDS):
        todo, newest = stale_cases(cases, entries)
        if not todo:
            return entries
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            sys.exit(f"{len(todo)} cases need Jev answers: set TYPESAFE_API_KEY")
        older = sum(c["id"] in entries and entries[c["id"]]["hash"] == case_hash(c) for c in todo)
        note = f", {older} of them answered by a Jev version older than {newest}" if older else ""
        print(f"asking Jev about {len(todo)} cases{note}", file=sys.stderr)
        ask_all(todo, key, model, workers, cache, entries, cache_path)
    if stale_cases(cases, entries)[0]:
        versions = sorted({entries[c["id"]]["model"] for c in cases}, key=version_key)
        sys.exit(f"Jev answered {model} with several versions ({', '.join(versions)}), rerun once the release has settled")
    return entries


def slot_count(question):
    """Answer slots of a question: options for Choice, levels for Score, 2 for Noul."""
    return 2 if question["type"] == "noul" else len(question["criteria"])


def nex_records(path, model, run, cases):
    """One model's records from a ``nex eval --out`` file, checked against the
    current cases. Exits when a case has no record or a record no longer
    matches its case in type, label, or number of slots."""
    by_case = {c["id"]: c for c in cases}
    records = [r for r in run["records"] if r["id"] in by_case]
    found = {r["id"] for r in records}
    if len(found) != len(records):
        sys.exit(f"{path} has duplicate case ids for {model}, rerun nex eval")
    if len(found) != len(cases):
        sys.exit(f"{path} has {len(found)} of {len(cases)} cases for {model}, rerun nex eval")
    stale = []
    for r in records:
        case = by_case[r["id"]]
        question = case["question"]
        slots = len(r.get("raw") or ())
        if r.get("type") != question["type"]:
            stale.append(f"{r['id']} is {r.get('type')} but the case is {question['type']}")
        elif r.get("truth") != truth_index(question, case["label"]):
            stale.append(f"{r['id']} has a different label")
        elif slots != slot_count(question):
            stale.append(f"{r['id']} has {slots} slots but the case has {slot_count(question)}")
    if stale:
        more = f" and {len(stale) - 3} more" if len(stale) > 3 else ""
        sys.exit(f"{path} is out of date for {model}: {', '.join(stale[:3])}{more}. Rerun nex eval")
    return records


def cross_fit_temperatures(records, folds=2):
    """The held-out temperatures of ``nex eval``: each record gets the one
    ``fit_type`` fits on the other folds of its question type, with folds from
    ``fold_of``. Returns ``({id: T}, {qtype: mean T over folds})``."""
    fold = {r["id"]: fold_of(r["id"], folds) for r in records}
    by_id, fitted = {}, {}
    for k in range(folds):
        train = [r for r in records if fold[r["id"]] != k]
        for qtype in QUESTION_TYPES:
            test = [r for r in records if fold[r["id"]] == k and r["type"] == qtype]
            if not test:
                continue
            t = fit_type(qtype, [(r["raw"], r["truth"]) for r in train if r["type"] == qtype]).temperature
            fitted.setdefault(qtype, []).append(t)
            by_id.update((r["id"], t) for r in test)
    return by_id, {q: sum(ts) / len(ts) for q, ts in fitted.items()}


def rescaled(records, temperature):
    """Copies of ``records`` with ``raw`` rescaled by ``temperature[id]``, so
    the scoring below can take them as they are."""
    return [dict(r, raw=apply_temperature(r["raw"], temperature.get(r["id"], 1.0))) for r in records]


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def confident(rows, threshold=0.9):
    """Share of answers with top probability >= threshold, and their accuracy."""
    picked = [argmax(p) == truth for p, truth in rows if max(p) >= threshold]
    return len(picked) / len(rows), (sum(picked) / len(picked) if picked else None)


def report(systems):
    """systems: {name: (records, temperatures)}, with records already
    rescaled and temperatures per question type for the summary. Prints one
    table."""
    print(f"{'system':32s} {'type':7s} {'n':>4s} {'acc':>6s} {'NLL':>6s} {'ECE':>6s} {'Brier':>6s} "
          f"{'p>=0.9 share, acc':>18s} {'MAE':>6s} {'p50 ms':>7s}")
    summary = {}
    for name, (records, temps) in systems.items():
        m = metrics(records)
        for qtype in QUESTION_TYPES:
            if qtype in m:
                m[qtype]["temperature"] = temps.get(qtype, 1.0)
        summary[name] = m
        for qtype in TYPES:
            if qtype not in m:
                continue
            group = [r for r in records if qtype == "all" or r["type"] == qtype]
            share, acc = confident([(r["raw"], r["truth"]) for r in group])
            row = m[qtype]
            row["confident_share"], row["confident_accuracy"] = share, acc
            mae = f"{row['mae']:.3f}" if "mae" in row else "-"
            acc_text = f"{acc:.1%}" if acc is not None else "-"
            print(f"{name:32s} {qtype:7s} {row['n']:4d} {row['accuracy']:6.3f} {row['nll']:6.3f} {row['ece']:6.3f} "
                  f"{row['brier']:6.3f} {share:9.0%}, {acc_text:>6s} {mae:>6s} {row['latency_ms']['p50']:7.0f}")
        print()
    return summary


def head_to_head(jev_records, name, records):
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
        nex_right = argmax(r["raw"]) == r["truth"]
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
    parser.add_argument("--nex-calibration", choices=("cv", "file"), default="cv",
                        help="Nex temperatures: cv fits them on the other folds like nex eval, so the numbers are "
                             "held out (default). file uses $NEX_CALIBRATION or nex/calibration.json, which is "
                             "in-sample if it was fitted on these cases")
    parser.add_argument("--folds", type=int, default=2, help="cross-fit folds for --nex-calibration cv (default: 2)")
    parser.add_argument("--out", help="write the comparison as JSON")
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    cases = load_cases(args.data or [str(ROOT / "evals" / "data")])
    if not cases:
        sys.exit("no eval cases found")
    nex_runs = json.loads(Path(args.nex_results).read_text())
    # Checked before any Jev request, so a stale file costs nothing.
    nex = {model: nex_records(args.nex_results, model, run, cases) for model, run in nex_runs.items()}
    if args.nex_calibration == "file":
        calibration = Calibration.load()
        how = {"method": "file", "path": str(calibration.path)}
        how_text = f"temperatures from {calibration.path}, in-sample if they were fitted on these cases"
    else:
        how = {"method": "cross-fit", "folds": args.folds}
        how_text = (f"{args.folds}-fold cross-fit like nex eval, each fold scored with temperatures "
                    "fit on the other folds (held out)")

    try:
        entries = fetch_jev(cases, Path(args.jev_cache), args.jev_model, args.workers)
    except RuntimeError as e:
        sys.exit(f"{e}\nanswers that arrived before the failure are saved in {args.jev_cache}")

    jev_records = []
    for case in cases:
        entry = entries[case["id"]]
        jev_records.append({
            "id": case["id"], "type": case["question"]["type"], "truth": truth_index(case["question"], case["label"]),
            "raw": jev_distribution(case["question"], entry["answer"]), "label_mass": 1.0,
            "latency_ms": entry["latency_ms"],
        })
    # fetch_jev guarantees a single version.
    version = entries[cases[0]["id"]]["model"]
    systems = {f"jev {version}": (jev_records, {})}
    for model, records in nex.items():
        if args.nex_calibration == "file":
            temps = temperatures_for(calibration, model)
            per_record = {r["id"]: temps[r["type"]] for r in records}
        else:
            per_record, temps = cross_fit_temperatures(records, args.folds)
        systems[f"nex {model}"] = (rescaled(records, per_record), temps)

    print(f"Nex calibration: {how_text}\n")
    summary = report(systems)
    print("p50 latency is per request for Jev (network included) and per question for Nex (local)\n")
    comparison = {}
    for name, (records, _) in list(systems.items())[1:]:
        comparison[name] = head_to_head(jev_records, name, records)
    if args.out:
        result = {"jev_model": version, "nex_calibration": how, "metrics": summary, "head_to_head": comparison}
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
