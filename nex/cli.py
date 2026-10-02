"""Command line interface: ``nex serve``, ``nex ask``, ``nex eval``, and
``nex calibrate``.

Every subcommand accepts ``--ollama-host`` and ``--backend-model``, before or
after the subcommand name.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .backend import DEFAULT_BACKEND_MODEL, BackendError, OllamaBackend
from .calibration import DEFAULT_PATH, QUESTION_TYPES, T_MAX, T_MIN, Calibration, at_bound, default_path
from .engine import Nex
from .evaluation import (
    collect,
    cross_fit,
    fit_temperatures,
    load_cases,
    metrics,
    temperatures_for,
)

DEFAULT_DATA = "evals/data"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
# Fewer records per type than this get a warning from calibrate.
MIN_RECORDS = 30
RECORD_FIELDS = ("id", "type", "raw", "truth", "label_mass", "latency_ms")


def main(argv=None, nex=None):
    """Run the CLI and return its exit code. Pass ``nex`` to use that
    instance instead of one built from ``--ollama-host`` and
    ``--backend-model``. Tests use this to inject a ``FakeBackend``."""
    args = build_parser().parse_args(argv)
    try:
        return args.run(args, nex) or 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (BackendError, ValueError, OSError, ImportError) as e:
        message = " ".join(str(e).split()) or type(e).__name__
        print(f"nex: error: {message}", file=sys.stderr)
        return 1


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nex", description="A small System One-style decision model on a local Ollama model."
    )
    _backend_options(parser, None)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # NEX_HOST and NEX_PORT are read in _cmd_serve, so a bad value cannot
    # break the other subcommands or --help.
    p = _subcommand(sub, "serve", "serve the System One HTTP API", _cmd_serve)
    p.add_argument("--host", help=f"address to bind (default: $NEX_HOST or {DEFAULT_HOST})")
    p.add_argument("--port", type=_port, help=f"port to bind, 0 to 65535 (default: $NEX_PORT or {DEFAULT_PORT})")

    p = _subcommand(sub, "ask", "answer questions about one state and print the response JSON", _cmd_ask)
    p.add_argument(
        "--state", required=True, metavar="TEXT|@FILE",
        help="the state as text or JSON, or @FILE to read it from a file (@- reads stdin)",
    )
    p.add_argument("--questions", metavar="JSON|@FILE", help="questions map in the System One wire format")
    p.add_argument("--noul", metavar="TEXT", help="quick yes/no question")
    p.add_argument("--choice", metavar="TEXT", help="quick choice question, needs --options")
    p.add_argument("--options", metavar="A,B,C", help="comma-separated options for --choice")
    p.add_argument("--score", metavar="TEXT", help="quick score question, needs --levels")
    p.add_argument("--levels", metavar="LOW,...,HIGH", help="comma-separated levels for --score, lowest first")
    p.add_argument("--debug", action="store_true", help="include per-question diagnostics")

    p = _subcommand(sub, "eval", "measure accuracy and calibration on a labeled dataset", _cmd_eval)
    _data_option(p)
    p.add_argument(
        "--model", action="append", metavar="MODEL",
        help="backend model to evaluate, repeatable (default: the backend model)",
    )
    _workers_option(p)
    p.add_argument("--folds", type=int, default=2, help="cross-fit folds (default: 2)")
    p.add_argument("--out", metavar="FILE", help="write records and metrics as JSON")

    p = _subcommand(sub, "calibrate", "fit per-type temperatures and save them", _cmd_calibrate)
    _data_option(p)
    p.add_argument("--model", metavar="MODEL", help="backend model to calibrate (default: the backend model)")
    _workers_option(p)
    p.add_argument(
        "--from", dest="from_file", metavar="FILE",
        help="reuse records from a file written by eval --out instead of running the model",
    )
    p.add_argument(
        "--path", metavar="FILE", help=f"calibration file to update (default: $NEX_CALIBRATION or {DEFAULT_PATH})"
    )
    return parser


def _backend_options(parser, default):
    model = os.environ.get("NEX_BACKEND_MODEL", DEFAULT_BACKEND_MODEL)
    parser.add_argument(
        "--ollama-host", default=default, metavar="URL",
        help="Ollama server (default: $OLLAMA_HOST or 127.0.0.1:11434)",
    )
    parser.add_argument(
        "--backend-model", default=default, metavar="MODEL",
        help=f"Ollama model that answers (default: {model}, from $NEX_BACKEND_MODEL)",
    )


def _subcommand(sub, name, help_text, run):
    p = sub.add_parser(name, help=help_text, description=help_text)
    # SUPPRESS keeps a subcommand that omits the option from overwriting a
    # value given before the subcommand name.
    _backend_options(p, argparse.SUPPRESS)
    p.set_defaults(run=run)
    return p


def _data_option(p):
    p.add_argument(
        "--data", action="append", metavar="PATH",
        help=f"dataset .jsonl file or directory, repeatable (default: {DEFAULT_DATA})",
    )


def _workers_option(p):
    p.add_argument(
        "--workers", type=int, default=1,
        help="concurrent requests (default: 1, Ollama serves one at a time anyway)",
    )


def _build_nex(args, calibration=None):
    return Nex(backend=OllamaBackend(model=args.backend_model, host=args.ollama_host), calibration=calibration)


def _warn(message):
    print(f"nex: warning: {message}", file=sys.stderr)


def _port(text):
    try:
        port = int(text)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"expected a port from 0 to 65535, got {text!r}")
    return port


def _serve_address(args):
    """Host and port for ``nex serve``. Options win over ``NEX_HOST`` and
    ``NEX_PORT``, and an empty value counts as unset. An empty host would
    otherwise bind every interface."""
    host = (args.host or "").strip() or os.environ.get("NEX_HOST", "").strip() or DEFAULT_HOST
    if args.port is not None:
        return host, args.port
    env_port = os.environ.get("NEX_PORT", "").strip()
    if not env_port:
        return host, DEFAULT_PORT
    try:
        return host, _port(env_port)
    except argparse.ArgumentTypeError as e:
        raise ValueError(f"NEX_PORT: {e}") from None


def _cmd_serve(args, nex):
    from .server import serve

    host, port = _serve_address(args)
    serve(nex or _build_nex(args), host, port)


def _cmd_ask(args, nex):
    state = read_value(args.state)
    questions = _ask_questions(args)
    nex = nex or _build_nex(args)
    response = nex.system_one(state, questions, model=args.backend_model)
    print(json.dumps(response.to_dict(include_diagnostics=args.debug), indent=2, ensure_ascii=False))


def read_value(value):
    """Text or ``@FILE``. Content that parses as a JSON object or array is
    used as JSON, anything else as a plain string."""
    if value.startswith("@"):
        value = _read_text(value).rstrip("\r\n")
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if isinstance(parsed, (dict, list)) else value


def _ask_questions(args):
    quick = {}
    if args.noul is not None:
        quick["noul"] = {"type": "noul", "instructions": args.noul}
    if args.choice is not None:
        if not args.options:
            raise ValueError("--choice needs --options, e.g. --options billing,technical,sales")
        quick["choice"] = {"type": "choice", "instructions": args.choice, "criteria": dict.fromkeys(_split(args.options, "--options"))}
    elif args.options:
        raise ValueError("--options only works with --choice")
    if args.score is not None:
        if not args.levels:
            raise ValueError("--score needs --levels, e.g. --levels low,medium,high")
        quick["score"] = {"type": "score", "instructions": args.score, "criteria": _split(args.levels, "--levels")}
    elif args.levels:
        raise ValueError("--levels only works with --score")

    if args.questions is not None:
        if quick:
            raise ValueError("use either --questions or the quick forms (--noul, --choice, --score), not both")
        try:
            questions = json.loads(_read_text(args.questions))
        except json.JSONDecodeError as e:
            raise ValueError(f"--questions is not valid JSON: {e}") from None
        return questions
    if not quick:
        raise ValueError("nothing to ask: give --questions, --noul, --choice, or --score")
    return quick


def _read_text(value):
    if value.startswith("@"):
        name = value[1:]
        return sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")
    return value


def _split(text, flag):
    items = [s.strip() for s in text.split(",") if s.strip()]
    if len(set(items)) != len(items):
        raise ValueError(f"{flag} lists the same entry twice")
    return items


def _load_data(paths):
    paths = paths or [DEFAULT_DATA]
    cases = load_cases(paths)
    if not cases:
        raise ValueError(f"no cases found in {', '.join(map(str, paths))}")
    return cases


def _check_counts(args):
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if getattr(args, "folds", 2) < 2:
        raise ValueError("--folds must be at least 2")


def _cmd_eval(args, nex):
    _check_counts(args)
    cases = _load_data(args.data)
    nex = nex or _build_nex(args)
    results = {}
    for model in args.model or [args.backend_model]:
        name = nex.resolve_backend(model).model
        if name in results:
            continue
        print(f"{name}: asking {len(cases)} cases", file=sys.stderr)
        records = collect(nex, cases, model, workers=args.workers, progress=_progress(name))
        results[name] = {
            "records": records,
            "raw": metrics(records),
            "cross_fit": cross_fit(records, args.folds),
            "current": metrics(records, temperatures_for(nex.calibration, name)),
        }
        if len(results) > 1:
            print()
        print(_eval_report(name, results[name]))
        # Written after every model so a failure later keeps earlier results.
        if args.out:
            _write_json(args.out, results)


def _cmd_calibrate(args, nex):
    _check_counts(args)
    # Loaded by explicit path, so a NEX_CALIBRATION file that does not exist
    # yet is created without the "running uncalibrated" warning. Loading
    # first also stops a malformed file before the model runs.
    path = Path(args.path) if args.path else default_path()
    calibration = Calibration.load(path)
    if not args.path and not os.environ.get("NEX_CALIBRATION"):
        print(
            f"nex: note: this updates the packaged calibration file {path}. "
            "Pass --path FILE or set NEX_CALIBRATION to write elsewhere.",
            file=sys.stderr,
        )
    nex = nex or _build_nex(args, calibration)
    model = args.model or args.backend_model
    if args.from_file:
        name, records = _records_from(args.from_file, model, nex)
    else:
        cases = _load_data(args.data)
        name = nex.resolve_backend(model).model
        print(f"{name}: asking {len(cases)} cases", file=sys.stderr)
        records = collect(nex, cases, model, workers=args.workers, progress=_progress(name))
    if not records:
        raise ValueError(f"no records for {name}")

    before = temperatures_for(calibration, name)
    notes = []
    for qtype, fit in fit_temperatures(records).items():
        if fit.n < MIN_RECORDS:
            _warn(f"{qtype}: only {fit.n} records, a temperature fitted on fewer than {MIN_RECORDS} may not hold on new cases")
        if at_bound(fit.temperature):
            side, bound, cases = ("lower", T_MIN, "right") if fit.temperature < 1 else ("upper", T_MAX, "wrong")
            _warn(
                f"{qtype}: not saved, kept T={before[qtype]:.2f}. The fit ran into the {side} bound "
                f"T={bound:g}, which happens when the model gets nearly every case {cases}."
            )
            continue
        if fit.mae_worse:
            _warn(
                f"{qtype}: T={fit.temperature:.3f} makes probabilities and confidence honest but pulls the "
                f"score field toward the middle of the scale (MAE {fit.mae_raw:.3f} -> {fit.mae_fitted:.3f}). "
                "This model's raw Score answers carry little uncertainty. Prefer a stronger model for Score "
                "questions, or read the argmax of probabilities instead of rounding score."
            )
        calibration.set(name, qtype, fit.temperature, fit.n)
    calibration.save(path)
    after = temperatures_for(calibration, name)
    print(_calibrate_report(name, path, metrics(records, before), metrics(records, after), notes))


def _records_from(path, model, nex):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or not all(isinstance(v, dict) and "records" in v for v in data.values()):
        raise ValueError(f"{path}: expected the JSON written by nex eval --out")
    if model is None and len(data) == 1:
        name = next(iter(data))
    else:
        name = nex.resolve_backend(model).model
    if name not in data:
        raise ValueError(f"{path} has no records for {name} (it has {', '.join(data)}), pick one with --model")
    records = data[name]["records"]
    for i, r in enumerate(records):
        if not isinstance(r, dict) or any(k not in r for k in RECORD_FIELDS):
            raise ValueError(f"{path}: record {i} of {name} lacks one of {', '.join(RECORD_FIELDS)}")
    return name, records


def _progress(name):
    tty = sys.stderr.isatty()

    def report(done, total):
        if tty:
            print(f"\r{name}: {done}/{total}", end="\n" if done == total else "", file=sys.stderr, flush=True)
        elif done == total or done % max(1, total // 10) == 0:
            print(f"{name}: {done}/{total}", file=sys.stderr, flush=True)

    return report


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _f(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def _pair(a, b):
    return f"{_f(a)} -> {_f(b)}"


def _table(header, rows):
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = []
    for row in [header, *rows]:
        cells = [c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _eval_report(name, result):
    raw, cv, current = result["raw"], result["cross_fit"], result["current"]
    header = [
        "type", "n", "acc", "NLL raw -> cv", "ECE raw -> cv", "Brier raw -> cv",
        "mean T", "MAE raw -> cv", "mass", "p50 ms", "p95 ms",
    ]
    rows = []
    for key in [*QUESTION_TYPES, "all"]:
        if key not in raw:
            continue
        r, c = raw[key], cv["metrics"].get(key, {})
        rows.append([
            key,
            str(r["n"]),
            _f(r["accuracy"]),
            _pair(r["nll"], c.get("nll")),
            _pair(r["ece"], c.get("ece")),
            _pair(r["brier"], c.get("brier")),
            _f(cv["temperatures"].get(key), 2),
            _pair(r["mae"], c.get("mae")) if "mae" in r else "-",
            _f(r["label_mass_mean"]),
            _f(r["latency_ms"]["p50"], 0),
            _f(r["latency_ms"]["p95"], 0),
        ])
    temps = ", ".join(f"{q} T={current[q]['temperature']:.2f}" for q in QUESTION_TYPES if q in current)
    return "\n".join([
        f"{name}: {raw['all']['n']} cases. raw = uncalibrated, cv = {cv['folds']}-fold cross-fit "
        "(each fold scored with temperatures fit on the other folds)",
        _table(header, rows),
        f"current calibration, in-sample if it was fitted on these cases ({temps}): "
        f"NLL {_f(current['all']['nll'])}, ECE {_f(current['all']['ece'])}, Brier {_f(current['all']['brier'])}",
    ])


def _calibrate_report(name, path, before, after, notes=()):
    header = ["type", "n", "T before", "T after", "NLL before -> after", "ECE before -> after", "MAE before -> after"]
    rows = []
    for key in [*QUESTION_TYPES, "all"]:
        if key not in after:
            continue
        b, a = before[key], after[key]
        rows.append([
            key,
            str(a["n"]),
            _f(b.get("temperature")),
            _f(a.get("temperature")),
            _pair(b["nll"], a["nll"]),
            _pair(b["ece"], a["ece"]),
            _pair(b["mae"], a["mae"]) if "mae" in a else "-",
        ])
    return "\n".join([
        f"{name}: fitted on {after['all']['n']} records, saved to {path}",
        _table(header, rows),
        *notes,
        "after is measured on the records the temperatures were fit on. nex eval gives a held-out estimate.",
    ])
