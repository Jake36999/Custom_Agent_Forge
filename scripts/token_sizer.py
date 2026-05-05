"""
Token-length preflight for Aletheia JSONL training artifacts.

Default target:
    output/qlora_skill_dataset.jsonl

The script estimates whether each row fits inside a target context window.
For ChatML-style rows it counts the rendered message text rather than the
raw JSON wrapper, with a small per-message overhead to reduce undercounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT = Path("output") / "qlora_skill_dataset.jsonl"
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 32768)
CHATML_MESSAGE_OVERHEAD = 4
CHATML_REPLY_OVERHEAD = 2


def _load_encoder(encoding_name: str) -> Tuple[Any, str]:
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding(encoding_name), encoding_name
    except Exception:
        return None, "chars_div_4_fallback"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _record_identity(record: Dict[str, Any], line_number: int) -> str:
    metadata = record.get("_aletheia_metadata")
    if isinstance(metadata, dict) and metadata.get("node_id"):
        return str(metadata["node_id"])
    for key in ("node_id", "failure_id", "trajectory_id", "id"):
        if record.get(key):
            return str(record[key])
    digest = hashlib.sha256(_stable_json(record).encode("utf-8")).hexdigest()[:12]
    return f"line:{line_number}:sha:{digest}"


def _message_text(record: Dict[str, Any]) -> Tuple[str, int]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return _stable_json(record), 0

    rendered: List[str] = []
    overhead = CHATML_REPLY_OVERHEAD
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "unknown")).strip() or "unknown"
        content = str(msg.get("content", ""))
        rendered.append(f"<|{role}|>\n{content}\n")
        overhead += CHATML_MESSAGE_OVERHEAD
    return "\n".join(rendered), overhead


def _record_text(record: Dict[str, Any], count_mode: str) -> Tuple[str, int]:
    if count_mode == "json":
        return _stable_json(record), 0
    if count_mode == "messages":
        return _message_text(record)
    raise ValueError(f"Unsupported count mode: {count_mode}")


def _count_tokens(text: str, encoder: Any, overhead: int) -> int:
    if encoder is not None:
        return len(encoder.encode(text, allowed_special="all")) + overhead
    return math.ceil(len(text) / 4.0) + overhead


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw = raw_line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSONL row {line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                yield line_number, obj
            else:
                print(f"[WARN] Skipping non-object JSONL row {line_number}", file=sys.stderr)


def _percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bucket_label(token_count: int, buckets: Tuple[int, ...]) -> str:
    for limit in buckets:
        if token_count <= limit:
            return f"<= {limit}"
    return f"> {buckets[-1]}"


def analyze_file(
    path: Path,
    context_window: int,
    encoding_name: str,
    count_mode: str,
    buckets: Tuple[int, ...] = DEFAULT_BUCKETS,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    encoder, tokenizer = _load_encoder(encoding_name)
    lengths: List[int] = []
    histogram: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    longest: List[Dict[str, Any]] = []
    breaches: List[Dict[str, Any]] = []

    for line_number, record in _iter_jsonl(path):
        text, overhead = _record_text(record, count_mode)
        token_count = _count_tokens(text, encoder, overhead)
        mode_name = str(record.get("_mode", "unknown"))
        identity = _record_identity(record, line_number)
        entry = {
            "line": line_number,
            "id": identity,
            "mode": mode_name,
            "tokens": token_count,
        }

        lengths.append(token_count)
        histogram[_bucket_label(token_count, buckets)] += 1
        mode_counts[mode_name] += 1
        longest.append(entry)
        if token_count > context_window:
            breaches.append(entry)

    longest = sorted(longest, key=lambda item: (-item["tokens"], item["line"]))
    row_count = len(lengths)
    report: Dict[str, Any] = {
        "input": str(path),
        "tokenizer": tokenizer,
        "count_mode": count_mode,
        "context_window": context_window,
        "rows": row_count,
        "mode_counts": dict(sorted(mode_counts.items())),
        "histogram": {label: histogram[label] for label in _histogram_order(buckets)},
        "breach_count": len(breaches),
        "breach_pct": round((len(breaches) / row_count) * 100.0, 4) if row_count else 0.0,
        "min": min(lengths) if lengths else 0,
        "mean": round(sum(lengths) / row_count, 2) if row_count else 0.0,
        "p50": round(_percentile(lengths, 50), 2),
        "p90": round(_percentile(lengths, 90), 2),
        "p95": round(_percentile(lengths, 95), 2),
        "p99": round(_percentile(lengths, 99), 2),
        "max": max(lengths) if lengths else 0,
        "longest": longest,
    }
    return report, breaches


def _histogram_order(buckets: Tuple[int, ...]) -> List[str]:
    return [f"<= {limit}" for limit in buckets] + [f"> {buckets[-1]}"]


def _print_report(report: Dict[str, Any], top_n: int) -> None:
    print("[Aletheia Token Preflight]")
    print(f"Input          : {report['input']}")
    print(f"Rows           : {report['rows']}")
    print(f"Tokenizer      : {report['tokenizer']}")
    print(f"Count mode     : {report['count_mode']}")
    print(f"Context window : {report['context_window']}")
    print(f"Breaches       : {report['breach_count']} ({report['breach_pct']}%)")
    print()
    print("Distribution")
    print(f"  min={report['min']} mean={report['mean']} p50={report['p50']} p90={report['p90']}")
    print(f"  p95={report['p95']} p99={report['p99']} max={report['max']}")
    print()
    print("Histogram")
    for label, count in report["histogram"].items():
        if count:
            print(f"  {label:>8}: {count}")
    print()
    print("Modes")
    for mode_name, count in report["mode_counts"].items():
        print(f"  {mode_name}: {count}")

    longest = report["longest"][:top_n]
    if longest:
        print()
        print(f"Top {len(longest)} Longest Rows")
        for item in longest:
            print(
                f"  line={item['line']} tokens={item['tokens']} "
                f"mode={item['mode']} id={item['id']}"
            )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight token sizes for Aletheia JSONL datasets.")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"JSONL dataset to inspect (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    parser.add_argument(
        "--count-mode",
        choices=("messages", "json"),
        default="messages",
        help="Count rendered ChatML messages or the raw JSON row.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--write-breaches", type=str, default=None)
    parser.add_argument("--json-report", type=str, default=None)
    parser.add_argument(
        "--hard-fail",
        action="store_true",
        help="Exit with code 2 when any row exceeds the context window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    report, breaches = analyze_file(
        input_path,
        context_window=args.context_window,
        encoding_name=args.encoding,
        count_mode=args.count_mode,
    )
    _print_report(report, top_n=max(0, args.top_n))

    if args.write_breaches:
        _write_jsonl(Path(args.write_breaches), breaches)
        print(f"\nWrote breach rows: {args.write_breaches}")

    if args.json_report:
        full_report = dict(report)
        full_report["longest"] = full_report["longest"][: max(0, args.top_n)]
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_report).write_text(
            json.dumps(full_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {args.json_report}")

    if args.hard_fail and report["breach_count"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
