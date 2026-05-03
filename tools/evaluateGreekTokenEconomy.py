#!/usr/bin/env python3
"""Evaluate held-out Greek token economy by comparing the base Apertus tokenizer against the local extended tokenizer, with an optional Greek reference tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer


DEFAULT_BASE_TOKENIZER = "artifacts/tokenizers/apertus-base"
DEFAULT_EXTENDED_TOKENIZER = "artifacts/tokenizers/apertus-greek-v1"
DEFAULT_REFERENCE_TOKENIZER = "artifacts/tokenizers/krikri-base"
DEFAULT_REPORT_PATH = Path("artifacts/reports/tokenizer_efficiency_eval.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate held-out Greek token economy by comparing the base Apertus tokenizer "
            "against the local extended tokenizer, with an optional Greek reference tokenizer."
        )
    )
    parser.add_argument(
        "--base-tokenizer",
        default=DEFAULT_BASE_TOKENIZER,
        help="Local path or Hugging Face id for the base Apertus tokenizer.",
    )
    parser.add_argument(
        "--extended-tokenizer",
        default=DEFAULT_EXTENDED_TOKENIZER,
        help="Local path or Hugging Face id for the extended Apertus tokenizer.",
    )
    parser.add_argument(
        "--reference-tokenizer",
        default=DEFAULT_REFERENCE_TOKENIZER,
        help="Optional comparison tokenizer. Pass an empty string to disable it.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        help="UTF-8 text file with one held-out Greek sample per line.",
    )
    parser.add_argument(
        "--jsonl-file",
        type=Path,
        help="Optional JSONL file with held-out Greek samples.",
    )
    parser.add_argument(
        "--jsonl-text-field",
        default="text",
        help="Field name to read from each JSONL row when --jsonl-file is used.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Inline sample text. Repeat this argument to pass multiple samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of unique samples to evaluate after filtering.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Discard very short samples before evaluation.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizers.",
    )
    return parser.parse_args()


def iter_text_file(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            yield text


def iter_jsonl_file(path: Path, field_name: str) -> Iterable[str]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        row = json.loads(raw_line)
        value = row.get(field_name)
        if isinstance(value, str):
            text = value.strip()
            if text:
                yield text


def load_samples(args: argparse.Namespace) -> List[str]:
    samples: List[str] = []

    if args.sample_file is not None:
        samples.extend(iter_text_file(args.sample_file))

    if args.jsonl_file is not None:
        samples.extend(iter_jsonl_file(args.jsonl_file, args.jsonl_text_field))

    samples.extend(text.strip() for text in args.text if text and text.strip())

    unique_samples: List[str] = []
    seen = set()
    for sample in samples:
        if len(sample) < args.min_chars:
            continue
        if sample in seen:
            continue
        unique_samples.append(sample)
        seen.add(sample)
        if len(unique_samples) >= args.limit:
            break

    return unique_samples


def chars_per_token(char_count: int, token_count: int) -> float:
    if token_count == 0:
        return 0.0
    return round(char_count / token_count, 4)


def reduction_pct(source_count: int, target_count: int) -> float:
    if source_count == 0:
        return 0.0
    return round(((source_count - target_count) / source_count) * 100, 2)


def analyze_tokenization(tokenizer, text: str) -> Dict[str, Any]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded_pieces = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]
    return {
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded_pieces": decoded_pieces,
    }


def compare_sample(base_tokenizer, extended_tokenizer, text: str, reference_tokenizer=None) -> Dict[str, Any]:
    base = analyze_tokenization(base_tokenizer, text)
    extended = analyze_tokenization(extended_tokenizer, text)

    char_count = len(text)
    base_count = base["token_count"]
    extended_count = extended["token_count"]

    item: Dict[str, Any] = {
        "text": text,
        "char_count": char_count,
        "base_token_count": base_count,
        "extended_token_count": extended_count,
        "base_chars_per_token": chars_per_token(char_count, base_count),
        "extended_chars_per_token": chars_per_token(char_count, extended_count),
        "base_to_extended_token_delta": base_count - extended_count,
        "base_to_extended_reduction_pct": reduction_pct(base_count, extended_count),
        "base_decoded_pieces": base["decoded_pieces"],
        "extended_decoded_pieces": extended["decoded_pieces"],
    }

    if reference_tokenizer is not None:
        reference = analyze_tokenization(reference_tokenizer, text)
        reference_count = reference["token_count"]
        item.update(
            {
                "reference_token_count": reference_count,
                "reference_chars_per_token": chars_per_token(char_count, reference_count),
                "base_to_reference_token_delta": base_count - reference_count,
                "base_to_reference_reduction_pct": reduction_pct(base_count, reference_count),
                "extended_to_reference_token_delta": extended_count - reference_count,
                "extended_to_reference_reduction_pct": reduction_pct(extended_count, reference_count),
                "reference_decoded_pieces": reference["decoded_pieces"],
            }
        )

    return item


def build_summary(items: List[Dict[str, Any]], has_reference: bool) -> Dict[str, Any]:
    total_chars = sum(item["char_count"] for item in items)
    total_base = sum(item["base_token_count"] for item in items)
    total_extended = sum(item["extended_token_count"] for item in items)

    improved = sum(1 for item in items if item["extended_token_count"] < item["base_token_count"])
    tied = sum(1 for item in items if item["extended_token_count"] == item["base_token_count"])
    worse = sum(1 for item in items if item["extended_token_count"] > item["base_token_count"])

    summary: Dict[str, Any] = {
        "sample_count": len(items),
        "total_chars": total_chars,
        "total_base_tokens": total_base,
        "total_extended_tokens": total_extended,
        "avg_base_tokens_per_sample": round(total_base / len(items), 4) if items else 0.0,
        "avg_extended_tokens_per_sample": round(total_extended / len(items), 4) if items else 0.0,
        "base_chars_per_token_weighted": chars_per_token(total_chars, total_base),
        "extended_chars_per_token_weighted": chars_per_token(total_chars, total_extended),
        "base_to_extended_token_delta": total_base - total_extended,
        "base_to_extended_reduction_pct": reduction_pct(total_base, total_extended),
        "improved_sample_count": improved,
        "tied_sample_count": tied,
        "worse_sample_count": worse,
    }

    if has_reference:
        total_reference = sum(item["reference_token_count"] for item in items)
        summary.update(
            {
                "total_reference_tokens": total_reference,
                "avg_reference_tokens_per_sample": round(total_reference / len(items), 4) if items else 0.0,
                "reference_chars_per_token_weighted": chars_per_token(total_chars, total_reference),
                "base_to_reference_token_delta": total_base - total_reference,
                "base_to_reference_reduction_pct": reduction_pct(total_base, total_reference),
                "extended_to_reference_token_delta": total_extended - total_reference,
                "extended_to_reference_reduction_pct": reduction_pct(total_extended, total_reference),
            }
        )

    return summary


def top_examples(items: List[Dict[str, Any]], count: int, reverse: bool) -> List[Dict[str, Any]]:
    sorted_items = sorted(
        items,
        key=lambda item: (
            item["base_to_extended_token_delta"],
            item["base_to_extended_reduction_pct"],
            len(item["text"]),
        ),
        reverse=reverse,
    )
    return [
        {
            "text": item["text"],
            "base_token_count": item["base_token_count"],
            "extended_token_count": item["extended_token_count"],
            "base_to_extended_token_delta": item["base_to_extended_token_delta"],
            "base_to_extended_reduction_pct": item["base_to_extended_reduction_pct"],
        }
        for item in sorted_items[:count]
    ]


def print_human_summary(summary: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    print("Tokenizer efficiency summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    best = top_examples(items, count=5, reverse=True)
    worst = top_examples(items, count=5, reverse=False)

    print("\nBest compression examples")
    for item in best:
        print(
            f"- delta={item['base_to_extended_token_delta']}, "
            f"reduction={item['base_to_extended_reduction_pct']}%, text={item['text']}"
        )

    print("\nWorst or flat examples")
    for item in worst:
        print(
            f"- delta={item['base_to_extended_token_delta']}, "
            f"reduction={item['base_to_extended_reduction_pct']}%, text={item['text']}"
        )


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    if not samples:
        raise SystemExit(
            "Provide held-out Greek text via --sample-file, --jsonl-file, or repeated --text arguments."
        )

    base_tokenizer = load_repo_tokenizer(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    extended_tokenizer = load_repo_tokenizer(
        args.extended_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    reference_tokenizer = None
    if args.reference_tokenizer.strip():
        reference_tokenizer = load_repo_tokenizer(
            args.reference_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

    items = [
        compare_sample(
            base_tokenizer=base_tokenizer,
            extended_tokenizer=extended_tokenizer,
            text=text,
            reference_tokenizer=reference_tokenizer,
        )
        for text in samples
    ]
    summary = build_summary(items, has_reference=reference_tokenizer is not None)

    report = {
        "base_tokenizer": args.base_tokenizer,
        "extended_tokenizer": args.extended_tokenizer,
        "reference_tokenizer": args.reference_tokenizer if args.reference_tokenizer.strip() else None,
        "summary": summary,
        "top_improvements": top_examples(items, count=20, reverse=True),
        "top_regressions_or_ties": top_examples(items, count=20, reverse=False),
        "samples": items,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_human_summary(summary, items)
    print(f"\nSaved report to {args.report_path}")


if __name__ == "__main__":
    main()
