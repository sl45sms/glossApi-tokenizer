import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


DEFAULT_BASE_TOKENIZER = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_REFERENCE_TOKENIZER = "ilsp/Llama-Krikri-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Greek segmentation between the Apertus tokenizer and a Greek reference tokenizer."
    )
    parser.add_argument(
        "--base-tokenizer",
        default=DEFAULT_BASE_TOKENIZER,
        help="Hugging Face model id or local path for the base tokenizer.",
    )
    parser.add_argument(
        "--reference-tokenizer",
        default=DEFAULT_REFERENCE_TOKENIZER,
        help="Hugging Face model id or local path for the comparison tokenizer.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        help="UTF-8 text file containing one sample per line.",
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
        default=20,
        help="Maximum number of samples to compare from the combined inputs.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional path to write the full JSON comparison report.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizers.",
    )
    return parser.parse_args()


def load_samples(sample_file: Optional[Path], inline_texts: List[str], limit: int) -> List[str]:
    samples = []

    if sample_file is not None:
        file_samples = [
            line.strip() for line in sample_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        samples.extend(file_samples)

    samples.extend(text.strip() for text in inline_texts if text.strip())

    unique_samples = []
    seen = set()
    for sample in samples:
        if sample not in seen:
            unique_samples.append(sample)
            seen.add(sample)

    return unique_samples[:limit]


def chars_per_token(text: str, token_count: int) -> float:
    if token_count == 0:
        return 0.0
    return round(len(text) / token_count, 4)


def compare_sample(base_tokenizer, reference_tokenizer, text: str) -> Dict[str, Any]:
    base_ids = base_tokenizer.encode(text, add_special_tokens=False)
    reference_ids = reference_tokenizer.encode(text, add_special_tokens=False)

    base_tokens = base_tokenizer.convert_ids_to_tokens(base_ids)
    reference_tokens = reference_tokenizer.convert_ids_to_tokens(reference_ids)
    base_decoded_pieces = [
        base_tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in base_ids
    ]
    reference_decoded_pieces = [
        reference_tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in reference_ids
    ]

    base_count = len(base_tokens)
    reference_count = len(reference_tokens)
    delta = base_count - reference_count
    reduction_pct = 0.0 if base_count == 0 else round((delta / base_count) * 100, 2)

    return {
        "text": text,
        "base_token_count": base_count,
        "reference_token_count": reference_count,
        "base_chars_per_token": chars_per_token(text, base_count),
        "reference_chars_per_token": chars_per_token(text, reference_count),
        "token_count_delta": delta,
        "token_count_reduction_pct": reduction_pct,
        "base_tokens": base_tokens,
        "reference_tokens": reference_tokens,
        "base_decoded_pieces": base_decoded_pieces,
        "reference_decoded_pieces": reference_decoded_pieces,
    }


def build_summary(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_base = sum(item["base_token_count"] for item in comparisons)
    total_reference = sum(item["reference_token_count"] for item in comparisons)
    sample_count = len(comparisons)
    delta = total_base - total_reference

    return {
        "sample_count": sample_count,
        "total_base_tokens": total_base,
        "total_reference_tokens": total_reference,
        "total_token_delta": delta,
        "avg_base_tokens_per_sample": round(total_base / sample_count, 4) if sample_count else 0.0,
        "avg_reference_tokens_per_sample": round(total_reference / sample_count, 4)
        if sample_count
        else 0.0,
        "relative_reduction_pct": round((delta / total_base) * 100, 2) if total_base else 0.0,
    }


def print_human_report(summary: Dict[str, Any], comparisons: List[Dict[str, Any]]) -> None:
    print("Tokenizer comparison summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for index, item in enumerate(comparisons, start=1):
        print(f"\nSample {index}")
        print(f"Text: {item['text']}")
        print(
            "Base tokens: "
            f"{item['base_token_count']} | Reference tokens: {item['reference_token_count']} | "
            f"Delta: {item['token_count_delta']}"
        )
        print(
            "Base chars/token: "
            f"{item['base_chars_per_token']} | Reference chars/token: {item['reference_chars_per_token']}"
        )
        print(f"Base tokenization (Apertus): {item['base_decoded_pieces']}")
        print(f"Reference tokenization (KriKri): {item['reference_decoded_pieces']}")


def main() -> None:
    args = parse_args()
    samples = load_samples(args.sample_file, args.text, args.limit)
    if not samples:
        raise SystemExit("Provide at least one sample via --text or --sample-file.")

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    reference_tokenizer = AutoTokenizer.from_pretrained(
        args.reference_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    comparisons = [compare_sample(base_tokenizer, reference_tokenizer, sample) for sample in samples]
    summary = build_summary(comparisons)

    report = {
        "base_tokenizer": args.base_tokenizer,
        "reference_tokenizer": args.reference_tokenizer,
        "summary": summary,
        "samples": comparisons,
    }

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print_human_report(summary, comparisons)


if __name__ == "__main__":
    main()