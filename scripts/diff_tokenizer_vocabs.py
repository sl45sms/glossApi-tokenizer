import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


DEFAULT_BASE_TOKENIZER = "artifacts/tokenizers/apertus-base"
DEFAULT_REFERENCE_TOKENIZER = "artifacts/tokenizers/krikri-base"
DEFAULT_REPORT_PATH = Path("artifacts/reports/tokenizer_vocab_diff.json")


def is_greek_text(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the vocabularies of two saved tokenizers and report overlap and differences."
    )
    parser.add_argument(
        "--base-tokenizer",
        default=DEFAULT_BASE_TOKENIZER,
        help="Local path or Hugging Face id for the first tokenizer.",
    )
    parser.add_argument(
        "--reference-tokenizer",
        default=DEFAULT_REFERENCE_TOKENIZER,
        help="Local path or Hugging Face id for the second tokenizer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of example tokens to include per difference set.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the JSON diff report.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["none", "greek"],
        default="none",
        help="Optional decoded-text filter to apply before diffing vocabularies.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizers.",
    )
    return parser.parse_args()


def sorted_vocab_items(vocab: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(vocab.items(), key=lambda item: item[1])


def build_examples(tokenizer, vocab: Dict[str, int], tokens: List[str], limit: int) -> List[Dict[str, str]]:
    examples = []
    for token in tokens[:limit]:
        token_id = vocab[token]
        examples.append(
            {
                "id": token_id,
                "raw": token,
                "decoded": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            }
        )
    return examples


def filter_vocab(tokenizer, vocab: Dict[str, int], filter_mode: str) -> Dict[str, int]:
    if filter_mode == "none":
        return vocab

    filtered_vocab = {}
    for token, token_id in vocab.items():
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if filter_mode == "greek" and is_greek_text(decoded):
            filtered_vocab[token] = token_id

    return filtered_vocab


def main() -> None:
    args = parse_args()

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    reference_tokenizer = AutoTokenizer.from_pretrained(
        args.reference_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    full_base_vocab = base_tokenizer.get_vocab()
    full_reference_vocab = reference_tokenizer.get_vocab()

    base_vocab = filter_vocab(base_tokenizer, full_base_vocab, args.filter_mode)
    reference_vocab = filter_vocab(reference_tokenizer, full_reference_vocab, args.filter_mode)

    base_tokens = set(base_vocab.keys())
    reference_tokens = set(reference_vocab.keys())

    common_tokens = sorted(base_tokens & reference_tokens)
    base_only_tokens = sorted(base_tokens - reference_tokens)
    reference_only_tokens = sorted(reference_tokens - base_tokens)

    report = {
        "base_tokenizer": args.base_tokenizer,
        "reference_tokenizer": args.reference_tokenizer,
        "filter_mode": args.filter_mode,
        "summary": {
            "full_base_vocab_size": len(full_base_vocab),
            "full_reference_vocab_size": len(full_reference_vocab),
            "base_vocab_size": len(base_vocab),
            "reference_vocab_size": len(reference_vocab),
            "common_token_count": len(common_tokens),
            "base_only_token_count": len(base_only_tokens),
            "reference_only_token_count": len(reference_only_tokens),
            "base_overlap_pct": round((len(common_tokens) / len(base_vocab)) * 100, 2) if base_vocab else 0.0,
            "reference_overlap_pct": round((len(common_tokens) / len(reference_vocab)) * 100, 2)
            if reference_vocab
            else 0.0,
        },
        "examples": {
            "base_only": build_examples(base_tokenizer, base_vocab, base_only_tokens, args.limit),
            "reference_only": build_examples(
                reference_tokenizer, reference_vocab, reference_only_tokens, args.limit
            ),
            "common": build_examples(base_tokenizer, base_vocab, common_tokens, args.limit),
        },
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Tokenizer vocabulary diff summary")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Filter mode: {args.filter_mode}")
    print(f"Base-only examples written: {min(args.limit, len(base_only_tokens))}")
    print(f"Reference-only examples written: {min(args.limit, len(reference_only_tokens))}")
    print(f"Common examples written: {min(args.limit, len(common_tokens))}")
    print(f"JSON report: {args.report_path}")


if __name__ == "__main__":
    main()