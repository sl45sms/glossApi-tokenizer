import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoTokenizer

from tokenizer_extract_common import build_readable_tokenizer_json


DEFAULT_BASE_TOKENIZER = Path("artifacts/tokenizers/apertus-base")
DEFAULT_TOKEN_FILE = Path("artifacts/vocab_candidates/selected_tokens_v1.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizers/apertus-greek-v1")
DEFAULT_REPORT_PATH = Path("artifacts/reports/tokenizer_apertus_greek_v1.json")
DEFAULT_READABLE_TOKENIZER_PATH = Path("artifacts/tokenizers/apertus-greek-v1/tokenizer_readable.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved Apertus tokenizer, add new tokens from a selected token list, and save the "
            "extended tokenizer with a JSON report."
        )
    )
    parser.add_argument(
        "--base-tokenizer",
        default=str(DEFAULT_BASE_TOKENIZER),
        help="Local tokenizer path or Hugging Face model id for the tokenizer to extend.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="Text file with one candidate token per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the extended tokenizer will be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the JSON report describing the extension run.",
    )
    parser.add_argument(
        "--readable-tokenizer-path",
        type=Path,
        default=DEFAULT_READABLE_TOKENIZER_PATH,
        help="Path for a schema-preserving readable tokenizer JSON export.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=50,
        help="Maximum number of example added/skipped tokens to include in the report.",
    )
    parser.add_argument(
        "--skip-leading-space-variants",
        action="store_true",
        help=(
            "Do not add a second candidate variant with a single leading space for tokens that do not already "
            "contain whitespace."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory or report files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.token_file.exists():
        raise SystemExit(f"Token file not found: {args.token_file}")
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit cannot be negative.")

    base_tokenizer_path = Path(args.base_tokenizer)
    if base_tokenizer_path.exists() and base_tokenizer_path.resolve() == args.output_dir.resolve():
        raise SystemExit("--output-dir must not be the same path as the local --base-tokenizer directory.")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_output_paths(args: argparse.Namespace) -> None:
    ensure_parent_dir(args.report_path)
    ensure_parent_dir(args.readable_tokenizer_path)

    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise SystemExit(f"Refusing to overwrite existing directory: {args.output_dir}. Use --overwrite.")

    for path in (args.report_path, args.readable_tokenizer_path):
        if path.exists():
            if args.overwrite:
                path.unlink()
            else:
                raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def load_candidate_tokens(token_file: Path) -> Tuple[List[str], Dict[str, int]]:
    raw_tokens: List[str] = []
    duplicate_input_count = 0
    seen = set()
    unique_tokens: List[str] = []

    for line in token_file.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue

        raw_tokens.append(token)
        if token in seen:
            duplicate_input_count += 1
            continue

        seen.add(token)
        unique_tokens.append(token)

    return unique_tokens, {
        "raw_non_empty_token_count": len(raw_tokens),
        "unique_input_token_count": len(unique_tokens),
        "duplicate_input_count": duplicate_input_count,
    }


def should_add_leading_space_variant(token: str) -> bool:
    return bool(token) and token == token.strip() and not any(character.isspace() for character in token)


def expand_candidate_tokens(tokens: Sequence[str], include_leading_space_variants: bool) -> Tuple[List[str], Dict[str, int]]:
    expanded_tokens: List[str] = []
    seen = set()
    duplicate_expanded_variant_count = 0
    leading_space_variant_count = 0

    for token in tokens:
        variants = [token]
        if include_leading_space_variants and should_add_leading_space_variant(token):
            variants.append(f" {token}")

        for variant in variants:
            if variant in seen:
                duplicate_expanded_variant_count += 1
                continue

            seen.add(variant)
            expanded_tokens.append(variant)
            if variant.startswith(" "):
                leading_space_variant_count += 1

    return expanded_tokens, {
        "expanded_input_token_count": len(expanded_tokens),
        "leading_space_variant_count": leading_space_variant_count,
        "duplicate_expanded_variant_count": duplicate_expanded_variant_count,
    }


def has_exact_single_token_coverage(tokenizer, token: str) -> Tuple[bool, List[int]]:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) if token_ids else ""
    return len(token_ids) == 1 and decoded == token, token_ids


def partition_tokens(tokenizer, tokens: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    tokens_to_add: List[str] = []
    skipped_tokens: List[Dict[str, Any]] = []

    for token in tokens:
        exact_single_token, token_ids = has_exact_single_token_coverage(tokenizer, token)
        if exact_single_token:
            skipped_tokens.append(
                {
                    "token": token,
                    "reason": "already_present_as_exact_single_token",
                    "existing_token_id": token_ids[0],
                }
            )
            continue

        tokens_to_add.append(token)

    return tokens_to_add, skipped_tokens


def write_readable_export(tokenizer, output_dir: Path, readable_tokenizer_path: Path) -> Dict[str, Any]:
    tokenizer_json_path = output_dir / "tokenizer.json"
    readable_payload, collision_fallbacks, has_decoded_collisions = build_readable_tokenizer_json(
        tokenizer,
        tokenizer_json_path,
    )
    readable_tokenizer_path.write_text(
        json.dumps(readable_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "tokenizer_json_path": str(tokenizer_json_path),
        "readable_tokenizer_path": str(readable_tokenizer_path),
        "readable_export_collision_fallbacks": collision_fallbacks,
        "readable_export_has_decoded_collisions": has_decoded_collisions,
    }


def build_report(
    args: argparse.Namespace,
    tokenizer,
    token_input_stats: Dict[str, int],
    skipped_tokens: Sequence[Dict[str, Any]],
    tokens_to_add: Sequence[str],
    num_added: int,
    readable_export_info: Dict[str, Any],
) -> Dict[str, Any]:
    initial_vocab_size = len(tokenizer) - num_added
    final_vocab_size = len(tokenizer)

    return {
        "base_tokenizer": args.base_tokenizer,
        "token_file": str(args.token_file),
        "output_dir": str(args.output_dir),
        "initial_vocab_size": initial_vocab_size,
        "final_vocab_size": final_vocab_size,
        "num_added": num_added,
        "token_input_stats": token_input_stats,
        "skipped_existing_token_count": len(skipped_tokens),
        "tokens_requested_for_addition": len(tokens_to_add),
        "tokenizer_class": tokenizer.__class__.__name__,
        "special_tokens_map": tokenizer.special_tokens_map,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "variant_strategy": {
            "include_leading_space_variants": not args.skip_leading_space_variants,
        },
        "samples": {
            "added_tokens": list(tokens_to_add[: args.sample_limit]),
            "skipped_tokens": list(skipped_tokens[: args.sample_limit]),
        },
        **readable_export_info,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_output_paths(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    unique_tokens, token_input_stats = load_candidate_tokens(args.token_file)
    expanded_tokens, expansion_stats = expand_candidate_tokens(
        unique_tokens,
        include_leading_space_variants=not args.skip_leading_space_variants,
    )
    token_input_stats.update(expansion_stats)

    tokens_to_add, skipped_tokens = partition_tokens(tokenizer, expanded_tokens)

    num_added = tokenizer.add_tokens(tokens_to_add)
    tokenizer.save_pretrained(args.output_dir)

    readable_export_info = write_readable_export(tokenizer, args.output_dir, args.readable_tokenizer_path)
    report = build_report(
        args,
        tokenizer,
        token_input_stats,
        skipped_tokens,
        tokens_to_add,
        num_added,
        readable_export_info,
    )

    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()