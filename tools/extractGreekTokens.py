#!/usr/bin/env python3
"""Extract Greek tokens from a HuggingFace tokenizer readable JSON export.

Σε αυτό το script δίνουμε σαν είσοδο ένα αρχείο JSON από εξαγωγή tokenizer
(π.χ. artifacts/tokenizers/apertus-greek-v1/tokenizer_readable.json)
και εξάγει τα ελληνικά tokens σε ένα αρχείο κειμένου.

Examples:
    # Default paths inside the repo:
    ./run_uenv.sh python tools/extractGreekTokens.py

    # Explicit input / output:
    ./run_uenv.sh python tools/extractGreekTokens.py \\
        --input artifacts/tokenizers/apertus-greek-v1/tokenizer_readable.json \\
        --output greek_tokens.txt

    # Include token ids alongside the token strings:
    ./run_uenv.sh python tools/extractGreekTokens.py --with-ids
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = REPO_ROOT / "artifacts" / "tokenizers" / "apertus-greek-v1" / "tokenizer_readable.json"
DEFAULT_OUTPUT = Path("greek_tokens.txt")


# ---------------------------------------------------------------------------
# Greek Unicode ranges
# ---------------------------------------------------------------------------
def _is_greek_char(ch: str) -> bool:
    """Return True when *ch* is a Greek-script code-point."""
    cp = ord(ch)
    # Greek and Coptic         U+0370 – U+03FF
    # Greek Extended           U+1F00 – U+1FFF
    return (0x0370 <= cp <= 0x03FF) or (0x1F00 <= cp <= 0x1FFF)


def token_contains_greek(token: str) -> bool:
    """Return True when *token* contains at least one Greek character."""
    return any(_is_greek_char(ch) for ch in token)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_greek_tokens(
    tokenizer_json_path: Path,
) -> list[tuple[str, int]]:
    """Read *tokenizer_json_path* and return (token, id) pairs for every
    vocabulary entry that contains at least one Greek character, sorted by id.

    Looks in both ``model.vocab`` (the base BPE vocabulary) and
    ``added_tokens`` (tokens added via :meth:`~transformers.PreTrainedTokenizerBase.add_tokens`).
    """
    with open(tokenizer_json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    seen_ids: set[int] = set()
    greek_entries: list[tuple[str, int]] = []

    # 1. Base BPE vocabulary
    vocab: dict[str, int] = data.get("model", {}).get("vocab", {})
    for token, tid in vocab.items():
        if tid in seen_ids:
            continue
        if token_contains_greek(token):
            greek_entries.append((token, tid))
            seen_ids.add(tid)

    # 2. Added tokens (typically the extended Greek tokens live here)
    for entry in data.get("added_tokens", []):
        token = entry.get("content", "")
        tid = entry.get("id")
        if tid is None or tid in seen_ids:
            continue
        if token_contains_greek(token):
            greek_entries.append((token, tid))
            seen_ids.add(tid)

    greek_entries.sort(key=lambda item: item[1])  # sort by token id
    return greek_entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Greek tokens from a tokenizer readable JSON export.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to tokenizer_readable.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file for Greek tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--with-ids",
        action="store_true",
        help="Include numeric token id next to each token string.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    entries = extract_greek_tokens(args.input)

    with open(args.output, "w", encoding="utf-8") as fh:
        for token, tid in entries:
            if args.with_ids:
                fh.write(f"{tid}\t{token}\n")
            else:
                fh.write(f"{token}\n")

    print(f"Extracted {len(entries)} Greek tokens → {args.output}")


if __name__ == "__main__":
    main()

