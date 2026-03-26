import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer


def build_parser(
    description: str,
    default_model_id: str,
    default_output_dir: Path,
    default_report_path: Path,
    default_readable_tokenizer_path: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model-id",
        default=default_model_id,
        help="Hugging Face model id for the source tokenizer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where the tokenizer will be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=default_report_path,
        help="Path for the JSON metadata report.",
    )
    parser.add_argument(
        "--readable-tokenizer-path",
        type=Path,
        default=default_readable_tokenizer_path,
        help="Path for a schema-preserving tokenizer JSON with decoded token strings where safe.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    return parser


def decode_token_ids(tokenizer, token_ids: List[int]) -> List[str]:
    return [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def build_readable_tokenizer_json(tokenizer, tokenizer_json_path: Path) -> Tuple[Dict[str, Any], int, bool]:
    tokenizer_payload = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    vocab = tokenizer_payload.get("model", {}).get("vocab", {})

    sorted_vocab_items = sorted(vocab.items(), key=lambda item: item[1])
    token_ids = [token_id for _, token_id in sorted_vocab_items]
    decoded_tokens = decode_token_ids(tokenizer, token_ids)
    decoded_counts = Counter(decoded_tokens)

    raw_to_display = {}
    readable_vocab = {}
    collision_fallbacks = 0
    for (raw_token, token_id), decoded_token in zip(sorted_vocab_items, decoded_tokens):
        display_token = decoded_token
        if decoded_counts[decoded_token] > 1:
            display_token = raw_token
            collision_fallbacks += 1

        raw_to_display[raw_token] = display_token
        readable_vocab[display_token] = token_id

    readable_added_tokens = []
    for item in tokenizer_payload.get("added_tokens", []):
        token_id = item.get("id")
        decoded = item.get("content", "") if item.get("special") else tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        readable_item = dict(item)
        if not item.get("special") and decoded_counts[decoded] == 1:
            readable_item["content"] = decoded
        readable_added_tokens.append(readable_item)

    readable_merges = []
    for pair in tokenizer_payload.get("model", {}).get("merges", []):
        if isinstance(pair, list) and len(pair) == 2:
            readable_merges.append(
                [raw_to_display.get(pair[0], pair[0]), raw_to_display.get(pair[1], pair[1])]
            )
        else:
            readable_merges.append(pair)

    readable_payload = dict(tokenizer_payload)
    readable_payload["added_tokens"] = readable_added_tokens
    readable_payload["model"] = dict(tokenizer_payload.get("model", {}))
    readable_payload["model"]["vocab"] = readable_vocab
    readable_payload["model"]["merges"] = readable_merges
    return readable_payload, collision_fallbacks, len(decoded_counts) != len(decoded_tokens)


def extract_tokenizer(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.readable_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(args.output_dir)

    tokenizer_json_path = args.output_dir / "tokenizer.json"
    readable_payload, collision_fallbacks, has_decoded_collisions = build_readable_tokenizer_json(
        tokenizer, tokenizer_json_path
    )
    args.readable_tokenizer_path.write_text(
        json.dumps(readable_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    backend_name = type(getattr(tokenizer, "backend_tokenizer", None)).__name__
    report = {
        "model_id": args.model_id,
        "output_dir": str(args.output_dir),
        "tokenizer_json_path": str(tokenizer_json_path),
        "readable_tokenizer_path": str(args.readable_tokenizer_path),
        "readable_export_collision_fallbacks": collision_fallbacks,
        "readable_export_has_decoded_collisions": has_decoded_collisions,
        "tokenizer_class": tokenizer.__class__.__name__,
        "backend_tokenizer_class": backend_name,
        "vocab_size": len(tokenizer),
        "special_tokens_map": tokenizer.special_tokens_map,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
    }

    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return report