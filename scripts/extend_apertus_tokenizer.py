# this script is meant to be run after selecting new tokens for addition to the Apertus tokenizer, 
# and it will load the base Apertus tokenizer, add the new tokens, and save the extended tokenizer 
# along with a JSON report describing the changes. 
# Optionally, if a base model checkpoint is provided, 
# it will also resize the model's token embeddings and initialize the new embeddings by mean-pooling over the original subtoken embeddings corresponding to each new token.
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer

from tokenizer_extract_common import build_readable_tokenizer_json, normalize_tokenizer_config


DEFAULT_BASE_TOKENIZER = Path("artifacts/tokenizers/apertus-base")
DEFAULT_TOKEN_FILE = Path("artifacts/vocab_candidates/selected_tokens_v1.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizers/apertus-greek-v1")
DEFAULT_REPORT_PATH = Path("artifacts/reports/tokenizer_apertus_greek_v1.json")
DEFAULT_READABLE_TOKENIZER_PATH = Path("artifacts/tokenizers/apertus-greek-v1/tokenizer_readable.json")


def default_model_output_dir() -> Path:
    scratch_root = os.environ.get("SCRATCH")
    if scratch_root:
        return Path(scratch_root) / "apertus-greek-init"
    return Path("artifacts/checkpoints/apertus-greek-init")


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
        "--base-model",
        help=(
            "Optional local model path or Hugging Face model id for a causal LM to resize and initialize from "
            "the base tokenizer's original subtoken embeddings."
        ),
    )
    parser.add_argument(
        "--model-output-dir",
        "--checkpoint-output-dir",
        "--checkpoint-storage-path",
        type=Path,
        default=default_model_output_dir(),
        help=(
            "Directory where the resized and mean-initialized model checkpoint will be saved when --base-model "
            "is provided. Defaults to $SCRATCH/apertus-greek-init when SCRATCH is set, otherwise "
            "artifacts/checkpoints/apertus-greek-init."
        ),
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
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype to use when loading --base-model.",
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

    if args.base_model:
        base_model_path = Path(args.base_model)
        if base_model_path.exists():
            if base_model_path.resolve() == args.output_dir.resolve():
                raise SystemExit("--output-dir must not be the same path as the local --base-model directory.")
            if base_model_path.resolve() == args.model_output_dir.resolve():
                raise SystemExit("--model-output-dir must not be the same path as the local --base-model directory.")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_output_paths(args: argparse.Namespace) -> None:
    ensure_parent_dir(args.output_dir)
    ensure_parent_dir(args.report_path)
    ensure_parent_dir(args.readable_tokenizer_path)
    if args.base_model:
        ensure_parent_dir(args.model_output_dir)

    output_directories = {args.output_dir.resolve(): args.output_dir}
    if args.base_model:
        output_directories[args.model_output_dir.resolve()] = args.model_output_dir

    for directory in output_directories.values():
        if directory.exists():
            if args.overwrite:
                shutil.rmtree(directory)
            else:
                raise SystemExit(f"Refusing to overwrite existing directory: {directory}. Use --overwrite.")

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
        token = line
        if not token.strip():
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


def has_exact_single_token_coverage(tokenizer, token: str) -> Tuple[bool, List[int]]:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) if token_ids else ""
    return len(token_ids) == 1 and decoded == token, token_ids


def has_leading_space_shadow_conflict(tokenizer, token: str) -> Tuple[bool, str, List[int]]:
    if not token or token[0].isspace():
        return False, "", []

    leading_space_token = f" {token}"
    exact_single_token, token_ids = has_exact_single_token_coverage(tokenizer, leading_space_token)
    return exact_single_token, leading_space_token, token_ids


def partition_tokens(
    tokenizer,
    tokens: Sequence[str],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[int]]]:
    tokens_to_add: List[str] = []
    skipped_tokens: List[Dict[str, Any]] = []
    initialization_source_ids: Dict[str, List[int]] = {}

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

        has_shadow_conflict, leading_space_token, leading_space_token_ids = has_leading_space_shadow_conflict(
            tokenizer,
            token,
        )
        if has_shadow_conflict:
            skipped_tokens.append(
                {
                    "token": token,
                    "reason": "would_shadow_existing_leading_space_single_token",
                    "conflicting_token": leading_space_token,
                    "conflicting_token_id": leading_space_token_ids[0],
                }
            )
            continue

        tokens_to_add.append(token)
        initialization_source_ids[token] = list(token_ids)

    return tokens_to_add, skipped_tokens, initialization_source_ids


def resolve_torch_dtype(torch_dtype_name: str):
    if torch_dtype_name == "auto":
        return "auto"

    import torch

    return getattr(torch, torch_dtype_name)


def build_initialization_samples(
    tokenizer,
    initialization_source_ids: Dict[str, List[int]],
    limit: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for token, source_ids in list(initialization_source_ids.items())[:limit]:
        samples.append(
            {
                "token": token,
                "source_token_count": len(source_ids),
                "source_token_ids": list(source_ids),
                "source_decoded_pieces": [
                    tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in source_ids
                ],
            }
        )
    return samples


def initialize_model_embeddings(
    args: argparse.Namespace,
    tokenizer,
    tokens_to_add: Sequence[str],
    initialization_source_ids: Dict[str, List[int]],
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        dtype=torch_dtype,
    )

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    input_embeddings = model.get_input_embeddings().weight
    output_embedding_layer = model.get_output_embeddings()
    output_embeddings = output_embedding_layer.weight if output_embedding_layer is not None else None
    output_embeddings_share_storage = bool(
        output_embeddings is not None and output_embeddings.data_ptr() == input_embeddings.data_ptr()
    )

    initialized_input_count = 0
    initialized_output_count = 0
    source_length_histogram: Dict[str, int] = {}

    with torch.no_grad():
        for token in tokens_to_add:
            source_ids = initialization_source_ids.get(token, [])
            if not source_ids:
                continue

            new_token_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings[new_token_id].copy_(input_embeddings[source_ids].mean(dim=0))
            initialized_input_count += 1

            if output_embeddings is not None and not output_embeddings_share_storage:
                output_embeddings[new_token_id].copy_(output_embeddings[source_ids].mean(dim=0))
                initialized_output_count += 1
            elif output_embeddings is not None:
                initialized_output_count += 1

            histogram_key = str(len(source_ids))
            source_length_histogram[histogram_key] = source_length_histogram.get(histogram_key, 0) + 1

    model.save_pretrained(args.model_output_dir)
    if args.model_output_dir.resolve() != args.output_dir.resolve():
        tokenizer.save_pretrained(args.model_output_dir)
        normalize_tokenizer_config(args.model_output_dir)

    return {
        "enabled": True,
        "base_model": args.base_model,
        "model_output_dir": str(args.model_output_dir),
        "torch_dtype": args.torch_dtype,
        "initialized_input_embeddings": initialized_input_count,
        "initialized_output_embeddings": initialized_output_count,
        "output_embeddings_share_storage": output_embeddings_share_storage,
        "source_subtoken_count_histogram": source_length_histogram,
        "tokenizer_saved_with_model": True,
        "samples": build_initialization_samples(tokenizer, initialization_source_ids, args.sample_limit),
    }


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
    model_initialization_info: Dict[str, Any],
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
        "token_file_strategy": {
            "use_input_tokens_verbatim": True,
        },
        "samples": {
            "added_tokens": list(tokens_to_add[: args.sample_limit]),
            "skipped_tokens": list(skipped_tokens[: args.sample_limit]),
        },
        "model_initialization": model_initialization_info,
        **readable_export_info,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_output_paths(args)

    tokenizer = load_repo_tokenizer(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    unique_tokens, token_input_stats = load_candidate_tokens(args.token_file)
    tokens_to_add, skipped_tokens, initialization_source_ids = partition_tokens(tokenizer, unique_tokens)

    num_added = tokenizer.add_tokens(tokens_to_add)
    tokenizer.save_pretrained(args.output_dir)
    normalize_tokenizer_config(args.output_dir)

    if args.base_model:
        model_initialization_info = initialize_model_embeddings(
            args,
            tokenizer,
            tokens_to_add,
            initialization_source_ids,
        )
    else:
        model_initialization_info = {
            "enabled": False,
            "reason": "no_base_model_provided",
        }

    readable_export_info = write_readable_export(tokenizer, args.output_dir, args.readable_tokenizer_path)
    report = build_report(
        args,
        tokenizer,
        token_input_stats,
        skipped_tokens,
        tokens_to_add,
        num_added,
        readable_export_info,
        model_initialization_info,
    )

    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()