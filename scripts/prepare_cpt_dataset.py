import argparse
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, Sequence

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_TOKENIZER_PATH = "artifacts/tokenizers/apertus-greek-v1"
DEFAULT_GREEK_DATASET = "epfml/FineWeb2-HQ"
DEFAULT_GREEK_CONFIG = "ell_Grek"
DEFAULT_ENGLISH_DATASET = "epfml/FineWeb-HQ"
DEFAULT_TEXT_COLUMN = "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare packed fixed-length CPT training sequences on disk so continued pretraining "
            "does not need to stream and tokenize raw text during the training step."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where parquet shards and metadata.json will be written.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer path used to produce packed sequences.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing the prepared dataset.",
    )

    parser.add_argument(
        "--greek-dataset",
        default=DEFAULT_GREEK_DATASET,
        help="Dataset id for the Greek CPT stream.",
    )
    parser.add_argument(
        "--greek-config",
        default=DEFAULT_GREEK_CONFIG,
        help="Optional dataset config for the Greek CPT stream.",
    )
    parser.add_argument(
        "--greek-split",
        default="train",
        help="Split to use from the Greek dataset.",
    )
    parser.add_argument(
        "--english-dataset",
        default=DEFAULT_ENGLISH_DATASET,
        help="Dataset id for the English anchor stream.",
    )
    parser.add_argument(
        "--english-config",
        help="Optional dataset config for the English anchor stream.",
    )
    parser.add_argument(
        "--english-split",
        default="train",
        help="Split to use from the English anchor dataset.",
    )
    parser.add_argument(
        "--greek-probability",
        type=float,
        default=0.9,
        help="Sampling weight for the Greek stream when interleaving datasets.",
    )
    parser.add_argument(
        "--english-probability",
        type=float,
        default=0.1,
        help="Sampling weight for the English anchor stream when interleaving datasets.",
    )
    parser.add_argument(
        "--stopping-strategy",
        default="first_exhausted",
        choices=("first_exhausted", "all_exhausted"),
        help="Stopping strategy used when interleaving the streaming datasets.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column in both datasets that contains raw text.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Sequence length for the packed training samples.",
    )
    parser.add_argument(
        "--sequences-per-shard",
        type=int,
        default=512,
        help="How many packed sequences to accumulate before writing one parquet shard.",
    )
    parser.add_argument(
        "--max-input-examples",
        type=int,
        help="Optional limit on the number of raw documents to read before stopping.",
    )
    parser.add_argument(
        "--max-output-sequences",
        type=int,
        help="Optional limit on the number of packed sequences to write.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Optional limit on the number of output tokens to write.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive.")
    if args.sequences_per_shard <= 0:
        raise SystemExit("--sequences-per-shard must be positive.")
    if args.max_input_examples is not None and args.max_input_examples <= 0:
        raise SystemExit("--max-input-examples must be positive when set.")
    if args.max_output_sequences is not None and args.max_output_sequences <= 0:
        raise SystemExit("--max-output-sequences must be positive when set.")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be positive when set.")
    if args.greek_probability < 0 or args.english_probability < 0:
        raise SystemExit("Dataset sampling probabilities cannot be negative.")
    if args.greek_probability == 0 and args.english_probability == 0:
        raise SystemExit("At least one dataset probability must be greater than zero.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory {args.output_dir} is not empty. Use --overwrite to replace it."
        )


def has_text(example: Dict[str, Any], text_column: str) -> bool:
    text = example.get(text_column)
    return isinstance(text, str) and bool(text.strip())


def select_text_and_source(example: Dict[str, Any], text_column: str, source_name: str) -> Dict[str, Any]:
    return {
        text_column: example[text_column],
        "__source": source_name,
    }


def load_streaming_dataset(dataset_name: str, dataset_config: str | None, split: str):
    dataset_kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if dataset_config:
        dataset_kwargs["name"] = dataset_config
    return load_dataset(**dataset_kwargs)


def build_source_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    source_name: str,
):
    dataset = load_streaming_dataset(dataset_name, dataset_config, split)
    dataset = dataset.filter(partial(has_text, text_column=text_column))
    map_kwargs: Dict[str, Any] = {
        "function": partial(select_text_and_source, text_column=text_column, source_name=source_name),
    }
    if getattr(dataset, "features", None):
        map_kwargs["remove_columns"] = list(dataset.features.keys())
    return dataset.map(**map_kwargs)


def load_tokenizer(tokenizer_path: str, trust_remote_code: bool):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise

        path = Path(tokenizer_path)
        tokenizer_file = path / "tokenizer.json"
        tokenizer_config_path = path / "tokenizer_config.json"
        if not tokenizer_file.exists():
            raise SystemExit(
                f"Tokenizer metadata references TokenizersBackend, but {tokenizer_file} was not found."
            ) from exc

        tokenizer_config: Dict[str, Any] = {}
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

        compatible_kwargs: Dict[str, Any] = {
            "tokenizer_file": str(tokenizer_file),
        }
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
            "add_prefix_space",
            "model_max_length",
            "padding_side",
            "truncation_side",
            "clean_up_tokenization_spaces",
            "model_input_names",
        ):
            if key in tokenizer_config:
                compatible_kwargs[key] = tokenizer_config[key]

        tokenizer = PreTrainedTokenizerFast(**compatible_kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def write_shard(output_dir: Path, shard_index: int, sequences: Sequence[Sequence[int]]) -> str:
    shard_path = output_dir / f"train-{shard_index:05d}.parquet"
    dataset = Dataset.from_dict({"input_ids": list(sequences)})
    dataset.to_parquet(str(shard_path))
    return shard_path.name


def output_sequence_limit(args: argparse.Namespace) -> int | None:
    limits = []
    if args.max_output_sequences is not None:
        limits.append(args.max_output_sequences)
    if args.max_output_tokens is not None:
        limits.append(args.max_output_tokens // args.max_seq_length)
    if not limits:
        return None
    positive_limits = [limit for limit in limits if limit > 0]
    if not positive_limits:
        raise SystemExit(
            "The requested output token limit is smaller than one full packed sequence."
        )
    return min(positive_limits)


def maybe_flush_shard(
    output_dir: Path,
    shard_index: int,
    shard_sequences: list[list[int]],
    shard_files: list[str],
) -> int:
    if not shard_sequences:
        return shard_index
    shard_files.append(write_shard(output_dir, shard_index, shard_sequences))
    shard_sequences.clear()
    return shard_index + 1


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path, args.trust_remote_code)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise SystemExit("Tokenizer must define eos_token_id so documents can be separated during packing.")

    greek_ds = None
    english_ds = None
    datasets = []
    probabilities = []

    if args.greek_probability > 0:
        greek_ds = build_source_dataset(
            dataset_name=args.greek_dataset,
            dataset_config=args.greek_config,
            split=args.greek_split,
            text_column=args.text_column,
            source_name="greek",
        )
        datasets.append(greek_ds)
        probabilities.append(args.greek_probability)

    if args.english_probability > 0:
        english_ds = build_source_dataset(
            dataset_name=args.english_dataset,
            dataset_config=args.english_config,
            split=args.english_split,
            text_column=args.text_column,
            source_name="english",
        )
        datasets.append(english_ds)
        probabilities.append(args.english_probability)

    if len(datasets) == 1:
        combined_ds = datasets[0]
    else:
        combined_ds = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=args.stopping_strategy,
        )

    max_sequences = output_sequence_limit(args)

    token_buffer: list[int] = []
    buffer_offset = 0
    shard_sequences: list[list[int]] = []
    shard_files: list[str] = []
    source_example_counts = {"greek": 0, "english": 0}
    source_byte_counts = {"greek": 0, "english": 0}
    total_examples = 0
    total_input_bytes = 0
    total_sequences = 0
    shard_index = 0

    for example in combined_ds:
        if args.max_input_examples is not None and total_examples >= args.max_input_examples:
            break
        if max_sequences is not None and total_sequences >= max_sequences:
            break

        text = example[args.text_column]
        source_name = example.get("__source", "unknown")

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_ids.append(eos_token_id)

        total_examples += 1
        total_input_bytes += len(text.encode("utf-8"))
        source_example_counts[source_name] = source_example_counts.get(source_name, 0) + 1
        source_byte_counts[source_name] = source_byte_counts.get(source_name, 0) + len(text.encode("utf-8"))

        token_buffer.extend(token_ids)

        while len(token_buffer) - buffer_offset >= args.max_seq_length:
            if max_sequences is not None and total_sequences >= max_sequences:
                break

            sequence = token_buffer[buffer_offset : buffer_offset + args.max_seq_length]
            shard_sequences.append(sequence)
            buffer_offset += args.max_seq_length
            total_sequences += 1

            if len(shard_sequences) >= args.sequences_per_shard:
                shard_index = maybe_flush_shard(args.output_dir, shard_index, shard_sequences, shard_files)

        if buffer_offset > 0 and buffer_offset >= len(token_buffer) // 2:
            token_buffer = token_buffer[buffer_offset:]
            buffer_offset = 0

    shard_index = maybe_flush_shard(args.output_dir, shard_index, shard_sequences, shard_files)

    metadata = {
        "tokenizer_path": args.tokenizer_path,
        "sequence_length": args.max_seq_length,
        "datasets": {
            "greek": {
                "dataset": args.greek_dataset,
                "config": args.greek_config,
                "split": args.greek_split,
                "probability": args.greek_probability,
            },
            "english": {
                "dataset": args.english_dataset,
                "config": args.english_config,
                "split": args.english_split,
                "probability": args.english_probability,
            },
        },
        "stopping_strategy": args.stopping_strategy,
        "text_column": args.text_column,
        "sequences_per_shard": args.sequences_per_shard,
        "max_input_examples": args.max_input_examples,
        "max_output_sequences": args.max_output_sequences,
        "max_output_tokens": args.max_output_tokens,
        "total_examples": total_examples,
        "total_input_bytes": total_input_bytes,
        "source_example_counts": source_example_counts,
        "source_byte_counts": source_byte_counts,
        "total_sequences": total_sequences,
        "total_output_tokens": total_sequences * args.max_seq_length,
        "discarded_tail_tokens": max(0, len(token_buffer) - buffer_offset),
        "parquet_shards": shard_files,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "total_sequences": total_sequences,
                "total_output_tokens": total_sequences * args.max_seq_length,
                "parquet_shards": len(shard_files),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    # Work around a reproducible interpreter-shutdown crash in the current
    # datasets/pyarrow stack after successful parquet export.
    os._exit(0)


if __name__ == "__main__":
    main()