#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SFT_MODULE_PATH = REPO_ROOT / "SFT" / "sft.py"
SFT_SPEC = importlib.util.spec_from_file_location("repo_sft_module", SFT_MODULE_PATH)
if SFT_SPEC is None or SFT_SPEC.loader is None:
    raise SystemExit(f"Failed to load SFT module from {SFT_MODULE_PATH}.")
SFT_MODULE = importlib.util.module_from_spec(SFT_SPEC)
SFT_SPEC.loader.exec_module(SFT_MODULE)

from repo_tokenizer import load_repo_tokenizer  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'user')}/prepared-datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render and tokenize the Apertus SFT mixture once, then write parquet shards that "
            "SFT/sft.py can load directly without repeating raw-chat preprocessing on every rank."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where train/ and optional eval/ parquet shards plus metadata.json will be written. "
            "Defaults to an IOPS scratch path derived from max sequence length, truncation side, and validation size."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing the prepared dataset.",
    )
    parser.add_argument(
        "--model-path",
        default=SFT_MODULE.DEFAULT_MODEL_PATH,
        help="Checkpoint path or model id whose tokenizer should be used for SFT rendering and tokenization.",
    )
    parser.add_argument(
        "--dataset-name",
        default=SFT_MODULE.DEFAULT_DATASET_NAME,
        help="Dataset id for the SFT mixture.",
    )
    parser.add_argument(
        "--dataset-config",
        help="Optional dataset config name.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to load. The published mixture currently exposes only train.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="Number of examples to hold out from the train split for evaluation.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Optional cap on the number of train examples after any validation split.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional cap on the number of held-out evaluation examples.",
    )
    parser.add_argument(
        "--preprocessing-batch-size",
        type=int,
        default=256,
        help="Batch size used while rendering and tokenizing chat examples.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=1,
        help="Worker count for dataset preprocessing. Keep 1 unless tokenizer multiprocessing is validated.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Sequence length used for SFT tokenization.",
    )
    parser.add_argument(
        "--truncation-side",
        choices=("left", "right"),
        default="left",
        help="Truncation side used when formatted chats exceed --max-seq-length.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed used for dataset slicing and shuffling.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Prepare only a short smoke-test slice of the dataset.",
    )
    parser.add_argument(
        "--smoke-train-samples",
        type=int,
        default=128,
        help="Train sample cap used by --smoke-test.",
    )
    parser.add_argument(
        "--smoke-validation-samples",
        type=int,
        default=16,
        help="Validation sample cap used by --smoke-test when no explicit holdout is set.",
    )
    parser.add_argument(
        "--examples-per-shard",
        type=int,
        default=10000,
        help="How many tokenized examples to store per parquet shard.",
    )
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    validation_tag = args.validation_samples
    if args.smoke_test and validation_tag == 0:
        validation_tag = args.smoke_validation_samples
    return DEFAULT_OUTPUT_ROOT / (
        f"apertus-greek-sft-{args.max_seq_length}-{args.truncation_side}-val{validation_tag}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.validation_samples < 0:
        raise SystemExit("--validation-samples cannot be negative.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise SystemExit("--max-train-samples must be positive when set.")
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        raise SystemExit("--max-eval-samples must be positive when set.")
    if args.preprocessing_batch_size <= 0:
        raise SystemExit("--preprocessing-batch-size must be positive.")
    if args.dataset_num_proc <= 0:
        raise SystemExit("--dataset-num-proc must be positive.")
    if args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive.")
    if args.smoke_train_samples <= 0:
        raise SystemExit("--smoke-train-samples must be positive.")
    if args.smoke_validation_samples < 0:
        raise SystemExit("--smoke-validation-samples cannot be negative.")
    if args.examples_per_shard <= 0:
        raise SystemExit("--examples-per-shard must be positive.")


def write_split(dataset: Dataset, split_dir: Path, split_name: str, examples_per_shard: int) -> list[str]:
    split_dir.mkdir(parents=True, exist_ok=True)
    parquet_shards: list[str] = []
    total_examples = len(dataset)
    shard_count = max(1, math.ceil(total_examples / examples_per_shard))

    for shard_index in range(shard_count):
        shard = dataset.shard(num_shards=shard_count, index=shard_index, contiguous=True)
        shard_path = split_dir / f"{split_name}-{shard_index:05d}.parquet"
        shard.to_parquet(str(shard_path))
        parquet_shards.append(shard_path.name)

    return parquet_shards


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args)
    validate_args(args)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory {args.output_dir} is not empty. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_repo_tokenizer(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.truncation_side = args.truncation_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise SystemExit("Tokenizer must expose either a pad token or eos token.")

    train_raw_dataset, eval_raw_dataset = SFT_MODULE.build_dataset_splits(args)
    train_dataset = SFT_MODULE.prepare_dataset(
        train_raw_dataset,
        tokenizer,
        max_seq_length=args.max_seq_length,
        preprocessing_batch_size=args.preprocessing_batch_size,
        dataset_num_proc=args.dataset_num_proc,
        split_name="train",
    )
    eval_dataset = None
    if eval_raw_dataset is not None:
        eval_dataset = SFT_MODULE.prepare_dataset(
            eval_raw_dataset,
            tokenizer,
            max_seq_length=args.max_seq_length,
            preprocessing_batch_size=args.preprocessing_batch_size,
            dataset_num_proc=args.dataset_num_proc,
            split_name="eval",
        )

    train_shards = write_split(
        train_dataset,
        args.output_dir / "train",
        "train",
        args.examples_per_shard,
    )
    eval_shards: list[str] = []
    if eval_dataset is not None:
        eval_shards = write_split(
            eval_dataset,
            args.output_dir / "eval",
            "eval",
            args.examples_per_shard,
        )

    metadata = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "validation_samples": args.validation_samples,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "smoke_test": args.smoke_test,
        "smoke_train_samples": args.smoke_train_samples if args.smoke_test else None,
        "smoke_validation_samples": args.smoke_validation_samples if args.smoke_test else None,
        "max_seq_length": args.max_seq_length,
        "truncation_side": args.truncation_side,
        "preprocessing_batch_size": args.preprocessing_batch_size,
        "dataset_num_proc": args.dataset_num_proc,
        "examples_per_shard": args.examples_per_shard,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "parquet_shards": {
            "train": train_shards,
            "eval": eval_shards,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
                "train_parquet_shards": len(train_shards),
                "eval_parquet_shards": len(eval_shards),
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