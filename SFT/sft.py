#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer


DEFAULT_MODEL_PATH = (
    "/capstor/scratch/cscs/p-skarvelis/"
    "apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-400steps/final"
)
DEFAULT_DATASET_NAME = "swiss-ai/apertus-sft-mixture"
DEFAULT_OUTPUT_DIR = f"/capstor/scratch/cscs/{os.environ.get('USER', 'user')}/apertus-greek-sft"
DEFAULT_RUN_NAME = "apertus-greek-sft"
SUPPORTED_CHAT_ROLES = {"user", "assistant", "tool"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised fine-tune an Apertus CPT checkpoint on the swiss-ai/apertus-sft-mixture "
            "chat dataset using the model tokenizer's chat template and assistant-only loss masking."
        )
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        help=(
            "Optional directory containing prepared parquet shards plus metadata.json produced by "
            "scripts/prepare_sft_dataset.py. When set, SFT loads tokenized examples from disk "
            "instead of rendering and tokenizing raw chats at startup."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Checkpoint path or model id for the tokenizer-aligned CPT model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where checkpoints, metrics, and the final SFT model are written.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Logical run name stored in Trainer state.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
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
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bf16 training in Trainer arguments.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Attention backend used when loading the model, e.g. eager, sdpa, or flash_attention_2.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing during SFT. Disabled by default for the current xIELU checkpoint path.",
    )
    parser.add_argument(
        "--distributed-strategy",
        choices=("ddp", "fsdp_full_shard"),
        default="ddp",
        help=(
            "Distributed training strategy. Use fsdp_full_shard to shard model and optimizer state "
            "across ranks when longer-context SFT runs exceed DDP memory headroom."
        ),
    )
    parser.add_argument(
        "--fsdp-min-num-params",
        type=int,
        default=100_000_000,
        help="Minimum parameter count used by FSDP auto-wrap when --distributed-strategy=fsdp_full_shard.",
    )
    parser.add_argument(
        "--fsdp-backward-prefetch",
        choices=("backward_pre", "backward_post"),
        default="backward_pre",
        help="Backward prefetch policy used by FSDP when --distributed-strategy=fsdp_full_shard.",
    )
    parser.add_argument(
        "--fsdp-limit-all-gathers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FSDP limit_all_gathers when --distributed-strategy=fsdp_full_shard.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer/model.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing into an existing output directory without resuming from a checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Optional checkpoint path to resume from. Defaults to the latest checkpoint under output dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=4,
        help="Expected distributed world size for Clariden launches.",
    )
    parser.add_argument(
        "--require-distributed",
        action="store_true",
        help="Fail fast unless launched under a distributed runner such as torchrun.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay passed to Trainer.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Epoch count used when --max-steps is negative.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional max optimizer steps. Use -1 to train for --num-train-epochs.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio used by the scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="Trainer learning rate scheduler type.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging interval in optimizer steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluation interval when a validation split is enabled.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint save interval.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Trainer report_to value. Use none to disable external reporters.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short smoke test with small dataset slices and step counts.",
    )
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=20,
        help="Max steps used by --smoke-test.",
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
    return parser.parse_args()


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
    if args.fsdp_min_num_params <= 0:
        raise SystemExit("--fsdp-min-num-params must be positive.")
    if args.expected_world_size <= 0:
        raise SystemExit("--expected-world-size must be positive.")
    if args.per_device_train_batch_size <= 0:
        raise SystemExit("--per-device-train-batch-size must be positive.")
    if args.per_device_eval_batch_size <= 0:
        raise SystemExit("--per-device-eval-batch-size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive.")
    if args.logging_steps <= 0:
        raise SystemExit("--logging-steps must be positive.")
    if args.eval_steps <= 0:
        raise SystemExit("--eval-steps must be positive.")
    if args.save_steps <= 0:
        raise SystemExit("--save-steps must be positive.")
    if args.save_total_limit <= 0:
        raise SystemExit("--save-total-limit must be positive.")
    if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
        raise SystemExit("--warmup-ratio must be in the range [0, 1).")
    if args.max_steps == 0 or args.max_steps < -1:
        raise SystemExit("--max-steps must be -1 or a positive integer.")
    if args.num_train_epochs <= 0:
        raise SystemExit("--num-train-epochs must be positive.")
    if args.smoke_max_steps <= 0:
        raise SystemExit("--smoke-max-steps must be positive.")
    if args.smoke_train_samples <= 0:
        raise SystemExit("--smoke-train-samples must be positive.")
    if args.smoke_validation_samples < 0:
        raise SystemExit("--smoke-validation-samples cannot be negative.")
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    if model_path.is_absolute() and output_dir.is_absolute() and model_path == output_dir:
        raise SystemExit("--output-dir must differ from --model-path.")
    if args.prepared_dataset_dir:
        prepared_dir = Path(args.prepared_dataset_dir)
        if not prepared_dir.exists() or not prepared_dir.is_dir():
            raise SystemExit(
                f"--prepared-dataset-dir must point to an existing directory, got {prepared_dir}."
            )


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_world_process_zero() -> bool:
    return global_rank() == 0


def rank_zero_print(message: str) -> None:
    if is_world_process_zero():
        print(message, flush=True)


def maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_text(parts: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            parts.append(stripped)
        return
    if isinstance(value, (dict, list)) and value:
        parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return
    text = str(value).strip()
    if text:
        parts.append(text)


def flatten_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return str(content).strip()

    parts: list[str] = []
    append_text(parts, content.get("text"))
    append_text(parts, content.get("formatted_tools"))
    append_text(parts, content.get("tools"))

    for part in content.get("parts") or []:
        if isinstance(part, dict):
            append_text(parts, part.get("text"))
        else:
            append_text(parts, part)

    for block in content.get("blocks") or []:
        if not isinstance(block, dict):
            append_text(parts, block)
            continue
        append_text(parts, block.get("text"))
        for call in block.get("calls") or []:
            if isinstance(call, dict) and any(bool(value) for value in call.values()):
                append_text(parts, {"tool_call": call})
        for output in block.get("outputs") or []:
            if isinstance(output, dict) and any(bool(value) for value in output.values()):
                append_text(parts, {"tool_output": output})

    return "\n".join(parts)


def decorate_message_content(role: str, content: str) -> str:
    if role == "developer":
        return f"Developer instructions:\n{content}"
    if role == "system":
        return f"System message:\n{content}"
    return f"{role}: {content}"


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> list[Dict[str, str]]:
    normalized: list[Dict[str, str]] = []
    system_parts: list[str] = []
    conversation_started = False

    for message in messages:
        role = str(message.get("role") or "").strip()
        if not role:
            continue
        content = flatten_message_content(message.get("content"))
        if not content:
            continue

        if role in {"system", "developer"} and not conversation_started:
            if role == "developer":
                system_parts.append(decorate_message_content(role, content))
            else:
                system_parts.append(content)
            continue

        if role in {"system", "developer"}:
            content = decorate_message_content(role, content)
            role = "user"

        if role not in SUPPORTED_CHAT_ROLES:
            content = decorate_message_content(role, content)
            role = "user"

        normalized.append({"role": role, "content": content})
        conversation_started = True

    if system_parts:
        normalized.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    return normalized


def build_labels(input_ids: Sequence[int], assistant_start_id: int, assistant_end_id: int) -> list[int]:
    labels = [-100] * len(input_ids)
    in_assistant_span = False
    for index, token_id in enumerate(input_ids):
        if token_id == assistant_start_id:
            in_assistant_span = True
            continue
        if token_id == assistant_end_id:
            in_assistant_span = False
            continue
        if in_assistant_span:
            labels[index] = token_id
    return labels


def has_supervised_tokens(example: Dict[str, Any]) -> bool:
    return any(label != -100 for label in example["labels"])


def render_and_tokenize_batch(
    examples: Dict[str, Sequence[Any]],
    tokenizer,
    max_seq_length: int,
    assistant_start_id: int,
    assistant_end_id: int,
) -> Dict[str, Any]:
    input_ids_batch = []
    attention_masks_batch = []
    labels_batch = []

    conversation_ids = examples.get("conversation_id") or [None] * len(examples["messages"])

    for raw_messages, conversation_id in zip(examples["messages"], conversation_ids):
        normalized_messages = normalize_messages(raw_messages)
        try:
            rendered_text = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            normalized_roles = [message["role"] for message in normalized_messages]
            raise ValueError(
                "Failed to render chat template for "
                f"conversation_id={conversation_id!r} with roles={normalized_roles}: {exc}"
            ) from exc
        tokenized = tokenizer(
            rendered_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = build_labels(input_ids, assistant_start_id, assistant_end_id)
        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_masks_batch,
        "labels": labels_batch,
    }


def maybe_limit_dataset(dataset: Dataset, sample_limit: int | None, seed: int) -> Dataset:
    if sample_limit is None or sample_limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_limit))


def load_prepared_dataset_metadata(prepared_dir: Path) -> Dict[str, Any] | None:
    metadata_path = prepared_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_prepared_dataset_split(prepared_dir: Path, split_name: str) -> Dataset | None:
    split_dir = prepared_dir / split_name
    if not split_dir.exists():
        return None
    shard_paths = sorted(split_dir.glob("*.parquet"))
    if not shard_paths:
        raise SystemExit(f"No parquet shards were found under {split_dir}.")
    return load_dataset(
        "parquet",
        data_files=[str(path) for path in shard_paths],
        split="train",
    )


def load_prepared_dataset_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    prepared_dir = Path(args.prepared_dataset_dir)
    metadata = load_prepared_dataset_metadata(prepared_dir)
    if metadata is not None:
        prepared_seq_length = metadata.get("max_seq_length") or metadata.get("sequence_length")
        if prepared_seq_length is not None and int(prepared_seq_length) != args.max_seq_length:
            raise SystemExit(
                "Prepared dataset sequence length mismatch: "
                f"metadata.json says {prepared_seq_length}, but the current run expects {args.max_seq_length}."
            )
        prepared_truncation_side = metadata.get("truncation_side")
        if prepared_truncation_side and prepared_truncation_side != args.truncation_side:
            raise SystemExit(
                "Prepared dataset truncation side mismatch: "
                f"metadata.json says {prepared_truncation_side}, but the current run expects {args.truncation_side}."
            )

    train_dataset = load_prepared_dataset_split(prepared_dir, "train")
    if train_dataset is None:
        raise SystemExit(f"Prepared dataset is missing a train split under {prepared_dir / 'train'}.")

    eval_dataset = load_prepared_dataset_split(prepared_dir, "eval")
    if args.validation_samples > 0 and eval_dataset is None:
        raise SystemExit(
            "--validation-samples was set, but the prepared dataset does not contain an eval split."
        )

    max_train_samples = args.max_train_samples
    if args.smoke_test:
        max_train_samples = args.smoke_train_samples
    train_dataset = maybe_limit_dataset(train_dataset, max_train_samples, args.seed)

    if eval_dataset is not None:
        max_eval_samples = args.max_eval_samples
        if args.smoke_test and args.validation_samples == 0:
            max_eval_samples = args.smoke_validation_samples
        eval_dataset = maybe_limit_dataset(eval_dataset, max_eval_samples, args.seed)

    return train_dataset, eval_dataset


def load_raw_dataset(args: argparse.Namespace) -> Dataset:
    dataset_kwargs: Dict[str, Any] = {
        "path": args.dataset_name,
        "split": args.dataset_split,
    }
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    rank_zero_print(
        f"Loading dataset {args.dataset_name!r} split={args.dataset_split!r} "
        f"config={args.dataset_config!r}."
    )
    return load_dataset(**dataset_kwargs)


def build_dataset_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    prepared_dataset_dir = getattr(args, "prepared_dataset_dir", None)
    if prepared_dataset_dir:
        rank_zero_print(f"Loading prepared SFT dataset from {prepared_dataset_dir}.")
        return load_prepared_dataset_splits(args)

    raw_dataset = load_raw_dataset(args)
    effective_validation_samples = args.validation_samples

    if args.smoke_test:
        smoke_holdout = effective_validation_samples or args.smoke_validation_samples
        smoke_total = args.smoke_train_samples + smoke_holdout
        raw_dataset = maybe_limit_dataset(raw_dataset, smoke_total, args.seed)
        effective_validation_samples = min(smoke_holdout, max(0, len(raw_dataset) - 1))

    if effective_validation_samples > 0:
        if effective_validation_samples >= len(raw_dataset):
            raise SystemExit(
                "--validation-samples must be smaller than the dataset size after any smoke-test cap."
            )
        split_dataset = raw_dataset.train_test_split(
            test_size=effective_validation_samples,
            shuffle=True,
            seed=args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = raw_dataset
        eval_dataset = None

    max_train_samples = args.max_train_samples
    if args.smoke_test:
        max_train_samples = args.smoke_train_samples
    train_dataset = maybe_limit_dataset(train_dataset, max_train_samples, args.seed)
    if eval_dataset is not None:
        max_eval_samples = args.max_eval_samples
        if args.smoke_test and args.validation_samples == 0:
            max_eval_samples = args.smoke_validation_samples
        eval_dataset = maybe_limit_dataset(eval_dataset, max_eval_samples, args.seed)

    return train_dataset, eval_dataset


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int,
    preprocessing_batch_size: int,
    dataset_num_proc: int,
    split_name: str,
) -> Dataset:
    assistant_start_id = tokenizer.convert_tokens_to_ids("<|assistant_start|>")
    assistant_end_id = tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    if assistant_start_id is None or assistant_end_id is None:
        raise SystemExit("Tokenizer is missing assistant boundary tokens required for assistant-only loss.")

    map_kwargs: Dict[str, Any] = {
        "function": lambda batch: render_and_tokenize_batch(
            batch,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            assistant_start_id=assistant_start_id,
            assistant_end_id=assistant_end_id,
        ),
        "batched": True,
        "batch_size": preprocessing_batch_size,
        "remove_columns": dataset.column_names,
        "desc": f"Rendering and tokenizing {split_name} split",
    }
    if dataset_num_proc > 1:
        map_kwargs["num_proc"] = dataset_num_proc
    tokenized_dataset = dataset.map(**map_kwargs)
    tokenized_dataset = tokenized_dataset.filter(has_supervised_tokens, desc=f"Filtering empty {split_name} rows")
    if len(tokenized_dataset) == 0:
        raise SystemExit(f"No usable assistant-supervised examples remained in the {split_name} split.")
    return tokenized_dataset


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    if args.overwrite_output_dir:
        return None
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        return None
    return get_last_checkpoint(str(output_dir))


def training_arguments(args: argparse.Namespace, has_eval_dataset: bool) -> TrainingArguments:
    max_steps = args.smoke_max_steps if args.smoke_test else args.max_steps
    logging_steps = max(1, args.logging_steps)
    eval_steps = max(1, args.eval_steps)
    save_steps = max(1, args.save_steps)
    if max_steps > 0:
        logging_steps = min(logging_steps, max_steps)
        eval_steps = min(eval_steps, max_steps)
        save_steps = min(save_steps, max_steps)

    training_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "run_name": args.run_name,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": max_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": args.bf16,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if has_eval_dataset else "no",
        "eval_steps": eval_steps if has_eval_dataset else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": args.report_to,
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
        "save_safetensors": True,
        "logging_first_step": True,
    }
    if world_size() > 1:
        if args.distributed_strategy == "fsdp_full_shard":
            training_kwargs["fsdp"] = "full_shard auto_wrap"
            training_kwargs["fsdp_config"] = {
                "min_num_params": args.fsdp_min_num_params,
                "backward_prefetch": args.fsdp_backward_prefetch,
                "limit_all_gathers": args.fsdp_limit_all_gathers,
                "activation_checkpointing": False,
            }
        else:
            training_kwargs["ddp_find_unused_parameters"] = False
    return TrainingArguments(**training_kwargs)


def validate_runtime(args: argparse.Namespace, model, tokenizer) -> None:
    current_world_size = world_size()
    if args.require_distributed and current_world_size == 1:
        raise SystemExit("This run requires a distributed launcher such as torchrun.")
    if args.distributed_strategy == "fsdp_full_shard" and current_world_size == 1:
        raise SystemExit("--distributed-strategy fsdp_full_shard requires world size greater than 1.")
    if current_world_size > 1 and current_world_size != args.expected_world_size:
        raise SystemExit(
            f"Expected distributed world size {args.expected_world_size}, but got {current_world_size}."
        )
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise SystemExit("Loaded model does not expose input embeddings.")
    if input_embeddings.num_embeddings != len(tokenizer):
        raise SystemExit(
            "Tokenizer/model vocab mismatch: "
            f"model embeddings={input_embeddings.num_embeddings}, tokenizer={len(tokenizer)}."
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise SystemExit("Loaded tokenizer does not expose apply_chat_template().")

    hidden_act = getattr(getattr(model, "config", None), "hidden_act", None)
    if hidden_act == "xielu" and args.gradient_checkpointing:
        rank_zero_print(
            "Warning: hidden_act=xielu with gradient checkpointing enabled has produced early NaN gradients "
            "on Clariden SFT runs in this repo. Prefer --no-gradient-checkpointing unless you are explicitly "
            "re-validating that combination."
        )
    if hidden_act == "xielu" and args.attn_implementation == "sdpa":
        rank_zero_print(
            "Warning: hidden_act=xielu with attn_implementation=sdpa has hit a fused MHA backward runtime "
            "failure on Clariden SFT runs in this repo. Prefer --attn-implementation eager unless you are "
            "explicitly re-validating sdpa for this checkpoint."
        )
    if (
        hidden_act == "xielu"
        and current_world_size > 1
        and args.distributed_strategy == "ddp"
        and args.max_seq_length > 1024
    ):
        rank_zero_print(
            "Warning: hidden_act=xielu with DDP and max_seq_length > 1024 has previously OOMed on Clariden "
            "at per-device batch size 1. Prefer --distributed-strategy fsdp_full_shard for 2048-token SFT runs."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=resolve_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        use_cache=not args.gradient_checkpointing,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False

    validate_runtime(args, model, tokenizer)

    train_raw_dataset, eval_raw_dataset = build_dataset_splits(args)
    if args.prepared_dataset_dir:
        train_dataset = train_raw_dataset
        eval_dataset = eval_raw_dataset
    else:
        train_dataset = prepare_dataset(
            train_raw_dataset,
            tokenizer,
            max_seq_length=args.max_seq_length,
            preprocessing_batch_size=args.preprocessing_batch_size,
            dataset_num_proc=args.dataset_num_proc,
            split_name="train",
        )
        eval_dataset = None
        if eval_raw_dataset is not None:
            eval_dataset = prepare_dataset(
                eval_raw_dataset,
                tokenizer,
                max_seq_length=args.max_seq_length,
                preprocessing_batch_size=args.preprocessing_batch_size,
                dataset_num_proc=args.dataset_num_proc,
                split_name="eval",
            )

    if is_world_process_zero():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "dataset_summary.json",
            {
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "prepared_dataset_dir": args.prepared_dataset_dir,
                "max_seq_length": args.max_seq_length,
                "truncation_side": args.truncation_side,
                "distributed_strategy": args.distributed_strategy,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
                "smoke_test": args.smoke_test,
            },
        )
        rank_zero_print(
            f"Prepared dataset: train_examples={len(train_dataset)}, "
            f"eval_examples={len(eval_dataset) if eval_dataset is not None else 0}."
        )
    maybe_barrier()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments(args, eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint:
        rank_zero_print(f"Resuming from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    if is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()