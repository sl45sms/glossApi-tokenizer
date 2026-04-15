import argparse
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from datasets import interleave_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_MODEL_PATH = "/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-init/"
DEFAULT_OUTPUT_DIR = "/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt"
DEFAULT_GREEK_DATASET = "epfml/FineWeb2-HQ"
DEFAULT_GREEK_CONFIG = "ell_Grek"
DEFAULT_ENGLISH_DATASET = "epfml/FineWeb-HQ"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_RUN_NAME = "apertus-greek-cpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue pretraining an Apertus checkpoint that is already aligned to the extended Greek tokenizer. "
            "This entry point is designed for single-node Clariden runs launched with torchrun."
        )
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local path or model id for the tokenizer-aligned checkpoint to continue training.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where warm-up, full CPT, and final artifacts will be written.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Logical run name recorded in the Trainer state.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the checkpoint tokenizer/model.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Torch dtype used when loading the aligned checkpoint.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bf16 training in Trainer arguments.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention backend passed to model loading, e.g. flash_attention_2 or eager.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing during training.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Trainer phase directories to be reused when they already contain checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed used for model, dataset, and Trainer state.",
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=4,
        help="Expected distributed world size. Set to 1 for single-process debugging.",
    )
    parser.add_argument(
        "--require-distributed",
        action="store_true",
        help="Fail fast when the script is not launched under a distributed runner such as torchrun.",
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
        "--tokenize-batch-size",
        type=int,
        default=1000,
        help="Batch size used by the streaming tokenization map.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length used during tokenization.",
    )

    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=16,
        help="Per-device batch size used by both phases.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps used by both phases.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers used by Trainer.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Base logging interval for both phases.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Base checkpoint save interval for both phases.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of Trainer checkpoints to retain per phase.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="Trainer learning-rate scheduler type.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Value passed to Trainer report_to. Use none to disable external reporters.",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the embedding-only warm-up phase and run full CPT directly.",
    )
    parser.add_argument(
        "--warmup-max-steps",
        type=int,
        default=2000,
        help="Number of optimizer steps in the embedding-only warm-up phase.",
    )
    parser.add_argument(
        "--warmup-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the embedding-only warm-up phase.",
    )
    parser.add_argument(
        "--full-max-steps",
        type=int,
        default=50000,
        help="Number of optimizer steps in the full CPT phase.",
    )
    parser.add_argument(
        "--full-learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for the full CPT phase.",
    )
    parser.add_argument(
        "--full-warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps for the full CPT cosine schedule.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Shrink both phases to a short validation run without changing the base configuration.",
    )
    parser.add_argument(
        "--smoke-warmup-steps",
        type=int,
        default=20,
        help="Warm-up step count used when --smoke-test is enabled.",
    )
    parser.add_argument(
        "--smoke-full-steps",
        type=int,
        default=40,
        help="Full-CPT step count used when --smoke-test is enabled.",
    )
    parser.add_argument(
        "--smoke-full-warmup-steps",
        type=int,
        default=5,
        help="Full-phase scheduler warmup steps used when --smoke-test is enabled.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive.")
    if args.tokenize_batch_size <= 0:
        raise SystemExit("--tokenize-batch-size must be positive.")
    if args.per_device_train_batch_size <= 0:
        raise SystemExit("--per-device-train-batch-size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive.")
    if args.logging_steps <= 0:
        raise SystemExit("--logging-steps must be positive.")
    if args.save_steps <= 0:
        raise SystemExit("--save-steps must be positive.")
    if args.save_total_limit <= 0:
        raise SystemExit("--save-total-limit must be positive.")
    if args.warmup_max_steps < 0 or args.full_max_steps < 0:
        raise SystemExit("Phase step counts cannot be negative.")
    if args.smoke_warmup_steps <= 0 or args.smoke_full_steps <= 0:
        raise SystemExit("Smoke-test step counts must be positive.")
    if args.smoke_full_warmup_steps < 0:
        raise SystemExit("--smoke-full-warmup-steps cannot be negative.")
    if args.full_warmup_steps < 0:
        raise SystemExit("--full-warmup-steps cannot be negative.")

    if args.greek_probability < 0 or args.english_probability < 0:
        raise SystemExit("Dataset sampling probabilities cannot be negative.")
    if args.greek_probability == 0 and args.english_probability == 0:
        raise SystemExit("At least one dataset probability must be greater than zero.")
    if args.expected_world_size <= 0:
        raise SystemExit("--expected-world-size must be positive.")

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    if model_path.is_absolute() and output_dir.is_absolute() and model_path == output_dir:
        raise SystemExit("--output-dir must differ from --model-path so the aligned checkpoint is not overwritten.")


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


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_phase_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def has_text(example: Dict[str, Any], text_column: str) -> bool:
    text = example.get(text_column)
    return isinstance(text, str) and bool(text.strip())


def tokenize_batch(
    examples: Dict[str, Sequence[Any]],
    tokenizer,
    text_column: str,
    max_seq_length: int,
) -> Dict[str, Any]:
    return tokenizer(
        list(examples[text_column]),
        truncation=True,
        max_length=max_seq_length,
    )


def load_streaming_dataset(dataset_name: str, dataset_config: str | None, split: str):
    dataset_kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if dataset_config:
        dataset_kwargs["name"] = dataset_config
    return load_dataset(**dataset_kwargs)


def build_training_dataset(args: argparse.Namespace, tokenizer):
    greek_ds = load_streaming_dataset(args.greek_dataset, args.greek_config, args.greek_split)

    datasets = []
    probabilities = []
    if args.greek_probability > 0:
        datasets.append(greek_ds)
        probabilities.append(args.greek_probability)

    if args.english_probability > 0:
        english_ds = load_streaming_dataset(args.english_dataset, args.english_config, args.english_split)
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

    filtered_ds = combined_ds.filter(partial(has_text, text_column=args.text_column))
    map_kwargs: Dict[str, Any] = {
        "function": partial(
            tokenize_batch,
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_seq_length=args.max_seq_length,
        ),
        "batched": True,
        "batch_size": args.tokenize_batch_size,
    }
    if getattr(filtered_ds, "features", None):
        map_kwargs["remove_columns"] = list(filtered_ds.features.keys())
    return filtered_ds.map(**map_kwargs)


def load_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise SystemExit("Tokenizer has no pad_token or eos_token. Set one before CPT.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args: argparse.Namespace):
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def validate_aligned_checkpoint(tokenizer, model) -> None:
    tokenizer_vocab_size = len(tokenizer)
    input_embedding_size = model.get_input_embeddings().num_embeddings
    if tokenizer_vocab_size != input_embedding_size:
        raise SystemExit(
            "Tokenizer/model vocabulary mismatch: "
            f"len(tokenizer)={tokenizer_vocab_size}, input_embeddings={input_embedding_size}. "
            "Point --model-path at the resized checkpoint produced by extend_apertus_tokenizer.py --base-model."
        )

    output_layer = model.get_output_embeddings()
    if output_layer is not None:
        output_embedding_size = output_layer.weight.shape[0]
        if tokenizer_vocab_size != output_embedding_size:
            raise SystemExit(
                "Tokenizer/model LM head mismatch: "
                f"len(tokenizer)={tokenizer_vocab_size}, lm_head={output_embedding_size}."
            )


def validate_runtime(args: argparse.Namespace) -> int:
    current_world_size = world_size()
    if args.require_distributed and current_world_size <= 1:
        raise SystemExit(
            "This run requires distributed launch. Use torchrun or python -m torch.distributed.run."
        )
    if args.expected_world_size and current_world_size != args.expected_world_size:
        raise SystemExit(
            f"Expected WORLD_SIZE={args.expected_world_size}, but found WORLD_SIZE={current_world_size}."
        )
    return current_world_size


def embedding_warmup_mode(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.get_input_embeddings().weight.requires_grad = True
    output_layer = model.get_output_embeddings()
    if output_layer is not None:
        output_layer.weight.requires_grad = True


def full_training_mode(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def phase_steps(args: argparse.Namespace) -> Dict[str, Dict[str, float | int]]:
    warmup_steps = 0 if args.skip_warmup else args.warmup_max_steps
    full_steps = args.full_max_steps
    full_warmup_steps = args.full_warmup_steps

    if args.smoke_test:
        warmup_steps = 0 if args.skip_warmup else args.smoke_warmup_steps
        full_steps = args.smoke_full_steps
        full_warmup_steps = args.smoke_full_warmup_steps

    full_warmup_steps = min(full_warmup_steps, full_steps)

    return {
        "warmup": {
            "max_steps": warmup_steps,
            "learning_rate": args.warmup_learning_rate,
            "warmup_steps": 0,
        },
        "full": {
            "max_steps": full_steps,
            "learning_rate": args.full_learning_rate,
            "warmup_steps": full_warmup_steps,
        },
    }


def training_arguments(
    args: argparse.Namespace,
    phase_name: str,
    phase_output_dir: Path,
    max_steps: int,
    learning_rate: float,
    warmup_steps: int,
) -> TrainingArguments:
    logging_steps = max(1, min(args.logging_steps, max_steps))
    save_steps = max(1, min(args.save_steps, max_steps))

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(phase_output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "run_name": f"{args.run_name}-{phase_name}",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "bf16": args.bf16,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "lr_scheduler_type": args.lr_scheduler_type,
        "dataloader_num_workers": args.dataloader_num_workers,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": True,
        "save_safetensors": True,
    }
    if world_size() > 1:
        training_kwargs["ddp_find_unused_parameters"] = False
    return TrainingArguments(**training_kwargs)


def run_phase(
    args: argparse.Namespace,
    phase_name: str,
    model,
    train_dataset,
    tokenizer,
    max_steps: int,
    learning_rate: float,
    warmup_steps: int,
) -> Trainer | None:
    if max_steps <= 0:
        rank_zero_print(f"Skipping phase '{phase_name}' because max_steps={max_steps}.")
        return None

    phase_output_dir = Path(args.output_dir) / phase_name
    if args.overwrite_output_dir:
        if is_world_process_zero():
            reset_phase_output_dir(phase_output_dir)
        maybe_barrier()
    else:
        ensure_output_dir(phase_output_dir)

    resume_checkpoint = None
    if not args.overwrite_output_dir:
        resume_checkpoint = get_last_checkpoint(str(phase_output_dir))

    trainer = Trainer(
        model=model,
        args=training_arguments(
            args,
            phase_name=phase_name,
            phase_output_dir=phase_output_dir,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        ),
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    rank_zero_print(
        f"Starting phase '{phase_name}' with max_steps={max_steps}, learning_rate={learning_rate}, "
        f"scheduler_warmup_steps={warmup_steps}."
    )
    if resume_checkpoint:
        rank_zero_print(f"Resuming phase '{phase_name}' from {resume_checkpoint}.")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    maybe_barrier()
    return trainer


def save_run_config(args: argparse.Namespace, current_world_size: int, tokenizer_vocab_size: int) -> None:
    if not is_world_process_zero():
        return

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    effective_global_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * current_world_size
    )
    payload = {
        "args": vars(args),
        "world_size": current_world_size,
        "effective_global_batch_size": effective_global_batch,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "phase_plan": phase_steps(args),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_final_checkpoint(trainer: Trainer, tokenizer, output_dir: str) -> None:
    final_dir = Path(output_dir) / "final"
    ensure_output_dir(final_dir)
    maybe_barrier()
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        trainer.state.save_to_json(str(final_dir / "trainer_state.json"))
    maybe_barrier()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    current_world_size = validate_runtime(args)
    tokenizer = load_tokenizer(args)
    model = load_model(args)
    validate_aligned_checkpoint(tokenizer, model)

    train_dataset = build_training_dataset(args, tokenizer)
    save_run_config(args, current_world_size, len(tokenizer))

    effective_global_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * current_world_size
    )
    rank_zero_print(
        "Loaded aligned checkpoint successfully. "
        f"world_size={current_world_size}, effective_global_batch_size={effective_global_batch}, "
        f"vocab_size={len(tokenizer)}."
    )

    plan = phase_steps(args)

    trainer: Trainer | None = None
    if not args.skip_warmup:
        embedding_warmup_mode(model)
        trainer = run_phase(
            args,
            phase_name="warmup",
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_steps=int(plan["warmup"]["max_steps"]),
            learning_rate=float(plan["warmup"]["learning_rate"]),
            warmup_steps=int(plan["warmup"]["warmup_steps"]),
        )

    full_training_mode(model)
    full_phase_trainer = run_phase(
        args,
        phase_name="full",
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_steps=int(plan["full"]["max_steps"]),
        learning_rate=float(plan["full"]["learning_rate"]),
        warmup_steps=int(plan["full"]["warmup_steps"]),
    )
    if full_phase_trainer is not None:
        trainer = full_phase_trainer

    if trainer is None:
        raise SystemExit("No training phase ran. Check the configured step counts.")

    save_final_checkpoint(trainer, tokenizer, args.output_dir)
    rank_zero_print(f"Saved final CPT checkpoint to {Path(args.output_dir) / 'final'}")


if __name__ == "__main__":
    main()