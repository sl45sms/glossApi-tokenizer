#!/usr/bin/env python3
"""Unified tokenizer-extension initialization and distillation entrypoint.

This script consolidates the previous distillation attempts into one entrypoint
that supports:
- weighted-mean initialization
- retok initialization
- retok-distill with real-context CLM tuning

The retok-distill path is designed for Alps/Clariden distributed launches via
`torchrun` across all GPUs and multiple nodes.
"""

import argparse
import copy
import gc
import json
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified token embedding initialization and distributed distillation."
    )
    parser.add_argument("--token-file", type=Path, required=True, help="Text file with one token per line.")
    parser.add_argument(
        "--base-tokenizer",
        default="artifacts/tokenizers/apertus-base",
        help="Base tokenizer used to derive source subtokens for initialization.",
    )
    parser.add_argument("--base-model", required=True, help="Base model id/path.")
    parser.add_argument(
        "--extended-tokenizer",
        required=True,
        help="Path to the extended tokenizer containing the target vocabulary.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the initialized checkpoint.")
    parser.add_argument(
        "--init-strategy",
        choices=("weighted-mean", "retok", "retok-distill"),
        default="retok-distill",
        help="Embedding initialization strategy.",
    )
    parser.add_argument(
        "--max-trainable-tokens",
        type=int,
        default=1500,
        help=(
            "Maximum number of new tokens to activate in this stage. "
            "Use 0 to disable the cap and activate all candidates at once."
        ),
    )
    parser.add_argument(
        "--deferred-token-file",
        type=Path,
        default=None,
        help=(
            "Optional path for writing deferred tokens for the next stage. "
            "Defaults to <output-dir>/deferred_tokens_next_stage.txt."
        ),
    )
    parser.add_argument(
        "--untied-output-init-strategy",
        choices=("zero", "mean"),
        default="zero",
        help=(
            "Initialization strategy for new untied lm_head rows. "
            "Use zero by default to avoid making new output tokens immediately competitive before CPT."
        ),
    )
    parser.add_argument(
        "--train-untied-output-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train new untied lm_head rows during tokenizer distillation. Disabled by default so distill focuses on "
            "input embeddings and leaves output-head adaptation to CPT."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--distill-steps", type=int, default=500)
    parser.add_argument("--distill-lr", type=float, default=5e-6)
    parser.add_argument("--distill-samples", type=int, default=5000)
    parser.add_argument("--distill-contexts-per-token", type=int, default=8)
    parser.add_argument("--distill-max-seq-length", type=int, default=1024)
    parser.add_argument("--distill-warmup-steps", type=int, default=50)
    parser.add_argument("--distill-batch-size", type=int, default=16)
    parser.add_argument("--distill-reg-weight", type=float, default=0.1)
    parser.add_argument("--distill-stream-timeout", type=int, default=600)
    parser.add_argument(
        "--distill-sync-interval",
        type=int,
        default=10,
        help="Distributed row-sync interval in optimizer steps.",
    )
    parser.add_argument(
        "--distill-sync-start-step",
        type=int,
        default=50,
        help="Do not run periodic distributed row-syncs before this step.",
    )
    parser.add_argument(
        "--distill-checkpoint-interval",
        type=int,
        default=100,
        help="Interval for writing distilled-row checkpoints.",
    )
    parser.add_argument(
        "--distill-max-invalid-steps",
        type=int,
        default=20,
        help="Abort distillation early if this many invalid (non-finite) steps are observed.",
    )
    parser.add_argument(
        "--distill-lr-decay-on-invalid",
        type=float,
        default=0.5,
        help="Multiply optimizer LR by this factor when invalid grads/rows are detected.",
    )
    parser.add_argument("--fineweb2-cache-dir", type=str, default=None)
    parser.add_argument("--refresh-context-cache", action="store_true")

    parser.add_argument(
        "--require-xielu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require xIELU CUDA kernel availability. Disable only for debugging.",
    )
    parser.add_argument("--report-path", type=Path, default=Path("artifacts/reports/unified_token_distill_report.json"))
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--run-greek-mmlu-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run GreekMMLU evaluation after saving the checkpoint (rank 0 only).",
    )
    parser.add_argument(
        "--eval-output-json",
        type=Path,
        default=Path("artifacts/reports/greek_mmlu_unified_distill_eval.json"),
    )
    parser.add_argument(
        "--fail-below-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if trained accuracy is below base accuracy when eval is enabled.",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward chat-template mode to GreekMMLU evaluator.",
    )

    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def distributed_enabled() -> bool:
    return world_size() > 1


def is_rank_zero() -> bool:
    return global_rank() == 0


def rank_zero_print(message: str) -> None:
    if is_rank_zero():
        print(message, flush=True)


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def init_distributed() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        device = torch.device("cuda", local_rank())
    else:
        device = torch.device("cpu")

    if distributed_enabled() and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        timeout_seconds = int(os.environ.get("DIST_TIMEOUT_SECONDS", "7200"))
        pg_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "timeout": timedelta(seconds=timeout_seconds),
        }
        if device.type == "cuda":
            pg_kwargs["device_id"] = local_rank()
        try:
            dist.init_process_group(**pg_kwargs)
        except TypeError:
            # Some torch builds do not support device_id in init_process_group.
            pg_kwargs.pop("device_id", None)
            dist.init_process_group(**pg_kwargs)
        rank_zero_print(
            f"Initialized process group backend={backend} timeout={timeout_seconds}s world_size={world_size()}"
        )

    return device


def teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def detect_xielu_cuda() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "xielu_cuda_available": False,
        "xielu_cuda_message": "",
    }
    try:
        import xielu.ops  # noqa: F401

        _ = torch.classes.xielu.XIELU()
        info["xielu_cuda_available"] = True
        info["xielu_cuda_message"] = "xIELU CUDA kernel loaded successfully."
    except Exception as exc:  # pragma: no cover - runtime dependent
        info["xielu_cuda_message"] = (
            f"xIELU CUDA kernel not available ({exc}). "
            "Install with: pip install --no-build-isolation git+https://github.com/nickjbrowning/XIELU"
        )
    return info


def validate_args(args: argparse.Namespace) -> None:
    if not args.token_file.exists():
        raise SystemExit(f"Token file not found: {args.token_file}")
    if args.distill_steps <= 0:
        raise SystemExit("--distill-steps must be positive.")
    if args.max_trainable_tokens < 0:
        raise SystemExit("--max-trainable-tokens cannot be negative.")
    if args.train_untied_output_rows and args.untied_output_init_strategy == "zero":
        rank_zero_print(
            "Training untied output rows with zero initialization. This is allowed, but expect CPT to carry most of the recovery."
        )
    if args.distill_batch_size <= 0:
        raise SystemExit("--distill-batch-size must be positive.")
    if args.distill_samples <= 0:
        raise SystemExit("--distill-samples must be positive.")
    if args.distill_contexts_per_token <= 0:
        raise SystemExit("--distill-contexts-per-token must be positive.")
    if args.distill_sync_interval <= 0:
        rank_zero_print("--distill-sync-interval <= 0: periodic sync disabled, final sync only.")
    if args.distill_sync_start_step < 0:
        raise SystemExit("--distill-sync-start-step cannot be negative.")
    if args.distill_checkpoint_interval <= 0:
        raise SystemExit("--distill-checkpoint-interval must be positive.")
    if args.distill_max_seq_length <= 0:
        raise SystemExit("--distill-max-seq-length must be positive.")
    if args.distill_max_invalid_steps <= 0:
        raise SystemExit("--distill-max-invalid-steps must be positive.")
    if not 0.0 < args.distill_lr_decay_on_invalid < 1.0:
        raise SystemExit("--distill-lr-decay-on-invalid must be in (0, 1).")


def rows_are_finite(embeddings: torch.Tensor, row_ids: torch.Tensor) -> bool:
    rows = embeddings.index_select(0, row_ids)
    return bool(torch.isfinite(rows).all().item())


def decay_optimizer_lr(optimizer: torch.optim.Optimizer, factor: float) -> float:
    current_lr = optimizer.param_groups[0]["lr"]
    new_lr = current_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    return new_lr



def prepare_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            rank_zero_print(f"Output directory exists, resume mode enabled: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)



def load_candidate_tokens(token_file: Path) -> List[str]:
    raw_tokens = []
    for line in token_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_tokens.append(line)
    return list(dict.fromkeys(raw_tokens))



def has_exact_single_token_coverage(tokenizer, token: str) -> bool:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        return False
    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    return decoded == token



def filter_trainable_tokens(base_tokenizer, extended_tokenizer, tokens: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[int]]]:
    trainable: List[str] = []
    skipped: List[Dict[str, Any]] = []
    source_map: Dict[str, List[int]] = {}

    for token in tokens:
        ext_id = extended_tokenizer.convert_tokens_to_ids(token)
        if not isinstance(ext_id, int):
            skipped.append({"token": token, "reason": "missing_in_extended_tokenizer"})
            continue

        source_ids = base_tokenizer.encode(token, add_special_tokens=False)
        if not source_ids:
            skipped.append({"token": token, "reason": "no_base_subtokens"})
            continue

        if has_exact_single_token_coverage(base_tokenizer, token):
            skipped.append({"token": token, "reason": "already_single_token_in_base"})
            continue

        trainable.append(token)
        source_map[token] = source_ids

    return trainable, skipped, source_map


def cap_trainable_tokens(
    trainable_tokens: Sequence[str],
    source_map: Dict[str, List[int]],
    max_trainable_tokens: int,
) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    if max_trainable_tokens == 0 or len(trainable_tokens) <= max_trainable_tokens:
        return list(trainable_tokens), [], dict(source_map)

    selected = list(trainable_tokens[:max_trainable_tokens])
    deferred = list(trainable_tokens[max_trainable_tokens:])
    selected_set = set(selected)
    selected_map = {token: source_map[token] for token in selected if token in selected_set}
    return selected, deferred, selected_map


def build_stage_tokenizer(base_tokenizer, stage_tokens: Sequence[str]):
    stage_tokenizer = copy.deepcopy(base_tokenizer)
    added = stage_tokenizer.add_tokens(list(stage_tokens))
    return stage_tokenizer, added


def write_deferred_tokens_file(args: argparse.Namespace, deferred_tokens: Sequence[str]) -> Optional[Path]:
    target_path = args.deferred_token_file or (args.output_dir / "deferred_tokens_next_stage.txt")
    if deferred_tokens:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("\n".join(deferred_tokens) + "\n", encoding="utf-8")
        return target_path

    if target_path.exists():
        target_path.unlink()
    return None


def serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload



def compute_weighted_mean(embeddings: torch.Tensor, source_ids: List[int]) -> torch.Tensor:
    if len(source_ids) == 1:
        return embeddings[source_ids[0]]
    weights = torch.tensor([1.0 / (tid + 1) for tid in source_ids], device=embeddings.device, dtype=embeddings.dtype)
    weights = weights / weights.sum()
    return (embeddings[source_ids] * weights.unsqueeze(1)).sum(dim=0)



def compute_retok_mean(embeddings: torch.Tensor, source_ids: List[int]) -> torch.Tensor:
    if len(source_ids) == 1:
        return embeddings[source_ids[0]]
    if len(source_ids) == 2:
        return (embeddings[source_ids[0]] + embeddings[source_ids[1]]) / 2.0
    return (embeddings[source_ids[:-1]].mean(dim=0) + embeddings[source_ids[-1]]) / 2.0



def resolve_fineweb2_cache_dir(args: argparse.Namespace) -> str:
    if args.fineweb2_cache_dir:
        return args.fineweb2_cache_dir
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return str(Path(scratch) / "FineWeb2-HQ")
    return os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))



def collect_contexts_ahocorasick(
    args: argparse.Namespace,
    tokenizer,
    tokens: Sequence[str],
    cache_dir: Path,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    map_path = cache_dir / "context_map.json"
    if map_path.exists() and not args.refresh_context_cache:
        rank_zero_print(f"Using cached contexts from {map_path}")
        return json.loads(map_path.read_text(encoding="utf-8"))

    if args.refresh_context_cache:
        for fpath in cache_dir.glob("t_*.pt"):
            fpath.unlink()

    fineweb2_cache = resolve_fineweb2_cache_dir(args)
    os.makedirs(fineweb2_cache, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = fineweb2_cache

    ds = load_dataset(
        "epfml/FineWeb2-HQ",
        "ell_Grek",
        split="train",
        streaming=True,
        cache_dir=fineweb2_cache,
    )

    token_info: Dict[str, Tuple[str, int]] = {}
    for token in tokens[: args.distill_samples]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int):
            continue
        clean = token.replace(" ", "").replace("Ġ", "").strip()
        if clean and clean not in token_info:
            token_info[clean] = (token, token_id)

    if not token_info:
        return []

    try:
        import ahocorasick

        automaton = ahocorasick.Automaton()
        for clean in token_info:
            automaton.add_word(clean, clean)
        automaton.make_automaton()
    except Exception:
        automaton = None

    contexts_needed = {clean: args.distill_contexts_per_token for clean in token_info}
    contexts: List[Dict[str, Any]] = []

    start_time = time.time()
    docs_scanned = 0
    for example in ds:
        docs_scanned += 1
        text = example.get("text", "")
        if not text:
            continue

        if automaton is not None:
            matches = [clean for _, clean in automaton.iter(text)]
        else:
            matches = [clean for clean in contexts_needed if clean in text]

        if not matches:
            if docs_scanned % 5000 == 0:
                elapsed = time.time() - start_time
                rank_zero_print(
                    f"Context collection progress: docs={docs_scanned}, contexts={len(contexts)}, remaining={len(contexts_needed)}, elapsed={elapsed:.0f}s"
                )
            if time.time() - start_time > args.distill_stream_timeout:
                break
            continue

        sentences = [
            sentence.strip()
            for sentence in text.replace("!", ".").replace(";", ".").split(".")
            if 30 < len(sentence.strip()) < 1000
        ]
        if not sentences:
            if time.time() - start_time > args.distill_stream_timeout:
                break
            continue

        for sentence in sentences:
            for clean in list(set(matches)):
                if clean not in contexts_needed:
                    continue
                if clean not in sentence:
                    continue
                token, token_id = token_info[clean]
                tokenized = tokenizer.encode(sentence, add_special_tokens=False)
                if token_id not in tokenized:
                    continue

                idx = len(contexts)
                sample = {"token": token, "text": sentence, "index": idx}
                torch.save(sample, cache_dir / f"t_{idx:06d}.pt")
                contexts.append({"token": token, "index": idx})
                contexts_needed[clean] -= 1
                if contexts_needed[clean] <= 0:
                    del contexts_needed[clean]

            if not contexts_needed:
                break

        elapsed = time.time() - start_time
        if docs_scanned % 5000 == 0:
            rank_zero_print(
                f"Context collection progress: docs={docs_scanned}, contexts={len(contexts)}, remaining={len(contexts_needed)}, elapsed={elapsed:.0f}s"
            )
        if not contexts_needed:
            break
        if elapsed > args.distill_stream_timeout:
            rank_zero_print("Context collection timeout reached, proceeding with collected contexts.")
            break

    map_path.write_text(json.dumps(contexts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rank_zero_print(
        f"Context collection finished: contexts={len(contexts)}, tokens_targeted={len(token_info)}, tokens_completed={len(token_info) - len(contexts_needed)}"
    )
    return contexts



def load_local_texts(cache_dir: Path, contexts: Sequence[Dict[str, Any]], rank: int, ws: int) -> List[str]:
    local_entries = contexts[rank::ws]
    texts: List[str] = []
    for entry in local_entries:
        index = int(entry["index"])
        fpath = cache_dir / f"t_{index:06d}.pt"
        if fpath.exists():
            payload = torch.load(fpath, map_location="cpu")
            text = payload.get("text", "")
            if text:
                texts.append(text)
    return texts



def build_grad_mask(new_ids_tensor: torch.Tensor, vocab_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    if new_ids_tensor.numel() > 0:
        mask[new_ids_tensor] = 1.0
    return mask



def sync_new_rows(
    input_embeddings: torch.Tensor,
    output_embeddings: Optional[torch.Tensor],
    new_ids_tensor: torch.Tensor,
) -> None:
    if not dist.is_available() or not dist.is_initialized() or new_ids_tensor.numel() == 0:
        return

    ws = world_size()
    with torch.no_grad():
        input_rows = input_embeddings.index_select(0, new_ids_tensor).contiguous()
        dist.all_reduce(input_rows, op=dist.ReduceOp.SUM)
        input_rows /= ws
        input_embeddings.index_copy_(0, new_ids_tensor, input_rows)

        if output_embeddings is not None:
            output_rows = output_embeddings.index_select(0, new_ids_tensor).contiguous()
            dist.all_reduce(output_rows, op=dist.ReduceOp.SUM)
            output_rows /= ws
            output_embeddings.index_copy_(0, new_ids_tensor, output_rows)



def save_distill_rows_checkpoint(
    checkpoint_path: Path,
    step: int,
    input_embeddings: torch.Tensor,
    output_embeddings: Optional[torch.Tensor],
    new_ids_tensor: torch.Tensor,
) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "new_ids": new_ids_tensor.detach().cpu(),
        "input_rows": input_embeddings.index_select(0, new_ids_tensor).detach().cpu(),
    }
    if output_embeddings is not None:
        payload["output_rows"] = output_embeddings.index_select(0, new_ids_tensor).detach().cpu()
    torch.save(payload, checkpoint_path)



def load_distill_rows_checkpoint(
    checkpoint_path: Path,
    input_embeddings: torch.Tensor,
    output_embeddings: Optional[torch.Tensor],
    new_ids_tensor: torch.Tensor,
) -> int:
    if not checkpoint_path.exists():
        return 0

    payload = torch.load(checkpoint_path, map_location="cpu")
    ckpt_new_ids = payload.get("new_ids")
    if ckpt_new_ids is None:
        return 0

    ckpt_new_ids = ckpt_new_ids.to(new_ids_tensor.device)
    if ckpt_new_ids.shape != new_ids_tensor.shape or not torch.equal(ckpt_new_ids, new_ids_tensor):
        raise SystemExit(
            "Distillation checkpoint token set does not match current token list. "
            "Use --overwrite or clear distill_checkpoints first."
        )

    with torch.no_grad():
        input_embeddings.index_copy_(0, new_ids_tensor, payload["input_rows"].to(input_embeddings.device))
        if output_embeddings is not None and "output_rows" in payload:
            output_embeddings.index_copy_(0, new_ids_tensor, payload["output_rows"].to(output_embeddings.device))

    step = int(payload.get("step", -1)) + 1
    return max(step, 0)



def evaluate_greek_mmlu(args: argparse.Namespace) -> Dict[str, Any]:
    import subprocess

    evaluator = REPO_ROOT / "evaluation" / "evaluate_greek_mmlu.py"
    if not evaluator.exists():
        return {"ran": False, "reason": f"Evaluator not found: {evaluator}"}

    cmd = [
        sys.executable,
        str(evaluator),
        "--base-model",
        args.base_model,
        "--trained-model",
        str(args.output_dir),
        "--output-json",
        str(args.eval_output_json),
    ]

    if args.use_chat_template:
        cmd.append("--use-chat-template")
    else:
        cmd.append("--no-use-chat-template")

    result = subprocess.run(cmd, text=True, capture_output=True)
    payload: Dict[str, Any] = {
        "ran": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }

    if result.returncode != 0:
        return payload

    if args.eval_output_json.exists():
        report = json.loads(args.eval_output_json.read_text(encoding="utf-8"))
        payload["report"] = report
        payload["base_accuracy"] = report.get("base_accuracy")
        payload["trained_accuracy"] = report.get("trained_accuracy")
        payload["accuracy_delta"] = report.get("accuracy_delta")

    return payload



def run_distillation(
    args: argparse.Namespace,
    model,
    tokenizer,
    trainable_tokens: Sequence[str],
    device: torch.device,
) -> Dict[str, Any]:
    if args.init_strategy != "retok-distill":
        return {"enabled": False, "reason": "init_strategy_is_not_retok_distill"}

    cache_dir = args.output_dir / "teacher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    contexts: List[Dict[str, Any]] = []
    if is_rank_zero():
        contexts = collect_contexts_ahocorasick(args, tokenizer, trainable_tokens, cache_dir)
    maybe_barrier()

    if not contexts:
        map_path = cache_dir / "context_map.json"
        if map_path.exists():
            contexts = json.loads(map_path.read_text(encoding="utf-8"))

    if not contexts:
        rank_zero_print("No contexts collected. Keeping retok initialization without distillation.")
        return {
            "enabled": True,
            "performed": False,
            "reason": "no_contexts_collected",
        }

    ws = world_size()
    rank = global_rank()
    local_texts = load_local_texts(cache_dir, contexts, rank=rank, ws=ws)
    if not local_texts:
        rank_zero_print("Warning: some ranks received no local contexts; those ranks will stay idle during training.")

    input_embeddings = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    output_embeddings = out_layer.weight if out_layer is not None else None
    tied = bool(output_embeddings is not None and output_embeddings.data_ptr() == input_embeddings.data_ptr())
    if tied:
        output_embeddings = None
    train_untied_output_rows = bool(output_embeddings is not None and args.train_untied_output_rows)

    for param in model.parameters():
        param.requires_grad = False
    input_embeddings.requires_grad = True
    if output_embeddings is not None:
        output_embeddings.requires_grad = train_untied_output_rows

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    new_ids = sorted(
        {
            tokenizer.convert_tokens_to_ids(token)
            for token in trainable_tokens
            if isinstance(tokenizer.convert_tokens_to_ids(token), int)
        }
    )
    if not new_ids:
        return {
            "enabled": True,
            "performed": False,
            "reason": "no_new_token_ids",
        }

    new_ids_tensor = torch.tensor(new_ids, device=device, dtype=torch.long)
    grad_mask = build_grad_mask(new_ids_tensor, input_embeddings.shape[0], device)
    new_ids_set = set(new_ids)

    w_in_init = input_embeddings.index_select(0, new_ids_tensor).detach().clone()

    params = [input_embeddings]
    if train_untied_output_rows and output_embeddings is not None:
        params.append(output_embeddings)

    optimizer = torch.optim.AdamW(params, lr=args.distill_lr, weight_decay=1e-4)
    if args.distill_warmup_steps > 0:
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.distill_warmup_steps,
            num_training_steps=args.distill_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.distill_steps)

    checkpoint_dir = args.output_dir / "distill_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "distill_rows.pt"

    start_step = 0
    if checkpoint_path.exists():
        start_step = load_distill_rows_checkpoint(
            checkpoint_path,
            input_embeddings,
            output_embeddings,
            new_ids_tensor,
        )
        rank_zero_print(f"Resuming distillation from step {start_step}.")
    maybe_barrier()

    if distributed_enabled():
        start_step_min = torch.tensor(start_step, device=device, dtype=torch.int32)
        start_step_max = torch.tensor(start_step, device=device, dtype=torch.int32)
        dist.all_reduce(start_step_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(start_step_max, op=dist.ReduceOp.MAX)
        if int(start_step_min.item()) != int(start_step_max.item()):
            raise SystemExit(
                "Inconsistent distillation resume step across ranks. "
                "Ensure all nodes see the same checkpoint directory."
            )

    if start_step >= args.distill_steps:
        rank_zero_print(
            f"Distillation already complete at step {start_step} (target {args.distill_steps}); skipping training loop."
        )
        return {
            "enabled": True,
            "performed": False,
            "reason": "already_completed",
            "resume_step": start_step,
            "num_contexts": len(contexts),
            "local_contexts": len(local_texts),
            "num_new_token_ids": len(new_ids_set),
            "recoveries": 0,
            "invalid_steps": 0,
        }

    model.train()
    start_time = time.time()
    invalid_steps = 0
    recoveries = 0

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def should_sync(step: int) -> bool:
        if not distributed_enabled():
            return False
        if step == args.distill_steps - 1:
            return True
        if args.distill_sync_interval <= 0:
            return False
        if step < args.distill_sync_start_step:
            return False
        return (step + 1) % args.distill_sync_interval == 0

    # Keep last known-good new-token rows to recover from non-finite updates.
    last_good_input_rows = input_embeddings.index_select(0, new_ids_tensor).detach().clone()
    last_good_output_rows = (
        output_embeddings.index_select(0, new_ids_tensor).detach().clone()
        if output_embeddings is not None
        else None
    )

    for step in range(start_step, args.distill_steps):
        if local_texts:
            if len(local_texts) <= args.distill_batch_size:
                batch_texts = local_texts
            else:
                batch_texts = random.sample(local_texts, args.distill_batch_size)
        else:
            batch_texts = []

        loss: Optional[torch.Tensor] = None

        if not batch_texts:
            if should_sync(step):
                sync_new_rows(input_embeddings, output_embeddings, new_ids_tensor)
            continue

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.distill_max_seq_length,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        target_mask = torch.isin(shift_labels, new_ids_tensor)
        if not target_mask.any():
            scheduler.step()
            if should_sync(step):
                sync_new_rows(input_embeddings, output_embeddings, new_ids_tensor)
            continue

        selected_logits = shift_logits[target_mask].float()
        selected_labels = shift_labels[target_mask]
        finite_rows = torch.isfinite(selected_logits).all(dim=1)
        if not finite_rows.any():
            invalid_steps += 1
            if invalid_steps % 5 == 0:
                new_lr = decay_optimizer_lr(optimizer, args.distill_lr_decay_on_invalid)
                rank_zero_print(
                    f"step {step}: all selected logits non-finite, decayed LR to {new_lr:.2e}"
                )
            scheduler.step()
            if should_sync(step):
                sync_new_rows(input_embeddings, output_embeddings, new_ids_tensor)
            local_fatal = 1 if invalid_steps >= args.distill_max_invalid_steps else 0
            fatal_tensor = torch.tensor(local_fatal, device=device, dtype=torch.int32)
            if distributed_enabled():
                dist.all_reduce(fatal_tensor, op=dist.ReduceOp.MAX)
            if fatal_tensor.item() > 0:
                rank_zero_print(
                    f"Stopping distillation early after {invalid_steps} invalid steps (non-finite logits)."
                )
                break
            continue

        loss = F.cross_entropy(selected_logits[finite_rows], selected_labels[finite_rows])

        if args.distill_reg_weight > 0:
            reg_loss = F.mse_loss(
                input_embeddings.index_select(0, new_ids_tensor),
                w_in_init,
            )
            loss = loss + args.distill_reg_weight * reg_loss

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            invalid_grad = False
            if input_embeddings.grad is not None:
                input_embeddings.grad *= grad_mask.unsqueeze(1)
                if not torch.isfinite(input_embeddings.grad).all():
                    invalid_grad = True
            if output_embeddings is not None and output_embeddings.grad is not None:
                output_embeddings.grad *= grad_mask.unsqueeze(1)
                if not torch.isfinite(output_embeddings.grad).all():
                    invalid_grad = True

            if invalid_grad:
                invalid_steps += 1
                optimizer.zero_grad(set_to_none=True)
                new_lr = decay_optimizer_lr(optimizer, args.distill_lr_decay_on_invalid)
                rank_zero_print(
                    f"step {step}: non-finite gradients detected, skipped update and decayed LR to {new_lr:.2e}"
                )
            else:
                optimizer.step()
                if rows_are_finite(input_embeddings, new_ids_tensor) and (
                    output_embeddings is None or rows_are_finite(output_embeddings, new_ids_tensor)
                ):
                    invalid_steps = 0
                    last_good_input_rows = input_embeddings.index_select(0, new_ids_tensor).detach().clone()
                    if output_embeddings is not None:
                        last_good_output_rows = (
                            output_embeddings.index_select(0, new_ids_tensor).detach().clone()
                        )
                else:
                    recoveries += 1
                    with torch.no_grad():
                        input_embeddings.index_copy_(0, new_ids_tensor, last_good_input_rows)
                        if output_embeddings is not None and last_good_output_rows is not None:
                            output_embeddings.index_copy_(0, new_ids_tensor, last_good_output_rows)
                    invalid_steps += 1
                    new_lr = decay_optimizer_lr(optimizer, args.distill_lr_decay_on_invalid)
                    rank_zero_print(
                        f"step {step}: recovered non-finite rows from last checkpoint, LR -> {new_lr:.2e}"
                    )
        else:
            invalid_steps += 1
            new_lr = decay_optimizer_lr(optimizer, args.distill_lr_decay_on_invalid)
            rank_zero_print(
                f"step {step}: non-finite loss detected, skipped update and decayed LR to {new_lr:.2e}"
            )

        scheduler.step()

        local_fatal = 1 if invalid_steps >= args.distill_max_invalid_steps else 0
        fatal_tensor = torch.tensor(local_fatal, device=device, dtype=torch.int32)
        if distributed_enabled():
            dist.all_reduce(fatal_tensor, op=dist.ReduceOp.MAX)
        if fatal_tensor.item() > 0:
            rank_zero_print(
                f"Stopping distillation early after {invalid_steps} consecutive invalid steps."
            )
            break

        do_sync = should_sync(step)
        if do_sync:
            sync_new_rows(input_embeddings, output_embeddings, new_ids_tensor)

        if is_rank_zero() and ((step + 1) % args.distill_checkpoint_interval == 0 or step == args.distill_steps - 1):
            save_distill_rows_checkpoint(
                checkpoint_path,
                step,
                input_embeddings,
                output_embeddings,
                new_ids_tensor,
            )

        if (step % 10 == 0 or step == args.distill_steps - 1) and loss is not None:
            elapsed = time.time() - start_time
            lr_value = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.distill_lr
            rank_zero_print(
                f"distill step {step}/{args.distill_steps} loss={loss.item():.4f} lr={lr_value:.2e} elapsed={elapsed:.0f}s"
            )

    # Final synchronization before saving.
    if distributed_enabled():
        # Best-effort final sync. Avoid hard barrier deadlocks at shutdown.
        try:
            sync_new_rows(input_embeddings, output_embeddings, new_ids_tensor)
        except Exception as exc:  # pragma: no cover - runtime distributed path
            rank_zero_print(f"Final distributed sync skipped due to exception: {exc}")

    return {
        "enabled": True,
        "performed": True,
        "num_contexts": len(contexts),
        "local_contexts": len(local_texts),
        "num_new_token_ids": len(new_ids_set),
        "train_untied_output_rows": train_untied_output_rows,
        "recoveries": recoveries,
        "invalid_steps": invalid_steps,
    }



def initialize_embeddings(
    args: argparse.Namespace,
    model,
    tokenizer,
    trainable_tokens: Sequence[str],
    source_map: Dict[str, List[int]],
) -> Dict[str, Any]:
    input_embeddings = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    output_embeddings = out_layer.weight if out_layer is not None else None
    tied = bool(output_embeddings is not None and output_embeddings.data_ptr() == input_embeddings.data_ptr())

    initialized = 0
    skipped = 0

    with torch.no_grad():
        for token in trainable_tokens:
            source_ids = source_map.get(token, [])
            if not source_ids:
                skipped += 1
                continue

            token_id = tokenizer.convert_tokens_to_ids(token)
            if not isinstance(token_id, int):
                skipped += 1
                continue

            if args.init_strategy == "weighted-mean":
                emb = compute_weighted_mean(input_embeddings, source_ids)
            else:
                emb = compute_retok_mean(input_embeddings, source_ids)

            input_embeddings[token_id].copy_(emb)
            if output_embeddings is not None and not tied:
                if args.untied_output_init_strategy == "mean":
                    output_embeddings[token_id].copy_(emb)
                else:
                    output_embeddings[token_id].zero_()
            initialized += 1

    return {
        "initialized": initialized,
        "skipped": skipped,
        "strategy": args.init_strategy,
        "output_embeddings_tied": tied,
        "untied_output_init_strategy": args.untied_output_init_strategy if not tied else "tied",
    }



def load_model_for_tokenizer(args: argparse.Namespace, tokenizer):
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": resolve_torch_dtype(args.torch_dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return model



def main() -> None:
    args = parse_args()
    validate_args(args)

    device = init_distributed()
    model = None

    try:
        if is_rank_zero():
            prepare_output_dir(args)
        maybe_barrier()

        xielu_status = detect_xielu_cuda()
        if args.require_xielu and not xielu_status["xielu_cuda_available"]:
            raise SystemExit(
                "xIELU CUDA kernel is required for this run but is not available. "
                f"Details: {xielu_status['xielu_cuda_message']}"
            )
        rank_zero_print(xielu_status["xielu_cuda_message"])

        candidate_tokens = load_candidate_tokens(args.token_file)

        base_tokenizer = load_repo_tokenizer(args.base_tokenizer, trust_remote_code=args.trust_remote_code)
        reference_extended_tokenizer = load_repo_tokenizer(
            args.extended_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

        trainable_tokens_all, skipped_tokens, source_map_all = filter_trainable_tokens(
            base_tokenizer,
            reference_extended_tokenizer,
            candidate_tokens,
        )
        if not trainable_tokens_all:
            raise SystemExit("No trainable tokens remained after filtering.")

        trainable_tokens, deferred_tokens, source_map = cap_trainable_tokens(
            trainable_tokens_all,
            source_map_all,
            args.max_trainable_tokens,
        )
        if not trainable_tokens:
            raise SystemExit("No trainable tokens remained after applying --max-trainable-tokens.")

        if deferred_tokens:
            rank_zero_print(
                f"Staged run enabled: training {len(trainable_tokens)} tokens now and deferring {len(deferred_tokens)} for next stage."
            )

        tokenizer, actually_added = build_stage_tokenizer(base_tokenizer, trainable_tokens)
        if actually_added != len(trainable_tokens):
            rank_zero_print(
                "Warning: stage tokenizer added fewer tokens than requested. "
                f"requested={len(trainable_tokens)} added={actually_added}"
            )

        deferred_tokens_path: Optional[Path] = None
        if is_rank_zero():
            deferred_tokens_path = write_deferred_tokens_file(args, deferred_tokens)
            if deferred_tokens_path is not None:
                rank_zero_print(f"Deferred tokens written to {deferred_tokens_path}")
        maybe_barrier()

        model = load_model_for_tokenizer(args, tokenizer)
        model = model.to(device)

        init_stats = initialize_embeddings(args, model, tokenizer, trainable_tokens, source_map)
        rank_zero_print(
            f"Initialized embeddings for {init_stats['initialized']} tokens (skipped={init_stats['skipped']})."
        )

        distill_stats = run_distillation(args, model, tokenizer, trainable_tokens, device)

        maybe_barrier()
        if is_rank_zero():
            rank_zero_print(f"Saving checkpoint to {args.output_dir}")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            eval_stats = {"ran": False}
            if args.run_greek_mmlu_eval:
                rank_zero_print("Starting GreekMMLU evaluation...")
                eval_stats = evaluate_greek_mmlu(args)
                rank_zero_print("Finished GreekMMLU evaluation.")
                if args.fail_below_base and eval_stats.get("ran"):
                    base_acc = eval_stats.get("base_accuracy")
                    trained_acc = eval_stats.get("trained_accuracy")
                    if base_acc is not None and trained_acc is not None and trained_acc < base_acc:
                        raise SystemExit(
                            "Post-training evaluation is below base accuracy and --fail-below-base is enabled. "
                            f"base={base_acc:.6f} trained={trained_acc:.6f}"
                        )

            report = {
                "init_strategy": args.init_strategy,
                "base_model": args.base_model,
                "base_tokenizer": args.base_tokenizer,
                "extended_tokenizer": args.extended_tokenizer,
                "token_file": str(args.token_file),
                "output_dir": str(args.output_dir),
                "args": serialize_args(args),
                "num_candidate_tokens": len(candidate_tokens),
                "num_trainable_tokens": len(trainable_tokens),
                "num_deferred_tokens": len(deferred_tokens),
                "num_skipped_tokens": len(skipped_tokens),
                "deferred_token_file": str(deferred_tokens_path) if deferred_tokens_path is not None else None,
                "xielu": xielu_status,
                "distributed": {
                    "world_size": world_size(),
                },
                "tokenizer_stage": {
                    "base_vocab_size": len(base_tokenizer),
                    "stage_added_tokens": actually_added,
                    "stage_vocab_size": len(tokenizer),
                },
                "initialization": init_stats,
                "distillation": distill_stats,
                "evaluation": eval_stats,
                "samples": {
                    "trainable_tokens": trainable_tokens[:50],
                    "deferred_tokens": deferred_tokens[:50],
                    "skipped_tokens": skipped_tokens[:50],
                },
            }

            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

        # Keep non-zero ranks alive until rank 0 fully finishes save/eval/report.
        maybe_barrier()

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        teardown_distributed()


if __name__ == "__main__":
    main()
