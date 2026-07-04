#!/usr/bin/env python3
"""Advanced token embedding initialization strategies for vocabulary extension.

Supports:
- weighted-mean:  Weight subtokens by inverse token ID (lower ID = more frequent)
- retok:          Use BPE merge heuristic (E = (E[first N-1 mean] + E[last])/2)
- retok-distill:  ReTok + causal language modeling fine-tuning on new embeddings & lm_head
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced token embedding initialization.")
    parser.add_argument("--token-file", type=Path, required=True,
                        help="Text file with one candidate token per line.")
    parser.add_argument("--base-model", required=True,
                        help="Model id or local path for the base LM.")
    parser.add_argument("--extended-tokenizer", required=True,
                        help="Path to the extended tokenizer directory.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where the re-initialized checkpoint is saved.")
    parser.add_argument("--init-strategy", choices=("weighted-mean", "retok", "retok-distill"),
                        default="retok", help="Embedding initialization strategy.")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--distill-steps", type=int, default=100,
                        help="Distillation steps for retok-distill.")
    parser.add_argument("--distill-lr", type=float, default=3e-5,
                        help="Safe Low LR for joint embed and lm_head tuning (default: 3e-5).")
    parser.add_argument("--distill-samples", type=int, default=256,
                        help="Number of tokens to sample contexts for.")
    parser.add_argument("--distill-contexts-per-token", type=int, default=8,
                        help="Number of real-text contexts to collect per token (default: 8).")
    parser.add_argument("--distill-max-seq-length", type=int, default=1024,
                        help="Max token length for distillation batches (default: 1024).")
    parser.add_argument("--distill-warmup-steps", type=int, default=None,
                        help="LR warmup steps for distillation (default: distill_steps // 10).")
    parser.add_argument("--distill-batch-size", type=int, default=16,
                        help="Per-GPU batch size for distillation (default: 16).")
    parser.add_argument("--fineweb2-cache-dir", type=str, default=None,
                        help="Local cache dir for FineWeb2-HQ (default: $SCRATCH/FineWeb2-HQ).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", type=Path,
                        default=Path("artifacts/reports/token_init_advanced.json"))
    parser.add_argument("--distill-worker", type=str, default=None,
                        help=argparse.SUPPRESS)  # internal: worker config file path
    return parser.parse_args()


# ── Helper utilities ────────────────────────────────────────────────────

def _build_grad_mask(new_ids_set: set, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Pre-build a boolean mask tensor for gradient masking (much faster than per-step loop)."""
    mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
    for tid in new_ids_set:
        if tid < vocab_size:
            mask[tid] = True
    return mask


def _resolve_fineweb2_cache_dir(args: argparse.Namespace) -> str:
    """Resolve FineWeb2-HQ cache directory, preferring local persistent storage."""
    import os as _os
    if args.fineweb2_cache_dir:
        return args.fineweb2_cache_dir
    scratch = _os.environ.get("SCRATCH", None)
    if scratch:
        return str(Path(scratch) / "FineWeb2-HQ")
    # Fallback: default HuggingFace datasets cache
    return _os.environ.get("HF_DATASETS_CACHE", _os.path.expanduser("~/.cache/huggingface/datasets"))


@torch.no_grad()
def _evaluate_perplexity(model, tokenizer, eval_texts: List[str], max_seq_length: int, device: torch.device) -> float:
    """Compute perplexity on held-out Greek texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in eval_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.numel() < 2:
            continue
        outputs = model(input_ids=input_ids, labels=input_ids)
        total_loss += outputs.loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
    model.train()
    if total_tokens == 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


# ── Shared distillation core ────────────────────────────────────────────

def _distill_core(
    *,
    model,
    tokenizer,
    texts: List[str],
    new_ids_set: set,
    device: torch.device,
    distill_steps: int,
    distill_lr: float,
    distill_warmup_steps: int,
    distill_batch_size: int,
    max_seq_length: int,
    output_dir: Path,
    ckpt_path: Path,
    start_step: int,
    label: str = "",
    grad_mask: Optional[torch.Tensor] = None,
    new_ids_tensor: Optional[torch.Tensor] = None,
) -> int:
    """Shared distillation training loop used by single-GPU and multi-GPU workers."""
    import random as _random, time as _time

    w_in = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    w_out = out_layer.weight if out_layer is not None else None
    tied = bool(w_out is not None and w_out.data_ptr() == w_in.data_ptr())

    params = [w_in]
    if w_out is not None and not tied:
        params.append(w_out)

    opt = torch.optim.AdamW(params, lr=distill_lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    warmup = distill_warmup_steps if distill_warmup_steps > 0 else 0
    if warmup > 0:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=distill_steps)
    else:
        sched = CosineAnnealingLR(opt, T_max=distill_steps)

    if grad_mask is None and device is not None:
        grad_mask = _build_grad_mask(new_ids_set, w_in.shape[0], device)
    if new_ids_tensor is None and device is not None:
        new_ids_tensor = torch.tensor(sorted(new_ids_set), device=device, dtype=torch.long)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _start_time = _time.time()
    prefix = f"[{label}] " if label else ""

    for step in range(start_step, distill_steps):
        if len(texts) < distill_batch_size:
            indices = list(range(len(texts)))
        else:
            indices = _random.sample(range(len(texts)), distill_batch_size)
        batch_texts = [texts[i] for i in indices]
        if not batch_texts:
            continue

        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Targeted loss: only positions where new tokens appear
        new_token_mask = torch.isin(shift_labels, new_ids_tensor)
        if new_token_mask.any():
            loss = F.cross_entropy(shift_logits[new_token_mask], shift_labels[new_token_mask])
        else:
            continue

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if w_in.grad is not None and grad_mask is not None:
                w_in.grad *= grad_mask.unsqueeze(1)
            if w_out is not None and not tied and w_out.grad is not None and grad_mask is not None:
                w_out.grad *= grad_mask.unsqueeze(1)

            opt.step()

        sched.step()

        if step > start_step and step % 50 == 0:
            save_dict = {
                "step": step,
                "w_in": w_in.detach().cpu(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
            }
            if w_out is not None and not tied:
                save_dict["w_out"] = w_out.detach().cpu()
            torch.save(save_dict, ckpt_path)

        if step % 10 == 0 or step == distill_steps - 1:
            elapsed = _time.time() - _start_time
            lr_val = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else distill_lr
            print(f"  {prefix}step {step}/{distill_steps}  loss={loss.item():.4f}  lr={lr_val:.2e}  elapsed={elapsed:.0f}s", flush=True)

    return start_step


# ── Aho-Corasick single-pass context collection ──────────────────────────

def _collect_contexts_ahocorasick(
    args: argparse.Namespace,
    tokenizer,
    all_tokens: List[str],
    cache_dir: Path,
) -> Tuple[List[dict], int]:
    """Single-pass context collection using Aho-Corasick automaton with local dataset cache."""
    import time as _time, os as _os

    fineweb2_cache = _resolve_fineweb2_cache_dir(args)
    print(f"FineWeb2-HQ cache dir: {fineweb2_cache}")
    _os.makedirs(fineweb2_cache, exist_ok=True)

    from datasets import load_dataset as _hf
    ds = _hf("epfml/FineWeb2-HQ", "ell_Grek", split="train", streaming=True, cache_dir=fineweb2_cache)

    # Build token lookup: clean form -> (token, new_id)
    token_info: Dict[str, tuple] = {}  # clean_search -> (token, new_id)
    clean_to_token: Dict[str, str] = {}
    for token in all_tokens:
        new_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(new_id, int):
            continue
        clean = token.replace(" ", "").replace("Ġ", "").strip()
        if clean and clean not in token_info:
            token_info[clean] = (token, new_id)
            clean_to_token[clean] = token

    if not token_info:
        print("No valid tokens to search for.")
        return [], 0

    # Build Aho-Corasick automaton
    try:
        import ahocorasick
        automaton = ahocorasick.Automaton()
        for clean, (_, _) in token_info.items():
            automaton.add_word(clean, clean)
        automaton.make_automaton()
        print(f"Built Aho-Corasick automaton with {len(token_info)} patterns.")
    except ImportError:
        print("WARNING: pyahocorasick not available, falling back to substring search (slower).")
        automaton = None

    CONTEXTS_PER_TOKEN = args.distill_contexts_per_token
    contexts_needed = {clean: CONTEXTS_PER_TOKEN for clean in token_info}
    global_map: List[dict] = []
    cache_idx = 0

    # Resume from streaming checkpoint
    stream_ckpt_path = cache_dir / "stream_checkpoint.pt"
    tokens_done = set()
    if stream_ckpt_path.exists():
        print(f"Resuming context collection from {stream_ckpt_path}...")
        ckpt = torch.load(stream_ckpt_path, map_location="cpu")
        cache_idx = ckpt.get("cache_idx", 0)
        global_map = ckpt.get("global_map", [])
        tokens_done = set(ckpt.get("tokens_done", []))
        for clean in tokens_done:
            contexts_needed.pop(clean, None)
        print(f"  Resumed at cache_idx={cache_idx}, {len(tokens_done)} tokens already satisfied, {len(contexts_needed)} remaining.")

    if not contexts_needed:
        print("All tokens already have sufficient contexts.")
        return global_map, cache_idx

    t0 = _time.time()
    found_total = 0
    save_interval = 500

    print(f"Streaming FineWeb2-HQ (ell_Grek) — searching for {len(contexts_needed)} tokens "
          f"({CONTEXTS_PER_TOKEN} contexts each, using local cache)...")

    for example in ds:
        text = example.get("text", "")
        if not text:
            continue

        # Find all candidate matches in this document
        if automaton is not None:
            matches = [(end_idx, clean) for end_idx, clean in automaton.iter(text)]
        else:
            # Fallback: substring search (slower but works without ahocorasick)
            matches = []
            for clean in contexts_needed:
                pos = text.find(clean)
                while pos != -1:
                    matches.append((pos + len(clean), clean))
                    pos = text.find(clean, pos + 1)

        if not matches:
            continue

        # Extract sentences around matches
        sentences_raw = [s.strip() for s in text.replace("!", ".").replace(";", ".").split(".")
                         if 30 < len(s.strip()) < 1000]
        if not sentences_raw:
            continue

        for sentence in sentences_raw:
            for _, clean in matches:
                if clean not in contexts_needed:
                    continue
                if clean not in sentence:
                    continue

                _, new_id = token_info[clean]
                tokenized_ids = tokenizer.encode(sentence, add_special_tokens=False)
                if new_id not in tokenized_ids:
                    continue

                # Valid context found
                token = clean_to_token[clean]
                cache_file = cache_dir / f"t_{cache_idx:06d}.pt"
                if not cache_file.exists():
                    torch.save({"text": sentence}, cache_file)
                global_map.append({"token": token, "global_idx": len(global_map)})
                cache_idx += 1
                found_total += 1

                contexts_needed[clean] -= 1
                if contexts_needed[clean] <= 0:
                    del contexts_needed[clean]
                    tokens_done.add(clean)

                break  # one context per sentence per clean token

        # Periodic checkpoint
        if found_total > 0 and found_total % save_interval == 0:
            torch.save({
                "cache_idx": cache_idx,
                "global_map": global_map,
                "tokens_done": list(tokens_done),
            }, stream_ckpt_path)
            elapsed = _time.time() - t0
            print(f"  [{elapsed:.0f}s] Collected {found_total} contexts, {len(contexts_needed)} tokens remaining.",
                  flush=True)

        # Early exit when all tokens satisfied
        if not contexts_needed:
            break

    elapsed = _time.time() - t0
    print(f"Context collection complete: {found_total} contexts in {elapsed:.0f}s "
          f"({len(tokens_done)}/{len(token_info)} tokens satisfied, "
          f"{len(contexts_needed)} tokens without sufficient real contexts).")

    # Final checkpoint save
    torch.save({
        "cache_idx": cache_idx,
        "global_map": global_map,
        "tokens_done": list(tokens_done),
    }, stream_ckpt_path)

    # Clean up iterator
    del ds
    import gc
    gc.collect()

    return global_map, cache_idx


def _load_context_texts(cache_dir: Path, num_expected: int) -> List[str]:
    """Load cached context texts into RAM."""
    texts = []
    for idx in range(num_expected):
        cf = cache_dir / f"t_{idx:06d}.pt"
        if cf.exists():
            texts.append(torch.load(cf, map_location="cpu")["text"])
        else:
            texts.append(None)
    # Filter out None entries
    return [t for t in texts if t is not None]


# ── Single GPU Distillation (Fallback) ──────────────────────────────────

def _distill_single(args, model, tokenizer, all_tokens_expanded, cache_dir):
    """Single GPU CLM Distillation (fallback) — delegates to shared _distill_core."""
    device = next(model.parameters()).device
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    w_in = model.get_input_embeddings().weight
    w_in.requires_grad = True
    out_layer = model.get_output_embeddings()
    w_out = out_layer.weight if out_layer is not None else None
    tied = bool(w_out is not None and w_out.data_ptr() == w_in.data_ptr())
    if w_out is not None and not tied:
        w_out.requires_grad = True

    new_ids_set = set()
    for t in all_tokens_expanded:
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int):
            new_ids_set.add(tid)

    grad_mask = _build_grad_mask(new_ids_set, w_in.shape[0], device)
    new_ids_tensor = torch.tensor(sorted(new_ids_set), device=device, dtype=torch.long)

    warmup_steps = args.distill_warmup_steps if args.distill_warmup_steps is not None else max(1, args.distill_steps // 10)

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ckpt_dir = Path(args.output_dir) / "distill_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "single_gpu_checkpoint.pt"

    # Resume
    start_step = 0
    if ckpt_path.exists():
        print(f"Found checkpoint {ckpt_path}. Resuming single GPU training...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        start_step = ckpt["step"] + 1
        w_in.data.copy_(ckpt["w_in"].to(device))
        if w_out is not None and not tied and "w_out" in ckpt:
            w_out.data.copy_(ckpt["w_out"].to(device))

    print("Loading text contexts into RAM...")
    texts = _load_context_texts(cache_dir, len(all_tokens_expanded))
    print(f"  Loaded {len(texts)} valid contexts.", flush=True)

    _distill_core(
        model=model, tokenizer=tokenizer, texts=texts, new_ids_set=new_ids_set,
        device=device, distill_steps=args.distill_steps, distill_lr=args.distill_lr,
        distill_warmup_steps=warmup_steps, distill_batch_size=args.distill_batch_size,
        max_seq_length=args.distill_max_seq_length, output_dir=args.output_dir,
        ckpt_path=ckpt_path, start_step=start_step, label="single",
        grad_mask=grad_mask, new_ids_tensor=new_ids_tensor,
    )

    if ckpt_path.exists():
        ckpt_path.unlink()


# ── Worker entry point (for multi-GPU) ──────────────────────────────────

def _distill_worker(cfg_file: str):
    """Run native Causal Language Modeling on pre-loaded text slices — delegates to _distill_core."""
    import pickle, random as _random, time as _time
    with open(cfg_file, "rb") as f:
        cfg = pickle.load(f)

    gpu = cfg["gpu"]
    global_indices = cfg.get("global_indices", list(range(len(cfg["tokens"]))))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {gpu}] Starting worker with {len(cfg['tokens'])} text contexts...", flush=True)

    # Load model
    dtype = getattr(torch, cfg["torch_dtype"]) if cfg["torch_dtype"] != "auto" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], trust_remote_code=cfg["trust_remote_code"], dtype=dtype,
    ).to(device)
    tokenizer = load_repo_tokenizer(cfg["extended_tokenizer"], trust_remote_code=cfg["trust_remote_code"])
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    input_emb = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    output_emb = out_layer.weight if out_layer is not None else None
    tied = bool(output_emb is not None and output_emb.data_ptr() == input_emb.data_ptr())
    base_tok = load_repo_tokenizer("artifacts/tokenizers/apertus-base", trust_remote_code=cfg["trust_remote_code"])

    # Initialize new token embeddings via retok heuristic
    unique_tokens = list(dict.fromkeys(cfg["tokens"]))
    with torch.no_grad():
        for token in unique_tokens:
            src_ids = base_tok.encode(token, add_special_tokens=False)
            if not src_ids:
                continue
            new_id = tokenizer.convert_tokens_to_ids(token)
            if not isinstance(new_id, int):
                continue
            if len(src_ids) == 1:
                emb = input_emb[src_ids[0]]
            elif len(src_ids) == 2:
                emb = (input_emb[src_ids[0]] + input_emb[src_ids[1]]) / 2.0
            else:
                emb = (input_emb[src_ids[:-1]].mean(dim=0) + input_emb[src_ids[-1]]) / 2.0
            input_emb[new_id].copy_(emb)
            if output_emb is not None and not tied:
                output_emb[new_id].copy_(emb)

    # Freeze/unfreeze
    for p in model.parameters():
        p.requires_grad = False
    input_emb.requires_grad = True
    if output_emb is not None and not tied:
        output_emb.requires_grad = True

    new_ids_set = set()
    for t in unique_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int):
            new_ids_set.add(tid)

    grad_mask = _build_grad_mask(new_ids_set, input_emb.shape[0], device)
    new_ids_tensor = torch.tensor(sorted(new_ids_set), device=device, dtype=torch.long)

    distill_warmup = cfg.get("distill_warmup_steps", max(1, cfg["distill_steps"] // 10))
    distill_batch_size = cfg.get("distill_batch_size", 16)
    max_seq_length = cfg.get("distill_max_seq_length", 1024)

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Resume
    start_step = 0
    ckpt_path = Path(cfg["output_dir"]) / f"checkpoint_worker_gpu{gpu}.pt"
    if ckpt_path.exists():
        print(f"[GPU {gpu}] Found active checkpoint. Resuming...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        start_step = ckpt["step"] + 1
        input_emb.data.copy_(ckpt["w_in"].to(device))
        if output_emb is not None and not tied and "w_out" in ckpt:
            output_emb.data.copy_(ckpt["w_out"].to(device))

    # Pre-load context texts into RAM
    cache_dir = Path(cfg["cache_dir"])
    print(f"[GPU {gpu}] Pre-loading context strings into RAM...", flush=True)
    worker_texts = []
    for global_idx in global_indices:
        cf = cache_dir / f"t_{global_idx:06d}.pt"
        if cf.exists():
            worker_texts.append(torch.load(cf, map_location="cpu")["text"])
    print(f"[GPU {gpu}] RAM pre-load complete. Total contexts: {len(worker_texts)}", flush=True)

    _distill_core(
        model=model, tokenizer=tokenizer, texts=worker_texts, new_ids_set=new_ids_set,
        device=device, distill_steps=cfg["distill_steps"], distill_lr=cfg["distill_lr"],
        distill_warmup_steps=distill_warmup, distill_batch_size=distill_batch_size,
        max_seq_length=max_seq_length, output_dir=Path(cfg["output_dir"]),
        ckpt_path=ckpt_path, start_step=start_step, label=f"GPU {gpu}",
        grad_mask=grad_mask, new_ids_tensor=new_ids_tensor,
    )

    # Save final chunk weights
    out_file = Path(cfg["output_dir"]) / f"embeddings_gpu{gpu}.pt"
    emb_data = {"input_embs": {}, "output_embs": {}}
    for token in unique_tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int):
            emb_data["input_embs"][str(tid)] = input_emb[tid].detach().cpu()
            if output_emb is not None and not tied:
                emb_data["output_embs"][str(tid)] = output_emb[tid].detach().cpu()
    torch.save(emb_data, out_file)
    print(f"[GPU {gpu}] Embeddings saved.", flush=True)
    if ckpt_path.exists():
        ckpt_path.unlink()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto": return "auto"
    return getattr(torch, dtype_name)


def load_bpe_merges(tokenizer_path: str) -> List[Tuple[str, str]]:
    tok_json = Path(tokenizer_path) / "tokenizer.json"
    with open(tok_json) as f:
        data = json.load(f)
    merges = data.get("model", {}).get("merges", [])
    return [(str(m[0]), str(m[1])) for m in merges]


def compute_weighted_mean(embeddings: torch.Tensor, source_ids: List[int]) -> torch.Tensor:
    if len(source_ids) == 1: return embeddings[source_ids[0]]
    weights = torch.tensor([1.0 / (tid + 1) for tid in source_ids], device=embeddings.device, dtype=embeddings.dtype)
    weights = weights / weights.sum()
    return (embeddings[source_ids] * weights.unsqueeze(1)).sum(dim=0)


def compute_retok_mean(embeddings: torch.Tensor, source_ids: List[int]) -> torch.Tensor:
    if len(source_ids) == 1: return embeddings[source_ids[0]]
    if len(source_ids) == 2: return (embeddings[source_ids[0]] + embeddings[source_ids[1]]) / 2.0
    emb_left = embeddings[source_ids[:-1]].mean(dim=0)
    emb_right = embeddings[source_ids[-1]]
    return (emb_left + emb_right) / 2.0


# ── Main initialization ─────────────────────────────────────────────────

def initialize_embeddings_advanced(
    args: argparse.Namespace,
    tokenizer,
    tokens_to_add: List[str],
    source_ids_map: Dict[str, List[int]],
) -> Dict[str, Any]:
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    print(f"Loading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code, dtype=torch_dtype,
    )

    orig_vocab = model.get_input_embeddings().num_embeddings
    target_vocab = len(tokenizer)
    print(f"Resizing embeddings  {orig_vocab} → {target_vocab}")
    model.resize_token_embeddings(target_vocab, mean_resizing=False)

    w_in = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    w_out = out_layer.weight if out_layer is not None else None
    tied = bool(w_out is not None and w_out.data_ptr() == w_in.data_ptr())

    if args.init_strategy in ("retok", "retok-distill"):
        merges = load_bpe_merges(str(args.extended_tokenizer))
        print(f"Loaded {len(merges)} BPE merges")

    stats: Dict[str, Any] = {"initialized": 0, "skipped": 0, "strategy": args.init_strategy}

    with torch.no_grad():
        for token in tokens_to_add:
            src = source_ids_map.get(token, [])
            if not src:
                stats["skipped"] += 1
                continue
            new_id = tokenizer.convert_tokens_to_ids(token)

            if args.init_strategy == "weighted-mean":
                emb = compute_weighted_mean(w_in, src)
            elif args.init_strategy in ("retok", "retok-distill"):
                emb = compute_retok_mean(w_in, src)
            else:
                emb = w_in[src].mean(dim=0)

            w_in[new_id].copy_(emb)
            if w_out is not None and not tied:
                w_out[new_id].copy_(emb)
            stats["initialized"] += 1

    if args.init_strategy == "retok-distill":
        print(f"\n=== CLM Vocabulary Extension Distillation ===")

        # ── Pre-distillation perplexity evaluation ──────────────────────
        device = next(model.parameters()).device
        eval_texts = [
            "Η ελληνική γλώσσα χρειάζεται καλύτερη κάλυψη στο tokenizer.",
            "Τα σχολικά βιβλία περιέχουν όρους που θέλουμε να γίνονται tokenize πιο αποδοτικά.",
            "Η εκπαίδευση στη σύγχρονη Ελλάδα αντιμετωπίζει προκλήσεις αλλά και ευκαιρίες.",
            "Η τεχνολογία και η καινοτομία είναι σημαντικοί πυλώνες ανάπτυξης.",
        ]
        pre_perplexity = _evaluate_perplexity(model, tokenizer, eval_texts,
                                               args.distill_max_seq_length, device)
        stats["pre_distill_perplexity"] = round(pre_perplexity, 2)
        print(f"  Pre-distill perplexity (Greek eval): {pre_perplexity:.2f}")

        run_token_distillation(args, model, tokenizer, tokens_to_add)
        stats["distillation_completed"] = True

        # ── Post-distillation perplexity evaluation ────────────────────
        post_perplexity = _evaluate_perplexity(model, tokenizer, eval_texts,
                                                args.distill_max_seq_length, device)
        stats["post_distill_perplexity"] = round(post_perplexity, 2)
        delta = post_perplexity - pre_perplexity
        stats["perplexity_delta"] = round(delta, 2)
        print(f"  Post-distill perplexity (Greek eval): {post_perplexity:.2f}  (Δ={delta:+.2f})")

    print(f"Saving finalized checkpoint → {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return stats


# ── Token Distillation ──────────────────────────────────────────────────

def run_token_distillation(
    args: argparse.Namespace,
    model,
    tokenizer,
    tokens_to_add: List[str],
) -> Optional[float]:
    """Token Distillation v17: Aho-Corasick streaming + local dataset cache + targeted CLM."""
    import time as _time, random as _random, gc, os as _os, subprocess, pickle

    num_gpus = torch.cuda.device_count()

    cache_dir = Path(args.output_dir) / "teacher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_tokens = tokens_to_add[:args.distill_samples]

    # ── Single-pass Aho-Corasick context collection ─────────────────────
    global_map, cache_idx = _collect_contexts_ahocorasick(
        args, tokenizer, all_tokens, cache_dir)

    if not global_map:
        print("WARNING: No contexts collected. Skipping distillation, keeping retok init only.")
        return None

    distill_warmup = args.distill_warmup_steps if args.distill_warmup_steps is not None else max(1, args.distill_steps // 10)

    # ── Single GPU or Multi GPU path ────────────────────────────────────
    use_parallel = num_gpus >= 2 and cache_idx > 100
    if not use_parallel:
        all_tokens_expanded = [meta["token"] for meta in global_map]
        return _distill_single(args, model, tokenizer, all_tokens_expanded, cache_dir)

    # ---- MULTI GPU DISTRIBUTED LAUNCH ----
    print(f"Spawning {num_gpus} isolated GPU processes.")
    token_chunks: List[List[str]] = [[] for _ in range(num_gpus)]
    global_indices_chunks: List[List[int]] = [[] for _ in range(num_gpus)]

    for c_idx in range(cache_idx):
        gpu_idx = c_idx % num_gpus
        meta = global_map[c_idx]
        token_chunks[gpu_idx].append(meta["token"])
        global_indices_chunks[gpu_idx].append(c_idx)

    worker_procs = []
    worker_configs = []
    tmp_dir = Path(args.output_dir) / "worker_tmp"
    tmp_dir.mkdir(exist_ok=True)

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i in range(num_gpus):
        if not token_chunks[i]:
            continue

        worker_cfg = {
            "gpu": i,
            "tokens": token_chunks[i],
            "global_indices": global_indices_chunks[i],
            "base_model": args.base_model,
            "extended_tokenizer": str(args.extended_tokenizer),
            "output_dir": str(args.output_dir),
            "cache_dir": str(cache_dir),
            "torch_dtype": args.torch_dtype,
            "trust_remote_code": args.trust_remote_code,
            "distill_steps": args.distill_steps,
            "distill_lr": args.distill_lr,
            "distill_warmup_steps": distill_warmup,
            "distill_batch_size": args.distill_batch_size,
            "distill_max_seq_length": args.distill_max_seq_length,
        }

        cfg_file = tmp_dir / f"worker_cfg_{i}.pkl"
        with open(cfg_file, "wb") as f:
            pickle.dump(worker_cfg, f)
        worker_configs.append(cfg_file)

        script_path = Path(__file__).resolve()
        worker_env = _os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(i)
        worker_env["PYTHONUNBUFFERED"] = "1"

        cmd = [sys.executable, str(script_path), "--distill-worker", str(cfg_file)]
        log_file = Path(args.output_dir) / f"worker_{i}.log"

        with open(log_file, "w") as log:
            proc = subprocess.Popen(cmd, env=worker_env, stdout=log, stderr=subprocess.STDOUT)
        worker_procs.append(proc)

    for proc in worker_procs:
        proc.wait()

    print("Merging generated multi-GPU structural weights...")
    w_in = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    w_out = out_layer.weight if out_layer is not None else None
    tied = bool(w_out is not None and w_out.data_ptr() == w_in.data_ptr())

    for i in range(num_gpus):
        emb_file = Path(args.output_dir) / f"embeddings_gpu{i}.pt"
        if not emb_file.exists():
            continue
        data = torch.load(emb_file, map_location="cpu")
        for tid_str, emb in data.get("input_embs", {}).items():
            w_in.data[int(tid_str)].copy_(emb)
        if w_out is not None and not tied:
            for tid_str, emb in data.get("output_embs", {}).items():
                w_out.data[int(tid_str)].copy_(emb)
        emb_file.unlink()

    for cfg_file in worker_configs:
        try:
            cfg_file.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print("Embeddings consolidated.")
    gc.collect()
    return 0.0


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    if "--distill-worker" in sys.argv:
        idx = sys.argv.index("--distill-worker")
        if idx + 1 < len(sys.argv):
            _distill_worker(sys.argv[idx + 1])
            return

    args = parse_args()

    if not args.token_file.exists():
        raise SystemExit(f"Token file not found: {args.token_file}")
        
    if args.output_dir.exists():
        if args.overwrite:
            print(f"Wiping directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            print(f"Output directory exists. Entering RESUME mode...")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading extended tokenizer {args.extended_tokenizer}")
    tok = load_repo_tokenizer(str(args.extended_tokenizer), trust_remote_code=args.trust_remote_code)

    print(f"Loading candidate tokens from {args.token_file}")
    raw = [line for line in args.token_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = list(dict.fromkeys(raw))

    base_tok_path = "artifacts/tokenizers/apertus-base"
    base_tok = load_repo_tokenizer(base_tok_path, trust_remote_code=args.trust_remote_code)

    source_map: Dict[str, List[int]] = {}
    for t in unique:
        source_map[t] = base_tok.encode(t, add_special_tokens=False)

    stats = initialize_embeddings_advanced(args, tok, unique, source_map)

    report = {
        "init_strategy": args.init_strategy,
        "token_file": str(args.token_file),
        "base_model": args.base_model,
        "extended_tokenizer": str(args.extended_tokenizer),
        "output_dir": str(args.output_dir),
        "num_tokens": len(unique),
        "stats": stats,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Execution complete!")


if __name__ == "__main__":
    main()
