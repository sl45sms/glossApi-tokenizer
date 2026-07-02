#!/usr/bin/env python3
"""Advanced token embedding initialization strategies for vocabulary extension.

Supports:
- weighted-mean:  Weight subtokens by inverse token ID (lower ID = more frequent)
- retok:          Use BPE merge heuristic (E = (E[first N-1 mean] + E[last])/2)
- retok-distill:  ReTok + gradient descent on new embeddings to preserve attention
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
    parser.add_argument("--distill-lr", type=float, default=5e-5,
                        help="Safe Lower LR for stable embedding tuning (default: 5e-5).")
    parser.add_argument("--distill-samples", type=int, default=256,
                        help="Number of text samples for distillation.")
    parser.add_argument("--distill-layer", type=int, default=None,
                        help="Attention layer to match (default: middle).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", type=Path,
                        default=Path("artifacts/reports/token_init_advanced.json"))
    parser.add_argument("--distill-worker", type=str, default=None,
                        help=argparse.SUPPRESS)  # internal: worker config file path
    return parser.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────

def _find_sentence_for_token(clean: str, args) -> str:
    """Fallback template generator if text cache missing."""
    return f"Η εξειδικευμένη λέξη {clean} χρησιμοποιείται σε αυτό το πλαίσιο."


# ── Single GPU Distillation (Fallback) ──────────────────────────────────

def _distill_single(args, model, tokenizer, all_tokens_expanded, cache_dir, distill_layers, layer_weights):
    """Single GPU distillation (fallback) with Early MSE and Safe Low LR."""
    import time as _time, random as _random
    device = next(model.parameters()).device
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    w_in = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    w_out = out_layer.weight if out_layer is not None else None
    tied = bool(w_out is not None and w_out.data_ptr() == w_in.data_ptr())
    w_in.requires_grad = True
    if w_out is not None and not tied:
        w_out.requires_grad = True

    new_ids_set = set()
    for t in all_tokens_expanded:
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int):
            new_ids_set.add(tid)

    opt = torch.optim.AdamW([w_in] if tied else [w_in, w_out], lr=args.distill_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.distill_steps)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ckpt_dir = Path(args.output_dir) / "distill_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    
    ckpt_path = ckpt_dir / "single_gpu_checkpoint.pt"
    if ckpt_path.exists():
        print(f"Found checkpoint {ckpt_path}. Resuming single GPU training...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        start_step = ckpt["step"] + 1
        w_in.data.copy_(ckpt["w_in"].to(device))
        if w_out is not None and not tied:
            w_out.data.copy_(ckpt["w_out"].to(device))
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])

    print("Loading teacher cache into RAM...")
    all_targets = []
    for idx in range(len(all_tokens_expanded)):
        cf = cache_dir / f"t_{idx:06d}.pt"
        if cf.exists():
            all_targets.append(torch.load(cf, map_location="cpu"))
        else:
            all_targets.append(None)
    print(f"Loaded {sum(1 for t in all_targets if t is not None)}/{len(all_targets)} targets into RAM")

    _start_time = _time.time()
    for step in range(start_step, args.distill_steps):
        indices = _random.sample(range(len(all_tokens_expanded)), min(64, len(all_tokens_expanded)))
        total_loss = 0.0
        for idx in indices:
            targets = all_targets[idx]
            if targets is None: continue
            token_str = all_tokens_expanded[idx]
            
            text = targets["text"]
            new_ids_t = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
            new_tok_id = tokenizer.convert_tokens_to_ids(token_str)
            if not isinstance(new_tok_id, int): continue
            pos_list = (new_ids_t[0] == new_tok_id).nonzero(as_tuple=True)[0]
            if len(pos_list) == 0: continue
            new_pos = pos_list[0].item()

            model.zero_grad(set_to_none=True)
            s_out = model(input_ids=new_ids_t, output_hidden_states=True)
            
            loss = torch.tensor(0.0, device=device)
            for layer, w in zip(distill_layers, layer_weights):
                t_tgt = targets[str(layer)].to(device)
                s_pred = s_out.hidden_states[layer + 1][0, new_pos, :]
                loss = loss + w * F.mse_loss(s_pred, t_tgt)

            if not torch.isnan(loss) and loss.item() != 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([w_in], max_norm=1.0)
                if w_in.grad is not None:
                    m = torch.zeros(w_in.grad.shape[0], 1, device=device)
                    for tid in new_ids_set:
                        if isinstance(tid, int) and 0 <= tid < m.shape[0]:
                            m[tid] = 1.0
                    w_in.grad *= m
                opt.step()
                total_loss += loss.item()

        sched.step()
        
        if step > start_step and step % 50 == 0:
            torch.save({
                "step": step,
                "w_in": w_in.detach().cpu(),
                "w_out": w_out.detach().cpu() if w_out is not None and not tied else None,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict()
            }, ckpt_path)

        if step % 10 == 0 or step < 3 or step == args.distill_steps - 1:
            avg = total_loss / len(indices) if indices else 0
            elapsed = _time.time() - _start_time
            eta = (elapsed/(step - start_step + 1))*(args.distill_steps-step-1) if step > start_step else 0
            print(f"  step {step}/{args.distill_steps}  loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
                  
    if ckpt_path.exists():
        ckpt_path.unlink()


# ── Worker entry point (for multi-GPU) ──────────────────────────────────

def _distill_worker(cfg_file: str):
    """Run early-layer MSE distillation with 100% RAM Pre-loaded Cache (No Disk Bottlenecks)."""
    import pickle
    with open(cfg_file, "rb") as f:
        cfg = pickle.load(f)

    gpu = cfg["gpu"]
    global_indices = cfg.get("global_indices", list(range(len(cfg["tokens"]))))
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {gpu}] Starting worker with {len(cfg['tokens'])} samples, {cfg['distill_steps']} steps", flush=True)

    # Load model
    from transformers import AutoModelForCausalLM
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

    unique_tokens = list(dict.fromkeys(cfg["tokens"]))
    with torch.no_grad():
        for token in unique_tokens:
            src_ids = base_tok.encode(token, add_special_tokens=False)
            if not src_ids: continue
            new_id = tokenizer.convert_tokens_to_ids(token)
            if not isinstance(new_id, int): continue
            if len(src_ids) == 1:
                emb = input_emb[src_ids[0]]
            elif len(src_ids) == 2:
                emb = (input_emb[src_ids[0]] + input_emb[src_ids[1]]) / 2.0
            else:
                emb = (input_emb[src_ids[:-1]].mean(dim=0) + input_emb[src_ids[-1]]) / 2.0
            input_emb[new_id].copy_(emb)
            if output_emb is not None and not tied:
                output_emb[new_id].copy_(emb)

    # Train setup
    for p in model.parameters(): p.requires_grad = False
    input_emb.requires_grad = True
    if output_emb is not None and not tied: output_emb.requires_grad = True

    new_ids_set = set()
    for t in unique_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int): new_ids_set.add(tid)

    params = [input_emb]
    if output_emb is not None and not tied: params.append(output_emb)
    opt = torch.optim.AdamW(params, lr=cfg["distill_lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["distill_steps"])
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()

    # Load worker checkpoint if exists
    start_step = 0
    ckpt_path = Path(cfg["output_dir"]) / f"checkpoint_worker_gpu{gpu}.pt"
    if ckpt_path.exists():
        print(f"[GPU {gpu}] Found active checkpoint. Resuming training...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        start_step = ckpt["step"] + 1
        input_emb.data.copy_(ckpt["input_emb"].to(device))
        if output_emb is not None and not tied:
            output_emb.data.copy_(ckpt["output_emb"].to(device))
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])

    # ── ΔΙΟΡΘΩΣΗ: Pre-loading των targets στη RAM του Worker για εξαφάνιση του I/O Bottleneck ──
    import random as _random, time as _time
    cache_dir = Path(cfg["cache_dir"])
    print(f"[GPU {gpu}] Pre-loading cache targets into node RAM...", flush=True)
    worker_targets = {}
    for local_idx, global_idx in enumerate(global_indices):
        cf = cache_dir / f"t_{global_idx:06d}.pt"
        if cf.exists():
            worker_targets[local_idx] = torch.load(cf, map_location="cpu")
        else:
            worker_targets[local_idx] = None
    print(f"[GPU {gpu}] Memory loading done. Active samples in RAM: {sum(1 for x in worker_targets.values() if x is not None)}", flush=True)

    distill_layers = cfg["layers"]
    layer_weights = cfg["layer_weights"]
    batch_size = 64

    for step in range(start_step, cfg["distill_steps"]):
        local_indices = _random.sample(range(len(cfg["tokens"])), min(batch_size, len(cfg["tokens"])))
        total_loss = 0.0
        skipped = 0
        for local_idx in local_indices:
            # Διαβάζουμε απευθείας από τη RAM (In-Memory Processing)
            targets = worker_targets.get(local_idx)
            if targets is None: 
                skipped += 1
                continue
            
            token_str = cfg["tokens"][local_idx]
            text = targets["text"]
            
            new_ids_t = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
            new_tok_id = tokenizer.convert_tokens_to_ids(token_str)
            if not isinstance(new_tok_id, int): 
                skipped += 1
                continue
                
            pos_list = (new_ids_t[0] == new_tok_id).nonzero(as_tuple=True)[0]
            if len(pos_list) == 0: 
                skipped += 1
                continue
            new_pos = pos_list[0].item()

            model.zero_grad(set_to_none=True)
            s_out = model(input_ids=new_ids_t, output_hidden_states=True)
            
            loss = torch.tensor(0.0, device=device)
            for layer, w in zip(distill_layers, layer_weights):
                t_tgt = targets[str(layer)].to(device)
                s_pred = s_out.hidden_states[layer + 1][0, new_pos, :]
                loss = loss + w * F.mse_loss(s_pred, t_tgt)

            if step == start_step and local_idx < 3:
                log_file = Path(cfg["output_dir"]) / f"debug_gpu{gpu}.log"
                with open(log_file, "a") as lf:
                    lf.write(f"step={step} local_idx={local_idx} global_idx={global_indices[local_idx]} loss={loss.item():.4f}\n")

            if not torch.isnan(loss) and loss.item() != 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([input_emb], max_norm=1.0)
                if input_emb.grad is not None:
                    m = torch.zeros(input_emb.grad.shape[0], 1, device=device)
                    for tid in new_ids_set:
                        if isinstance(tid, int) and 0 <= tid < m.shape[0]:
                            m[tid] = 1.0
                    input_emb.grad *= m
                opt.step()
                total_loss += loss.item()
            else:
                skipped += 1

        sched.step()
        
        if step > start_step and step % 50 == 0:
            torch.save({
                "step": step,
                "input_emb": input_emb.detach().cpu(),
                "output_emb": output_emb.detach().cpu() if output_emb is not None and not tied else None,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict()
            }, ckpt_path)

        if step % 10 == 0 or step < 3:
            avg = total_loss / (len(local_indices) - skipped) if (len(local_indices) - skipped) > 0 else 0
            print(f"[GPU {gpu}] step {step}/{cfg['distill_steps']}  loss={avg:.4f}  skipped={skipped}", flush=True)

    # Save final embeddings
    out_file = Path(cfg["output_dir"]) / f"embeddings_gpu{gpu}.pt"
    emb_data = {"input_embs": {}, "output_embs": {}}
    for token in unique_tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int):
            emb_data["input_embs"][str(tid)] = input_emb[tid].detach().cpu()
            if output_emb is not None and not tied:
                emb_data["output_embs"][str(tid)] = output_emb[tid].detach().cpu()
    torch.save(emb_data, out_file)
    print(f"[GPU {gpu}] Saved embeddings to {out_file}", flush=True)
    
    if ckpt_path.exists():
        ckpt_path.unlink()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def load_bpe_merges(tokenizer_path: str) -> List[Tuple[str, str]]:
    tok_json = Path(tokenizer_path) / "tokenizer.json"
    with open(tok_json) as f:
        data = json.load(f)
    merges = data.get("model", {}).get("merges", [])
    return [(str(m[0]), str(m[1])) for m in merges]


def compute_weighted_mean(embeddings: torch.Tensor, source_ids: List[int]) -> torch.Tensor:
    if len(source_ids) == 1:
        return embeddings[source_ids[0]]
    weights = torch.tensor([1.0 / (tid + 1) for tid in source_ids],
                           device=embeddings.device, dtype=embeddings.dtype)
    weights = weights / weights.sum()
    return (embeddings[source_ids] * weights.unsqueeze(1)).sum(dim=0)


def compute_retok_mean(embeddings: torch.Tensor, source_ids: List[int]) -> torch.Tensor:
    if len(source_ids) == 1:
        return embeddings[source_ids[0]]
    if len(source_ids) == 2:
        return (embeddings[source_ids[0]] + embeddings[source_ids[1]]) / 2.0
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
    print(f"Loading base model {args.base_model}  dtype={torch_dtype} ...")
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

    print(f"Initialized {stats['initialized']}  skipped {stats['skipped']}")

    if args.init_strategy == "retok-distill":
        print(f"\n=== Token Distillation  steps={args.distill_steps}  lr={args.distill_lr} ===")
        distillation_loss = run_token_distillation(
            args, model, tokenizer, tokens_to_add, source_ids_map,
        )
        stats["distillation_final_loss"] = distillation_loss

    print(f"Saving model → {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return stats


# ── Token Distillation ──────────────────────────────────────────────────

def run_token_distillation(
    args: argparse.Namespace,
    model,
    tokenizer,
    tokens_to_add: List[str],
    source_ids_map: Dict[str, List[int]],
) -> Optional[float]:
    """Token Distillation v14: FineWeb2-HQ Streaming + Integer Subsequence ID Matching + Multi-GPU Checkpointing."""
    import time as _time, random as _random, gc, os as _os, subprocess, pickle
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = next(model.parameters()).dtype
    num_gpus = torch.cuda.device_count()

    base_tok = load_repo_tokenizer("artifacts/tokenizers/apertus-base",
                                    trust_remote_code=args.trust_remote_code)
    
    #distill_layers = [1, 2]
    #layer_weights = [0.5, 0.5]
    # το MSE στα layers 1-2 πιάνει μόνο positional/lexical features — όχι σημασιολογία. Τα layers 4-16 περιέχουν semantic content που χρειάζεται το MMLU.
    distill_layers = [4, 8, 16]
    layer_weights = [0.2, 0.5, 0.3]

    # ── ΔΙΟΡΘΩΣΗ: Επιστρέφει το πρώτο match ακόμα κι αν η λέξη εμφανίζεται 2+ φορές στην ίδια πρόταση
    def _find_id_subsequence(main_list: List[int], sub_list: List[int]) -> Optional[Tuple[int, int]]:
        matches = []
        n, m = len(main_list), len(sub_list)
        for i in range(n - m + 1):
            if main_list[i : i + m] == sub_list:
                matches.append((i, i + m))
        return matches[0] if matches else None

    # ---- Pre-compute teacher cache ----
    cache_dir = Path(args.output_dir) / "teacher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_tokens = tokens_to_add[:args.distill_samples]
    
    print(f"Loading FineWeb2-HQ (ell_Grek) stream for real context matching...")
    from datasets import load_dataset as _hf
    ds = _hf("epfml/FineWeb2-HQ", "ell_Grek", split="train", streaming=True)
    
    print(f"Pre-computing teacher cache for {len(all_tokens)} tokens using early layer hidden states...")
    
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code, dtype=dtype,
    ).to(device)
    teacher.eval()
    
    CONTEXTS_PER_TOKEN = 3
    cache_idx = 0
    global_map = []
    t0 = _time.time()
    ds_iterator = iter(ds)
    
    for idx, token in enumerate(all_tokens):
        clean = token.strip()
        token_ids = base_tok.encode(clean, add_special_tokens=False)
        if not token_ids:
            continue
            
        found_contexts = 0
        attempts = 0
        
        while found_contexts < CONTEXTS_PER_TOKEN and attempts < 1500:
            attempts += 1
            try:
                example = next(ds_iterator)
                text_content = example.get("text", "")
            except StopIteration:
                ds_iterator = iter(ds)
                continue
                
            if clean not in text_content:
                continue
                
            sentences = [s.strip() for s in text_content.replace("!", ".").replace(";", ".").split(".") if len(s.strip()) > 30]
            for sentence in sentences:
                if clean not in sentence:
                    continue
                    
                old_ids_tensor = base_tok.encode(sentence, add_special_tokens=True, return_tensors="pt").to(device)
                old_ids_list = old_ids_tensor[0].tolist()
                
                bounds = _find_id_subsequence(old_ids_list, token_ids)
                if bounds is None:
                    continue
                    
                start_pos, end_pos = bounds
                cache_file = cache_dir / f"t_{cache_idx:06d}.pt"
                global_map.append({"token": token, "global_idx": idx, "template_idx": found_contexts})
                
                if cache_file.exists():
                    found_contexts += 1
                    cache_idx += 1
                    if found_contexts >= CONTEXTS_PER_TOKEN:
                        break
                    continue
                
                with torch.no_grad():
                    t_out = teacher(old_ids_tensor, output_hidden_states=True)
                    targets = {"text": sentence}
                    
                    for layer in distill_layers:
                        hidden_slice = t_out.hidden_states[layer + 1][0, start_pos:end_pos, :]
                        targets[str(layer)] = hidden_slice.mean(dim=0).cpu()
                    
                torch.save(targets, cache_file)
                found_contexts += 1
                cache_idx += 1
                if found_contexts >= CONTEXTS_PER_TOKEN:
                    break

        # Fallback templates
        if found_contexts < CONTEXTS_PER_TOKEN:
            while found_contexts < CONTEXTS_PER_TOKEN:
                fallback_text = f"Η εξειδικευμένη λέξη {clean} χρησιμοποιείται σε αυτό το πλαίσιο."
                old_ids_tensor = base_tok.encode(fallback_text, add_special_tokens=True, return_tensors="pt").to(device)
                old_ids_list = old_ids_tensor[0].tolist()
                bounds = _find_id_subsequence(old_ids_list, token_ids)
                
                start_pos, end_pos = bounds if bounds is not None else (len(base_tok.encode("Η εξειδικευμένη λέξη ", add_special_tokens=True)), len(base_tok.encode("Η εξειδικευμένη λέξη ", add_special_tokens=True)) + len(token_ids))
                
                cache_file = cache_dir / f"t_{cache_idx:06d}.pt"
                global_map.append({"token": token, "global_idx": idx, "template_idx": found_contexts})
                
                if not cache_file.exists():
                    with torch.no_grad():
                        t_out = teacher(old_ids_tensor, output_hidden_states=True)
                        targets = {"text": fallback_text}
                        for layer in distill_layers:
                            hidden_slice = t_out.hidden_states[layer + 1][0, start_pos:end_pos, :]
                            targets[str(layer)] = hidden_slice.mean(dim=0).cpu()
                    torch.save(targets, cache_file)
                found_contexts += 1
                cache_idx += 1
                
        if (idx + 1) % 20 == 0:
            print(f"  cached {idx+1}/{len(all_tokens)} tokens  total samples={cache_idx}  eta={(_time.time()-t0)/(idx+1)*(len(all_tokens)-idx-1):.0f}s", flush=True)
            
    del teacher; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Cache generated successfully ({cache_idx} samples) in {_time.time()-t0:.0f}s")

    use_parallel = num_gpus >= 2 and cache_idx > 250
    if not use_parallel:
        print("Running single-GPU distillation fallback.")
        all_tokens_expanded = [meta["token"] for meta in global_map]
        return _distill_single(args, model, tokenizer, all_tokens_expanded, cache_dir, distill_layers, layer_weights)

    # ---- MULTI GPU ----
    print(f"Starting multi-GPU distillation on {num_gpus} GPUs.")

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
    if torch.cuda.is_available(): torch.cuda.empty_cache()

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
            "layers": distill_layers,
            "layer_weights": layer_weights,
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
        print(f"Spawning worker for GPU {i} with {len(token_chunks[i])} samples. Log: {log_file}")
        with open(log_file, "w") as log:
            proc = subprocess.Popen(cmd, env=worker_env, stdout=log, stderr=subprocess.STDOUT)
        worker_procs.append(proc)

    num_finished = 0
    for proc in worker_procs:
        proc.wait()
        if proc.returncode == 0:
            print(f"Worker process {proc.pid} finished successfully.")
            num_finished += 1
        else:
            print(f"Worker process {proc.pid} failed with code {proc.returncode}.")

    if num_finished != len(worker_procs):
        print("Warning: Some workers failed. Check logs.")

    print("Merging embeddings across workers...")
    w_in = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    w_out = out_layer.weight if out_layer is not None else None
    tied = bool(w_out is not None and w_out.data_ptr() == w_in.data_ptr())

    for i in range(num_gpus):
        emb_file = Path(args.output_dir) / f"embeddings_gpu{i}.pt"
        if not emb_file.exists(): continue

        data = torch.load(emb_file, map_location="cpu")
        for tid_str, emb in data.get("input_embs", {}).items():
            w_in.data[int(tid_str)].copy_(emb)
        if w_out is not None and not tied:
            for tid_str, emb in data.get("output_embs", {}).items():
                w_out.data[int(tid_str)].copy_(emb)
        emb_file.unlink()

    for cfg_file in worker_configs:
        try: cfg_file.unlink()
        except OSError: pass
    try: tmp_dir.rmdir()
    except OSError: pass

    print("Embeddings successfully merged.")
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
            print(f"Wiping output directory as requested by --overwrite: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            print(f"Output directory exists. Entering RESUME mode (reusing cache and active checkpoints)...")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading extended tokenizer  {args.extended_tokenizer}")
    tok = load_repo_tokenizer(str(args.extended_tokenizer), trust_remote_code=args.trust_remote_code)
    print(f"Vocab size: {len(tok)}")

    print(f"Loading candidates  {args.token_file}")
    raw = [line for line in args.token_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = list(dict.fromkeys(raw))
    print(f"{len(unique)} unique tokens")

    base_tok_path = "artifacts/tokenizers/apertus-base"
    print(f"Loading base tokenizer  {base_tok_path}")
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
    print(f"Report → {args.report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
