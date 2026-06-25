#!/usr/bin/env python3
"""Compositional Geometric Alignment (CGA) Pipeline.

Orchestrates the full CGA vocabulary extension workflow:

Phase 1 - Morpho-BPE:
    Load Greek morphological anchors (prefixes, suffixes, stems, forced tokens)
    and score/filter candidate tokens by morphological coherence.

Phase 2 - Geometric Alignment:
    Download/load Greek FastText vectors, find anchor tokens common to both
    the base LLM and FastText, compute Orthogonal Procrustes transformation W,
    and project new token embeddings from FastText space into LLM space.

Phase 3 - Compositional Residuals:
    For compound/derived words not found in FastText, attempt morphological
    decomposition and tensor-based composition of root + affix embeddings.

Output:
    - Extended tokenizer saved to --output-dir
    - Resized model checkpoint saved to --model-output-dir (when --base-model given)
    - JSON report with alignment statistics

Usage:
    # Full CGA pipeline with model initialization
    python vocab-extension/cga_pipeline.py \
        --base-tokenizer artifacts/tokenizers/apertus-base \
        --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
        --base-model swiss-ai/Apertus-8B-Instruct-2509 \
        --output-dir artifacts/tokenizers/apertus-greek-cga-v1 \
        --model-output-dir /scratch/$USER/apertus-greek-cga-v1 \
        --trust-remote-code --torch-dtype bfloat16

    # Tokenizer-only (no model loading)
    python vocab-extension/cga_pipeline.py \
        --base-tokenizer artifacts/tokenizers/apertus-base \
        --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
        --output-dir artifacts/tokenizers/apertus-greek-cga-v1
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add vocab-extension dir for cross-module imports (hyphen in dir name)
_VOCAB_EXT_DIR = Path(__file__).resolve().parent
if str(_VOCAB_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_VOCAB_EXT_DIR))

from repo_tokenizer import load_repo_tokenizer

# Import CGA modules (now importable because _VOCAB_EXT_DIR is on sys.path)
from morpho_bpe import (
    MorphologicalAnchorSet,
    filter_by_morphological_quality,
    score_candidate_tokens,
)
from fasttext_utils import (
    FastTextVectorModel,
    build_anchor_embeddings,
    extract_anchor_tokens,
    load_greek_fasttext,
)
from geometric_alignment import (
    GeometricAligner,
    apply_cga_to_model_embeddings,
    extract_base_embeddings_for_tokens,
    initialize_new_embeddings_cga,
)
from compositional_residuals import (
    CompositionalEmbeddingModel,
    batch_compose_compound_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cga_pipeline")


# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_BASE_TOKENIZER = Path("artifacts/tokenizers/apertus-base")
DEFAULT_TOKEN_FILE = Path("artifacts/vocab_candidates/selected_tokens_v1.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizers/apertus-greek-cga-v1")
DEFAULT_REPORT_PATH = Path("artifacts/reports/cga_pipeline_report.json")
DEFAULT_STATIC_DIR = Path("vocabularyGen/static")


def default_model_output_dir() -> Path:
    scratch_root = os.environ.get("SCRATCH")
    if scratch_root:
        return Path(scratch_root) / "apertus-greek-cga-v1"
    return Path("artifacts/checkpoints/apertus-greek-cga-v1")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compositional Geometric Alignment (CGA) pipeline for Greek vocabulary extension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Tokenizer inputs
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

    # Model initialization
    parser.add_argument(
        "--base-model",
        help="Optional local model path or Hugging Face model id for a causal LM to resize and initialize via CGA.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=default_model_output_dir(),
        help="Directory for the CGA-initialized model checkpoint.",
    )

    # CGA-specific options
    parser.add_argument(
        "--fasttext-cache-dir",
        type=Path,
        default=None,
        help="Directory to cache downloaded FastText models.",
    )
    parser.add_argument(
        "--fasttext-use-subword",
        action="store_true",
        help="Use fastText .bin model for subword-aware vector lookup (requires `fasttext` package).",
    )
    parser.add_argument(
        "--min-anchors",
        type=int,
        default=200,
        help="Minimum number of anchor tokens for Procrustes alignment.",
    )
    parser.add_argument(
        "--no-morpho-filter",
        action="store_true",
        help="Skip morphological quality filtering of candidate tokens.",
    )
    parser.add_argument(
        "--morpho-min-score",
        type=float,
        default=0.2,
        help="Minimum morphological coherence score to keep a candidate (0.0-1.0).",
    )
    parser.add_argument(
        "--no-compositional",
        action="store_true",
        help="Skip compositional residual computation for compound words.",
    )
    parser.add_argument(
        "--pca-bridge",
        action="store_true",
        help=(
            "Use PCA bridge: reduce LLM embeddings to FastText dimension via PCA, "
            "align in same-dimensional space (disparity ≈ 0), then reconstruct via PCA⁻¹. "
            "Recommended when D_base >> D_target (e.g., 4096 vs 300)."
        ),
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="Target PCA dimension (default: FastText dimension, usually 300).",
    )
    parser.add_argument(
        "--pca-min-variance",
        type=float,
        default=0.95,
        help="Minimum cumulative variance to retain when pca-dim is not set.",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=DEFAULT_STATIC_DIR,
        help="Directory with morphological anchor files (prothimata.txt, epithemata.txt, forced.txt).",
    )

    # Model loading options
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer/model.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype to use when loading --base-model.",
    )
    parser.add_argument(
        "--untied-output-init-strategy",
        choices=("zero", "mean", "keep-resized"),
        default="zero",
        help="How to initialize new lm_head rows when not tied.",
    )

    # Output control
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=50,
        help="Maximum number of example tokens in the report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output directories/files.",
    )
    parser.add_argument(
        "--save-alignment",
        type=Path,
        default=None,
        help="If set, save the Procrustes W matrix and alignment metadata to this .npz file.",
    )

    return parser.parse_args()


# ── Validation & preparation ───────────────────────────────────────────

def validate_args(args: argparse.Namespace) -> None:
    if not args.token_file.exists():
        raise SystemExit(f"Token file not found: {args.token_file}")

    base_tok_path = Path(args.base_tokenizer)
    if base_tok_path.exists() and base_tok_path.resolve() == args.output_dir.resolve():
        raise SystemExit("--output-dir must differ from --base-tokenizer.")

    if args.base_model:
        base_model_path = Path(args.base_model)
        if base_model_path.exists():
            if base_model_path.resolve() == args.output_dir.resolve():
                raise SystemExit("--output-dir must differ from --base-model.")
            if base_model_path.resolve() == args.model_output_dir.resolve():
                raise SystemExit("--model-output-dir must differ from --base-model.")


def prepare_output_paths(args: argparse.Namespace) -> None:
    for path in (args.report_path,):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_dirs = {args.output_dir.resolve(): args.output_dir}
    if args.base_model:
        output_dirs[args.model_output_dir.resolve()] = args.model_output_dir

    for directory in output_dirs.values():
        if directory.exists():
            if args.overwrite:
                shutil.rmtree(directory)
            else:
                raise SystemExit(f"Refusing to overwrite existing directory: {directory}. Use --overwrite.")

    for path in (args.report_path,):
        if path.exists():
            if args.overwrite:
                path.unlink()
            else:
                raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


# ── Token loading (mirrors extend_apertus_tokenizer.py) ────────────────

def load_candidate_tokens(token_file: Path) -> Tuple[List[str], Dict[str, int]]:
    raw_tokens: List[str] = []
    seen: set = set()
    unique_tokens: List[str] = []
    duplicate_count = 0

    for line in token_file.read_text(encoding="utf-8").splitlines():
        token = line
        if not token.strip():
            continue
        raw_tokens.append(token)
        if token in seen:
            duplicate_count += 1
            continue
        seen.add(token)
        unique_tokens.append(token)

    return unique_tokens, {
        "raw_non_empty_token_count": len(raw_tokens),
        "unique_input_token_count": len(unique_tokens),
        "duplicate_input_count": duplicate_count,
    }


def has_exact_single_token_coverage(tokenizer, token: str) -> Tuple[bool, List[int]]:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) if token_ids else ""
    return len(token_ids) == 1 and decoded == token, token_ids


def partition_tokens(
    tokenizer,
    tokens: Sequence[str],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[int]]]:
    tokens_to_add: List[str] = []
    skipped: List[Dict[str, Any]] = []
    source_ids: Dict[str, List[int]] = {}

    for token in tokens:
        exact, token_ids = has_exact_single_token_coverage(tokenizer, token)
        if exact:
            skipped.append({
                "token": token,
                "reason": "already_present_as_exact_single_token",
                "existing_token_id": token_ids[0],
            })
            continue

        tokens_to_add.append(token)
        source_ids[token] = list(token_ids)

    return tokens_to_add, skipped, source_ids


# ── Main pipeline ──────────────────────────────────────────────────────

def resolve_torch_dtype(torch_dtype_name: str):
    if torch_dtype_name == "auto":
        return "auto"
    import torch
    return getattr(torch, torch_dtype_name)


def run_cga_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the full CGA pipeline and return a report dict."""
    import torch
    from transformers import AutoModelForCausalLM

    report: Dict[str, Any] = {
        "pipeline": "cga",
        "base_tokenizer": args.base_tokenizer,
        "token_file": str(args.token_file),
        "output_dir": str(args.output_dir),
    }

    # ── Load tokenizer ─────────────────────────────────────────────
    logger.info("Loading base tokenizer from %s ...", args.base_tokenizer)
    tokenizer = load_repo_tokenizer(args.base_tokenizer, trust_remote_code=args.trust_remote_code)
    report["tokenizer_class"] = tokenizer.__class__.__name__
    report["initial_vocab_size"] = len(tokenizer)

    # ── Load & partition candidate tokens ──────────────────────────
    unique_tokens, token_input_stats = load_candidate_tokens(args.token_file)
    report["token_input_stats"] = token_input_stats

    tokens_to_add, skipped_tokens, initialization_source_ids = partition_tokens(
        tokenizer, unique_tokens,
    )
    report["tokens_requested_for_addition"] = len(tokens_to_add)
    report["skipped_existing_token_count"] = len(skipped_tokens)
    logger.info(
        "Candidates: %d unique, %d to add, %d skipped (already present).",
        len(unique_tokens), len(tokens_to_add), len(skipped_tokens),
    )

    # ── Phase 1: Morpho-BPE filtering ──────────────────────────────
    morpho_report: Dict[str, Any] = {"enabled": not args.no_morpho_filter}
    if not args.no_morpho_filter:
        logger.info("Phase 1: Morpho-BPE filtering ...")
        anchors = MorphologicalAnchorSet.from_static_dir(args.static_dir)
        logger.info(
            "Loaded %d prefixes, %d suffixes, %d forced tokens.",
            len(anchors.prefixes), len(anchors.suffixes), len(anchors.forced),
        )

        scored = score_candidate_tokens(tokens_to_add, anchors)
        tokens_to_add = filter_by_morphological_quality(
            scored, min_score=args.morpho_min_score,
        )
        morpho_report["prefix_count"] = len(anchors.prefixes)
        morpho_report["suffix_count"] = len(anchors.suffixes)
        morpho_report["forced_count"] = len(anchors.forced)
        morpho_report["tokens_after_morpho_filter"] = len(tokens_to_add)
        morpho_report["min_score"] = args.morpho_min_score
        logger.info("After morpho filter: %d tokens remain.", len(tokens_to_add))
    report["morpho_bpe"] = morpho_report

    if not tokens_to_add:
        logger.warning("No tokens to add after filtering. Exiting.")
        report["num_added"] = 0
        report["error"] = "no_tokens_to_add"
        return report

    # Snapshot base vocabulary BEFORE extension so anchor extraction
    # only considers tokens that exist in the base model's embeddings.
    base_vocab_before_extend = list(tokenizer.get_vocab().keys())
    initial_vocab_size = len(tokenizer)

    # ── Add tokens to tokenizer ────────────────────────────────────
    num_added = tokenizer.add_tokens(tokens_to_add)
    report["num_added"] = num_added
    report["final_vocab_size"] = len(tokenizer)

    tokenizer.save_pretrained(args.output_dir)
    # Also normalize config (import from tokenizer_extract_common)
    try:
        from tokenizer_extract_common import normalize_tokenizer_config
        normalize_tokenizer_config(args.output_dir)
    except Exception:
        logger.warning("Could not normalize tokenizer config; continuing.")

    logger.info("Extended tokenizer saved to %s (%d tokens).", args.output_dir, len(tokenizer))

    # ── If no base model, we're done ───────────────────────────────
    if not args.base_model:
        report["model_initialization"] = {"enabled": False, "reason": "no_base_model_provided"}
        return report

    # ── Load base model ────────────────────────────────────────────
    logger.info("Loading base model from %s ...", args.base_model)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        dtype=torch_dtype,
    )
    input_embeddings = model.get_input_embeddings().weight
    D_base = input_embeddings.shape[1]
    report["model_info"] = {
        "base_model": args.base_model,
        "D_base": D_base,
        "torch_dtype": args.torch_dtype,
        "vocab_size_before_resize": input_embeddings.shape[0],
    }

    # ── Phase 2: Geometric Alignment ───────────────────────────────
    logger.info("Phase 2: Loading Greek FastText vectors ...")
    ft_model = load_greek_fasttext(
        cache_dir=args.fasttext_cache_dir,
        use_subword=args.fasttext_use_subword,
    )
    D_target = ft_model.dim
    report["fasttext"] = {
        "num_vectors": len(ft_model) if hasattr(ft_model, "__len__") else "subword_model",
        "dim": D_target,
        "use_subword": args.fasttext_use_subword,
    }

    # Extract anchor tokens common to both spaces.
    # Use the pre-extension vocab so anchors are guaranteed to have valid
    # token IDs within the base model's embedding matrix.
    logger.info("Extracting anchor tokens (from base vocab of %d tokens) ...", initial_vocab_size)
    anchors = extract_anchor_tokens(base_vocab_before_extend, ft_model, min_anchor_count=args.min_anchors)
    report["anchors"] = {
        "extracted_count": len(anchors),
        "min_required": args.min_anchors,
        "base_vocab_size": initial_vocab_size,
    }

    if len(anchors) < 3:
        logger.warning("Too few anchors (%d). Falling back to mean initialization.", len(anchors))
        model_initialization_info = _fallback_mean_init(
            args, model, tokenizer, tokens_to_add, initialization_source_ids,
        )
        report["model_initialization"] = model_initialization_info
        return report

    # Build X (base LLM embeddings for anchors) and Y (FastText embeddings)
    logger.info("Building anchor embedding matrices ...")
    X, found_x, missing_x = extract_base_embeddings_for_tokens(
        input_embeddings, tokenizer, anchors,
    )
    Y, found_y, missing_y = build_anchor_embeddings(anchors, ft_model)

    # Keep only anchors found in both spaces
    common_anchors = sorted(set(found_x) & set(found_y), key=lambda a: list(anchors).index(a) if a in anchors else 9999)
    logger.info("Common anchors for alignment: %d", len(common_anchors))

    if len(common_anchors) < 10:
        logger.warning("Too few common anchors (%d). Falling back to mean init.", len(common_anchors))
        model_initialization_info = _fallback_mean_init(
            args, model, tokenizer, tokens_to_add, initialization_source_ids,
        )
        report["model_initialization"] = model_initialization_info
        return report

    # Re-extract with common anchors only
    X_common, _, _ = extract_base_embeddings_for_tokens(
        input_embeddings, tokenizer, common_anchors,
    )
    Y_common, _, _ = build_anchor_embeddings(common_anchors, ft_model)

    X_np = X_common.detach().cpu().float().numpy().astype(np.float64)
    Y_np = Y_common.astype(np.float64)

    # Compute Procrustes alignment (direct or PCA bridge)
    aligner = GeometricAligner()
    if args.pca_bridge:
        logger.info(
            "Computing PCA-bridge Procrustes alignment "
            "(PCA: %d → %d, then %d ↔ %d) ...",
            D_base, args.pca_dim or D_target, args.pca_dim or D_target, D_target,
        )
        aligner.fit_with_pca_bridge(
            X_np, Y_np,
            pca_dim=args.pca_dim,
            min_variance=args.pca_min_variance,
            center=True,
        )
    else:
        logger.info("Computing direct Orthogonal Procrustes alignment ...")
        aligner.fit(X_np, Y_np, center=True)

    report["procrustes"] = {
        "mode": "pca_bridge" if args.pca_bridge else "direct",
        "num_common_anchors": len(common_anchors),
        "disparity": aligner.disparity,
        "D_base": D_base,
        "D_target": D_target,
    }
    if args.pca_bridge:
        report["procrustes"]["pca_variance_retained"] = aligner.pca_variance_retained
        report["procrustes"]["pca_components_shape"] = (
            list(aligner.pca_components.shape) if aligner.pca_components is not None else None
        )
    logger.info("Procrustes disparity: %.6f", aligner.disparity)

    # Save alignment if requested
    if args.save_alignment:
        np.savez(
            args.save_alignment,
            W=aligner.W,
            X_mean=aligner.X_mean,
            Y_mean=aligner.Y_mean,
            D_base=D_base,
            D_target=D_target,
            anchors=np.array(common_anchors),
        )
        logger.info("Alignment saved to %s", args.save_alignment)

    # ── Phase 3: Compositional residuals (for compound words) ──────
    compositional_report: Dict[str, Any] = {"enabled": not args.no_compositional}
    compound_embeddings: Dict[str, np.ndarray] = {}

    if not args.no_compositional:
        logger.info("Phase 3: Computing compositional embeddings for compound words ...")
        composition_model = CompositionalEmbeddingModel(dim=D_target, per_dimension=True)
        composition_model.eval()

        # Identify compound words among new tokens (those not directly in FastText)
        compound_candidates: List[str] = []
        for token in tokens_to_add:
            clean = token.lstrip("Ġ▁ ")
            if clean not in ft_model:
                compound_candidates.append(token)

        if compound_candidates:
            compound_embeddings, failed_compounds = batch_compose_compound_embeddings(
                compound_candidates, ft_model, aligner, composition_model,
            )
            compositional_report["compound_candidates"] = len(compound_candidates)
            compositional_report["composed_successfully"] = len(compound_embeddings)
            compositional_report["composition_failed"] = len(failed_compounds)
            if failed_compounds:
                compositional_report["failed_examples"] = failed_compounds[:20]
            logger.info(
                "Composition: %d succeeded, %d failed.",
                len(compound_embeddings), len(failed_compounds),
            )
        else:
            compositional_report["compound_candidates"] = 0
            compositional_report["note"] = "All new tokens found directly in FastText."
    report["compositional_residuals"] = compositional_report

    # ── Initialize new embeddings via CGA ──────────────────────────
    logger.info("Initializing new token embeddings via CGA ...")

    # First try CGA projection via FastText
    cga_initialized = initialize_new_embeddings_cga(
        tokens_to_add,
        ft_model,
        aligner,
        input_embeddings,
        tokenizer,
        initialization_source_ids,
        fallback_strategy="skip",  # We'll handle fallback ourselves
    )

    # Merge in compound embeddings
    for token, emb in compound_embeddings.items():
        if token not in cga_initialized:
            cga_initialized[token] = torch.from_numpy(emb).to(
                device=input_embeddings.device, dtype=input_embeddings.dtype,
            )

    # For remaining tokens, fall back to mean
    for token in tokens_to_add:
        if token not in cga_initialized:
            source_ids = initialization_source_ids.get(token, [])
            if source_ids:
                cga_initialized[token] = input_embeddings[source_ids].mean(dim=0).clone()

    # Apply to model
    model_init_stats = apply_cga_to_model_embeddings(
        model,
        tokenizer,
        tokens_to_add,
        cga_initialized,
        initialization_source_ids,
        output_init_strategy=args.untied_output_init_strategy,
    )

    # Save model
    logger.info("Saving CGA-initialized model to %s ...", args.model_output_dir)
    model.save_pretrained(args.model_output_dir)
    if args.model_output_dir.resolve() != args.output_dir.resolve():
        tokenizer.save_pretrained(args.model_output_dir)
        try:
            from tokenizer_extract_common import normalize_tokenizer_config
            normalize_tokenizer_config(args.model_output_dir)
        except Exception:
            pass

    model_init_stats["model_output_dir"] = str(args.model_output_dir)
    model_init_stats["total_new_tokens"] = len(tokens_to_add)
    report["model_initialization"] = model_init_stats

    logger.info("CGA pipeline complete.")
    return report


def _fallback_mean_init(
    args: argparse.Namespace,
    model,
    tokenizer,
    tokens_to_add: List[str],
    initialization_source_ids: Dict[str, List[int]],
) -> Dict[str, Any]:
    """Fallback: mean initialization (same as extend_apertus_tokenizer.py)."""
    import torch

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_embeddings = model.get_input_embeddings().weight
    output_embedding_layer = model.get_output_embeddings()
    output_embeddings = output_embedding_layer.weight if output_embedding_layer is not None else None
    output_share = bool(output_embeddings is not None and output_embeddings.data_ptr() == input_embeddings.data_ptr())

    input_init_count = 0
    output_init_count = 0

    with torch.no_grad():
        for token in tokens_to_add:
            source_ids = initialization_source_ids.get(token, [])
            if not source_ids:
                continue
            new_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings[new_id].copy_(input_embeddings[source_ids].mean(dim=0))
            input_init_count += 1

            if output_embeddings is not None and not output_share:
                if args.untied_output_init_strategy == "mean":
                    output_embeddings[new_id].copy_(output_embeddings[source_ids].mean(dim=0))
                elif args.untied_output_init_strategy == "zero":
                    output_embeddings[new_id].zero_()
                output_init_count += 1

    model.save_pretrained(args.model_output_dir)
    if args.model_output_dir.resolve() != args.output_dir.resolve():
        tokenizer.save_pretrained(args.model_output_dir)
        try:
            from tokenizer_extract_common import normalize_tokenizer_config
            normalize_tokenizer_config(args.model_output_dir)
        except Exception:
            pass

    return {
        "enabled": True,
        "strategy": "mean_fallback",
        "reason": "insufficient_anchors_for_cga",
        "initialized_input_embeddings": input_init_count,
        "initialized_output_embeddings": output_init_count,
        "output_embeddings_share_storage": output_share,
        "model_output_dir": str(args.model_output_dir),
    }


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_output_paths(args)

    report = run_cga_pipeline(args)

    # Write report
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Signal any issues
    if report.get("error"):
        logger.error("Pipeline ended with error: %s", report["error"])
        sys.exit(1)


if __name__ == "__main__":
    main()
