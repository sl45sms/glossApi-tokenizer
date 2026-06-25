#!/usr/bin/env python3
"""Geometric Alignment via Orthogonal Procrustes for vocabulary extension.

Implements the second phase of Compositional Geometric Alignment (CGA):
- Extract base model embeddings for anchor tokens shared with FastText.
- Compute the Orthogonal Procrustes transformation W that maps from the
  LLM embedding space to the FastText space (or vice versa).
- Project new token embeddings from FastText into the LLM space.

Math:
    Given X ∈ R^{N×D_base} (LLM embeddings of anchors) and
          Y ∈ R^{N×D_target} (FastText embeddings of anchors),

    Find W = argmin ||XW - Y||_F  subject to  W^T W = I

    Solution via SVD:
        X^T Y = U Σ V^T  →  W = U V^T

    For a new token with FastText vector y_new:
        x_new = y_new W^T   (projects from target → base space)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Allow cross-module imports within vocab-extension (hyphen in dir name prevents package imports)
_VOCAB_EXT_DIR = Path(__file__).resolve().parent
if str(_VOCAB_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_VOCAB_EXT_DIR))

logger = logging.getLogger(__name__)


# ── Orthogonal Procrustes ──────────────────────────────────────────────

def orthogonal_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    center: bool = True,
    scale: bool = False,
) -> Tuple[np.ndarray, float]:
    """Solve the orthogonal Procrustes problem: min ||XW - Y||_F  s.t. W^T W = I.

    Args:
        X: Source matrix of shape (N, d1).
        Y: Target matrix of shape (N, d2). Must have d1 >= d2.
        center: If True, center X and Y by subtracting column means.
        scale: If True, also scale to unit variance (not usually needed).

    Returns:
        (W, disparity) where W is (d1, d2) and disparity = ||XW - Y||_F / ||Y||_F.
    """
    N, d1 = X.shape
    _, d2 = Y.shape

    if N < max(d1, d2):
        raise ValueError(
            f"Number of anchor points ({N}) must be >= max dimension ({max(d1, d2)}). "
            "Consider using more anchors or dimensionality reduction."
        )

    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

    if scale:
        X_std = X.std(axis=0, keepdims=True)
        Y_std = Y.std(axis=0, keepdims=True)
        X_std[X_std == 0] = 1.0
        Y_std[Y_std == 0] = 1.0
        X = X / X_std
        Y = Y / Y_std

    # X^T Y  (d1 × d2)
    M = X.T @ Y  # (d1, d2)

    # SVD
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # U: (d1, min(d1,d2)), s: (min(d1,d2),), Vt: (min(d1,d2), d2)

    # W = U V^T  (d1 × d2)
    W = U @ Vt  # (d1, d2)

    # Compute disparity
    XW = X @ W  # (N, d2)
    disparity = float(np.linalg.norm(XW - Y) / max(np.linalg.norm(Y), 1e-10))

    return W, disparity


def orthogonal_procrustes_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    center: bool = True,
) -> Tuple[torch.Tensor, float]:
    """PyTorch version of orthogonal_procrustes (useful when embeddings are already on GPU)."""
    if center:
        X = X - X.mean(dim=0, keepdims=True)
        Y = Y - Y.mean(dim=0, keepdims=True)

    M = X.T @ Y  # (d1, d2)

    U, s, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt

    XW = X @ W
    disparity = float((torch.norm(XW - Y) / max(torch.norm(Y), 1e-10)).item())

    return W, disparity


# ── Embedding extraction ───────────────────────────────────────────────

def extract_base_embeddings_for_tokens(
    input_embeddings: torch.Tensor,
    tokenizer,
    tokens: Sequence[str],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Extract base model input embeddings for a list of token strings.

    For tokens that are not single tokens in the base tokenizer, we use
    the mean of the subtoken embeddings.

    Args:
        input_embeddings: The model's input embedding weight matrix (vocab_size × dim).
        tokenizer: The tokenizer (for encoding tokens).
        tokens: Token strings to extract embeddings for.
        device: Device to place the output tensor on.

    Returns:
        (X matrix N×D, found_tokens, missing_tokens)
    """
    vocab_size = input_embeddings.shape[0]
    vectors: List[torch.Tensor] = []
    found: List[str] = []
    missing: List[str] = []
    oob_skipped = 0

    with torch.no_grad():
        for token in tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                missing.append(token)
                continue

            # Safety: skip tokens whose IDs exceed the base embedding matrix.
            # This can happen if anchors were extracted from an extended vocab.
            if any(tid >= vocab_size for tid in token_ids):
                oob_skipped += 1
                missing.append(token)
                continue

            ids_tensor = torch.tensor(token_ids, device=input_embeddings.device)
            # Mean-pool subtoken embeddings
            vec = input_embeddings[ids_tensor].mean(dim=0)
            vectors.append(vec)
            found.append(token)

    if oob_skipped:
        logger.warning(
            "Skipped %d tokens with IDs out of bounds for embedding matrix (size %d).",
            oob_skipped, vocab_size,
        )

    if not vectors:
        return (
            torch.empty(0, input_embeddings.shape[1], device=device or input_embeddings.device),
            found,
            missing,
        )

    X = torch.stack(vectors, dim=0)
    if device is not None:
        X = X.to(device)
    return X, found, missing


# ── Main CGA alignment class ───────────────────────────────────────────

class GeometricAligner:
    """Orchestrates the Orthogonal Procrustes alignment between LLM and FastText spaces.

    Supports two modes:

    1. **Direct mode** (fit): Maps D_base → D_target directly via Procrustes.
       Works when D_base >= D_target (e.g., 4096 ≥ 300) but the dimension
       gap causes high disparity and reconstruction error.

    2. **PCA bridge mode** (fit_with_pca_bridge): Reduces D_base to D_target
       via PCA before Procrustes, then reconstructs via inverse PCA after projection.
       This gives near-zero disparity since both sides have the same dimension.

    Usage:
        # Direct mode
        aligner = GeometricAligner()
        aligner.fit(X_4096, Y_300)

        # PCA bridge mode (recommended when D_base >> D_target)
        aligner = GeometricAligner()
        aligner.fit_with_pca_bridge(X_4096, Y_300, pca_dim=300)
    """

    def __init__(self):
        self.W: Optional[np.ndarray] = None  # Procrustes rotation matrix
        self.W_t: Optional[np.ndarray] = None  # Cached transpose of W
        self.X_mean: Optional[np.ndarray] = None  # Base space mean
        self.Y_mean: Optional[np.ndarray] = None  # Target space mean
        self.disparity: Optional[float] = None
        self.d_base: Optional[int] = None
        self.d_target: Optional[int] = None
        self.num_anchors: Optional[int] = None
        self.fitted: bool = False

        # PCA bridge state
        self.pca_bridge: bool = False
        self.pca_components: Optional[np.ndarray] = None  # (pca_dim, D_base)
        self.pca_mean: Optional[np.ndarray] = None  # (D_base,)
        self.pca_variance_retained: Optional[float] = None

    # ── Direct fit (original path) ────────────────────────────────

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        center: bool = True,
        use_torch: bool = False,
    ) -> "GeometricAligner":
        """Compute the Procrustes transformation W from anchor pairs.

        Args:
            X: Base model embeddings for anchors, shape (N, D_base).
            Y: FastText embeddings for the same anchors, shape (N, D_target).
            center: Whether to center the embeddings before alignment.
            use_torch: Use PyTorch SVD (useful for GPU tensors).

        Returns:
            self, for chaining.
        """
        self.pca_bridge = False

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().float().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if isinstance(Y, torch.Tensor):
            Y_np = Y.detach().cpu().float().numpy().astype(np.float64)
        else:
            Y_np = np.asarray(Y, dtype=np.float64)

        N, d1 = X_np.shape
        _, d2 = Y_np.shape

        if d1 < d2:
            raise ValueError(
                f"Base dimension ({d1}) must be >= target dimension ({d2}) "
                f"for orthogonal Procrustes with orthonormal columns. "
                f"Consider using fit_with_pca_bridge() for dimension-mismatched spaces."
            )

        self.d_base = d1
        self.d_target = d2
        self.num_anchors = N

        # Store means for later projection
        self.X_mean = X_np.mean(axis=0, keepdims=True) if center else np.zeros((1, d1))
        self.Y_mean = Y_np.mean(axis=0, keepdims=True) if center else np.zeros((1, d2))

        if use_torch:
            X_t = torch.from_numpy(X_np)
            Y_t = torch.from_numpy(Y_np)
            W_t, disparity = orthogonal_procrustes_torch(X_t, Y_t, center=center)
            self.W = W_t.detach().cpu().float().numpy()
        else:
            self.W, self.disparity = orthogonal_procrustes(X_np, Y_np, center=center)

        self.W_t = self.W.T
        # Recompute disparity with stored W for consistency
        if not use_torch:
            X_centered = X_np - self.X_mean if center else X_np
            Y_centered = Y_np - self.Y_mean if center else Y_np
            XW = X_centered @ self.W
            self.disparity = float(
                np.linalg.norm(XW - Y_centered) / max(np.linalg.norm(Y_centered), 1e-10)
            )
        else:
            self.disparity = disparity

        self.fitted = True
        logger.info(
            "Procrustes fitted (direct): %d anchors, D_base=%d → D_target=%d, disparity=%.6f",
            N, d1, d2, self.disparity,
        )
        return self

    # ── PCA bridge fit ────────────────────────────────────────────

    def fit_with_pca_bridge(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        pca_dim: Optional[int] = None,
        min_variance: float = 0.95,
        center: bool = True,
    ) -> "GeometricAligner":
        """Fit using PCA bridge: reduce LLM space to FastText dimension,
        align in same-dimensional space, then reconstruct via inverse PCA.

        Pipeline:
            X (N, D_base)  →  PCA  →  X_pca (N, pca_dim)
            Y (N, D_target) unchanged
            Procrustes: X_pca @ W ≈ Y  →  W (pca_dim, pca_dim)
            Projection: y_new → y_new @ W^T  →  PCA⁻¹ → LLM space

        Args:
            X: Base model embeddings for anchors, shape (N, D_base).
            Y: FastText embeddings for anchors, shape (N, D_target).
            pca_dim: Target PCA dimension. Defaults to D_target (FastText dim).
            min_variance: Minimum cumulative variance to retain (used if pca_dim is None).
            center: Whether to center before PCA and Procrustes.

        Returns:
            self, for chaining.
        """
        self.pca_bridge = True

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().float().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if isinstance(Y, torch.Tensor):
            Y_np = Y.detach().cpu().float().numpy().astype(np.float64)
        else:
            Y_np = np.asarray(Y, dtype=np.float64)

        N, d_base = X_np.shape
        _, d_target = Y_np.shape

        # Determine PCA dimension
        if pca_dim is None:
            pca_dim = d_target
        pca_dim = min(pca_dim, d_target, N, d_base)

        self.d_base = d_base
        self.d_target = d_target
        self.num_anchors = N

        # ── Step 1: Fit PCA on X ──────────────────────────────────
        logger.info("Fitting PCA: %d anchors, %d → %d dimensions ...", N, d_base, pca_dim)
        self.pca_mean = X_np.mean(axis=0) if center else np.zeros(d_base)
        X_centered = X_np - self.pca_mean if center else X_np

        # Use covariance matrix for efficiency (N >> d)
        cov = (X_centered.T @ X_centered) / (N - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order, we want descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top pca_dim components
        self.pca_components = eigenvectors[:, :pca_dim].T.copy()  # (pca_dim, d_base)
        total_var = eigenvalues.sum()
        retained_var = eigenvalues[:pca_dim].sum()
        self.pca_variance_retained = float(retained_var / total_var) if total_var > 0 else 1.0

        logger.info(
            "PCA: %d dims retain %.2f%% variance (total=%d dims).",
            pca_dim, self.pca_variance_retained * 100, d_base,
        )

        # ── Step 2: Reduce X via PCA ──────────────────────────────
        X_pca = X_centered @ self.pca_components.T  # (N, pca_dim)

        # ── Step 3: Procrustes on same-dimensional spaces ──────────
        self.X_mean = X_pca.mean(axis=0, keepdims=True) if center else np.zeros((1, pca_dim))
        self.Y_mean = Y_np.mean(axis=0, keepdims=True) if center else np.zeros((1, d_target))

        # Sanity: if pca_dim != d_target (unlikely), pad/truncate Y to match
        if pca_dim != d_target:
            logger.warning(
                "PCA dimension (%d) ≠ FastText dimension (%d). "
                "Padding Y with zeros to match.",
                pca_dim, d_target,
            )
            if pca_dim > d_target:
                Y_padded = np.pad(Y_np, ((0, 0), (0, pca_dim - d_target)))
            else:
                Y_padded = Y_np[:, :pca_dim]
            self.W, self.disparity = orthogonal_procrustes(X_pca, Y_padded, center=center)
            self.W = self.W[:d_target, :pca_dim]  # trim back
        else:
            self.W, self.disparity = orthogonal_procrustes(X_pca, Y_np, center=center)

        self.W_t = self.W.T  # (d_target, pca_dim) or (pca_dim, pca_dim) when equal

        self.fitted = True
        logger.info(
            "Procrustes fitted (PCA bridge): %d anchors, %d→%d via PCA, "
            "disparity=%.6f, PCA variance=%.2f%%",
            N, d_base, d_target, self.disparity, self.pca_variance_retained * 100,
        )
        return self

    # ── Projection ────────────────────────────────────────────────

    def project_target_to_base(
        self,
        Y_new: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Project vectors from FastText space → LLM embedding space.

        In direct mode:
            x_new = (y_new - Y_mean) @ W^T + X_mean

        In PCA bridge mode:
            x_reduced = (y_new - Y_mean) @ W^T + X_mean   (in PCA space)
            x_new = x_reduced @ pca_components + pca_mean   (reconstruct to LLM space)

        Args:
            Y_new: FastText vectors, shape (M, D_target) or (D_target,).

        Returns:
            Projected vectors in LLM space, shape (M, D_base) or (D_base,).
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() or fit_with_pca_bridge() before project_target_to_base().")

        if isinstance(Y_new, torch.Tensor):
            Y_new = Y_new.detach().cpu().float().numpy()

        Y_np = np.asarray(Y_new, dtype=np.float64)
        was_1d = Y_np.ndim == 1
        if was_1d:
            Y_np = Y_np.reshape(1, -1)

        # Center in target space
        Y_centered = Y_np - self.Y_mean

        # Project through W^T into (PCA or base) space
        X_proj = Y_centered @ self.W_t  # (M, pca_dim)

        # De-center
        X_proj = X_proj + self.X_mean

        # PCA bridge: reconstruct to full LLM space
        if self.pca_bridge and self.pca_components is not None:
            X_proj = X_proj @ self.pca_components  # (M, pca_dim) @ (pca_dim, d_base) = (M, d_base)
            X_proj = X_proj + self.pca_mean

        if was_1d:
            X_proj = X_proj.reshape(-1)

        return X_proj.astype(np.float32)

    def project_base_to_target(
        self,
        X_new: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Project vectors from LLM space → FastText space (for analysis).

        In PCA bridge mode, first reduces via PCA, then projects.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() or fit_with_pca_bridge() before project_base_to_target().")

        if isinstance(X_new, torch.Tensor):
            X_new = X_new.detach().cpu().float().numpy()

        X_np = np.asarray(X_new, dtype=np.float64)
        was_1d = X_np.ndim == 1
        if was_1d:
            X_np = X_np.reshape(1, -1)

        # PCA bridge: reduce to PCA space first
        if self.pca_bridge and self.pca_components is not None:
            X_np = (X_np - self.pca_mean) @ self.pca_components.T  # (M, pca_dim)

        X_centered = X_np - self.X_mean
        Y_proj = X_centered @ self.W
        Y_proj = Y_proj + self.Y_mean

        if was_1d:
            Y_proj = Y_proj.reshape(-1)

        return Y_proj.astype(np.float32)

    def to_dict(self) -> Dict:
        """Serialize alignment parameters to a dictionary."""
        if not self.fitted:
            return {"fitted": False}

        result = {
            "fitted": True,
            "d_base": self.d_base,
            "d_target": self.d_target,
            "num_anchors": self.num_anchors,
            "disparity": self.disparity,
            "W_shape": list(self.W.shape) if self.W is not None else None,
            "pca_bridge": self.pca_bridge,
        }
        if self.pca_bridge:
            result["pca_variance_retained"] = self.pca_variance_retained
            result["pca_components_shape"] = (
                list(self.pca_components.shape) if self.pca_components is not None else None
            )
        return result


# ── New token initialization via CGA ───────────────────────────────────

def initialize_new_embeddings_cga(
    new_tokens: Sequence[str],
    ft_model,  # FastTextVectorModel or FastTextSubwordModel
    aligner: GeometricAligner,
    input_embeddings: torch.Tensor,
    tokenizer,
    initialization_source_ids: Optional[Dict[str, List[int]]] = None,
    fallback_strategy: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Initialize embeddings for new tokens using CGA projection.

    For each new token:
    1. Get its FastText vector y_new.
    2. Project to LLM space: x_cga = aligner.project_target_to_base(y_new).
    3. If FastText vector is unavailable, fall back to mean/zero strategy.

    Args:
        new_tokens: Token strings to initialize.
        ft_model: Loaded FastText model.
        aligner: Fitted GeometricAligner.
        input_embeddings: The model's input embedding weight tensor (vocab_size × D_base).
        tokenizer: The tokenizer.
        initialization_source_ids: Optional dict mapping token → source subtoken IDs
            (used for fallback mean initialization).
        fallback_strategy: 'mean' (mean of subtokens), 'zero', or 'skip'.

    Returns:
        Dict mapping token → initialized embedding vector (torch.Tensor on same device).
    """
    from morpho_bpe import normalize_greek_token_for_matching, strip_greek_accents

    if initialization_source_ids is None:
        initialization_source_ids = {}

    initialized: Dict[str, torch.Tensor] = {}
    device = input_embeddings.device
    D_base = input_embeddings.shape[1]

    for token in new_tokens:
        new_id = tokenizer.convert_tokens_to_ids(token)
        if new_id is None or new_id < 0:
            continue

        # Try FastText lookup
        clean = token.lstrip("Ġ▁ ")
        ft_vec = None

        if hasattr(ft_model, "get_vector"):
            # Try multiple forms
            for form in (clean, clean.lower(), strip_greek_accents(clean.lower())):
                try:
                    vec = ft_model.get_vector(form)
                    if vec is not None and (hasattr(vec, "any") and vec.any() or np.any(vec)):
                        ft_vec = vec
                        break
                except Exception:
                    continue

        if ft_vec is not None:
            # CGA projection
            x_cga = aligner.project_target_to_base(ft_vec)
            initialized[token] = torch.from_numpy(x_cga).to(device=device, dtype=input_embeddings.dtype)

        elif fallback_strategy == "mean":
            # Fallback: mean of subtoken embeddings
            source_ids = initialization_source_ids.get(token, [])
            if source_ids:
                initialized[token] = input_embeddings[source_ids].mean(dim=0).clone()
            else:
                # Tokenize and mean-pool
                token_ids = tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    ids_tensor = torch.tensor(token_ids, device=device)
                    initialized[token] = input_embeddings[ids_tensor].mean(dim=0).clone()
                elif fallback_strategy != "skip":
                    initialized[token] = torch.zeros(D_base, device=device, dtype=input_embeddings.dtype)

        elif fallback_strategy == "zero":
            initialized[token] = torch.zeros(D_base, device=device, dtype=input_embeddings.dtype)

        # 'skip' does nothing

    logger.info(
        "Initialized %d / %d new token embeddings via CGA (fallback=%s).",
        len(initialized), len(new_tokens), fallback_strategy,
    )
    return initialized


def apply_cga_to_model_embeddings(
    model,
    tokenizer,
    new_tokens: Sequence[str],
    cga_initialized: Dict[str, torch.Tensor],
    initialization_source_ids: Dict[str, List[int]],
    output_init_strategy: str = "zero",
) -> Dict:
    """Apply CGA-initialized embeddings to the model's weight matrices.

    Args:
        model: The causal LM whose embeddings to resize and initialize.
        tokenizer: The extended tokenizer.
        new_tokens: Tokens that were added.
        cga_initialized: Dict token → CGA-projected embedding.
        initialization_source_ids: Dict token → source subtoken IDs (for fallback).
        output_init_strategy: 'zero', 'mean', or 'keep-resized' for output head.

    Returns:
        Statistics dict.
    """
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    input_embeddings = model.get_input_embeddings().weight
    output_embedding_layer = model.get_output_embeddings()
    output_embeddings = output_embedding_layer.weight if output_embedding_layer is not None else None
    output_embeddings_share_storage = bool(
        output_embeddings is not None
        and output_embeddings.data_ptr() == input_embeddings.data_ptr()
    )

    cga_count = 0
    mean_fallback_count = 0
    zero_count = 0
    output_initialized = 0

    with torch.no_grad():
        for token in new_tokens:
            new_id = tokenizer.convert_tokens_to_ids(token)
            if new_id is None or new_id < 0:
                continue

            if token in cga_initialized:
                input_embeddings[new_id].copy_(cga_initialized[token])
                cga_count += 1
            else:
                source_ids = initialization_source_ids.get(token, [])
                if source_ids:
                    input_embeddings[new_id].copy_(
                        input_embeddings[source_ids].mean(dim=0)
                    )
                    mean_fallback_count += 1
                else:
                    # Try tokenizing
                    token_ids = tokenizer.encode(token, add_special_tokens=False)
                    if token_ids:
                        ids_tensor = torch.tensor(token_ids, device=input_embeddings.device)
                        input_embeddings[new_id].copy_(
                            input_embeddings[ids_tensor].mean(dim=0)
                        )
                        mean_fallback_count += 1
                    else:
                        zero_count += 1

            # Output head
            if output_embeddings is not None and not output_embeddings_share_storage:
                if output_init_strategy == "mean":
                    source_ids = initialization_source_ids.get(token, [])
                    if source_ids:
                        output_embeddings[new_id].copy_(
                            output_embeddings[source_ids].mean(dim=0)
                        )
                    output_initialized += 1
                elif output_init_strategy == "zero":
                    output_embeddings[new_id].zero_()
                    output_initialized += 1
                # 'keep-resized': do nothing

    return {
        "cga_initialized": cga_count,
        "mean_fallback_initialized": mean_fallback_count,
        "zero_initialized": zero_count,
        "output_embeddings_initialized": output_initialized,
        "output_embeddings_share_storage": output_embeddings_share_storage,
        "output_init_strategy": output_init_strategy if not output_embeddings_share_storage else "shared-with-input",
    }
