#!/usr/bin/env python3
"""Compositional Residuals (Morphological Tensor) for vocabulary extension.

Implements the third phase of Compositional Geometric Alignment (CGA):
- Decompose compound/derived Greek words into root + affix components.
- Compute embeddings for unknown compound words using tensor composition.

For a compound word like "ανθρωπότητα" (humanity):
    - root: "άνθρωπ-" (human)
    - suffix: "-ότητα" (-ity)

The embedding is computed as:
    E_new = W^T ( f(E_root) ⊗ g(E_suffix) )

Where:
    - f, g are learnable linear projections of the component embeddings
    - ⊗ is an element-wise interaction (Hadamard product or outer product)
    - W^T projects from the intermediate space back to LLM space

For practical implementation, we use:
    E_new = α * E_root + β * E_suffix + γ * (E_root ⊙ E_suffix)

where ⊙ is the Hadamard (element-wise) product and α, β, γ are learned scalars
(or per-dimension weights).

When the root or suffix is not in FastText, we fall back to using the base model's
subtoken mean embeddings.
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


# ── Morphological decomposition ────────────────────────────────────────

# Common Greek derivational suffixes with their canonical forms
GREEK_DERIVATIONAL_SUFFIXES: Dict[str, str] = {
    # Nominalization suffixes
    "-ότητα": "-ότητα",   # -ότητα (quality): ανθρωπότητα, βεβαιότητα
    "-ότητα": "-ότητα",
    "-ωση": "-ωση",       # -ωση (process): εκβιομηχάνιση → οργάνωση
    "-ιση": "-ιση",       # -ιση (action): ποίηση, κίνηση
    "-εια": "-εια",       # -εια (state): αλήθεια, βοήθεια
    "-εια": "-εια",
    "-ία": "-ία",         # -ία (quality): σοφία, κακία
    "-ία": "-ία",
    "-μός": "-μός",       # -μός (act/result): σεισμός, παλμός
    "-ση": "-ση",         # -ση (action/result): στάση, πράξη → γνώση
    "-μα": "-μα",         # -μα (result): πράγμα, γράμμα
    "-ιμο": "-ιμο",       # -ιμο (action): τρέξιμο, γράψιμο
    "-άδα": "-άδα",       # -άδα (collective): δεκάδα, ντουζίνα → ομάδα
    "-άρι": "-άρι",       # -άρι (diminutive/place): σχολάρι, ψαράδικο
    "-άκι": "-άκι",       # -άκι (diminutive): παιδάκι, σπιτάκι
    "-ούλα": "-ούλα",     # -ούλα (diminutive): μανούλα, καρδούλα
    "-άκος": "-άκος",     # -άκος (diminutive masc): Γιωργάκης → δρόμος

    # Agent/doer suffixes
    "-τής": "-τής",       # -τής (agent): ποιητής, μαθητής
    "-της": "-της",       # -της (agent/measure): καθρέφτης, μέτρο → πλάτης
    "-τήρας": "-τήρας",   # -τήρας (instrument): κινητήρας, θερμαντήρας
    "-ιστής": "-ιστής",   # -ιστής (professional): καλλιτέχνης → καθηγητής
    "-έας": "-έας",       # -έας (agent): γραμματέας, συγγραφέας
    "-άρχης": "-άρχης",   # -άρχης (ruler/chief): γυμνασιάρχης
    "-ποιός": "-ποιός",   # -ποιός (maker): υποδηματοποιός, θαλασσοπόρος

    # Adjective suffixes
    "-ικός": "-ικός",     # -ικός (adjective): φυσικός, ελληνικός
    "-ικος": "-ικος",     # -ικος (adj. variant)
    "-ιμος": "-ιμος",     # -ιμος (capable): χρήσιμος, φαγώσιμος
    "-τος": "-τος",       # -τος (-able): γραπτός, προσιτός
    "-μένος": "-μένος",   # -μένος (passive participle): γραμμένος
    "-όμενος": "-όμενος", # -όμενος (present participle): ερχόμενος
    "-ων": "-ων",         # -ων (active participle): γράφων
    "-ούχος": "-ούχος",   # -ούχος (possessing): διπλωματούχος
    "-ωδός": "-ωδός",     # -ωδός (like): αστεροειδής → μελωδός
    "-ώδης": "-ώδης",     # -ώδης (resembling): ακανθώδης
    "-ωπός": "-ωπός",     # -ωπός (looking): αγριωπός

    # Verb suffixes (common endings)
    "-ώνω": "-ώνω",
    "-άρω": "-άρω",
    "-ίζω": "-ίζω",
    "-εύω": "-εύω",
}

# Common Greek prefixes with canonical forms
GREEK_PREFIXES: Dict[str, str] = {
    "αντι-": "αντι-",     # anti-
    "αντί-": "αντι-",
    "απο-": "απο-",       # from, de-
    "από-": "απο-",
    "δια-": "δια-",       # through, inter-
    "διά-": "δια-",
    "εις-": "εις-",       # into
    "εκ-": "εκ-",         # out, ex-
    "εξ-": "εκ-",
    "εν-": "εν-",         # in, en-
    "επι-": "επι-",       # on, epi-
    "επί-": "επι-",
    "κατα-": "κατα-",     # down, against
    "κατά-": "κατα-",
    "μετα-": "μετα-",     # after, meta-
    "μετά-": "μετα-",
    "παρα-": "παρα-",     # beside, para-
    "παρά-": "παρα-",
    "περι-": "περι-",     # around, peri-
    "περί-": "περι-",
    "προ-": "προ-",       # before, pro-
    "προς-": "προς-",     # toward
    "συν-": "συν-",       # with, syn-
    "συμ-": "συν-",
    "συγ-": "συν-",
    "συλ-": "συν-",
    "υπερ-": "υπερ-",     # over, hyper-
    "υπέρ-": "υπερ-",
    "υπο-": "υπο-",       # under, hypo-
    "υπό-": "υπο-",
    "α-": "α-",           # un-, a- (privative)
    "αν-": "αν-",
    "ανα-": "ανα-",       # up, re-
    "ανά-": "ανα-",
    "ξε-": "ξε-",         # un- (reversal)
    "ξανα-": "ξανα-",     # re-, again
    "δυσ-": "δυσ-",       # dys-, difficult
    "ευ-": "ευ-",         # eu-, well
    "αρχι-": "αρχι-",     # arch-, chief
    "αρχί-": "αρχι-",
    "παν-": "παν-",       # pan-, all
    "ολο-": "ολο-",       # holo-, whole
    "ημι-": "ημι-",       # hemi-, half
    "μισο-": "μισο-",     # half-
    "τηλε-": "τηλε-",     # tele-, distant
}


def canonicalize_affix(affix: str) -> str:
    """Map an affix to its canonical form."""
    affix_lower = affix.lower().strip("-")
    # Check prefixes
    for prefix, canonical in GREEK_PREFIXES.items():
        prefix_clean = prefix.strip("-")
        if affix_lower == prefix_clean:
            return canonical.strip("-")
    # Check suffixes
    for suffix, canonical in GREEK_DERIVATIONAL_SUFFIXES.items():
        suffix_clean = suffix.strip("-")
        if affix_lower == suffix_clean or affix_lower.endswith(suffix_clean):
            return canonical.strip("-")
    return affix_lower


def decompose_greek_word(
    word: str,
    known_prefixes: Optional[Dict[str, str]] = None,
    known_suffixes: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, str]]:
    """Attempt to decompose a Greek word into (prefix, root, suffix).

    Returns a list of possible decompositions, ordered by likelihood.
    Each decomposition is (prefix_or_empty, root, suffix_or_empty).

    Args:
        word: The Greek word to decompose.
        known_prefixes: Optional extra prefix -> canonical mapping.
        known_suffixes: Optional extra suffix -> canonical mapping.
    """
    from morpho_bpe import strip_greek_accents

    clean = word.lstrip("Ġ▁ ")
    word_norm = strip_greek_accents(clean.lower())

    all_prefixes = dict(GREEK_PREFIXES)
    if known_prefixes:
        all_prefixes.update(known_prefixes)
    all_suffixes = dict(GREEK_DERIVATIONAL_SUFFIXES)
    if known_suffixes:
        all_suffixes.update(known_suffixes)

    decompositions: List[Tuple[str, str, str]] = []

    # Try prefix + root + suffix
    for prefix_form, prefix_canon in sorted(all_prefixes.items(), key=lambda x: -len(x[0])):
        prefix_clean = strip_greek_accents(prefix_canon.strip("-"))
        if word_norm.startswith(prefix_clean) and len(word_norm) > len(prefix_clean) + 2:
            rest = word_norm[len(prefix_clean):]
            for suffix_form, suffix_canon in sorted(all_suffixes.items(), key=lambda x: -len(x[0])):
                suffix_clean = strip_greek_accents(suffix_canon.strip("-"))
                if rest.endswith(suffix_clean) and len(rest) > len(suffix_clean):
                    root = rest[: -len(suffix_clean)] if suffix_clean else rest
                    if len(root) >= 2:
                        decompositions.append((prefix_clean, root, suffix_clean))
                        break  # take longest suffix match
            else:
                # Prefix only, no known suffix
                decompositions.append((prefix_clean, rest, ""))

    # Try root + suffix (no prefix)
    for suffix_form, suffix_canon in sorted(all_suffixes.items(), key=lambda x: -len(x[0])):
        suffix_clean = strip_greek_accents(suffix_canon.strip("-"))
        if word_norm.endswith(suffix_clean) and len(word_norm) > len(suffix_clean) + 2:
            root = word_norm[: -len(suffix_clean)]
            decompositions.append(("", root, suffix_clean))
            break  # take longest suffix match

    # If no decomposition found, treat whole word as root
    if not decompositions:
        decompositions.append(("", word_norm, ""))

    return decompositions


# ── Tensor composition ─────────────────────────────────────────────────

class CompositionalEmbeddingModel(torch.nn.Module):
    """Learnable tensor-composition model for compound word embeddings.

    Implements: E_new = α * E_root + β * E_suffix + γ * (E_root ⊙ E_suffix)

    Where α, β, γ are learnable per-dimension parameters (or scalars).
    """

    def __init__(self, dim: int, per_dimension: bool = True):
        super().__init__()
        self.dim = dim
        self.per_dimension = per_dimension

        if per_dimension:
            self.alpha = torch.nn.Parameter(torch.ones(dim) * 0.5)
            self.beta = torch.nn.Parameter(torch.ones(dim) * 0.5)
            self.gamma = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.alpha = torch.nn.Parameter(torch.tensor(0.5))
            self.beta = torch.nn.Parameter(torch.tensor(0.5))
            self.gamma = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        E_root: torch.Tensor,
        E_suffix: torch.Tensor,
        E_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compose compound embedding from components.

        Args:
            E_root: Root embedding (batch × dim) or (dim,).
            E_suffix: Suffix embedding.
            E_prefix: Optional prefix embedding.

        Returns:
            Composed embedding of same shape as E_root.
        """
        if E_prefix is not None:
            # If prefix is present, merge prefix+root first
            E_root = self.alpha * E_root + self.beta * E_prefix + self.gamma * (E_root * E_prefix)

        result = self.alpha * E_root + self.beta * E_suffix + self.gamma * (E_root * E_suffix)
        return result

    def compose_numpy(self, E_root: np.ndarray, E_suffix: np.ndarray) -> np.ndarray:
        """Compose using numpy (no gradients needed)."""
        with torch.no_grad():
            root_t = torch.from_numpy(E_root).float()
            suffix_t = torch.from_numpy(E_suffix).float()
            result = self.forward(root_t, suffix_t)
            return result.numpy()


def compute_compositional_embedding(
    word: str,
    ft_model,  # FastText model for component lookups
    aligner,  # GeometricAligner for space projection
    composition_model: Optional[CompositionalEmbeddingModel] = None,
    base_input_embeddings: Optional[torch.Tensor] = None,
) -> Optional[np.ndarray]:
    """Compute a compositional embedding for a compound Greek word.

    Steps:
    1. Decompose the word into (prefix, root, suffix).
    2. Look up each component's embedding.
    3. Compose: E = α·E_root + β·E_suffix + γ·(E_root ⊙ E_suffix).
    4. Project into LLM space via the aligner.

    Args:
        word: The compound word to embed.
        ft_model: FastText model for component vectors.
        aligner: Fitted GeometricAligner.
        composition_model: Optional pre-trained CompositionalEmbeddingModel.
        base_input_embeddings: Optional base model embeddings for fallback.

    Returns:
        Embedding vector in LLM space, or None if composition is impossible.
    """
    from morpho_bpe import strip_greek_accents

    decompositions = decompose_greek_word(word)

    for prefix, root, suffix in decompositions:
        # Get component embeddings from FastText
        E_root_vec = _lookup_component_embedding(root, ft_model)
        E_suffix_vec = _lookup_component_embedding(suffix, ft_model) if suffix else None
        E_prefix_vec = _lookup_component_embedding(prefix, ft_model) if prefix else None

        if E_root_vec is None:
            continue  # try next decomposition

        if E_suffix_vec is None and suffix:
            continue  # need at least root + suffix for composition

        # Compose
        root_vec = np.asarray(E_root_vec, dtype=np.float32)
        suffix_vec = np.asarray(E_suffix_vec, dtype=np.float32) if E_suffix_vec is not None else root_vec.copy()

        if composition_model is not None:
            composed = composition_model.compose_numpy(root_vec, suffix_vec)
        else:
            # Simple average with slight root bias
            composed = 0.6 * root_vec + 0.4 * suffix_vec

        # Project to LLM space
        if aligner is not None and aligner.fitted:
            return aligner.project_target_to_base(composed)
        else:
            return composed  # return in FastText space

    return None


def _lookup_component_embedding(
    component: str,
    ft_model,
) -> Optional[np.ndarray]:
    """Look up a morphological component's embedding in FastText.

    Tries multiple forms: exact, lowercase, accent-stripped, canonical.
    """
    from morpho_bpe import strip_greek_accents

    if not component or not component.strip():
        return None

    clean = component.strip("- ")
    forms_to_try = [
        clean,
        clean.lower(),
        strip_greek_accents(clean.lower()),
        canonicalize_affix(clean),
    ]

    for form in forms_to_try:
        if not form:
            continue
        try:
            if hasattr(ft_model, "get_vector"):
                vec = ft_model.get_vector(form)
                if vec is not None:
                    vec_np = np.asarray(vec, dtype=np.float32)
                    if np.any(vec_np):
                        return vec_np
        except Exception:
            continue

        # Fallback: try direct dict lookup
        if hasattr(ft_model, "word_to_vec") and form in ft_model.word_to_vec:
            return np.asarray(ft_model.word_to_vec[form], dtype=np.float32)

    return None


# ── Batch composition ──────────────────────────────────────────────────

def batch_compose_compound_embeddings(
    compound_words: Sequence[str],
    ft_model,
    aligner,
    composition_model: Optional[CompositionalEmbeddingModel] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Batch-compose embeddings for multiple compound words.

    Returns:
        (successful {word: embedding_in_llm_space}, failed_words)
    """
    successful: Dict[str, np.ndarray] = {}
    failed: List[str] = []

    for word in compound_words:
        emb = compute_compositional_embedding(
            word, ft_model, aligner, composition_model,
        )
        if emb is not None:
            successful[word] = emb
        else:
            failed.append(word)

    logger.info(
        "Compositional embedding: %d succeeded, %d failed.",
        len(successful), len(failed),
    )
    return successful, failed
