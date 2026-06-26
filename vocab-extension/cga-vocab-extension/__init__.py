"""Compositional Geometric Alignment (CGA) for vocabulary extension.

A morphology-aware, mathematically principled approach to extending
an LLM's tokenizer for morphologically rich languages like Greek.

Modules:
    morpho_bpe              - Morphological anchors & syllable constraints
    fasttext_utils          - FastText vector loading & anchor extraction
    geometric_alignment     - Orthogonal Procrustes alignment & projection
    compositional_residuals - Morphological tensor for compound words
    cga_pipeline            - Full CGA pipeline CLI

Note: The directory is named 'vocab-extension' (with hyphen) so Python
cannot do `import vocab_extension`. The modules use sys.path injection
so they can be imported via `from morpho_bpe import ...` after adding
this directory to sys.path.
"""

import sys
from pathlib import Path

# Ensure this directory is on sys.path for cross-module imports
_VOCAB_EXT_DIR = Path(__file__).resolve().parent
if str(_VOCAB_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_VOCAB_EXT_DIR))

from morpho_bpe import (  # noqa: E402
    MorphologicalAnchorSet,
    score_candidate_tokens,
    filter_by_morphological_quality,
)

from fasttext_utils import (  # noqa: E402
    FastTextVectorModel,
    load_greek_fasttext,
    extract_anchor_tokens,
    build_anchor_embeddings,
)

from geometric_alignment import (  # noqa: E402
    GeometricAligner,
    initialize_new_embeddings_cga,
    apply_cga_to_model_embeddings,
)

from compositional_residuals import (  # noqa: E402
    CompositionalEmbeddingModel,
    decompose_greek_word,
    compute_compositional_embedding,
    batch_compose_compound_embeddings,
)

__all__ = [
    # morpho_bpe
    "MorphologicalAnchorSet",
    "score_candidate_tokens",
    "filter_by_morphological_quality",
    # fasttext_utils
    "FastTextVectorModel",
    "load_greek_fasttext",
    "extract_anchor_tokens",
    "build_anchor_embeddings",
    # geometric_alignment
    "GeometricAligner",
    "initialize_new_embeddings_cga",
    "apply_cga_to_model_embeddings",
    # compositional_residuals
    "CompositionalEmbeddingModel",
    "decompose_greek_word",
    "compute_compositional_embedding",
    "batch_compose_compound_embeddings",
]
