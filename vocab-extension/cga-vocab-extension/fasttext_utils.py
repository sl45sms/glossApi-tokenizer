#!/usr/bin/env python3
"""FastText utilities for Compositional Geometric Alignment.

Handles:
- Downloading/loading Facebook fastText Greek word vectors (cc.el.300)
- Building embedding matrices for anchor tokens and new candidate tokens
- OOV handling via fastText subword composition (using the fasttext package)
  or via character n-gram fallback when only word vectors are available
- Caching downloaded models in a configurable directory
"""

import gzip
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow cross-module imports within vocab-extension (hyphen in dir name prevents package imports)
_VOCAB_EXT_DIR = Path(__file__).resolve().parent
if str(_VOCAB_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_VOCAB_EXT_DIR))

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

FASTTEXT_GREEK_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.vec.gz"
FASTTEXT_GREEK_BIN_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.bin.gz"
FASTTEXT_DIM = 300
DEFAULT_CACHE_DIR = Path(os.environ.get("SCRATCH", Path.home() / ".cache")) / "fasttext"

# Common Greek words guaranteed to be in any large Greek corpus embedding
# These serve as fallback anchors when other matches fail.
_GREEK_FALLBACK_ANCHORS: List[str] = [
    "και", "είναι", "δεν", "από", "για", "με", "σε", "το", "τα", "την",
    "του", "της", "των", "τους", "ένα", "μια", "έχει", "ήταν", "ότι",
    "όπως", "αυτό", "αυτή", "αυτός", "πολύ", "άλλα", "άλλες", "θα", "να",
    "άνθρωπος", "κόσμος", "χώρα", "πόλη", "γλώσσα", "λόγος", "χρόνος",
    "μέρα", "ζωή", "νερό", "φως", "αγάπη", "δύναμη", "αλήθεια", "γνώση",
]


# ── Model loading ──────────────────────────────────────────────────────

class FastTextVectorModel:
    """A simple fastText word-vector model loaded from a .vec or .vec.gz file.

    Stores word -> vector mapping and provides lookup with optional subword fallback.
    """

    def __init__(self):
        self.word_to_vec: Dict[str, np.ndarray] = {}
        self.dim: int = 0
        self.num_words: int = 0

    def load_vec(self, path: Path) -> None:
        """Load word vectors from a .vec or .vec.gz text file."""
        open_fn = gzip.open if path.suffix == ".gz" else open

        with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split()
            self.num_words = int(header[0])
            self.dim = int(header[1])

            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) < self.dim + 1:
                    continue  # skip malformed lines
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1 : self.dim + 1]], dtype=np.float32)
                except ValueError:
                    continue
                self.word_to_vec[word] = vec

        logger.info(
            "Loaded %d word vectors (dim=%d) from %s",
            len(self.word_to_vec),
            self.dim,
            path,
        )

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_vec

    def is_in_vocab(self, word: str) -> bool:
        return word in self.word_to_vec

    def get_words(self) -> List[str]:
        return list(self.word_to_vec.keys())

    def __len__(self) -> int:
        return len(self.word_to_vec)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get the vector for a word, or None if OOV."""
        return self.word_to_vec.get(word)

    def get_vectors_batch(
        self,
        words: Sequence[str],
        oov_strategy: str = "zero",
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Get vectors for a batch of words.

        Args:
            words: Words to look up.
            oov_strategy: 'zero' (zero vector), 'skip' (omit), or 'mean' (mean of all vectors).

        Returns:
            (matrix N×dim, found_words, oov_words)
        """
        vectors: List[np.ndarray] = []
        found: List[str] = []
        oov: List[str] = []

        for word in words:
            vec = self.get_vector(word)
            if vec is not None:
                vectors.append(vec)
                found.append(word)
            else:
                oov.append(word)
                if oov_strategy == "zero":
                    vectors.append(np.zeros(self.dim, dtype=np.float32))
                    found.append(word)  # count as found for matrix alignment
                # 'skip' just doesn't add

        if not vectors:
            return np.array([], dtype=np.float32).reshape(0, self.dim), found, oov

        return np.stack(vectors, axis=0), found, oov


class FastTextSubwordModel:
    """Wrapper around the `fasttext` Python package for subword-aware lookups.

    This can produce vectors for any string, not just known words.
    """

    def __init__(self):
        self._model = None

    def load_bin(self, path: Path) -> None:
        """Load a fastText .bin model (includes subword information)."""
        try:
            import fasttext  # type: ignore
        except ImportError:
            raise ImportError(
                "The 'fasttext' package is required for subword-aware FastText. "
                "Install it with: pip install fasttext"
            )
        self._model = fasttext.load_model(str(path))
        self.dim = self._model.get_dimension()
        logger.info("Loaded fastText subword model from %s (dim=%d)", path, self.dim)

    def get_vector(self, word: str) -> np.ndarray:
        """Get the vector for any word using subword composition."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_bin() first.")
        return np.array(self._model.get_word_vector(word), dtype=np.float32)

    def is_in_vocab(self, word: str) -> bool:
        if self._model is None:
            return False
        return self._model.get_word_id(word) != -1

    def get_words(self) -> List[str]:
        if self._model is None:
            return []
        return self._model.get_words()

    def __contains__(self, word: str) -> bool:
        # Subword models can produce vectors for any string
        return True


# ── Download helpers ───────────────────────────────────────────────────

def _download_fasttext_greek_vec(download_dir: Path) -> Path:
    """Download the Greek fastText .vec.gz file if not already present."""
    import urllib.request

    download_dir.mkdir(parents=True, exist_ok=True)
    dest = download_dir / "cc.el.300.vec.gz"

    if dest.exists():
        logger.info("FastText Greek vectors already at %s", dest)
        return dest

    logger.info("Downloading Greek fastText vectors from %s ...", FASTTEXT_GREEK_URL)
    urllib.request.urlretrieve(FASTTEXT_GREEK_URL, dest)
    logger.info("Downloaded to %s", dest)
    return dest


def _download_fasttext_greek_bin(download_dir: Path) -> Path:
    """Download the Greek fastText .bin.gz file if not already present."""
    import urllib.request

    download_dir.mkdir(parents=True, exist_ok=True)
    dest_compressed = download_dir / "cc.el.300.bin.gz"
    dest = download_dir / "cc.el.300.bin"

    if dest.exists():
        logger.info("FastText Greek binary model already at %s", dest)
        return dest

    if not dest_compressed.exists():
        logger.info("Downloading Greek fastText binary model from %s ...", FASTTEXT_GREEK_BIN_URL)
        urllib.request.urlretrieve(FASTTEXT_GREEK_BIN_URL, dest_compressed)

    logger.info("Decompressing %s ...", dest_compressed)
    import gzip
    import shutil

    with gzip.open(dest_compressed, "rb") as f_in:
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info("Decompressed to %s", dest)
    return dest


# ── Main loader ────────────────────────────────────────────────────────

def load_greek_fasttext(
    cache_dir: Optional[Path] = None,
    use_subword: bool = False,
) -> FastTextVectorModel:
    """Load Greek fastText word vectors.

    Args:
        cache_dir: Directory to cache downloaded models. Defaults to $SCRATCH/fasttext or ~/.cache/fasttext.
        use_subword: If True, attempt to load the .bin model for subword awareness.
                     Requires the `fasttext` Python package.

    Returns:
        A FastTextVectorModel (or FastTextSubwordModel if use_subword=True) with Greek vectors loaded.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    if use_subword:
        bin_path = _download_fasttext_greek_bin(cache_dir)
        model = FastTextSubwordModel()
        model.load_bin(bin_path)
        return model  # type: ignore[return-value]

    vec_path = _download_fasttext_greek_vec(cache_dir)
    model = FastTextVectorModel()
    model.load_vec(vec_path)
    return model


# ── Anchor extraction ──────────────────────────────────────────────────

def extract_anchor_tokens(
    base_vocab: Sequence[str],
    ft_model: FastTextVectorModel,
    extra_anchors: Optional[Sequence[str]] = None,
    min_anchor_count: int = 100,
) -> List[str]:
    """Find tokens that exist in both the base tokenizer vocabulary and FastText.

    Args:
        base_vocab: All token strings from the base tokenizer.
        ft_model: Loaded FastText vector model.
        extra_anchors: Additional tokens to force-include as anchors.
        min_anchor_count: Minimum number of anchors required. If not enough natural
                         overlaps are found, fallback Greek anchors are added.

    Returns:
        List of anchor tokens common to both spaces.
    """
    from morpho_bpe import normalize_greek_token_for_matching

    # Build normalized index for base vocab
    base_set: Dict[str, str] = {}  # normalized -> original
    for token in base_vocab:
        norm = normalize_greek_token_for_matching(token)
        base_set[norm] = token

    # Find overlaps
    anchors: List[str] = []
    ft_words = ft_model.get_words()

    for ft_word in ft_words:
        norm = normalize_greek_token_for_matching(ft_word)
        if norm in base_set:
            anchors.append(base_set[norm])  # use the base tokenizer's form

    # Add extra anchors if provided
    if extra_anchors:
        for anchor in extra_anchors:
            if anchor not in anchors:
                anchors.append(anchor)

    # If we don't have enough natural overlaps, add fallback Greek anchors
    if len(anchors) < min_anchor_count:
        logger.warning(
            "Only %d natural overlaps found (need %d). Adding fallback Greek anchors.",
            len(anchors),
            min_anchor_count,
        )
        for fallback in _GREEK_FALLBACK_ANCHORS:
            norm = normalize_greek_token_for_matching(fallback)
            # Check if the fallback word exists in FastText
            if fallback in ft_model and norm in base_set:
                token = base_set[norm]
                if token not in anchors:
                    anchors.append(token)

    logger.info("Extracted %d anchor tokens common to base vocab and FastText.", len(anchors))
    return anchors


def build_anchor_embeddings(
    anchors: Sequence[str],
    ft_model: FastTextVectorModel,
    ft_token_to_vec: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build the FastText embedding matrix Y for anchor tokens.

    Args:
        anchors: Anchor token strings.
        ft_model: FastText vector model.
        ft_token_to_vec: Optional pre-computed mapping for non-word tokens.

    Returns:
        (Y matrix N×D, found_anchors, missing_anchors)
    """
    if ft_token_to_vec is None:
        ft_token_to_vec = {}

    from morpho_bpe import normalize_greek_token_for_matching

    vectors: List[np.ndarray] = []
    found: List[str] = []
    missing: List[str] = []

    for anchor in anchors:
        clean = anchor.lstrip("Ġ▁ ").lower()
        vec = None

        # Try direct lookup
        if clean in ft_model:
            vec = ft_model.get_vector(clean)
        # Try with accent-stripped form
        if vec is None:
            from morpho_bpe import strip_greek_accents
            norm = strip_greek_accents(clean)
            if norm in ft_model:
                vec = ft_model.get_vector(norm)
        # Try pre-computed mapping
        if vec is None and clean in ft_token_to_vec:
            vec = ft_token_to_vec[clean]

        if vec is not None:
            vectors.append(vec)
            found.append(anchor)
        else:
            missing.append(anchor)

    if not vectors:
        return np.array([], dtype=np.float32).reshape(0, ft_model.dim), found, missing

    Y = np.stack(vectors, axis=0)
    return Y, found, missing
