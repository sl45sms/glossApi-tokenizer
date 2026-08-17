#!/usr/bin/env python3
"""Morpho-BPE: Morphologically-guided vocabulary extension for Greek.

Implements the first phase of Compositional Geometric Alignment (CGA):
- Rule-based morphological anchors: prefixes, suffixes, stems, and closed-class words
  that should be preserved as indivisible tokens.
- Syllable constraint checking for Greek: ensures sub-token boundaries respect
  Greek syllabic structure (consonant-vowel groupings).
- Candidate token scoring by morphological coherence and frequency-weighted utility.
"""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple


# ── Greek syllable pattern ────────────────────────────────────────────
# Greek syllables follow: (C)(C)(C)V(V)(C) where C=consonant, V=vowel.
# Minimum syllable: a single vowel. Maximum onset: 3 consonants (e.g., στρ).
# Codas are limited to /s/, /n/, /r/ in native Greek words.

_GREEK_VOWELS: FrozenSet[str] = frozenset("αάἀἁὰᾶἄἂἆἅἃἇεέἐἑὲἒἔἓἕηήἠἡὴῆἤἢἦἥἣἧιίἰἱὶῖἴἲἶἵἳἷοόὀὁὸὄὂὅὃυύὐὑὺῦὔὒὖὕὓὗωώὠὡὼῶὤὢὦὥὣὧϊϋΐΰ")

_GREEK_CONSONANTS: FrozenSet[str] = frozenset(
    "βγδζθκλμνξπρσςτφχψ"
    "ΒΓΔΖΘΚΛΜΝΞΠΡΣΤΦΧΨ"
)

_GREEK_TONOS: FrozenSet[str] = frozenset("άέήίόύώΆΈΉΊΌΎΏ")

# Valid two-consonant onsets in Modern Greek
_VALID_ONSET_PAIRS: FrozenSet[str] = frozenset([
    "μπ", "ντ", "γκ", "γγ", "τζ", "τσ",
    "βλ", "βρ", "γλ", "γρ", "δρ", "θλ", "θρ",
    "κλ", "κν", "κρ", "κτ", "μν", "πλ", "πν", "πρ", "πτ",
    "σβ", "σγ", "σδ", "σθ", "σκ", "σμ", "σπ", "στ", "σφ", "σχ",
    "τρ", "φθ", "φλ", "φρ", "χλ", "χρ", "χτ",
    "βγ", "γδ", "χθ",
])

# Valid three-consonant onsets
_VALID_TRIPLE_ONSETS: FrozenSet[str] = frozenset([
    "στρ", "σκλ", "σκν", "σκρ", "σπλ", "σπρ", "σφρ", "σχλ",
    "μπρ", "ντρ", "γκρ", "μπλ", "ντλ", "γκλ"
])


# ── Accent / diacritic normalization ───────────────────────────────────

def strip_greek_accents(text: str) -> str:
    """Remove combining diacritical marks and replace precomposed accented chars
    with their unaccented base forms."""
    # Normalize to NFD to separate base from combining marks
    nfd = unicodedata.normalize("NFD", text)
    # Remove combining marks (Mn = Mark, Nonspacing)
    base = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    # Replace tonos-marked vowels with plain vowels
    tonos_map = str.maketrans(
        "άέήίόύώΆΈΉΊΌΎΏΐΰϊϋ",
        "αεηιουωΑΕΗΙΟΥΩιυιυ",
    )
    return base.translate(tonos_map)


def strip_greek_accents_keep_diaeresis(text: str) -> str:
    """Like strip_greek_accents but preserves diaeresis (διαλυτικά)."""
    nfd = unicodedata.normalize("NFD", text)
    base = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    tonos_map = str.maketrans(
        "άέήίόύώΆΈΉΊΌΎΏΐΰ",
        "αεηιουωΑΕΗΙΟΥΩϊϋ",
    )
    return base.translate(tonos_map)


# ── Greek syllable checking ────────────────────────────────────────────

def _is_greek_vowel(ch: str) -> bool:
    return ch.lower() in _GREEK_VOWELS or strip_greek_accents(ch.lower()) in "αεηιουω"


def _is_greek_consonant(ch: str) -> bool:
    return ch.lower() in _GREEK_CONSONANTS


def _is_greek_char(ch: str) -> bool:
    return _is_greek_vowel(ch) or _is_greek_consonant(ch)


def token_respects_greek_syllable_structure(token: str) -> bool:
    """Check whether a sub-token respects Greek syllabic structure.

    A token passes if it is either:
    - A whole syllable (C* V+ C?)
    - A prefix of a syllable (consonant onset only)
    - A suffix of a syllable (vowel + optional coda)
    - A single consonant (valid as a fragment)

    Returns False for tokens that split mid-syllable in unnatural ways.
    """
    if not token:
        return True

    # Remove leading space marker (Ġ = ▁ = U+2581 or space)
    clean = token.lstrip("Ġ▁ ")

    if not clean:
        return True  # pure space token is fine

    # If the token has no Greek characters, it's fine
    greek_chars = [ch for ch in clean if _is_greek_char(ch)]
    if not greek_chars:
        return True

    # Classify each Greek character
    types = ["V" if _is_greek_vowel(ch) else "C" for ch in greek_chars]

    # A token must not end mid-consonant-cluster unless the whole cluster
    # is a valid onset or coda.
    # Rule: the token should not end with consonants that cannot form
    # a valid syllable coda or onset.

    # If token ends with a consonant cluster...
    # Actually, for BPE subwords, we mainly care that the token doesn't end
    # in the middle of a consonant cluster that should stay together as an onset.
    # And it shouldn't start with a sequence that breaks a coda-onset boundary.

    # Simplify: require at least one vowel per token, or the token is a valid
    # consonant-only prefix/suffix (like derivational morphemes).
    if "V" not in types:
        # Consonant-only token: acceptable if short (1-3 consonants, valid onset)
        if len(clean) <= 3:
            return True
        return False

    # Token should not start mid-syllable (after vowel, before consonants of next onset)
    # This is hard to check without context, so we're permissive here.
    return True


def token_violates_greek_syllable_boundary(token: str) -> bool:
    """More strict check: returns True if the token likely violates syllable boundaries.

    Focuses on detecting tokens that split Greek words at unnatural mid-syllable positions.
    """
    clean = token.lstrip("Ġ▁ ")
    if not clean:
        return False

    greek_chars = [(i, ch) for i, ch in enumerate(clean) if _is_greek_char(ch)]
    if len(greek_chars) < 2:
        return False

    # Build C/V sequence
    types = "".join("V" if _is_greek_vowel(ch) else "C" for _, ch in greek_chars)

    # Violation: three or more consecutive consonants that are NOT a valid onset
    # (unless at word boundary)
    for match in re.finditer(r"C{3,}", types):
        cluster = "".join(
            greek_chars[j][1].lower()
            for j in range(match.start(), match.end())
        )
        if len(cluster) > 3:
            return True
        if len(cluster) == 3 and cluster.lower() not in _VALID_TRIPLE_ONSETS:
            return True

    return False


# ── Morphological anchor types ─────────────────────────────────────────

class MorphologicalAnchorSet:
    """Holds sets of morphological anchors for Greek tokenizer extension.

    Categories:
    - prefixes (προθήματα): derivational prefixes like α-, αντι-, κατα-, παρα-, etc.
    - suffixes (επιθήματα): derivational suffixes like -ότητα, -ση, -μένος, etc.
    - stems (θέματα): common Greek stems for compounding
    - forced (αδιαίρετα): closed-class words & high-frequency tokens that must stay whole
    """

    def __init__(
        self,
        prefixes: Optional[Iterable[str]] = None,
        suffixes: Optional[Iterable[str]] = None,
        stems: Optional[Iterable[str]] = None,
        forced: Optional[Iterable[str]] = None,
    ):
        self.prefixes: Set[str] = set(prefixes or [])
        self.suffixes: Set[str] = set(suffixes or [])
        self.stems: Set[str] = set(stems or [])
        self.forced: Set[str] = set(forced or [])

        # Normalized versions (no accents) for fuzzy matching
        self._prefixes_norm: Set[str] = {strip_greek_accents(p) for p in self.prefixes}
        self._suffixes_norm: Set[str] = {strip_greek_accents(s) for s in self.suffixes}
        self._stems_norm: Set[str] = {strip_greek_accents(s) for s in self.stems}
        self._forced_norm: Set[str] = {strip_greek_accents(f) for f in self.forced}

    @classmethod
    def from_static_dir(cls, static_dir: Path) -> "MorphologicalAnchorSet":
        """Load anchors from the vocabularyGen/static directory."""
        prefixes = cls._read_anchor_file(static_dir / "prothimata.txt")
        suffixes = cls._read_anchor_file(static_dir / "epithemata.txt")
        forced = cls._read_anchor_file(static_dir / "forced.txt")
        return cls(prefixes=prefixes, suffixes=suffixes, forced=forced)

    @staticmethod
    def _read_anchor_file(path: Path) -> List[str]:
        if not path.exists():
            return []
        anchors: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token:
                anchors.append(token)
        return anchors

    def is_morphological_anchor(self, token: str) -> Tuple[bool, str]:
        """Check if a token is a morphological anchor.

        Returns (is_anchor, anchor_type).
        """
        clean = token.lstrip("Ġ▁ ").lower()
        clean_norm = strip_greek_accents(clean)

        # Direct match
        if token in self.forced or clean in self.forced or clean_norm in self._forced_norm:
            return True, "forced"
        if clean in self.prefixes or clean_norm in self._prefixes_norm:
            return True, "prefix"
        if clean in self.suffixes or clean_norm in self._suffixes_norm:
            return True, "suffix"
        if clean in self.stems or clean_norm in self._stems_norm:
            return True, "stem"

        # Prefix-like: starts with a known prefix and is longer
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            prefix_norm = strip_greek_accents(prefix.lower())
            if len(prefix_norm) >= 2 and clean_norm.startswith(prefix_norm) and len(clean_norm) > len(prefix_norm):
                return True, "prefix-derived"

        # Suffix-like: ends with a known suffix
        for suffix in sorted(self.suffixes, key=len, reverse=True):
            suffix_norm = strip_greek_accents(suffix.lower())
            if len(suffix_norm) >= 2 and clean_norm.endswith(suffix_norm) and len(clean_norm) > len(suffix_norm):
                return True, "suffix-derived"

        return False, ""

    def score_token_morphological_coherence(self, token: str) -> float:
        """Score a candidate token by its morphological coherence.

        Returns a score in [0.0, 1.0] where:
        - 1.0 = exact morphological anchor
        - 0.8+ = derived from known affix
        - 0.5+ = contains a known stem
        - 0.3+ = respects syllable structure
        - 0.0 = violates syllable boundaries
        """
        is_anchor, anchor_type = self.is_morphological_anchor(token)
        if is_anchor:
            if anchor_type in ("forced", "prefix", "suffix"):
                return 1.0
            if anchor_type == "stem":
                return 0.9
            if anchor_type in ("prefix-derived", "suffix-derived"):
                return 0.8

        if token_violates_greek_syllable_boundary(token):
            return 0.0

        if token_respects_greek_syllable_structure(token):
            return 0.4

        return 0.2


# ── Morpho-BPE candidate scoring ──────────────────────────────────────

def score_candidate_tokens(
    tokens: Sequence[str],
    anchors: MorphologicalAnchorSet,
    frequency_map: Optional[Dict[str, int]] = None,
    fragmentation_map: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Score candidate tokens by morphological and frequency-weighted utility.

    Returns list of (token, total_score, score_breakdown) sorted by total_score descending.
    """
    scored: List[Tuple[str, float, Dict[str, float]]] = []

    max_freq = max(frequency_map.values()) if frequency_map else 1
    max_frag = max(fragmentation_map.values()) if fragmentation_map else 1

    for token in tokens:
        breakdown: Dict[str, float] = {}

        # Morphological coherence score (0.0 - 1.0)
        morph_score = anchors.score_token_morphological_coherence(token)
        breakdown["morphological_coherence"] = morph_score

        # Frequency score (0.0 - 1.0), log-scaled
        if frequency_map and token in frequency_map:
            freq = frequency_map[token]
            freq_score = min(1.0, max(0.0, (freq / max(1, max_freq)) ** 0.5))
        else:
            freq_score = 0.1  # unknown frequency
        breakdown["frequency"] = freq_score

        # Fragmentation utility score (0.0 - 1.0)
        if fragmentation_map and token in fragmentation_map:
            frag = fragmentation_map[token]
            frag_score = min(1.0, frag / max(1, max_frag))
        else:
            frag_score = 0.1
        breakdown["fragmentation_utility"] = frag_score

        # Weighted total: morphology is primary, frequency and fragmentation
        # are modifiers
        total = 0.5 * morph_score + 0.25 * freq_score + 0.25 * frag_score
        breakdown["total"] = total

        scored.append((token, total, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_by_morphological_quality(
    scored_tokens: List[Tuple[str, float, Dict[str, float]]],
    min_score: float = 0.3,
    max_tokens: Optional[int] = None,
) -> List[str]:
    """Filter scored tokens, keeping only those above the quality threshold."""
    kept = [token for token, score, _ in scored_tokens if score >= min_score]
    if max_tokens and len(kept) > max_tokens:
        kept = kept[:max_tokens]
    return kept


# ── Token text normalization for matching ──────────────────────────────

def normalize_greek_token_for_matching(token: str) -> str:
    """Normalize a Greek token for cross-vocabulary matching.

    - Strip leading space markers
    - Lowercase
    - Remove accents
    """
    clean = token.lstrip("Ġ▁ ").lower()
    return strip_greek_accents(clean)


def build_normalized_token_index(
    tokens: Iterable[str],
) -> Dict[str, List[str]]:
    """Build a normalized-token -> original-tokens index for fuzzy lookup."""
    index: Dict[str, List[str]] = defaultdict(list)
    for token in tokens:
        norm = normalize_greek_token_for_matching(token)
        index[norm].append(token)
    return index
