from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging as hf_logging


MISTRAL_REGEX_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_MISTRAL_REGEX_WARNING_FRAGMENT = "with an incorrect regex pattern"
_MISTRAL_REGEX_LOGGER = "transformers.tokenization_utils_tokenizers"
_MISTRAL_LIKE_MODEL_TYPES = {"apertus", "mistral", "mistral3", "voxtral", "ministral", "pixtral"}


class _MistralRegexWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _MISTRAL_REGEX_WARNING_FRAGMENT not in record.getMessage()


@contextmanager
def suppress_mistral_regex_warning():
    logger = logging.getLogger(_MISTRAL_REGEX_LOGGER)
    warning_filter = _MistralRegexWarningFilter()
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)
        hf_logging.set_verbosity(previous_verbosity)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_compatible_fast_tokenizer(model_ref: str):
    model_path = Path(model_ref)
    tokenizer_file = model_path / "tokenizer.json"
    tokenizer_config = _read_json(model_path / "tokenizer_config.json")
    if not tokenizer_file.exists():
        raise SystemExit(
            f"Tokenizer metadata references TokenizersBackend, but {tokenizer_file} was not found."
        )

    compatible_kwargs: Dict[str, Any] = {
        "tokenizer_file": str(tokenizer_file),
    }
    for key in (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
        "add_prefix_space",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "model_input_names",
    ):
        if key in tokenizer_config:
            compatible_kwargs[key] = tokenizer_config[key]

    tokenizer = PreTrainedTokenizerFast(**compatible_kwargs)
    chat_template_path = model_path / "chat_template.jinja"
    if chat_template_path.exists():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def _model_type_for(model_ref: str) -> str | None:
    model_path = Path(model_ref)
    if not model_path.exists():
        return None
    return _read_json(model_path / "config.json").get("model_type")


def _is_mistral_regex_already_fixed(tokenizer) -> bool:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    if pre_tokenizer is None:
        return False
    return MISTRAL_REGEX_PATTERN in repr(pre_tokenizer)


def maybe_fix_mistral_regex(tokenizer, model_ref: str):
    if not hasattr(tokenizer, "backend_tokenizer"):
        return tokenizer

    try:
        vocab_size = len(tokenizer)
    except TypeError:
        return tokenizer

    model_type = _model_type_for(model_ref)
    if vocab_size <= 100000:
        return tokenizer
    if model_type is not None and model_type not in _MISTRAL_LIKE_MODEL_TYPES:
        return tokenizer

    if _is_mistral_regex_already_fixed(tokenizer):
        setattr(tokenizer, "fix_mistral_regex", True)
        return tokenizer

    try:
        import tokenizers
    except ImportError:
        return tokenizer

    split_pretokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(MISTRAL_REGEX_PATTERN),
        behavior="isolated",
    )
    current_pretokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    try:
        if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
            tokenizer.backend_tokenizer.pre_tokenizer[0] = split_pretokenizer
        else:
            if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Metaspace):
                current_pretokenizer = tokenizers.pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    use_regex=False,
                )
            tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
                [split_pretokenizer, current_pretokenizer]
            )
    except Exception:
        return tokenizer

    setattr(tokenizer, "fix_mistral_regex", True)
    return tokenizer


def load_repo_tokenizer(model_ref: str, trust_remote_code: bool = False):
    with suppress_mistral_regex_warning():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as exc:
            if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
                raise
            tokenizer = _load_compatible_fast_tokenizer(model_ref)
    return maybe_fix_mistral_regex(tokenizer, model_ref)