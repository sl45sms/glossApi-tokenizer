#!/usr/bin/env python3

import argparse
import html
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


TEXTBOX_SIGNATURE = inspect.signature(gr.Textbox.__init__)
TEXTBOX_SUPPORTS_BUTTONS = "buttons" in TEXTBOX_SIGNATURE.parameters
TEXTBOX_SUPPORTS_SHOW_COPY_BUTTON = "show_copy_button" in TEXTBOX_SIGNATURE.parameters
BLOCKS_SUPPORTS_CSS = "css" in inspect.signature(gr.Blocks.__init__).parameters
BLOCKS_LAUNCH_SUPPORTS_SHOW_API = "show_api" in inspect.signature(gr.Blocks.launch).parameters

DEFAULT_CAPSTOR_MODEL_PATH = "/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

CUSTOM_CSS = """
:root {
    --surface-0: #f5efe2;
    --surface-1: rgba(255, 250, 241, 0.88);
    --surface-2: rgba(245, 233, 210, 0.92);
    --ink-0: #201715;
    --ink-1: #59463f;
    --ink-2: #7b655d;
    --hero-ink: #2a1c17;
    --editor-bg: #1d1f25;
    --editor-bg-soft: #252831;
    --editor-ink: #f6efe2;
    --editor-muted: #b7ab98;
    --editor-border: rgba(246, 239, 226, 0.14);
    --accent: #0c7c59;
    --accent-soft: rgba(12, 124, 89, 0.14);
    --shadow-soft: 0 24px 70px rgba(63, 31, 18, 0.12);
    --radius-xl: 24px;
    --radius-lg: 18px;
    --font-sans: "IBM Plex Sans", "Noto Sans", "Liberation Sans", sans-serif;
    --font-serif: "Source Serif 4", "Iowan Old Style", Georgia, serif;
}

body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(12, 124, 89, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(197, 114, 61, 0.12), transparent 26%),
        linear-gradient(180deg, #f8f2e7 0%, #efe6d5 100%);
    color: var(--ink-0);
    font-family: var(--font-sans);
}

.gradio-container {
    max-width: 1440px !important;
    padding: 24px 20px 40px !important;
}

.app-shell {
    background: linear-gradient(180deg, rgba(255, 252, 246, 0.86), rgba(255, 249, 239, 0.94));
    border: 1px solid rgba(89, 70, 63, 0.12);
    border-radius: 32px;
    box-shadow: var(--shadow-soft);
    overflow: hidden;
    padding: 20px;
    backdrop-filter: blur(14px);
}

.hero-panel {
    background:
        linear-gradient(140deg, rgba(12, 124, 89, 0.10), transparent 32%),
        linear-gradient(220deg, rgba(197, 114, 61, 0.12), transparent 36%),
        #fffaf0;
    border: 1px solid rgba(89, 70, 63, 0.10);
    border-radius: 28px;
    padding: 28px 24px 20px 24px;
    margin-bottom: 18px;
}

.hero-panel h1 {
    font-family: var(--font-serif);
    font-size: clamp(2rem, 4vw, 3.5rem);
    line-height: 0.96;
    letter-spacing: -0.03em;
    margin: 0 0 8px 0;
    color: var(--hero-ink) !important;
}

.hero-panel p,
.hero-panel li,
.meta-strip,
.status-strip {
    color: var(--ink-1);
    font-size: 0.97rem;
    line-height: 1.55;
}

.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.meta-card {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(89, 70, 63, 0.12);
    border-radius: 18px;
    padding: 14px 16px;
}

.meta-card strong {
    display: block;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    color: #6f554b;
}

.panel-card {
    background: var(--surface-1);
    border: 1px solid rgba(89, 70, 63, 0.12);
    border-radius: var(--radius-xl);
    padding: 16px;
    box-shadow: 0 12px 30px rgba(51, 28, 16, 0.07);
    margin-top: 12px;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
}

.probe-card {
    background:
        linear-gradient(180deg, rgba(12, 124, 89, 0.08), transparent 42%),
        var(--surface-1);
    border-color: rgba(12, 124, 89, 0.26);
}

.console-card {
    background:
        linear-gradient(180deg, rgba(197, 114, 61, 0.08), transparent 42%),
        var(--surface-1);
}

.section-title {
    font-family: var(--font-serif);
    font-size: 1.45rem;
    margin: 6px 0 2px 0;
    color: var(--hero-ink) !important;
}

.lane-chip {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 8px;
    background: var(--accent-soft);
    color: var(--accent);
}

.status-strip {
    background: rgba(255, 255, 255, 0.66);
    border: 1px solid rgba(89, 70, 63, 0.10);
    border-radius: 16px;
    padding: 10px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
}

.probe-library {
    display: grid;
    gap: 10px;
    margin-top: 12px;
}

.probe-library-item {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(89, 70, 63, 0.10);
    border-radius: 16px;
    padding: 12px 14px;
}

.probe-library-item strong {
    display: block;
    color: var(--hero-ink);
    margin-bottom: 4px;
}

.history-stack {
    display: grid;
    gap: 12px;
}

.history-card {
    background: rgba(255, 255, 255, 0.66);
    border: 1px solid rgba(89, 70, 63, 0.10);
    border-radius: 18px;
    padding: 14px;
}

.history-head {
    font-family: var(--font-serif);
    font-size: 1.08rem;
    color: var(--hero-ink);
    margin-bottom: 8px;
}

.history-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-2);
    margin-top: 10px;
    margin-bottom: 6px;
}

.history-card pre,
.history-empty {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: var(--font-sans);
    line-height: 1.5;
    color: var(--ink-0);
}

.history-empty {
    padding: 16px;
    background: rgba(255, 255, 255, 0.62);
    border: 1px dashed rgba(89, 70, 63, 0.16);
    border-radius: 18px;
}

.gr-button-primary {
    background: linear-gradient(135deg, #1d8c68, #0c7c59) !important;
    border: 0 !important;
    box-shadow: 0 10px 26px rgba(12, 124, 89, 0.18) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #d37b2f, #c5723d) !important;
    border: 0 !important;
    box-shadow: 0 10px 26px rgba(197, 114, 61, 0.18) !important;
    color: white !important;
}

.gr-button {
    border-radius: 999px !important;
    min-height: 44px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}

textarea,
input,
.gr-textbox textarea,
.gr-textbox input {
    font-family: var(--font-sans) !important;
    background: linear-gradient(180deg, var(--editor-bg-soft), var(--editor-bg)) !important;
    color: var(--editor-ink) !important;
    border: 1px solid var(--editor-border) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 8px 22px rgba(8, 10, 14, 0.18) !important;
    caret-color: var(--editor-ink) !important;
}

textarea::placeholder,
input::placeholder,
.gr-textbox textarea::placeholder,
.gr-textbox input::placeholder {
    color: var(--editor-muted) !important;
    opacity: 1 !important;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose strong,
.gradio-container label,
.gradio-container .form label,
.gradio-container .gr-markdown,
.gradio-container .gr-accordion summary,
.gradio-container .gr-accordion button,
.gradio-container .gr-accordion * {
    color: var(--hero-ink);
}

footer {
    display: none !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 14px 12px 24px !important;
    }

    .app-shell {
        padding: 14px;
    }

    .hero-panel {
        padding: 20px 18px 16px 18px;
    }
}
"""

EVALUATION_SCENARIOS: List[Dict[str, str]] = [
    {
        "title": "Greek Explainer",
        "focus": "Greek fluency, domain relevance, and concise explanation.",
        "prompt": (
            "Απάντησε στα ελληνικά σε 4 σύντομες προτάσεις. "
            "Τι κερδίζει ένα γλωσσικό μοντέλο όταν ο tokenizer του χειρίζεται "
            "καλύτερα συχνές ελληνικές λέξεις και μορφήματα;"
        ),
    },
    {
        "title": "Instruction Fidelity",
        "focus": "Checks whether the model follows exact formatting and length constraints.",
        "prompt": (
            "Δώσε ακριβώς 3 bullets στα ελληνικά. Κάθε bullet να έχει 8 έως 12 λέξεις. "
            "Θέμα: τι ελέγχουμε σε ένα CPT smoke test;"
        ),
    },
    {
        "title": "Structured JSON",
        "focus": "Checks structured output and whether Greek content survives strict formatting.",
        "prompt": (
            "Απάντησε μόνο με valid JSON. Χρησιμοποίησε τα πεδία task, risk, mitigation. "
            "Θέμα: φόρτωση ενός checkpoint μετά από CPT."
        ),
    },
    {
        "title": "English Anchor",
        "focus": "Checks whether English capability still holds after Greek-focused CPT.",
        "prompt": (
            "Answer in English with 4 concise sentences: what is continued pretraining, "
            "and why can it help domain adaptation without changing the model architecture?"
        ),
    },
    {
        "title": "Bilingual Response",
        "focus": "Checks clean switching between Greek and English in one response.",
        "prompt": (
            "Δώσε πρώτα 2 προτάσεις στα ελληνικά και μετά 2 προτάσεις στα αγγλικά. "
            "Εξήγησε τι είναι το gradient accumulation και γιατί είναι χρήσιμο όταν η GPU μνήμη είναι περιορισμένη."
        ),
    },
    {
        "title": "Uncertainty Guardrail",
        "focus": "Checks whether the model avoids inventing facts for unknown information.",
        "prompt": (
            "Αν δεν έχεις επαρκή στοιχεία, πες το καθαρά και μην επινοήσεις λεπτομέρειες. "
            "Ερώτηση: Ποια ήταν τα ακριβή έσοδα του φανταστικού οργανισμού 'Αστερισμός Παιδείας' τον Μάρτιο του 2025;"
        ),
    },
]

SCENARIOS_BY_TITLE = {scenario["title"]: scenario for scenario in EVALUATION_SCENARIOS}
DEFAULT_SCENARIO_TITLE = EVALUATION_SCENARIOS[0]["title"]


def discover_default_model() -> str:
    env_model = os.environ.get("APERTUS_MODEL_UI_MODEL_PATH")
    if env_model:
        return env_model

    user_name = os.environ.get("USER", "")
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    scratch_dir = os.environ.get("SCRATCH")
    if scratch_dir:
        candidates.append(Path(scratch_dir) / "apertus-greek-cpt" / "final")

    if user_name:
        candidates.append(Path("/capstor/scratch/cscs") / user_name / "apertus-greek-cpt" / "final")

    candidates.extend(
        [
            Path(DEFAULT_CAPSTOR_MODEL_PATH),
            repo_root / "artifacts" / "checkpoints" / "apertus-greek-cpt-smoke" / "final",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def default_dtype(device: str) -> str:
    if device.startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return "float32"


def resolve_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def format_path(value: str) -> str:
    return html.escape(value or "-")


def textbox_copy_kwargs() -> Dict[str, Any]:
    if TEXTBOX_SUPPORTS_BUTTONS:
        return {"buttons": ["copy"]}
    if TEXTBOX_SUPPORTS_SHOW_COPY_BUTTON:
        return {"show_copy_button": True}
    return {}


def load_tokenizer(model_ref: str, trust_remote_code: bool):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise

        model_path = Path(model_ref)
        tokenizer_file = model_path / "tokenizer.json"
        tokenizer_config_path = model_path / "tokenizer_config.json"
        if not tokenizer_file.exists():
            raise SystemExit(
                f"Tokenizer metadata references TokenizersBackend, but {tokenizer_file} was not found."
            ) from exc

        tokenizer_config: Dict[str, Any] = {}
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

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

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def render_probe_details(title: str) -> str:
    scenario = SCENARIOS_BY_TITLE[title]
    return (
        f"### {scenario['title']}\n"
        f"**What to inspect:** {scenario['focus']}\n\n"
        "**Ready-made question**\n\n"
        f"```text\n{scenario['prompt']}\n```"
    )


def render_probe_library() -> str:
    items = []
    for scenario in EVALUATION_SCENARIOS:
        items.append(
            "<div class='probe-library-item'>"
            f"<strong>{html.escape(scenario['title'])}</strong>"
            f"<div>{html.escape(scenario['focus'])}</div>"
            "</div>"
        )
    return "<div class='probe-library'>" + "".join(items) + "</div>"


def render_history(entries: List[Dict[str, str]]) -> str:
    if not entries:
        return (
            "<div class='history-empty'>"
            "No evaluation runs yet. The log shown here is for review only; prompts are sent statelessly."
            "</div>"
        )

    cards = []
    for entry in reversed(entries):
        cards.append(
            "<div class='history-card'>"
            f"<div class='history-head'>Run {html.escape(entry['index'])} - {html.escape(entry['label'])}</div>"
            f"<div class='meta-strip'>{html.escape(entry['summary'])}</div>"
            "<div class='history-label'>Prompt</div>"
            f"<pre>{html.escape(entry['prompt'])}</pre>"
            "<div class='history-label'>Response</div>"
            f"<pre>{html.escape(entry['response'])}</pre>"
            "</div>"
        )
    return "<div class='history-stack'>" + "".join(cards) + "</div>"


@dataclass
class GenerationResult:
    text: str
    seconds: float
    device: str
    token_count: int


@dataclass
class ModelRuntime:
    label: str
    model_ref: str
    preferred_device: str
    dtype_name: str
    trust_remote_code: bool
    attn_implementation: Optional[str] = None
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    generate_lock: threading.Lock = field(default_factory=threading.Lock)

    def _device_map(self):
        if not self.preferred_device or self.preferred_device == "auto":
            return "auto"
        return {"": self.preferred_device}

    def ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        with self.load_lock:
            if self.model is not None and self.tokenizer is not None:
                return

            tokenizer = load_tokenizer(self.model_ref, self.trust_remote_code)
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": resolve_dtype(self.dtype_name),
                "low_cpu_mem_usage": True,
                "device_map": self._device_map(),
            }
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation

            model = AutoModelForCausalLM.from_pretrained(self.model_ref, **load_kwargs)
            model.eval()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True

            self.tokenizer = tokenizer
            self.model = model

    def primary_device(self) -> str:
        if self.model is None:
            return self.preferred_device or "cpu"

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            for device_name in hf_device_map.values():
                if isinstance(device_name, str):
                    return device_name

        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:
            return self.preferred_device or "cpu"

    def _encode_prompt(self, prompt: str):
        assert self.tokenizer is not None

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                pass

        return self.tokenizer(prompt, return_tensors="pt")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("The prompt is empty.")

        self.ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None

        start_time = time.perf_counter()
        with self.generate_lock:
            encoded = self._encode_prompt(prompt)
            model_device = self.primary_device()
            encoded = encoded.to(model_device)
            input_ids = encoded["input_ids"]

            generate_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if "attention_mask" in encoded:
                generate_kwargs["attention_mask"] = encoded["attention_mask"]
            if temperature > 0:
                generate_kwargs["temperature"] = float(temperature)
                generate_kwargs["top_p"] = float(top_p)

            with torch.inference_mode():
                output_ids = self.model.generate(**generate_kwargs)

            generated_ids = output_ids[0][input_ids.shape[-1] :]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not text:
                text = "[empty response]"

        elapsed = time.perf_counter() - start_time
        return GenerationResult(
            text=text,
            seconds=elapsed,
            device=model_device,
            token_count=int(generated_ids.shape[-1]),
        )


def build_status_message(runtime_label: str, message: str) -> str:
    return f"<div class='status-strip'><strong>{html.escape(runtime_label)}</strong> {message}</div>"


def run_generation(
    runtime: ModelRuntime,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    history: List[Dict[str, str]],
    label: str,
):
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Write a prompt first.")

    next_history = list(history or [])
    try:
        result = runtime.generate(prompt, max_new_tokens, temperature, top_p)
        tokens_per_second = result.token_count / result.seconds if result.seconds > 0 else 0.0
        status = build_status_message(
            runtime.label,
            f"completed in {result.seconds:.2f}s on {html.escape(result.device)} for "
            f"{result.token_count} new tokens ({tokens_per_second:.1f} tok/s).",
        )
        response_text = result.text
        summary = (
            f"{label} | {result.token_count} tokens | {result.seconds:.2f}s | "
            f"{tokens_per_second:.1f} tok/s | {result.device}"
        )
    except Exception as exc:
        response_text = f"[ERROR]\n{exc}"
        status = build_status_message(runtime.label, f"failed: {html.escape(str(exc))}")
        summary = f"{label} | error | {exc}"

    next_history.append(
        {
            "index": str(len(next_history) + 1),
            "label": label,
            "prompt": prompt,
            "response": response_text,
            "summary": summary,
        }
    )
    return response_text, status, next_history, render_history(next_history)


def selected_probe_prompt(title: str) -> str:
    return SCENARIOS_BY_TITLE[title]["prompt"]


def run_selected_probe(
    runtime: ModelRuntime,
    title: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    history: List[Dict[str, str]],
):
    prompt = selected_probe_prompt(title)
    response, status, next_history, history_html = run_generation(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        history=history,
        label=title,
    )
    return prompt, response, status, next_history, history_html


def clear_current_view():
    return "", ""


def clear_history():
    return [], render_history([])


def build_app(args: argparse.Namespace):
    runtime = ModelRuntime(
        label=args.model_label,
        model_ref=args.model_path,
        preferred_device=args.device,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    hero_html = f"""
    <div class='hero-panel'>
        <div class='lane-chip'>Single-model CPT evaluation</div>
        <h1>Apertus CPT Console</h1>
        <p>
            This UI is focused on post-training inspection of one tokenizer-aligned CPT checkpoint.
            The preset probes are meant to stress Greek fluency, instruction following, structured output,
            English retention, bilingual switching, and uncertainty handling. The freeform prompt box stays editable,
            so you can replace any preset with your own test immediately.
        </p>
        <div class='meta-grid'>
            <div class='meta-card'>
                <strong>Checkpoint</strong>
                <div class='meta-strip'>{format_path(args.model_path)}</div>
            </div>
            <div class='meta-card'>
                <strong>Runtime</strong>
                <div class='meta-strip'>device: {format_path(args.device)}</div>
                <div class='meta-strip'>dtype: {format_path(args.dtype)}</div>
                <div class='meta-strip'>attention: {format_path(args.attn_implementation or 'default')}</div>
            </div>
            <div class='meta-card'>
                <strong>Serve endpoint</strong>
                <div class='meta-strip'>{format_path(args.host)}:{args.port}</div>
                <div class='meta-strip'>first request loads weights and can take a while</div>
                <div class='meta-strip'>runs are stateless; the log below is not sent back to the model</div>
            </div>
        </div>
    </div>
    """

    block_kwargs: Dict[str, Any] = {"title": "Apertus CPT Console"}
    if BLOCKS_SUPPORTS_CSS:
        block_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**block_kwargs) as demo:
        history_state = gr.State([])

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(hero_html)

            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    with gr.Group(elem_classes=["panel-card", "probe-card"]):
                        gr.HTML("<div class='lane-chip'>Ready-made probes</div><div class='section-title'>Evaluation Deck</div>")
                        probe_selector = gr.Dropdown(
                            label="Choose a probe",
                            choices=[scenario["title"] for scenario in EVALUATION_SCENARIOS],
                            value=DEFAULT_SCENARIO_TITLE,
                        )
                        probe_details = gr.Markdown(value=render_probe_details(DEFAULT_SCENARIO_TITLE))
                        with gr.Row():
                            load_probe_button = gr.Button("Load selected probe")
                            run_probe_button = gr.Button("Run selected probe", variant="secondary")
                        gr.HTML(render_probe_library())

                with gr.Column(scale=8):
                    with gr.Group(elem_classes=["panel-card", "console-card"]):
                        gr.HTML("<div class='lane-chip'>Free prompting</div><div class='section-title'>Prompt Console</div>")
                        prompt_box = gr.Textbox(
                            label="Prompt",
                            lines=9,
                            value=selected_probe_prompt(DEFAULT_SCENARIO_TITLE),
                            placeholder="Write a freeform prompt or load one of the preset probes...",
                        )
                        with gr.Row():
                            ask_button = gr.Button("Ask model", variant="primary")
                            clear_current_button = gr.Button("Clear current view")

                        with gr.Accordion("Generation settings", open=False):
                            with gr.Row():
                                max_new_tokens = gr.Slider(
                                    minimum=64,
                                    maximum=1024,
                                    step=32,
                                    value=args.max_new_tokens,
                                    label="Max new tokens",
                                )
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.2,
                                    step=0.05,
                                    value=args.temperature,
                                    label="Temperature",
                                )
                                top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.05,
                                    value=args.top_p,
                                    label="Top-p",
                                )

                        answer_box = gr.Textbox(
                            label="Model answer",
                            lines=18,
                            **textbox_copy_kwargs(),
                        )
                        status_box = gr.HTML()

            with gr.Group(elem_classes=["panel-card"]):
                gr.HTML("<div class='lane-chip'>Review log</div><div class='section-title'>Session Log</div>")
                with gr.Row():
                    clear_history_button = gr.Button("Clear session log")
                history_html = gr.HTML(render_history([]))

        probe_selector.change(fn=render_probe_details, inputs=[probe_selector], outputs=[probe_details])
        load_probe_button.click(fn=selected_probe_prompt, inputs=[probe_selector], outputs=[prompt_box])
        run_probe_button.click(
            fn=lambda title, tokens, temp, nucleus, history: run_selected_probe(
                runtime,
                title,
                int(tokens),
                float(temp),
                float(nucleus),
                history,
            ),
            inputs=[probe_selector, max_new_tokens, temperature, top_p, history_state],
            outputs=[prompt_box, answer_box, status_box, history_state, history_html],
        )
        ask_button.click(
            fn=lambda prompt, tokens, temp, nucleus, history: run_generation(
                runtime=runtime,
                prompt=prompt,
                max_new_tokens=int(tokens),
                temperature=float(temp),
                top_p=float(nucleus),
                history=history,
                label="Freeform prompt",
            ),
            inputs=[prompt_box, max_new_tokens, temperature, top_p, history_state],
            outputs=[answer_box, status_box, history_state, history_html],
        )
        clear_current_button.click(fn=clear_current_view, outputs=[answer_box, status_box])
        clear_history_button.click(fn=clear_history, outputs=[history_state, history_html])

    demo.queue(default_concurrency_limit=4)
    return demo


def parse_args() -> argparse.Namespace:
    resolved_device = os.environ.get("APERTUS_MODEL_UI_DEVICE", default_device())
    resolved_dtype = os.environ.get("APERTUS_MODEL_UI_DTYPE", default_dtype(resolved_device))

    parser = argparse.ArgumentParser(description="Run a Gradio UI to inspect a trained CPT checkpoint.")
    parser.add_argument("--host", default=os.environ.get("APERTUS_MODEL_UI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("APERTUS_MODEL_UI_PORT", "7861")))
    parser.add_argument("--model-path", default=discover_default_model())
    parser.add_argument("--model-label", default=os.environ.get("APERTUS_MODEL_UI_LABEL", "Trained CPT model"))
    parser.add_argument("--device", default=resolved_device)
    parser.add_argument("--dtype", default=resolved_dtype, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=os.environ.get("APERTUS_MODEL_UI_ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=parse_bool(os.environ.get("APERTUS_MODEL_UI_TRUST_REMOTE_CODE", "false")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("APERTUS_MODEL_UI_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("APERTUS_MODEL_UI_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("APERTUS_MODEL_UI_TOP_P", str(DEFAULT_TOP_P))),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(args)

    launch_kwargs: Dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
        "share": False,
        "inbrowser": False,
    }
    if BLOCKS_LAUNCH_SUPPORTS_SHOW_API:
        launch_kwargs["show_api"] = False

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()