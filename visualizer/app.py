import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_tokenizers.py"
DEFAULT_BASE_TOKENIZER = str(REPO_ROOT / "artifacts" / "tokenizers" / "apertus-base")
DEFAULT_EXTENDED_TOKENIZER = str(REPO_ROOT / "artifacts" / "tokenizers" / "apertus-greek-v1")
DEFAULT_REFERENCE_TOKENIZER = str(REPO_ROOT / "artifacts" / "tokenizers" / "krikri-base")


def build_command(
    samples: List[str],
    base_tokenizer: str,
    extended_tokenizer: str,
    reference_tokenizer: str,
    report_path: Path,
) -> List[str]:
    command = [
        "python",
        str(COMPARE_SCRIPT),
        "--trust-remote-code",
        "--base-tokenizer",
        base_tokenizer,
        "--report-path",
        str(report_path),
    ]
    if extended_tokenizer.strip():
        command.extend(["--extended-tokenizer", extended_tokenizer])
    command.extend(["--reference-tokenizer", reference_tokenizer])
    for sample in samples:
        command.extend(["--text", sample])
    return command


def parse_samples(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def render_summary(summary: Dict[str, Any]) -> str:
    has_extended = "total_extended_tokens" in summary
    if has_extended:
        return (
            f"Samples: {summary['sample_count']}\n"
            f"Apertus base total tokens: {summary['total_base_tokens']}\n"
            f"Apertus Greek v1 total tokens: {summary['total_extended_tokens']}\n"
            f"Krikri total tokens: {summary['total_reference_tokens']}\n"
            f"Base -> Extended delta: {summary['base_to_extended_token_delta']}\n"
            f"Base -> Krikri delta: {summary['total_token_delta']}\n"
            f"Extended -> Krikri delta: {summary['extended_to_reference_token_delta']}\n"
            f"Avg Base tokens/sample: {summary['avg_base_tokens_per_sample']}\n"
            f"Avg Extended tokens/sample: {summary['avg_extended_tokens_per_sample']}\n"
            f"Avg Krikri tokens/sample: {summary['avg_reference_tokens_per_sample']}\n"
            f"Base -> Extended reduction: {summary['base_to_extended_reduction_pct']}%\n"
            f"Base -> Krikri reduction: {summary['relative_reduction_pct']}%\n"
            f"Extended -> Krikri reduction: {summary['extended_to_reference_reduction_pct']}%"
        )

    return (
        f"Samples: {summary['sample_count']}\n"
        f"Apertus total tokens: {summary['total_base_tokens']}\n"
        f"Krikri total tokens: {summary['total_reference_tokens']}\n"
        f"Token delta: {summary['total_token_delta']}\n"
        f"Avg Apertus tokens/sample: {summary['avg_base_tokens_per_sample']}\n"
        f"Avg Krikri tokens/sample: {summary['avg_reference_tokens_per_sample']}\n"
        f"Relative reduction: {summary['relative_reduction_pct']}%"
    )


def render_sample_markdown(sample: Dict[str, Any], index: int) -> str:
    has_extended = "extended_token_count" in sample
    lines = [
        f"### Sample {index}",
        f"**Text**: {sample['text']}",
        "",
        f"- Apertus token count: {sample['base_token_count']}",
    ]

    if has_extended:
        lines.extend(
            [
                f"- Apertus Greek v1 token count: {sample['extended_token_count']}",
                f"- Krikri token count: {sample['reference_token_count']}",
                f"- Base -> Extended delta: {sample['base_to_extended_token_delta']}",
                f"- Base -> Krikri delta: {sample['token_count_delta']}",
                f"- Extended -> Krikri delta: {sample['extended_to_reference_token_delta']}",
                f"- Apertus chars/token: {sample['base_chars_per_token']}",
                f"- Apertus Greek v1 chars/token: {sample['extended_chars_per_token']}",
                f"- Krikri chars/token: {sample['reference_chars_per_token']}",
                "",
                "**Apertus tokenization**",
                "```text",
                " | ".join(sample["base_decoded_pieces"]),
                "```",
                "**Apertus Greek v1 tokenization**",
                "```text",
                " | ".join(sample["extended_decoded_pieces"]),
                "```",
                "**Krikri tokenization**",
                "```text",
                " | ".join(sample["reference_decoded_pieces"]),
                "```",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"- Krikri token count: {sample['reference_token_count']}",
            f"- Token delta: {sample['token_count_delta']}",
            f"- Apertus chars/token: {sample['base_chars_per_token']}",
            f"- Krikri chars/token: {sample['reference_chars_per_token']}",
            "",
            "**Apertus tokenization**",
            "```text",
            " | ".join(sample["base_decoded_pieces"]),
            "```",
            "**Krikri tokenization**",
            "```text",
            " | ".join(sample["reference_decoded_pieces"]),
            "```",
        ]
    )
    return "\n".join(lines)


def compare_text(text: str, base_tokenizer: str, extended_tokenizer: str, reference_tokenizer: str):
    samples = parse_samples(text)
    if not samples:
        raise gr.Error("Provide at least one non-empty line of text.")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as report_file:
        report_path = Path(report_file.name)

    try:
        command = build_command(samples, base_tokenizer, extended_tokenizer, reference_tokenizer, report_path)
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise gr.Error(f"compare_tokenizers.py failed: {stderr}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)

    summary_text = render_summary(report["summary"])
    sample_sections = [render_sample_markdown(sample, index) for index, sample in enumerate(report["samples"], start=1)]
    comparison_markdown = "\n\n".join(sample_sections)
    raw_report = json.dumps(report, ensure_ascii=False, indent=2)
    return summary_text, comparison_markdown, raw_report


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Tokenizer Visualizer") as app:
        gr.Markdown(
            "# Tokenizer Visualizer\n"
            "Compare Greek tokenization between base Apertus, extended Apertus, and Krikri by sending one sentence per line."
        )

        with gr.Row():
            base_tokenizer = gr.Textbox(label="Base tokenizer", value=DEFAULT_BASE_TOKENIZER)
            extended_tokenizer = gr.Textbox(label="Extended tokenizer", value=DEFAULT_EXTENDED_TOKENIZER)
            reference_tokenizer = gr.Textbox(label="Reference tokenizer", value=DEFAULT_REFERENCE_TOKENIZER)

        text_input = gr.Textbox(
            label="Input text",
            lines=8,
            placeholder="Enter one sentence per line...",
            value=(
                "Η ελληνική γλώσσα χρειάζεται καλύτερη κάλυψη στο tokenizer.\n"
                "Τα σχολικά βιβλία περιέχουν όρους που θέλουμε να γίνονται tokenize πιο αποδοτικά."
            ),
        )

        compare_button = gr.Button("Compare tokenizers", variant="primary")
        summary_output = gr.Textbox(label="Summary", lines=13)
        comparison_output = gr.Markdown(label="Detailed comparison")
        raw_report_output = gr.Code(label="Raw JSON report", language="json")

        compare_button.click(
            fn=compare_text,
            inputs=[text_input, base_tokenizer, extended_tokenizer, reference_tokenizer],
            outputs=[summary_output, comparison_output, raw_report_output],
        )

    return app


if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1", server_port=7860)