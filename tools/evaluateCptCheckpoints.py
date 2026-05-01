#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_BASE_REPORT_CACHE = REPO_ROOT / "artifacts/reports/greek_mmlu_base_eval.json"
DEFAULT_EVALUATION_SCRIPT = REPO_ROOT / "evaluation/evaluate_greek_mmlu.py"
TOKENIZER_ARTIFACT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class CheckpointCandidate:
    name: str
    checkpoint_dir: Path
    step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all saved full-phase CPT checkpoints for a run, including intermediate "
            "Trainer checkpoints that need tokenizer metadata borrowed from the run final directory."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the CPT run directory that contains full/ and final/.",
    )
    parser.add_argument(
        "--checkpoint-subdir",
        default="full",
        help="Subdirectory inside --run-dir that contains the Trainer checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="checkpoint-*",
        help="Glob used under --checkpoint-subdir to find saved checkpoints.",
    )
    parser.add_argument(
        "--include-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate the run final/ directory.",
    )
    parser.add_argument(
        "--final-dir",
        help="Override the final checkpoint directory. Defaults to <run-dir>/final.",
    )
    parser.add_argument(
        "--view-root",
        help="Where to build tokenizer-aware eval views for intermediate checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write per-checkpoint JSON reports plus summary files.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model id or local path passed through to evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--base-report-cache",
        default=str(DEFAULT_BASE_REPORT_CACHE),
        help="Persistent base-model cache report reused across checkpoint evaluations.",
    )
    parser.add_argument(
        "--evaluation-script",
        default=str(DEFAULT_EVALUATION_SCRIPT),
        help="Path to evaluation/evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch the evaluation script. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device passed through to evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype passed through to evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--attn-implementation",
        help="Optional attention backend passed through to evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model and tokenizer during eval.",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the tokenizer chat template when available.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Progress print interval passed through to evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on evaluation examples for quick sweeps.",
    )
    parser.add_argument(
        "--subject",
        nargs="*",
        help="Optional subject filter passed through to evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include per-example predictions in each checkpoint report.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-checkpoint reports when they already exist.",
    )
    parser.add_argument(
        "--overwrite-views",
        action="store_true",
        help="Rebuild eval views even when they already exist.",
    )
    return parser.parse_args()


def numeric_step_from_name(name: str) -> int:
    prefix = "checkpoint-"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    if name == "final":
        return 10**12
    return 10**15


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir must point to an existing directory, got {run_dir}.")

    final_dir = Path(args.final_dir).expanduser().resolve() if args.final_dir else run_dir / "final"
    if args.include_final and not final_dir.is_dir():
        raise SystemExit(f"Final checkpoint directory was not found at {final_dir}.")

    checkpoint_root = run_dir / args.checkpoint_subdir
    if not checkpoint_root.is_dir():
        raise SystemExit(
            f"Checkpoint subdirectory {checkpoint_root} was not found under {run_dir}."
        )

    view_root = Path(args.view_root).expanduser().resolve() if args.view_root else run_dir / "eval_views"
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = REPO_ROOT / "artifacts" / "reports" / f"{run_dir.name}_checkpoint_sweep"

    evaluation_script = Path(args.evaluation_script).expanduser().resolve()
    if not evaluation_script.is_file():
        raise SystemExit(f"Evaluation script not found at {evaluation_script}.")

    return run_dir, checkpoint_root, final_dir, view_root, output_dir


def discover_checkpoints(
    checkpoint_root: Path,
    checkpoint_glob: str,
    final_dir: Path,
    include_final: bool,
) -> list[CheckpointCandidate]:
    candidates = [
        CheckpointCandidate(path.name, path, numeric_step_from_name(path.name))
        for path in checkpoint_root.glob(checkpoint_glob)
        if path.is_dir()
    ]
    candidates.sort(key=lambda candidate: (candidate.step, candidate.name))
    if include_final:
        candidates.append(CheckpointCandidate("final", final_dir, numeric_step_from_name("final")))
    if not candidates:
        raise SystemExit(f"No checkpoints matched under {checkpoint_root}.")
    return candidates


def checkpoint_has_tokenizer_artifacts(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "tokenizer.json").is_file() and (checkpoint_dir / "tokenizer_config.json").is_file()


def recreate_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src)


def build_eval_view(checkpoint_dir: Path, final_dir: Path, view_root: Path, overwrite: bool) -> Path:
    if checkpoint_has_tokenizer_artifacts(checkpoint_dir):
        return checkpoint_dir

    view_dir = view_root / checkpoint_dir.name
    if overwrite or not view_dir.exists():
        recreate_directory(view_dir)
    else:
        view_dir.mkdir(parents=True, exist_ok=True)

    for src in checkpoint_dir.iterdir():
        ensure_symlink(src, view_dir / src.name)

    for file_name in TOKENIZER_ARTIFACT_FILES:
        src = final_dir / file_name
        if src.exists():
            ensure_symlink(src, view_dir / file_name)

    if not checkpoint_has_tokenizer_artifacts(view_dir):
        raise SystemExit(
            f"Failed to build tokenizer-aware eval view for {checkpoint_dir}; tokenizer artifacts are still missing in {view_dir}."
        )
    return view_dir


def build_eval_command(
    args: argparse.Namespace,
    evaluation_script: Path,
    model_ref: Path,
    output_json: Path,
) -> list[str]:
    command = [
        args.python_executable,
        str(evaluation_script),
        "--base-model",
        args.base_model,
        "--trained-model",
        str(model_ref),
        "--output-json",
        str(output_json),
        "--base-report-cache",
        str(Path(args.base_report_cache).expanduser().resolve()),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--progress-interval",
        str(args.progress_interval),
        "--use-chat-template" if args.use_chat_template else "--no-use-chat-template",
    ]
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.subject:
        command.extend(["--subject", *args.subject])
    if args.save_predictions:
        command.append("--save-predictions")
    return command


def nested_get(payload: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def summary_record(candidate: CheckpointCandidate, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "checkpoint": candidate.name,
        "step": candidate.step,
        "model_ref": nested_get(report, ("models", "trained", "model_ref")),
        "base_accuracy": nested_get(report, ("models", "base", "overall", "accuracy")),
        "overall_accuracy": nested_get(report, ("models", "trained", "overall", "accuracy")),
        "overall_accuracy_delta": nested_get(report, ("comparison", "overall_accuracy_delta")),
        "social_sciences_accuracy": nested_get(
            report,
            ("models", "trained", "group_accuracy", "Social Sciences", "accuracy"),
        ),
        "social_sciences_delta": nested_get(
            report,
            ("comparison", "group_accuracy_delta", "Social Sciences"),
        ),
        "modern_greek_language_accuracy": nested_get(
            report,
            ("models", "trained", "subject_accuracy", "Modern Greek Language", "accuracy"),
        ),
        "modern_greek_language_delta": nested_get(
            report,
            ("comparison", "subject_accuracy_delta", "Modern Greek Language"),
        ),
        "law_accuracy": nested_get(report, ("models", "trained", "subject_accuracy", "Law", "accuracy")),
        "law_delta": nested_get(report, ("comparison", "subject_accuracy_delta", "Law")),
        "mathematics_accuracy": nested_get(
            report,
            ("models", "trained", "subject_accuracy", "Mathematics", "accuracy"),
        ),
        "mathematics_delta": nested_get(
            report,
            ("comparison", "subject_accuracy_delta", "Mathematics"),
        ),
        "secondary_school_accuracy": nested_get(
            report,
            ("models", "trained", "level_accuracy", "Secondary School", "accuracy"),
        ),
        "secondary_school_delta": nested_get(
            report,
            ("comparison", "level_accuracy_delta", "Secondary School"),
        ),
        "report_json": str(report_path),
    }


def best_checkpoint(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(
        records,
        key=lambda record: (
            float(record["overall_accuracy"]),
            -int(record["step"]),
        ),
    )


def format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def write_summary_files(output_dir: Path, records: Sequence[dict[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    summary_tsv = output_dir / "summary.tsv"
    champion = best_checkpoint(records)
    write_json(
        summary_json,
        {
            "best_checkpoint_by_overall_accuracy": champion,
            "checkpoints": list(records),
        },
    )

    columns = [
        "checkpoint",
        "step",
        "overall_accuracy",
        "overall_accuracy_delta",
        "social_sciences_accuracy",
        "social_sciences_delta",
        "modern_greek_language_accuracy",
        "modern_greek_language_delta",
        "law_accuracy",
        "law_delta",
        "mathematics_accuracy",
        "mathematics_delta",
        "secondary_school_accuracy",
        "secondary_school_delta",
        "report_json",
    ]
    lines = ["\t".join(columns)]
    for record in records:
        lines.append(
            "\t".join(
                [
                    str(record["checkpoint"]),
                    str(record["step"]),
                    format_float(record["overall_accuracy"]),
                    format_float(record["overall_accuracy_delta"]),
                    format_float(record["social_sciences_accuracy"]),
                    format_float(record["social_sciences_delta"]),
                    format_float(record["modern_greek_language_accuracy"]),
                    format_float(record["modern_greek_language_delta"]),
                    format_float(record["law_accuracy"]),
                    format_float(record["law_delta"]),
                    format_float(record["mathematics_accuracy"]),
                    format_float(record["mathematics_delta"]),
                    format_float(record["secondary_school_accuracy"]),
                    format_float(record["secondary_school_delta"]),
                    str(record["report_json"]),
                ]
            )
        )
    summary_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(records: Sequence[dict[str, Any]]) -> None:
    print()
    print(f"{'checkpoint':<15} {'overall':>10} {'delta':>10} {'social':>10} {'modern_el':>10}")
    for record in records:
        print(
            f"{record['checkpoint']:<15} "
            f"{format_float(record['overall_accuracy']):>10} "
            f"{format_float(record['overall_accuracy_delta']):>10} "
            f"{format_float(record['social_sciences_delta']):>10} "
            f"{format_float(record['modern_greek_language_delta']):>10}"
        )
    champion = best_checkpoint(records)
    print()
    print(
        "Best checkpoint by overall GreekMMLU accuracy: "
        f"{champion['checkpoint']} ({format_float(champion['overall_accuracy'])}, "
        f"delta {format_float(champion['overall_accuracy_delta'])})."
    )


def main() -> None:
    args = parse_args()
    run_dir, checkpoint_root, final_dir, view_root, output_dir = resolve_paths(args)
    evaluation_script = Path(args.evaluation_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    view_root.mkdir(parents=True, exist_ok=True)

    candidates = discover_checkpoints(
        checkpoint_root=checkpoint_root,
        checkpoint_glob=args.checkpoint_glob,
        final_dir=final_dir,
        include_final=args.include_final,
    )

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        model_ref = build_eval_view(
            checkpoint_dir=candidate.checkpoint_dir,
            final_dir=final_dir,
            view_root=view_root,
            overwrite=args.overwrite_views,
        )
        output_json = output_dir / f"{candidate.name}.json"
        if args.skip_existing and output_json.is_file():
            print(f"Reusing existing report for {candidate.name}: {output_json}")
            report = load_json(output_json)
        else:
            command = build_eval_command(
                args=args,
                evaluation_script=evaluation_script,
                model_ref=model_ref,
                output_json=output_json,
            )
            print(f"Evaluating {candidate.name} from {model_ref}")
            subprocess.run(command, cwd=str(REPO_ROOT), check=True)
            report = load_json(output_json)
        records.append(summary_record(candidate, report, output_json))

    records.sort(key=lambda record: (int(record["step"]), str(record["checkpoint"])))
    write_summary_files(output_dir, records)
    print_summary(records)
    print(f"\nPer-checkpoint reports and summary files were written under {output_dir}")


if __name__ == "__main__":
    main()