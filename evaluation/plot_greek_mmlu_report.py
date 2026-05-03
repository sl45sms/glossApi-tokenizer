#!/usr/bin/env python3
"""Convert a GreekMMLU evaluation JSON report into PNG bar charts for overall, group, level, and subject comparisons."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_REPORT_JSON = "artifacts/reports/greek_mmlu_eval.json"
DEFAULT_TOP_SUBJECTS = 20
BASE_COLOR = "#4e79a7"
KRIKRI_COLOR = "#f28e2b"
TRAINED_COLOR = "#e15759"
POSITIVE_DELTA_COLOR = "#59a14f"
NEGATIVE_DELTA_COLOR = "#d37295"
MODEL_ORDER = ("base", "krikri", "trained")
MODEL_FALLBACK_LABELS = {
    "base": "Base",
    "krikri": "KriKri",
    "trained": "Trained",
}
MODEL_COLORS = {
    "base": BASE_COLOR,
    "krikri": KRIKRI_COLOR,
    "trained": TRAINED_COLOR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a GreekMMLU evaluation JSON report into PNG bar charts for overall, group, "
            "level, and subject comparisons."
        )
    )
    parser.add_argument(
        "report_json",
        nargs="?",
        default=DEFAULT_REPORT_JSON,
        help="Path to the JSON report produced by evaluation/evaluate_greek_mmlu.py.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where PNG images will be written. Defaults to a sibling folder named "
            "<report-stem>_plots."
        ),
    )
    parser.add_argument(
        "--top-subjects",
        type=int,
        default=DEFAULT_TOP_SUBJECTS,
        help="Number of subjects to include in the subject charts. Use 0 to include every subject.",
    )
    parser.add_argument(
        "--subject-order",
        choices=("abs-delta", "delta", "trained", "alphabetical"),
        default="abs-delta",
        help="How to order subjects before slicing them for the subject charts.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution in dots per inch.",
    )
    return parser.parse_args()


def load_report(report_path: Path) -> Dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    available_models = report.get("models", {})
    required_paths = [
        ("models",),
        ("models", "base", "overall"),
        ("models", "trained", "overall"),
        ("models", "base", "group_accuracy"),
        ("models", "trained", "group_accuracy"),
        ("models", "base", "level_accuracy"),
        ("models", "trained", "level_accuracy"),
        ("models", "base", "subject_accuracy"),
        ("models", "trained", "subject_accuracy"),
    ]
    for model_key in ("krikri",):
        if model_key in available_models:
            required_paths.extend(
                [
                    ("models", model_key, "overall"),
                    ("models", model_key, "group_accuracy"),
                    ("models", model_key, "level_accuracy"),
                    ("models", model_key, "subject_accuracy"),
                ]
            )
    for path in required_paths:
        current: Any = report
        for key in path:
            if key not in current:
                dotted_path = ".".join(path)
                raise KeyError(f"Missing required report key: {dotted_path}")
            current = current[key]
    return report


def resolve_output_dir(report_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return report_path.parent / f"{report_path.stem}_plots"


def model_label(model_ref: str, fallback: str) -> str:
    cleaned = str(model_ref).rstrip("/")
    if not cleaned:
        return fallback
    return Path(cleaned).name or fallback


def ordered_model_keys(report: Dict[str, Any]) -> List[str]:
    return [model_key for model_key in MODEL_ORDER if model_key in report["models"]]


def extract_overall(
    report: Dict[str, Any],
    model_keys: Sequence[str],
) -> Tuple[List[str], List[float]]:
    labels = [MODEL_FALLBACK_LABELS[model_key] for model_key in model_keys]
    values = [float(report["models"][model_key]["overall"]["accuracy"]) for model_key in model_keys]
    return labels, values


def extract_breakdown(
    report: Dict[str, Any],
    breakdown_key: str,
) -> List[Tuple[str, Dict[str, float], float]]:
    model_keys = ordered_model_keys(report)
    breakdowns = {
        model_key: report["models"][model_key][breakdown_key]
        for model_key in model_keys
    }
    labels = sorted(
        {
            label
            for breakdown in breakdowns.values()
            for label in breakdown
        }
    )

    rows: List[Tuple[str, Dict[str, float], float]] = []
    for label in labels:
        accuracies = {
            model_key: float(breakdowns[model_key].get(label, {}).get("accuracy", 0.0))
            for model_key in model_keys
        }
        trained_accuracy = accuracies.get("trained", 0.0)
        base_accuracy = accuracies.get("base", 0.0)
        rows.append((label, accuracies, trained_accuracy - base_accuracy))
    return rows


def order_subject_rows(
    rows: Sequence[Tuple[str, Dict[str, float], float]],
    subject_order: str,
    top_subjects: int,
) -> List[Tuple[str, Dict[str, float], float]]:
    if subject_order == "alphabetical":
        ordered = sorted(rows, key=lambda row: row[0].lower())
    elif subject_order == "trained":
        ordered = sorted(rows, key=lambda row: (row[1].get("trained", 0.0), row[0].lower()), reverse=True)
    elif subject_order == "delta":
        ordered = sorted(rows, key=lambda row: (row[2], row[0].lower()), reverse=True)
    else:
        ordered = sorted(rows, key=lambda row: (abs(row[2]), row[0].lower()), reverse=True)

    if top_subjects > 0:
        ordered = ordered[:top_subjects]
    return ordered


def figure_size(label_count: int, wide: bool = False) -> Tuple[float, float]:
    width = max(8.0, min(24.0, 5.5 + (0.55 if wide else 0.45) * label_count))
    height = 6.0 if label_count <= 10 else 7.5
    return width, height


def annotate_bars(ax: plt.Axes, bars: Sequence[Any], value_scale: float) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height / value_scale:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def positive_upper_limit(values: Sequence[float], fallback: float = 1.0) -> float:
    if not values:
        return fallback
    return max(fallback, max(values) * 1.15)


def symmetric_delta_limit(values: Sequence[float], fallback: float = 1.0) -> float:
    if not values:
        return fallback
    return max(fallback, max(abs(value) for value in values) * 1.2)


def save_grouped_accuracy_chart(
    rows: Sequence[Tuple[str, Dict[str, float], float]],
    title: str,
    output_path: Path,
    model_keys: Sequence[str],
    model_labels: Dict[str, str],
    dpi: int,
    annotate: bool,
) -> None:
    labels = [row[0] for row in rows]
    positions = list(range(len(labels)))
    series_count = max(len(model_keys), 1)
    width = min(0.38, 0.82 / series_count)

    fig, ax = plt.subplots(figsize=figure_size(len(labels), wide=True))
    all_values: List[float] = []
    bar_series: List[Sequence[Any]] = []
    for series_index, model_key in enumerate(model_keys):
        offset = (series_index - (series_count - 1) / 2.0) * width
        model_values = [row[1].get(model_key, 0.0) * 100.0 for row in rows]
        all_values.extend(model_values)
        bars = ax.bar(
            [position + offset for position in positions],
            model_values,
            width=width,
            color=MODEL_COLORS[model_key],
            label=model_labels[model_key],
        )
        bar_series.append(bars)

    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, positive_upper_limit(all_values, fallback=100.0 if not labels else 1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    if annotate:
        for bars in bar_series:
            annotate_bars(ax, bars, 100.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_delta_chart(
    rows: Sequence[Tuple[str, Dict[str, float], float]],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    labels = [row[0] for row in rows]
    delta_values = [row[2] * 100.0 for row in rows]
    colors = [POSITIVE_DELTA_COLOR if value >= 0 else NEGATIVE_DELTA_COLOR for value in delta_values]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=figure_size(len(labels), wide=True))
    bars = ax.bar(positions, delta_values, color=colors)

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Accuracy delta")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    limit = symmetric_delta_limit(delta_values)
    ax.set_ylim(-limit, limit)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, delta_values):
        offset = 4 if value >= 0 else -14
        ax.annotate(
            f"{value / 100.0:+.1%}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_overall_chart(
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    positions = list(range(len(labels)))
    bar_values = [value * 100.0 for value in values]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    bars = ax.bar(positions, bar_values, color=colors[: len(labels)], width=0.55)

    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, positive_upper_limit(bar_values, fallback=100.0 if not bar_values else 1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    annotate_bars(ax, bars, 100.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    report = load_report(report_path)
    output_dir = resolve_output_dir(report_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = ordered_model_keys(report)
    model_labels = {
        model_key: model_label(
            report["models"][model_key].get("model_ref", model_key),
            MODEL_FALLBACK_LABELS[model_key],
        )
        for model_key in model_keys
    }

    overall_labels, overall_values = extract_overall(report, model_keys)
    group_rows = extract_breakdown(report, "group_accuracy")
    level_rows = extract_breakdown(report, "level_accuracy")
    subject_rows = extract_breakdown(report, "subject_accuracy")
    ordered_subject_rows = order_subject_rows(subject_rows, args.subject_order, args.top_subjects)

    save_overall_chart(
        labels=overall_labels,
        values=overall_values,
        colors=[MODEL_COLORS[model_key] for model_key in model_keys],
        title="GreekMMLU overall accuracy",
        output_path=output_dir / "overall_accuracy.png",
        dpi=args.dpi,
    )
    save_grouped_accuracy_chart(
        rows=group_rows,
        title="GreekMMLU accuracy by group",
        output_path=output_dir / "group_accuracy.png",
        model_keys=model_keys,
        model_labels=model_labels,
        dpi=args.dpi,
        annotate=True,
    )
    save_grouped_accuracy_chart(
        rows=level_rows,
        title="GreekMMLU accuracy by level",
        output_path=output_dir / "level_accuracy.png",
        model_keys=model_keys,
        model_labels=model_labels,
        dpi=args.dpi,
        annotate=True,
    )
    save_grouped_accuracy_chart(
        rows=ordered_subject_rows,
        title="GreekMMLU subject accuracy comparison",
        output_path=output_dir / "subject_accuracy_comparison.png",
        model_keys=model_keys,
        model_labels=model_labels,
        dpi=args.dpi,
        annotate=False,
    )
    save_delta_chart(
        rows=ordered_subject_rows,
        title="GreekMMLU subject accuracy delta (trained - base)",
        output_path=output_dir / "subject_accuracy_delta.png",
        dpi=args.dpi,
    )

    manifest = {
        "report_json": str(report_path),
        "output_dir": str(output_dir),
        "images": [
            str(output_dir / "overall_accuracy.png"),
            str(output_dir / "group_accuracy.png"),
            str(output_dir / "level_accuracy.png"),
            str(output_dir / "subject_accuracy_comparison.png"),
            str(output_dir / "subject_accuracy_delta.png"),
        ],
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()