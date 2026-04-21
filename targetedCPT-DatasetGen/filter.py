import argparse
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

try:
    import ahocorasick
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python module 'ahocorasick'. Install the pip package 'pyahocorasick' into .venv-uenv, "
        "for example with ./run_uenv.sh python -m pip install -r requirements.txt"
    ) from exc

from datasets import load_dataset


DEFAULT_DATASET_ID = "epfml/FineWeb2-HQ"
DEFAULT_CONFIG = "ell_Grek"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_TOKEN_FILE = Path("artifacts/vocab_candidates/selected_tokens_v1.txt")
DEFAULT_REPORT_PATH = Path("artifacts/reports/targeted_cpt_filter_summary.json")

_WORKER_AUTOMATON: ahocorasick.Automaton | None = None
_WORKER_TARGETS: tuple["TargetSpec", ...] = ()
_WORKER_NORMALIZATION = "nfkc"
_WORKER_CASEFOLD = True


@dataclass(frozen=True)
class TargetSpec:
    target_id: int
    raw_token: str
    surface_form: str
    search_text: str
    requires_word_boundaries: bool


@dataclass
class RunState:
    counts: list[int]
    selected_documents: int = 0
    matched_documents: int = 0
    matched_target_samples: int = 0
    satisfied_targets: int = 0
    output_bytes: int = 0
    stop_reason: str | None = None


def default_output_path() -> Path:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "targeted-cpt" / "curated_greek_cpt.jsonl"
    return Path("artifacts/curated_greek_cpt.jsonl")


def default_cache_dir() -> Path | None:
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if cache_dir:
        return Path(cache_dir)

    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "hf_datasets"

    return None


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    return max(1, min(16, cpu_count // 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the FineWeb2-HQ Greek split, search for every selected tokenizer candidate with "
            "Aho-Corasick, and write only the documents that help cover still-underfilled targets."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset id to load.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Dataset configuration name.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to iterate.",
    )
    parser.add_argument(
        "--text-field",
        default=DEFAULT_TEXT_FIELD,
        help="Dataset field that contains the document text.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="Tokenizer candidate file to target, usually selected_tokens_v1.txt.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path(),
        help=(
            "Destination JSONL file with entries shaped like {\"text\": ...}. Defaults to "
            "$SCRATCH/targeted-cpt/curated_greek_cpt.jsonl when SCRATCH is set."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination JSON summary report.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="Optional Hugging Face cache directory override.",
    )
    parser.add_argument(
        "--limit-per-word",
        type=int,
        default=50,
        help="Maximum number of selected documents that may contribute to each target token.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of worker processes used for Aho-Corasick matching.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="How many documents to send to one worker task at a time.",
    )
    parser.add_argument(
        "--max-pending-batches",
        type=int,
        help="Maximum number of worker batches allowed in flight. Defaults to workers * 2.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Optional cap on scanned dataset documents for a shorter dry run.",
    )
    parser.add_argument(
        "--max-selected-documents",
        type=int,
        help="Optional cap on written JSONL rows.",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        help="Optional cap on output JSONL size in bytes.",
    )
    parser.add_argument(
        "--quality-score-min",
        type=float,
        help="Optional lower bound for the dataset quality_score field.",
    )
    parser.add_argument(
        "--normalization",
        choices=("none", "nfc", "nfkc"),
        default="nfkc",
        help="Unicode normalization applied before matching.",
    )
    parser.add_argument(
        "--no-casefold",
        action="store_true",
        help="Disable casefolding before matching. By default matching is case-insensitive.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10000,
        help="Emit a progress line every N scanned documents. Use 0 to disable progress output.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=25,
        help="How many underfilled targets to include in the JSON report.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the split eagerly instead of using datasets streaming.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output JSONL and report files.",
    )
    return parser.parse_args()


def normalize_for_matching(text: str, normalization: str, casefold: bool) -> str:
    normalized = text
    if normalization != "none":
        normalized = unicodedata.normalize(normalization.upper(), normalized)
    normalized = normalized.replace("\u2019", "'").replace("\u02bc", "'")
    if casefold:
        normalized = normalized.casefold()
    return normalized


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_clean_target(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        path.unlink()
        return
    raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --overwrite to replace it.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.token_file.exists():
        raise SystemExit(f"Token file not found: {args.token_file}")
    if args.limit_per_word <= 0:
        raise SystemExit("--limit-per-word must be greater than 0.")
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than 0.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")
    if args.max_pending_batches is not None and args.max_pending_batches <= 0:
        raise SystemExit("--max-pending-batches must be greater than 0 when set.")
    if args.max_documents is not None and args.max_documents <= 0:
        raise SystemExit("--max-documents must be greater than 0 when set.")
    if args.max_selected_documents is not None and args.max_selected_documents <= 0:
        raise SystemExit("--max-selected-documents must be greater than 0 when set.")
    if args.max_output_bytes is not None and args.max_output_bytes <= 0:
        raise SystemExit("--max-output-bytes must be greater than 0 when set.")
    if args.report_every < 0:
        raise SystemExit("--report-every cannot be negative.")
    if args.preview_limit < 0:
        raise SystemExit("--preview-limit cannot be negative.")


def prepare_paths(args: argparse.Namespace) -> None:
    ensure_parent_dir(args.output_path)
    ensure_parent_dir(args.report_path)
    ensure_clean_target(args.output_path, args.overwrite)
    ensure_clean_target(args.report_path, args.overwrite)


def load_targets(args: argparse.Namespace) -> tuple[list[TargetSpec], Dict[str, int]]:
    raw_lines = args.token_file.read_text(encoding="utf-8").splitlines()
    casefold = not args.no_casefold
    seen_keys: set[tuple[str, str, bool]] = set()
    targets: list[TargetSpec] = []
    stats = {
        "raw_lines": len(raw_lines),
        "skipped_empty_lines": 0,
        "deduplicated_tokens": 0,
        "boundary_sensitive_targets": 0,
    }

    for raw_line in raw_lines:
        if not raw_line.strip():
            stats["skipped_empty_lines"] += 1
            continue

        requires_word_boundaries = raw_line[:1].isspace()
        surface_form = raw_line.lstrip()
        search_text = normalize_for_matching(surface_form, args.normalization, casefold)
        if not search_text:
            stats["skipped_empty_lines"] += 1
            continue

        dedupe_key = (raw_line, search_text, requires_word_boundaries)
        if dedupe_key in seen_keys:
            stats["deduplicated_tokens"] += 1
            continue

        seen_keys.add(dedupe_key)
        target_id = len(targets)
        targets.append(
            TargetSpec(
                target_id=target_id,
                raw_token=raw_line,
                surface_form=surface_form,
                search_text=search_text,
                requires_word_boundaries=requires_word_boundaries,
            )
        )
        if requires_word_boundaries:
            stats["boundary_sensitive_targets"] += 1

    stats["active_targets"] = len(targets)

    if not targets:
        raise SystemExit(f"No usable targets were loaded from {args.token_file}.")

    return targets, stats


def build_automaton(targets: Sequence[TargetSpec]) -> ahocorasick.Automaton:
    automaton = ahocorasick.Automaton()
    grouped_target_ids: dict[str, list[int]] = defaultdict(list)
    for target in targets:
        grouped_target_ids[target.search_text].append(target.target_id)

    for search_text, target_ids in grouped_target_ids.items():
        automaton.add_word(search_text, (len(search_text), tuple(target_ids)))

    automaton.make_automaton()
    return automaton


def init_worker(targets: Sequence[TargetSpec], normalization: str, casefold: bool) -> None:
    global _WORKER_AUTOMATON, _WORKER_TARGETS, _WORKER_NORMALIZATION, _WORKER_CASEFOLD

    _WORKER_TARGETS = tuple(targets)
    _WORKER_NORMALIZATION = normalization
    _WORKER_CASEFOLD = casefold
    _WORKER_AUTOMATON = build_automaton(_WORKER_TARGETS)


def is_word_char(character: str) -> bool:
    return character.isalnum() or character == "_"


def has_word_boundaries(text: str, start_index: int, end_index: int) -> bool:
    left_is_boundary = start_index == 0 or not is_word_char(text[start_index - 1])
    right_is_boundary = end_index == len(text) - 1 or not is_word_char(text[end_index + 1])
    return left_is_boundary and right_is_boundary


def process_batch(batch_texts: Sequence[str]) -> list[tuple[int, list[int]]]:
    if _WORKER_AUTOMATON is None:
        raise RuntimeError("Worker automaton was not initialized.")

    selected_rows: list[tuple[int, list[int]]] = []
    for row_index, text in enumerate(batch_texts):
        normalized_text = normalize_for_matching(text, _WORKER_NORMALIZATION, _WORKER_CASEFOLD)
        if not normalized_text:
            continue

        matched_target_ids: set[int] = set()
        for end_index, (pattern_length, target_ids) in _WORKER_AUTOMATON.iter(normalized_text):
            start_index = end_index - pattern_length + 1
            for target_id in target_ids:
                if target_id in matched_target_ids:
                    continue
                target = _WORKER_TARGETS[target_id]
                if target.requires_word_boundaries and not has_word_boundaries(
                    normalized_text,
                    start_index,
                    end_index,
                ):
                    continue
                matched_target_ids.add(target_id)

        if matched_target_ids:
            selected_rows.append((row_index, sorted(matched_target_ids)))

    return selected_rows


def dataset_iterator(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    return load_dataset(
        args.dataset_id,
        args.config,
        split=args.split,
        streaming=not args.no_streaming,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
    )


def future_result(
    pending: dict[Future[list[tuple[int, list[int]]]], list[str]],
) -> tuple[list[str], list[tuple[int, list[int]]]]:
    done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
    future = next(iter(done))
    batch_texts = pending.pop(future)
    return batch_texts, future.result()


def progress_line(
    args: argparse.Namespace,
    state: RunState,
    scanned_documents: int,
    eligible_documents: int,
    total_targets: int,
    pending_batches: int,
    started_at: float,
) -> str:
    elapsed = round(time.time() - started_at, 2)
    output_mib = state.output_bytes / (1024 * 1024)
    return (
        f"Scanned {scanned_documents} documents | Eligible {eligible_documents} | "
        f"Documents with matches {state.matched_documents} | Selected {state.selected_documents} | "
        f"Targets satisfied {state.satisfied_targets}/{total_targets} | Pending batches {pending_batches} | "
        f"Output {output_mib:.2f} MiB | Elapsed {elapsed}s"
    )


def apply_batch_results(
    args: argparse.Namespace,
    batch_texts: Sequence[str],
    batch_matches: Sequence[tuple[int, list[int]]],
    output_file,
    targets: Sequence[TargetSpec],
    state: RunState,
) -> None:
    if state.stop_reason is not None:
        return

    state.matched_documents += len(batch_matches)
    for row_index, matched_target_ids in batch_matches:
        if args.max_selected_documents is not None and state.selected_documents >= args.max_selected_documents:
            state.stop_reason = f"selected-documents-limit:{args.max_selected_documents}"
            break

        contributing_target_ids = [
            target_id for target_id in matched_target_ids if state.counts[target_id] < args.limit_per_word
        ]
        if not contributing_target_ids:
            continue

        output_line = json.dumps({"text": batch_texts[row_index]}, ensure_ascii=False) + "\n"
        output_line_bytes = len(output_line.encode("utf-8"))
        if args.max_output_bytes is not None and state.output_bytes + output_line_bytes > args.max_output_bytes:
            state.stop_reason = f"output-bytes-limit:{args.max_output_bytes}"
            break

        output_file.write(output_line)
        state.output_bytes += output_line_bytes
        state.selected_documents += 1

        for target_id in contributing_target_ids:
            state.counts[target_id] += 1
            state.matched_target_samples += 1
            if state.counts[target_id] == args.limit_per_word:
                state.satisfied_targets += 1

        if state.satisfied_targets >= len(targets):
            state.stop_reason = "all-targets-satisfied"
            break


def underfilled_targets(
    args: argparse.Namespace,
    targets: Sequence[TargetSpec],
    counts: Sequence[int],
) -> list[dict[str, object]]:
    missing_targets = [
        {
            "token": targets[index].raw_token,
            "surface_form": targets[index].surface_form,
            "count": counts[index],
            "missing": args.limit_per_word - counts[index],
            "requires_word_boundaries": targets[index].requires_word_boundaries,
        }
        for index in range(len(targets))
        if counts[index] < args.limit_per_word
    ]
    missing_targets.sort(key=lambda item: (item["count"], item["token"]))
    if args.preview_limit == 0:
        return []
    return missing_targets[: args.preview_limit]


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_filter(args: argparse.Namespace) -> dict[str, object]:
    casefold = not args.no_casefold
    targets, target_stats = load_targets(args)
    max_pending_batches = args.max_pending_batches or max(1, args.workers * 2)
    started_at = time.time()

    state = RunState(counts=[0] * len(targets))
    scanned_documents = 0
    eligible_documents = 0
    reader_stop_reason: str | None = None
    pending: dict[Future[list[tuple[int, list[int]]]], list[str]] = {}
    batch: list[str] = []

    with args.output_path.open("w", encoding="utf-8") as output_file:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(targets, args.normalization, casefold),
        )
        try:
            for record in dataset_iterator(args):
                if state.stop_reason is not None:
                    reader_stop_reason = state.stop_reason
                    break
                if args.max_documents is not None and scanned_documents >= args.max_documents:
                    reader_stop_reason = f"max-documents-limit:{args.max_documents}"
                    break

                scanned_documents += 1

                if args.quality_score_min is not None:
                    quality_score = record.get("quality_score")
                    if quality_score is None or quality_score < args.quality_score_min:
                        continue

                text = record.get(args.text_field)
                if not isinstance(text, str) or not text.strip():
                    continue

                eligible_documents += 1
                batch.append(text)

                if len(batch) >= args.batch_size:
                    pending[executor.submit(process_batch, tuple(batch))] = batch
                    batch = []

                while pending and len(pending) >= max_pending_batches:
                    completed_batch, batch_matches = future_result(pending)
                    apply_batch_results(args, completed_batch, batch_matches, output_file, targets, state)
                    if state.stop_reason is not None:
                        reader_stop_reason = state.stop_reason
                        break

                if args.report_every and scanned_documents % args.report_every == 0:
                    print(
                        progress_line(
                            args,
                            state,
                            scanned_documents,
                            eligible_documents,
                            len(targets),
                            len(pending),
                            started_at,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

            if batch and state.stop_reason is None:
                pending[executor.submit(process_batch, tuple(batch))] = batch

            if state.stop_reason is None:
                while pending:
                    completed_batch, batch_matches = future_result(pending)
                    apply_batch_results(args, completed_batch, batch_matches, output_file, targets, state)
            else:
                for future in pending:
                    future.cancel()
                pending.clear()
        finally:
            executor.shutdown(wait=state.stop_reason is None, cancel_futures=True)

        output_file.flush()

    final_stop_reason = state.stop_reason or reader_stop_reason or "dataset-exhausted"
    elapsed_seconds = round(time.time() - started_at, 2)
    report = {
        "dataset": {
            "dataset_id": args.dataset_id,
            "config": args.config,
            "split": args.split,
            "text_field": args.text_field,
            "streaming": not args.no_streaming,
            "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        },
        "paths": {
            "token_file": str(args.token_file),
            "output_path": str(args.output_path),
            "report_path": str(args.report_path),
        },
        "settings": {
            "limit_per_word": args.limit_per_word,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "max_pending_batches": max_pending_batches,
            "max_documents": args.max_documents,
            "max_selected_documents": args.max_selected_documents,
            "max_output_bytes": args.max_output_bytes,
            "quality_score_min": args.quality_score_min,
            "normalization": args.normalization,
            "casefold": casefold,
            "report_every": args.report_every,
        },
        "target_summary": target_stats,
        "run_summary": {
            "stop_reason": final_stop_reason,
            "scanned_documents": scanned_documents,
            "eligible_documents": eligible_documents,
            "documents_with_any_target_match": state.matched_documents,
            "selected_documents": state.selected_documents,
            "matched_target_samples": state.matched_target_samples,
            "satisfied_targets": state.satisfied_targets,
            "unsatisfied_targets": len(targets) - state.satisfied_targets,
            "output_bytes": state.output_bytes,
            "elapsed_seconds": elapsed_seconds,
        },
        "coverage_preview": {
            "underfilled_targets": underfilled_targets(args, targets, state.counts),
        },
    }
    write_report(args.report_path, report)
    return report


def main() -> int:
    args = parse_args()
    validate_args(args)
    prepare_paths(args)
    report = run_filter(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)