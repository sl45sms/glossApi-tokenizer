import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer


DEFAULT_INPUT_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.sqlite3")
DEFAULT_INPUT_COUNTS_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.json")
DEFAULT_QUOTED_INPUT_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_word_counts.sqlite3")
DEFAULT_CAPITALIZED_INPUT_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_capitalized_word_counts.sqlite3")
DEFAULT_BASE_TOKENIZER = "artifacts/tokenizers/apertus-base"
DEFAULT_OUTPUT_TSV_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_candidates.tsv")
DEFAULT_OUTPUT_TOKENS_PATH = Path("artifacts/vocab_candidates/selected_tokens_v1.txt")
DEFAULT_REPORT_PATH = Path("artifacts/reports/fineweb2_hq_ell_grek_candidate_selection.json")
DEFAULT_STATIC_DIR = Path("vocabularyGen/static")
COUNT_SOURCE_WORDS = "words"
COUNT_SOURCE_QUOTED = "quoted"
COUNT_SOURCE_CAPITALIZED = "capitalized"
COUNT_SOURCE_TABLE_NAMES = {
    COUNT_SOURCE_WORDS: "word_counts",
    COUNT_SOURCE_QUOTED: "quoted_word_counts",
    COUNT_SOURCE_CAPITALIZED: "capitalized_word_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select tokenizer candidate tokens from counted Greek words by keeping words that the base "
            "tokenizer splits into at least N pieces and ranking them by frequency-weighted fragmentation."
        )
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "db", "json"),
        default="auto",
        help="Where to read the counted words from. Auto prefers SQLite when available.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_INPUT_DB_PATH,
        help="Primary SQLite all-word count database produced by vocabularyGen/countWords.py.",
    )
    parser.add_argument(
        "--counts-path",
        type=Path,
        default=DEFAULT_INPUT_COUNTS_PATH,
        help="Primary JSON all-word count export produced by vocabularyGen/countWords.py.",
    )
    parser.add_argument(
        "--quoted-db-path",
        type=Path,
        default=DEFAULT_QUOTED_INPUT_DB_PATH,
        help="Optional SQLite quoted-word count database produced by vocabularyGen/countWords.py.",
    )
    parser.add_argument(
        "--capitalized-db-path",
        type=Path,
        default=DEFAULT_CAPITALIZED_INPUT_DB_PATH,
        help="Optional SQLite capitalized-word count database produced by vocabularyGen/countWords.py.",
    )
    parser.add_argument(
        "--skip-quoted-counts",
        action="store_true",
        help="Do not combine quoted-word counts even when the quoted SQLite database is available.",
    )
    parser.add_argument(
        "--skip-capitalized-counts",
        action="store_true",
        help="Do not combine capitalized-word counts even when the capitalized SQLite database is available.",
    )
    parser.add_argument(
        "--base-tokenizer",
        default=DEFAULT_BASE_TOKENIZER,
        help="Base tokenizer path or Hugging Face model id to score against.",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=DEFAULT_STATIC_DIR,
        help="Directory of curated static token files to append. Every non-empty line in each file is read and hyphens are removed.",
    )
    parser.add_argument(
        "--skip-static-files",
        "--skip-static-affixes",
        action="store_true",
        help="Disable extra token injection from curated files in the static directory.",
    )
    parser.add_argument(
        "--output-tsv-path",
        type=Path,
        default=DEFAULT_OUTPUT_TSV_PATH,
        help="TSV file with candidate metadata sorted by selection rank.",
    )
    parser.add_argument(
        "--output-tokens-path",
        type=Path,
        default=DEFAULT_OUTPUT_TOKENS_PATH,
        help="Plain text file with one selected token per line, ready for tokenizer.add_tokens(...).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON summary report for the candidate selection run.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Only consider words with at least this many occurrences.",
    )
    parser.add_argument(
        "--top-k-input",
        type=int,
        default=200000,
        help="Only score the top K counted words after filtering by count. Use 0 to score all rows.",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=5000,
        help="Maximum number of selected candidates to export. Use 0 to export every passing candidate.",
    )
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=4,
        help="Ignore candidates shorter than this many characters.",
    )
    parser.add_argument(
        "--max-word-length",
        type=int,
        default=40,
        help="Ignore candidates longer than this many characters. Use 0 to disable the upper bound.",
    )
    parser.add_argument(
        "--include-non-greek",
        action="store_true",
        help="Keep non-Greek or mixed-script words instead of filtering to Greek-containing words.",
    )
    parser.add_argument(
        "--preserve-case-variants",
        action="store_true",
        help="Keep uppercase/lowercase variants as separate candidates instead of collapsing them.",
    )
    parser.add_argument(
        "--min-base-token-count",
        type=int,
        default=3,
        help="Only keep words that currently require at least this many base-tokenizer tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of words to score per tokenizer batch.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=25,
        help="Number of selected examples to include in the JSON report.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def contains_greek(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
            return True
    return False


def validate_args(args: argparse.Namespace) -> None:
    if args.min_count <= 0:
        raise SystemExit("--min-count must be greater than 0.")
    if args.top_k_input is not None and args.top_k_input < 0:
        raise SystemExit("--top-k-input cannot be negative.")
    if args.max_selected is not None and args.max_selected < 0:
        raise SystemExit("--max-selected cannot be negative.")
    if args.min_word_length <= 0:
        raise SystemExit("--min-word-length must be greater than 0.")
    if args.max_word_length < 0:
        raise SystemExit("--max-word-length cannot be negative.")
    if args.max_word_length and args.max_word_length < args.min_word_length:
        raise SystemExit("--max-word-length must be at least --min-word-length.")
    if args.min_base_token_count <= 0:
        raise SystemExit("--min-base-token-count must be greater than 0.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")
    if args.example_limit < 0:
        raise SystemExit("--example-limit cannot be negative.")
    if not args.skip_static_files:
        if not args.static_dir.exists():
            raise SystemExit(f"Static directory not found: {args.static_dir}")
        if not args.static_dir.is_dir():
            raise SystemExit(f"Static path is not a directory: {args.static_dir}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_clean_target(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        path.unlink()
        return
    raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --overwrite to replace it.")


def prepare_paths(args: argparse.Namespace) -> None:
    for path in (args.output_tsv_path, args.output_tokens_path, args.report_path):
        ensure_parent_dir(path)
        ensure_clean_target(path, args.overwrite)


def resolve_input_source(args: argparse.Namespace) -> Tuple[str, Path]:
    if args.input_format == "db":
        if not args.db_path.exists():
            raise SystemExit(f"SQLite count database not found: {args.db_path}")
        return "db", args.db_path

    if args.input_format == "json":
        if not args.counts_path.exists():
            raise SystemExit(f"JSON count file not found: {args.counts_path}")
        return "json", args.counts_path

    if args.db_path.exists():
        return "db", args.db_path
    if args.counts_path.exists():
        return "json", args.counts_path

    raise SystemExit(
        "No count input found. Run vocabularyGen/countWords.py first or provide --db-path/--counts-path."
    )


def load_rows_from_db(
    db_path: Path,
    table_name: str,
    min_count: int,
    top_k_input: Optional[int],
) -> List[Tuple[str, int]]:
    query = f"SELECT word, count FROM {table_name} WHERE count >= ? ORDER BY count DESC, word ASC"
    params: List[Any] = [min_count]
    if top_k_input:
        query += " LIMIT ?"
        params.append(top_k_input)

    connection = sqlite3.connect(str(db_path))
    try:
        return [(str(word), int(count)) for word, count in connection.execute(query, tuple(params))]
    finally:
        connection.close()


def load_rows_from_json(counts_path: Path, min_count: int, top_k_input: Optional[int]) -> List[Tuple[str, int]]:
    payload = json.loads(counts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Expected a JSON array in {counts_path}.")

    rows: List[Tuple[str, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        count = item.get("count")
        if not isinstance(word, str) or not isinstance(count, int):
            continue
        if count < min_count:
            continue
        rows.append((word, count))

    rows.sort(key=lambda entry: (-entry[1], entry[0]))
    if top_k_input:
        rows = rows[:top_k_input]
    return rows


def sort_rows_by_count(rows: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return sorted(rows, key=lambda entry: (-entry[1], entry[0]))


def accumulate_source_rows(
    source_name: str,
    rows: Sequence[Tuple[str, int]],
    combined_counts: Dict[str, int],
    source_counts_by_word: DefaultDict[str, Dict[str, int]],
) -> None:
    for word, count in rows:
        combined_counts[word] = combined_counts.get(word, 0) + count
        word_source_counts = source_counts_by_word[word]
        word_source_counts[source_name] = word_source_counts.get(source_name, 0) + count


def load_source_rows(args: argparse.Namespace) -> Dict[str, Any]:
    input_format, input_path = resolve_input_source(args)
    top_k_input = None if args.top_k_input == 0 else args.top_k_input

    if input_format == "db":
        primary_rows = load_rows_from_db(
            input_path,
            COUNT_SOURCE_TABLE_NAMES[COUNT_SOURCE_WORDS],
            args.min_count,
            top_k_input,
        )
    else:
        primary_rows = load_rows_from_json(input_path, args.min_count, top_k_input)

    combined_counts: Dict[str, int] = {}
    source_counts_by_word: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    input_source_stats: Dict[str, Any] = {
        "combined_row_count_before_merge": 0,
        "merged_unique_word_count": 0,
        "included_source_count": 0,
        "sources": [],
    }

    accumulate_source_rows(COUNT_SOURCE_WORDS, primary_rows, combined_counts, source_counts_by_word)
    input_source_stats["combined_row_count_before_merge"] += len(primary_rows)
    input_source_stats["included_source_count"] += 1
    input_source_stats["sources"].append(
        {
            "name": COUNT_SOURCE_WORDS,
            "format": input_format,
            "path": str(input_path),
            "included": True,
            "row_count": len(primary_rows),
        }
    )

    optional_sources = [
        (COUNT_SOURCE_QUOTED, args.quoted_db_path, args.skip_quoted_counts),
        (COUNT_SOURCE_CAPITALIZED, args.capitalized_db_path, args.skip_capitalized_counts),
    ]

    for source_name, db_path, skip_source in optional_sources:
        if skip_source:
            input_source_stats["sources"].append(
                {
                    "name": source_name,
                    "format": "db",
                    "path": str(db_path),
                    "included": False,
                    "row_count": 0,
                    "reason": "skipped-by-flag",
                }
            )
            continue

        if not db_path.exists():
            input_source_stats["sources"].append(
                {
                    "name": source_name,
                    "format": "db",
                    "path": str(db_path),
                    "included": False,
                    "row_count": 0,
                    "reason": "db-not-found",
                }
            )
            continue

        optional_rows = load_rows_from_db(
            db_path,
            COUNT_SOURCE_TABLE_NAMES[source_name],
            args.min_count,
            top_k_input,
        )
        accumulate_source_rows(source_name, optional_rows, combined_counts, source_counts_by_word)
        input_source_stats["combined_row_count_before_merge"] += len(optional_rows)
        input_source_stats["included_source_count"] += 1
        input_source_stats["sources"].append(
            {
                "name": source_name,
                "format": "db",
                "path": str(db_path),
                "included": True,
                "row_count": len(optional_rows),
            }
        )

    merged_rows = sort_rows_by_count(list(combined_counts.items()))
    input_source_stats["merged_unique_word_count"] = len(merged_rows)

    return {
        "primary_input_format": input_format,
        "primary_input_path": input_path,
        "rows": merged_rows,
        "source_counts_by_word": {
            word: dict(sorted(source_counts.items()))
            for word, source_counts in source_counts_by_word.items()
        },
        "input_source_stats": input_source_stats,
    }


def filter_source_rows(args: argparse.Namespace, rows: Sequence[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    stats = {
        "input_rows": len(rows),
        "skipped_short": 0,
        "skipped_long": 0,
        "skipped_non_greek": 0,
        "eligible_rows": 0,
    }
    filtered_rows: List[Tuple[str, int]] = []

    for word, count in rows:
        if len(word) < args.min_word_length:
            stats["skipped_short"] += 1
            continue
        if args.max_word_length and len(word) > args.max_word_length:
            stats["skipped_long"] += 1
            continue
        if not args.include_non_greek and not contains_greek(word):
            stats["skipped_non_greek"] += 1
            continue

        filtered_rows.append((word, count))
        stats["eligible_rows"] += 1

    return filtered_rows, stats


def batched(rows: Sequence[Tuple[str, int]], batch_size: int) -> Iterator[Sequence[Tuple[str, int]]]:
    for start_index in range(0, len(rows), batch_size):
        yield rows[start_index : start_index + batch_size]


def choose_case_variant_representative(variant_counts: Dict[str, int]) -> str:
    lowercase_variants = [word for word in variant_counts if word == word.lower()]
    candidate_pool = lowercase_variants if lowercase_variants else list(variant_counts)
    candidate_pool.sort(key=lambda word: (-variant_counts[word], word))
    return candidate_pool[0]


def collapse_case_variants(
    args: argparse.Namespace,
    rows: Sequence[Tuple[str, int]],
) -> Tuple[List[Tuple[str, int]], Dict[str, Dict[str, int]], Dict[str, Any]]:
    variant_groups: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    for word, count in rows:
        case_key = word.casefold()
        current_count = variant_groups[case_key].get(word, 0)
        variant_groups[case_key][word] = current_count + count

    stats: Dict[str, Any] = {
        "preserve_case_variants": args.preserve_case_variants,
        "input_rows": len(rows),
        "casefold_group_count": len(variant_groups),
        "collapsed_group_count": 0,
        "merged_variant_rows": 0,
    }

    if args.preserve_case_variants:
        variant_details = {word: {word: count} for word, count in rows}
        return list(rows), variant_details, stats

    collapsed_rows: List[Tuple[str, int]] = []
    variant_details: Dict[str, Dict[str, int]] = {}

    for variant_counts in variant_groups.values():
        representative = choose_case_variant_representative(variant_counts)
        collapsed_rows.append((representative, sum(variant_counts.values())))
        variant_details[representative] = dict(sorted(variant_counts.items(), key=lambda item: (-item[1], item[0])))
        if len(variant_counts) > 1:
            stats["collapsed_group_count"] += 1
            stats["merged_variant_rows"] += len(variant_counts) - 1

    collapsed_rows.sort(key=lambda entry: (-entry[1], entry[0]))
    return collapsed_rows, variant_details, stats


def build_representative_source_details(
    variant_details: Dict[str, Dict[str, int]],
    source_counts_by_word: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    representative_source_details: Dict[str, Dict[str, int]] = {}

    for representative, variants in variant_details.items():
        aggregated_source_counts: Dict[str, int] = {}
        for variant_word in variants:
            for source_name, count in source_counts_by_word.get(variant_word, {}).items():
                aggregated_source_counts[source_name] = aggregated_source_counts.get(source_name, 0) + count
        representative_source_details[representative] = dict(sorted(aggregated_source_counts.items()))

    return representative_source_details


def sanitize_static_entry(raw_entry: str, strip_hyphen: bool) -> str:
    cleaned_entry = raw_entry
    if strip_hyphen:
        cleaned_entry = cleaned_entry.replace("-", "")
    return cleaned_entry


def list_static_files(static_dir: Path) -> List[Path]:
    return sorted(path for path in static_dir.iterdir() if path.is_file())


def load_static_token_groups(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if args.skip_static_files:
        return [], {
            "enabled": False,
            "file_count": 0,
            "raw_entry_count": 0,
            "cleaned_entry_count": 0,
            "group_count": 0,
            "duplicate_exact_entry_count": 0,
            "static_files": [],
        }

    static_files = list_static_files(args.static_dir)
    raw_entries: List[Tuple[str, str, str]] = []
    for path in static_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw_entry = line
            if not raw_entry.strip():
                continue

            cleaned_entry = sanitize_static_entry(raw_entry, strip_hyphen=True)
            if not cleaned_entry:
                continue

            raw_entries.append((cleaned_entry, raw_entry, path.name))

    grouped_entries: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "raw_entries": [],
            "source_files": set(),
        }
    )

    for cleaned_entry, raw_entry, source_file in raw_entries:
        grouped_entry = grouped_entries[cleaned_entry]
        grouped_entry["count"] += 1
        grouped_entry["raw_entries"].append(raw_entry)
        grouped_entry["source_files"].add(source_file)

    static_entries: List[Dict[str, Any]] = []
    duplicate_exact_entry_count = 0
    for grouped_entry in grouped_entries.values():
        raw_entry_list = sorted(set(grouped_entry["raw_entries"]))
        if grouped_entry["count"] > 1:
            duplicate_exact_entry_count += grouped_entry["count"] - 1

        representative = sanitize_static_entry(raw_entry_list[0], strip_hyphen=True)
        static_entries.append(
            {
                "word": representative,
                "token": representative,
                "clean_variants": {representative: grouped_entry["count"]},
                "raw_entries": raw_entry_list,
                "source_files": sorted(grouped_entry["source_files"]),
            }
        )

    static_entries.sort(key=lambda item: item["word"])
    return static_entries, {
        "enabled": True,
        "file_count": len(static_files),
        "raw_entry_count": len(raw_entries),
        "cleaned_entry_count": len(raw_entries),
        "group_count": len(grouped_entries),
        "duplicate_exact_entry_count": duplicate_exact_entry_count,
        "static_files": [str(path) for path in static_files],
    }


def has_exact_single_token_coverage(tokenizer, word: str) -> Tuple[bool, List[int]]:
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) if token_ids else ""
    return len(token_ids) == 1 and decoded == word, token_ids


def candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -candidate["utility_score"],
        -candidate["base_token_count"],
        -candidate["count"],
        candidate["word"],
    )


def select_candidates(
    args: argparse.Namespace,
    rows: Sequence[Tuple[str, int]],
    base_tokenizer,
    variant_details: Dict[str, Dict[str, int]],
    representative_source_details: Dict[str, Dict[str, int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "scored_rows": 0,
        "skipped_below_token_threshold": 0,
        "selected_before_cap": 0,
    }
    selected_candidates: List[Dict[str, Any]] = []

    for batch in batched(rows, args.batch_size):
        words = [word for word, _ in batch]
        base_inputs = base_tokenizer(words, add_special_tokens=False)
        base_input_ids = base_inputs["input_ids"]

        for index, (word, count) in enumerate(batch):
            stats["scored_rows"] += 1

            base_token_count = len(base_input_ids[index])
            if base_token_count < args.min_base_token_count:
                stats["skipped_below_token_threshold"] += 1
                continue

            base_fragmentation = max(base_token_count - 1, 0)
            utility_score = count * base_fragmentation
            source_counts = dict(representative_source_details.get(word, {COUNT_SOURCE_WORDS: count}))

            selected_candidates.append(
                {
                    "word": word,
                    "token": f" {word}",
                    "count": count,
                    "count_sources": ",".join(source_counts),
                    "source_counts": source_counts,
                    "source_type": "corpus",
                    "static_source_files": "",
                    "source_variant_count": len(variant_details.get(word, {word: count})),
                    "source_variants": variant_details.get(word, {word: count}),
                    "base_token_count": base_token_count,
                    "base_fragmentation": base_fragmentation,
                    "utility_score": utility_score,
                }
            )

    stats["selected_before_cap"] = len(selected_candidates)
    selected_candidates.sort(key=candidate_sort_key)
    stats["selected_after_cap"] = len(selected_candidates)
    return selected_candidates, stats


def apply_total_selection_cap(
    args: argparse.Namespace,
    selected_candidates: Sequence[Dict[str, Any]],
    static_candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cap_stats: Dict[str, Any] = {
        "max_selected": None if args.max_selected == 0 else args.max_selected,
        "corpus_candidates_before_cap": len(selected_candidates),
        "static_candidates_before_cap": len(static_candidates),
    }

    if not args.max_selected:
        cap_stats.update(
            {
                "corpus_candidates_after_cap": len(selected_candidates),
                "static_candidates_after_cap": len(static_candidates),
                "total_candidates_after_cap": len(selected_candidates) + len(static_candidates),
                "dropped_corpus_candidates": 0,
                "dropped_static_candidates": 0,
            }
        )
        return list(selected_candidates), list(static_candidates), cap_stats

    kept_static_candidates = list(static_candidates[: args.max_selected])
    remaining_corpus_slots = max(args.max_selected - len(kept_static_candidates), 0)
    kept_selected_candidates = list(selected_candidates[:remaining_corpus_slots])

    cap_stats.update(
        {
            "corpus_candidates_after_cap": len(kept_selected_candidates),
            "static_candidates_after_cap": len(kept_static_candidates),
            "total_candidates_after_cap": len(kept_selected_candidates) + len(kept_static_candidates),
            "dropped_corpus_candidates": len(selected_candidates) - len(kept_selected_candidates),
            "dropped_static_candidates": len(static_candidates) - len(kept_static_candidates),
        }
    )
    return kept_selected_candidates, kept_static_candidates, cap_stats


def build_static_candidates(
    static_token_groups: Sequence[Dict[str, Any]],
    base_tokenizer,
    existing_tokens: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    existing_token_set = set(existing_tokens)
    stats: Dict[str, Any] = {
        "input_group_count": len(static_token_groups),
        "exact_single_token_in_base": 0,
        "skipped_exact_single_token_in_base": 0,
        "already_selected_by_token": 0,
        "missing_static_candidates": 0,
    }
    static_candidates: List[Dict[str, Any]] = []

    for static_group in static_token_groups:
        word = static_group["word"]
        exact_single_token, token_ids = has_exact_single_token_coverage(base_tokenizer, word)
        if exact_single_token:
            stats["exact_single_token_in_base"] += 1
            stats["skipped_exact_single_token_in_base"] += 1
            continue

        if word in existing_token_set:
            stats["already_selected_by_token"] += 1
            continue

        static_candidates.append(
            {
                "word": word,
                "token": word,
                "count": 0,
                "count_sources": "",
                "source_counts": {},
                "source_type": "static",
                "static_source_files": ",".join(static_group["source_files"]),
                "source_variant_count": len(static_group["raw_entries"]),
                "source_variants": static_group["clean_variants"],
                "static_raw_entries": static_group["raw_entries"],
                "base_token_count": len(token_ids),
                "base_fragmentation": max(len(token_ids) - 1, 0),
                "utility_score": 0,
            }
        )

    static_candidates.sort(key=lambda candidate: (-candidate["base_token_count"], candidate["word"]))
    stats["missing_static_candidates"] = len(static_candidates)
    return static_candidates, stats


def decode_token_ids(tokenizer, token_ids: Sequence[int]) -> List[str]:
    return [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in token_ids]


def build_examples(
    selected_candidates: Sequence[Dict[str, Any]],
    base_tokenizer,
    limit: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for candidate in selected_candidates[:limit]:
        word = candidate["word"]
        base_ids = base_tokenizer.encode(word, add_special_tokens=False)
        example = dict(candidate)
        example.setdefault("source_variants", {word: candidate["count"]})
        example["base_decoded_pieces"] = decode_token_ids(base_tokenizer, base_ids)

        examples.append(example)
    return examples


def write_candidate_tsv(path: Path, selected_candidates: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        writer.writerow(
            [
                "word",
                "token",
                "count",
                "count_sources",
                "source_counts_json",
                "source_type",
                "static_source_files",
                "source_variant_count",
                "base_token_count",
                "base_fragmentation",
                "utility_score",
            ]
        )
        for candidate in selected_candidates:
            writer.writerow(
                [
                    candidate["word"],
                    candidate["token"],
                    candidate["count"],
                    candidate.get("count_sources", ""),
                    json.dumps(candidate.get("source_counts", {}), ensure_ascii=False, sort_keys=True),
                    candidate["source_type"],
                    candidate["static_source_files"],
                    candidate["source_variant_count"],
                    candidate["base_token_count"],
                    candidate["base_fragmentation"],
                    candidate["utility_score"],
                ]
            )


def write_token_list(path: Path, selected_candidates: Sequence[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(candidate["token"] for candidate in selected_candidates) + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_paths(args)

    source_bundle = load_source_rows(args)
    filtered_rows, source_filter_stats = filter_source_rows(args, source_bundle["rows"])
    collapsed_rows, variant_details, case_variant_stats = collapse_case_variants(args, filtered_rows)
    representative_source_details = build_representative_source_details(
        variant_details,
        source_bundle["source_counts_by_word"],
    )

    base_tokenizer = load_repo_tokenizer(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    selected_candidates, selection_stats = select_candidates(
        args,
        collapsed_rows,
        base_tokenizer,
        variant_details,
        representative_source_details,
    )

    static_token_groups, static_input_stats = load_static_token_groups(args)
    static_candidates, static_stats = build_static_candidates(
        static_token_groups,
        base_tokenizer,
        [candidate["token"] for candidate in selected_candidates],
    )
    selected_candidates, static_candidates, selection_cap_stats = apply_total_selection_cap(
        args,
        selected_candidates,
        static_candidates,
    )
    all_selected_candidates = selected_candidates + static_candidates

    write_candidate_tsv(args.output_tsv_path, all_selected_candidates)
    write_token_list(args.output_tokens_path, all_selected_candidates)

    report = {
        "input": {
            "format": source_bundle["primary_input_format"],
            "path": str(source_bundle["primary_input_path"]),
        },
        "input_sources": source_bundle["input_source_stats"],
        "tokenizers": {
            "base_tokenizer": args.base_tokenizer,
        },
        "filters": {
            "min_count": args.min_count,
            "top_k_input": None if args.top_k_input == 0 else args.top_k_input,
            "max_selected": None if args.max_selected == 0 else args.max_selected,
            "min_word_length": args.min_word_length,
            "max_word_length": None if args.max_word_length == 0 else args.max_word_length,
            "include_non_greek": args.include_non_greek,
            "preserve_case_variants": args.preserve_case_variants,
            "min_base_token_count": args.min_base_token_count,
            "skip_quoted_counts": args.skip_quoted_counts,
            "skip_capitalized_counts": args.skip_capitalized_counts,
            "skip_static_files": args.skip_static_files,
            "batch_size": args.batch_size,
        },
        "static_dir": str(args.static_dir),
        "source_filter_stats": source_filter_stats,
        "case_variant_stats": case_variant_stats,
        "selection_stats": selection_stats,
        "selection_cap_stats": selection_cap_stats,
        "static_input_stats": static_input_stats,
        "static_stats": static_stats,
        "outputs": {
            "output_tsv_path": str(args.output_tsv_path),
            "output_tokens_path": str(args.output_tokens_path),
            "report_path": str(args.report_path),
        },
        "selected_examples": build_examples(
            all_selected_candidates,
            base_tokenizer,
            args.example_limit,
        ),
        "selected_static_examples": build_examples(
            static_candidates,
            base_tokenizer,
            args.example_limit,
        ),
    }

    write_report(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()