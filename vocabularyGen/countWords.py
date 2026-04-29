import argparse
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

from datasets import load_dataset


DEFAULT_DATASET_ID = "epfml/FineWeb2-HQ"
DEFAULT_CONFIG = "ell_Grek"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_OUTPUT_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.json")
DEFAULT_REPORT_PATH = Path("artifacts/reports/fineweb2_hq_ell_grek_word_count_summary.json")
DEFAULT_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.sqlite3")
DEFAULT_QUOTED_OUTPUT_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_words.txt")
DEFAULT_QUOTED_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_word_counts.sqlite3")
DEFAULT_CAPITALIZED_OUTPUT_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_capitalized_words.txt")
DEFAULT_CAPITALIZED_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_capitalized_word_counts.sqlite3")
WORD_CONNECTORS = {"'", "-"}
COUNT_MODE_WORDS = "words"
COUNT_MODE_QUOTED = "quoted"
COUNT_MODE_CAPITALIZED = "capitalized"
COUNT_MODES = (COUNT_MODE_WORDS, COUNT_MODE_QUOTED, COUNT_MODE_CAPITALIZED)
MODE_TABLE_NAMES = {
	COUNT_MODE_WORDS: "word_counts",
	COUNT_MODE_QUOTED: "quoted_word_counts",
	COUNT_MODE_CAPITALIZED: "capitalized_word_counts",
}
QUOTE_TRANSLATION = str.maketrans(
	{
		"\u2018": "'",
		"\u2019": "'",
		"\u201a": "'",
		"\u201b": "'",
		"\u2032": "'",
		"\u201c": '"',
		"\u201d": '"',
		"\u201e": '"',
		"\u00ab": '"',
		"\u00bb": '"',
	}
)
QUOTED_SEGMENT_PATTERNS = (
	re.compile(r'"([^"\n]+)"'),
	re.compile(r'`([^`\n]+)`'),
	re.compile(r"(?<!\w)'([^'\n]+)'(?!\w)"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Stream FineWeb2-HQ Greek text on Clariden, count regular word frequencies, and optionally "
			"also export quoted and capitalized word lists for tokenizer static candidates."
		)
	)
	parser.add_argument(
		"--count-modes",
		nargs="+",
		choices=COUNT_MODES,
		default=[COUNT_MODE_WORDS],
		help=(
			"Which counters to run. Use one or more of: words, quoted, capitalized. "
			"The default keeps the legacy all-word counting behavior."
		),
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
		"--output-path",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Destination JSON file for all-word counts with entries shaped like {\"word\": ..., \"count\": ...}.",
	)
	parser.add_argument(
		"--quoted-output-path",
		type=Path,
		default=DEFAULT_QUOTED_OUTPUT_PATH,
		help="Destination text file with one quoted-word candidate per line.",
	)
	parser.add_argument(
		"--capitalized-output-path",
		type=Path,
		default=DEFAULT_CAPITALIZED_OUTPUT_PATH,
		help="Destination text file with one capitalized-word candidate per line.",
	)
	parser.add_argument(
		"--report-path",
		type=Path,
		default=DEFAULT_REPORT_PATH,
		help="Destination JSON summary report.",
	)
	parser.add_argument(
		"--db-path",
		type=Path,
		default=DEFAULT_DB_PATH,
		help="SQLite sidecar used to store exact all-word counts while streaming the dataset.",
	)
	parser.add_argument(
		"--quoted-db-path",
		type=Path,
		default=DEFAULT_QUOTED_DB_PATH,
		help="SQLite sidecar used to store exact quoted-word counts while streaming the dataset.",
	)
	parser.add_argument(
		"--capitalized-db-path",
		type=Path,
		default=DEFAULT_CAPITALIZED_DB_PATH,
		help="SQLite sidecar used to store exact capitalized-word counts while streaming the dataset.",
	)
	parser.add_argument(
		"--cache-dir",
		type=Path,
		help="Optional Hugging Face cache directory override.",
	)
	parser.add_argument(
		"--max-documents",
		type=int,
		help="Optional cap on the number of streamed documents for a shorter dry run.",
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
		help="Unicode normalization applied before extraction.",
	)
	parser.add_argument(
		"--casefold",
		action="store_true",
		help="Case-fold regular and quoted words before counting so uppercase and lowercase forms merge.",
	)
	parser.add_argument(
		"--strip-accents",
		action="store_true",
		help="Remove combining marks after normalization.",
	)
	parser.add_argument(
		"--include-non-greek",
		action="store_true",
		help="Keep words without Greek characters. By default only Greek-containing words are counted.",
	)
	parser.add_argument(
		"--min-word-length",
		type=int,
		default=1,
		help="Ignore counted words shorter than this many characters after normalization.",
	)
	parser.add_argument(
		"--min-count",
		type=int,
		default=1,
		help="Only export all-word rows whose final count is at least this value.",
	)
	parser.add_argument(
		"--quoted-min-count",
		type=int,
		default=1,
		help="Only export quoted words whose final count is at least this value.",
	)
	parser.add_argument(
		"--capitalized-min-count",
		type=int,
		default=1,
		help="Only export capitalized words whose final count is at least this value.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		help="Only export the top K all-word rows after filtering. Use 0 to export every row.",
	)
	parser.add_argument(
		"--quoted-top-k",
		type=int,
		default=1000,
		help="Only export the top K quoted words after filtering. Use 0 to export every row.",
	)
	parser.add_argument(
		"--capitalized-top-k",
		type=int,
		default=1000,
		help="Only export the top K capitalized words after filtering. Use 0 to export every row.",
	)
	parser.add_argument(
		"--flush-threshold",
		type=int,
		default=250000,
		help="Flush in-memory counts to SQLite after this many distinct pending words.",
	)
	parser.add_argument(
		"--report-every",
		type=int,
		default=10000,
		help="Emit a progress line every N scanned documents. Use 0 to disable progress output.",
	)
	parser.add_argument(
		"--no-streaming",
		action="store_true",
		help="Load the split eagerly instead of using datasets streaming.",
	)
	parser.add_argument(
		"--reuse-db",
		action="store_true",
		help="Skip dataset processing and export directly from the existing SQLite databases for the selected count modes.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace existing output files, report JSON, or SQLite databases.",
	)
	return parser.parse_args(argv)


def resolve_count_modes(args: argparse.Namespace) -> list[str]:
	resolved_modes: list[str] = []
	seen_modes: set[str] = set()
	for mode in args.count_modes:
		if mode in seen_modes:
			continue
		resolved_modes.append(mode)
		seen_modes.add(mode)
	return resolved_modes


def contains_greek(text: str) -> bool:
	for char in text:
		codepoint = ord(char)
		if 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
			return True
	return False


def strip_accents(text: str) -> str:
	decomposed = unicodedata.normalize("NFD", text)
	stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
	return unicodedata.normalize("NFC", stripped)


def normalize_text(text: str, args: argparse.Namespace) -> str:
	normalized = text
	if args.normalization != "none":
		normalized = unicodedata.normalize(args.normalization.upper(), normalized)
	normalized = normalized.replace("\u2019", "'").replace("\u02bc", "'")
	return normalized.translate(QUOTE_TRANSLATION)


def normalize_word(word: str, args: argparse.Namespace, preserve_case: bool = False) -> str:
	normalized = word if preserve_case else (word.casefold() if args.casefold else word)
	if args.strip_accents:
		normalized = strip_accents(normalized)
	return normalized


def iter_words(text: str) -> Iterator[str]:
	current = []
	previous_was_letter = False

	for char in text:
		if char.isalpha():
			current.append(char)
			previous_was_letter = True
			continue

		if current and previous_was_letter and char in WORD_CONNECTORS:
			current.append(char)
			previous_was_letter = False
			continue

		if current:
			if current[-1] in WORD_CONNECTORS:
				current.pop()
			if current:
				yield "".join(current)
			current = []
		previous_was_letter = False

	if current:
		if current[-1] in WORD_CONNECTORS:
			current.pop()
		if current:
			yield "".join(current)


def iter_quoted_segments(normalized_text: str) -> Iterator[str]:
	matches: list[tuple[int, int, str]] = []

	for pattern in QUOTED_SEGMENT_PATTERNS:
		for match in pattern.finditer(normalized_text):
			segment = match.group(1).strip()
			if segment:
				matches.append((match.start(), match.end(), segment))

	matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
	selected_spans: list[tuple[int, int, str]] = []
	for start_index, end_index, segment in matches:
		if any(start_index < existing_end and end_index > existing_start for existing_start, existing_end, _ in selected_spans):
			continue
		selected_spans.append((start_index, end_index, segment))

	selected_spans.sort(key=lambda item: item[0])
	for _, _, segment in selected_spans:
		yield segment


def word_passes_filters(word: str, args: argparse.Namespace) -> bool:
	if len(word) < args.min_word_length:
		return False
	if not args.include_non_greek and not contains_greek(word):
		return False
	return True


def filtered_words_from_normalized(normalized_text: str, args: argparse.Namespace, preserve_case: bool = False) -> Iterator[str]:
	for word in iter_words(normalized_text):
		normalized_word = normalize_word(word, args, preserve_case=preserve_case)
		if not word_passes_filters(normalized_word, args):
			continue
		yield normalized_word


def starts_with_capital(word: str) -> bool:
	return bool(word) and word[0].isalpha() and word[0].isupper()


def filtered_capitalized_words_from_normalized(normalized_text: str, args: argparse.Namespace) -> Iterator[str]:
	for word in iter_words(normalized_text):
		if not starts_with_capital(word):
			continue
		normalized_word = normalize_word(word, args, preserve_case=True)
		if not word_passes_filters(normalized_word, args):
			continue
		yield normalized_word


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def mode_output_path(args: argparse.Namespace, mode: str) -> Path:
	if mode == COUNT_MODE_WORDS:
		return args.output_path
	if mode == COUNT_MODE_QUOTED:
		return args.quoted_output_path
	return args.capitalized_output_path


def mode_db_path(args: argparse.Namespace, mode: str) -> Path:
	if mode == COUNT_MODE_WORDS:
		return args.db_path
	if mode == COUNT_MODE_QUOTED:
		return args.quoted_db_path
	return args.capitalized_db_path


def mode_min_count(args: argparse.Namespace, mode: str) -> int:
	if mode == COUNT_MODE_WORDS:
		return args.min_count
	if mode == COUNT_MODE_QUOTED:
		return args.quoted_min_count
	return args.capitalized_min_count


def mode_top_k(args: argparse.Namespace, mode: str) -> Optional[int]:
	if mode == COUNT_MODE_WORDS:
		return None if args.top_k == 0 else args.top_k
	if mode == COUNT_MODE_QUOTED:
		return None if args.quoted_top_k == 0 else args.quoted_top_k
	return None if args.capitalized_top_k == 0 else args.capitalized_top_k


def validate_args(args: argparse.Namespace) -> None:
	args.count_modes = resolve_count_modes(args)

	if not args.count_modes:
		raise SystemExit("At least one count mode must be selected.")
	if args.max_documents is not None and args.max_documents <= 0:
		raise SystemExit("--max-documents must be greater than 0.")
	if args.min_word_length <= 0:
		raise SystemExit("--min-word-length must be greater than 0.")
	if args.min_count <= 0:
		raise SystemExit("--min-count must be greater than 0.")
	if args.quoted_min_count <= 0:
		raise SystemExit("--quoted-min-count must be greater than 0.")
	if args.capitalized_min_count <= 0:
		raise SystemExit("--capitalized-min-count must be greater than 0.")
	if args.flush_threshold <= 0:
		raise SystemExit("--flush-threshold must be greater than 0.")
	if args.report_every < 0:
		raise SystemExit("--report-every cannot be negative.")
	if args.top_k is not None and args.top_k < 0:
		raise SystemExit("--top-k cannot be negative.")
	if args.quoted_top_k is not None and args.quoted_top_k < 0:
		raise SystemExit("--quoted-top-k cannot be negative.")
	if args.capitalized_top_k is not None and args.capitalized_top_k < 0:
		raise SystemExit("--capitalized-top-k cannot be negative.")
	if args.reuse_db and args.overwrite:
		raise SystemExit("--reuse-db and --overwrite cannot be used together.")
	if args.reuse_db:
		missing_db_paths = [str(mode_db_path(args, mode)) for mode in args.count_modes if not mode_db_path(args, mode).exists()]
		if missing_db_paths:
			missing_display = ", ".join(missing_db_paths)
			raise SystemExit(f"SQLite database not found for selected count modes: {missing_display}")


def ensure_clean_target(path: Path, overwrite: bool) -> None:
	if not path.exists():
		return
	if overwrite:
		path.unlink()
		return
	raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --overwrite to replace it.")


def prepare_paths(args: argparse.Namespace) -> None:
	ensure_parent_dir(args.report_path)
	ensure_clean_target(args.report_path, args.overwrite)

	for mode in args.count_modes:
		output_path = mode_output_path(args, mode)
		ensure_parent_dir(output_path)
		ensure_clean_target(output_path, args.overwrite)

	if args.reuse_db:
		return

	for mode in args.count_modes:
		db_path = mode_db_path(args, mode)
		ensure_parent_dir(db_path)
		if not db_path.exists():
			continue
		if args.overwrite:
			db_path.unlink()
			continue
		raise SystemExit(
			f"Refusing to overwrite existing SQLite database: {db_path}. "
			"Use --overwrite to replace it or --reuse-db to export from it."
		)


def open_database(db_path: Path, table_name: str) -> sqlite3.Connection:
	connection = sqlite3.connect(str(db_path))
	connection.execute("PRAGMA journal_mode=WAL")
	connection.execute("PRAGMA synchronous=NORMAL")
	connection.execute("PRAGMA temp_store=MEMORY")
	connection.execute(
		f"CREATE TABLE IF NOT EXISTS {table_name} (word TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
	)
	connection.commit()
	return connection


def flush_counts(connection: sqlite3.Connection, table_name: str, pending_counts: Counter) -> int:
	if not pending_counts:
		return 0

	with connection:
		connection.executemany(
			f"INSERT INTO {table_name}(word, count) VALUES(?, ?) "
			f"ON CONFLICT(word) DO UPDATE SET count = count + excluded.count",
			pending_counts.items(),
		)

	flushed = len(pending_counts)
	pending_counts.clear()
	return flushed


def dataset_iterator(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
	return load_dataset(
		args.dataset_id,
		args.config,
		split=args.split,
		streaming=not args.no_streaming,
		cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
	)


def initialize_mode_stats(mode: str) -> Dict[str, int]:
	stats = {
		"counted_words": 0,
		"flushes": 0,
		"flushed_unique_batches": 0,
	}
	if mode == COUNT_MODE_QUOTED:
		stats["documents_with_quoted_text"] = 0
		stats["quoted_segments"] = 0
	else:
		stats["counted_documents"] = 0
	return stats


def maybe_flush_mode(
	mode: str,
	args: argparse.Namespace,
	connections: Dict[str, sqlite3.Connection],
	pending_counts_by_mode: Dict[str, Counter],
	mode_stats: Dict[str, Dict[str, int]],
) -> None:
	if len(pending_counts_by_mode[mode]) < args.flush_threshold:
		return
	mode_stats[mode]["flushed_unique_batches"] += flush_counts(
		connections[mode],
		MODE_TABLE_NAMES[mode],
		pending_counts_by_mode[mode],
	)
	mode_stats[mode]["flushes"] += 1


def progress_line(
	args: argparse.Namespace,
	scanned_documents: int,
	mode_stats: Dict[str, Dict[str, int]],
	pending_counts_by_mode: Dict[str, Counter],
	elapsed_seconds: float,
) -> str:
	segments = [f"Scanned {scanned_documents} documents"]

	if COUNT_MODE_WORDS in args.count_modes:
		word_stats = mode_stats[COUNT_MODE_WORDS]
		segments.append(
			(
				f"Words docs {word_stats['counted_documents']} | Words {word_stats['counted_words']} | "
				f"Words pending {len(pending_counts_by_mode[COUNT_MODE_WORDS])} | Words flushes {word_stats['flushes']}"
			)
		)

	if COUNT_MODE_QUOTED in args.count_modes:
		quoted_stats = mode_stats[COUNT_MODE_QUOTED]
		segments.append(
			(
				f"Quoted docs {quoted_stats['documents_with_quoted_text']} | Quoted segments {quoted_stats['quoted_segments']} | "
				f"Quoted words {quoted_stats['counted_words']} | Quoted pending {len(pending_counts_by_mode[COUNT_MODE_QUOTED])} | "
				f"Quoted flushes {quoted_stats['flushes']}"
			)
		)

	if COUNT_MODE_CAPITALIZED in args.count_modes:
		capitalized_stats = mode_stats[COUNT_MODE_CAPITALIZED]
		segments.append(
			(
				f"Capitalized docs {capitalized_stats['counted_documents']} | Capitalized words {capitalized_stats['counted_words']} | "
				f"Capitalized pending {len(pending_counts_by_mode[COUNT_MODE_CAPITALIZED])} | "
				f"Capitalized flushes {capitalized_stats['flushes']}"
			)
		)

	segments.append(f"Elapsed {round(elapsed_seconds, 2)}s")
	return " | ".join(segments)


def count_selected_modes(args: argparse.Namespace, connections: Dict[str, sqlite3.Connection]) -> Dict[str, object]:
	pending_counts_by_mode = {mode: Counter() for mode in args.count_modes}
	mode_stats = {mode: initialize_mode_stats(mode) for mode in args.count_modes}
	started_at = time.time()

	scanned_documents = 0

	for record in dataset_iterator(args):
		if args.max_documents is not None and scanned_documents >= args.max_documents:
			break

		scanned_documents += 1

		if args.quality_score_min is not None:
			quality_score = record.get("quality_score")
			if quality_score is None or quality_score < args.quality_score_min:
				continue

		text = record.get(args.text_field)
		if not isinstance(text, str) or not text.strip():
			continue

		normalized_text = normalize_text(text, args)

		if COUNT_MODE_WORDS in args.count_modes:
			matched_words = 0
			for word in filtered_words_from_normalized(normalized_text, args):
				pending_counts_by_mode[COUNT_MODE_WORDS][word] += 1
				mode_stats[COUNT_MODE_WORDS]["counted_words"] += 1
				matched_words += 1
			if matched_words:
				mode_stats[COUNT_MODE_WORDS]["counted_documents"] += 1
			maybe_flush_mode(COUNT_MODE_WORDS, args, connections, pending_counts_by_mode, mode_stats)

		if COUNT_MODE_QUOTED in args.count_modes:
			quoted_segments = 0
			for segment in iter_quoted_segments(normalized_text):
				quoted_segments += 1
				for word in filtered_words_from_normalized(segment, args):
					pending_counts_by_mode[COUNT_MODE_QUOTED][word] += 1
					mode_stats[COUNT_MODE_QUOTED]["counted_words"] += 1
			if quoted_segments:
				mode_stats[COUNT_MODE_QUOTED]["documents_with_quoted_text"] += 1
				mode_stats[COUNT_MODE_QUOTED]["quoted_segments"] += quoted_segments
			maybe_flush_mode(COUNT_MODE_QUOTED, args, connections, pending_counts_by_mode, mode_stats)

		if COUNT_MODE_CAPITALIZED in args.count_modes:
			matched_capitalized_words = 0
			for word in filtered_capitalized_words_from_normalized(normalized_text, args):
				pending_counts_by_mode[COUNT_MODE_CAPITALIZED][word] += 1
				mode_stats[COUNT_MODE_CAPITALIZED]["counted_words"] += 1
				matched_capitalized_words += 1
			if matched_capitalized_words:
				mode_stats[COUNT_MODE_CAPITALIZED]["counted_documents"] += 1
			maybe_flush_mode(COUNT_MODE_CAPITALIZED, args, connections, pending_counts_by_mode, mode_stats)

		if args.report_every and scanned_documents % args.report_every == 0:
			print(
				progress_line(
					args,
					scanned_documents,
					mode_stats,
					pending_counts_by_mode,
					time.time() - started_at,
				),
				file=sys.stderr,
				flush=True,
			)

	for mode in args.count_modes:
		if not pending_counts_by_mode[mode]:
			continue
		mode_stats[mode]["flushed_unique_batches"] += flush_counts(
			connections[mode],
			MODE_TABLE_NAMES[mode],
			pending_counts_by_mode[mode],
		)
		mode_stats[mode]["flushes"] += 1

	return {
		"mode": "count-and-export",
		"dataset_id": args.dataset_id,
		"config": args.config,
		"split": args.split,
		"streaming": not args.no_streaming,
		"max_documents": args.max_documents,
		"quality_score_min": args.quality_score_min,
		"enabled_modes": args.count_modes,
		"scanned_documents": scanned_documents,
		"mode_summaries": mode_stats,
		"elapsed_seconds": round(time.time() - started_at, 2),
	}


def build_export_query(table_name: str, min_count: int, top_k: Optional[int]) -> tuple[str, tuple[object, ...]]:
	query = f"SELECT word, count FROM {table_name} WHERE count >= ? ORDER BY count DESC, word ASC"
	params: list[object] = [min_count]

	if top_k:
		query += " LIMIT ?"
		params.append(top_k)

	return query, tuple(params)


def fetch_single_value(connection: sqlite3.Connection, query: str, params: tuple[object, ...] = ()) -> int:
	result = connection.execute(query, params).fetchone()
	return int(result[0]) if result is not None and result[0] is not None else 0


def fetch_preview_rows(
	connection: sqlite3.Connection,
	export_query: str,
	export_params: tuple[object, ...],
	preview_limit: int = 25,
) -> list[dict[str, object]]:
	preview_query = export_query
	preview_params = export_params
	if " LIMIT ?" in export_query:
		preview_limit = min(preview_limit, int(export_params[-1]))
		preview_params = export_params[:-1] + (preview_limit,)
	else:
		preview_query = export_query + " LIMIT ?"
		preview_params = export_params + (preview_limit,)

	return [
		{"word": word, "count": count}
		for word, count in connection.execute(preview_query, preview_params)
	]


def export_json_counts(output_path: Path, connection: sqlite3.Connection, export_query: str, export_params: tuple[object, ...]) -> int:
	exported_rows = 0
	with output_path.open("w", encoding="utf-8") as output_file:
		output_file.write("[\n")
		for word, count in connection.execute(export_query, export_params):
			if exported_rows:
				output_file.write(",\n")
			output_file.write(json.dumps({"word": word, "count": count}, ensure_ascii=False))
			exported_rows += 1
		output_file.write("\n]\n")
	return exported_rows


def export_text_counts(output_path: Path, connection: sqlite3.Connection, export_query: str, export_params: tuple[object, ...]) -> int:
	exported_rows = 0
	with output_path.open("w", encoding="utf-8") as output_file:
		for word, _ in connection.execute(export_query, export_params):
			output_file.write(f"{word}\n")
			exported_rows += 1
	return exported_rows


def export_counts(args: argparse.Namespace, connections: Dict[str, sqlite3.Connection]) -> Dict[str, object]:
	export_summary: Dict[str, object] = {}

	for mode in args.count_modes:
		connection = connections[mode]
		table_name = MODE_TABLE_NAMES[mode]
		min_count = mode_min_count(args, mode)
		top_k = mode_top_k(args, mode)
		output_path = mode_output_path(args, mode)
		db_path = mode_db_path(args, mode)
		export_query, export_params = build_export_query(table_name, min_count, top_k)

		if mode == COUNT_MODE_WORDS:
			exported_rows = export_json_counts(output_path, connection, export_query, export_params)
			output_format = "json-array"
		else:
			exported_rows = export_text_counts(output_path, connection, export_query, export_params)
			output_format = "line-delimited-text"

		total_unique_words = fetch_single_value(connection, f"SELECT COUNT(*) FROM {table_name}")
		filtered_unique_words = fetch_single_value(
			connection,
			f"SELECT COUNT(*) FROM {table_name} WHERE count >= ?",
			(min_count,),
		)

		export_summary[mode] = {
			"output_path": str(output_path),
			"db_path": str(db_path),
			"output_format": output_format,
			"min_count": min_count,
			"top_k": top_k,
			"total_unique_words": total_unique_words,
			"filtered_unique_words": filtered_unique_words,
			"exported_rows": exported_rows,
			"top_examples": fetch_preview_rows(connection, export_query, export_params),
		}

	return export_summary


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
	report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	validate_args(args)
	prepare_paths(args)

	connections = {
		mode: open_database(mode_db_path(args, mode), MODE_TABLE_NAMES[mode])
		for mode in args.count_modes
	}

	try:
		if args.reuse_db:
			count_summary = {
				"mode": "export-only",
				"dataset_id": args.dataset_id,
				"config": args.config,
				"split": args.split,
				"streaming": not args.no_streaming,
				"enabled_modes": args.count_modes,
			}
		else:
			count_summary = count_selected_modes(args, connections)

		export_summary = export_counts(args, connections)
		report = {
			"dataset": {
				"dataset_id": args.dataset_id,
				"config": args.config,
				"split": args.split,
				"text_field": args.text_field,
			},
			"filters": {
				"count_modes": args.count_modes,
				"normalization": args.normalization,
				"casefold": args.casefold,
				"strip_accents": args.strip_accents,
				"include_non_greek": args.include_non_greek,
				"min_word_length": args.min_word_length,
				"quality_score_min": args.quality_score_min,
				"min_count": args.min_count,
				"quoted_min_count": args.quoted_min_count,
				"capitalized_min_count": args.capitalized_min_count,
				"top_k": mode_top_k(args, COUNT_MODE_WORDS),
				"quoted_top_k": mode_top_k(args, COUNT_MODE_QUOTED),
				"capitalized_top_k": mode_top_k(args, COUNT_MODE_CAPITALIZED),
			},
			"quoted_span_patterns": [pattern.pattern for pattern in QUOTED_SEGMENT_PATTERNS],
			"count_summary": count_summary,
			"export_summary": export_summary,
		}
		write_report(args.report_path, report)
		print(json.dumps(report, ensure_ascii=False, indent=2))
	finally:
		for connection in connections.values():
			connection.close()

	return 0


if __name__ == "__main__":
	exit_code = main()
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(exit_code)