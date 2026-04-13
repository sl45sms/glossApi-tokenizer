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
from typing import Dict, Iterable, Iterator

from datasets import load_dataset


DEFAULT_DATASET_ID = "epfml/FineWeb2-HQ"
DEFAULT_CONFIG = "ell_Grek"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_OUTPUT_PATH = Path("vocabularyGen/static/quoted_words.txt")
DEFAULT_REPORT_PATH = Path("artifacts/reports/fineweb2_hq_ell_grek_quoted_word_count_summary.json")
DEFAULT_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_word_counts.sqlite3")
WORD_CONNECTORS = {"'", "-"}
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Stream FineWeb2-HQ Greek text on Clariden, count words that appear inside quotes, "
			"double quotes, or backticks, and export the top results to a static token file."
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
		"--output-path",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Destination text file with one extracted quoted word per line.",
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
		help="SQLite sidecar used to store exact quoted-word counts while streaming the dataset.",
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
		help="Unicode normalization applied before quote extraction and tokenization.",
	)
	parser.add_argument(
		"--casefold",
		action="store_true",
		help="Case-fold words before counting so uppercase and lowercase forms merge.",
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
		help="Only export words whose final count is at least this value.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=1000,
		help="Only export the top K most frequent quoted words. Use 0 to export every matching row.",
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
		help="Skip dataset processing and export directly from an existing SQLite quoted-word database.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace an existing output text file, report JSON, or SQLite database.",
	)
	return parser.parse_args()


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


def normalize_word(word: str, args: argparse.Namespace) -> str:
	normalized = word.casefold() if args.casefold else word
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


def iter_quoted_segments(text: str, args: argparse.Namespace) -> Iterator[str]:
	normalized_text = normalize_text(text, args)
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


def filtered_quoted_words(text: str, args: argparse.Namespace) -> Iterator[str]:
	for segment in iter_quoted_segments(text, args):
		for word in iter_words(segment):
			normalized_word = normalize_word(word, args)
			if len(normalized_word) < args.min_word_length:
				continue
			if not args.include_non_greek and not contains_greek(normalized_word):
				continue
			yield normalized_word


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
	if args.max_documents is not None and args.max_documents <= 0:
		raise SystemExit("--max-documents must be greater than 0.")
	if args.min_word_length <= 0:
		raise SystemExit("--min-word-length must be greater than 0.")
	if args.min_count <= 0:
		raise SystemExit("--min-count must be greater than 0.")
	if args.top_k is not None and args.top_k < 0:
		raise SystemExit("--top-k cannot be negative.")
	if args.flush_threshold <= 0:
		raise SystemExit("--flush-threshold must be greater than 0.")
	if args.report_every < 0:
		raise SystemExit("--report-every cannot be negative.")
	if args.reuse_db and args.overwrite:
		raise SystemExit("--reuse-db and --overwrite cannot be used together.")
	if args.reuse_db and not args.db_path.exists():
		raise SystemExit(f"SQLite database not found: {args.db_path}")


def ensure_clean_target(path: Path, overwrite: bool) -> None:
	if not path.exists():
		return
	if overwrite:
		path.unlink()
		return
	raise SystemExit(f"Refusing to overwrite existing file: {path}. Use --overwrite to replace it.")


def prepare_paths(args: argparse.Namespace) -> None:
	ensure_parent_dir(args.output_path)
	ensure_parent_dir(args.report_path)
	ensure_parent_dir(args.db_path)

	ensure_clean_target(args.output_path, args.overwrite)
	ensure_clean_target(args.report_path, args.overwrite)

	if args.reuse_db:
		return

	if args.db_path.exists():
		if args.overwrite:
			args.db_path.unlink()
		else:
			raise SystemExit(
				f"Refusing to overwrite existing SQLite database: {args.db_path}. "
				"Use --overwrite to replace it or --reuse-db to export from it."
			)


def open_database(db_path: Path) -> sqlite3.Connection:
	connection = sqlite3.connect(str(db_path))
	connection.execute("PRAGMA journal_mode=WAL")
	connection.execute("PRAGMA synchronous=NORMAL")
	connection.execute("PRAGMA temp_store=MEMORY")
	connection.execute(
		"CREATE TABLE IF NOT EXISTS quoted_word_counts (word TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
	)
	connection.commit()
	return connection


def flush_counts(connection: sqlite3.Connection, pending_counts: Counter) -> int:
	if not pending_counts:
		return 0

	with connection:
		connection.executemany(
			"INSERT INTO quoted_word_counts(word, count) VALUES(?, ?) "
			"ON CONFLICT(word) DO UPDATE SET count = count + excluded.count",
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


def count_quoted_words(args: argparse.Namespace, connection: sqlite3.Connection) -> Dict[str, object]:
	pending_counts: Counter = Counter()
	started_at = time.time()

	scanned_documents = 0
	counted_documents = 0
	counted_segments = 0
	counted_words = 0
	flushes = 0
	flushed_entries = 0

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

		segments_in_document = list(iter_quoted_segments(text, args))
		if segments_in_document:
			counted_documents += 1
			counted_segments += len(segments_in_document)

		for segment in segments_in_document:
			for word in iter_words(segment):
				normalized_word = normalize_word(word, args)
				if len(normalized_word) < args.min_word_length:
					continue
				if not args.include_non_greek and not contains_greek(normalized_word):
					continue
				pending_counts[normalized_word] += 1
				counted_words += 1

		if len(pending_counts) >= args.flush_threshold:
			flushed_entries += flush_counts(connection, pending_counts)
			flushes += 1

		if args.report_every and scanned_documents % args.report_every == 0:
			elapsed = round(time.time() - started_at, 2)
			print(
				(
					f"Scanned {scanned_documents} documents | Documents with quoted text {counted_documents} | "
					f"Quoted segments {counted_segments} | Counted quoted words {counted_words} | "
					f"Pending unique words {len(pending_counts)} | Flushes {flushes} | Elapsed {elapsed}s"
				),
				file=sys.stderr,
				flush=True,
			)

	if pending_counts:
		flushed_entries += flush_counts(connection, pending_counts)
		flushes += 1

	return {
		"mode": "count-and-export",
		"dataset_id": args.dataset_id,
		"config": args.config,
		"split": args.split,
		"streaming": not args.no_streaming,
		"max_documents": args.max_documents,
		"quality_score_min": args.quality_score_min,
		"scanned_documents": scanned_documents,
		"documents_with_quoted_text": counted_documents,
		"quoted_segments": counted_segments,
		"counted_words": counted_words,
		"flushes": flushes,
		"flushed_unique_batches": flushed_entries,
		"elapsed_seconds": round(time.time() - started_at, 2),
	}


def build_export_query(args: argparse.Namespace) -> tuple[str, tuple[object, ...]]:
	query = "SELECT word, count FROM quoted_word_counts WHERE count >= ? ORDER BY count DESC, word ASC"
	params: list[object] = [args.min_count]

	if args.top_k:
		query += " LIMIT ?"
		params.append(args.top_k)

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
		preview_query = export_query
		preview_params = export_params[:-1] + (preview_limit,)
	else:
		preview_query = export_query + " LIMIT ?"
		preview_params = export_params + (preview_limit,)

	return [
		{"word": word, "count": count}
		for word, count in connection.execute(preview_query, preview_params)
	]


def export_counts(args: argparse.Namespace, connection: sqlite3.Connection) -> Dict[str, object]:
	export_query, export_params = build_export_query(args)
	exported_rows = 0

	with args.output_path.open("w", encoding="utf-8") as output_file:
		for word, _ in connection.execute(export_query, export_params):
			output_file.write(f"{word}\n")
			exported_rows += 1

	total_unique_words = fetch_single_value(connection, "SELECT COUNT(*) FROM quoted_word_counts")
	filtered_unique_words = fetch_single_value(
		connection,
		"SELECT COUNT(*) FROM quoted_word_counts WHERE count >= ?",
		(args.min_count,),
	)

	return {
		"output_path": str(args.output_path),
		"db_path": str(args.db_path),
		"min_count": args.min_count,
		"top_k": None if args.top_k == 0 else args.top_k,
		"total_unique_words": total_unique_words,
		"filtered_unique_words": filtered_unique_words,
		"exported_rows": exported_rows,
		"top_examples": fetch_preview_rows(connection, export_query, export_params),
	}


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
	report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
	args = parse_args()
	validate_args(args)
	prepare_paths(args)

	connection = open_database(args.db_path)

	try:
		if args.reuse_db:
			count_summary = {
				"mode": "export-only",
				"dataset_id": args.dataset_id,
				"config": args.config,
				"split": args.split,
				"streaming": not args.no_streaming,
			}
		else:
			count_summary = count_quoted_words(args, connection)

		export_summary = export_counts(args, connection)
		report = {
			"dataset": {
				"dataset_id": args.dataset_id,
				"config": args.config,
				"split": args.split,
				"text_field": args.text_field,
			},
			"filters": {
				"normalization": args.normalization,
				"casefold": args.casefold,
				"strip_accents": args.strip_accents,
				"include_non_greek": args.include_non_greek,
				"min_word_length": args.min_word_length,
				"quality_score_min": args.quality_score_min,
				"min_count": args.min_count,
				"top_k": None if args.top_k == 0 else args.top_k,
			},
			"quoted_span_patterns": [pattern.pattern for pattern in QUOTED_SEGMENT_PATTERNS],
			"count_summary": count_summary,
			"export_summary": export_summary,
		}
		write_report(args.report_path, report)
		print(json.dumps(report, ensure_ascii=False, indent=2))
	finally:
		connection.close()

	return 0


if __name__ == "__main__":
	exit_code = main()
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(exit_code)