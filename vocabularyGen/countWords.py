import argparse
import json
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from datasets import load_dataset


DEFAULT_DATASET_ID = "epfml/FineWeb2-HQ"
DEFAULT_CONFIG = "ell_Grek"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_OUTPUT_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.json")
DEFAULT_REPORT_PATH = Path("artifacts/reports/fineweb2_hq_ell_grek_word_count_summary.json")
DEFAULT_DB_PATH = Path("artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.sqlite3")
WORD_CONNECTORS = {"'", "-"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Stream FineWeb2-HQ Greek text on Clariden, count word frequencies, and export a JSON list "
			"sorted by descending count."
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
		help="Destination JSON file with entries shaped like {\"word\": ..., \"count\": ...}.",
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
		help="SQLite sidecar used to store exact counts while streaming the dataset.",
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
		help="Unicode normalization applied before tokenization.",
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
		help="Only export the top K most frequent words after filtering.",
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
		help="Skip dataset processing and export directly from an existing SQLite count database.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace an existing output JSON, report JSON, or SQLite database.",
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
	return normalized


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


def filtered_words(text: str, args: argparse.Namespace) -> Iterator[str]:
	normalized_text = normalize_text(text, args)
	for word in iter_words(normalized_text):
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
	if args.flush_threshold <= 0:
		raise SystemExit("--flush-threshold must be greater than 0.")
	if args.report_every < 0:
		raise SystemExit("--report-every cannot be negative.")
	if args.top_k is not None and args.top_k <= 0:
		raise SystemExit("--top-k must be greater than 0.")
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
		"CREATE TABLE IF NOT EXISTS word_counts (word TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
	)
	connection.commit()
	return connection


def flush_counts(connection: sqlite3.Connection, pending_counts: Counter) -> int:
	if not pending_counts:
		return 0

	with connection:
		connection.executemany(
			"INSERT INTO word_counts(word, count) VALUES(?, ?) "
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


def count_words(args: argparse.Namespace, connection: sqlite3.Connection) -> Dict[str, object]:
	pending_counts: Counter = Counter()
	started_at = time.time()

	scanned_documents = 0
	counted_documents = 0
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

		matched_in_document = 0
		for word in filtered_words(text, args):
			pending_counts[word] += 1
			counted_words += 1
			matched_in_document += 1

		if matched_in_document:
			counted_documents += 1

		if len(pending_counts) >= args.flush_threshold:
			flushed_entries += flush_counts(connection, pending_counts)
			flushes += 1

		if args.report_every and scanned_documents % args.report_every == 0:
			elapsed = round(time.time() - started_at, 2)
			print(
				(
					f"Scanned {scanned_documents} documents | Counted {counted_documents} documents | "
					f"Counted words {counted_words} | Pending unique words {len(pending_counts)} | "
					f"Flushes {flushes} | Elapsed {elapsed}s"
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
		"counted_documents": counted_documents,
		"counted_words": counted_words,
		"flushes": flushes,
		"flushed_unique_batches": flushed_entries,
		"elapsed_seconds": round(time.time() - started_at, 2),
	}


def build_export_query(args: argparse.Namespace) -> tuple[str, tuple[object, ...]]:
	query = "SELECT word, count FROM word_counts WHERE count >= ? ORDER BY count DESC, word ASC"
	params: list[object] = [args.min_count]

	if args.top_k is not None:
		query += " LIMIT ?"
		params.append(args.top_k)

	return query, tuple(params)


def fetch_single_value(connection: sqlite3.Connection, query: str, params: tuple[object, ...] = ()) -> int:
	result = connection.execute(query, params).fetchone()
	return int(result[0]) if result is not None and result[0] is not None else 0


def export_counts(args: argparse.Namespace, connection: sqlite3.Connection) -> Dict[str, object]:
	export_query, export_params = build_export_query(args)
	exported_rows = 0

	with args.output_path.open("w", encoding="utf-8") as output_file:
		output_file.write("[\n")
		for word, count in connection.execute(export_query, export_params):
			if exported_rows:
				output_file.write(",\n")
			output_file.write(json.dumps({"word": word, "count": count}, ensure_ascii=False))
			exported_rows += 1
		output_file.write("\n]\n")

	total_unique_words = fetch_single_value(connection, "SELECT COUNT(*) FROM word_counts")
	filtered_unique_words = fetch_single_value(
		connection,
		"SELECT COUNT(*) FROM word_counts WHERE count >= ?",
		(args.min_count,),
	)

	return {
		"output_path": str(args.output_path),
		"db_path": str(args.db_path),
		"min_count": args.min_count,
		"top_k": args.top_k,
		"total_unique_words": total_unique_words,
		"filtered_unique_words": filtered_unique_words,
		"exported_rows": exported_rows,
	}


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
	report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
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
			count_summary = count_words(args, connection)

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
				"top_k": args.top_k,
			},
			"count_summary": count_summary,
			"export_summary": export_summary,
		}
		write_report(args.report_path, report)
		print(json.dumps(report, ensure_ascii=False, indent=2))
	finally:
		connection.close()


if __name__ == "__main__":
	main()