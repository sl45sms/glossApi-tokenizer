Use this script to stream the Greek `ell_Grek` subset of `epfml/FineWeb2-HQ`, count word frequencies, and export a JSON list sorted by descending count.

The main output file is:

```json
[
  {"word": "xxx", "count": 123},
  {"word": "zzz", "count": 67},
  {"word": "yyy", "count": 45}
]
```

The default output locations are:

- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.json`
- `artifacts/reports/fineweb2_hq_ell_grek_word_count_summary.json`
- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.sqlite3`

The SQLite file is intentional. The Greek FineWeb2-HQ split is about 84GB, so keeping exact counts only in memory is not reliable. The script streams the dataset and flushes batched counts to SQLite, then exports the final sorted JSON from that database.

Run it through `uenv` like the other repo scripts:

```bash
./run_uenv.sh python vocabularyGen/countWords.py \
  --report-every 10000
```

Useful options:

- `--min-count 5` exports only words with at least five occurrences.
- `--top-k 50000` exports only the most frequent 50k words.
- `--casefold` merges uppercase and lowercase forms.
- `--strip-accents` removes combining accents after normalization.
- `--include-non-greek` keeps Latin-script or mixed-script words instead of filtering to Greek-containing words only.
- `--reuse-db` skips dataset processing and re-exports JSON from an existing SQLite database.

This is a CPU-side preprocessing step and fits the Clariden `uenv` workflow described in the repository runbook.

After generating the counts, you can rank real observed words into tokenizer candidates:

```bash
./run_uenv.sh python vocabularyGen/selectTokenizerCandidates.py \
  --min-count 5 \
  --min-base-token-count 3 \
  --max-selected 5000 \
  --overwrite
```

That selector writes:

- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_candidates.tsv`
- `artifacts/vocab_candidates/selected_tokens_v1.txt`
- `artifacts/reports/fineweb2_hq_ell_grek_candidate_selection.json`

The selector uses only the base Apertus tokenizer. By default it keeps words that the base tokenizer splits into 3 or more tokens, then ranks them by real corpus frequency and base-tokenizer fragmentation. It does not generate artificial stems as tokens.
Case variants are collapsed by default, so entries like `Δημιουργία` and `δημιουργία` become a single candidate and the lowercase form is preferred when it exists.
Corpus-selected tokens are written to `selected_tokens_v1.txt` with a single leading space so they match normal in-text word boundaries for this tokenizer family.
The selector also reads every file in `vocabularyGen/static/` by default. Each non-empty line is treated as a static token candidate, hyphens are removed from the line, and the cleaned static token is appended exactly as written when the base tokenizer does not already contain it as an exact single token.

Useful selector options:

- `--top-k-input 200000` only scores the top 200k counted words.
- `--min-base-token-count 3` keeps only words that currently split into at least 3 base-tokenizer pieces.
- `--min-base-token-count 4` is a stricter pass if you want only heavily fragmented words.
- `--preserve-case-variants` keeps uppercase and lowercase variants as separate candidates if you explicitly want that.
- `--skip-static-files` disables the extra static token injection step.
- `--max-selected 2000` gives you a smaller, more conservative first token list.

To turn the selected token list into a saved tokenizer directory, run:

```bash
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
  --overwrite
```

That writes:

- `artifacts/tokenizers/apertus-greek-v1`
- `artifacts/tokenizers/apertus-greek-v1/tokenizer_readable.json`
- `artifacts/reports/tokenizer_apertus_greek_v1.json`

The extension script now uses the token list verbatim. It does not create extra with-space and without-space variants automatically, so the spacing behavior is controlled entirely by what `selectTokenizerCandidates.py` writes.

To extract words that appear inside quoted spans and write the most common ones into the static token folder, run:

```bash
./run_uenv.sh python vocabularyGen/countQuotedWords.py \
  --overwrite
```

That writes:

- `vocabularyGen/static/quoted_words.txt`
- `artifacts/reports/fineweb2_hq_ell_grek_quoted_word_count_summary.json`
- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_word_counts.sqlite3`

By default it exports the top 1000 quoted words. Use `--top-k 2000` for a larger static file or `--top-k 0` to export every quoted word above `--min-count`.
It extracts words from text enclosed by single quotes, double quotes, or backticks, and it also normalizes common curly quote variants such as `“...”` and `«...»` before matching.