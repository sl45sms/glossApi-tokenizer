Use `countWords.py` as the unified preprocessor for the Greek `ell_Grek` subset of `epfml/FineWeb2-HQ`.
It can count regular words, extract words from quoted spans, and collect observed capitalized forms such as sentence-initial words, proper names, countries, months, weekdays, and holidays in a single streaming pass.


Run it through `uenv` like the other repo scripts:

```bash
./run_uenv.sh python vocabularyGen/countWords.py \
  --report-every 10000
```

To generate the full tokenizer-mining bundle in one pass, including quoted words and capitalized words, run:

```bash
./run_uenv.sh python vocabularyGen/countWords.py \
  --count-modes words quoted capitalized \
  --report-every 10000 \
  --overwrite
```

Useful options:

- `--count-modes words quoted capitalized` enables all three extractors in one run.
- `--min-count 5` exports only words with at least five occurrences.
- `--top-k 50000` exports only the most frequent 50k words.
- `--quoted-top-k 2000` exports a larger quoted-word text export.
- `--capitalized-top-k 2000` exports a larger capitalized-word text export.
- `--casefold` merges uppercase and lowercase forms.
- `--strip-accents` removes combining accents after normalization.
- `--include-non-greek` keeps Latin-script or mixed-script words instead of filtering to Greek-containing words only.
- `--reuse-db` skips dataset processing and re-exports from the existing SQLite databases for the selected count modes.

This is a CPU-side preprocessing step and fits the Clariden `uenv` workflow described in the repository runbook.

The default all-word output locations are:

- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.json`
- `artifacts/reports/fineweb2_hq_ell_grek_word_count_summary.json`
- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_word_counts.sqlite3`


When `quoted` mode is enabled, the same run also writes:

- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_words.txt`
- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_word_counts.sqlite3`

When `capitalized` mode is enabled, the same run also writes:

- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_capitalized_words.txt`
- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_capitalized_word_counts.sqlite3`

The summary report stays in:

- `artifacts/reports/fineweb2_hq_ell_grek_word_count_summary.json`

It includes separate export sections for every enabled mode.


Quoted mode extracts words from text enclosed by single quotes, double quotes, or backticks, and it also normalizes common curly quote variants such as `“...”` and `«...»` before matching.
Capitalized mode keeps observed forms whose first letter is uppercase in the corpus, so sentence-initial variants and proper-name-like spellings still contribute through the capitalized SQLite source even when corpus word counts are case-folded or later case-collapsed by the selector.
The generated quoted/capitalized text exports are inspection artifacts, not curated exact-token inputs. If you want to force exact surface forms into the final token list, copy only the chosen entries into a curated file under `vocabularyGen/static/`.

If you only want quoted-word extraction, run the unified script directly in quoted-only mode:

```bash
./run_uenv.sh python vocabularyGen/countWords.py \
  --count-modes quoted \
  --overwrite
```

That writes:

- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_words.txt`
- `artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_word_counts.sqlite3`

The shared summary report remains at:

- `artifacts/reports/fineweb2_hq_ell_grek_word_count_summary.json`

By default quoted and capitalized exports both write the top 1000 rows for their mode. Use `--quoted-top-k 0` or `--capitalized-top-k 0` to export every matching row above the corresponding `--quoted-min-count` or `--capitalized-min-count` threshold.


* The SQLite files are intentional. The Greek FineWeb2-HQ split is about 84GB, so keeping exact counts only in memory is not reliable. The script streams the dataset and flushes batched counts to SQLite, then exports the final sorted JSON from that database.

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

The selector uses only the base Apertus tokenizer. By default it combines the regular word-count database with the quoted-word and capitalized-word SQLite databases when those exist, then ranks the merged catalog by real corpus frequency and base-tokenizer fragmentation. It does not generate artificial stems as tokens.
Case variants are collapsed by default for corpus-derived counts, so entries like `Δημιουργία` and `δημιουργία` become a single candidate and the lowercase form is preferred when it exists unless you opt into `--preserve-case-variants`.
Corpus-selected tokens are written to `selected_tokens_v1.txt` with a single leading space so they match normal in-text word boundaries for this tokenizer family.
The selector also reads curated files in `vocabularyGen/static/` by default. Each non-empty line is treated as a static token candidate, hyphens are removed from the line, and the cleaned static token is appended to `selected_tokens_v1.txt` exactly as written.

Useful selector options:

- `--top-k-input 200000` only scores the top 200k counted words.
- `--skip-quoted-counts` falls back to the regular word-count source plus curated static files only.
- `--skip-capitalized-counts` excludes the capitalized SQLite counts from the merged catalog while still allowing curated static files.
- `--min-base-token-count 3` keeps only words that currently split into at least 3 base-tokenizer pieces.
- `--min-base-token-count 4` is a stricter pass if you want only heavily fragmented words.
- `--preserve-case-variants` keeps uppercase and lowercase variants as separate candidates if you explicitly want that.
- `--skip-static-files` disables the extra curated static token injection step.
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

If you also want a model checkpoint with resized embeddings initialized from the mean of the original subtoken embeddings, pass a base model path or model id:

```bash
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --checkpoint-output-dir /iopsstor/scratch/cscs/p-skarvelis/apertus-greek-init \
  --torch-dtype bfloat16 \
  --overwrite
```

In that mode, the script computes each new token's initialization from the base tokenizer decomposition before the token is added, averages the corresponding input embeddings, and applies the same mean initialization to the LM head when it is not tied to the input embedding matrix.
The checkpoint path flags `--model-output-dir`, `--checkpoint-output-dir`, and `--checkpoint-storage-path` are aliases for the same setting. When `SCRATCH` is defined, the default checkpoint location is `$SCRATCH/apertus-greek-init`.
