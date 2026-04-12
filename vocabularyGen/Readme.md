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