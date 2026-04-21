To train with the full ~80GB `ell_Grek` corpus from `epfml/FineWeb2-HQ` gives the best result, but it is slow for quick iteration.

Estimate on four Clariden nodes (16x GH200)

|--|80GB Full CPT|100MB Targeted CPT|1GB Curated CPT|
|---|---|---|---|
|Time (4x GH200)|~2-4 Days|~30 Minutes|~3-5 Hours|
|Embedding Quality|Excellent|Good|Very Good|
|Language Flow|Native|Slightly "broken"|Natural|
|Overfitting Risk|None|High|Low|

The filtering step is CPU-side preprocessing, so it should run through `uenv` like the other tokenizer and dataset scripts in this repo.

Run the filter with:

```bash
./run_uenv.sh python targetedCPT-DatasetGen/filter.py \
	--overwrite
```

Important notes:

- The script uses the `pyahocorasick` pip package and imports it as `ahocorasick`.
- `requirements.txt` now includes `pyahocorasick`, and `run_uenv.sh` will sync updated repo requirements into `.venv-uenv` before running the command.
- When `SCRATCH` is defined, the default output path is `$SCRATCH/targeted-cpt/curated_greek_cpt.jsonl`.
- The default JSON summary report is written to `artifacts/reports/targeted_cpt_filter_summary.json`.

Filtering strategy:

- Build one Aho-Corasick automaton for all selected tokenizer candidates from `artifacts/vocab_candidates/selected_tokens_v1.txt`.
- Stream the FineWeb2-HQ Greek split so RAM stays bounded.
- Use multiprocessing so Clariden Grace CPU cores do the substring matching in parallel.
- Keep a global counter in the main process and only write documents that still help underfilled targets.
- Stop early once every target has reached `--limit-per-word`, or when an optional document or byte cap is reached.

One implementation detail matters for this repo: many corpus-derived tokenizer candidates are stored with a leading space in `selected_tokens_v1.txt` because that is the right format for tokenizer insertion. The filter script strips that leading space for text search and treats those entries as boundary-sensitive surface-form matches, so they still match raw documents correctly.

Useful options:

- `--limit-per-word 50` keeps up to fifty selected documents per target token.
- `--workers 16` uses more Clariden CPU processes for matching.
- `--max-output-bytes 104857600` stops around a 100MB targeted dataset.
- `--max-output-bytes 1073741824` stops around a 1GB curated dataset.
- `--max-documents 200000` does a shorter dry run.
- `--quality-score-min 4.0` ignores lower-quality FineWeb rows.
- `--report-every 10000` prints progress every 10k scanned documents.

Examples:

```bash
./run_uenv.sh python targetedCPT-DatasetGen/filter.py \
	--limit-per-word 10 \
	--max-output-bytes 104857600 \
	--overwrite
```

```bash
./run_uenv.sh python targetedCPT-DatasetGen/filter.py \
	--limit-per-word 50 \
	--max-output-bytes 1073741824 \
	--workers 16 \
	--overwrite
```

keep in mind the TARGET_WORDS should be in the same format as expected by the tokenizer (e.g., with a leading space if that's how they appear in the tokenized text).
