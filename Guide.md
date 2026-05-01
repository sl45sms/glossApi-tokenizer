# Greek Tokenizer -> CPT -> SFT Guide

The short version is:

- do not continue from degraded CPT or SFT checkpoints
- keep one clean checkpoint lineage from tokenizer-aligned init -> CPT -> SFT
- use the curated targeted CPT dataset only as a probe, not as the main production CPT corpus
- do not move to the next stage unless the current stage passes a validation gate
- always probe instruct checkpoints with the chat template, not raw text


## 1. Non-Negotiable Rules

Follow these for every new run:

1. Never mix checkpoint lineages.
   - The exact CPT checkpoint you evaluate must be the same one you use for SFT dataset preparation, SFT training, UI checks, and EOS checks.
2. Never promote a checkpoint that loses more than a small amount on GreekMMLU.
   - A large drop means the training recipe is bad, even if the model looks more Greek in a few prompts.
3. Do not use the curated 1GB targeted CPT dataset as the main production CPT corpus.
   - It is fine as a quick probe only.
4. Always pass `MODEL_PATH` explicitly for SFT launchers.
   - Do not rely on launcher defaults.

## 2. Directory Conventions

Use these environment variables consistently:

```bash
export IOPS_ROOT=/iopsstor/scratch/cscs/${USER}
export CAPSTOR_ROOT=/capstor/scratch/cscs/${USER}
export BASE_MODEL=swiss-ai/Apertus-8B-Instruct-2509
export INIT_MODEL=${IOPS_ROOT}/apertus-greek-init
```

Use one explicit run name per stage.

Examples:

- `apertus-greek-cpt-probe-curated-1GB-100steps`
- `apertus-greek-cpt-full-1btok-500steps`
- `apertus-greek-sft-from-cpt-full-1btok-500steps`


## 3. Stage A: Tokenizer And Aligned Init Checkpoint

### A1. Extract word statistics

```bash
./run_uenv.sh python vocabularyGen/countWords.py \
  --count-modes words quoted capitalized \
  --report-every 10000 \
  --overwrite
```
that creates the main word-count JSON plus the generated text exports `artifacts/vocab_candidates/fineweb2_hq_ell_grek_quoted_words.txt` and `artifacts/vocab_candidates/fineweb2_hq_ell_grek_capitalized_words.txt`, together with their SQLite sidecars. The generated text exports are inspection artifacts; `vocabularyGen/static/` stays reserved for curated manual token files.

### A2. Select tokenizer candidates

```bash
./run_uenv.sh python vocabularyGen/selectTokenizerCandidates.py \
  --min-count 5 \
  --min-base-token-count 4 \
  --min-base-token-count-high-frequency 5 \
  --max-selected 5000 \
  --overwrite
```
that writes `artifacts/vocab_candidates/fineweb2_hq_ell_grek_candidates.tsv`, `artifacts/vocab_candidates/selected_tokens_v1.txt`, and `artifacts/reports/fineweb2_hq_ell_grek_candidate_selection.json` for the next step. The selector merges the regular, quoted, and capitalized SQLite count sources, keeps a stricter default guard for very frequent corpus words, and only then appends curated static tokens from `vocabularyGen/static/`.

### A3. Build tokenizer and aligned init checkpoint

```bash
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
  --base-model ${BASE_MODEL} \
  --checkpoint-output-dir ${INIT_MODEL} \
  --torch-dtype bfloat16 \
  --overwrite
```
By default, the script mean-initializes the new input embeddings from the original subtoken decomposition and uses a more conservative zero initialization for untied output-head rows. If you explicitly want the older untied-head behavior, pass `--untied-output-init-strategy mean`.

### A4. Validation gate for aligned init

Before any CPT, compare the aligned init checkpoint against base.

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model ${BASE_MODEL} \
  --trained-model ${INIT_MODEL} \
  --output-json artifacts/reports/greek_mmlu_init_eval.json
```

Acceptance rule:

- If aligned init is already badly worse than base, stop here and debug tokenizer extension or embedding initialization.


## 4. Stage B: Targeted CPT Probe Only

This stage is only a probe to see whether the new tokens can be introduced without obvious damage.

Do not treat this as the production CPT run.

### B1. Build the curated targeted Greek dataset

```bash
./run_uenv.sh python targetedCPT-DatasetGen/filter.py \
  --limit-per-word 50 \
  --max-output-bytes 1073741824 \
  --workers 16 \
  --overwrite
```

### B2. Pack the curated CPT dataset

```bash
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
  --output-dir ${IOPS_ROOT}/prepared-datasets/apertus-greek-targeted-packed-2048 \
  --greek-dataset ${IOPS_ROOT}/targeted-cpt/curated_greek_cpt.jsonl \
  --greek-probability 0.9 \
  --english-dataset epfml/FineWeb-HQ \
  --english-probability 0.1 \
  --overwrite
```

The current repo artifact for this dataset is about `114,731,008` output tokens.

At global batch `256` and sequence length `2048`, each optimizer step consumes:

$$
256 \times 2048 = 524{,}288 \text{ tokens/step}
$$

That means one full pass over the curated dataset is only about:

$$
114{,}731{,}008 / 524{,}288 \approx 219 \text{ steps}
$$

So the old 400-step and 1000-step curated runs were already too much.

### B3. Run a short CPT probe only

Use a small step budget that stays well below one full pass.

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${INIT_MODEL}
export OUTPUT_DIR=${CAPSTOR_ROOT}/apertus-greek-cpt-probe-curated-1GB-100steps
export PREPARED_TRAIN_DATASET_DIR=${IOPS_ROOT}/prepared-datasets/apertus-greek-targeted-packed-2048
export ATTN_IMPLEMENTATION=sdpa
export GRADIENT_CHECKPOINTING=0
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export TARGET_GLOBAL_BATCH_SIZE=256
unset GRADIENT_ACCUMULATION_STEPS
export BENCHMARK_MODE=0
export SMOKE_TEST=0
export SKIP_WARMUP=0
export WARMUP_MAX_STEPS=25
export FULL_MAX_STEPS=75
export FULL_WARMUP_STEPS=10

sbatch --nodes=4 --time=12:00:00 scripts/run_apertus_greek_cpt_clariden_multinode.sh
```

### B4. Validation gate for the probe

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model ${BASE_MODEL} \
  --trained-model ${CAPSTOR_ROOT}/apertus-greek-cpt-probe-curated-1GB-100steps/final \
  --output-json artifacts/reports/greek_mmlu_probe_eval.json
```

Acceptance rule:

- If this probe loses more than a small amount against base, do not continue training this curated branch.
- If it regresses hard, discard the branch and move to the larger production CPT path below.


## 6. Stage C: Production CPT On A Larger Corpus

This is the recommended main path.

Use a much larger Greek+English mixture so the model is not overfit to a tiny biased corpus.

### C1. Prepare a much larger packed CPT dataset

Start with roughly one billion output tokens.

```bash
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
  --output-dir ${IOPS_ROOT}/prepared-datasets/apertus-greek-full-packed-2048-1btok \
  --greek-dataset epfml/FineWeb2-HQ \
  --greek-config ell_Grek \
  --greek-probability 0.9 \
  --english-dataset epfml/FineWeb-HQ \
  --english-probability 0.1 \
  --max-output-tokens 1073741824 \
  --overwrite
```

This gives a dataset large enough that a few hundred steps are no longer repeated passes over a tiny curated set.

### C2. Run the first production CPT pass

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${INIT_MODEL}
export OUTPUT_DIR=${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps
export PREPARED_TRAIN_DATASET_DIR=${IOPS_ROOT}/prepared-datasets/apertus-greek-full-packed-2048-1btok
export ATTN_IMPLEMENTATION=sdpa
export GRADIENT_CHECKPOINTING=0
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export TARGET_GLOBAL_BATCH_SIZE=256
unset GRADIENT_ACCUMULATION_STEPS
export BENCHMARK_MODE=0
export SMOKE_TEST=0
export SKIP_WARMUP=0
export WARMUP_MAX_STEPS=100
export FULL_MAX_STEPS=400
export FULL_WARMUP_STEPS=50

sbatch --nodes=4 --time=12:00:00 scripts/run_apertus_greek_cpt_clariden_multinode.sh
```

### C3. Validation gate for production CPT

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model ${BASE_MODEL} \
  --trained-model ${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final \
  --output-json artifacts/reports/greek_mmlu_cpt_full_eval.json
```

Acceptance rule:

- Only continue from the best CPT checkpoint that is stable or improved.
- If it is still clearly worse than base, do not move to SFT yet.


## 7. Stage D: Manual EOS And Chat-Format Checks

Before SFT, run a quick manual probe with the fixed EOS tool.

```bash
./run_uenv.sh python tools/analyzeEos.py \
  --model-path ${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final \
  --device cuda \
  --max-new-tokens 128
```

Expected behavior:

- the script should report `Chat template used: True`
- the answer should stop naturally more often than before
- if EOS is still never emitted, check the exact checkpoint you are probing and do not proceed blindly

You can also inspect the same checkpoint in the UI:

```bash
APERTUS_MODEL_UI_PORT=8631 ./run_model_ui.sh \
  --base-device cuda:0 \
  --device cuda:1 \
  --model-path ${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final
```


## 8. Stage E: Prepare The SFT Dataset

Use the exact same CPT checkpoint lineage that passed the CPT gate.

```bash
OVERWRITE=1 \
VALIDATION_SAMPLES=2048 \
MAX_SEQ_LENGTH=1024 \
MODEL_PATH=${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final \
sbatch scripts/run_prepare_sft_dataset_clariden.sh
```

This writes:

```text
${IOPS_ROOT}/prepared-datasets/apertus-greek-sft-1024-left-val2048
```


## 9. Stage F: SFT Smoke Test First

Do not jump directly into a long multi-node SFT run.

```bash
SMOKE_TEST=1 \
VALIDATION_SAMPLES=16 \
MODEL_PATH=${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final \
PREPARED_DATASET_DIR=${IOPS_ROOT}/prepared-datasets/apertus-greek-sft-1024-left-val2048 \
OUTPUT_DIR=${CAPSTOR_ROOT}/apertus-greek-sft-smoke \
sbatch --nodes=1 SFT/run_apertus_greek_sft_clariden.sh
```

Then probe EOS again on the smoke checkpoint or final smoke output.


## 10. Stage G: Full SFT

Only after the SFT smoke test looks sane:

```bash
MODEL_PATH=${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final \
PREPARED_DATASET_DIR=${IOPS_ROOT}/prepared-datasets/apertus-greek-sft-1024-left-val2048 \
VALIDATION_SAMPLES=2048 \
OUTPUT_DIR=${CAPSTOR_ROOT}/apertus-greek-sft-full-1btok-500steps \
sbatch --nodes=4 --time=12:00:00 SFT/run_apertus_greek_sft_clariden_multinode.sh
```

Resume from the latest checkpoint if the job hits the 12-hour limit:

```bash
RESUME_FROM_CHECKPOINT=${CAPSTOR_ROOT}/apertus-greek-sft-full-1btok-500steps/checkpoint-XXXX \
MODEL_PATH=${CAPSTOR_ROOT}/apertus-greek-cpt-full-1btok-500steps/final \
PREPARED_DATASET_DIR=${IOPS_ROOT}/prepared-datasets/apertus-greek-sft-1024-left-val2048 \
VALIDATION_SAMPLES=2048 \
OUTPUT_DIR=${CAPSTOR_ROOT}/apertus-greek-sft-full-1btok-500steps \
sbatch --nodes=4 --time=12:00:00 SFT/run_apertus_greek_sft_clariden_multinode.sh
```


## 11. Stage H: Final Evaluation

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model ${BASE_MODEL} \
  --trained-model ${CAPSTOR_ROOT}/apertus-greek-sft-full-1btok-500steps \
  --output-json artifacts/reports/greek_mmlu_sft_final_eval.json
```

Optional plots:

```bash
./run_uenv.sh python evaluation/plot_greek_mmlu_report.py \
  artifacts/reports/greek_mmlu_sft_final_eval.json \
  --output-dir artifacts/reports/greek_mmlu_sft_final_eval_plots
```


## 12. Quick Checklist

Before moving from one stage to the next, verify all of the following:

- Same checkpoint lineage is used everywhere.
- `MODEL_PATH` is always set explicitly for SFT.
- GreekMMLU does not collapse relative to the previous stage.
- EOS checks are run with the fixed `tools/analyzeEos.py`.
- You do not run long CPT on the small curated 1GB dataset.


## 13. What Not To Repeat

Do not do these again:

- do not use the 1GB curated targeted dataset for 400-1000 total CPT steps at global batch 256
- do not compare one CPT checkpoint in eval and a different one in SFT prep/training
- do not use raw-text prompting to judge instruct checkpoint EOS behavior
- do not start a long SFT run before a smoke run and an EOS sanity check


## 14. One-Line Summary

Restart from the aligned init checkpoint, use the curated targeted CPT dataset only for a short probe, move the real CPT run onto a much larger Greek+English corpus, keep one clean checkpoint lineage, and only then do SFT with the fixed EOS supervision.