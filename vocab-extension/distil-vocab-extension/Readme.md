# Unified Distillation Pipeline

This folder had scripts for token initialization and
distillation. The entrypoint is:

- `vocab-extension/distil-vocab-extension/unified_token_distill.py`

It has three strategies for CLI with:

- `--init-strategy weighted-mean`
- `--init-strategy retok`
- `--init-strategy retok-distill`

`retok-distill` supports distributed runs through `torchrun` across all GPUs and
multiple Alps/Clariden nodes.

The script enforces xIELU CUDA by default (`--require-xielu`).


## Single-Node Example (uenv)

```bash
cd /users/${USER}/glossApi-Tokenizer

./run_uenv.sh python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  vocab-extension/distil-vocab-extension/unified_token_distill.py \
  --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
  --base-tokenizer artifacts/tokenizers/apertus-base \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --extended-tokenizer artifacts/tokenizers/apertus-greek-v1 \
  --output-dir "${SCRATCH}/apertus-greek-tokenizer-distill-unified" \
  --init-strategy retok-distill \
  --max-trainable-tokens 1500 \
  --torch-dtype bfloat16 \
  --attn-implementation sdpa \
  --distill-steps 500 \
  --distill-lr 5e-6 \
  --distill-samples 5000 \
  --distill-contexts-per-token 8 \
  --distill-max-seq-length 1024 \
  --distill-warmup-steps 50 \
  --distill-batch-size 16 \
  --distill-reg-weight 0.1 \
  --distill-stream-timeout 600 \
  --distill-sync-interval 10 \
  --distill-sync-start-step 50 \
  --distill-checkpoint-interval 100 \
  --distill-max-invalid-steps 20 \
  --distill-lr-decay-on-invalid 0.5 \
  --report-path artifacts/reports/unified_token_distill_report.json \
  --require-xielu
```


## Multi-Node Clariden Example

Use the dedicated launcher:

- `scripts/run_apertus_tokenizer_distill_clariden_multinode.sh`

```bash
cd /users/${USER}/glossApi-Tokenizer

# Optional overrides
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified"
export DISTILL_STEPS=500
export DISTILL_LR=5e-6
export MAX_TRAINABLE_TOKENS=1500
export NPROC_PER_NODE=4
export DIST_TIMEOUT_SECONDS=7200
export DISTILL_SYNC_START_STEP=50
export DISTILL_MAX_INVALID_STEPS=20
export DISTILL_LR_DECAY_ON_INVALID=0.5

sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```

Notes:

- `OUTPUT_DIR` must be on a shared filesystem visible to all nodes.
- The launcher verifies xIELU before starting distributed training.
- Context caching is reused automatically unless `REFRESH_CONTEXT_CACHE=1`.
- If `OUTPUT_DIR` already has distill checkpoints, the run resumes. For a clean restart set `OVERWRITE_OUTPUT_DIR=1`.
- The pipeline is now stage-first by default (`MAX_TRAINABLE_TOKENS=1500`) to avoid a one-shot 5000-token shift.
- Deferred tokens are written to `OUTPUT_DIR/deferred_tokens_next_stage.txt`.


## Stage Continuation (Recommended)

After stage 1 finishes, continue with the deferred token list:

```bash
export BASE_MODEL="${OUTPUT_DIR}"
export BASE_TOKENIZER="${OUTPUT_DIR}"
export TOKEN_FILE="${OUTPUT_DIR}/deferred_tokens_next_stage.txt"
export OVERWRITE_OUTPUT_DIR=0
sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```

Repeat until the deferred token file is empty.


## Optional GreekMMLU Gate

You can evaluate after distillation and fail the run if it is below base:

```bash
export RUN_GREEK_MMLU_EVAL=1
export FAIL_BELOW_BASE=1
export USE_CHAT_TEMPLATE=0
export DIST_TIMEOUT_SECONDS=7200
export DISTILL_SYNC_START_STEP=50
export DISTILL_MAX_INVALID_STEPS=20
export DISTILL_LR_DECAY_ON_INVALID=0.5
sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```