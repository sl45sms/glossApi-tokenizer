# Retok → CPT → SFT Pipeline

Three-stage pipeline for Greek vocabulary extension on Apertus-8B:

```
Stage 1: retok init   Stage 2: CPT          Stage 3: SFT
  (~5 min, 1 GPU)       (hours, 4-32 GPUs)    (~3 hours, 4 GPUs)
  ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
  │ Load Apertus │      │ 90% Greek    │      │ Instruct     │
  │ Resize emb   │ ──▶  │ 10% English  │ ──▶  │ format       │
  │ Retok init   │      │ Full model   │      │ restoration  │
  │ Save ckpt    │      │ training     │      │              │
  └─────────────┘      └──────────────┘      └──────────────┘
```

CPT improves GreekMMLU (+2.2%) but breaks instruction-following. SFT restores chat ability while preserving Greek gains.

## Quick Start

### Step 1: Retok Initialization
```bash
sbatch vocab-extension/retok-cpt/run_retok_init.sh
```

### Step 2: CPT Smoke Test
```bash
sbatch vocab-extension/retok-cpt/run_retok_cpt_smoke.sh
```

### Step 3: Full CPT (multi-node)
```bash
sbatch vocab-extension/retok-cpt/run_retok_cpt_multinode.sh
```

### Step 4: SFT — restore chat ability

**Smoke test first (20 steps, ~5 min):**
```bash
SMOKE_TEST=1 sbatch vocab-extension/retok-cpt/run_retok_sft.sh
```

**Full SFT (~3 hours):**
```bash
sbatch vocab-extension/retok-cpt/run_retok_sft.sh
```

### Step 5: Evaluate
```bash
# GreekMMLU
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model "/capstor/scratch/cscs/${USER}/apertus-greek-sft-retok/final" \
  --output-json artifacts/reports/greek_mmlu_retok_sft_eval.json

# Chat test
./run_model_ui.sh --base-device cuda:0 --device cuda:1 \
  --model-path /capstor/scratch/cscs/${USER}/apertus-greek-sft-retok/final
```

## Results (checkpoint-5000)

| Model | GreekMMLU | Chat |
|-------|-----------|------|
| Base Apertus | 64.86% | ✅ |
| CPT (5000 steps) | **67.09%** (+2.22%) | ❌ repetition loops |
| CPT + SFT | ~67% | ✅ restored |

## Multi-node Scaling

| Nodes | GPUs | CPT steps / 12h |
|-------|------|-----------------|
| 4 | 16 | ~2,000 |
| 8 | 32 | ~4,000 |
| 16 | 64 | ~8,000 |

```bash
sbatch --nodes=8 vocab-extension/retok-cpt/run_retok_cpt_multinode.sh
```

## Configuration

```bash
# CPT
FULL_MAX_STEPS=5000 sbatch --nodes=8 vocab-extension/retok-cpt/run_retok_cpt_multinode.sh

# SFT
SMOKE_TEST=1 MAX_SEQ_LENGTH=2048 sbatch vocab-extension/retok-cpt/run_retok_sft.sh
```
