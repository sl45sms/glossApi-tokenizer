#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=retok-init
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/retok_init_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/retok_init_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=%u@cscs.ch
#
# ── Stage 1: Retok Embedding Initialization ─────────────────────────────
#
# Creates an aligned checkpoint where new Greek token embeddings are
# initialized from their BPE subword composition (retok heuristic).
# No distillation — keeps base model quality intact.
#
# Output: ${SCRATCH}/apertus-greek-tokenizer-retok-v1/

set -euo pipefail

PROJECT_DIR="/users/p-skarvelis/glossApi-Tokenizer"
cd "${PROJECT_DIR}"

RETOK_OUTPUT="${SCRATCH}/apertus-greek-tokenizer-retok-v1"
LOG_DIR="${SCRATCH:-/iopsstor/scratch/cscs/${USER}}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================="
echo " Retok Embedding Init — $(date)"
echo " Base: swiss-ai/Apertus-8B-Instruct-2509"
echo " Extended tokenizer: artifacts/tokenizers/apertus-greek-v1"
echo " Tokens: artifacts/vocab_candidates/selected_tokens_v1.txt"
echo " Output: ${RETOK_OUTPUT}"
echo "========================================="

./run_uenv.sh python -u "${PROJECT_DIR}/vocab-extension/distil-vocab-extension/advanced_with_freeweb_and_casual_lm.py" \
  --token-file "${PROJECT_DIR}/artifacts/vocab_candidates/selected_tokens_v1.txt" \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --extended-tokenizer "${PROJECT_DIR}/artifacts/tokenizers/apertus-greek-v1" \
  --output-dir "${RETOK_OUTPUT}" \
  --init-strategy retok \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --overwrite \
  2>&1 | tee "${LOG_DIR}/retok_init_${SLURM_JOB_ID}_console.log"

echo ""
echo "========================================="
echo " Retok checkpoint saved → ${RETOK_OUTPUT}"
echo " Now run: sbatch vocab-extension/retok-cpt/run_retok_cpt_smoke.sh"
echo "========================================="
