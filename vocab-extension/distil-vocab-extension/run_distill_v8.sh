#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=token-distill-v8
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

PROJECT_DIR="/users/p-skarvelis/glossApi-Tokenizer"
cd "${PROJECT_DIR}"

echo "========================================="
echo " Token Distillation v8 (multi-GPU) — $(date)"
echo " Approach: Pre-compute + stochastic + multi-layer (parallel)"
echo " GPU: 4, Samples: 5000, Steps: 5000, Batch: 64"
echo " Layers: [4,8,16] weights=[0.2,0.5,0.3]"
echo " Resume: YES (every 50 steps)"
echo "========================================="

./run_uenv.sh python -u ${PROJECT_DIR}/vocab-extension/distil-vocab-extension/advanced_token_init.py \
  --token-file ${PROJECT_DIR}/artifacts/vocab_candidates/selected_tokens_v1.txt \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --extended-tokenizer ${PROJECT_DIR}/artifacts/tokenizers/apertus-greek-v1 \
  --output-dir "${SCRATCH}/apertus-greek-tokenizer-distill" \
  --init-strategy retok-distill \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --distill-steps 5000 \
  --distill-lr 1e-3 \
  --distill-samples 5000 \
  2>&1

echo "========================================="
echo " Done — $(date)"
echo "========================================="
