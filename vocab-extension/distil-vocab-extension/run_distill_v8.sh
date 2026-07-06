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
echo " Unified Token Distillation (single-node, 4 GPU) — $(date)"
echo " Strategy: retok-distill with context sampling and row sync"
echo " GPU: 4, Samples: 5000, Steps: 500, Batch: 16"
echo "========================================="

./run_uenv.sh python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  ${PROJECT_DIR}/vocab-extension/distil-vocab-extension/unified_token_distill.py \
  --token-file ${PROJECT_DIR}/artifacts/vocab_candidates/selected_tokens_v1.txt \
  --base-tokenizer ${PROJECT_DIR}/artifacts/tokenizers/apertus-base \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --extended-tokenizer ${PROJECT_DIR}/artifacts/tokenizers/apertus-greek-v1 \
  --output-dir "${SCRATCH}/apertus-greek-tokenizer-distill-unified-v8" \
  --init-strategy retok-distill \
  --torch-dtype bfloat16 \
  --attn-implementation sdpa \
  --trust-remote-code \
  --distill-steps 500 \
  --distill-lr 1e-5 \
  --distill-samples 5000 \
  --distill-contexts-per-token 8 \
  --distill-max-seq-length 1024 \
  --distill-warmup-steps 50 \
  --distill-batch-size 16 \
  --distill-reg-weight 0.1 \
  --distill-stream-timeout 600 \
  --distill-sync-interval 10 \
  --distill-checkpoint-interval 100 \
  --require-xielu \
  2>&1

echo "========================================="
echo " Done — $(date)"
echo "========================================="
