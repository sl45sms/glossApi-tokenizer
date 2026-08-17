#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=token-distill-v19
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/distill_v19_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/distill_v19_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=panagiotis@skarvelis.gr

VERSION="v19"
PROJECT_DIR="/users/p-skarvelis/glossApi-Tokenizer"
cd "${PROJECT_DIR}"

# Ensure log dir exists
LOG_DIR="${SCRATCH:-/iopsstor/scratch/cscs/${USER}}/logs"
mkdir -p "${LOG_DIR}"

# Pre-set HF cache so datasets library caches FineWeb2-HQ locally
export HF_DATASETS_CACHE="${SCRATCH}/FineWeb2-HQ"
mkdir -p "${HF_DATASETS_CACHE}"

echo "========================================="
echo " Unified Token Distillation ${VERSION} — $(date)"
echo " Approach: Aho-Corasick + distributed row-sync + xIELU required"
echo " GPU: 4, Samples: 5000, Steps: 500, Batch: 16"
echo " Contexts/token: 8, Max seq len: 1024, LR warmup: 50 steps"
echo " LR: 1e-5, Reg weight: 0.1, Stream timeout: 600s"
echo " FineWeb2-HQ cache: ${HF_DATASETS_CACHE}"
echo " Log: ${LOG_DIR}/distill_v19_${SLURM_JOB_ID}_console.log"
echo "========================================="

# Unified distillation with distributed torchrun on all local GPUs.
./run_uenv.sh python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  ${PROJECT_DIR}/vocab-extension/distil-vocab-extension/unified_token_distill.py \
  --token-file ${PROJECT_DIR}/artifacts/vocab_candidates/selected_tokens_v1.txt \
  --base-tokenizer ${PROJECT_DIR}/artifacts/tokenizers/apertus-base \
  --base-model swiss-ai/Apertus-8B-2509 \
  --extended-tokenizer ${PROJECT_DIR}/artifacts/tokenizers/apertus-greek-v1 \
  --output-dir "${SCRATCH}/apertus-greek-tokenizer-distill-unified-freeweb-${VERSION}" \
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
  --fineweb2-cache-dir "${SCRATCH}/FineWeb2-HQ" \
  --require-xielu \
  2>&1 | tee "${LOG_DIR}/distill_v19_${SLURM_JOB_ID}_console.log"

echo "========================================="
echo " Done — $(date)"
echo "========================================="
