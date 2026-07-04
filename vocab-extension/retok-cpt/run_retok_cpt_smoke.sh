#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=retok-cpt-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/retok_cpt_smoke_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/retok_cpt_smoke_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=%u@cscs.ch
#
# ── Stage 2a: CPT Smoke Test (100 steps) ───────────────────────────────
#
# Quick validation that the retok checkpoint loads and trains correctly.
# Run this BEFORE the full CPT run.
#
# Prerequisite: run_retok_init.sh must have completed successfully.

set -euo pipefail

REPO_ROOT="/users/p-skarvelis/glossApi-Tokenizer"

ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
	set -a
	source "${ENV_FILE}"
	set +a
fi

export HF_TOKEN="${HF_TOKEN:-}"

CE_ENVIRONMENT="${CE_ENVIRONMENT:-apertus-greek-clariden}"
EDF_PATH="${HOME}/.edf/${CE_ENVIRONMENT}.toml"
IOPS_SCRATCH_ROOT="${IOPS_SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}"
CAPSTOR_SCRATCH_ROOT="${CAPSTOR_SCRATCH_ROOT:-/capstor/scratch/cscs/${USER}}"

# ── Point to the retok checkpoint ──────────────────────────────────────
MODEL_PATH="${SCRATCH}/apertus-greek-tokenizer-retok-v1"
OUTPUT_DIR="${CAPSTOR_SCRATCH_ROOT}/apertus-greek-cpt-retok-smoke"
RUN_NAME="apertus-greek-cpt-retok-smoke"
STAGE_ROOT="${SCRATCH}/glossapi-tokenizer_${SLURM_JOB_ID}"

if [[ ! -d "${MODEL_PATH}" ]]; then
	echo "ERROR: Retok checkpoint not found at ${MODEL_PATH}" >&2
	echo "Run sbatch vocab-extension/retok-cpt/run_retok_init.sh first." >&2
	exit 1
fi

# ── Smoke test settings ────────────────────────────────────────────────
NPROC_PER_NODE=4
SMOKE_TEST=1
SMOKE_WARMUP_STEPS=20
SMOKE_FULL_STEPS=80
SMOKE_FULL_WARMUP_STEPS=10
MAX_SEQ_LENGTH=1024
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
ATTN_IMPLEMENTATION=sdpa
GRADIENT_CHECKPOINTING=0
TRUST_REMOTE_CODE=1
SAVE_TOTAL_LIMIT=all

export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get

# Validate EDF
if [[ -f "${EDF_PATH}" ]]; then
	image_line="$(grep -E '^[[:space:]]*image[[:space:]]*=' "${EDF_PATH}" | head -n 1 || true)"
	if [[ -n "${image_line}" ]]; then
		image_expr="${image_line#*=}"; image_expr="${image_expr# }"
		image_expr="${image_expr%\"}"; image_expr="${image_expr#\"}"
		expanded_image="${image_expr}"
		eval "expanded_image=\"${expanded_image}\""
		if [[ "${expanded_image}" == *.sqsh && ! -f "${expanded_image}" ]]; then
			echo "ERROR: CE image not found: ${expanded_image}" >&2
			echo "Build it with: sbatch scripts/build_apertus_greek_clariden_image.sh" >&2
			exit 1
		fi
	fi
fi

# Stage repo
rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"
tar -C "${REPO_ROOT}" -cz CPT scripts Agents.md Readme.md requirements.txt | tar -xz -C "${STAGE_ROOT}"
cp "${REPO_ROOT}/repo_tokenizer.py" "${STAGE_ROOT}/repo_tokenizer.py"

echo "========================================="
echo " Retok CPT SMOKE TEST — $(date)"
echo " Model: ${MODEL_PATH}"
echo " Output: ${OUTPUT_DIR}"
echo " Steps: 20 warmup + 80 full"
echo " Seq len: ${MAX_SEQ_LENGTH}, Batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}x4"
echo "========================================="

SRUN_EXPORT="ALL"
SRUN_EXPORT+=",STAGE_ROOT=${STAGE_ROOT}"
SRUN_EXPORT+=",MODEL_PATH=${MODEL_PATH}"
SRUN_EXPORT+=",OUTPUT_DIR=${OUTPUT_DIR}"
SRUN_EXPORT+=",IOPS_SCRATCH_ROOT=${IOPS_SCRATCH_ROOT}"
SRUN_EXPORT+=",CAPSTOR_SCRATCH_ROOT=${CAPSTOR_SCRATCH_ROOT}"
SRUN_EXPORT+=",RUN_NAME=${RUN_NAME}"
SRUN_EXPORT+=",NPROC_PER_NODE=${NPROC_PER_NODE}"
SRUN_EXPORT+=",TORCH_DTYPE=bfloat16"
SRUN_EXPORT+=",ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
SRUN_EXPORT+=",BF16=1"
SRUN_EXPORT+=",GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
SRUN_EXPORT+=",TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
SRUN_EXPORT+=",OVERWRITE_OUTPUT_DIR=1"
SRUN_EXPORT+=",SKIP_WARMUP=0"
SRUN_EXPORT+=",SMOKE_TEST=${SMOKE_TEST}"
SRUN_EXPORT+=",SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",SMOKE_GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",SMOKE_MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",SEED=42"
SRUN_EXPORT+=",MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",DATALOADER_NUM_WORKERS=4"
SRUN_EXPORT+=",LOGGING_STEPS=5"
SRUN_EXPORT+=",SAVE_STEPS=1000"
SRUN_EXPORT+=",SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
SRUN_EXPORT+=",LR_SCHEDULER_TYPE=cosine"
SRUN_EXPORT+=",REPORT_TO=none"
SRUN_EXPORT+=",BENCHMARK_MODE=0"
SRUN_EXPORT+=",WARMUP_MAX_STEPS=${SMOKE_WARMUP_STEPS}"
SRUN_EXPORT+=",WARMUP_LEARNING_RATE=1e-4"
SRUN_EXPORT+=",FULL_MAX_STEPS=${SMOKE_FULL_STEPS}"
SRUN_EXPORT+=",FULL_LEARNING_RATE=2e-5"
SRUN_EXPORT+=",FULL_WARMUP_STEPS=${SMOKE_FULL_WARMUP_STEPS}"
SRUN_EXPORT+=",GREEK_DATASET=epfml/FineWeb2-HQ"
SRUN_EXPORT+=",GREEK_CONFIG=ell_Grek"
SRUN_EXPORT+=",GREEK_SPLIT=train"
SRUN_EXPORT+=",GREEK_PROBABILITY=0.9"
SRUN_EXPORT+=",ENGLISH_DATASET=epfml/FineWeb-HQ"
SRUN_EXPORT+=",ENGLISH_CONFIG="
SRUN_EXPORT+=",ENGLISH_SPLIT=train"
SRUN_EXPORT+=",ENGLISH_PROBABILITY=0.1"
SRUN_EXPORT+=",PREPARED_TRAIN_DATASET_DIR="

srun --environment="${CE_ENVIRONMENT}" \
	--export="${SRUN_EXPORT}" \
	--ntasks=1 bash <<'INNER'
set -euo pipefail

if [[ -f /opt/apertus-greek-venv/bin/activate ]]; then
	. /opt/apertus-greek-venv/bin/activate
elif [[ -f /opt/gsdg-venv/bin/activate ]]; then
	. /opt/gsdg-venv/bin/activate
fi

export HF_HOME="${IOPS_SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${IOPS_SCRATCH_ROOT}/hf_datasets"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="${IOPS_SCRATCH_ROOT}/triton/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "$(dirname "${OUTPUT_DIR}")"
cd "${STAGE_ROOT}"

echo "CPT smoke test starting..." >&2
echo "Model path: ${MODEL_PATH}" >&2
echo "Output dir: ${OUTPUT_DIR}" >&2

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" CPT/cpt.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --torch-dtype bfloat16 \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --expected-world-size "${NPROC_PER_NODE}" \
  --require-distributed \
  --seed 42 \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --dataloader-num-workers 4 \
  --logging-steps 5 \
  --save-steps 1000 \
  --save-total-limit all \
  --lr-scheduler-type cosine \
  --report-to none \
  --warmup-max-steps "${WARMUP_MAX_STEPS}" \
  --warmup-learning-rate 1e-4 \
  --full-max-steps "${FULL_MAX_STEPS}" \
  --full-learning-rate 2e-5 \
  --full-warmup-steps "${FULL_WARMUP_STEPS}" \
  --smoke-test \
  --smoke-warmup-steps "${WARMUP_MAX_STEPS}" \
  --smoke-full-steps "${FULL_MAX_STEPS}" \
  --smoke-full-warmup-steps "${FULL_WARMUP_STEPS}" \
  --smoke-per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --smoke-gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --smoke-max-seq-length "${MAX_SEQ_LENGTH}" \
  --greek-dataset epfml/FineWeb2-HQ \
  --greek-config ell_Grek \
  --greek-split train \
  --greek-probability 0.9 \
  --english-dataset epfml/FineWeb-HQ \
  --english-split train \
  --english-probability 0.1 \
  --trust-remote-code \
  --overwrite-output-dir \
  --no-gradient-checkpointing

echo ""
echo "========================================="
echo " Smoke test PASSED — $(date)"
echo " Ready for full CPT: sbatch vocab-extension/retok-cpt/run_retok_cpt.sh"
echo "========================================="
INNER
