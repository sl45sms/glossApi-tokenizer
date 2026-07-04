#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=retok-sft
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/retok_sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/retok_sft_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=%u@cscs.ch
#
# ── Stage 3: SFT on Retok CPT Checkpoint ───────────────────────────────
#
# Restores instruction-following ability lost during CPT.
# Uses swiss-ai/apertus-sft-mixture (same dataset that made Apertus instruct).
#
# Prerequisite: CPT must have completed and saved the final checkpoint.

set -euo pipefail

REPO_ROOT="/users/p-skarvelis/glossApi-Tokenizer"
if [[ ! -d "${REPO_ROOT}" ]]; then
	echo "Repository root not found at ${REPO_ROOT}" >&2
	exit 1
fi

ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
	set -a
	source "${ENV_FILE}"
	set +a
fi

export HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH must be set before submitting this job." >&2
	exit 1
fi

CE_ENVIRONMENT="${CE_ENVIRONMENT:-apertus-greek-clariden}"
IOPS_SCRATCH_ROOT="${IOPS_SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}"
CAPSTOR_SCRATCH_ROOT="${CAPSTOR_SCRATCH_ROOT:-/capstor/scratch/cscs/${USER}}"

# ── Point to the CPT final checkpoint ──────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${CAPSTOR_SCRATCH_ROOT}/apertus-greek-cpt-retok/final}"
OUTPUT_DIR="${OUTPUT_DIR:-${CAPSTOR_SCRATCH_ROOT}/apertus-greek-sft-retok}"
RUN_NAME="${RUN_NAME:-apertus-greek-sft-retok}"
STAGE_ROOT="${SCRATCH}/glossapi-tokenizer-sft_${SLURM_JOB_ID}"

if [[ ! -d "${MODEL_PATH}" ]]; then
	echo "ERROR: CPT final checkpoint not found at ${MODEL_PATH}" >&2
	echo "Run the retok CPT pipeline first." >&2
	exit 1
fi

if [[ ! -f "${MODEL_PATH}/tokenizer_config.json" ]]; then
	echo "ERROR: tokenizer_config.json missing from ${MODEL_PATH}" >&2
	echo "Copy tokenizer files: cp .../apertus-greek-tokenizer-retok-v1/tokenizer* ${MODEL_PATH}/" >&2
	exit 1
fi

# ── SFT settings ───────────────────────────────────────────────────────
DATASET_NAME="${DATASET_NAME:-swiss-ai/apertus-sft-mixture}"
DATASET_SPLIT="train"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"

NPROC_PER_NODE=4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
GRADIENT_CHECKPOINTING=0
DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-ddp}"
TRUST_REMOTE_CODE=1
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

# ── Smoke test ─────────────────────────────────────────────────────────
SMOKE_TEST="${SMOKE_TEST:-0}"

export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get

# Stage repo
rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"
tar -C "${REPO_ROOT}" -cz SFT | tar -xz -C "${STAGE_ROOT}"
cp "${REPO_ROOT}/repo_tokenizer.py" "${STAGE_ROOT}/repo_tokenizer.py"

echo "========================================="
echo " Retok SFT — $(date)"
echo " Model:     ${MODEL_PATH}"
echo " Output:    ${OUTPUT_DIR}"
echo " Dataset:   ${DATASET_NAME}"
echo " Seq len:   ${MAX_SEQ_LENGTH}"
echo " Epochs:    ${NUM_TRAIN_EPOCHS}, LR: ${LEARNING_RATE}"
echo " Batch:     ${PER_DEVICE_TRAIN_BATCH_SIZE}/GPU × ${GRADIENT_ACCUMULATION_STEPS} accum × 4 GPUs"
echo " Attn:      ${ATTN_IMPLEMENTATION}, Strategy: ${DISTRIBUTED_STRATEGY}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
	echo " MODE:      SMOKE TEST (20 steps)"
else
	echo " MODE:      FULL SFT"
fi
echo "========================================="

SRUN_EXPORT="ALL"
SRUN_EXPORT+=",STAGE_ROOT=${STAGE_ROOT}"
SRUN_EXPORT+=",MODEL_PATH=${MODEL_PATH}"
SRUN_EXPORT+=",OUTPUT_DIR=${OUTPUT_DIR}"
SRUN_EXPORT+=",DATASET_NAME=${DATASET_NAME}"
SRUN_EXPORT+=",DATASET_CONFIG="
SRUN_EXPORT+=",DATASET_SPLIT=${DATASET_SPLIT}"
SRUN_EXPORT+=",PREPARED_DATASET_DIR="
SRUN_EXPORT+=",VALIDATION_SAMPLES=0"
SRUN_EXPORT+=",RUN_NAME=${RUN_NAME}"
SRUN_EXPORT+=",NPROC_PER_NODE=${NPROC_PER_NODE}"
SRUN_EXPORT+=",EXPECTED_WORLD_SIZE=${NPROC_PER_NODE}"
SRUN_EXPORT+=",TORCH_DTYPE=bfloat16"
SRUN_EXPORT+=",ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
SRUN_EXPORT+=",BF16=1"
SRUN_EXPORT+=",GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
SRUN_EXPORT+=",DISTRIBUTED_STRATEGY=${DISTRIBUTED_STRATEGY}"
SRUN_EXPORT+=",FSDP_MIN_NUM_PARAMS=100000000"
SRUN_EXPORT+=",FSDP_BACKWARD_PREFETCH=backward_pre"
SRUN_EXPORT+=",TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
SRUN_EXPORT+=",OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR}"
SRUN_EXPORT+=",SEED=42"
SRUN_EXPORT+=",PREPROCESSING_BATCH_SIZE=256"
SRUN_EXPORT+=",DATASET_NUM_PROC=1"
SRUN_EXPORT+=",MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",LEARNING_RATE=${LEARNING_RATE}"
SRUN_EXPORT+=",WEIGHT_DECAY=0.0"
SRUN_EXPORT+=",NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
SRUN_EXPORT+=",MAX_STEPS=-1"
SRUN_EXPORT+=",WARMUP_RATIO=0.03"
SRUN_EXPORT+=",LR_SCHEDULER_TYPE=cosine"
SRUN_EXPORT+=",PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",PER_DEVICE_EVAL_BATCH_SIZE=1"
SRUN_EXPORT+=",GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",DATALOADER_NUM_WORKERS=4"
SRUN_EXPORT+=",LOGGING_STEPS=10"
SRUN_EXPORT+=",SAVE_STEPS=500"
SRUN_EXPORT+=",SAVE_TOTAL_LIMIT=3"
SRUN_EXPORT+=",REPORT_TO=none"
SRUN_EXPORT+=",SMOKE_TEST=${SMOKE_TEST}"
SRUN_EXPORT+=",SMOKE_MAX_STEPS=20"
SRUN_EXPORT+=",SMOKE_TRAIN_SAMPLES=128"
SRUN_EXPORT+=",SMOKE_VALIDATION_SAMPLES=16"
SRUN_EXPORT+=",RESUME_FROM_CHECKPOINT="
SRUN_EXPORT+=",IOPS_SCRATCH_ROOT=${IOPS_SCRATCH_ROOT}"
SRUN_EXPORT+=",NCCL_SOCKET_IFNAME=nmn0"
SRUN_EXPORT+=",GLOO_SOCKET_IFNAME=nmn0"
SRUN_EXPORT+=",NCCL_CROSS_NIC=1"
SRUN_EXPORT+=",FI_PROVIDER=cxi"

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
export TRITON_CACHE_DIR="${IOPS_SCRATCH_ROOT}/triton/cache"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=nmn0
export GLOO_SOCKET_IFNAME=nmn0
export NCCL_CROSS_NIC=1
export FI_PROVIDER=cxi
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "$(dirname "${OUTPUT_DIR}")"
cd "${STAGE_ROOT}"

echo "SFT training starting..." >&2
echo "Model path: ${MODEL_PATH}" >&2
echo "Output dir: ${OUTPUT_DIR}" >&2

sft_args=(
	SFT/sft.py
	--model-path "${MODEL_PATH}"
	--output-dir "${OUTPUT_DIR}"
	--run-name "${RUN_NAME}"
	--dataset-name "${DATASET_NAME}"
	--dataset-split "${DATASET_SPLIT}"
	--validation-samples 0
	--preprocessing-batch-size 256
	--dataset-num-proc 1
	--max-seq-length "${MAX_SEQ_LENGTH}"
	--torch-dtype bfloat16
	--attn-implementation "${ATTN_IMPLEMENTATION}"
	--distributed-strategy "${DISTRIBUTED_STRATEGY}"
	--expected-world-size "${NPROC_PER_NODE}"
	--require-distributed
	--seed 42
	--learning-rate "${LEARNING_RATE}"
	--weight-decay 0.0
	--num-train-epochs "${NUM_TRAIN_EPOCHS}"
	--max-steps -1
	--warmup-ratio 0.03
	--lr-scheduler-type cosine
	--per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
	--per-device-eval-batch-size 1
	--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
	--dataloader-num-workers 4
	--logging-steps 10
	--save-steps 500
	--save-total-limit 3
	--report-to none
	--smoke-max-steps 20
	--smoke-train-samples 128
	--smoke-validation-samples 16
	--trust-remote-code
)

if [[ "${SMOKE_TEST}" == "1" ]]; then
	sft_args+=(--smoke-test)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
	sft_args+=(--no-gradient-checkpointing)
fi
if [[ "${OVERWRITE_OUTPUT_DIR}" == "1" ]]; then
	sft_args+=(--overwrite-output-dir)
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${sft_args[@]}"

echo ""
echo "========================================="
echo " SFT complete — $(date)"
echo " Model:    ${OUTPUT_DIR}"
echo " Test it:"
echo "   ./run_model_ui.sh --base-device cuda:0 --device cuda:1 \\"
echo "     --model-path ${OUTPUT_DIR}"
echo " Evaluate:"
echo "   ./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \\"
echo "     --base-model swiss-ai/Apertus-8B-Instruct-2509 \\"
echo "     --trained-model ${OUTPUT_DIR} \\"
echo "     --output-json artifacts/reports/greek_mmlu_retok_sft_eval.json"
echo "========================================="
INNER
