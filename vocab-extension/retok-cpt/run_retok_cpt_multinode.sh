#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=retok-cpt-mn
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/retok_cpt_mn_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/retok_cpt_mn_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=panagiotis@skarvelis.gr
#
# ── Multi-Node Retok CPT ───────────────────────────────────────────────
#
# 4 nodes × 4 GPUs = 16 GPUs total.
# Resumes automatically from the last checkpoint on timeout.
#
# Quick size guide (global batch 256, seq 2048):
#   --nodes=4  → ~2,000 steps / 12h  (25 resubmits for 50K)
#   --nodes=8  → ~4,000 steps / 12h  (12 resubmits for 50K)
#   --nodes=16 → ~8,000 steps / 12h  (6 resubmits for 50K)
#
# To change node count:
#   sbatch --nodes=8 vocab-extension/retok-cpt/run_retok_cpt_multinode.sh
#
# Prerequisites:
#   1. run_retok_init.sh must have completed
#   2. run_retok_cpt_smoke.sh should have passed

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

if [[ -z "${SLURM_JOB_NODELIST:-}" || -z "${SLURM_NNODES:-}" ]]; then
	echo "This launcher must run under a Slurm allocation." >&2
	exit 1
fi

CE_ENVIRONMENT="${CE_ENVIRONMENT:-apertus-greek-clariden}"
EDF_PATH="${HOME}/.edf/${CE_ENVIRONMENT}.toml"
IOPS_SCRATCH_ROOT="${IOPS_SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}"
CAPSTOR_SCRATCH_ROOT="${CAPSTOR_SCRATCH_ROOT:-/capstor/scratch/cscs/${USER}}"

# ── Point to the retok checkpoint ──────────────────────────────────────
MODEL_PATH="${SCRATCH}/apertus-greek-tokenizer-retok-v1"
OUTPUT_DIR="${OUTPUT_DIR:-${CAPSTOR_SCRATCH_ROOT}/apertus-greek-cpt-retok}"
RUN_NAME="apertus-greek-cpt-retok-multinode"
STAGE_ROOT="${SCRATCH}/glossapi-tokenizer_${SLURM_JOB_ID}"

if [[ ! -d "${MODEL_PATH}" ]]; then
	echo "ERROR: Retok checkpoint not found at ${MODEL_PATH}" >&2
	echo "Run sbatch vocab-extension/retok-cpt/run_retok_init.sh first." >&2
	exit 1
fi

# ── Distributed setup ──────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_WORLD_SIZE_COMPUTED=$((SLURM_NNODES * NPROC_PER_NODE))
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-${EXPECTED_WORLD_SIZE_COMPUTED}}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-${SLURM_NNODES}}"

# ── CPT settings ───────────────────────────────────────────────────────
FULL_MAX_STEPS="${FULL_MAX_STEPS:-50000}"
FULL_LEARNING_RATE="${FULL_LEARNING_RATE:-2e-5}"
FULL_WARMUP_STEPS="${FULL_WARMUP_STEPS:-1000}"
WARMUP_MAX_STEPS="${WARMUP_MAX_STEPS:-2000}"
if [[ "${SKIP_WARMUP:-0}" == "1" ]]; then
	WARMUP_MAX_STEPS=0
fi
WARMUP_LEARNING_RATE="${WARMUP_LEARNING_RATE:-1e-4}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-256}"
PER_DEVICE_TRAIN_BATCH_SIZE=1

# Derive gradient accumulation
denominator=$((PER_DEVICE_TRAIN_BATCH_SIZE * EXPECTED_WORLD_SIZE))
if (( TARGET_GLOBAL_BATCH_SIZE % denominator == 0 )); then
	GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / denominator))
else
	echo "ERROR: TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE} not divisible by per_device_batch * world_size = ${denominator}" >&2
	exit 1
fi

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
TRUST_REMOTE_CODE=1
OVERWRITE_OUTPUT_DIR=0  # MUST be 0 for resume to work
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

# Prepared dataset (optional)
PREPARED_TRAIN_DATASET_DIR="${PREPARED_TRAIN_DATASET_DIR:-}"
DEFAULT_PREPARED="${IOPS_SCRATCH_ROOT}/prepared-datasets/apertus-greek-packed-${MAX_SEQ_LENGTH}"
if [[ -z "${PREPARED_TRAIN_DATASET_DIR}" && -d "${DEFAULT_PREPARED}" ]]; then
	PREPARED_TRAIN_DATASET_DIR="${DEFAULT_PREPARED}"
fi

# ── Multi-node network settings ────────────────────────────────────────
export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-nmn0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-nmn0}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export FI_PROVIDER="${FI_PROVIDER:-cxi}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-16384}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"

# ── Validate EDF ───────────────────────────────────────────────────────
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

# ── Stage repo ─────────────────────────────────────────────────────────
rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"
tar -C "${REPO_ROOT}" -cz CPT scripts Agents.md Readme.md requirements.txt | tar -xz -C "${STAGE_ROOT}"
cp "${REPO_ROOT}/repo_tokenizer.py" "${STAGE_ROOT}/repo_tokenizer.py"

effective_global_batch=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * EXPECTED_WORLD_SIZE))

echo "========================================="
echo " Retok CPT MULTI-NODE — $(date)"
echo " Nodes:     ${SLURM_NNODES} × ${NPROC_PER_NODE} GPUs = ${EXPECTED_WORLD_SIZE} total"
echo " Model:     ${MODEL_PATH}"
echo " Output:    ${OUTPUT_DIR}"
echo " Steps:     ${WARMUP_MAX_STEPS} warmup + ${FULL_MAX_STEPS} full"
echo " Seq len:   ${MAX_SEQ_LENGTH}"
echo " Batch:     ${PER_DEVICE_TRAIN_BATCH_SIZE}/GPU × ${GRADIENT_ACCUMULATION_STEPS} accum × ${EXPECTED_WORLD_SIZE} world = ${effective_global_batch} global"
echo " LR:        ${WARMUP_LEARNING_RATE} warmup, ${FULL_LEARNING_RATE} full"
if [[ "${SKIP_WARMUP:-0}" == "1" ]]; then
	echo "             (warmup SKIPPED via SKIP_WARMUP=1)"
fi
echo " Attn:      ${ATTN_IMPLEMENTATION}, Grad ckpt: ${GRADIENT_CHECKPOINTING}"
echo " Master:    ${MASTER_ADDR}:${MASTER_PORT}"
if [[ -n "${PREPARED_TRAIN_DATASET_DIR}" ]]; then
	echo " Dataset:   prepared (${PREPARED_TRAIN_DATASET_DIR})"
else
	echo " Dataset:   streaming"
fi
echo " Resume:    OVERWRITE_OUTPUT_DIR=0 (auto-resume on timeout)"
echo "========================================="

# ── SRUN_EXPORT ────────────────────────────────────────────────────────
SRUN_EXPORT="ALL"
SRUN_EXPORT+=",STAGE_ROOT=${STAGE_ROOT}"
SRUN_EXPORT+=",MODEL_PATH=${MODEL_PATH}"
SRUN_EXPORT+=",OUTPUT_DIR=${OUTPUT_DIR}"
SRUN_EXPORT+=",IOPS_SCRATCH_ROOT=${IOPS_SCRATCH_ROOT}"
SRUN_EXPORT+=",CAPSTOR_SCRATCH_ROOT=${CAPSTOR_SCRATCH_ROOT}"
SRUN_EXPORT+=",PREPARED_TRAIN_DATASET_DIR=${PREPARED_TRAIN_DATASET_DIR}"
SRUN_EXPORT+=",RUN_NAME=${RUN_NAME}"
SRUN_EXPORT+=",MASTER_ADDR=${MASTER_ADDR}"
SRUN_EXPORT+=",MASTER_PORT=${MASTER_PORT}"
SRUN_EXPORT+=",NNODES=${NNODES}"
SRUN_EXPORT+=",NPROC_PER_NODE=${NPROC_PER_NODE}"
SRUN_EXPORT+=",EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE}"
SRUN_EXPORT+=",TORCH_DTYPE=bfloat16"
SRUN_EXPORT+=",ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
SRUN_EXPORT+=",BF16=1"
SRUN_EXPORT+=",GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
SRUN_EXPORT+=",TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
SRUN_EXPORT+=",OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR}"
SRUN_EXPORT+=",SKIP_WARMUP=${SKIP_WARMUP:-0}"
SRUN_EXPORT+=",SMOKE_TEST=0"
SRUN_EXPORT+=",SEED=42"
SRUN_EXPORT+=",MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
SRUN_EXPORT+=",GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",DATALOADER_NUM_WORKERS=2"
SRUN_EXPORT+=",LOGGING_STEPS=10"
SRUN_EXPORT+=",SAVE_STEPS=1000"
SRUN_EXPORT+=",SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
SRUN_EXPORT+=",LR_SCHEDULER_TYPE=cosine"
SRUN_EXPORT+=",REPORT_TO=none"
SRUN_EXPORT+=",BENCHMARK_MODE=0"
SRUN_EXPORT+=",WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS}"
SRUN_EXPORT+=",WARMUP_LEARNING_RATE=${WARMUP_LEARNING_RATE}"
SRUN_EXPORT+=",FULL_MAX_STEPS=${FULL_MAX_STEPS}"
SRUN_EXPORT+=",FULL_LEARNING_RATE=${FULL_LEARNING_RATE}"
SRUN_EXPORT+=",FULL_WARMUP_STEPS=${FULL_WARMUP_STEPS}"
SRUN_EXPORT+=",SMOKE_WARMUP_STEPS=20"
SRUN_EXPORT+=",SMOKE_FULL_STEPS=40"
SRUN_EXPORT+=",SMOKE_FULL_WARMUP_STEPS=5"
SRUN_EXPORT+=",SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE=1"
SRUN_EXPORT+=",SMOKE_GRADIENT_ACCUMULATION_STEPS=1"
SRUN_EXPORT+=",SMOKE_MAX_SEQ_LENGTH=1024"
SRUN_EXPORT+=",GREEK_DATASET=epfml/FineWeb2-HQ"
SRUN_EXPORT+=",GREEK_CONFIG=ell_Grek"
SRUN_EXPORT+=",GREEK_SPLIT=train"
SRUN_EXPORT+=",GREEK_PROBABILITY=${GREEK_PROBABILITY:-0.9}"
SRUN_EXPORT+=",ENGLISH_DATASET=epfml/FineWeb-HQ"
SRUN_EXPORT+=",ENGLISH_CONFIG="
SRUN_EXPORT+=",ENGLISH_SPLIT=train"
SRUN_EXPORT+=",ENGLISH_PROBABILITY=${ENGLISH_PROBABILITY:-0.1}"
SRUN_EXPORT+=",NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
SRUN_EXPORT+=",GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
SRUN_EXPORT+=",NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
SRUN_EXPORT+=",FI_PROVIDER=${FI_PROVIDER}"
SRUN_EXPORT+=",FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE}"
SRUN_EXPORT+=",FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE}"
SRUN_EXPORT+=",FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER}"
SRUN_EXPORT+=",FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE}"
SRUN_EXPORT+=",FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR}"

# ── Launch ─────────────────────────────────────────────────────────────
srun --environment="${CE_ENVIRONMENT}" \
	--export="${SRUN_EXPORT}" \
	--ntasks-per-node=1 \
	--kill-on-bad-exit=1 bash <<'INNER'
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
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC}"
export FI_PROVIDER="${FI_PROVIDER}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "$(dirname "${OUTPUT_DIR}")"
cd "${STAGE_ROOT}"

echo "CPT multi-node training starting..." >&2
echo "Model path: ${MODEL_PATH}" >&2
echo "Output dir: ${OUTPUT_DIR}" >&2
echo "World size: ${EXPECTED_WORLD_SIZE}, Master: ${MASTER_ADDR}:${MASTER_PORT}" >&2

cpt_args=(
	CPT/cpt.py
	--model-path "${MODEL_PATH}"
	--output-dir "${OUTPUT_DIR}"
	--run-name "${RUN_NAME}"
	--torch-dtype bfloat16
	--attn-implementation "${ATTN_IMPLEMENTATION}"
	--expected-world-size "${EXPECTED_WORLD_SIZE}"
	--require-distributed
	--seed 42
	--max-seq-length "${MAX_SEQ_LENGTH}"
	--per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
	--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
	--dataloader-num-workers 2
	--logging-steps 10
	--save-steps 1000
	--save-total-limit "${SAVE_TOTAL_LIMIT}"
	--lr-scheduler-type cosine
	--report-to none
	--warmup-max-steps "${WARMUP_MAX_STEPS}"
	--warmup-learning-rate "${WARMUP_LEARNING_RATE}"
	--full-max-steps "${FULL_MAX_STEPS}"
	--full-learning-rate "${FULL_LEARNING_RATE}"
	--full-warmup-steps "${FULL_WARMUP_STEPS}"
	--greek-dataset epfml/FineWeb2-HQ
	--greek-config ell_Grek
	--greek-split train
	--greek-probability "${GREEK_PROBABILITY}"
	--english-dataset epfml/FineWeb-HQ
	--english-split train
	--english-probability "${ENGLISH_PROBABILITY}"
	--trust-remote-code
)

if [[ -n "${PREPARED_TRAIN_DATASET_DIR}" ]]; then
	cpt_args+=(--prepared-train-dataset-dir "${PREPARED_TRAIN_DATASET_DIR}")
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
	cpt_args+=(--no-gradient-checkpointing)
fi
if [[ "${SKIP_WARMUP:-0}" == "1" ]]; then
	cpt_args+=(--skip-warmup)
fi

python -m torch.distributed.run \
	--nnodes="${NNODES}" \
	--nproc_per_node="${NPROC_PER_NODE}" \
	--rdzv_id="${SLURM_JOB_ID}" \
	--rdzv_backend=c10d \
	--rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
	"${cpt_args[@]}"

echo ""
echo "========================================="
echo " CPT complete — $(date)"
echo " Checkpoint: ${OUTPUT_DIR}/final"
echo " Evaluate:"
echo "   ./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \\"
echo "     --base-model swiss-ai/Apertus-8B-Instruct-2509 \\"
echo "     --trained-model ${OUTPUT_DIR}/final \\"
echo "     --output-json artifacts/reports/greek_mmlu_retok_cpt_eval.json"
echo "========================================="
INNER
