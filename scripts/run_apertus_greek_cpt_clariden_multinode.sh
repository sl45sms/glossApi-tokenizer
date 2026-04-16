#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=apertus-greek-cpt-mn
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="/users/p-skarvelis/glossApi-Tokenizer"
if [[ ! -d "${REPO_ROOT}" ]]; then
	echo "Repository root not found at ${REPO_ROOT}" >&2
	exit 1
fi

ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
	set -a
	# shellcheck disable=SC1090
	source "${ENV_FILE}"
	set +a
	echo "Loaded optional environment from ${ENV_FILE}" >&2
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
MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-init/}"
OUTPUT_DIR="${OUTPUT_DIR:-${CAPSTOR_SCRATCH_ROOT}/apertus-greek-cpt}"
PREPARED_TRAIN_DATASET_DIR="${PREPARED_TRAIN_DATASET_DIR:-}"
RUN_NAME="${RUN_NAME:-apertus-greek-cpt-multinode}"
STAGE_ROOT="${STAGE_ROOT:-${SCRATCH}/glossapi-tokenizer_${SLURM_JOB_ID}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_WORLD_SIZE_COMPUTED=$((SLURM_NNODES * NPROC_PER_NODE))
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-${EXPECTED_WORLD_SIZE_COMPUTED}}"
if (( EXPECTED_WORLD_SIZE != EXPECTED_WORLD_SIZE_COMPUTED )); then
	echo "EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE} does not match SLURM_NNODES * NPROC_PER_NODE = ${EXPECTED_WORLD_SIZE_COMPUTED}." >&2
	exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-${SLURM_NNODES}}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
BF16="${BF16:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SMOKE_GRADIENT_ACCUMULATION_STEPS="${SMOKE_GRADIENT_ACCUMULATION_STEPS:-1}"
SMOKE_MAX_SEQ_LENGTH="${SMOKE_MAX_SEQ_LENGTH:-1024}"

SEED="${SEED:-42}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-1000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
	denominator=$((PER_DEVICE_TRAIN_BATCH_SIZE * EXPECTED_WORLD_SIZE))
	if (( TARGET_GLOBAL_BATCH_SIZE % denominator != 0 )); then
		echo "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE} is not divisible by PER_DEVICE_TRAIN_BATCH_SIZE * EXPECTED_WORLD_SIZE (${denominator})." >&2
		echo "Set GRADIENT_ACCUMULATION_STEPS explicitly for this launch shape." >&2
		exit 1
	fi
	GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / denominator))
fi
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
REPORT_TO="${REPORT_TO:-none}"
BENCHMARK_MODE="${BENCHMARK_MODE:-0}"

if [[ "${BENCHMARK_MODE}" == "1" && "${SMOKE_TEST}" == "1" ]]; then
	echo "BENCHMARK_MODE=1 cannot be combined with SMOKE_TEST=1. Unset SMOKE_TEST or export SMOKE_TEST=0 before launching the benchmark." >&2
	exit 1
fi

WARMUP_MAX_STEPS="${WARMUP_MAX_STEPS:-2000}"
WARMUP_LEARNING_RATE="${WARMUP_LEARNING_RATE:-1e-4}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:-50000}"
FULL_LEARNING_RATE="${FULL_LEARNING_RATE:-2e-5}"
FULL_WARMUP_STEPS="${FULL_WARMUP_STEPS:-1000}"
SMOKE_WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-20}"
SMOKE_FULL_STEPS="${SMOKE_FULL_STEPS:-40}"
SMOKE_FULL_WARMUP_STEPS="${SMOKE_FULL_WARMUP_STEPS:-5}"

GREEK_DATASET="${GREEK_DATASET:-epfml/FineWeb2-HQ}"
GREEK_CONFIG="${GREEK_CONFIG:-ell_Grek}"
GREEK_SPLIT="${GREEK_SPLIT:-train}"
GREEK_PROBABILITY="${GREEK_PROBABILITY:-0.9}"
ENGLISH_DATASET="${ENGLISH_DATASET:-epfml/FineWeb-HQ}"
ENGLISH_CONFIG="${ENGLISH_CONFIG:-}"
ENGLISH_SPLIT="${ENGLISH_SPLIT:-train}"
ENGLISH_PROBABILITY="${ENGLISH_PROBABILITY:-0.1}"

DEFAULT_PREPARED_TRAIN_DATASET_DIR="${IOPS_SCRATCH_ROOT}/prepared-datasets/apertus-greek-packed-${MAX_SEQ_LENGTH}"
if [[ -z "${PREPARED_TRAIN_DATASET_DIR}" && -d "${DEFAULT_PREPARED_TRAIN_DATASET_DIR}" ]]; then
	PREPARED_TRAIN_DATASET_DIR="${DEFAULT_PREPARED_TRAIN_DATASET_DIR}"
fi

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

if [[ -f "${EDF_PATH}" ]]; then
	image_line="$(grep -E '^[[:space:]]*image[[:space:]]*=' "${EDF_PATH}" | head -n 1 || true)"
	if [[ -n "${image_line}" ]]; then
		image_expr="${image_line#*=}"
		image_expr="${image_expr# }"
		image_expr="${image_expr%\"}"
		image_expr="${image_expr#\"}"
		expanded_image="${image_expr}"
		eval "expanded_image=\"${expanded_image}\""
		if [[ "${expanded_image}" == *.sqsh && ! -f "${expanded_image}" ]]; then
			echo "CE image referenced by ${EDF_PATH} does not exist: ${expanded_image}" >&2
			echo "Build it first with: sbatch scripts/build_apertus_greek_clariden_image.sh" >&2
			exit 1
		fi
	fi
fi

rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"
tar -C "${REPO_ROOT}" -cz CPT scripts Agents.md Readme.md requirements.txt | tar -xz -C "${STAGE_ROOT}"

effective_global_batch=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * EXPECTED_WORLD_SIZE))

echo "Using CE environment: ${CE_ENVIRONMENT}" >&2
echo "Using model path: ${MODEL_PATH}" >&2
echo "Using output dir: ${OUTPUT_DIR}" >&2
echo "Using iops scratch root: ${IOPS_SCRATCH_ROOT}" >&2
echo "Using capstor scratch root: ${CAPSTOR_SCRATCH_ROOT}" >&2
if [[ -n "${PREPARED_TRAIN_DATASET_DIR}" ]]; then
	echo "Using prepared train dataset dir: ${PREPARED_TRAIN_DATASET_DIR}" >&2
fi
if [[ "${BENCHMARK_MODE}" == "1" ]]; then
	echo "Benchmark mode enabled: checkpoints and final model export will be skipped." >&2
fi
echo "Staged workspace into ${STAGE_ROOT}" >&2
echo "Using MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}" >&2
echo "Using ${SLURM_NNODES} node(s), ${NPROC_PER_NODE} process(es) per node, world_size=${EXPECTED_WORLD_SIZE}" >&2
echo "Using max_seq_length=${MAX_SEQ_LENGTH}, per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}, gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}, effective_global_batch_size=${effective_global_batch}" >&2
echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} FI_PROVIDER=${FI_PROVIDER}" >&2

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
SRUN_EXPORT+=",TORCH_DTYPE=${TORCH_DTYPE}"
SRUN_EXPORT+=",ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
SRUN_EXPORT+=",BF16=${BF16}"
SRUN_EXPORT+=",GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
SRUN_EXPORT+=",TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
SRUN_EXPORT+=",OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR}"
SRUN_EXPORT+=",SKIP_WARMUP=${SKIP_WARMUP}"
SRUN_EXPORT+=",SMOKE_TEST=${SMOKE_TEST}"
SRUN_EXPORT+=",SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE=${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",SMOKE_GRADIENT_ACCUMULATION_STEPS=${SMOKE_GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",SMOKE_MAX_SEQ_LENGTH=${SMOKE_MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",SEED=${SEED}"
SRUN_EXPORT+=",MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",TOKENIZE_BATCH_SIZE=${TOKENIZE_BATCH_SIZE}"
SRUN_EXPORT+=",PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
SRUN_EXPORT+=",GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
SRUN_EXPORT+=",LOGGING_STEPS=${LOGGING_STEPS}"
SRUN_EXPORT+=",SAVE_STEPS=${SAVE_STEPS}"
SRUN_EXPORT+=",SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
SRUN_EXPORT+=",LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}"
SRUN_EXPORT+=",REPORT_TO=${REPORT_TO}"
SRUN_EXPORT+=",BENCHMARK_MODE=${BENCHMARK_MODE}"
SRUN_EXPORT+=",WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS}"
SRUN_EXPORT+=",WARMUP_LEARNING_RATE=${WARMUP_LEARNING_RATE}"
SRUN_EXPORT+=",FULL_MAX_STEPS=${FULL_MAX_STEPS}"
SRUN_EXPORT+=",FULL_LEARNING_RATE=${FULL_LEARNING_RATE}"
SRUN_EXPORT+=",FULL_WARMUP_STEPS=${FULL_WARMUP_STEPS}"
SRUN_EXPORT+=",SMOKE_WARMUP_STEPS=${SMOKE_WARMUP_STEPS}"
SRUN_EXPORT+=",SMOKE_FULL_STEPS=${SMOKE_FULL_STEPS}"
SRUN_EXPORT+=",SMOKE_FULL_WARMUP_STEPS=${SMOKE_FULL_WARMUP_STEPS}"
SRUN_EXPORT+=",GREEK_DATASET=${GREEK_DATASET}"
SRUN_EXPORT+=",GREEK_CONFIG=${GREEK_CONFIG}"
SRUN_EXPORT+=",GREEK_SPLIT=${GREEK_SPLIT}"
SRUN_EXPORT+=",GREEK_PROBABILITY=${GREEK_PROBABILITY}"
SRUN_EXPORT+=",ENGLISH_DATASET=${ENGLISH_DATASET}"
SRUN_EXPORT+=",ENGLISH_CONFIG=${ENGLISH_CONFIG}"
SRUN_EXPORT+=",ENGLISH_SPLIT=${ENGLISH_SPLIT}"
SRUN_EXPORT+=",ENGLISH_PROBABILITY=${ENGLISH_PROBABILITY}"
SRUN_EXPORT+=",NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
SRUN_EXPORT+=",GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
SRUN_EXPORT+=",NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
SRUN_EXPORT+=",FI_PROVIDER=${FI_PROVIDER}"
SRUN_EXPORT+=",FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE}"
SRUN_EXPORT+=",FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE}"
SRUN_EXPORT+=",FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER}"
SRUN_EXPORT+=",FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE}"
SRUN_EXPORT+=",FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR}"

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

export HF_HOME="${HF_HOME:-${IOPS_SCRATCH_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${IOPS_SCRATCH_ROOT}/hf_datasets}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${IOPS_SCRATCH_ROOT}/triton/cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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

cpt_args=(
	CPT/cpt.py
	--model-path "${MODEL_PATH}"
	--output-dir "${OUTPUT_DIR}"
	--run-name "${RUN_NAME}"
	--torch-dtype "${TORCH_DTYPE}"
	--attn-implementation "${ATTN_IMPLEMENTATION}"
	--expected-world-size "${EXPECTED_WORLD_SIZE}"
	--require-distributed
	--seed "${SEED}"
	--max-seq-length "${MAX_SEQ_LENGTH}"
	--tokenize-batch-size "${TOKENIZE_BATCH_SIZE}"
	--per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
	--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
	--dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
	--logging-steps "${LOGGING_STEPS}"
	--save-steps "${SAVE_STEPS}"
	--save-total-limit "${SAVE_TOTAL_LIMIT}"
	--lr-scheduler-type "${LR_SCHEDULER_TYPE}"
	--report-to "${REPORT_TO}"
	--warmup-max-steps "${WARMUP_MAX_STEPS}"
	--warmup-learning-rate "${WARMUP_LEARNING_RATE}"
	--full-max-steps "${FULL_MAX_STEPS}"
	--full-learning-rate "${FULL_LEARNING_RATE}"
	--full-warmup-steps "${FULL_WARMUP_STEPS}"
	--smoke-warmup-steps "${SMOKE_WARMUP_STEPS}"
	--smoke-full-steps "${SMOKE_FULL_STEPS}"
	--smoke-full-warmup-steps "${SMOKE_FULL_WARMUP_STEPS}"
	--smoke-per-device-train-batch-size "${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE}"
	--smoke-gradient-accumulation-steps "${SMOKE_GRADIENT_ACCUMULATION_STEPS}"
	--smoke-max-seq-length "${SMOKE_MAX_SEQ_LENGTH}"
	--greek-dataset "${GREEK_DATASET}"
	--greek-config "${GREEK_CONFIG}"
	--greek-split "${GREEK_SPLIT}"
	--greek-probability "${GREEK_PROBABILITY}"
	--english-dataset "${ENGLISH_DATASET}"
	--english-split "${ENGLISH_SPLIT}"
	--english-probability "${ENGLISH_PROBABILITY}"
)

if [[ -n "${PREPARED_TRAIN_DATASET_DIR}" ]]; then
	cpt_args+=(--prepared-train-dataset-dir "${PREPARED_TRAIN_DATASET_DIR}")
fi
if [[ "${BENCHMARK_MODE}" == "1" ]]; then
	cpt_args+=(--benchmark-mode)
fi

if [[ -n "${ENGLISH_CONFIG}" ]]; then
	cpt_args+=(--english-config "${ENGLISH_CONFIG}")
fi
if [[ "${BF16}" == "0" ]]; then
	cpt_args+=(--no-bf16)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
	cpt_args+=(--no-gradient-checkpointing)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
	cpt_args+=(--trust-remote-code)
fi
if [[ "${OVERWRITE_OUTPUT_DIR}" == "1" ]]; then
	cpt_args+=(--overwrite-output-dir)
fi
if [[ "${SKIP_WARMUP}" == "1" ]]; then
	cpt_args+=(--skip-warmup)
fi
if [[ "${SMOKE_TEST}" == "1" ]]; then
	cpt_args+=(--smoke-test)
fi

python -m torch.distributed.run \
	--nnodes="${NNODES}" \
	--nproc_per_node="${NPROC_PER_NODE}" \
	--node_rank="${SLURM_NODEID}" \
	--master_addr="${MASTER_ADDR}" \
	--master_port="${MASTER_PORT}" \
	"${cpt_args[@]}"
INNER