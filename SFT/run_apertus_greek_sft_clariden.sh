#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=apertus-greek-sft
#SBATCH --partition=normal
#SBATCH --nodes=1
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

EDF_PATH="${HOME}/.edf/${CE_ENVIRONMENT:-apertus-greek-clariden}.toml"

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH must be set before submitting this job." >&2
	exit 1
fi

CE_ENVIRONMENT="${CE_ENVIRONMENT:-apertus-greek-clariden}"
IOPS_SCRATCH_ROOT="${IOPS_SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}"
CAPSTOR_SCRATCH_ROOT="${CAPSTOR_SCRATCH_ROOT:-/capstor/scratch/cscs/${USER}}"
MODEL_PATH="${MODEL_PATH:-/capstor/scratch/cscs/p-skarvelis/apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-400steps/final}"
OUTPUT_DIR="${OUTPUT_DIR:-${CAPSTOR_SCRATCH_ROOT}/apertus-greek-sft}"
DATASET_NAME="${DATASET_NAME:-swiss-ai/apertus-sft-mixture}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
PREPARED_DATASET_DIR="${PREPARED_DATASET_DIR:-}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
RUN_NAME="${RUN_NAME:-apertus-greek-sft}"
STAGE_ROOT="${STAGE_ROOT:-${SCRATCH}/glossapi-tokenizer-sft_${SLURM_JOB_ID}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-${NPROC_PER_NODE}}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
BF16="${BF16:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
REQUESTED_DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-auto}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-100000000}"
FSDP_BACKWARD_PREFETCH="${FSDP_BACKWARD_PREFETCH:-backward_pre}"
FSDP_LIMIT_ALL_GATHERS="${FSDP_LIMIT_ALL_GATHERS:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
SEED="${SEED:-42}"

PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-256}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
TRUNCATION_SIDE="${TRUNCATION_SIDE:-left}"

DEFAULT_PREPARED_DATASET_DIR="${IOPS_SCRATCH_ROOT}/prepared-datasets/apertus-greek-sft-${MAX_SEQ_LENGTH}-${TRUNCATION_SIDE}-val${VALIDATION_SAMPLES}"
if [[ -z "${PREPARED_DATASET_DIR}" && -d "${DEFAULT_PREPARED_DATASET_DIR}" ]]; then
	PREPARED_DATASET_DIR="${DEFAULT_PREPARED_DATASET_DIR}"
fi

if [[ "${REQUESTED_DISTRIBUTED_STRATEGY}" == "auto" ]]; then
	if [[ "${NPROC_PER_NODE}" -gt 1 && "${MAX_SEQ_LENGTH}" -gt 1024 ]]; then
		DISTRIBUTED_STRATEGY="fsdp_full_shard"
	else
		DISTRIBUTED_STRATEGY="ddp"
	fi
else
	DISTRIBUTED_STRATEGY="${REQUESTED_DISTRIBUTED_STRATEGY}"
fi

if [[ "${DISTRIBUTED_STRATEGY}" != "ddp" && "${DISTRIBUTED_STRATEGY}" != "fsdp_full_shard" ]]; then
	echo "Unsupported DISTRIBUTED_STRATEGY=${DISTRIBUTED_STRATEGY}. Use ddp, fsdp_full_shard, or auto." >&2
	exit 1
fi
if [[ "${DISTRIBUTED_STRATEGY}" == "fsdp_full_shard" && "${NPROC_PER_NODE}" -le 1 ]]; then
	echo "DISTRIBUTED_STRATEGY=fsdp_full_shard requires NPROC_PER_NODE greater than 1." >&2
	exit 1
fi

LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
MAX_STEPS="${MAX_STEPS:--1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REPORT_TO="${REPORT_TO:-none}"

SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-20}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-128}"
SMOKE_VALIDATION_SAMPLES="${SMOKE_VALIDATION_SAMPLES:-16}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-nmn0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-nmn0}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export FI_PROVIDER="${FI_PROVIDER:-cxi}"

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
tar -C "${REPO_ROOT}" -cz SFT | tar -xz -C "${STAGE_ROOT}"
cp "${REPO_ROOT}/repo_tokenizer.py" "${STAGE_ROOT}/repo_tokenizer.py"

echo "Using CE environment: ${CE_ENVIRONMENT}" >&2
echo "Using model path: ${MODEL_PATH}" >&2
echo "Using output dir: ${OUTPUT_DIR}" >&2
echo "Using dataset: ${DATASET_NAME}" >&2
echo "Using dataset split: ${DATASET_SPLIT}" >&2
if [[ -n "${PREPARED_DATASET_DIR}" ]]; then
	echo "Using prepared dataset dir: ${PREPARED_DATASET_DIR}" >&2
fi
echo "Using distributed strategy: ${DISTRIBUTED_STRATEGY}" >&2
echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_CROSS_NIC=${NCCL_CROSS_NIC} FI_PROVIDER=${FI_PROVIDER}" >&2
echo "Staged workspace into ${STAGE_ROOT}" >&2

SRUN_EXPORT="ALL"
SRUN_EXPORT+=",STAGE_ROOT=${STAGE_ROOT}"
SRUN_EXPORT+=",MODEL_PATH=${MODEL_PATH}"
SRUN_EXPORT+=",OUTPUT_DIR=${OUTPUT_DIR}"
SRUN_EXPORT+=",DATASET_NAME=${DATASET_NAME}"
SRUN_EXPORT+=",DATASET_CONFIG=${DATASET_CONFIG}"
SRUN_EXPORT+=",DATASET_SPLIT=${DATASET_SPLIT}"
SRUN_EXPORT+=",PREPARED_DATASET_DIR=${PREPARED_DATASET_DIR}"
SRUN_EXPORT+=",VALIDATION_SAMPLES=${VALIDATION_SAMPLES}"
SRUN_EXPORT+=",MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}"
SRUN_EXPORT+=",MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}"
SRUN_EXPORT+=",RUN_NAME=${RUN_NAME}"
SRUN_EXPORT+=",NPROC_PER_NODE=${NPROC_PER_NODE}"
SRUN_EXPORT+=",EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE}"
SRUN_EXPORT+=",TORCH_DTYPE=${TORCH_DTYPE}"
SRUN_EXPORT+=",ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
SRUN_EXPORT+=",BF16=${BF16}"
SRUN_EXPORT+=",GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
SRUN_EXPORT+=",DISTRIBUTED_STRATEGY=${DISTRIBUTED_STRATEGY}"
SRUN_EXPORT+=",FSDP_MIN_NUM_PARAMS=${FSDP_MIN_NUM_PARAMS}"
SRUN_EXPORT+=",FSDP_BACKWARD_PREFETCH=${FSDP_BACKWARD_PREFETCH}"
SRUN_EXPORT+=",FSDP_LIMIT_ALL_GATHERS=${FSDP_LIMIT_ALL_GATHERS}"
SRUN_EXPORT+=",TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
SRUN_EXPORT+=",OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR}"
SRUN_EXPORT+=",SEED=${SEED}"
SRUN_EXPORT+=",PREPROCESSING_BATCH_SIZE=${PREPROCESSING_BATCH_SIZE}"
SRUN_EXPORT+=",DATASET_NUM_PROC=${DATASET_NUM_PROC}"
SRUN_EXPORT+=",MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",TRUNCATION_SIDE=${TRUNCATION_SIDE}"
SRUN_EXPORT+=",LEARNING_RATE=${LEARNING_RATE}"
SRUN_EXPORT+=",WEIGHT_DECAY=${WEIGHT_DECAY}"
SRUN_EXPORT+=",NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
SRUN_EXPORT+=",MAX_STEPS=${MAX_STEPS}"
SRUN_EXPORT+=",WARMUP_RATIO=${WARMUP_RATIO}"
SRUN_EXPORT+=",LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}"
SRUN_EXPORT+=",PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
SRUN_EXPORT+=",PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
SRUN_EXPORT+=",GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
SRUN_EXPORT+=",DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
SRUN_EXPORT+=",LOGGING_STEPS=${LOGGING_STEPS}"
SRUN_EXPORT+=",EVAL_STEPS=${EVAL_STEPS}"
SRUN_EXPORT+=",SAVE_STEPS=${SAVE_STEPS}"
SRUN_EXPORT+=",SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
SRUN_EXPORT+=",REPORT_TO=${REPORT_TO}"
SRUN_EXPORT+=",SMOKE_TEST=${SMOKE_TEST}"
SRUN_EXPORT+=",SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS}"
SRUN_EXPORT+=",SMOKE_TRAIN_SAMPLES=${SMOKE_TRAIN_SAMPLES}"
SRUN_EXPORT+=",SMOKE_VALIDATION_SAMPLES=${SMOKE_VALIDATION_SAMPLES}"
SRUN_EXPORT+=",RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
SRUN_EXPORT+=",IOPS_SCRATCH_ROOT=${IOPS_SCRATCH_ROOT}"
SRUN_EXPORT+=",NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
SRUN_EXPORT+=",GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
SRUN_EXPORT+=",NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
SRUN_EXPORT+=",FI_PROVIDER=${FI_PROVIDER}"

srun --environment="${CE_ENVIRONMENT}" \
	--export="${SRUN_EXPORT}" \
	--ntasks=1 bash <<'INNER'
set -euo pipefail

if [[ -f /opt/apertus-greek-venv/bin/activate ]]; then
	. /opt/apertus-greek-venv/bin/activate
elif [[ -f /opt/gsdg-venv/bin/activate ]]; then
	. /opt/gsdg-venv/bin/activate
fi

export HF_HOME="${HF_HOME:-${IOPS_SCRATCH_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${IOPS_SCRATCH_ROOT}/hf_datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${IOPS_SCRATCH_ROOT}/triton/cache}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC}"
export FI_PROVIDER="${FI_PROVIDER}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "$(dirname "${OUTPUT_DIR}")"
cd "${STAGE_ROOT}"

sft_args=(
	SFT/sft.py
	--model-path "${MODEL_PATH}"
	--output-dir "${OUTPUT_DIR}"
	--run-name "${RUN_NAME}"
	--dataset-name "${DATASET_NAME}"
	--dataset-split "${DATASET_SPLIT}"
	--validation-samples "${VALIDATION_SAMPLES}"
	--preprocessing-batch-size "${PREPROCESSING_BATCH_SIZE}"
	--dataset-num-proc "${DATASET_NUM_PROC}"
	--max-seq-length "${MAX_SEQ_LENGTH}"
	--truncation-side "${TRUNCATION_SIDE}"
	--torch-dtype "${TORCH_DTYPE}"
	--attn-implementation "${ATTN_IMPLEMENTATION}"
	--distributed-strategy "${DISTRIBUTED_STRATEGY}"
	--fsdp-min-num-params "${FSDP_MIN_NUM_PARAMS}"
	--fsdp-backward-prefetch "${FSDP_BACKWARD_PREFETCH}"
	--expected-world-size "${EXPECTED_WORLD_SIZE}"
	--require-distributed
	--seed "${SEED}"
	--learning-rate "${LEARNING_RATE}"
	--weight-decay "${WEIGHT_DECAY}"
	--num-train-epochs "${NUM_TRAIN_EPOCHS}"
	--max-steps "${MAX_STEPS}"
	--warmup-ratio "${WARMUP_RATIO}"
	--lr-scheduler-type "${LR_SCHEDULER_TYPE}"
	--per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
	--per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
	--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
	--dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
	--logging-steps "${LOGGING_STEPS}"
	--eval-steps "${EVAL_STEPS}"
	--save-steps "${SAVE_STEPS}"
	--save-total-limit "${SAVE_TOTAL_LIMIT}"
	--report-to "${REPORT_TO}"
	--smoke-max-steps "${SMOKE_MAX_STEPS}"
	--smoke-train-samples "${SMOKE_TRAIN_SAMPLES}"
	--smoke-validation-samples "${SMOKE_VALIDATION_SAMPLES}"
)

if [[ -n "${PREPARED_DATASET_DIR}" ]]; then
	sft_args+=(--prepared-dataset-dir "${PREPARED_DATASET_DIR}")
fi
if [[ -n "${DATASET_CONFIG}" ]]; then
	sft_args+=(--dataset-config "${DATASET_CONFIG}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
	sft_args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
	sft_args+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
	sft_args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${BF16}" == "0" ]]; then
	sft_args+=(--no-bf16)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
	sft_args+=(--no-gradient-checkpointing)
fi
if [[ "${FSDP_LIMIT_ALL_GATHERS}" == "0" ]]; then
	sft_args+=(--no-fsdp-limit-all-gathers)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
	sft_args+=(--trust-remote-code)
fi
if [[ "${OVERWRITE_OUTPUT_DIR}" == "1" ]]; then
	sft_args+=(--overwrite-output-dir)
fi
if [[ "${SMOKE_TEST}" == "1" ]]; then
	sft_args+=(--smoke-test)
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${sft_args[@]}"
INNER