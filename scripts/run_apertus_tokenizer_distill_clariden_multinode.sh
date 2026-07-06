#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=apertus-token-distill-mn
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
STAGE_ROOT="${STAGE_ROOT:-${SCRATCH}/glossapi-token-distill_${SLURM_JOB_ID}}"

BASE_MODEL="${BASE_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
OUTPUT_DIR="${OUTPUT_DIR:-${CAPSTOR_SCRATCH_ROOT}/apertus-greek-tokenizer-distill-unified}"
TOKEN_FILE="${TOKEN_FILE:-${STAGE_ROOT}/artifacts/vocab_candidates/selected_tokens_v1.txt}"
BASE_TOKENIZER="${BASE_TOKENIZER:-${STAGE_ROOT}/artifacts/tokenizers/apertus-base}"
EXTENDED_TOKENIZER="${EXTENDED_TOKENIZER:-${STAGE_ROOT}/artifacts/tokenizers/apertus-greek-v1}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/unified_token_distill_report.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_WORLD_SIZE_COMPUTED=$((SLURM_NNODES * NPROC_PER_NODE))
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-${EXPECTED_WORLD_SIZE_COMPUTED}}"
if (( EXPECTED_WORLD_SIZE != EXPECTED_WORLD_SIZE_COMPUTED )); then
	echo "EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE} does not match SLURM_NNODES * NPROC_PER_NODE = ${EXPECTED_WORLD_SIZE_COMPUTED}." >&2
	exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29601}"
NNODES="${NNODES:-${SLURM_NNODES}}"

INIT_STRATEGY="${INIT_STRATEGY:-retok-distill}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
REQUIRE_XIELU="${REQUIRE_XIELU:-1}"
RUN_GREEK_MMLU_EVAL="${RUN_GREEK_MMLU_EVAL:-0}"
FAIL_BELOW_BASE="${FAIL_BELOW_BASE:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-0}"
REFRESH_CONTEXT_CACHE="${REFRESH_CONTEXT_CACHE:-0}"

DISTILL_STEPS="${DISTILL_STEPS:-500}"
DISTILL_LR="${DISTILL_LR:-5e-6}"
DISTILL_SAMPLES="${DISTILL_SAMPLES:-5000}"
DISTILL_CONTEXTS_PER_TOKEN="${DISTILL_CONTEXTS_PER_TOKEN:-8}"
DISTILL_MAX_SEQ_LENGTH="${DISTILL_MAX_SEQ_LENGTH:-1024}"
DISTILL_WARMUP_STEPS="${DISTILL_WARMUP_STEPS:-50}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-16}"
DISTILL_REG_WEIGHT="${DISTILL_REG_WEIGHT:-0.1}"
DISTILL_STREAM_TIMEOUT="${DISTILL_STREAM_TIMEOUT:-600}"
DISTILL_SYNC_INTERVAL="${DISTILL_SYNC_INTERVAL:-10}"
DISTILL_SYNC_START_STEP="${DISTILL_SYNC_START_STEP:-50}"
DISTILL_CHECKPOINT_INTERVAL="${DISTILL_CHECKPOINT_INTERVAL:-100}"
DISTILL_MAX_INVALID_STEPS="${DISTILL_MAX_INVALID_STEPS:-20}"
DISTILL_LR_DECAY_ON_INVALID="${DISTILL_LR_DECAY_ON_INVALID:-0.5}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-7200}"
FINEWEB2_CACHE_DIR="${FINEWEB2_CACHE_DIR:-${IOPS_SCRATCH_ROOT}/FineWeb2-HQ}"
EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${OUTPUT_DIR}/greek_mmlu_unified_distill_eval.json}"

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
tar -C "${REPO_ROOT}" -cz \
	vocab-extension \
	evaluation \
	scripts \
	repo_tokenizer.py \
	Agents.md \
	Readme.md \
	requirements.txt \
	artifacts/vocab_candidates/selected_tokens_v1.txt \
	artifacts/tokenizers/apertus-base \
	artifacts/tokenizers/apertus-greek-v1 \
| tar -xz -C "${STAGE_ROOT}"

echo "Using CE environment: ${CE_ENVIRONMENT}" >&2
echo "Using MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}" >&2
echo "Using ${SLURM_NNODES} node(s), ${NPROC_PER_NODE} process(es) per node, world_size=${EXPECTED_WORLD_SIZE}" >&2
echo "Using OUTPUT_DIR=${OUTPUT_DIR}" >&2
echo "Using TOKEN_FILE=${TOKEN_FILE}" >&2
echo "Using EXTENDED_TOKENIZER=${EXTENDED_TOKENIZER}" >&2
echo "Using FINEWEB2_CACHE_DIR=${FINEWEB2_CACHE_DIR}" >&2
echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} FI_PROVIDER=${FI_PROVIDER}" >&2

SRUN_EXPORT="ALL"
SRUN_EXPORT+=",STAGE_ROOT=${STAGE_ROOT}"
SRUN_EXPORT+=",TOKEN_FILE=${TOKEN_FILE}"
SRUN_EXPORT+=",BASE_TOKENIZER=${BASE_TOKENIZER}"
SRUN_EXPORT+=",EXTENDED_TOKENIZER=${EXTENDED_TOKENIZER}"
SRUN_EXPORT+=",BASE_MODEL=${BASE_MODEL}"
SRUN_EXPORT+=",OUTPUT_DIR=${OUTPUT_DIR}"
SRUN_EXPORT+=",REPORT_PATH=${REPORT_PATH}"
SRUN_EXPORT+=",MASTER_ADDR=${MASTER_ADDR}"
SRUN_EXPORT+=",MASTER_PORT=${MASTER_PORT}"
SRUN_EXPORT+=",NNODES=${NNODES}"
SRUN_EXPORT+=",NPROC_PER_NODE=${NPROC_PER_NODE}"
SRUN_EXPORT+=",EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE}"
SRUN_EXPORT+=",INIT_STRATEGY=${INIT_STRATEGY}"
SRUN_EXPORT+=",TORCH_DTYPE=${TORCH_DTYPE}"
SRUN_EXPORT+=",ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
SRUN_EXPORT+=",TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
SRUN_EXPORT+=",OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR}"
SRUN_EXPORT+=",REQUIRE_XIELU=${REQUIRE_XIELU}"
SRUN_EXPORT+=",RUN_GREEK_MMLU_EVAL=${RUN_GREEK_MMLU_EVAL}"
SRUN_EXPORT+=",FAIL_BELOW_BASE=${FAIL_BELOW_BASE}"
SRUN_EXPORT+=",USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
SRUN_EXPORT+=",REFRESH_CONTEXT_CACHE=${REFRESH_CONTEXT_CACHE}"
SRUN_EXPORT+=",DISTILL_STEPS=${DISTILL_STEPS}"
SRUN_EXPORT+=",DISTILL_LR=${DISTILL_LR}"
SRUN_EXPORT+=",DISTILL_SAMPLES=${DISTILL_SAMPLES}"
SRUN_EXPORT+=",DISTILL_CONTEXTS_PER_TOKEN=${DISTILL_CONTEXTS_PER_TOKEN}"
SRUN_EXPORT+=",DISTILL_MAX_SEQ_LENGTH=${DISTILL_MAX_SEQ_LENGTH}"
SRUN_EXPORT+=",DISTILL_WARMUP_STEPS=${DISTILL_WARMUP_STEPS}"
SRUN_EXPORT+=",DISTILL_BATCH_SIZE=${DISTILL_BATCH_SIZE}"
SRUN_EXPORT+=",DISTILL_REG_WEIGHT=${DISTILL_REG_WEIGHT}"
SRUN_EXPORT+=",DISTILL_STREAM_TIMEOUT=${DISTILL_STREAM_TIMEOUT}"
SRUN_EXPORT+=",DISTILL_SYNC_INTERVAL=${DISTILL_SYNC_INTERVAL}"
SRUN_EXPORT+=",DISTILL_SYNC_START_STEP=${DISTILL_SYNC_START_STEP}"
SRUN_EXPORT+=",DISTILL_CHECKPOINT_INTERVAL=${DISTILL_CHECKPOINT_INTERVAL}"
SRUN_EXPORT+=",DISTILL_MAX_INVALID_STEPS=${DISTILL_MAX_INVALID_STEPS}"
SRUN_EXPORT+=",DISTILL_LR_DECAY_ON_INVALID=${DISTILL_LR_DECAY_ON_INVALID}"
SRUN_EXPORT+=",DIST_TIMEOUT_SECONDS=${DIST_TIMEOUT_SECONDS}"
SRUN_EXPORT+=",FINEWEB2_CACHE_DIR=${FINEWEB2_CACHE_DIR}"
SRUN_EXPORT+=",EVAL_OUTPUT_JSON=${EVAL_OUTPUT_JSON}"
SRUN_EXPORT+=",IOPS_SCRATCH_ROOT=${IOPS_SCRATCH_ROOT}"
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${IOPS_SCRATCH_ROOT}/triton/cache}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="false"
export DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC}"
export FI_PROVIDER="${FI_PROVIDER}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "$(dirname "${OUTPUT_DIR}")" "${FINEWEB2_CACHE_DIR}"
cd "${STAGE_ROOT}"

python - <<'PY'
import sys
import torch

try:
    import xielu.ops  # noqa: F401
    _ = torch.classes.xielu.XIELU()
except Exception as exc:
    print(f"xIELU CUDA kernel unavailable: {exc}", file=sys.stderr)
    sys.exit(42)
print("xIELU CUDA kernel check: OK")
PY

common_args=(
	vocab-extension/distil-vocab-extension/unified_token_distill.py
	--token-file "${TOKEN_FILE}"
	--base-tokenizer "${BASE_TOKENIZER}"
	--base-model "${BASE_MODEL}"
	--extended-tokenizer "${EXTENDED_TOKENIZER}"
	--output-dir "${OUTPUT_DIR}"
	--init-strategy "${INIT_STRATEGY}"
	--torch-dtype "${TORCH_DTYPE}"
	--attn-implementation "${ATTN_IMPLEMENTATION}"
	--distill-steps "${DISTILL_STEPS}"
	--distill-lr "${DISTILL_LR}"
	--distill-samples "${DISTILL_SAMPLES}"
	--distill-contexts-per-token "${DISTILL_CONTEXTS_PER_TOKEN}"
	--distill-max-seq-length "${DISTILL_MAX_SEQ_LENGTH}"
	--distill-warmup-steps "${DISTILL_WARMUP_STEPS}"
	--distill-batch-size "${DISTILL_BATCH_SIZE}"
	--distill-reg-weight "${DISTILL_REG_WEIGHT}"
	--distill-stream-timeout "${DISTILL_STREAM_TIMEOUT}"
	--distill-sync-interval "${DISTILL_SYNC_INTERVAL}"
	--distill-sync-start-step "${DISTILL_SYNC_START_STEP}"
	--distill-checkpoint-interval "${DISTILL_CHECKPOINT_INTERVAL}"
	--distill-max-invalid-steps "${DISTILL_MAX_INVALID_STEPS}"
	--distill-lr-decay-on-invalid "${DISTILL_LR_DECAY_ON_INVALID}"
	--fineweb2-cache-dir "${FINEWEB2_CACHE_DIR}"
	--report-path "${REPORT_PATH}"
	--eval-output-json "${EVAL_OUTPUT_JSON}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
	common_args+=(--trust-remote-code)
fi
if [[ "${OVERWRITE_OUTPUT_DIR}" == "1" ]]; then
	common_args+=(--overwrite)
fi
if [[ "${REQUIRE_XIELU}" == "1" ]]; then
	common_args+=(--require-xielu)
else
	common_args+=(--no-require-xielu)
fi
if [[ "${RUN_GREEK_MMLU_EVAL}" == "1" ]]; then
	common_args+=(--run-greek-mmlu-eval)
else
	common_args+=(--no-run-greek-mmlu-eval)
fi
if [[ "${FAIL_BELOW_BASE}" == "1" ]]; then
	common_args+=(--fail-below-base)
else
	common_args+=(--no-fail-below-base)
fi
if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
	common_args+=(--use-chat-template)
else
	common_args+=(--no-use-chat-template)
fi
if [[ "${REFRESH_CONTEXT_CACHE}" == "1" ]]; then
	common_args+=(--refresh-context-cache)
fi

python -m torch.distributed.run \
	--nnodes "${NNODES}" \
	--nproc_per_node "${NPROC_PER_NODE}" \
	--node_rank "${SLURM_NODEID}" \
	--master_addr "${MASTER_ADDR}" \
	--master_port "${MASTER_PORT}" \
	"${common_args[@]}"
INNER
