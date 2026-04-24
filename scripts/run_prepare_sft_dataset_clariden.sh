#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=prepare-sft
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00

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

IOPS_SCRATCH_ROOT="${IOPS_SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}"
HF_HOME="${HF_HOME:-${IOPS_SCRATCH_ROOT}/hf}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${IOPS_SCRATCH_ROOT}/hf_datasets}"
TMPDIR="${TMPDIR:-${IOPS_SCRATCH_ROOT}/tmp}"

MODEL_PATH="${MODEL_PATH:-/capstor/scratch/cscs/p-skarvelis/apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-400steps/final}"
DATASET_NAME="${DATASET_NAME:-swiss-ai/apertus-sft-mixture}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-2048}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
TRUNCATION_SIDE="${TRUNCATION_SIDE:-left}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-512}"
EXAMPLES_PER_SHARD="${EXAMPLES_PER_SHARD:-20000}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SEED="${SEED:-42}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-128}"
SMOKE_VALIDATION_SAMPLES="${SMOKE_VALIDATION_SAMPLES:-16}"
OVERWRITE="${OVERWRITE:-0}"

validation_tag="${VALIDATION_SAMPLES}"
if [[ "${SMOKE_TEST}" == "1" && "${validation_tag}" == "0" ]]; then
	validation_tag="${SMOKE_VALIDATION_SAMPLES}"
fi
PREPARED_DATASET_DIR="${PREPARED_DATASET_DIR:-${IOPS_SCRATCH_ROOT}/prepared-datasets/apertus-greek-sft-${MAX_SEQ_LENGTH}-${TRUNCATION_SIDE}-val${validation_tag}}"

default_dataset_num_proc=$(( ${SLURM_CPUS_PER_TASK:-64} / 2 ))
if (( default_dataset_num_proc < 1 )); then
	default_dataset_num_proc=1
fi
DATASET_NUM_PROC="${DATASET_NUM_PROC:-${default_dataset_num_proc}}"

export HF_HOME
export HF_DATASETS_CACHE
export TMPDIR
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" "$(dirname "${PREPARED_DATASET_DIR}")"

echo "Using model path: ${MODEL_PATH}" >&2
echo "Using dataset: ${DATASET_NAME}" >&2
echo "Using dataset split: ${DATASET_SPLIT}" >&2
echo "Using prepared dataset dir: ${PREPARED_DATASET_DIR}" >&2
echo "Using max_seq_length=${MAX_SEQ_LENGTH}, truncation_side=${TRUNCATION_SIDE}" >&2
echo "Using cpus_per_task=${SLURM_CPUS_PER_TASK:-64}, dataset_num_proc=${DATASET_NUM_PROC}, preprocessing_batch_size=${PREPROCESSING_BATCH_SIZE}" >&2
echo "Using HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} TMPDIR=${TMPDIR}" >&2

prep_args=(
	python
	scripts/prepare_sft_dataset.py
	--model-path "${MODEL_PATH}"
	--dataset-name "${DATASET_NAME}"
	--dataset-split "${DATASET_SPLIT}"
	--validation-samples "${VALIDATION_SAMPLES}"
	--preprocessing-batch-size "${PREPROCESSING_BATCH_SIZE}"
	--dataset-num-proc "${DATASET_NUM_PROC}"
	--max-seq-length "${MAX_SEQ_LENGTH}"
	--truncation-side "${TRUNCATION_SIDE}"
	--seed "${SEED}"
	--examples-per-shard "${EXAMPLES_PER_SHARD}"
	--output-dir "${PREPARED_DATASET_DIR}"
)

if [[ -n "${DATASET_CONFIG}" ]]; then
	prep_args+=(--dataset-config "${DATASET_CONFIG}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
	prep_args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
	prep_args+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
	prep_args+=(--trust-remote-code)
fi
if [[ "${SMOKE_TEST}" == "1" ]]; then
	prep_args+=(--smoke-test)
	prep_args+=(--smoke-train-samples "${SMOKE_TRAIN_SAMPLES}")
	prep_args+=(--smoke-validation-samples "${SMOKE_VALIDATION_SAMPLES}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
	prep_args+=(--overwrite)
fi

cd "${REPO_ROOT}"
srun --ntasks=1 --cpu-bind=cores ./run_uenv.sh "${prep_args[@]}"