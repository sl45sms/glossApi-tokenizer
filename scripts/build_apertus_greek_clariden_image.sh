#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=build-apertus-greek-img
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="/users/p-skarvelis/glossApi-Tokenizer"
if [[ ! -d "${REPO_ROOT}" ]]; then
	echo "Repository root not found at ${REPO_ROOT}" >&2
	exit 1
fi

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH must be set before building the Clariden image." >&2
	exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
	echo "This image build must run on an aarch64 system such as Clariden." >&2
	exit 1
fi

log() {
	printf '[%s] %s\n' "$(date -Is)" "$*"
}

detect_base_sqsh() {
	local candidates=()
	if [[ -n "${BASE_SQSH:-}" ]]; then
		candidates+=("${BASE_SQSH}")
	fi
	candidates+=(
		"${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh"
		"${SCRATCH}/images/gsdg-qwen3_clariden_flashinfer_latest.sqsh"
	)

	for candidate in "${candidates[@]}"; do
		if [[ -f "${candidate}" ]]; then
			printf '%s\n' "${candidate}"
			return 0
		fi
	done

	return 1
}

if ! BASE_SQSH="$(detect_base_sqsh)"; then
	cat >&2 <<EOF
No Clariden-compatible base .sqsh image was found.

Set BASE_SQSH to an existing aarch64 CE image, for example:
  export BASE_SQSH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh

Then re-run:
  sbatch scripts/build_apertus_greek_clariden_image.sh
EOF
	exit 1
fi

OUT_SQSH="${OUT_SQSH:-${SCRATCH}/images/apertus-greek-aarch64.sqsh}"
NAME="${NAME:-apertus-greek-${SLURM_JOB_ID:-manual}}"
ENROOT_CREATE_PROCS="${ENROOT_CREATE_PROCS:-8}"
MKSQUASHFS_PROCS="${MKSQUASHFS_PROCS:-8}"

log "Using BASE_SQSH=${BASE_SQSH}"
log "Writing OUT_SQSH=${OUT_SQSH}"
log "Enroot container name: ${NAME}"
log "Using ENROOT_CREATE_PROCS=${ENROOT_CREATE_PROCS}"
log "Using MKSQUASHFS_PROCS=${MKSQUASHFS_PROCS}"

build_cpu_list() {
	local limit="$1"
	python3 - "$limit" <<'PY'
import os
import sys

limit = max(1, int(sys.argv[1]))
cpus = sorted(os.sched_getaffinity(0))
selected = cpus[: min(limit, len(cpus))]
print(",".join(str(cpu) for cpu in selected))
PY
}

run_with_limited_cpus() {
	local limit="$1"
	shift

	if command -v taskset >/dev/null 2>&1; then
		local cpu_list
		cpu_list="$(build_cpu_list "$limit")"
		log "Running with CPU affinity: ${cpu_list}"
		taskset -c "${cpu_list}" "$@"
	else
		"$@"
	fi
}

mkdir -p "$(dirname "${OUT_SQSH}")"
rm -f "${OUT_SQSH}"
enroot remove -f "${NAME}" >/dev/null 2>&1 || true

export ENROOT_SLURM_HOOK=off
export OCI_ANNOTATION_com__hooks__cxi__enabled=false

log "enroot create"
run_with_limited_cpus "${ENROOT_CREATE_PROCS}" enroot create -n "${NAME}" "${BASE_SQSH}"

cleanup() {
	enroot remove -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "enroot start: install repo runtime dependencies"
tar -C "${REPO_ROOT}" -cz requirements.txt requirements-clariden-runtime.txt CPT scripts edf Agents.md Readme.md | \
	enroot start --rw --mount "${SCRATCH}:/mnt/cache" "${NAME}" bash -lc '
	set -euo pipefail

	mkdir -p /workspace
	tar -xz -C /workspace

	python3 -m venv --system-site-packages /opt/apertus-greek-venv
	. /opt/apertus-greek-venv/bin/activate

	python -m pip install --upgrade pip
	python -m pip install -r /workspace/requirements-clariden-runtime.txt

	if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
		python -m pip install --upgrade wheel cmake ninja packaging
		python -m pip install --no-build-isolation flash-attn
	fi

	python - <<"PY"
import torch
import transformers
import datasets
import accelerate

print("torch_version=" + torch.__version__)
print("transformers_version=" + transformers.__version__)
print("datasets_version=" + datasets.__version__)
print("accelerate_version=" + accelerate.__version__)
PY
	'

ROOTFS=""
for candidate in "$HOME/.local/share/enroot/$NAME" "/dev/shm/$(id -nu)/enrootdata/$NAME"; do
	if [[ -d "$candidate" ]]; then
		ROOTFS="$candidate"
		break
	fi
done

if [[ -z "${ROOTFS}" ]]; then
	log "ERROR: cannot find enroot rootfs for ${NAME}"
	exit 2
fi

log "mksquashfs export"
run_with_limited_cpus "${MKSQUASHFS_PROCS}" \
	mksquashfs "${ROOTFS}" "${OUT_SQSH}" -processors "${MKSQUASHFS_PROCS}" \
	-comp zstd -b 131072 -noappend -all-root \
	-e etc/motd etc/xthostname mnt/cache

log "DONE"