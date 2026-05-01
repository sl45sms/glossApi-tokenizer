# CPT

This stage runs continued pretraining on an Apertus checkpoint that has already been resized to the extended tokenizer.

The training mixture is:

- 90% Greek from `epfml/FineWeb2-HQ`, config `ell_Grek`
- 10% English anchor from `epfml/FineWeb-HQ`

The English anchor is kept to reduce reasoning drift while the model adapts to Greek text.

## Preconditions

Before launching CPT, you need an aligned checkpoint, not just the tokenizer directory.

Build or refresh it with:

```bash
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
	--base-tokenizer artifacts/tokenizers/apertus-base \
	--token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
	--base-model swiss-ai/Apertus-8B-Instruct-2509 \
	--checkpoint-output-dir "${SCRATCH}/apertus-greek-init" \
	--torch-dtype bfloat16 \
	--overwrite
```

That checkpoint path is what `CPT/cpt.py` must receive through `--model-path`.

Older aligned checkpoints saved from the earlier tokenizer workflow may advertise `tokenizer_class: TokenizersBackend` in `tokenizer_config.json`. The current CPT loader handles that compatibility case automatically, and the tokenizer extraction/extension scripts now normalize future saved tokenizer configs to `PreTrainedTokenizerFast`.

## Training entry point

`CPT/cpt.py` is a CLI training script.
Useful options:

- `--model-path`: aligned checkpoint directory
- `--output-dir`: persistent training output directory
- `--prepared-train-dataset-dir`: optional directory of packed parquet shards produced offline by `scripts/prepare_cpt_dataset.py`
- `--save-total-limit`: maximum number of retained intermediate checkpoints per phase; use `all` to disable pruning entirely
- `--benchmark-mode`: disable checkpoint saves and final export, while still writing phase metrics JSON
- `--smoke-test`: short validation run
- `--skip-warmup`: skip the embedding-only phase
- `--attn-implementation sdpa`: default backend that works without extra attention kernels
- streaming text is tokenized with `padding="max_length"` so distributed smoke runs do not fail on variable-length iterable batches
- smoke mode defaults to `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`, and `smoke_max_seq_length=1024` so the first Clariden validation run stays well below the full 2048-token production footprint
- `--expected-world-size 4 --require-distributed`: fail fast if the launch shape is wrong

Inspect the full CLI with:

```bash
python CPT/cpt.py --help
```

## Prepare packed training data offline

The repository includes the `scripts/prepare_cpt_dataset.py` to move tokenization and packing out of the training hot path.

The script streams the configured Greek and English datasets once, tokenizes them with the extended tokenizer, inserts EOS separators between documents, packs fixed-length sequences, and writes parquet shards plus `metadata.json`.

`--greek-dataset` and `--english-dataset` accept either a Hugging Face dataset id or a local `.json` / `.jsonl` file. This is useful when you want to pack the filtered targeted-CPT output from `targetedCPT-DatasetGen/filter.py`, for example `${SCRATCH}/targeted-cpt/curated_greek_cpt.jsonl`.

Recommended placement on Clariden:

- prepared parquet shards: `/iopsstor/scratch`
- checkpoints and model outputs: `/capstor/scratch`
- Hugging Face and Triton caches: `/iopsstor/scratch`

Example:

```bash
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
	--tokenizer-path artifacts/tokenizers/apertus-greek-v1 \
	--output-dir /iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-2048 \
	--max-seq-length 2048 \
	--sequences-per-shard 512 \
	--overwrite
```

For a short benchmark corpus instead of a full pass, cap the output:

```bash
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
	--tokenizer-path artifacts/tokenizers/apertus-greek-v1 \
	--output-dir /iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-bench \
	--max-seq-length 2048 \
	--max-output-sequences 4096 \
	--overwrite
```

To pack only the filtered Greek JSONL without the English anchor stream:

```bash
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
	--tokenizer-path artifacts/tokenizers/apertus-greek-v1 \
	--output-dir /iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-targeted-packed-2048 \
	--greek-dataset ${SCRATCH}/targeted-cpt/curated_greek_cpt.jsonl \
	--greek-probability 1.0 \
	--english-probability 0 \
	--max-seq-length 2048 \
	--overwrite
```

When `--prepared-train-dataset-dir` is set, `CPT/cpt.py` loads the parquet shards from disk and skips the live streaming/tokenization path entirely. The sequence length in `metadata.json` must match the run's effective sequence length.

## Clariden launcher

Use the tracked Slurm launcher in `scripts/run_apertus_greek_cpt_clariden.sh` for the single-node 4xGH200 path.

The matching Container Engine template is tracked in `edf/apertus-greek-clariden.toml`.

The launcher automatically sources `${REPO_ROOT}/.env` before calling `srun`, so keeping `HF_TOKEN=...` in the repo-local `.env` file is enough for the CE environment expansion.

The tracked launchers now default to the following layout unless you override them:

- `IOPS_SCRATCH_ROOT=/iopsstor/scratch/cscs/${USER}` for prepared datasets and runtime caches
- `CAPSTOR_SCRATCH_ROOT=/capstor/scratch/cscs/${USER}` for `OUTPUT_DIR`
- `PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-${MAX_SEQ_LENGTH}` when that directory already exists

## Build the Clariden image

You need the CE image before the smoke test can run. The EDF points at `${SCRATCH}/images/apertus-greek-aarch64.sqsh`, so that file must exist first.

The repo includes `scripts/build_apertus_greek_clariden_image.sh` to create it.

Default behavior:

- output image: `${SCRATCH}/images/apertus-greek-aarch64.sqsh`
- base image: `${BASE_SQSH}` when set, otherwise the script auto-detects an existing Clariden-compatible base image such as `${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh`
- runtime Python env inside the image: `/opt/apertus-greek-venv`
- xIELU install: enabled by default from `git+https://github.com/nickjbrowning/XIELU`

Build it with:

```bash
sbatch scripts/build_apertus_greek_clariden_image.sh
```

If you want to pin the base image explicitly:

```bash
export BASE_SQSH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh
sbatch scripts/build_apertus_greek_clariden_image.sh
```

The build script requests `128G` by default, installs only the runtime packages needed for CPT, limits `enroot create` CPU affinity, and caps `mksquashfs` to 8 workers to reduce the export-time memory spike.

The build now also attempts to install `xielu` by default with `--no-build-isolation` so it compiles against the container's existing Torch build, and prints `xielu_import=OK` during the verification step when the package imports successfully.

If you need to disable that install for troubleshooting, set:

```bash
export INSTALL_XIELU=0
sbatch scripts/build_apertus_greek_clariden_image.sh
```

If you hit memory pressure during squash export, lower the worker counts further when submitting:

```bash
export BASE_SQSH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh
export ENROOT_CREATE_PROCS=4
export MKSQUASHFS_PROCS=4
sbatch --mem=160G scripts/build_apertus_greek_clariden_image.sh
```

If you later want to experiment with `flash_attention_2`, you can ask the build job to try installing `flash-attn`:

```bash
export BASE_SQSH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh
export INSTALL_FLASH_ATTN=1
sbatch scripts/build_apertus_greek_clariden_image.sh
```

If you want to pin a different xIELU pip source, override the package spec directly:

```bash
export XIELU_PIP_SPEC='git+https://github.com/nickjbrowning/XIELU'
sbatch scripts/build_apertus_greek_clariden_image.sh
```

For the default repo image path, the tracked EDF already matches the produced filename. If needed, refresh your local CE definition with:

```bash
mkdir -p ~/.edf
cp edf/apertus-greek-clariden.toml ~/.edf/apertus-greek-clariden.toml
```

Smoke test example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-smoke
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-1024
export SMOKE_TEST=1
sbatch --time=01:00:00 scripts/run_apertus_greek_cpt_clariden.sh
```

Short benchmark example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-bench
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-2048
export BENCHMARK_MODE=1
export SMOKE_TEST=0
export SKIP_WARMUP=1
export FULL_MAX_STEPS=50
sbatch --time=00:30:00 scripts/run_apertus_greek_cpt_clariden.sh
```

Production-length example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-2048
export SMOKE_TEST=0
sbatch --time=12:00:00 scripts/run_apertus_greek_cpt_clariden.sh
```

Multi-node production example on 4 Clariden nodes / 16 GPUs:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-multinode
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-2048
export SMOKE_TEST=0
sbatch --nodes=4 --time=12:00:00 scripts/run_apertus_greek_cpt_clariden_multinode.sh
```

The multi-node launcher is data-parallel, not tensor-parallel. That means it improves throughput, but it does not reduce per-GPU model memory. The tracked multi-node defaults target the same effective global batch size of `256` as the single-node launcher by deriving:

- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `TARGET_GLOBAL_BATCH_SIZE=256`
- `GRADIENT_ACCUMULATION_STEPS=256 / (PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE)`

On 4 nodes with 4 GPUs each, that resolves to `GRADIENT_ACCUMULATION_STEPS=16`.

The multi-node launcher also exports the Clariden network settings required beyond one node:

- `NCCL_SOCKET_IFNAME=nmn0`
- `GLOO_SOCKET_IFNAME=nmn0`
- `NCCL_CROSS_NIC=1`
- `FI_PROVIDER=cxi`

The tracked launcher now uses conservative non-smoke defaults for the first real 2048-token run:

- `MAX_SEQ_LENGTH=2048`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRADIENT_ACCUMULATION_STEPS=64`

That preserves the intended effective global batch of `256` across 4 GPUs while avoiding the previously observed OOM-prone `16 x 4` microbatch shape. Only raise the per-device batch size after a short 2048-token probe confirms memory headroom.

Useful overrides for the launcher:

- `sbatch --time=HH:MM:SS scripts/run_apertus_greek_cpt_clariden.sh`
- `FULL_MAX_STEPS=50000`
- `WARMUP_MAX_STEPS=2000`
- `FULL_WARMUP_STEPS=1000`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRADIENT_ACCUMULATION_STEPS=64`
- `SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `SMOKE_GRADIENT_ACCUMULATION_STEPS=1`
- `SMOKE_MAX_SEQ_LENGTH=1024`
- `PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-2048`
- `BENCHMARK_MODE=1`
- `ATTN_IMPLEMENTATION=sdpa`
- `ATTN_IMPLEMENTATION=flash_attention_2`
- `OVERWRITE_OUTPUT_DIR=1`
- `SKIP_WARMUP=1`

Useful overrides for the multi-node launcher:

- `sbatch --nodes=4 --time=HH:MM:SS scripts/run_apertus_greek_cpt_clariden_multinode.sh`
- `TARGET_GLOBAL_BATCH_SIZE=256`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRADIENT_ACCUMULATION_STEPS=16`
- `PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-2048`
- `BENCHMARK_MODE=1`
- `MASTER_PORT=29501`
- `NPROC_PER_NODE=4`

The launcher stages the repo into `${SCRATCH}`, exports the Hugging Face cache paths inside the container, disables the Clariden CXI hook that can interfere with container imports, and starts the training with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 CPT/cpt.py ...
```

The default attention backend is now `sdpa`, which works without extra attention kernels in the image. Use `ATTN_IMPLEMENTATION=flash_attention_2` only when the container actually includes `flash-attn`.
The launcher also pins `HF_HOME`, `HF_DATASETS_CACHE`, and `TRITON_CACHE_DIR` under `/iopsstor/scratch/cscs/${USER}` so active data and compiler caches stay on the fast scratch tier.
It also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce allocator fragmentation on the first backward pass.

On the Clariden `normal` partition, the maximum walltime is `12:00:00`. The tracked Slurm script now uses that as its default, and you can request a shorter walltime with `sbatch --time=...`.

Longer CPT runs are expected to span multiple allocations. Re-submit the same launcher with the same `OUTPUT_DIR` and the training script will automatically resume from the latest checkpoint in `OUTPUT_DIR/warmup/` or `OUTPUT_DIR/full/`. Use `OVERWRITE_OUTPUT_DIR=1` only when you explicitly want to discard the previous phase checkpoints and restart from scratch.

By default, the tracked CPT launchers keep up to `3` intermediate checkpoints per phase through `SAVE_TOTAL_LIMIT=3`. To keep every intermediate checkpoint instead of pruning older `checkpoint-*` directories, submit the job with:

```bash
export SAVE_TOTAL_LIMIT=all
```

That works in both `scripts/run_apertus_greek_cpt_clariden.sh` and `scripts/run_apertus_greek_cpt_clariden_multinode.sh`. The final export in `OUTPUT_DIR/final/` is still written separately at the end of a non-benchmark run.

## Outputs

The training script writes:

- `OUTPUT_DIR/run_config.json`: resolved run configuration and effective global batch size
- `OUTPUT_DIR/warmup/`: warm-up phase checkpoints
- `OUTPUT_DIR/full/`: full CPT phase checkpoints
- `OUTPUT_DIR/final/`: final model and tokenizer artifacts

Each phase directory also gets `phase_metrics.json` with:

- world size
- effective global batch size
- effective sequence length
- phase token budget
- cluster tokens per second
- tokens per second per GPU

In `--benchmark-mode`, the phase metrics are still written, but checkpoint saves and `OUTPUT_DIR/final/` are skipped.

The offline dataset preparation script writes:

- `PREPARED_TRAIN_DATASET_DIR/metadata.json`: tokenizer path, dataset mix, sequence length, and shard inventory
- `PREPARED_TRAIN_DATASET_DIR/*.parquet`: packed `input_ids` shards ready for direct loading in `CPT/cpt.py`

The script validates at startup that:

- the tokenizer loads from `--model-path`
- the model loads from `--model-path`
- `len(tokenizer)` matches both the input embeddings and LM head size

That catches the main failure mode where the tokenizer exists but the checkpoint was never resized to it.