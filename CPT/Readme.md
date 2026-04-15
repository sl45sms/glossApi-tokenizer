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

`CPT/cpt.py` is now a proper CLI training script instead of a hardcoded prototype.

Useful options:

- `--model-path`: aligned checkpoint directory
- `--output-dir`: persistent training output directory
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

## Clariden launcher

Use the tracked Slurm launcher in `scripts/run_apertus_greek_cpt_clariden.sh` for the single-node 4xGH200 path.

The matching Container Engine template is tracked in `edf/apertus-greek-clariden.toml`.

The launcher automatically sources `${REPO_ROOT}/.env` before calling `srun`, so keeping `HF_TOKEN=...` in the repo-local `.env` file is enough for the CE environment expansion.

## Build the Clariden image

You need the CE image before the smoke test can run. The EDF points at `${SCRATCH}/images/apertus-greek-aarch64.sqsh`, so that file must exist first.

The repo includes `scripts/build_apertus_greek_clariden_image.sh` to create it.

Default behavior:

- output image: `${SCRATCH}/images/apertus-greek-aarch64.sqsh`
- base image: `${BASE_SQSH}` when set, otherwise the script auto-detects an existing Clariden-compatible base image such as `${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh`
- runtime Python env inside the image: `/opt/apertus-greek-venv`

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

For the default repo image path, the tracked EDF already matches the produced filename. If needed, refresh your local CE definition with:

```bash
mkdir -p ~/.edf
cp edf/apertus-greek-clariden.toml ~/.edf/apertus-greek-clariden.toml
```

Smoke test example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=${SCRATCH}/apertus-greek-cpt-smoke
export SMOKE_TEST=1
sbatch --time=01:00:00 scripts/run_apertus_greek_cpt_clariden.sh
```

Production-length example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=${SCRATCH}/apertus-greek-cpt
export SMOKE_TEST=0
sbatch --time=12:00:00 scripts/run_apertus_greek_cpt_clariden.sh
```

Multi-node production example on 4 Clariden nodes / 16 GPUs:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=${SCRATCH}/apertus-greek-cpt-multinode
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
- `ATTN_IMPLEMENTATION=sdpa`
- `ATTN_IMPLEMENTATION=flash_attention_2`
- `OVERWRITE_OUTPUT_DIR=1`
- `SKIP_WARMUP=1`

Useful overrides for the multi-node launcher:

- `sbatch --nodes=4 --time=HH:MM:SS scripts/run_apertus_greek_cpt_clariden_multinode.sh`
- `TARGET_GLOBAL_BATCH_SIZE=256`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRADIENT_ACCUMULATION_STEPS=16`
- `MASTER_PORT=29501`
- `NPROC_PER_NODE=4`

The launcher stages the repo into `${SCRATCH}`, exports the Hugging Face cache paths inside the container, disables the Clariden CXI hook that can interfere with container imports, and starts the training with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 CPT/cpt.py ...
```

The default attention backend is now `sdpa`, which works without extra attention kernels in the image. Use `ATTN_IMPLEMENTATION=flash_attention_2` only when the container actually includes `flash-attn`.
The launcher also pins `TRITON_CACHE_DIR` under `${SCRATCH}` so Triton does not try to cache autotune state under an unavailable home-directory path inside the container.
It also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce allocator fragmentation on the first backward pass.

On the Clariden `normal` partition, the maximum walltime is `12:00:00`. The tracked Slurm script now uses that as its default, and you can request a shorter walltime with `sbatch --time=...`.

Longer CPT runs are expected to span multiple allocations. Re-submit the same launcher with the same `OUTPUT_DIR` and the training script will automatically resume from the latest checkpoint in `OUTPUT_DIR/warmup/` or `OUTPUT_DIR/full/`. Use `OVERWRITE_OUTPUT_DIR=1` only when you explicitly want to discard the previous phase checkpoints and restart from scratch.

## Outputs

The training script writes:

- `OUTPUT_DIR/run_config.json`: resolved run configuration and effective global batch size
- `OUTPUT_DIR/warmup/`: warm-up phase checkpoints
- `OUTPUT_DIR/full/`: full CPT phase checkpoints
- `OUTPUT_DIR/final/`: final model and tokenizer artifacts

The script validates at startup that:

- the tokenizer loads from `--model-path`
- the model loads from `--model-path`
- `len(tokenizer)` matches both the input embeddings and LM head size

That catches the main failure mode where the tokenizer exists but the checkpoint was never resized to it.