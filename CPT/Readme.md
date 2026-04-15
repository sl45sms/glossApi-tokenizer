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

## Training entry point

`CPT/cpt.py` is now a proper CLI training script instead of a hardcoded prototype.

Useful options:

- `--model-path`: aligned checkpoint directory
- `--output-dir`: persistent training output directory
- `--smoke-test`: short validation run
- `--skip-warmup`: skip the embedding-only phase
- `--attn-implementation eager`: fallback when `flash_attention_2` is unavailable
- `--expected-world-size 4 --require-distributed`: fail fast if the launch shape is wrong

Inspect the full CLI with:

```bash
python CPT/cpt.py --help
```

## Clariden launcher

Use the tracked Slurm launcher in `scripts/run_apertus_greek_cpt_clariden.sh` for the single-node 4xGH200 path.

The matching Container Engine template is tracked in `edf/apertus-greek-clariden.toml`.

Smoke test example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=${SCRATCH}/apertus-greek-cpt-smoke
export SMOKE_TEST=1
sbatch scripts/run_apertus_greek_cpt_clariden.sh
```

Production-length example:

```bash
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=${SCRATCH}/apertus-greek-cpt
export SMOKE_TEST=0
sbatch scripts/run_apertus_greek_cpt_clariden.sh
```

Useful overrides for the launcher:

- `FULL_MAX_STEPS=50000`
- `WARMUP_MAX_STEPS=2000`
- `FULL_WARMUP_STEPS=1000`
- `PER_DEVICE_TRAIN_BATCH_SIZE=16`
- `GRADIENT_ACCUMULATION_STEPS=4`
- `ATTN_IMPLEMENTATION=eager`
- `OVERWRITE_OUTPUT_DIR=1`
- `SKIP_WARMUP=1`

The launcher stages the repo into `${SCRATCH}`, exports the Hugging Face cache paths inside the container, disables the Clariden CXI hook that can interfere with container imports, and starts the training with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 CPT/cpt.py ...
```

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