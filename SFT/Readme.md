Use the SFT mixture to fine-tune the CPT checkpoint from stage 3 so the model learns to use the expanded Greek vocabulary in instruction-following contexts.

Defaults wired into the trainer:

- Curated CPT model path: `/capstor/scratch/cscs/p-skarvelis/apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-400steps/final`
- SFT dataset: `swiss-ai/apertus-sft-mixture`
- Related reference repo: `/users/p-skarvelis/glossApi-Trainer`

What is now in this folder:

- `sft.py`: full-finetuning entrypoint built on `transformers.Trainer`
- `run_apertus_greek_sft_clariden.sh`: single-node Clariden launcher using the existing CE/EDF workflow from this repo
- `run_apertus_greek_sft_clariden_multinode.sh`: multi-node Clariden launcher using Slurm rendezvous and `torch.distributed.run` across nodes
- `../scripts/prepare_sft_dataset.py`: one-time SFT dataset exporter that renders chats, tokenizes them, and writes prepared parquet shards for reuse
- `../scripts/run_prepare_sft_dataset_clariden.sh`: single-node Clariden CPU-heavy launcher for the prepared-SFT-dataset export path

Trainer behavior:

- loads the tokenizer and model from the same CPT final checkpoint path
- reads `swiss-ai/apertus-sft-mixture`, which currently exposes only a `train` split
- normalizes each nested `messages` record into plain chat turns, renders it with the Apertus chat template, and tokenizes the rendered chat
- applies assistant-only loss masking by training only on tokens between `<|assistant_start|>` and `<|assistant_end|>`
- optionally loads a prepared dataset directory produced by `scripts/prepare_sft_dataset.py`, which avoids repeating the raw-chat render/tokenize step on every distributed rank
- defaults to `max_seq_length=1024`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `GRADIENT_CHECKPOINTING=0`, and `ATTN_IMPLEMENTATION=eager` for a safer first full-finetune shape on 4 GH200 GPUs
- supports `--distributed-strategy fsdp_full_shard` to shard model and optimizer state across the 4 Clariden ranks when longer-context runs exceed DDP memory headroom

Prepared dataset workflow:

- Use `scripts/prepare_sft_dataset.py` to render and tokenize the SFT mixture once, then store `train/` and optional `eval/` parquet shards plus `metadata.json` on shared storage.
- `SFT/sft.py` now accepts `--prepared-dataset-dir` and loads those shards directly instead of running `dataset.map(...)` over the raw chat corpus at startup.
- Both Clariden launchers now auto-use `PREPARED_DATASET_DIR` when set, and they also auto-detect `${IOPS_SCRATCH_ROOT}/prepared-datasets/apertus-greek-sft-${MAX_SEQ_LENGTH}-${TRUNCATION_SIDE}-val${VALIDATION_SAMPLES}` when that directory already exists.

Prepare a reusable 1024-token dataset with a held-out eval split:

```bash
./run_uenv.sh python scripts/prepare_sft_dataset.py \
	--model-path /capstor/scratch/cscs/p-skarvelis/apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-400steps/final \
	--validation-samples 2048 \
	--max-seq-length 1024 \
	--output-dir /iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-sft-1024-left-val2048 \
	--overwrite
```

Or submit the CPU-heavy single-node Clariden launcher, which defaults to a larger CPU allocation and derives `DATASET_NUM_PROC` from the Slurm CPU count:

```bash
OVERWRITE=1 \
VALIDATION_SAMPLES=2048 \
MAX_SEQ_LENGTH=1024 \
sbatch scripts/run_prepare_sft_dataset_clariden.sh
```

Then launch SFT with the prepared dataset explicitly:

```bash
PREPARED_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-sft-1024-left-val2048 \
VALIDATION_SAMPLES=2048 \
OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-sft \
sbatch SFT/run_apertus_greek_sft_clariden_multinode.sh
```

Because the dataset has no published validation split, evaluation is off by default. To enable periodic eval, hold out a subset from train with `VALIDATION_SAMPLES` or `--validation-samples`.

The current curated CPT checkpoint uses `hidden_act=xielu`. In this repo's Clariden SFT runs, enabling gradient checkpointing on that checkpoint caused NaN gradients almost immediately, `ATTN_IMPLEMENTATION=sdpa` hit a fused MHA backward runtime failure, and `ATTN_IMPLEMENTATION=eager` under plain DDP at `max_seq_length=2048` ran out of memory at per-device batch size 1. The launcher now defaults to `GRADIENT_CHECKPOINTING=0`, `ATTN_IMPLEMENTATION=eager`, and `MAX_SEQ_LENGTH=1024`, and it auto-selects `DISTRIBUTED_STRATEGY=fsdp_full_shard` when you request a longer context such as `MAX_SEQ_LENGTH=2048` on multiple GPUs.

Smoke test on Clariden:

```bash
SMOKE_TEST=1 \
VALIDATION_SAMPLES=16 \
sbatch SFT/run_apertus_greek_sft_clariden.sh
```

Default full SFT launch:

```bash
VALIDATION_SAMPLES=2048 \
OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-sft \
sbatch SFT/run_apertus_greek_sft_clariden.sh
```

Multi-node Clariden launch:

- `SFT/run_apertus_greek_sft_clariden_multinode.sh` starts from 4 nodes / 16 GPUs and derives `GRADIENT_ACCUMULATION_STEPS` from `TARGET_GLOBAL_BATCH_SIZE=32` by default so the effective global batch stays aligned with the single-node default shape.
- The multi-node launcher reuses the same Clariden network settings as the CPT multi-node path and uses `MASTER_ADDR` / `MASTER_PORT` rendezvous instead of `--standalone`.

Example multi-node full SFT launch:

```bash
VALIDATION_SAMPLES=2048 \
OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-sft \
sbatch SFT/run_apertus_greek_sft_clariden_multinode.sh
```

Example multi-node 2048-token launch:

```bash
MAX_SEQ_LENGTH=2048 \
DISTRIBUTED_STRATEGY=fsdp_full_shard \
VALIDATION_SAMPLES=2048 \
OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-sft \
sbatch SFT/run_apertus_greek_sft_clariden_multinode.sh
```



Useful overrides:
Clariden `normal` currently has `MaxTime=12:00:00`. If a full SFT run needs more wall-clock time, resume from the latest checkpoint in a follow-up job instead of requesting a longer time on this partition.
For example, if the latest checkpoint is `checkpoint-1000`:
```bash
# resume a partially completed run
RESUME_FROM_CHECKPOINT=/capstor/scratch/cscs/${USER}/apertus-greek-sft/checkpoint-1000 \
sbatch SFT/run_apertus_greek_sft_clariden.sh

# keep a shorter context length for the first stable production pass
MAX_SEQ_LENGTH=1024 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
ATTN_IMPLEMENTATION=eager \
GRADIENT_CHECKPOINTING=0 \
GRADIENT_ACCUMULATION_STEPS=8 \
sbatch SFT/run_apertus_greek_sft_clariden.sh

# try a longer context only after confirming memory headroom on your exact image
MAX_SEQ_LENGTH=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
ATTN_IMPLEMENTATION=eager \
GRADIENT_CHECKPOINTING=0 \
GRADIENT_ACCUMULATION_STEPS=8 \
sbatch SFT/run_apertus_greek_sft_clariden.sh

# force the sharded path explicitly if you want 2048 tokens without relying on auto-selection
MAX_SEQ_LENGTH=2048 \
DISTRIBUTED_STRATEGY=fsdp_full_shard \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
ATTN_IMPLEMENTATION=eager \
GRADIENT_CHECKPOINTING=0 \
GRADIENT_ACCUMULATION_STEPS=8 \
sbatch SFT/run_apertus_greek_sft_clariden.sh

# resume a multi-node run from a saved checkpoint
RESUME_FROM_CHECKPOINT=/capstor/scratch/cscs/${USER}/apertus-greek-sft/checkpoint-1000 \
sbatch SFT/run_apertus_greek_sft_clariden_multinode.sh
```

Minimal local validation:

```bash
./run_uenv.sh python -m py_compile SFT/sft.py
./run_uenv.sh python SFT/sft.py --help
```