# Agents / Runbook (CSCS Alps Clariden)

This repository is intended to run on CSCS Alps, targeting Clariden, with the goal of adapting `swiss-ai/Apertus-8B-Instruct-2509` for better Greek coverage and then continuing pretraining with the updated vocabulary.

The practical workflow in this repo now has three stages:

1. Extract the tokenizer from the target model.
2. Extend the tokenizer and build a resized model checkpoint aligned to it.
3. Run CPT on Clariden with the aligned checkpoint, using `CPT/cpt.py` as the training entry point.

Important constraint:

- `artifacts/tokenizers/apertus-greek-v1` is the tokenizer artifact.
- `CPT/cpt.py` must load a model checkpoint whose embeddings were already resized to that tokenizer. A tokenizer directory alone is not enough.

There is also a local inspection tool in this repository:

- a tokenizer visualizer web UI served on `http://localhost:7860/` for side-by-side comparison of tokenization results

## 1. Platform assumptions

- Cluster: CSCS Alps, Clariden.
- Scheduler: Slurm.
- Runtime: CSCS Container Engine with an EDF file.
- Build-time tooling: `uenv` for Python and local preprocessing on Alps.
- Target model: `swiss-ai/Apertus-8B-Instruct-2509`.
- Greek reference tokenizer: `ilsp/Llama-Krikri-8B-Instruct`.
- Extended tokenizer artifact: `artifacts/tokenizers/apertus-greek-v1`.
- CPT entry point: `CPT/cpt.py`.
- Greek corpus source for CPT: `epfml/FineWeb2-HQ`, config `ell_Grek`, plus any additional curated Greek text.
- English anchor corpus for CPT: `epfml/FineWeb-HQ`.

Recommended operating split:

- Use `uenv` for Python tooling, dataset inspection, token mining, and tokenizer preparation.
- Use a container launched through Clariden's Container Engine for model initialization and GPU-side CPT jobs.
- Do not mix `uenv` and the runtime container inside the same training job.

References:

- Clariden: https://docs.cscs.ch/clusters/clariden/
- Container Engine / EDF: https://docs.cscs.ch/software/container-engine/
- CE quick start: https://docs.cscs.ch/software/container-engine/#step-2-launch-a-program
- uenv quick start: https://docs.cscs.ch/software/uenv/
- Build containers on Alps: https://docs.cscs.ch/build-install/containers/
- HuggingFace tokenizers: https://huggingface.co/docs/tokenizers/index
- HuggingFace transformers: https://huggingface.co/docs/transformers/index

## 2. High-level workflow

The intended workflow is:

1. Reuse or regenerate the extended tokenizer under `artifacts/tokenizers/apertus-greek-v1`.
2. Create a model checkpoint whose embeddings are resized and initialized for that tokenizer.
3. Point `CPT/cpt.py` at that aligned checkpoint through `model_path`.
4. Launch a single-node Clariden CPT run with 4 GPUs using `torchrun`.
5. Validate checkpoint save/reload and only then scale the run length or cluster shape.

Important constraints:

- Extending a tokenizer changes the embedding matrix shape. The model must be loaded and resized with the updated tokenizer before training or inference.
- `CPT/cpt.py` only gets the intended 4-GPU global batch if it is launched with a distributed launcher such as `torchrun --nproc_per_node=4`. A plain `python CPT/cpt.py` run will stay single-process.

## 3. Clariden environment strategy

Clariden nodes are GH200 systems, so the runtime environment must be compatible with `aarch64`.

That means:

- Build or use an `aarch64` container image.
- Use Container Engine for GPU execution.
- Keep HuggingFace caches in `${SCRATCH}`.

If you try to run an `x86_64` container on Clariden, container startup may fail before Python even starts.

### 3.1 Minimal sanity check

If you already have a working EDF test file, confirm that CE is alive before doing any tokenizer or training work.

Example:

```bash
srun -Aa0140 --environment=test echo "Its Alive"
```

### 3.2 Build-time Python via uenv

Use `uenv` for all CPU-side preparation steps.

Example:

```bash
uenv image pull prgenv-gnu/24.11:v1
uenv run prgenv-gnu/24.11:v1 --view=default -- bash -lc 'python3 --version; which python3'
```

Create a local environment for tokenizer work:

```bash
uenv run prgenv-gnu/24.11:v1 --view=default -- bash -lc '
python3 -m venv .venv-uenv
source .venv-uenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
'
```

This stage is for:

- dataset inspection
- vocabulary analysis
- tokenizer extraction and extension
- dry-run preprocessing
- running the local tokenizer visualizer under `uenv`

### 3.3 Runtime container for Clariden

For training jobs, use a Clariden-compatible container image through CE.

The image should include at least:

- Python 3.10+
- PyTorch with CUDA support appropriate for Alps/Clariden
- `transformers`
- `datasets`
- `tokenizers`
- `sentencepiece`
- `accelerate`
- `safetensors`
- optionally `deepspeed` or another distributed training stack if needed later

### 3.4 Example EDF for tokenizer/model-init/CPT jobs

Create an EDF file such as `~/.edf/apertus-greek-clariden.toml`:

```toml
image = "${SCRATCH}/images/apertus-greek-aarch64.sqsh"
mounts = [
	"${SCRATCH}:${SCRATCH}",
	"/capstor/store/cscs/swissai/a0140/p-skarvelis:/capstor/store/cscs/swissai/a0140/p-skarvelis",
]
workdir = "${SCRATCH}"

[env]
HF_HOME = "${SCRATCH}/hf"
TRANSFORMERS_CACHE = "${SCRATCH}/hf"
HF_DATASETS_CACHE = "${SCRATCH}/hf_datasets"
HF_TOKEN = "${HF_TOKEN}"
```

Then test it:

```bash
srun -Aa0140 --environment=apertus-greek-clariden python -c 'import torch, transformers; print(torch.__version__); print(torch.cuda.device_count())'
```

If you keep `model_path` and `output_dir` under `${SCRATCH}`, you can drop the `/capstor/...` mount. Keep the mount only when the training script uses that storage path directly.

### 3.5 Known-good Clariden launch settings

The related Clariden project under `/users/p-skarvelis/GSDG` uses the following single-node CE settings, and they are a good starting point here as well:

```bash
export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get
```

Those settings are useful when the host CXI/libfabric hook interferes with the container runtime on Clariden.

If you later scale CPT beyond one node, also pin the network stack explicitly:

```bash
export NCCL_SOCKET_IFNAME=nmn0
export GLOO_SOCKET_IFNAME=nmn0
export NCCL_CROSS_NIC=1
export FI_PROVIDER=cxi
```

## 4. Stage 1: Extract the target tokenizer

The first required artifact is the tokenizer from `swiss-ai/Apertus-8B-Instruct-2509`.

The goal of this stage is to save a local copy and inspect:

- tokenizer class
- vocabulary size
- special tokens
- whether it is BPE, SentencePiece, or another backend

Minimal extraction example:

```python
from transformers import AutoTokenizer

model_id = "swiss-ai/Apertus-8B-Instruct-2509"
out_dir = "artifacts/tokenizers/apertus-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(out_dir)

print(type(tokenizer))
print("vocab size:", len(tokenizer))
print("special tokens:", tokenizer.special_tokens_map)
```

Expected output of this stage:

- a saved tokenizer directory under `artifacts/tokenizers/apertus-base`
- a short report recording vocab size and tokenizer type
- optionally a local visual inspection session through `visualizer/app.py`

Before moving to stage 2, verify that Greek sample text is currently segmented inefficiently enough to justify extension.

Use checks such as:

- average tokens per Greek sentence
- fragmentation of common Greek morphemes
- fragmentation of domain-specific GlossAPI terms

## 5. Stage 2: Extend the tokenizer for Greek

This is the critical stage. The objective is to improve Greek tokenization while keeping changes controlled.

There are two candidate sources for new vocabulary:

1. The tokenizer of `ilsp/Llama-Krikri-8B-Instruct`.
2. Tokens mined directly from Greek corpora, especially GlossAPI data.

### 5.1 Recommended approach

Do not blindly merge another tokenizer wholesale.

Instead:

1. Extract candidate tokens from the Greek-oriented tokenizer.
2. Extract candidate substrings or terms from the Greek corpus.
3. Remove anything already represented well by the Apertus tokenizer.
4. Keep only additions that materially reduce token count or capture recurring domain terms.

This is safer than replacing the tokenizer or doing an uncontrolled full merge.

### 5.2 Candidate sources

Candidate additions should include:

- frequent whole Greek words
- high-frequency subwords not represented efficiently in Apertus
- punctuation-adjacent Greek patterns if they occur repeatedly
- educational or GlossAPI-specific terms
- normalized forms only if normalization is part of preprocessing

Avoid adding:

- extremely rare whole words
- noisy OCR artifacts
- duplicate variants caused only by inconsistent whitespace
- tokens that are already split efficiently enough

### 5.3 Practical filtering rules

Keep a candidate token only if at least one of the following is true:

- it appears frequently in the Greek corpus
- it reduces token count relative to the base tokenizer
- it is a stable domain term that matters for downstream tasks

Additions should be incremental. Start with a conservative vocabulary expansion, not a massive one.

### 5.4 Example extension flow

At a high level:

1. Load the base Apertus tokenizer.
2. Load the Krikri tokenizer.
3. Read Greek corpus text.
4. Build a candidate set.
5. Filter candidates by utility.
6. Call `tokenizer.add_tokens(...)` on the surviving set.
7. Save the extended tokenizer.

Minimal example:

```python
from transformers import AutoTokenizer

base_id = "swiss-ai/Apertus-8B-Instruct-2509"
greek_id = "ilsp/Llama-Krikri-8B-Instruct"

base_tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
greek_tok = AutoTokenizer.from_pretrained(greek_id, trust_remote_code=True)

base_vocab = set(base_tok.get_vocab().keys())
greek_vocab = set(greek_tok.get_vocab().keys())

candidates = sorted(greek_vocab - base_vocab)

# Replace this with corpus-driven filtering before adding tokens.
selected = candidates[:2000]

num_added = base_tok.add_tokens(selected)
print("added:", num_added)

base_tok.save_pretrained("artifacts/tokenizers/apertus-greek-v1")
```

The placeholder slice above is only a workflow example. In actual use, the selected set should come from measured filtering, not an arbitrary cutoff.

### 5.5 GlossAPI-driven token mining

GlossAPI should be used to find additions that the base tokenizer handles poorly.

Recommended process:

1. Stream or load the target GlossAPI datasets.
2. Extract the best text-bearing fields.
3. Normalize consistently.
4. Count frequent words and substrings.
5. Compare segmentation under the base tokenizer.
6. Promote only the items that improve compression or preserve important Greek forms.

Useful measurements:

- token count before and after extension
- average characters per token on Greek text
- coverage of common Greek educational terms
- rate of unknown or fragmented patterns if present

Expected output of stage 2:

- `artifacts/tokenizers/apertus-greek-v1`
- a candidate list with frequency counts
- a short evaluation note comparing base and extended tokenizer behavior

### 5.6 Build the aligned model checkpoint for CPT

For CPT, the tokenizer directory alone is not sufficient. `CPT/cpt.py` loads both tokenizer and model from `model_path`, so that path must already contain a checkpoint whose embeddings were resized to `artifacts/tokenizers/apertus-greek-v1`.

Use `scripts/extend_apertus_tokenizer.py` with `--base-model` to create that aligned checkpoint:

```bash
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
	--base-tokenizer artifacts/tokenizers/apertus-base \
	--token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
	--base-model swiss-ai/Apertus-8B-Instruct-2509 \
	--checkpoint-output-dir "${SCRATCH}/apertus-greek-init" \
	--torch-dtype bfloat16 \
	--overwrite
```

Notes:

- When `--base-model` is enabled, the script loads the full base LM in order to resize and initialize the new embeddings. Run this step on a machine or job with enough memory.
- If you already have an initialized checkpoint at a persistent path such as `/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-init`, reuse it and point `CPT/cpt.py` there.
- The resulting checkpoint directory is the value that should go into `model_path` in `CPT/cpt.py`.

## 6. Stage 3: Continue training with the new tokenizer

This repo's current stage-3 path is `CPT/cpt.py`. It assumes that the tokenizer extension work is already done and that `model_path` points to a checkpoint whose embeddings were resized to the extended tokenizer.

### 6.1 What `CPT/cpt.py` does

The current script:

- loads the aligned checkpoint from `model_path`
- loads the tokenizer from the same directory
- streams a 90% Greek / 10% English mixture
- uses `bfloat16`, `flash_attention_2`, and gradient checkpointing for GH200-class GPUs
- runs two phases:
	1. embedding-only warm-up for 2000 steps at `1e-4`
	2. full CPT for 50000 steps at `2e-5` with `warmup_steps=1000`

Because the datasets are streamed, the schedule is step-driven rather than epoch-driven.

### 6.2 Required preconditions

Before launching the script:

- `model_path` must point to the aligned checkpoint from stage 2, not just `artifacts/tokenizers/apertus-greek-v1`.
- `output_dir` must point to a persistent mounted path such as `${SCRATCH}/apertus-greek-cpt` or `/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt`.
- The Clariden container must have working support for `torch`, `transformers`, `datasets`, and the `flash_attention_2` path used by the script.
- If `model_path` or `output_dir` stay under `/capstor/...`, that path must be mounted in the EDF.

### 6.3 Launch shape on Clariden

Start with a single Clariden node:

- 1 node
- 4 GPUs
- 1 Slurm task that launches 4 local training workers with `torchrun`

Important constraint:

- The batch-size comment inside `CPT/cpt.py` assumes 4 GPUs. That is only true if the job is launched with `torchrun --nproc_per_node=4` or an equivalent distributed launcher.
- If you run `python CPT/cpt.py` directly, `Trainer` will stay single-process and the effective global batch will be smaller than intended.

Treat the current script as a single-node Clariden starting point. Only move to multi-node after the one-node path is stable; multi-node training will require a proper distributed launcher and the explicit Clariden network settings from section 3.5.

### 6.4 Example single-node Clariden launcher

A minimal Slurm pattern, adapted from the Clariden usage in the related `GSDG` project, is:

```bash
#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=apertus-greek-cpt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail

CE_ENVIRONMENT="${CE_ENVIRONMENT:-apertus-greek-clariden}"
STAGE_ROOT="${SCRATCH}/glossapi-tokenizer_${SLURM_JOB_ID}"

export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get

rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"
tar -C /users/p-skarvelis/glossApi-Tokenizer -cz CPT Agents.md | tar -xz -C "${STAGE_ROOT}"

srun --environment="${CE_ENVIRONMENT}" --ntasks=1 bash -lc '
	set -euo pipefail
	export HF_HOME="${SCRATCH}/hf"
	export TRANSFORMERS_CACHE="${SCRATCH}/hf"
	export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
	export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
	cd "'"${STAGE_ROOT}"'"
	python -m torch.distributed.run --standalone --nproc_per_node=4 CPT/cpt.py
'
```

This launcher deliberately uses a single Slurm task and lets `torchrun` spawn the 4 local worker processes inside the Clariden container.

### 6.5 First-run strategy

Do not start with the full 50000-step job.

First validate:

1. the aligned checkpoint loads inside the CE environment
2. all 4 GPUs are visible inside the job
3. a short run writes checkpoints successfully
4. the saved checkpoint can be reloaded with the tokenizer

After that, scale the step counts back up to the intended long run.

## 7. Suggested directory layout

To keep the work reproducible, use a simple artifact structure:

```text
artifacts/
	tokenizers/
		apertus-base/
		apertus-greek-v1/
	vocab_candidates/
		krikri_candidates.txt
		glossapi_candidates.tsv
		selected_tokens_v1.txt
	reports/
		tokenizer_baseline.md
		tokenizer_eval_v1.md
	checkpoints/
		apertus-greek-init/
		apertus-greek-cpt-smoke/

CPT/
	cpt.py

external persistent storage/
	${SCRATCH}/apertus-greek-init/
	${SCRATCH}/apertus-greek-cpt/
	or /capstor/store/cscs/swissai/a0140/p-skarvelis/...
```

## 8. Validation checklist

Before accepting the new tokenizer and model, verify:

- the tokenizer can be loaded and saved cleanly
- the aligned checkpoint can be loaded with the new tokenizer
- `len(tokenizer)` matches the checkpoint embedding matrix size
- the training paths used in `CPT/cpt.py` are visible inside the Clariden container
- the job is launched with `torchrun` and sees all 4 Clariden GPUs
- Greek sample text uses fewer or better tokens than before
- no special tokens were broken or removed
- training data preprocessing is consistent across runs
- the CPT job writes intermediate checkpoints successfully
- a saved checkpoint can be reloaded for inference

Useful evaluation comparisons:

- base tokenizer vs extended tokenizer on the same Greek corpus
- base initialized model vs continued-pretrained model on Greek prompts
- token count reduction on representative GlossAPI samples
- manual inspection in the visualizer for representative Greek sentences and domain terms

## 9. Recommended execution order

The practical order for this repository should be:

1. Set up `uenv` on Alps for preprocessing.
2. Create or validate the Clariden EDF and container.
3. Reuse or regenerate the base and extended tokenizer artifacts.
4. Create or reuse the aligned checkpoint with `scripts/extend_apertus_tokenizer.py --base-model`.
5. Set `model_path` and `output_dir` in `CPT/cpt.py` to mounted persistent storage.
6. Run a short single-node Clariden smoke test with `torchrun`.
7. Launch the longer 4-GPU CPT job.
8. Reload the saved checkpoint and evaluate Greek behavior.
9. Only then consider multi-node scaling.

## 10. Immediate next milestone

The first concrete deliverable for the current repo state should be:

1. a persistent `apertus-greek-init` checkpoint aligned with `apertus-greek-v1`
2. a successful single-node Clariden CPT smoke run from `CPT/cpt.py`
3. a saved checkpoint that reloads cleanly with the tokenizer
4. a production-length Clariden CPT run after the smoke path is stable
