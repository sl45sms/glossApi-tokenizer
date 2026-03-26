# Agents / Runbook (CSCS Alps Clariden)

This repository is intended to run on CSCS Alps, targeting Clariden, with the goal of adapting `swiss-ai/Apertus-8B-Instruct-2509` for better Greek coverage.

The work is split into three stages:

1. Extract the tokenizer from the target model.
2. Extend the tokenizer with Greek-specific coverage.
3. Continue training the model on Greek text with the updated tokenizer.

The practical target is not only to add arbitrary tokens, but to improve tokenization efficiency and Greek text coverage without destabilizing the base model.

There is also a local inspection tool in this repository:

- a tokenizer visualizer web UI served on `http://localhost:7860/` for side-by-side comparison of tokenization results

## 1. Platform assumptions

- Cluster: CSCS Alps, Clariden.
- Scheduler: Slurm.
- Runtime: CSCS Container Engine with an EDF file.
- Build-time tooling: `uenv` for Python and local preprocessing on Alps.
- Target model: `swiss-ai/Apertus-8B-Instruct-2509`.
- Greek reference tokenizer: `ilsp/Llama-Krikri-8B-Instruct`.
- Greek corpus source: GlossAPI datasets and any additional curated Greek text.

Recommended operating split:

- Use `uenv` for Python tooling, dataset inspection, token mining, and tokenizer preparation.
- Use a container launched through Clariden's Container Engine for GPU jobs.
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

1. Pull the tokenizer for `swiss-ai/Apertus-8B-Instruct-2509` and inspect its tokenizer type.
2. Collect candidate Greek additions from:
	 - the tokenizer vocabulary of `ilsp/Llama-Krikri-8B-Instruct`
	 - high-frequency substrings, words, and domain terms extracted from GlossAPI corpora
3. Filter candidates so that only useful new entries are added.
4. Resize model embeddings to match the new tokenizer.
5. Continue pretraining on Greek corpora.
6. Evaluate tokenization efficiency and downstream Greek behavior.

Important constraint:

- Extending a tokenizer changes the embedding matrix shape. The model must be loaded and resized with the updated tokenizer before training or inference.

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

### 3.4 Example EDF for tokenizer/training jobs

Create an EDF file such as `~/.edf/apertus-greek.toml`:

```toml
image = "${SCRATCH}/images/apertus-greek-aarch64.sqsh"
mounts = ["${SCRATCH}:${SCRATCH}"]
workdir = "${SCRATCH}"

[env]
HF_HOME = "${SCRATCH}/hf"
TRANSFORMERS_CACHE = "${SCRATCH}/hf"
HF_DATASETS_CACHE = "${SCRATCH}/hf_datasets"
HF_TOKEN = "${HF_TOKEN}"
```

Then test it:

```bash
srun -Aa0140 --environment=apertus-greek python -c 'import torch, transformers; print(torch.__version__)'
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

## 6. Stage 3: Continue training with the new tokenizer

Once the tokenizer is extended, the model must be aligned to it before training.

At minimum, the training pipeline needs to:

1. load the base model
2. load the updated tokenizer
3. resize token embeddings
4. continue pretraining on Greek corpora
5. save the adapted checkpoint

Core step:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "swiss-ai/Apertus-8B-Instruct-2509"
tok_dir = "artifacts/tokenizers/apertus-greek-v1"

tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer))
```

Important note:

- Newly added embeddings start untrained. They only become useful after continued training.

### 6.1 Training objective

The simplest first objective is continued pretraining on Greek plain text or instruction-like Greek corpora.

Possible progression:

1. Continued pretraining on broad Greek text.
2. Optional instruction tuning on Greek Q/A data.
3. Optional domain adaptation on GlossAPI-specific educational material.

### 6.2 Training strategy on Clariden

For the first pass, keep the plan simple:

- start with a small-scale continued pretraining run
- validate the pipeline on a subset of Greek data
- confirm that checkpoint save and reload works with the new tokenizer
- only then scale to a longer run

Depending on the final model memory profile, use one of:

- single-node fine-tuning if the model fits comfortably
- distributed training across multiple GPUs if needed

The exact launcher can be decided later, but the environment should remain:

- Clariden-compatible container
- Slurm allocation
- caches under `${SCRATCH}`

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
		apertus-greek-continued-pretrain/
```

## 8. Validation checklist

Before accepting the new tokenizer and model, verify:

- the tokenizer can be loaded and saved cleanly
- the model can be loaded with the new tokenizer
- `resize_token_embeddings` runs without mismatch errors
- Greek sample text uses fewer or better tokens than before
- no special tokens were broken or removed
- training data preprocessing is consistent across runs
- a saved checkpoint can be reloaded for inference

Useful evaluation comparisons:

- base tokenizer vs extended tokenizer on the same Greek corpus
- base model vs continued-pretrained model on Greek prompts
- token count reduction on representative GlossAPI samples
- manual inspection in the visualizer for representative Greek sentences and domain terms

## 9. Recommended execution order

The practical order for this repository should be:

1. Set up `uenv` on Alps for preprocessing.
2. Create or validate the Clariden EDF and container.
3. Extract and save the Apertus tokenizer.
4. Compare it against the Krikri tokenizer.
5. Use the local visualizer to inspect representative Greek tokenization cases.
6. Mine candidate tokens from GlossAPI Greek text.
7. Build a conservative extended tokenizer.
8. Evaluate tokenization efficiency on Greek samples.
9. Resize Apertus embeddings and run a small continued-pretraining job.
10. Scale up only after the small run is stable.

## 10. Immediate next milestone

The first concrete deliverable should be:

1. a saved local copy of the Apertus tokenizer
2. a script or notebook that compares Greek segmentation between Apertus and Krikri
3. a local visualizer for interactive tokenization inspection
4. a first filtered candidate token list derived from GlossAPI text

After that, the repo can move to the first `apertus-greek-v1` tokenizer release.
