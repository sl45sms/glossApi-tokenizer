# commond line steps
## 1 extract words
```
./run_uenv.sh python vocabularyGen/countWords.py \
  --report-every 10000
  ```
  ```
./run_uenv.sh python vocabularyGen/countQuotedWords.py \
  --overwrite
  ```
## 2 select candidates
```
./run_uenv.sh python vocabularyGen/selectTokenizerCandidates.py \
  --min-count 5 \
  --min-base-token-count 3 \
  --max-selected 10000 \
  --overwrite
```
## 3 create tokenizer
```
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --checkpoint-output-dir /iopsstor/scratch/cscs/p-skarvelis/apertus-greek-init \
  --torch-dtype bfloat16 \
  --overwrite
```
at this point you have a new tokenizer for the model, but the model does not know how to use the new tokens yet. The next step is to create a targeted CPT dataset that contains documents with the new tokens, and then do a short CPT training run on that dataset to teach the model how to use the new tokens.
# small fast full training (Curated CPT)
## 4 filter dataset
```
./run_uenv.sh python targetedCPT-DatasetGen/filter.py \
	--limit-per-word 50 \
	--max-output-bytes 1073741824 \
	--workers 16 \
	--overwrite
```
## 5 prepare a CPT training dataset
(ie mix with english data, pack into sequences, write to parquet)
```
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
  --output-dir "$SCRATCH/prepared-datasets/apertus-greek-targeted-packed-2048" \
  --greek-dataset "$SCRATCH/targeted-cpt/curated_greek_cpt.jsonl" \
  --greek-probability 0.9 \
  --english-dataset "epfml/FineWeb-HQ" \
  --english-probability 0.1 \
  --overwrite
```
this creates a new CPT training dataset that is a mix of the filtered Greek documents and some English documents from FineWeb-HQ, packed into sequences of 2048 tokens and written to parquet format for efficient loading during training.
```json
{"output_dir": "/iopsstor/scratch/cscs/p-skarvelis/prepared-datasets/apertus-greek-targeted-packed-2048", "total_sequences": 56021, "total_output_tokens": 114731008, "parquet_shards": 110}
```
## 6 train a CPT model
 use the same training script as for the full CPT, but with the new prepared dataset.
 The steps used below are an example for the ~1GB curated dataset, created above.
```
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=/iopsstor/scratch/cscs/${USER}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-prod-xielu-sdpa-nogc-curated-1GB-2048seq-1000steps
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-targeted-packed-2048
export ATTN_IMPLEMENTATION=sdpa
export GRADIENT_CHECKPOINTING=0
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export TARGET_GLOBAL_BATCH_SIZE=256
unset GRADIENT_ACCUMULATION_STEPS
export BENCHMARK_MODE=0
export SMOKE_TEST=0
export SKIP_WARMUP=0
export FULL_MAX_STEPS=700
export WARMUP_MAX_STEPS=300
export FULL_WARMUP_STEPS=100

sbatch --nodes=4 --time=12:00:00 scripts/run_apertus_greek_cpt_clariden_multinode.sh
```
* train will get arround 4H

* Use the same OUTPUT_DIR on posible rerun.
  If you keep the same output directory, warmup resumes from output_dir/warmup and full resumes from output_dir/full. If warmup is already complete, the code loads that checkpoint and skips warmup cleanly before moving on.

at this point you have a new CPT-trained model that knows how to use the new tokens in some degree but may behave like base model, so need to procced with SFT to further teach the model to use the new tokens in a more natural way.
# SFT training