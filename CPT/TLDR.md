## System preperation

### build image
```
sbatch scripts/build_apertus_greek_clariden_image.sh
```
#### the image build now installs xIELU by default with `--no-build-isolation`; check the build log for `xielu_import=OK`

#### disable xIELU only if you need to troubleshoot the image build
```
export INSTALL_XIELU=0
sbatch scripts/build_apertus_greek_clariden_image.sh
```

#### wait for the job to complete, then check the image is in place and has a reasonable size
```
ls -lh "${SCRATCH}/images/apertus-greek-aarch64.sqsh"
```

### prepare edf
```
cp edf/apertus-greek-clariden.toml ~/.edf/apertus-greek-clariden.toml
```
* create the folder .edf in your home directory if it doesn't exist


## benchmarking

### create token for benchmarking
```
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
  --tokenizer-path artifacts/tokenizers/apertus-greek-v1 \
  --output-dir /iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-bench-2048 \
  --max-seq-length 2048 \
  --max-output-sequences 4096 \
  --overwrite
```
### single node benchmark
```
RUN_TAG=$(date +%Y%m%d-%H%M%S)
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-bench-${RUN_TAG}
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-bench-2048
export ATTN_IMPLEMENTATION=sdpa
export BENCHMARK_MODE=1
export SMOKE_TEST=0
export SKIP_WARMUP=1
export FULL_MAX_STEPS=50
JOBID=$(sbatch --parsable --time=01:30:00 scripts/run_apertus_greek_cpt_clariden.sh)
echo "${JOBID}"
tail -f slurm-${JOBID}.out
```
#### get the benchmark results after the job completes
```
python3 -m json.tool "${OUTPUT_DIR}/full/phase_metrics.json"
```

### 4 nodes benchmark
```
RUN_TAG=$(date +%Y%m%d-%H%M%S)
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-init
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-bench-multinode-${RUN_TAG}
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-packed-bench-2048
export ATTN_IMPLEMENTATION=sdpa
export BENCHMARK_MODE=1
export SMOKE_TEST=0
export SKIP_WARMUP=1
export FULL_MAX_STEPS=50
JOBID=$(sbatch --parsable --nodes=4 --time=01:30:00 scripts/run_apertus_greek_cpt_clariden_multinode.sh)
echo "${JOBID}"
tail -f slurm-${JOBID}.out
```
#### get the benchmark results after the job completes
```
python3 -m json.tool "${OUTPUT_DIR}/full/phase_metrics.json"
```

#### compare the results
The numbers to compare between the 4-GPU and 16-GPU runs are these fields inside phase_metrics.json:

* cluster_tokens_per_second
* tokens_per_second_per_gpu
* steps_completed
* effective_global_batch_size
* effective_max_seq_length

