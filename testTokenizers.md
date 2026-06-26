# Ελεγχος tokenizers


Έχουμε ήδη 3 tokenizers στο tokenizers:

Tokenizer	Περιγραφή
apertus-base	Το original Apertus (χωρίς ελληνικά tokens)
apertus-greek-v1	Επέκταση με frequency-based επιλογή
apertus-greek-cga-v1	Επέκταση με CGA (Compositional Geometric Alignmen



# Προετοιμασία σύνολο δεδομένων CPT για το targeted greek cpt (~1GB)
```bash
./run_uenv.sh python scripts/prepare_cpt_dataset.py \
    --tokenizer-path artifacts/tokenizers/apertus-greek-v1 \
    --greek-dataset ${SCRATCH}/targeted-cpt/curated_greek_cpt.jsonl \
    --greek-probability 1.0 --english-probability 0 \
    --max-seq-length 2048 \
    --output-dir /iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-v1-targeted-packed-2048 \
    --overwrite
```


# === CPT με apertus-greek-v1 (mean-init) ===
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-tokenizer-v1
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-v1-targeted
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-v1-targeted-packed-2048
export SKIP_WARMUP=1           # Το targeted dataset είναι ~1GB, το warmup θα overfit-άρει
export SMOKE_TEST=0            # Production run
export FULL_MAX_STEPS=50000    # 50k βήματα
export MAX_SEQ_LENGTH=2048
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=64
export SAVE_TOTAL_LIMIT=all    # Κράτα όλα τα checkpoints για να συγκρίνεις
sbatch --time=12:00:00 scripts/run_apertus_greek_cpt_clariden.sh


# === CPT με apertus-greek-cga-v1 (CGA) ===
export CE_ENVIRONMENT=apertus-greek-clariden
export MODEL_PATH=${SCRATCH}/apertus-greek-cga-v1
export OUTPUT_DIR=/capstor/scratch/cscs/${USER}/apertus-greek-cpt-cga-targeted
export PREPARED_TRAIN_DATASET_DIR=/iopsstor/scratch/cscs/${USER}/prepared-datasets/apertus-greek-cga-v1-targeted-packed-2048
export SKIP_WARMUP=1
export SMOKE_TEST=0
export FULL_MAX_STEPS=50000
export MAX_SEQ_LENGTH=2048
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=64
export SAVE_TOTAL_LIMIT=all
sbatch --time=12:00:00 scripts/run_apertus_greek_cpt_clariden.sh


# Αξιολόγηση CPT με apertus-greek-v1
```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
    --base-model swiss-ai/Apertus-8B-Instruct-2509 \
    --trained-model /capstor/scratch/cscs/${USER}/apertus-greek-cpt-v1-targeted/final \
    --skip-krikri \
    --output-json artifacts/reports/greek_mmlu_cpt_v1_targeted_eval.json
```

# Αξιολόγηση CPT με CGA tokenizer
```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
    --base-model swiss-ai/Apertus-8B-Instruct-2509 \
    --trained-model /capstor/scratch/cscs/${USER}/apertus-greek-cpt-cga-targeted/final \
    --skip-krikri \
    --output-json artifacts/reports/greek_mmlu_cpt_cga_targeted_eval.json
```

# Σύγκριση αποτελεσμάτων αξιολόγησης
```bash
./run_uenv.sh python evaluation/plot_greek_mmlu_report.py \
    --reports \
        artifacts/reports/greek_mmlu_base_eval.json \
        artifacts/reports/greek_mmlu_cpt_v1_targeted_eval.json \
        artifacts/reports/greek_mmlu_cpt_cga_targeted_eval.json
```
* Ή απλά διάβασε τα JSON — το key που σε ενδιαφέρει είναι το accuracy__All:
```bash
for f in artifacts/reports/greek_mmlu_*targeted*.json; do
    echo "$f: $(python -c "import json; d=json.load(open('$f')); print(d.get('results',{}).get('All',{}).get('accuracy','N/A'))")"
done
```