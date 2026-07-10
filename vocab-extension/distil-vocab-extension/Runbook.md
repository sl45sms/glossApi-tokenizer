
# πρώτο run με MAX_TRAINABLE_TOKENS=1500
```bash
unset BASE_MODEL BASE_TOKENIZER TOKEN_FILE STAGE1_OUTPUT STAGE2_OUTPUT
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-init-run-1"
export OVERWRITE_OUTPUT_DIR=1
export DISTILL_STEPS=500
export DISTILL_LR=5e-6
export MAX_TRAINABLE_TOKENS=1500
export NPROC_PER_NODE=4
export DIST_TIMEOUT_SECONDS=7200
export DISTILL_SYNC_START_STEP=50
export DISTILL_MAX_INVALID_STEPS=20
export DISTILL_LR_DECAY_ON_INVALID=0.5
export UNTIED_OUTPUT_INIT_STRATEGY=zero
export TRAIN_UNTIED_OUTPUT_ROWS=0
export RUN_GREEK_MMLU_EVAL=0
export FAIL_BELOW_BASE=0
sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```

* Νέο default ασφαλείας: για untied μοντέλο τα νέα `lm_head` rows αρχικοποιούνται σε μηδέν και δεν εκπαιδεύονται στο distill.
  Αυτό μειώνει το pre-CPT quality drop επειδή τα νέα tokens δεν μπαίνουν αμέσως να ανταγωνίζονται στο output softmax.

* Μόνο για ablation/experiment: `export UNTIED_OUTPUT_INIT_STRATEGY=mean` και `export TRAIN_UNTIED_OUTPUT_ROWS=1`.

# evaluation
```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
--base-model swiss-ai/Apertus-8B-Instruct-2509 \
--trained-model /capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-init-run-1/ \
--output-json artifacts/reports/greek_mmlu_distill_init_run_1_eval.json \
--no-use-chat-template
```
προηγούμενο αποτέλεσμα evaluate
```
{
  "output_json": "artifacts/reports/greek_mmlu_distill_init_run_1_eval.json",
  "xielu_cuda_available": true,
  "base_report_cache": "artifacts/reports/greek_mmlu_base_eval.json",
  "base_report_cache_hit": true,
  "krikri_report_cache": "artifacts/reports/greek_mmlu_krikri_eval.json",
  "krikri_report_cache_hit": true,
  "maistros_report_cache": "artifacts/reports/greek_mmlu_maistros_eval.json",
  "maistros_report_cache_hit": false,
  "base_accuracy": 0.6997955747955747,
  "krikri_accuracy": 0.7003367003367004,
  "maistros_accuracy": null,
  "trained_accuracy": 0.6772486772486772,
  "accuracy_delta": -0.022546897546897537
}
```
``` νεο
{
  "output_json": "artifacts/reports/greek_mmlu_distill_init_run_1_eval.json",
  "xielu_cuda_available": true,
  "base_report_cache": "artifacts/reports/greek_mmlu_base_eval.json",
  "base_report_cache_hit": true,
  "krikri_report_cache": "artifacts/reports/greek_mmlu_krikri_eval.json",
  "krikri_report_cache_hit": true,
  "maistros_report_cache": "artifacts/reports/greek_mmlu_maistros_eval.json",
  "maistros_report_cache_hit": false,
  "base_accuracy": 0.6997955747955747,
  "krikri_accuracy": 0.7003367003367004,
  "maistros_accuracy": null,
  "trained_accuracy": 0.6779100529100529,
  "accuracy_delta": -0.02188552188552184
}
```




# δεύτερο run 
BASE_MODEL και BASE_TOKENIZER στο output του stage 1 και TOKEN_FILE στο deferred_tokens_next_stage.txt.
```bash
export STAGE1_OUTPUT="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-init-run-1"
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-init-run-2"
export BASE_MODEL="${STAGE1_OUTPUT}"
export BASE_TOKENIZER="${STAGE1_OUTPUT}"
export TOKEN_FILE="${STAGE1_OUTPUT}/deferred_tokens_next_stage.txt"
export MAX_TRAINABLE_TOKENS=1000
export OVERWRITE_OUTPUT_DIR=1
sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```

* Σημείωση: Μην κάνεις stage 2 στο ίδιο OUTPUT_DIR του stage 1 γιατί το distill_checkpoints/distill_rows.pt κρατά το token set του stage 1 και θα χτυπήσει mismatch.

* Απόφαση με τα τωρινά δεδομένα: champion είναι το stage 1 (0.6772), όχι το stage 2 (0.6652). Δεν συνεχίζουμε αυτόματα σε stage 3.

* Προτεινόμενη συνέχεια: προχώρα σε CPT/SFT με βάση το stage 1 checkpoint αντί να προσθέσεις τώρα τα υπόλοιπα deferred tokens.

* Αν θες παρ' όλα αυτά πειραματικό stage 3, κάν' το μόνο με αυστηρό gate και μικρό cap:

```bash
export STAGE2_OUTPUT="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-s2"
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-s3-exp"
export BASE_MODEL="${STAGE2_OUTPUT}"
export BASE_TOKENIZER="${STAGE2_OUTPUT}"
export TOKEN_FILE="${STAGE2_OUTPUT}/deferred_tokens_next_stage.txt"
export MAX_TRAINABLE_TOKENS=300
export DISTILL_STEPS=200
export OVERWRITE_OUTPUT_DIR=1
sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```

* Gate stage 3: αν το trained_accuracy δεν ξεπεράσει το stage 1, απορρίπτεται.

# evaluation stage 2
```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
--base-model swiss-ai/Apertus-8B-Instruct-2509 \
--trained-model /capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified-s2/ \
--output-json artifacts/reports/greek_mmlu_distill_stage2_eval.json \
--no-use-chat-template
```
προηγούμενο αποτέλεσμα evaluate stage 2
```
{
  "output_json": "artifacts/reports/greek_mmlu_distill_stage2_eval.json",
  "xielu_cuda_available": true,
  "base_report_cache": "artifacts/reports/greek_mmlu_base_eval.json",
  "base_report_cache_hit": true,
  "krikri_report_cache": "artifacts/reports/greek_mmlu_krikri_eval.json",
  "krikri_report_cache_hit": true,
  "maistros_report_cache": "artifacts/reports/greek_mmlu_maistros_eval.json",
  "maistros_report_cache_hit": false,
  "base_accuracy": 0.6997955747955747,
  "krikri_accuracy": 0.7003367003367004,
  "maistros_accuracy": null,
  "trained_accuracy": 0.6652236652236653,
  "accuracy_delta": -0.034571909571909476
}
```


# Protocol 3 runs για full 5000 χωρίς blind risk

Στόχος: να κρατήσεις το quality anchor του stage 1 αλλά να δοκιμάσεις και full 5000 tokenizer με recovery CPT,
πριν αποφασίσεις production path.


## Run 1: χτίσε full-5000 tokenizer candidate από stage 1

```bash
export STAGE1_OUTPUT="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified"
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-full5000-candidate"
export BASE_MODEL="${STAGE1_OUTPUT}"
export BASE_TOKENIZER="${STAGE1_OUTPUT}"
unset TOKEN_FILE
export MAX_TRAINABLE_TOKENS=0
export DISTILL_STEPS=150
export DISTILL_LR=2e-6
export DISTILL_REG_WEIGHT=0.2
export DISTILL_CONTEXTS_PER_TOKEN=4
export RUN_GREEK_MMLU_EVAL=0
export FAIL_BELOW_BASE=0
export OVERWRITE_OUTPUT_DIR=1
sbatch scripts/run_apertus_tokenizer_distill_clariden_multinode.sh
```

Σημείωση: το `MAX_TRAINABLE_TOKENS=0` σημαίνει no cap. Με base το stage 1, θα προστεθούν μόνο τα υπόλοιπα deferred.


## Run 2: short CPT baseline από stage 1 (A)

```bash
export MODEL_PATH="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-unified"
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-cpt-stage1-short"
export MAX_SEQ_LENGTH=1024
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=32
export WARMUP_MAX_STEPS=300
export FULL_MAX_STEPS=3000
export FULL_WARMUP_STEPS=200
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=all
export OVERWRITE_OUTPUT_DIR=1
sbatch scripts/run_apertus_greek_cpt_clariden.sh
```


## Run 3: short CPT από full-5000 candidate (B)

```bash
export MODEL_PATH="/capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-full5000-candidate"
export OUTPUT_DIR="/capstor/scratch/cscs/${USER}/apertus-greek-cpt-full5000-short"
export MAX_SEQ_LENGTH=1024
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=32
export WARMUP_MAX_STEPS=300
export FULL_MAX_STEPS=3000
export FULL_WARMUP_STEPS=200
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=all
export OVERWRITE_OUTPUT_DIR=1
sbatch scripts/run_apertus_greek_cpt_clariden.sh
```


## Eval και απόφαση

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model /capstor/scratch/cscs/${USER}/apertus-greek-cpt-stage1-short/final \
  --output-json artifacts/reports/greek_mmlu_cpt_stage1_short_eval.json \
  --no-use-chat-template

./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model /capstor/scratch/cscs/${USER}/apertus-greek-cpt-full5000-short/final \
  --output-json artifacts/reports/greek_mmlu_cpt_full5000_short_eval.json \
  --no-use-chat-template
```

Προαιρετικά economy check:

```bash
./run_uenv.sh python tools/evaluateGreekTokenEconomy.py \
  --base-tokenizer artifacts/tokenizers/apertus-base \
  --extended-tokenizer /capstor/scratch/cscs/${USER}/apertus-greek-tokenizer-distill-full5000-candidate \
  --sample-file greek_samples.txt \
  --report-path artifacts/reports/token_economy_full5000_candidate.json
```

Decision gate:

* Αν το `cpt-full5000-short` είναι καλύτερο ή κοντά (<= 0.005 χειρότερο) από το `cpt-stage1-short`, κράτα full 5000 για production CPT.
* Αν είναι χειρότερο πάνω από 0.005, κράτα stage 1 ως production path και φύλαξε το full 5000 μόνο για πειραματισμό.