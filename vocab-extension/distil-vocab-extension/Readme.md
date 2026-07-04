cd /users/${USER}/glossApi-Tokenizer/

rm -rf "${SCRATCH}/apertus-greek-tokenizer-distill" && sbatch ./vocab-extension/distil-vocab-extension/run_distill_v8.sh

to resume just sbatch ./vocab-extension/distil-vocab-extension/run_distill_v8.sh


./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model "${SCRATCH}/apertus-greek-tokenizer-distill-freeweb-v18" \
  --output-json artifacts/reports/greek_mmlu_distill_freeweb_v18_eval.json

rm -rf "${SCRATCH}/apertus-greek-tokenizer-distill-freeweb-v18" && sbatch ./vocab-extension/distil-vocab-extension/run_distill_freeweb.sh