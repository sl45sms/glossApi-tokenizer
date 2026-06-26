cd /users/${USER}/glossApi-Tokenizer/

rm -rf "${SCRATCH}/apertus-greek-tokenizer-distill" && sbatch ./vocab-extension/distil-vocab-extension/run_distill_v8.sh

to resume just sbatch ./vocab-extension/distil-vocab-extension/run_distill_v8.sh