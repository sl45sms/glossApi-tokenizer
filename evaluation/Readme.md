here will run GreekMMLU to evaluate the final CPT checkpoint against the original Apertus base model and the Krikri reference model. The evaluation script is `evaluation/evaluate_greek_mmlu.py`, and it produces a JSON report with category-wise and overall accuracy.

# run

Use the repo's `uenv` wrapper so the evaluator runs with the same Python environment as the rest of the project:

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
	--base-model swiss-ai/Apertus-8B-Instruct-2509 \
	--krikri-model ilsp/Llama-Krikri-8B-Instruct \
	--trained-model /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final \
	--output-json artifacts/reports/greek_mmlu_eval.json
```

By default, the script caches the Apertus base-model evaluation at `artifacts/reports/greek_mmlu_base_eval.json` and the Krikri evaluation at `artifacts/reports/greek_mmlu_krikri_eval.json`, then reuses both on later runs so repeated evaluations only score the current trained checkpoint.

If you need to recompute the base model, add `--refresh-base-report-cache`. If you need to recompute Krikri, add `--refresh-krikri-report-cache`.

If the model requires remote code during loading, add `--trust-remote-code`.

The script defaults to:

- dataset `dascim/GreekMMLU`
- config `All`
- evaluation split `test`
- few-shot source split `dev`
- `5` few-shot examples per question

For a quick smoke run without loading the 8B checkpoints:

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
	--base-model sshleifer/tiny-gpt2 \
	--trained-model sshleifer/tiny-gpt2 \
	--no-evaluate-krikri \
	--device cpu \
	--num-few-shot 0 \
	--limit 1 \
	--progress-interval 0 \
	--output-json /tmp/greek_mmlu_smoke.json
```

Useful flags:

- `--device cuda:0` to force a specific GPU
- `--limit 100` to evaluate only part of the benchmark
- `--subject Agriculture` to restrict evaluation to one or more subjects
- `--num-few-shot 0` to run zero-shot instead of the default 5-shot setup
- `--save-predictions` to include per-example predictions in the output JSON
- `--refresh-base-report-cache` to force a fresh base-model evaluation instead of reusing the cached one
- `--refresh-krikri-report-cache` to force a fresh Krikri evaluation instead of reusing the cached one
- `--no-evaluate-krikri` to skip the Krikri reference lane entirely

The output report includes overall accuracy, group-wise accuracy, subject-wise accuracy, level-wise accuracy, the trained-minus-base accuracy deltas, and when Krikri is enabled the Krikri-vs-base and trained-vs-Krikri comparisons.

# plot

After generating the JSON report, convert it into PNG bar charts with:

```bash
./run_uenv.sh python evaluation/plot_greek_mmlu_report.py \
	artifacts/reports/greek_mmlu_eval.json \
	--output-dir artifacts/reports/greek_mmlu_eval_plots
```

This writes the following images under `artifacts/reports/greek_mmlu_eval_plots`:

- `overall_accuracy.png`
- `group_accuracy.png`
- `level_accuracy.png`
- `subject_accuracy_comparison.png`
- `subject_accuracy_delta.png`

When the JSON report contains a Krikri lane, the overall, group, level, and subject comparison charts include it automatically; the delta chart remains trained-minus-base.

Useful plotting flags:

- `--top-subjects 10` to keep only the top 10 subjects in the subject charts
- `--subject-order trained` to sort subjects by trained-model accuracy
- `--subject-order delta` to sort subjects by trained-minus-base improvement
- `--dpi 240` to export higher-resolution PNGs
- `--output-dir /tmp/greek_mmlu_plots` to write the images somewhere else

# useful links
- GreekMMLU dataset card: https://huggingface.co/datasets/dascim/GreekMMLU
- GreekMMLU paper: https://arxiv.org/abs/2212.14096
- GreekMMLU GitHub: https://github.com/dascim/GreekMMLU
- GreekMMLU leaderboard: https://greekmmlu.dascim.com/leaderboard
