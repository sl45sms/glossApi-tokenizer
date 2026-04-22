here will run GreekMMLU to evaluate the final CPT checkpoint against the original base model. The evaluation script is `evaluation/evaluate_greek_mmlu.py`, and it produces a JSON report with category-wise and overall accuracy.

# run

Use the repo's `uenv` wrapper so the evaluator runs with the same Python environment as the rest of the project:

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
	--base-model swiss-ai/Apertus-8B-Instruct-2509 \
	--trained-model /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final \
	--output-json artifacts/reports/greek_mmlu_eval.json
```

By default, the script caches the base-model evaluation at `artifacts/reports/greek_mmlu_base_eval.json` and reuses it on later runs, so repeated evaluations only score the current trained checkpoint.

If you need to recompute the base model, add `--refresh-base-report-cache`.

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

The output report includes overall accuracy, group-wise accuracy, subject-wise accuracy, level-wise accuracy, and trained-minus-base accuracy deltas.

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
