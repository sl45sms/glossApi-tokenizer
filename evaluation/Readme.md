here will run GreekMMLU to evaluate the final CPT checkpoint against the original base model. The evaluation script is `evaluation/evaluate_greek_mmlu.py`, and it produces a JSON report with category-wise and overall accuracy.

# run

Use the repo's `uenv` wrapper so the evaluator runs with the same Python environment as the rest of the project:

```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
	--base-model swiss-ai/Apertus-8B-Instruct-2509 \
	--trained-model /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final \
	--output-json artifacts/reports/greek_mmlu_eval.json
```

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

The output report includes overall accuracy, group-wise accuracy, subject-wise accuracy, level-wise accuracy, and trained-minus-base accuracy deltas.

# useful links
- GreekMMLU dataset card: https://huggingface.co/datasets/dascim/GreekMMLU
- GreekMMLU paper: https://arxiv.org/abs/2212.14096
- GreekMMLU GitHub: https://github.com/dascim/GreekMMLU
- GreekMMLU leaderboard: https://greekmmlu.dascim.com/leaderboard
