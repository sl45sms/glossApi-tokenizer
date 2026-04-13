# Tokenizer Visualizer

The visualizer takes one sentence or a set of sentences and compares the tokenization results from three tokenizers, reporting differences in token counts, average characters per token, and example tokenizations.

Implementation details:

- It runs as a local web service on `http://localhost:7860/`.
- It provides a multiline text input where each non-empty line is treated as a separate sample.
- It compares the base Apertus tokenizer, the extended Apertus Greek tokenizer, and the Krikri reference tokenizer.
- It calls `scripts/compare_tokenizers.py` under the hood and renders the JSON report in a more readable format.

## Run

Make sure `.venv-uenv` exists and includes `gradio`:

```bash
./run_uenv.sh python -m pip install gradio
```

Then start the visualizer:

```bash
./run_visualizer.sh
```

Open:

```text
http://localhost:7860/
```

## Default tokenizer paths

- Base tokenizer: `artifacts/tokenizers/apertus-base`
- Extended tokenizer: `artifacts/tokenizers/apertus-greek-v1`
- Reference tokenizer: `artifacts/tokenizers/krikri-base`

You can override all three paths from the UI.
