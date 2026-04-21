# CPT Model UI

This folder contains a single-model Gradio console for probing the checkpoint produced by `CPT/cpt.py`.

The UI is intended for post-training inspection only. It does not compare multiple models side by side. Instead, it loads one tokenizer-aligned checkpoint and gives you two ways to test it:

- ready-made evaluation probes for Greek fluency, formatting, JSON output, English retention, bilingual switching, and uncertainty handling
- a freeform prompt box for any manual check you want to run

The session log is only for review inside the browser. Previous runs are shown there, but they are not sent back into the model context.

## Run

Use the repo helper:

```bash
./run_model_ui.sh --model-path /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final
```

Or call the app directly through the existing `uenv` wrapper:

```bash
./run_uenv.sh python modelUI/app.py \
	--model-path /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final \
	--device cuda:0 \
	--dtype bfloat16
```

By default the UI binds to `127.0.0.1:7861`.

## Useful arguments

- `--model-path`: local checkpoint directory to load, ideally the `final` directory written by CPT
- `--device`: inference device such as `cuda:0` or `cpu`
- `--dtype`: one of `bfloat16`, `float16`, or `float32`
- `--attn-implementation`: attention backend, default `sdpa`
- `--host` and `--port`: web UI bind address

The first request can take a while because the model weights are loaded lazily on demand.
