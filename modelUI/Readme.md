# CPT Model UI

This folder contains a Gradio console for probing the checkpoint produced by `CPT/cpt.py` against the original Apertus base model.

The UI is intended for post-training inspection. It sends the same prompt to two lanes:

- the original base model, by default `swiss-ai/Apertus-8B-Instruct-2509`
- the trained tokenizer-aligned CPT checkpoint, usually the `final` directory written by `CPT/cpt.py`

It keeps the same two testing modes:

- ready-made evaluation probes for Greek fluency, formatting, JSON output, English retention, bilingual switching, and uncertainty handling
- a freeform prompt box for any manual check you want to run

The session log is only for review inside the browser. Previous runs are shown there, but they are not sent back into either model.

## Run

Use the repo helper:

```bash
./run_model_ui.sh \
	--base-device cuda:0 \
	--device cuda:1 \
	--model-path /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final
```

Or call the app directly through the existing `uenv` wrapper:

```bash
./run_uenv.sh python modelUI/app.py \
	--base-model swiss-ai/Apertus-8B-Instruct-2509 \
	--base-device cuda:0 \
	--base-dtype bfloat16 \
	--model-path /capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final \
	--device cuda:1 \
	--dtype bfloat16
```

By default the UI binds to `127.0.0.1:7861`.

If you only have one GPU, use the trained checkpoint on GPU and the base model on CPU to avoid loading both 8B models onto the same card:

```bash
./run_model_ui.sh \
	--base-device cpu \
	--base-dtype float32 \
	--device cuda:0 \
	--dtype bfloat16
```

## Useful arguments

- `--base-model`: local path or model id for the original base lane
- `--base-device` and `--base-dtype`: runtime placement for the base lane
- `--model-path`: local checkpoint directory to load, ideally the `final` directory written by CPT
- `--device`: inference device such as `cuda:0` or `cpu`
- `--dtype`: one of `bfloat16`, `float16`, or `float32`
- `--attn-implementation`: attention backend, default `sdpa`
- `--host` and `--port`: web UI bind address

The first request can take a while because both model weights are loaded lazily on demand. Compare runs only execute in parallel when the two lanes use different explicit devices.
