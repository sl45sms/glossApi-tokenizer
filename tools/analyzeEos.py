import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_tokenizer import load_repo_tokenizer

# Usage:
# Basic:
# ./run_uenv.sh python tools/analyzeEos.py --model-path /capstor/scratch/cscs/p-skarvelis/apertus-greek-sft/
# Force CUDA:
# ./run_uenv.sh python tools/analyzeEos.py --model-path /capstor/scratch/cscs/p-skarvelis/apertus-greek-sft/ --device cuda
# Custom prompt:
# ./run_uenv.sh python tools/analyzeEos.py --model-path /capstor/scratch/cscs/p-skarvelis/apertus-greek-sft/ --prompt "Γράψε μια σύντομη περίληψη για την ελληνική ιστορία."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EOS behavior for a local checkpoint.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/capstor/scratch/cscs/p-skarvelis/apertus-greek-sft/"),
        help="Path to local model/checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Ποιος είναι ο πρωθυπουργός της Ελλάδας;",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the prompt with the tokenizer chat template when available.",
    )
    parser.add_argument(
        "--attn-implementation",
        help="Optional attention backend passed to model loading, e.g. eager or sdpa.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {requested_device} was requested but CUDA is not available in this environment.")
    return requested_device


def build_prompt(tokenizer, prompt: str, use_chat_template: bool) -> tuple[str, bool]:
    uses_chat_template = bool(use_chat_template and getattr(tokenizer, "chat_template", None))
    if not uses_chat_template:
        return prompt, False
    return (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        True,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from {args.model_path} on device={device} with dtype={dtype}...")
    tokenizer = load_repo_tokenizer(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Έλεγχος των ειδικών tokens
    print("\n--- Tokenizer Config ---")
    print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # 2. Προετοιμασία του Prompt
    prompt_text, uses_chat_template = build_prompt(tokenizer, args.prompt, args.use_chat_template)
    print(f"Chat template used: {uses_chat_template}")
    inputs = tokenizer(
        prompt_text,
        add_special_tokens=not uses_chat_template,
        return_tensors="pt",
    ).to(device)

    # 3. Παραγωγή κειμένου
    print("\n--- Generating ---")
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    # 4. Ανάλυση των IDs
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]  # Παίρνουμε μόνο τη νέα απάντηση
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print(f"Response: {decoded_text}")
    print("\n--- Token ID Analysis ---")
    last_tokens = generated_ids[-10:].tolist()  # Δες τα τελευταία 10 tokens
    print(f"Last 10 token IDs: {last_tokens}")

    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated_ids:
        pos = (generated_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
        print(f"✅ EOS found at position {pos}!")
    elif tokenizer.eos_token_id is None:
        print("⚠️ EOS token is not configured in tokenizer.")
    else:
        print("❌ EOS NOT FOUND in the output. The model just kept talking.")


if __name__ == "__main__":
    main()