#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_TRAINED_MODEL = "/capstor/store/cscs/swissai/a0140/p-skarvelis/apertus-greek-cpt/final"
DEFAULT_DATASET = "dascim/GreekMMLU"
DEFAULT_DATASET_CONFIG = "All"
DEFAULT_OUTPUT_JSON = "artifacts/reports/greek_mmlu_eval.json"
DEFAULT_BASE_REPORT_CACHE = "artifacts/reports/greek_mmlu_base_eval.json"
ANSWER_LABELS = ("Α", "Β", "Γ", "Δ")
PROMPT_INSTRUCTION = (
	"Απάντησε στην ακόλουθη ερώτηση πολλαπλής επιλογής δίνοντας μόνο το γράμμα "
	"της σωστής επιλογής."
)


@dataclass(frozen=True)
class GreekMMLUExample:
	question: str
	choices: Sequence[str]
	answer: int
	group: str
	subject: str
	level: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate the original Apertus base model and a Greek-CPT checkpoint on GreekMMLU, "
			"then write a JSON report with overall and category-wise accuracy."
		)
	)
	parser.add_argument(
		"--base-model",
		default=DEFAULT_BASE_MODEL,
		help="Model id or local path for the original base model.",
	)
	parser.add_argument(
		"--trained-model",
		default=DEFAULT_TRAINED_MODEL,
		help="Model id or local path for the final CPT checkpoint.",
	)
	parser.add_argument(
		"--dataset",
		default=DEFAULT_DATASET,
		help="GreekMMLU dataset id.",
	)
	parser.add_argument(
		"--dataset-config",
		default=DEFAULT_DATASET_CONFIG,
		help="GreekMMLU configuration to evaluate. Use All for the aggregate benchmark.",
	)
	parser.add_argument(
		"--split",
		default="test",
		help="Dataset split to score.",
	)
	parser.add_argument(
		"--dev-split",
		default="dev",
		help="Few-shot source split.",
	)
	parser.add_argument(
		"--num-few-shot",
		type=int,
		default=5,
		help="Number of same-subject dev examples to prepend to each prompt.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		help="Optional cap on the number of evaluation examples.",
	)
	parser.add_argument(
		"--subject",
		nargs="*",
		help="Optional subject filter applied after loading the dataset.",
	)
	parser.add_argument(
		"--output-json",
		default=DEFAULT_OUTPUT_JSON,
		help="Where to write the evaluation report.",
	)
	parser.add_argument(
		"--base-report-cache",
		default=DEFAULT_BASE_REPORT_CACHE,
		help="Path where the base-model evaluation cache is stored and reused across runs.",
	)
	parser.add_argument(
		"--use-base-report-cache",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Reuse a persistent base-model evaluation cache instead of recomputing the base model every run.",
	)
	parser.add_argument(
		"--refresh-base-report-cache",
		action="store_true",
		help="Ignore any existing cached base evaluation and recompute it before updating the cache.",
	)
	parser.add_argument(
		"--device",
		default="auto",
		help="Torch device for inference, for example auto, cpu, cuda, or cuda:0.",
	)
	parser.add_argument(
		"--torch-dtype",
		choices=("auto", "float32", "float16", "bfloat16"),
		default="auto",
		help="Torch dtype used when loading the models.",
	)
	parser.add_argument(
		"--attn-implementation",
		help="Optional attention backend passed to model loading, e.g. sdpa or flash_attention_2.",
	)
	parser.add_argument(
		"--trust-remote-code",
		action="store_true",
		help="Pass trust_remote_code=True when loading the model and tokenizer.",
	)
	parser.add_argument(
		"--use-chat-template",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Use the tokenizer chat template when available.",
	)
	parser.add_argument(
		"--progress-interval",
		type=int,
		default=100,
		help="Progress print interval per model. Set to 0 to disable.",
	)
	parser.add_argument(
		"--save-predictions",
		action="store_true",
		help="Include per-example predictions in the output JSON report.",
	)
	return parser.parse_args()


def resolve_device(device_name: str) -> str:
	if device_name == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return device_name


def resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
	dtype_map = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	if dtype_name == "auto":
		return None
	return dtype_map[dtype_name]


def load_examples(
	dataset_name: str,
	dataset_config: str,
	split: str,
	subject_filter: Optional[Sequence[str]] = None,
	limit: Optional[int] = None,
) -> List[GreekMMLUExample]:
	dataset = load_dataset(dataset_name, dataset_config, split=split)
	subject_filter_set = {value.strip() for value in subject_filter or [] if value and value.strip()}

	examples: List[GreekMMLUExample] = []
	for row in dataset:
		if subject_filter_set and str(row["subject"]).strip() not in subject_filter_set:
			continue

		choices = [str(choice).strip() for choice in row["choices"]]
		if not 2 <= len(choices) <= len(ANSWER_LABELS):
			raise ValueError(
				f"Unsupported number of answer choices: expected 2-4, got {len(choices)} for question {row['question']!r}."
			)

		answer = int(row["answer"])
		if answer < 0 or answer >= len(choices):
			raise ValueError(
				f"Answer index {answer} is out of range for question {row['question']!r}."
			)

		examples.append(
			GreekMMLUExample(
				question=str(row["question"]).strip(),
				choices=choices,
				answer=answer,
				group=str(row["group"]).strip(),
				subject=str(row["subject"]).strip(),
				level=str(row["level"]).strip(),
			)
		)

		if limit is not None and len(examples) >= limit:
			break

	return examples


def normalize_subject_filter(subject_filter: Optional[Sequence[str]]) -> List[str]:
	return sorted({value.strip() for value in subject_filter or [] if value and value.strip()})


def compute_examples_fingerprint(examples: Sequence[GreekMMLUExample]) -> str:
	hasher = hashlib.sha256()
	for example in examples:
		for value in (example.question, *example.choices, str(example.answer), example.group, example.subject, example.level):
			hasher.update(value.encode("utf-8"))
			hasher.update(b"\0")
		hasher.update(b"\1")
	return hasher.hexdigest()


def build_base_cache_key(
	args: argparse.Namespace,
	examples: Sequence[GreekMMLUExample],
	few_shot_examples: Sequence[GreekMMLUExample],
) -> Dict[str, Any]:
	return {
		"cache_format": 1,
		"base_model": args.base_model,
		"dataset": args.dataset,
		"dataset_config": args.dataset_config,
		"split": args.split,
		"dev_split": args.dev_split,
		"num_few_shot": args.num_few_shot,
		"limit": args.limit,
		"subject_filter": normalize_subject_filter(args.subject),
		"torch_dtype": args.torch_dtype,
		"trust_remote_code": args.trust_remote_code,
		"use_chat_template": args.use_chat_template,
		"attn_implementation": args.attn_implementation,
		"save_predictions": args.save_predictions,
		"prompt_instruction": PROMPT_INSTRUCTION,
		"answer_labels": list(ANSWER_LABELS),
		"evaluation_examples_fingerprint": compute_examples_fingerprint(examples),
		"few_shot_examples_fingerprint": compute_examples_fingerprint(few_shot_examples),
	}


def is_valid_model_report(report: Any) -> bool:
	return isinstance(report, dict) and all(
		key in report
		for key in ("model_ref", "overall", "group_accuracy", "subject_accuracy", "level_accuracy", "runtime_seconds")
	)


def load_cached_base_report(cache_path: Path, expected_cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	if not cache_path.exists():
		return None

	try:
		payload = json.loads(cache_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		print(
			f"Ignoring unreadable base cache at {cache_path}; recomputing the base model evaluation.",
			file=sys.stderr,
			flush=True,
		)
		return None

	cache_key = payload.get("cache_key")
	base_report = payload.get("base_report")
	if cache_key != expected_cache_key:
		print(
			f"Base cache at {cache_path} does not match the current evaluation settings; recomputing.",
			file=sys.stderr,
			flush=True,
		)
		return None
	if not is_valid_model_report(base_report):
		print(
			f"Base cache at {cache_path} is missing the expected report fields; recomputing.",
			file=sys.stderr,
			flush=True,
		)
		return None
	return base_report


def save_cached_base_report(
	cache_path: Path,
	cache_key: Dict[str, Any],
	base_report: Dict[str, Any],
) -> None:
	payload = {
		"cache_key": cache_key,
		"base_report": base_report,
	}
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_question_block(example: GreekMMLUExample, include_answer: bool) -> str:
	lines = [f"Ερώτηση: {example.question}", "Επιλογές:"]
	for index, choice in enumerate(example.choices):
		lines.append(f"{ANSWER_LABELS[index]}. {choice}")

	answer_line = "Απάντηση:"
	if include_answer:
		answer_line += f" {ANSWER_LABELS[example.answer]}"
	lines.append(answer_line)
	return "\n".join(lines)


def build_prompt_body(example: GreekMMLUExample, few_shot_examples: Sequence[GreekMMLUExample]) -> str:
	sections = [PROMPT_INSTRUCTION]
	for few_shot_example in few_shot_examples:
		sections.append(render_question_block(few_shot_example, include_answer=True))
	sections.append(render_question_block(example, include_answer=False))
	return "\n\n".join(sections)


def build_few_shot_index(examples: Iterable[GreekMMLUExample]) -> Dict[str, List[GreekMMLUExample]]:
	index: Dict[str, List[GreekMMLUExample]] = defaultdict(list)
	for example in examples:
		index[example.subject].append(example)
	return index


def select_few_shot_examples(
	example: GreekMMLUExample,
	few_shot_index: Dict[str, List[GreekMMLUExample]],
	all_few_shot_examples: Sequence[GreekMMLUExample],
	num_few_shot: int,
) -> Sequence[GreekMMLUExample]:
	if num_few_shot <= 0:
		return []

	subject_examples = [
		candidate
		for candidate in few_shot_index.get(example.subject, [])
		if candidate.question != example.question
	]
	if len(subject_examples) >= num_few_shot:
		return subject_examples[:num_few_shot]

	fallback: List[GreekMMLUExample] = list(subject_examples)
	for candidate in all_few_shot_examples:
		if candidate.question == example.question:
			continue
		if candidate in fallback:
			continue
		fallback.append(candidate)
		if len(fallback) >= num_few_shot:
			break

	return fallback


def init_counter() -> Dict[str, int]:
	return {"correct": 0, "total": 0}


def update_counter(counter: Dict[str, int], is_correct: bool) -> None:
	counter["total"] += 1
	counter["correct"] += int(is_correct)


def finalize_counter(counter: Dict[str, int]) -> Dict[str, Any]:
	total = counter["total"]
	correct = counter["correct"]
	accuracy = correct / total if total else 0.0
	return {"correct": correct, "total": total, "accuracy": accuracy}


def finalize_breakdown(breakdown: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
	return {
		key: finalize_counter(counter)
		for key, counter in sorted(breakdown.items(), key=lambda item: item[0])
	}


def summarize_dataset(examples: Sequence[GreekMMLUExample]) -> Dict[str, Any]:
	groups: Dict[str, Dict[str, int]] = defaultdict(init_counter)
	subjects: Dict[str, Dict[str, int]] = defaultdict(init_counter)
	levels: Dict[str, Dict[str, int]] = defaultdict(init_counter)

	for example in examples:
		update_counter(groups[example.group], True)
		update_counter(subjects[example.subject], True)
		update_counter(levels[example.level], True)

	return {
		"total_examples": len(examples),
		"group_counts": {key: value["total"] for key, value in sorted(groups.items())},
		"subject_counts": {key: value["total"] for key, value in sorted(subjects.items())},
		"level_counts": {key: value["total"] for key, value in sorted(levels.items())},
	}


def compute_accuracy_delta(
	base_breakdown: Dict[str, Dict[str, Any]],
	trained_breakdown: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
	deltas: Dict[str, float] = {}
	shared_keys = sorted(set(base_breakdown) & set(trained_breakdown))
	for key in shared_keys:
		deltas[key] = trained_breakdown[key]["accuracy"] - base_breakdown[key]["accuracy"]
	return deltas


class ModelScorer:
	def __init__(
		self,
		model_ref: str,
		device: str,
		torch_dtype: Optional[torch.dtype],
		trust_remote_code: bool,
		use_chat_template: bool,
		attn_implementation: Optional[str],
	) -> None:
		self.model_ref = model_ref
		self.device = device
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_ref,
			trust_remote_code=trust_remote_code,
		)

		model_kwargs: Dict[str, Any] = {
			"trust_remote_code": trust_remote_code,
			"low_cpu_mem_usage": True,
		}
		if torch_dtype is not None:
			model_kwargs["torch_dtype"] = torch_dtype
		if attn_implementation:
			model_kwargs["attn_implementation"] = attn_implementation

		self.model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
		self.model.to(device)
		self.model.eval()

		if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		self.uses_chat_template = bool(use_chat_template and getattr(self.tokenizer, "chat_template", None))
		context_candidates = []
		max_positions = getattr(self.model.config, "max_position_embeddings", None)
		if isinstance(max_positions, int) and max_positions > 0:
			context_candidates.append(max_positions)
		tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
		if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
			context_candidates.append(tokenizer_limit)
		self.context_limit = min(context_candidates) if context_candidates else None

	def unload(self) -> None:
		del self.model
		del self.tokenizer
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def wrap_prompt(self, prompt_body: str) -> str:
		if not self.uses_chat_template:
			return prompt_body
		return self.tokenizer.apply_chat_template(
			[{"role": "user", "content": prompt_body}],
			tokenize=False,
			add_generation_prompt=True,
		)

	def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
		prompt_inputs = self.tokenizer(
			prompt_text,
			add_special_tokens=not self.uses_chat_template,
			return_tensors="pt",
		)
		return prompt_inputs.input_ids.to(self.device)

	def _encode_candidates(self, candidate_texts: Sequence[str]) -> List[torch.Tensor]:
		candidate_token_ids: List[torch.Tensor] = []
		for candidate_text in candidate_texts:
			candidate_ids = self.tokenizer(
				candidate_text,
				add_special_tokens=False,
				return_tensors="pt",
			).input_ids[0]
			if candidate_ids.numel() == 0:
				raise ValueError(f"Candidate {candidate_text!r} tokenized to zero tokens.")
			candidate_token_ids.append(candidate_ids.to(self.device))
		return candidate_token_ids

	@torch.inference_mode()
	def score_candidates(self, prompt_text: str, candidate_texts: Sequence[str]) -> List[float]:
		prompt_ids = self._encode_prompt(prompt_text)
		candidate_token_ids = self._encode_candidates(candidate_texts)

		if self.context_limit is not None:
			required_context = prompt_ids.shape[1] + max(candidate_ids.numel() for candidate_ids in candidate_token_ids)
			if required_context > self.context_limit:
				raise ValueError(
					f"Prompt requires {required_context} tokens but model context limit is {self.context_limit}. "
					"Reduce --num-few-shot or evaluate fewer/shorter examples."
				)

		prompt_outputs = self.model(prompt_ids, use_cache=True)
		next_token_log_probs = torch.log_softmax(prompt_outputs.logits[:, -1, :].float(), dim=-1)

		scores: List[float] = []
		for candidate_ids in candidate_token_ids:
			score = next_token_log_probs[0, candidate_ids[0]].item()
			if candidate_ids.numel() > 1:
				continuation_outputs = self.model(
					candidate_ids[:-1].unsqueeze(0),
					past_key_values=prompt_outputs.past_key_values,
					use_cache=False,
				)
				continuation_log_probs = torch.log_softmax(continuation_outputs.logits[0].float(), dim=-1)
				token_indexes = torch.arange(candidate_ids.numel() - 1, device=self.device)
				score += continuation_log_probs[token_indexes, candidate_ids[1:]].sum().item()
			scores.append(score)
		return scores


def evaluate_model(
	model_label: str,
	model_ref: str,
	examples: Sequence[GreekMMLUExample],
	few_shot_index: Dict[str, List[GreekMMLUExample]],
	all_few_shot_examples: Sequence[GreekMMLUExample],
	args: argparse.Namespace,
) -> Dict[str, Any]:
	scorer = ModelScorer(
		model_ref=model_ref,
		device=resolve_device(args.device),
		torch_dtype=resolve_torch_dtype(args.torch_dtype),
		trust_remote_code=args.trust_remote_code,
		use_chat_template=args.use_chat_template,
		attn_implementation=args.attn_implementation,
	)

	overall = init_counter()
	groups: Dict[str, Dict[str, int]] = defaultdict(init_counter)
	subjects: Dict[str, Dict[str, int]] = defaultdict(init_counter)
	levels: Dict[str, Dict[str, int]] = defaultdict(init_counter)
	predictions: List[Dict[str, Any]] = []
	start_time = time.time()

	try:
		for index, example in enumerate(examples, start=1):
			few_shot_examples = select_few_shot_examples(
				example=example,
				few_shot_index=few_shot_index,
				all_few_shot_examples=all_few_shot_examples,
				num_few_shot=args.num_few_shot,
			)
			prompt_body = build_prompt_body(example, few_shot_examples)
			prompt_text = scorer.wrap_prompt(prompt_body)
			candidate_texts = [f" {ANSWER_LABELS[i]}" for i in range(len(example.choices))]
			candidate_scores = scorer.score_candidates(prompt_text, candidate_texts)
			predicted_answer = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
			is_correct = predicted_answer == example.answer

			update_counter(overall, is_correct)
			update_counter(groups[example.group], is_correct)
			update_counter(subjects[example.subject], is_correct)
			update_counter(levels[example.level], is_correct)

			if args.save_predictions:
				predictions.append(
					{
						"index": index - 1,
						"question": example.question,
						"choices": list(example.choices),
						"group": example.group,
						"subject": example.subject,
						"level": example.level,
						"gold_label": ANSWER_LABELS[example.answer],
						"predicted_label": ANSWER_LABELS[predicted_answer],
						"is_correct": is_correct,
						"scores": {
							ANSWER_LABELS[choice_index]: candidate_scores[choice_index]
							for choice_index in range(len(candidate_scores))
						},
					}
				)

			if args.progress_interval and index % args.progress_interval == 0:
				elapsed = time.time() - start_time
				current_accuracy = overall["correct"] / overall["total"]
				print(
					(
						f"[{model_label}] {index}/{len(examples)} examples "
						f"accuracy={current_accuracy:.4f} elapsed={elapsed:.1f}s"
					),
					file=sys.stderr,
					flush=True,
				)
	finally:
		scorer.unload()

	report = {
		"model_ref": model_ref,
		"overall": finalize_counter(overall),
		"group_accuracy": finalize_breakdown(groups),
		"subject_accuracy": finalize_breakdown(subjects),
		"level_accuracy": finalize_breakdown(levels),
		"runtime_seconds": time.time() - start_time,
	}
	if args.save_predictions:
		report["predictions"] = predictions
	return report


def build_report(
	args: argparse.Namespace,
	examples: Sequence[GreekMMLUExample],
	few_shot_examples: Sequence[GreekMMLUExample],
	base_report: Dict[str, Any],
	trained_report: Dict[str, Any],
	base_report_cache_path: Optional[Path],
	base_report_cache_hit: bool,
) -> Dict[str, Any]:
	return {
		"dataset": {
			"name": args.dataset,
			"config": args.dataset_config,
			"split": args.split,
			"dev_split": args.dev_split,
			"num_examples": len(examples),
			"num_few_shot": args.num_few_shot,
			"num_few_shot_source_examples": len(few_shot_examples),
			"summary": summarize_dataset(examples),
		},
		"inference": {
			"device": resolve_device(args.device),
			"torch_dtype": args.torch_dtype,
			"use_chat_template": args.use_chat_template,
			"trust_remote_code": args.trust_remote_code,
			"attn_implementation": args.attn_implementation,
		},
		"models": {
			"base": base_report,
			"trained": trained_report,
		},
		"cache": {
			"base_report_cache": str(base_report_cache_path) if base_report_cache_path is not None else None,
			"base_report_cache_hit": base_report_cache_hit,
		},
		"comparison": {
			"overall_accuracy_delta": (
				trained_report["overall"]["accuracy"] - base_report["overall"]["accuracy"]
			),
			"group_accuracy_delta": compute_accuracy_delta(
				base_report["group_accuracy"],
				trained_report["group_accuracy"],
			),
			"subject_accuracy_delta": compute_accuracy_delta(
				base_report["subject_accuracy"],
				trained_report["subject_accuracy"],
			),
			"level_accuracy_delta": compute_accuracy_delta(
				base_report["level_accuracy"],
				trained_report["level_accuracy"],
			),
		},
	}


def main() -> None:
	args = parse_args()
	base_report_cache_path = Path(args.base_report_cache) if args.use_base_report_cache and args.base_report_cache else None
	examples = load_examples(
		dataset_name=args.dataset,
		dataset_config=args.dataset_config,
		split=args.split,
		subject_filter=args.subject,
		limit=args.limit,
	)
	if not examples:
		raise ValueError("No evaluation examples were loaded. Check --dataset-config, --split, or --subject.")

	few_shot_examples = load_examples(
		dataset_name=args.dataset,
		dataset_config=args.dataset_config,
		split=args.dev_split,
		subject_filter=args.subject,
		limit=None,
	)
	few_shot_index = build_few_shot_index(few_shot_examples)
	base_cache_key = build_base_cache_key(args, examples, few_shot_examples)

	print(
		(
			f"Loaded {len(examples)} evaluation examples from {args.dataset}/{args.dataset_config}:{args.split} "
			f"and {len(few_shot_examples)} few-shot examples from {args.dev_split}."
		),
		file=sys.stderr,
		flush=True,
	)

	base_report_cache_hit = False
	base_report: Optional[Dict[str, Any]] = None
	if base_report_cache_path is not None and not args.refresh_base_report_cache:
		base_report = load_cached_base_report(base_report_cache_path, base_cache_key)
		if base_report is not None:
			base_report_cache_hit = True
			print(
				f"Using cached base evaluation from {base_report_cache_path}.",
				file=sys.stderr,
				flush=True,
			)

	if base_report is None:
		base_report = evaluate_model(
			model_label="base",
			model_ref=args.base_model,
			examples=examples,
			few_shot_index=few_shot_index,
			all_few_shot_examples=few_shot_examples,
			args=args,
		)
		if base_report_cache_path is not None:
			save_cached_base_report(base_report_cache_path, base_cache_key, base_report)
			print(
				f"Saved base evaluation cache to {base_report_cache_path}.",
				file=sys.stderr,
				flush=True,
			)

	trained_report = evaluate_model(
		model_label="trained",
		model_ref=args.trained_model,
		examples=examples,
		few_shot_index=few_shot_index,
		all_few_shot_examples=few_shot_examples,
		args=args,
	)

	report = build_report(
		args=args,
		examples=examples,
		few_shot_examples=few_shot_examples,
		base_report=base_report,
		trained_report=trained_report,
		base_report_cache_path=base_report_cache_path,
		base_report_cache_hit=base_report_cache_hit,
	)

	output_path = Path(args.output_json)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	summary = {
		"output_json": str(output_path),
		"base_report_cache": str(base_report_cache_path) if base_report_cache_path is not None else None,
		"base_report_cache_hit": base_report_cache_hit,
		"base_accuracy": base_report["overall"]["accuracy"],
		"trained_accuracy": trained_report["overall"]["accuracy"],
		"accuracy_delta": report["comparison"]["overall_accuracy_delta"],
	}
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
