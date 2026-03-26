import json
from pathlib import Path

from tokenizer_extract_common import build_parser, extract_tokenizer


DEFAULT_MODEL_ID = "ilsp/Llama-Krikri-8B-Instruct"
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizers/krikri-base")
DEFAULT_REPORT_PATH = Path("artifacts/reports/tokenizer_krikri_baseline.json")
DEFAULT_READABLE_TOKENIZER_PATH = Path("artifacts/tokenizers/krikri-base/tokenizer_readable.json")


def parse_args():
    parser = build_parser(
        description="Download and save the Krikri tokenizer with a small metadata report.",
        default_model_id=DEFAULT_MODEL_ID,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        default_report_path=DEFAULT_REPORT_PATH,
        default_readable_tokenizer_path=DEFAULT_READABLE_TOKENIZER_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = extract_tokenizer(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()