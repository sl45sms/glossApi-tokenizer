#!/usr/bin/env python3
"""Convert a Parquet file to JSON."""

# Usage:
# ./tools/parquet2json.sh xxx.parquet -o xxx.json
# ./tools/parquet2json.sh xxx.parquet --lines        # JSON Lines
# ./tools/parquet2json.sh xxx.parquet --indent 0     # compact

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert a Parquet file to JSON.")
    parser.add_argument("input", type=Path, help="Input .parquet file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .json file (default: same name as input with .json extension)",
    )
    parser.add_argument(
        "--orient", default="records",
        choices=["records", "split", "index", "columns", "values", "table"],
        help="JSON orientation passed to pandas (default: records)",
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="JSON indentation (default: 2, use 0 for compact)",
    )
    parser.add_argument(
        "--lines", action="store_true",
        help="Write one JSON object per line (JSON Lines format)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_suffix(".json")

    df = pd.read_parquet(args.input)

    if args.lines:
        with open(output, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        indent = args.indent if args.indent > 0 else None
        df.to_json(output, orient=args.orient, force_ascii=False, indent=indent)

    print(f"wrote {len(df)} rows -> {output}")


if __name__ == "__main__":
    main()

