"""
Create a small manifest for quick experiments by sampling an existing manifest.

Usage:
  python recipes/asr/prepare_quick_manifest.py --input data/manifests/train.json --output data/manifests/train_small.jsonl --max-examples 200
"""

import argparse
import json
import random
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(inp, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    if args.sample_frac and 0.0 < args.sample_frac < 1.0:
        k = max(1, int(len(lines) * args.sample_frac))
        lines = random.sample(lines, k)
    if args.max_examples and len(lines) > args.max_examples:
        lines = random.sample(lines, args.max_examples)

    with open(out, "w", encoding="utf-8") as fh:
        for l in lines:
            fh.write(l + "\n")
    print(f"Wrote {len(lines)} lines to {out}")
    return 0


if __name__ == "__main__":
    main()
