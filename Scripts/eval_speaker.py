"""
Evaluate speaker predictions in `predictions.jsonl` for speaker classification accuracy and confusion.

Input: predictions.jsonl with objects containing at least: {speaker, pred_spk}
Outputs: writes `metrics_summary_spk.json` and `metrics_summary_spk.csv` to the same folder.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_predictions(path: Path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line.strip()))
    return out


def compute_speaker_metrics(preds):
    total = 0
    correct = 0
    conf = defaultdict(lambda: defaultdict(int))
    speakers = set()
    for p in preds:
        ref = p.get("speaker")
        pred = p.get("pred_spk")
        if ref is None or pred is None:
            continue
        total += 1
        speakers.add(ref)
        speakers.add(pred)
        if ref == pred:
            correct += 1
        conf[ref][pred] += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "total": total, "confusion": {r: dict(conf[r]) for r in conf}}


def write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def write_csv(path: Path, summary: dict):
    # Simple CSV: metric,value
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("metric,value\n")
        fh.write(f"accuracy,{summary['accuracy']:.4f}\n")
        fh.write(f"total,{summary['total']}\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str, help="Path to predictions.jsonl")
    parser.add_argument(
        "--outdir",
        type=str,
        help="Directory to write summary files (defaults to same folder as predictions)",
    )
    args = parser.parse_args(argv)

    preds_path = Path(args.predictions)
    assert preds_path.exists(), f"Predictions file {preds_path} not found"
    preds = load_predictions(preds_path)
    summary = compute_speaker_metrics(preds)

    outdir = Path(args.outdir) if args.outdir else preds_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    summary_json = outdir / "metrics_summary_spk.json"
    summary_csv = outdir / "metrics_summary_spk.csv"

    write_json(summary_json, summary)
    write_csv(summary_csv, summary)

    print("Speaker summary written:", summary_json, summary_csv)
    print(f"Accuracy={summary['accuracy']:.4f}, Total={summary['total']}")


if __name__ == "__main__":
    main()
