"""
Evaluate ASR predictions produced by `train_full.py` or other systems.

Input: predictions.jsonl with objects containing at minimum: {ref, hyp, wav, start, end, speaker}
Outputs: prints summary metrics and writes `metrics_summary.json` and `metrics_summary.csv` in same folder.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def simple_wer(ref: str, hyp: str) -> float:
    r = ref.strip().split()
    h = hyp.strip().split()
    # DP
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return d[len(r)][len(h)] / len(r)


def simple_cer(ref: str, hyp: str) -> float:
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return d[len(r)][len(h)] / len(r)


def load_predictions(path: Path) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line.strip()))
    return out


def summarize(preds: List[Dict]):
    # compute overall WER/CER and per-epoch if present
    per_epoch = defaultdict(list)
    for p in preds:
        epoch = p.get("epoch", "all")
        per_epoch[epoch].append(p)

    summary = {}
    for epoch, items in per_epoch.items():
        wers = [simple_wer(it.get("ref", ""), it.get("hyp", "")) for it in items]
        cers = [simple_cer(it.get("ref", ""), it.get("hyp", "")) for it in items]
        summary[epoch] = {
            "wer": sum(wers) / max(1, len(wers)),
            "cer": sum(cers) / max(1, len(cers)),
            "count": len(items),
        }
    return summary


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
    summary = summarize(preds)

    outdir = Path(args.outdir) if args.outdir else preds_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    summary_json = outdir / "metrics_summary.json"
    summary_csv = outdir / "metrics_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # CSV: epoch, wer, cer, count
    with open(summary_csv, "w", encoding="utf-8") as fh:
        fh.write("epoch,wer,cer,count\n")
        for epoch, s in summary.items():
            fh.write(f"{epoch},{s['wer']:.4f},{s['cer']:.4f},{s['count']}\n")

    print("Summary written:", summary_json, summary_csv)
    print("Summary:")
    for epoch, s in summary.items():
        print(f"Epoch {epoch}: WER={s['wer']:.4f} CER={s['cer']:.4f} Cnt={s['count']}")


if __name__ == "__main__":
    main()
