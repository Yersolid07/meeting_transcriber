"""
Lightweight SpeechBrain training entrypoint + research-oriented artifact outputs.

Features:
- Loads hyperparams YAML (optional)
- Accepts train/valid manifest paths or uses small dev-manifest for `--small-data`
- Dry-run mode when SpeechBrain is not installed or `--dry-run` is used: produces example checkpoints, predictions, and metrics for research workflows
- Produces research artifacts:
  - `experiments/<run_name>/metrics.csv` (epoch, split, wer, cer, loss)
  - `experiments/<run_name>/predictions.jsonl` (per-utterance: wav, start, end, ref, hyp, speaker)
  - `experiments/<run_name>/checkpoints/` (checkpoint files)
  - TensorBoard logs under `experiments/<run_name>/tensorboard`

This file intentionally provides a robust dry-run and a scaffold for a full SpeechBrain integration.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

# Optional imports used only if available
try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    torch = None
    SummaryWriter = None


def load_yaml(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception as e:
        # HyperPyYAML / SpeechBrain hyperparams often include custom YAML tags that
        # are not resolvable in a generic environment; for dry-run and smoke tests
        # we return an empty dict and emit a short warning. A real training run
        # with SpeechBrain should use SpeechBrain's YAML loader.
        print(f"Warning: failed to parse hyperparams file {path}: {e}")
        return {}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def simple_wer(ref: str, hyp: str) -> float:
    # Very small WER calculation using word edit distance
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


def write_predictions_jsonl(preds: List[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def append_metrics(path: Path, row: Dict):
    header = ["epoch", "split", "wer", "cer", "loss", "speaker_acc"]
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8") as fh:
        if is_new:
            fh.write(",".join(header) + "\n")
        fh.write(",".join(str(row.get(h, "")) for h in header) + "\n")


def create_dummy_artifacts(exp_dir: Path, epochs: int = 3):
    # Write checkpoints, predictions, and metrics emulating a training run
    ckpt_dir = exp_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    preds_path = exp_dir / "predictions.jsonl"
    metrics_path = exp_dir / "metrics.csv"
    tensorboard_dir = exp_dir / "tensorboard"
    ensure_dir(tensorboard_dir)

    sample_refs = [
        {
            "wav": "audio/utt1.wav",
            "start": 0.0,
            "end": 3.2,
            "ref": "saya mau makan",
            "speaker": "spk1",
        },
        {
            "wav": "audio/utt2.wav",
            "start": 3.3,
            "end": 6.0,
            "ref": "apa kabar semuanya",
            "speaker": "spk2",
        },
    ]

    preds = []
    for e in range(1, epochs + 1):
        # checkpoint
        ckpt_file = ckpt_dir / f"ckpt_epoch_{e}.pt"
        with open(ckpt_file, "w") as fh:
            fh.write("dummy checkpoint epoch %d" % e)
        # generate predictions with slight corruption per epoch
        epoch_preds = []
        for s in sample_refs:
            hyp = s["ref"]
            pred_spk = s["speaker"]
            if e < epochs:
                # minor corruption: perturb hypothesis and sometimes speaker
                hyp = hyp.replace("makan", "makann")
                if s["speaker"] == "spk1":
                    pred_spk = "spk2"
            obj = {
                "wav": s["wav"],
                "start": s["start"],
                "end": s["end"],
                "ref": s["ref"],
                "hyp": hyp,
                "speaker": s["speaker"],
                "pred_spk": pred_spk,
                "epoch": e,
            }
            preds.append(obj)
            epoch_preds.append(obj)
        # compute mock metrics
        wers = [simple_wer(item["ref"], item["hyp"]) for item in epoch_preds]
        cers = [simple_cer(item["ref"], item["hyp"]) for item in epoch_preds]
        spk_acc = sum(
            1 for item in epoch_preds if item.get("pred_spk") == item.get("speaker")
        ) / max(1, len(epoch_preds))
        append_metrics(
            metrics_path,
            {
                "epoch": e,
                "split": "dev",
                "wer": round(sum(wers) / max(1, len(wers)), 4),
                "cer": round(sum(cers) / max(1, len(cers)), 4),
                "loss": round(1.0 / e, 4),
                "speaker_acc": round(spk_acc, 4),
            },
        )

    write_predictions_jsonl(preds, preds_path)
    return {
        "predictions": str(preds_path),
        "metrics": str(metrics_path),
        "checkpoints": str(ckpt_dir),
        "tensorboard": str(tensorboard_dir),
    }


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(description="ASR training entrypoint with research artifacts")
    parser.add_argument(
        "--hparams",
        type=str,
        help="Path to hyperparams YAML",
        default="recipes/asr/hyperparams.yaml",
    )
    parser.add_argument("--train-manifest", type=str, help="JSONL manifest for training")
    parser.add_argument("--valid-manifest", type=str, help="JSONL manifest for validation")
    parser.add_argument("--outdir", type=str, help="Experiment output dir", default="experiments")
    parser.add_argument("--run-name", type=str, help="Run name (used as subfolder)", default=None)
    parser.add_argument("--epochs", type=int, help="Number of epochs for dry run", default=3)
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip real training and produce dummy artifacts"
    )
    parser.add_argument(
        "--small-data", action="store_true", help="Use built-in small dev set for CI/smoke tests"
    )

    args = parser.parse_args(argv)

    hparams = {}
    if args.hparams and Path(args.hparams).exists():
        hparams = load_yaml(args.hparams)

    run_name = args.run_name or f"run_{int(time.time())}"
    exp_dir = Path(args.outdir) / run_name
    if exp_dir.exists():
        # safe backup
        bk = str(exp_dir) + ".bak"
        if Path(bk).exists():
            shutil.rmtree(bk)
        shutil.move(str(exp_dir), bk)
    ensure_dir(exp_dir)

    # if small-data or dry-run, produce dummy artifacts and exit
    if args.small_data or args.dry_run or torch is None:
        print(
            "Running in dry-run/small-data mode. Producing example artifacts for research and testing."
        )
        artifacts = create_dummy_artifacts(exp_dir, epochs=args.epochs)
        print("Artifacts written:")
        print(json.dumps(artifacts, indent=2))
        print(
            "To run real training, install SpeechBrain and re-run without --dry-run; see docs/integration_with_speechbrain.md for full instructions."
        )
        return 0

    # Real training path: attempt to import SpeechBrain and run a full train loop
    try:
        import speechbrain as sb  # type: ignore
    except Exception as e:
        print("SpeechBrain not available or failed to import:", str(e))
        print("Run with --dry-run for CI, or install SpeechBrain to run full training.")
        return 2

    # Placeholder scaffold: the full SpeechBrain training loop belongs here.
    # Minimal scaffold given here so researchers can extend with Brain subclass.
    print(
        "SpeechBrain detected. Running training scaffold — please extend this file to implement the Brain training loop."
    )
    # TODO: implement full SpeechBrain recipe flow here (datasets, tokenization, Brain class, optimizers, checkpointer, WER/CER computation, optional speaker classification head)
    # For now, produce placeholders so downstream evaluation tools and logging are exercised
    artifacts = create_dummy_artifacts(exp_dir, epochs=args.epochs)
    print(
        "NOTE: This release includes scaffold + dry-run outputs. A full SpeechBrain training implementation is TODO; artifacts were produced at:",
        artifacts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
