"""
Prepare CommonVoice (cv-corpus) manifests for ASR training.

This script reads a CommonVoice `*.tsv` file (train/dev/test) and writes a manifest JSON
where each entry contains: wav (absolute path), duration, text, and client_id (if available).

Usage:
  python scripts/prepare_commonvoice.py --cv-dir data/dataset_mentah/mcv-scripted-id-v24.0/cv-corpus-24.0-2025-12-05 --subset train --out data/manifests/cv_train.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf


def prepare_commonvoice(cv_dir: Path, subset: str, out_json: Path):
    tsv = cv_dir / f"{subset}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv}")

    rows = []
    with tsv.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for r in reader:
            # CommonVoice columns: client_id, path, sentence, up_votes, down_votes, age, gender, locale, segment
            path_field = r.get("path") or r.get("audio")
            if not path_field:
                continue
            wav_path = cv_dir / "clips" / path_field
            if not wav_path.exists():
                # try absolute
                wav_path = Path(path_field)
                if not wav_path.exists():
                    continue
            try:
                info = sf.info(str(wav_path))
                duration = float(info.frames) / float(info.samplerate)
            except Exception:
                duration = None

            rows.append(
                {
                    "wav": str(wav_path.resolve()),
                    "duration": duration,
                    "text": r.get("sentence", "").strip(),
                    "client_id": r.get("client_id", ""),
                }
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} entries to {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", required=True, help="Path to cv-corpus-*/id directory")
    parser.add_argument(
        "--subset", required=True, choices=["train", "dev", "test"], help="TSV subset to convert"
    )
    parser.add_argument("--out", required=True, help="Output manifest JSON path")
    args = parser.parse_args()

    prepare_commonvoice(Path(args.cv_dir), args.subset, Path(args.out))


if __name__ == "__main__":
    main()
