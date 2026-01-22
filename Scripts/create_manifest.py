"""
Create JSON manifests for ASR training from a directory of wav files and a CSV/TSV of transcripts.

Example usage:
  python scripts/create_manifest.py --audio-dir data/audio --transcripts transcripts.csv --out data/manifests/train.json

CSV format: filename,wav_path,text
If `wav_path` is a path relative to the audio directory, the script will resolve it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import soundfile as sf


def make_manifest(audio_dir: Path, transcripts_csv: Path, out_json: Path):
    rows = []
    with transcripts_csv.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            fname = r.get("filename") or r.get("file") or r.get("wav")
            wav_path = r.get("wav_path") or r.get("path") or r.get("file")
            text = r.get("text") or r.get("transcript") or r.get("label") or ""

            if not fname and not wav_path:
                continue

            if wav_path and not os.path.isabs(wav_path):
                wav_full = audio_dir / wav_path
            else:
                wav_full = Path(wav_path)

            if not wav_full.exists():
                # fallback: file may be given by filename column
                wav_full = audio_dir / fname

            if not wav_full.exists():
                print(f"Warning: file not found {wav_full}, skipping")
                continue

            try:
                info = sf.info(str(wav_full))
                duration = float(info.frames) / float(info.samplerate)
            except Exception:
                duration = None

            rows.append(
                {
                    "wav": str(wav_full.resolve()),
                    "duration": duration,
                    "text": text,
                }
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} entries to {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    make_manifest(Path(args.audio_dir), Path(args.transcripts), Path(args.out))


if __name__ == "__main__":
    main()
