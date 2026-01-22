#!/usr/bin/env python3
"""Create manifests from Indonesian Conversational Speech Corpus (ICorpus)
Parses TXT timestamp files and creates a JSONL manifest suitable for training.
"""
import argparse
import json
from pathlib import Path


def parse_txt_to_manifest(txt_path: Path, wav_dir: Path):
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines:
        if not line.strip():
            continue
        # Expect: [start,end]\tSPEAKER_ID\tGENDER\ttext
        parts = line.split("\t", 3)
        if len(parts) < 4:
            # Try an alternative split if the format is slightly different
            # skip malformed
            continue
        time_part, speaker_id, gender, text = parts
        try:
            start_s, end_s = time_part.strip("[]").split(",")
            start, end = float(start_s), float(end_s)
        except Exception:
            continue
        wav_path = wav_dir / (txt_path.stem + ".wav")
        if not wav_path.exists():
            continue
        rec = {
            "audio_filepath": str(wav_path.resolve()),
            "start": start,
            "end": end,
            "duration": end - start,
            "text": text.strip(),
            "speaker_id": speaker_id,
        }
        records.append(rec)
    return records


def build_manifest(input_dir: str, out_path: str):
    data_dir = Path(input_dir)
    wav_dir = data_dir / "WAV"
    txt_dir = data_dir / "TXT"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as fw:
        for txt in txt_dir.glob("*.txt"):
            recs = parse_txt_to_manifest(txt, wav_dir)
            for r in recs:
                fw.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
    print(f"Wrote {total} records to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/dataset_mentah/Indonesian_Conversational_Speech_Corpus",
        help="Path to ICorpus folder",
    )
    p.add_argument("--out", default="data/manifests/icorpus.jsonl", help="Output manifest path")
    args = p.parse_args()
    build_manifest(args.input, args.out)


if __name__ == "__main__":
    main()
