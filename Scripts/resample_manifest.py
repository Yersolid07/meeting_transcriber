#!/usr/bin/env python3
"""Resample audio files referenced by a JSONL manifest to 16kHz mono
Writes resampled files under data/resampled/<relative-path> and updates manifest entries.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

try:
    import torchaudio
except Exception:
    torchaudio = None


def resample_file(src: Path, dst: Path, target_sr: int = 16000):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if torchaudio:
        waveform, sr = torchaudio.load(str(src))
        if sr != target_sr or waveform.shape[0] != 1:
            # convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            resampled = torchaudio.transforms.Resample(sr, target_sr)(waveform)
            torchaudio.save(str(dst), resampled, target_sr)
        else:
            shutil.copy2(src, dst)
    else:
        # Fallback to ffmpeg CLI
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            str(dst),
        ]
        subprocess.run(cmd, check=True)


def resample_manifest(
    manifest_path: str,
    out_prefix: str = "data/resampled",
    target_sr: int = 16000,
    overwrite: bool = False,
):
    manifest = Path(manifest_path)
    out_prefix = Path(out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)
    out_entries = []
    with manifest.open("r", encoding="utf-8") as fr:
        for line in fr:
            j = json.loads(line)
            src = Path(j["audio_filepath"])
            rel = src.name
            dst = out_prefix / rel
            if dst.exists() and not overwrite:
                j["audio_filepath"] = str(dst.resolve())
                out_entries.append(j)
                continue
            resample_file(src, dst, target_sr)
            # If resampling succeeded, update filepath and (re)compute duration
            import soundfile as sf

            info = sf.info(str(dst))
            j["audio_filepath"] = str(dst.resolve())
            j["duration"] = info.frames / float(info.samplerate)
            out_entries.append(j)
    out_manifest = manifest.parent / f"{manifest.stem}.resampled.jsonl"
    with out_manifest.open("w", encoding="utf-8") as fw:
        for e in out_entries:
            fw.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote resampled manifest to {out_manifest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/icorpus.jsonl")
    p.add_argument("--outprefix", default="data/resampled")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    resample_manifest(args.manifest, args.outprefix, overwrite=args.overwrite)
