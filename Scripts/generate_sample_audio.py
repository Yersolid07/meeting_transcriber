"""
Generate a small synthetic WAV file useful for a dev manifest or CI smoke tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def generate(path: Path, duration_s: float = 1.0, sr: int = 16000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # simple sine wave at 440Hz
    wav = 0.05 * np.sin(2 * np.pi * 440 * t).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)
    print(f"Generated sample audio: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()
    generate(Path(args.out), duration_s=args.duration, sr=args.sr)


if __name__ == "__main__":
    main()
