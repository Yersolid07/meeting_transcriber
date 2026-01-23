#!/usr/bin/env python3
"""
Convert a speaker-labeled ground-truth text file into:
 - a plain transcript `.txt` (no speaker labels) for WER
 - an RTTM `.rttm` with estimated or audio-aligned timestamps for DER

Usage examples:
  python Scripts/convert_gt_to_rttm.py --gt data/ground_truth/rapatsingkat_gt.txt --audio data/audio/kondisi_multispeaker/rapatsingkat.mp3 --inplace
  python Scripts/convert_gt_to_rttm.py --gt my_gt_with_labels.txt --out-dir data/ground_truth

The script will create files named like `<prefix>_gt.txt` and `<prefix>_gt.rttm`.
If `--inplace` is provided, the plain .txt will overwrite the input (after creating a .bak)
and the .rttm will be written next to it.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def parse_speaker_labeled_file(gt_path: Path) -> List[Tuple[str, str]]:
    """Parse a speaker-labeled GT file into list of (speaker, text) in order.

    Recognizes lines that start with `Name:` (case-insensitive) as speaker labels.
    Consecutive non-label lines are appended to the current speaker utterance.
    If no labels found, returns a single entry with speaker 'SPK_01' and whole text.
    """
    label_re = re.compile(r"^\s*([^:\n\r]{1,80}):\s*(.*)$")
    items: List[Tuple[str, str]] = []

    with open(gt_path, "r", encoding="utf-8") as f:
        cur_speaker = None
        cur_lines: List[str] = []
        any_label = False
        for raw in f:
            line = raw.rstrip("\n\r")
            m = label_re.match(line)
            if m:
                any_label = True
                # flush existing
                if cur_speaker is not None:
                    items.append((cur_speaker, " ".join(l.strip() for l in cur_lines if l.strip())))
                cur_speaker = m.group(1).strip()
                first = m.group(2).strip()
                cur_lines = [first] if first else []
            else:
                # append to current speaker text if non-empty
                if line.strip():
                    cur_lines.append(line.strip())
        # flush last
        if cur_speaker is not None:
            items.append((cur_speaker, " ".join(l.strip() for l in cur_lines if l.strip())))

    if not items:
        # no labels found: treat whole file as single utterance
        with open(gt_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            return []
        return [("SPK_01", txt)]

    return items


def sanitize_speaker_id(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_\-]", "_", s.strip())
    if not s:
        return "SPK_01"
    return s


def words_in_text(t: str) -> int:
    return len([w for w in re.findall(r"\w+", t)])


def get_audio_duration_seconds(audio_path: Path) -> Optional[float]:
    # Prefer soundfile for quick info
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    # Fallback: torchaudio
    try:
        import torchaudio

        info = torchaudio.info(str(audio_path))
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    return None


def allocate_durations_proportional(
    utterances: List[Tuple[str, str]], total_duration: float, min_dur: float = 0.2
):
    # Compute words per utterance
    counts = [max(1, words_in_text(t)) for _, t in utterances]
    total_words = sum(counts)
    if total_words <= 0:
        # evenly split
        n = len(utterances)
        base = total_duration / n if n > 0 else 0
        return [max(min_dur, base) for _ in utterances]

    durations = [max(min_dur, total_duration * (c / total_words)) for c in counts]

    # Adjust sum to equal total_duration
    s = sum(durations)
    if s == 0:
        return durations
    factor = total_duration / s
    durations = [d * factor for d in durations]

    # final adjustment to avoid rounding gaps
    diff = total_duration - sum(durations)
    if abs(diff) > 1e-6 and durations:
        durations[-1] += diff

    return durations


def build_rttm_lines(
    file_id: str,
    utterances: List[Tuple[str, str]],
    durations: List[float],
    start_offset: float = 0.0,
):
    lines: List[str] = []
    t = start_offset
    for (speaker, text), dur in zip(utterances, durations):
        spk = sanitize_speaker_id(speaker)
        # Format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
        lines.append(f"SPEAKER {file_id} 1 {t:.2f} {dur:.2f} <NA> <NA> {spk} <NA> <NA>")
        t += dur
    return lines


def convert(
    gt: Path,
    audio: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    inplace: bool = False,
    wps: float = 3.5,
):
    gt = Path(gt)
    if out_dir is None:
        out_dir = gt.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    utterances = parse_speaker_labeled_file(gt)
    if not utterances:
        raise RuntimeError("No utterances parsed from GT file")

    # Create plain transcript (no speaker labels)
    plain_text = "\n".join([t for _, t in utterances if t.strip()])

    prefix = gt.stem
    # Normalize common naming: if file already ends with '_gt', avoid doubling (e.g. rapatsingkat_gt -> rapatsingkat)
    if prefix.lower().endswith("_gt"):
        # remove trailing _gt to keep output names clean
        prefix = prefix[:-3]
    plain_path = out_dir / f"{prefix}_gt.txt"

    if inplace:
        # backup original
        bak = gt.with_suffix(gt.suffix + ".bak")
        if not bak.exists():
            gt.rename(bak)
        # write plain to original filename
        plain_path = gt

    with open(plain_path, "w", encoding="utf-8") as f:
        f.write(plain_text.strip() + "\n")

    # Build RTTM
    file_id = prefix
    audio_duration = None
    if audio is not None:
        audio = Path(audio)
        audio_duration = get_audio_duration_seconds(audio)
        if audio_duration is None:
            print(
                "Warning: could not read audio duration; RTTM will be estimated from words-per-second"
            )

    if audio_duration is None:
        # estimate by words-per-second
        total_words = sum(max(1, words_in_text(t)) for _, t in utterances)
        audio_duration = max(1.0, total_words / wps)

    durations = allocate_durations_proportional(utterances, audio_duration)

    rttm_lines = build_rttm_lines(file_id, utterances, durations)

    rttm_path = out_dir / f"{prefix}_gt.rttm"
    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rttm_lines) + "\n")

    return str(plain_path), str(rttm_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert speaker-labeled GT text to plain .txt and .rttm"
    )
    parser.add_argument("--gt", required=True, help="Input speaker-labeled GT text file")
    parser.add_argument(
        "--audio", help="Optional path to audio file to get duration for RTTM (recommended)"
    )
    parser.add_argument("--out-dir", help="Output directory (defaults to input file parent)")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input .txt with plain transcript (backup created)",
    )
    parser.add_argument(
        "--wps",
        type=float,
        default=3.5,
        help="Words-per-second rate used when audio not available (default 3.5)",
    )

    args = parser.parse_args()

    plain, rttm = convert(
        Path(args.gt),
        Path(args.audio) if args.audio else None,
        Path(args.out_dir) if args.out_dir else None,
        inplace=args.inplace,
        wps=args.wps,
    )
    print(f"Plain transcript written: {plain}")
    print(f"RTTM written: {rttm}")


if __name__ == "__main__":
    main()
