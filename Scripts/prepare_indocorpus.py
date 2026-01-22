"""
Prepare Indonesian Conversational Speech Corpus manifests and RTTM files.

The corpus layout expected:
- WAV/GroupID_ConversationID_0_SpeakerID.wav
- TXT/GroupID_ConversationID_0_SpeakerID.txt  (lines: [start,end]\tspeaker_id\tgender\ttranscript)

This script will output:
- Manifest JSON with entries for each utterance: wav, start, end, duration, text, speaker_id
- RTTM files per conversation ID (useful for diarization evaluation)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf

UTT_LINE_RE = re.compile(
    r"\[(?P<start>[0-9\.]+),(?P<end>[0-9\.]+)\]\s+(?P<speaker>[^\t]+)\s+(?P<gender>[^\t]+)\s+(?P<text>.*)"
)


def prepare_indocorpus(corpus_dir: Path, out_manifest: Path, out_rttm_dir: Path):
    wav_dir = corpus_dir / "WAV"
    txt_dir = corpus_dir / "TXT"

    if not wav_dir.exists() or not txt_dir.exists():
        raise FileNotFoundError("Expected WAV/ and TXT/ directories in corpus_dir")

    entries = []
    out_rttm_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in txt_dir.glob("*.txt"):
        convo_id = txt_file.stem.rsplit("_", 1)[0]
        rttm_lines = []

        with txt_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                m = UTT_LINE_RE.match(line)
                if not m:
                    # sometimes transcripts may not follow exact format; skip if not matching
                    continue
                start = float(m.group("start"))
                end = float(m.group("end"))
                speaker = m.group("speaker").strip()
                text = m.group("text").strip()

                # Determine corresponding wav file: use same stem convention
                # Original naming: GroupID_ConversationID_0_SpeakerID.wav
                # We can find any wav that starts with the convo prefix
                possible_wavs = list(wav_dir.glob(f"{convo_id}*.wav"))
                if not possible_wavs:
                    # skip utterance if wav not found
                    continue
                wav_path = possible_wavs[0]

                try:
                    info = sf.info(str(wav_path))
                    sr = info.samplerate
                    duration_wav = info.frames / sr
                except Exception:
                    sr = None
                    duration_wav = None

                utt_duration = end - start

                entries.append(
                    {
                        "wav": str(wav_path.resolve()),
                        "start": start,
                        "end": end,
                        "duration": utt_duration,
                        "text": text,
                        "speaker": speaker,
                    }
                )

                # RTTM line: SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
                file_id = wav_path.stem
                rttm_lines.append(
                    f"SPEAKER {file_id} 1 {start:.3f} {utt_duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
                )

        # write RTTM for this conversation
        if rttm_lines:
            out_file = out_rttm_dir / f"{convo_id}.rttm"
            with out_file.open("w", encoding="utf-8") as fout:
                fout.write("\n".join(rttm_lines))

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as fout:
        json.dump(entries, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} utterances to {out_manifest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", required=True, help="Path to Indonesian_Conversational_Speech_Corpus"
    )
    parser.add_argument("--out-manifest", required=True, help="Output manifest path")
    parser.add_argument("--out-rttm-dir", required=True, help="Output RTTM directory")
    args = parser.parse_args()

    prepare_indocorpus(Path(args.corpus_dir), Path(args.out_manifest), Path(args.out_rttm_dir))


if __name__ == "__main__":
    main()
