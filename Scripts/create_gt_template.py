#!/usr/bin/env python3
"""
Script untuk membuat template ground truth dari file audio.
"""

import argparse
from pathlib import Path

# Import project modules lazily inside functions to avoid module-level side-effects
from src.utils import format_duration


def create_template(audio_path: str, output_dir: str, num_speakers: int = 2):
    """Create ground truth template files"""

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get audio info (lazy import)
    from src.audio_processor import AudioProcessor

    processor = AudioProcessor()
    waveform, sr = processor.load_audio(str(audio_path))
    duration = processor.get_duration(waveform, sr)

    sample_name = audio_path.stem

    # Create transcript template
    transcript_path = output_dir / f"{sample_name}.txt"
    transcript_content = f"""# Ground Truth Transcript untuk: {audio_path.name}
# Durasi audio: {format_duration(duration)} ({duration:.2f} detik)
#
# INSTRUKSI:
# 1. Dengarkan audio dengan seksama
# 2. Tulis transkripsi lengkap di bawah (hapus semua baris komentar)
# 3. Gunakan tanda baca yang tepat
# 4. Tulis apa adanya (termasuk kata-kata tidak jelas)
#
# Mulai transkripsi di bawah baris ini:
# ============================================

[Tulis transkripsi di sini...]

"""

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_content)

    print(f"✅ Transcript template: {transcript_path}")

    # Create RTTM template
    rttm_path = output_dir / f"{sample_name}.rttm"
    rttm_content = f"""# Ground Truth Diarization (RTTM Format)
# File: {audio_path.name}
# Durasi: {format_duration(duration)} ({duration:.2f} detik)
# Estimasi speakers: {num_speakers}
#
# FORMAT:
# SPEAKER <file_id> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
#
# CONTOH:
# SPEAKER {sample_name} 1 0.00 5.50 <NA> <NA> SPEAKER_00 <NA> <NA>
# SPEAKER {sample_name} 1 5.50 3.20 <NA> <NA> SPEAKER_01 <NA> <NA>
#
# INSTRUKSI:
# 1. Dengarkan audio dan identifikasi pergantian speaker
# 2. Catat waktu mulai (dalam detik) dan durasi setiap segmen
# 3. Gunakan label SPEAKER_00, SPEAKER_01, dst.
# 4. Hapus semua baris komentar sebelum evaluasi
#
# Mulai anotasi di bawah:
# ============================================

SPEAKER {sample_name} 1 0.00 0.00 <NA> <NA> SPEAKER_00 <NA> <NA>

"""

    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write(rttm_content)

    print(f"✅ RTTM template: {rttm_path}")

    # Create info file
    info_path = output_dir / f"{sample_name}_info.json"
    info_content = {
        "audio_file": audio_path.name,
        "duration_seconds": duration,
        "duration_formatted": format_duration(duration),
        "estimated_speakers": num_speakers,
        "transcript_file": f"{sample_name}.txt",
        "rttm_file": f"{sample_name}.rttm",
        "status": "pending",
    }

    import json

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_content, f, indent=2)

    print(f"✅ Info file: {info_path}")

    return transcript_path, rttm_path


def main():
    parser = argparse.ArgumentParser(description="Create ground truth templates")
    parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--speakers", "-s", type=int, default=2, help="Estimated number of speakers"
    )

    args = parser.parse_args()

    create_template(args.audio, args.output, args.speakers)

    print("\n✅ Template created successfully!")
    print("Next steps:")
    print("  1. Dengarkan audio dan isi transkripsi di file .txt")
    print("  2. Anotasi speaker segments di file .rttm")
    print("  3. Hapus semua baris komentar")


if __name__ == "__main__":
    main()
