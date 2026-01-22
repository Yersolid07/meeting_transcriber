#!/usr/bin/env python3
"""
Script untuk augmentasi audio - menambahkan noise untuk membuat kondisi noisy.
Berguna untuk membuat dataset kondisi_noisy dari kondisi_bersih.
"""

import argparse
from pathlib import Path

import torch
import torchaudio

# Import project modules lazily inside functions to avoid executing code at module import time
from src.utils import ensure_dir, list_audio_files


def add_white_noise(waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """
    Add white noise to waveform at specified SNR.

    Args:
        waveform: Input waveform [1, T]
        snr_db: Signal-to-Noise Ratio in dB

    Returns:
        Noisy waveform
    """
    # Calculate signal power
    signal_power = torch.mean(waveform**2)

    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate white noise
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)

    # Add noise to signal
    noisy_waveform = waveform + noise

    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(noisy_waveform))
    if max_val > 1.0:
        noisy_waveform = noisy_waveform / max_val

    return noisy_waveform


def add_ambient_noise(
    waveform: torch.Tensor, noise_file: str, snr_db: float = 15.0, sample_rate: int = 16000
) -> torch.Tensor:
    """
    Add ambient noise from file to waveform.

    Args:
        waveform: Input waveform [1, T]
        noise_file: Path to noise audio file
        snr_db: Signal-to-Noise Ratio in dB
        sample_rate: Sample rate

    Returns:
        Noisy waveform
    """
    # Load noise file
    noise_waveform, noise_sr = torchaudio.load(noise_file)

    # Convert to mono if needed
    if noise_waveform.shape[0] > 1:
        noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)

    # Resample if needed
    if noise_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(noise_sr, sample_rate)
        noise_waveform = resampler(noise_waveform)

    # Match length
    signal_length = waveform.shape[1]
    noise_length = noise_waveform.shape[1]

    if noise_length < signal_length:
        # Repeat noise
        repeats = (signal_length // noise_length) + 1
        noise_waveform = noise_waveform.repeat(1, repeats)

    # Trim to match signal length
    noise_waveform = noise_waveform[:, :signal_length]

    # Calculate powers
    signal_power = torch.mean(waveform**2)
    noise_power = torch.mean(noise_waveform**2)

    # Scale noise for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr_linear * noise_power))

    scaled_noise = noise_waveform * scale

    # Add noise
    noisy_waveform = waveform + scaled_noise

    # Normalize
    max_val = torch.max(torch.abs(noisy_waveform))
    if max_val > 1.0:
        noisy_waveform = noisy_waveform / max_val

    return noisy_waveform


def add_reverb(
    waveform: torch.Tensor, sample_rate: int = 16000, room_scale: float = 0.5
) -> torch.Tensor:
    """
    Add simple reverb effect to simulate room acoustics.

    Args:
        waveform: Input waveform [1, T]
        sample_rate: Sample rate
        room_scale: Room size scale (0-1)

    Returns:
        Waveform with reverb
    """
    # Simple delay-based reverb
    delays_ms = [20, 40, 60, 80, 100]
    decays = [0.6, 0.4, 0.3, 0.2, 0.1]

    output = waveform.clone()

    for delay_ms, decay in zip(delays_ms, decays):
        delay_samples = int(delay_ms * room_scale * sample_rate / 1000)

        if delay_samples < waveform.shape[1]:
            delayed = torch.zeros_like(waveform)
            delayed[:, delay_samples:] = waveform[:, :-delay_samples] * decay
            output = output + delayed

    # Normalize
    max_val = torch.max(torch.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output


def augment_audio_file(
    input_path: str,
    output_path: str,
    augmentation_type: str = "white_noise",
    snr_db: float = 20.0,
    noise_file: str = None,
    sample_rate: int = 16000,
):
    """
    Augment a single audio file.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        augmentation_type: Type of augmentation
        snr_db: SNR for noise augmentation
        noise_file: Path to noise file (for ambient noise)
        sample_rate: Sample rate
    """
    # Load audio (lazy import to keep module import-time clean)
    from src.audio_processor import AudioProcessor

    processor = AudioProcessor()
    waveform, sr = processor.load_audio(input_path)

    # Apply augmentation
    if augmentation_type == "white_noise":
        augmented = add_white_noise(waveform, snr_db)
    elif augmentation_type == "ambient_noise" and noise_file:
        augmented = add_ambient_noise(waveform, noise_file, snr_db, sr)
    elif augmentation_type == "reverb":
        augmented = add_reverb(waveform, sr)
    elif augmentation_type == "combined":
        augmented = add_white_noise(waveform, snr_db + 5)
        augmented = add_reverb(augmented, sr, 0.3)
    else:
        augmented = waveform

    # Save
    torchaudio.save(output_path, augmented, sr)

    return output_path


def augment_directory(
    input_dir: str,
    output_dir: str,
    augmentation_type: str = "white_noise",
    snr_db: float = 20.0,
    noise_file: str = None,
):
    """
    Augment all audio files in a directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    audio_files = list_audio_files(input_dir)

    print(f"Found {len(audio_files)} audio files")
    print(f"Augmentation: {augmentation_type}, SNR: {snr_db} dB")
    print(f"Output: {output_dir}")
    print("-" * 50)

    for i, audio_path in enumerate(audio_files, 1):
        output_path = output_dir / audio_path.name

        print(f"[{i}/{len(audio_files)}] {audio_path.name}...", end=" ")

        try:
            augment_audio_file(
                str(audio_path), str(output_path), augmentation_type, snr_db, noise_file
            )
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("-" * 50)
    print(f"✅ Done! Augmented files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Augment audio files to create noisy conditions")

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input audio file or directory"
    )

    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output audio file or directory"
    )

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="white_noise",
        choices=["white_noise", "ambient_noise", "reverb", "combined"],
        help="Type of augmentation",
    )

    parser.add_argument(
        "--snr", type=float, default=20.0, help="Signal-to-Noise Ratio in dB (default: 20)"
    )

    parser.add_argument(
        "--noise-file",
        type=str,
        default=None,
        help="Path to ambient noise file (for ambient_noise type)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        augment_directory(args.input, args.output, args.type, args.snr, args.noise_file)
    else:
        output_path = Path(args.output)
        ensure_dir(output_path.parent)

        augment_audio_file(args.input, args.output, args.type, args.snr, args.noise_file)
        print(f"✅ Augmented file saved: {args.output}")


if __name__ == "__main__":
    main()
