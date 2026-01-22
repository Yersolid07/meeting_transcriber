"""
Audio Processor Module
======================
Handles audio loading, preprocessing, and segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class AudioConfig:
    """Configuration for audio processing"""

    sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True
    trim_silence: bool = False
    silence_threshold_db: float = -40.0
    max_duration_seconds: Optional[float] = None


@dataclass
class AudioInfo:
    """Information about loaded audio"""

    path: str
    duration_seconds: float
    sample_rate: int
    num_channels: int
    num_samples: int


class AudioProcessor:
    """
    Handles all audio preprocessing operations.
    Converts input audio to standardized format for downstream processing.

    Attributes:
        config: AudioConfig object with processing settings

    Example:
        >>> processor = AudioProcessor()
        >>> waveform, sr = processor.load_audio("meeting.wav")
        >>> print(f"Duration: {processor.get_duration(waveform, sr):.2f}s")
    """

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize AudioProcessor.

        Args:
            config: AudioConfig object (uses defaults if None)
        """
        self.config = config or AudioConfig()
        self._resampler_cache: dict = {}

    def load_audio(
        self,
        audio_path: Union[str, Path],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            Tuple of (waveform tensor [1, T], sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
        """
        audio_path = Path(audio_path)

        # Validate file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate format
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        # Load audio
        try:
            waveform, orig_sr = torchaudio.load(str(audio_path))
        except Exception as e:
            # Fallback to librosa if torchaudio fails
            if LIBROSA_AVAILABLE:
                try:
                    audio_np, orig_sr = librosa.load(str(audio_path), sr=None, mono=False)
                    if audio_np.ndim == 1:
                        audio_np = audio_np[np.newaxis, :]
                    waveform = torch.from_numpy(audio_np).float()
                except Exception:
                    # Try pydub (requires ffmpeg) as a robust fallback
                    try:
                        from pydub import AudioSegment

                        seg = AudioSegment.from_file(str(audio_path))
                        orig_sr = seg.frame_rate
                        samples = np.array(seg.get_array_of_samples())

                        if seg.channels > 1:
                            samples = samples.reshape((-1, seg.channels)).T
                        else:
                            samples = samples[np.newaxis, :]

                        # Normalize based on sample width
                        max_val = float(1 << (8 * seg.sample_width - 1))
                        audio_np = samples.astype(np.float32) / max_val
                        waveform = torch.from_numpy(audio_np).float()
                    except Exception:
                        # Try ffmpeg CLI (system binary) to decode to WAV in-memory (no extra Python packages required)
                        try:
                            import subprocess
                            import io
                            import soundfile as sf

                            proc = subprocess.run(
                                ["ffmpeg", "-i", str(audio_path), "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                check=True,
                            )
                            out = proc.stdout

                            audio_np, orig_sr = sf.read(io.BytesIO(out), dtype="float32")
                            if audio_np.ndim == 1:
                                audio_np = audio_np[np.newaxis, :]
                            else:
                                audio_np = audio_np.T
                            waveform = torch.from_numpy(audio_np).float()
                        except Exception:
                            # Last resort: use ffmpeg-python to decode into WAV bytes and read via soundfile
                            try:
                                import ffmpeg
                                import io
                                import soundfile as sf

                                out, _ = (
                                    ffmpeg.input(str(audio_path))
                                    .output("pipe:", format="wav", acodec="pcm_s16le")
                                    .run(capture_stdout=True, capture_stderr=True)
                                )

                                audio_np, orig_sr = sf.read(io.BytesIO(out), dtype="float32")
                                if audio_np.ndim == 1:
                                    audio_np = audio_np[np.newaxis, :]
                                else:
                                    audio_np = audio_np.T
                                waveform = torch.from_numpy(audio_np).float()
                            except Exception:
                                raise RuntimeError(
                                    "Format file tidak didukung atau backend decoding (ffmpeg) tidak tersedia. "
                                    "Silakan install ffmpeg (pastikan tersedia di PATH) atau gunakan format WAV/MP3 yang didukung."
                                )
            else:
                raise RuntimeError(f"Failed to load audio: {e}")

        # Trim to time range if specified
        if start_time is not None or end_time is not None:
            waveform = self._trim_to_range(waveform, orig_sr, start_time, end_time)

        # Convert to mono if needed
        if self.config.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != self.config.sample_rate:
            waveform = self._resample(waveform, orig_sr, self.config.sample_rate)

        # Normalize amplitude
        if self.config.normalize:
            waveform = self._normalize(waveform)

        # Trim silence if requested
        if self.config.trim_silence:
            waveform = self._trim_silence(waveform)

        # Enforce max duration
        if self.config.max_duration_seconds:
            max_samples = int(self.config.max_duration_seconds * self.config.sample_rate)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[:, :max_samples]

        return waveform, self.config.sample_rate

    def get_audio_info(self, audio_path: Union[str, Path]) -> AudioInfo:
        """
        Get information about audio file without loading full waveform.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioInfo object with file details
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        info = torchaudio.info(str(audio_path))

        return AudioInfo(
            path=str(audio_path),
            duration_seconds=info.num_frames / info.sample_rate,
            sample_rate=info.sample_rate,
            num_channels=info.num_channels,
            num_samples=info.num_frames,
        )

    def _trim_to_range(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> torch.Tensor:
        """Trim waveform to specified time range"""
        start_sample = int((start_time or 0) * sample_rate)
        end_sample = int((end_time or waveform.shape[-1] / sample_rate) * sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[-1], end_sample)

        return waveform[:, start_sample:end_sample]

    def _resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate with caching"""
        cache_key = (orig_sr, target_sr)

        if cache_key not in self._resampler_cache:
            self._resampler_cache[cache_key] = Resample(orig_freq=orig_sr, new_freq=target_sr)

        return self._resampler_cache[cache_key](waveform)

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to [-1, 1] range"""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Remove leading and trailing silence"""
        # Convert threshold from dB to amplitude
        threshold = 10 ** (self.config.silence_threshold_db / 20)

        # Find non-silent regions
        amplitude = torch.abs(waveform).squeeze()
        non_silent = amplitude > threshold

        if not non_silent.any():
            return waveform

        # Find first and last non-silent sample
        non_silent_indices = torch.where(non_silent)[0]
        start_idx = non_silent_indices[0].item()
        end_idx = non_silent_indices[-1].item() + 1

        return waveform[:, start_idx:end_idx]

    def get_duration(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """Get duration of waveform in seconds"""
        return waveform.shape[-1] / sample_rate

    def cut_segment(
        self, waveform: torch.Tensor, start_sec: float, end_sec: float, sample_rate: int
    ) -> torch.Tensor:
        """
        Extract a segment from waveform.

        Args:
            waveform: Input waveform [C, T]
            start_sec: Start time in seconds
            end_sec: End time in seconds
            sample_rate: Sample rate of waveform

        Returns:
            Segment waveform [C, t]
        """
        start_sample = int(max(0, start_sec) * sample_rate)
        end_sample = int(min(end_sec * sample_rate, waveform.shape[-1]))

        return waveform[:, start_sample:end_sample]

    def split_into_chunks(
        self,
        waveform: torch.Tensor,
        chunk_duration: float,
        overlap: float = 0.0,
        sample_rate: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Split waveform into overlapping chunks.

        Args:
            waveform: Input waveform
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            sample_rate: Sample rate (uses config if None)

        Returns:
            List of (chunk_waveform, start_sec, end_sec)
        """
        sample_rate = sample_rate or self.config.sample_rate
        total_duration = self.get_duration(waveform, sample_rate)

        chunks = []
        start = 0.0

        while start < total_duration:
            end = min(start + chunk_duration, total_duration)
            chunk = self.cut_segment(waveform, start, end, sample_rate)
            chunks.append((chunk, start, end))
            start += chunk_duration - overlap

        return chunks

    def add_noise(
        self, waveform: torch.Tensor, noise_level: float = 0.01, noise_type: str = "gaussian"
    ) -> torch.Tensor:
        """
        Add noise to waveform (for data augmentation).

        Args:
            waveform: Input waveform
            noise_level: Noise amplitude (0-1)
            noise_type: Type of noise ("gaussian", "uniform")

        Returns:
            Waveform with added noise
        """
        if noise_type == "gaussian":
            noise = torch.randn_like(waveform) * noise_level
        elif noise_type == "uniform":
            noise = (torch.rand_like(waveform) * 2 - 1) * noise_level
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return waveform + noise

    def save_audio(
        self,
        waveform: torch.Tensor,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ):
        """
        Save waveform to audio file.

        Args:
            waveform: Waveform to save
            output_path: Output file path
            sample_rate: Sample rate (uses config if None)
        """
        sample_rate = sample_rate or self.config.sample_rate
        torchaudio.save(str(output_path), waveform, sample_rate)
