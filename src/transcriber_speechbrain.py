"""
SpeechBrain ASR wrapper (optional)
Provides a lightweight adapter around SpeechBrain's EncoderASR/EncoderDecoderASR to be used
as an optional backend in `meeting_transcriber`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from src.diarization import SpeakerSegment
from src.transcriber import TranscriptSegment


@dataclass
class SpeechBrainASRConfig:
    model_id: str = "speechbrain/asr-crdnn-rnnlm-librispeech"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_length_s: float = 30.0


class SpeechBrainTranscriber:
    """Adapter for SpeechBrain ASR models.

    Usage:
        t = SpeechBrainTranscriber(config)
        t.transcribe_segments(waveform, segments, sample_rate)
    """

    def __init__(self, config: Optional[SpeechBrainASRConfig] = None, models_dir: str = "./models"):
        self.config = config or SpeechBrainASRConfig()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from speechbrain.pretrained import (  # type: ignore
                EncoderASR,
                EncoderDecoderASR,
            )

            # Try EncoderDecoderASR first (seq2seq), fall back to EncoderASR
            try:
                self._model = EncoderDecoderASR.from_hparams(
                    source=self.config.model_id, savedir=str(self.models_dir)
                )
            except Exception:
                self._model = EncoderASR.from_hparams(
                    source=self.config.model_id, savedir=str(self.models_dir)
                )

        except Exception as e:
            print(f"[SpeechBrain] Could not load model: {e}")
            self._model = None

    def transcribe_full_audio(self, waveform: torch.Tensor, sample_rate: int = 16000) -> str:
        """Transcribe full audio waveform. Returns post-processed text (raw)."""
        self._load_model()
        if self._model is None:
            return ""

        # SpeechBrain typically expects a file path for convenience; some models accept numpy arrays
        try:
            audio_np = waveform.squeeze().cpu().numpy()
            # Many SpeechBrain models accept numpy arrays for `transcribe_batch`/`transcribe_file`
            # Use transcribe_batch for in-memory audio
            try:
                res = self._model.transcribe_batch([audio_np])
                if isinstance(res, list):
                    return str(res[0])
                return str(res)
            except Exception:
                # Fallback: write temporary file
                import tempfile

                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, audio_np.astype("float32"), sample_rate)
                    return str(self._model.transcribe_file(tmp.name))
        except Exception as e:
            print(f"[SpeechBrain] Full audio transcription failed: {e}")
            return ""

    def transcribe_segments(
        self, waveform: torch.Tensor, segments: List[SpeakerSegment], sample_rate: int = 16000
    ) -> List[TranscriptSegment]:
        """Transcribe each segment and return list of TranscriptSegment objects."""
        self._load_model()
        transcripts: List[TranscriptSegment] = []

        if self._model is None:
            return transcripts

        for seg in segments:
            start = int(seg.start * sample_rate)
            end = int(seg.end * sample_rate)
            segment_np = waveform[:, start:end].squeeze().cpu().numpy()

            if segment_np.size == 0:
                continue

            # Skip extremely short segments
            if seg.end - seg.start < 0.2:
                continue

            try:
                # prefer in-memory transcribe_batch
                res = self._model.transcribe_batch([segment_np])
                text = str(res[0]) if isinstance(res, list) else str(res)
            except Exception:
                # fallback to temporary file path
                try:
                    import tempfile

                    import soundfile as sf

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        sf.write(tmp.name, segment_np.astype("float32"), sample_rate)
                        text = str(self._model.transcribe_file(tmp.name))
                except Exception as e:
                    print(f"[SpeechBrain] Segment transcription failed: {e}")
                    text = ""

            if not text or not text.strip():
                continue

            transcripts.append(
                TranscriptSegment(
                    speaker_id=seg.speaker_id,
                    start=seg.start,
                    end=seg.end,
                    text=text.strip(),
                    confidence=getattr(seg, "confidence", 1.0),
                    is_overlap=getattr(seg, "is_overlap", False),
                )
            )

        return transcripts
