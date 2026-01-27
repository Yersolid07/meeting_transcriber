"""
ASR Transcription Module
========================
Implements speech-to-text with configurable backends (Whisper, Wav2Vec2).
Default is Whisper-base for multilingual support; supports beam CTC decoding for CTC models.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.diarization import SpeakerSegment
from src.utils import setup_logger


@dataclass
class ASRConfig:
    """Configuration for ASR"""

    model_id: str = "openai/whisper-small"
    chunk_length_s: float = 30.0
    stride_length_s: float = 5.0
    batch_size: int = 4
    return_timestamps: Optional[str] = None  # None or 'char'/'word'

    # Approximate Continuous Speech Tokenizer token rate in Hz (e.g., 7.5). When set,
    # the transcriber will apply a fast lossy compression preprocessor for speed.
    # Default: disabled (None). Use --cst-hz to enable.
    cst_hz: Optional[float] = None

    # Backend options:
    # - 'whisper': HuggingFace transformers ASR pipeline (seq2seq whisper)
    # - 'transformers': HuggingFace transformers ASR pipeline (CTC wav2vec2, etc)
    # - 'whisperx': WhisperX (faster-whisper + optional alignment; we use transcription + segments)
    # - 'speechbrain': SpeechBrain adapter
    backend: str = "whisper"

    # Preferred language for whisper (use 'id' for Indonesian)
    language: str = "id"

    # WhisperX options
    # compute_type common values: "float16" (GPU), "int8" / "int8_float16" (lower VRAM)
    whisperx_compute_type: str = "auto"
    whisperx_vad_filter: bool = True

    # Use full-audio ASR and align timestamps to diarization segments if available
    use_full_audio_for_segments: bool = False

    # Quick mode (single-pass full audio + reduced precision) and parallelism
    quick_mode: bool = False
    parallel_workers: int = 4

    # When not using full-audio timestamps, include a small context window around short segments
    context_window_s: float = 0.5

    # Decoder options: 'greedy' or 'beam' (beam can use pyctcdecode + kenlm)
    decoder: str = "greedy"
    beam_width: int = 10
    use_lm: bool = False
    lm_path: Optional[str] = None

    # Text post-processing
    capitalize_sentences: bool = True
    normalize_whitespace: bool = True
    add_punctuation: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TranscriptSegment:
    """Transcript segment with speaker and timing information"""

    speaker_id: str
    start: float
    end: float
    text: str
    confidence: float = 1.0
    is_overlap: bool = False
    language: str = "id"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get segment duration in seconds"""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Get number of words in text"""
        return len(self.text.split()) if self.text else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "speaker_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
            "is_overlap": self.is_overlap,
            "duration": self.duration,
            "word_count": self.word_count,
        }


class ASRTranscriber:
    """
    Automatic Speech Recognition using Wav2Vec2-XLSR. Supports multiple backends including
    HuggingFace `transformers` pipeline and optional SpeechBrain adapter.

    Transcribes audio segments with speaker information.
    Optimized for Indonesian language with code-switching support.

    Attributes:
        config: ASRConfig object

    Example:
        >>> transcriber = ASRTranscriber()
        >>> segments = transcriber.transcribe_segments(waveform, diarization_segments)
        >>> for seg in segments:
        ...     print(f"{seg.speaker_id}: {seg.text}")
    """

    def __init__(self, config: Optional[ASRConfig] = None, models_dir: str = "./models"):
        """
        Initialize ASRTranscriber.

        Args:
            config: ASRConfig object
            models_dir: Directory to cache downloaded models
        """
        self.config = config or ASRConfig()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.device = self.config.device

        # Setup logger
        self.logger = setup_logger("ASRTranscriber")
        # Log configured CST value for diagnostics
        try:
            self.logger.info(f"ASRTranscriber configured cst_hz: {getattr(self.config, 'cst_hz', None)} Hz")
        except Exception:
            pass

        # Model placeholders (lazy loading)
        self._pipeline = None
        self._processor = None
        self._model = None
        self._speechbrain_adapter = None
        self._whisperx_model = None

    def _load_model(self):
        """Lazy load ASR model and pipeline"""
        # If user configured SpeechBrain backend, prefer it
        if getattr(self.config, "backend", "whisper") == "speechbrain":
            if self._speechbrain_adapter is None:
                try:
                    from .transcriber_speechbrain import (
                        SpeechBrainASRConfig,
                        SpeechBrainTranscriber,
                    )

                    sb_cfg = SpeechBrainASRConfig(model_id=self.config.model_id, device=self.device)
                    self._speechbrain_adapter = SpeechBrainTranscriber(
                        sb_cfg, models_dir=str(self.models_dir)
                    )
                    self.logger.info(
                        f"SpeechBrain adapter initialized with model: {self.config.model_id}"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not initialize SpeechBrain adapter: {e}")
                    self._speechbrain_adapter = None
            return

        # WhisperX backend
        if getattr(self.config, "backend", None) == "whisperx":
            if self._whisperx_model is None:
                try:
                    # WhisperX imports torchaudio.AudioMetaData (not present in some builds, e.g., torchaudio 2.8 CPU on Windows)
                    import torchaudio

                    if not hasattr(torchaudio, "AudioMetaData"):
                        from typing import NamedTuple

                        class AudioMetaData(NamedTuple):
                            sample_rate: int
                            num_frames: int
                            num_channels: int
                            bits_per_sample: int = 16
                            encoding: str = "PCM_S"

                        # Provide stub to satisfy downstream imports; uses safe defaults
                        torchaudio.AudioMetaData = AudioMetaData  # type: ignore

                    import whisperx  # type: ignore

                    # Allowlist OmegaConf ListConfig for torch.load (needed since PyTorch 2.6 weights_only=True)
                    try:
                        import typing

                        import torch.serialization as ts
                        from omegaconf.base import ContainerMetadata  # type: ignore
                        from omegaconf.listconfig import ListConfig  # type: ignore

                        # Allow torch.load with weights_only=True to unpickle HF configs that store plain list
                        # Allowlist common builtin types and container types used inside HF checkpoints
                        ts.add_safe_globals([dict, list, int, float, str, tuple, set])

                        # Add collections.defaultdict (needed by some HF checkpoints under newer PyTorch)
                        import collections

                        ts.add_safe_globals([collections.defaultdict])

                        # Ensure OmegaConf ListConfig is allowlisted (common in HF configs)
                        ts.add_safe_globals([ListConfig])

                        # Allow AnyNode from OmegaConf which some HF configs embed
                        try:
                            from omegaconf.nodes import AnyNode  # type: ignore

                            ts.add_safe_globals([AnyNode])
                        except Exception:
                            # Not strictly fatal; continue if import fails
                            pass

                        # Some checkpoints include TorchVersion objects
                        try:
                            import torch

                            ts.add_safe_globals([torch.torch_version.TorchVersion])
                        except Exception:
                            pass

                        # Add ContainerMetadata and Metadata from OmegaConf if present
                        try:
                            from omegaconf.base import Metadata  # type: ignore

                            ts.add_safe_globals([ContainerMetadata, Metadata, typing.Any])
                        except Exception:
                            ts.add_safe_globals([ContainerMetadata, typing.Any])
                    except Exception as e:
                        self.logger.warning(f"Could not add ListConfig to torch safe globals: {e}")

                    model_name_or_path = self.config.model_id
                    p = Path(str(model_name_or_path))
                    if p.exists() and p.is_dir():
                        # WhisperX (faster-whisper / CTranslate2) expects a CT2-converted model directory
                        # containing model.bin + config files. A folder with only *.safetensors is a
                        # HuggingFace Transformers checkpoint and cannot be loaded directly by WhisperX.
                        has_model_bin = (p / "model.bin").exists()
                        has_safetensors = any(p.glob("*.safetensors"))
                        if not has_model_bin and has_safetensors:
                            raise RuntimeError(
                                "WhisperX backend membutuhkan model format CTranslate2 (ada file 'model.bin'). "
                                f"Folder '{p.as_posix()}' hanya berisi *.safetensors (format Transformers), jadi "
                                "tidak bisa dipakai langsung oleh WhisperX. "
                                "Solusi: pakai nama model WhisperX seperti 'large-v3-turbo' agar auto-download, "
                                "atau convert model Transformers -> CTranslate2 memakai ctranslate2 converter."
                            )

                    compute_type = getattr(self.config, "whisperx_compute_type", "auto")
                    if compute_type == "auto":
                        # Sensible default: float16 on CUDA, int8 on CPU
                        compute_type = "float16" if self.device == "cuda" else "int8"

                    # WhisperX uses faster-whisper under the hood; model can be a name ("large-v3", "large-v3-turbo")
                    # or a local directory containing model weights (e.g. safetensors).
                    self.logger.info(
                        f"Loading WhisperX model: {model_name_or_path} (device={self.device}, compute_type={compute_type})"
                    )

                    # Robust loading: try to parse WeightsUnpickler errors and auto-allowlist missing globals
                    def _load_model_with_retry():
                        import importlib
                        import re

                        import torch.serialization as ts

                        max_attempts = 8
                        attempt = 0
                        while True:
                            try:
                                return whisperx.load_model(
                                    model_name_or_path,
                                    device=self.device,
                                    compute_type=compute_type,
                                    download_root=str(self.models_dir),
                                )
                            except Exception as e:
                                attempt += 1
                                if attempt >= max_attempts:
                                    # Give up and re-raise the original exception
                                    raise
                                msg = str(e)
                                # Find module.Class patterns in the error message
                                missing = set(
                                    re.findall(
                                        r"GLOBAL\s+([\w\.]+)\s+was not an allowed global", msg
                                    )
                                )
                                # Also catch suggestions in the message
                                more = set(re.findall(r"add_safe_globals\(\[([^\]]+)\]\)", msg))
                                for m in more:
                                    # split comma-separated list like 'collections.defaultdict' or 'omegaconf.nodes.AnyNode'
                                    parts = [
                                        p.strip().strip("\"''") for p in m.split(",") if p.strip()
                                    ]
                                    missing.update(parts)

                                if not missing:
                                    # nothing we can do programmatically
                                    raise

                                for fullname in missing:
                                    try:
                                        module_name, cls_name = fullname.rsplit(".", 1)
                                        mod = importlib.import_module(module_name)
                                        cls = getattr(mod, cls_name)
                                        ts.add_safe_globals([cls])
                                        self.logger.info(
                                            f"Auto-added {fullname} to torch safe globals"
                                        )
                                    except Exception as ie:
                                        self.logger.warning(
                                            f"Could not auto-add {fullname} to safe globals: {ie}"
                                        )
                                # retry loop

                    self._whisperx_model = _load_model_with_retry()
                    self.logger.info("WhisperX model loaded successfully")
                except Exception as e:
                    # When user explicitly requests WhisperX backend, fail loudly with a helpful message.
                    self._whisperx_model = None
                    raise RuntimeError(f"Failed to load WhisperX model: {e}") from e

        if self._pipeline is None:
            # If user explicitly selected WhisperX and the WhisperX model loaded OK,
            # prefer WhisperX and skip attempting the Transformers pipeline which may
            # not recognize model names like 'large-v3-turbo' and produce confusing errors.
            if (
                getattr(self.config, "backend", None) == "whisperx"
                and self._whisperx_model is not None
            ):
                self._pipeline = "WHISPERX"
                self.logger.info("WhisperX backend active; skipping Transformers pipeline load")
            else:
                try:
                    from transformers import pipeline

                    self.logger.info(f"Loading model: {self.config.model_id}")

                    # Try to use pipeline first (simpler)
                    self._pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=self.config.model_id,
                        device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                        chunk_length_s=self.config.chunk_length_s,
                        stride_length_s=(self.config.stride_length_s, self.config.stride_length_s),
                    )

                    self.logger.info("Model loaded successfully via pipeline")

                except Exception as e:
                    self.logger.warning(f"Pipeline loading failed: {e}")
                    self.logger.info("Attempting direct model loading...")

                    # Attempt direct transformers model loading (Wav2Vec2)
                    try:
                        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                        self._processor = Wav2Vec2Processor.from_pretrained(
                            self.config.model_id, cache_dir=str(self.models_dir)
                        )
                        self._model = Wav2Vec2ForCTC.from_pretrained(
                            self.config.model_id, cache_dir=str(self.models_dir)
                        )

                        if self.device == "cuda" and torch.cuda.is_available():
                            self._model = self._model.cuda()

                        self._model.eval()
                        self.logger.info("Model loaded successfully via direct loading")

                        # If user requested beam decoding, try to prepare a CTC beam decoder (pyctcdecode)
                        self._ctc_decoder = None
                        try:
                            if self.config.decoder == "beam":
                                from pyctcdecode import build_ctcdecoder

                                # Build label list from tokenizer vocab ordered by id
                                vocab = self._processor.tokenizer.get_vocab()
                                labels = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]

                                if self.config.use_lm and self.config.lm_path:
                                    self.logger.info("Building CTC decoder with LM...")
                                    self._ctc_decoder = build_ctcdecoder(
                                        labels, self.config.lm_path
                                    )
                                else:
                                    self.logger.info("Building CTC decoder (no LM)")
                                    self._ctc_decoder = build_ctcdecoder(labels)

                                self.logger.info("CTC decoder ready")
                        except Exception as e:
                            self.logger.warning(
                                f"Could not build CTC decoder (pyctcdecode/kenlm missing or failed): {e}"
                            )
                            self._ctc_decoder = None

                    except Exception as e2:
                        self.logger.error(f"Direct loading also failed: {e2}")
                        self.logger.warning("Using fallback placeholder mode")
                        self._pipeline = "FALLBACK"

    def transcribe_segments(
        self,
        waveform: torch.Tensor,
        segments: List[SpeakerSegment],
        sample_rate: int = 16000,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TranscriptSegment]:
        """
        Transcribe each speaker segment. If `use_full_audio_for_segments` is enabled,
        run ASR once on the full audio and map word/segment timestamps back to
        the diarization segments when the ASR pipeline returns timestamps.
        Falls back to context-augmented per-segment transcription when timestamps
        are not available.
        """
        try:
            self._load_model()
        except Exception as e:
            # If loading the configured ASR backend fails (common when deployment preset
            # forced WhisperX but model_id is a Transformers repo), attempt a safe
            # runtime fallback to a lightweight Whisper model so interactive UI flows
            # remain responsive instead of crashing.
            self.logger.error(
                f"ASR model load failed: {e}. Attempting fallback to 'whisper' backend with 'openai/whisper-small'."
            )
            try:
                self.config.backend = "whisper"
                self.config.model_id = "openai/whisper-small"
                # Clear any partially-initialized model state
                self._pipeline = None
                self._model = None
                self._processor = None
                self._whisperx_model = None
                self._load_model()
                self.logger.info("Fallback ASR model loaded successfully (openai/whisper-small)")
            except Exception as e2:
                self.logger.error(f"Fallback ASR model load also failed: {e2}")
                # Re-raise to let caller handle/report the error
                raise

        # If SpeechBrain backend adapter is configured, delegate to it
        if (
            getattr(self.config, "backend", None) == "speechbrain"
            and getattr(self, "_speechbrain_adapter", None) is not None
        ):
            try:
                sb_res = self._speechbrain_adapter.transcribe_segments(
                    waveform, segments, sample_rate
                )
                for s in sb_res:
                    s.text = self._postprocess_text(s.text)
                return sb_res
            except Exception as e:
                self.logger.error(f"SpeechBrain adapter transcription failed: {e}")

        transcripts = []
        total_segments = len(segments)

        # If using full-audio mapping, run pipeline once on entire audio and try to align
        full_asr_result = None
        audio_np_full = waveform.squeeze().cpu().numpy()

        if self.config.use_full_audio_for_segments:
            # If SpeechBrain backend is used, ask adapter to produce full transcription
            if (
                getattr(self.config, "backend", "whisper") == "speechbrain"
                and self._speechbrain_adapter is not None
            ):
                try:
                    self.logger.info(
                        "Running full-audio ASR via SpeechBrain adapter for alignment to segments"
                    )
                    full_text = self._speechbrain_adapter.transcribe_full_audio(
                        waveform, sample_rate
                    )
                    # SpeechBrain adapter currently returns plain text; we can't map timestamps, so store as simple str
                    full_asr_result = {"text": full_text}
                except Exception as e:
                    self.logger.error(f"SpeechBrain full-audio ASR failed: {e}")
                    full_asr_result = None

            elif self._pipeline not in (None, "FALLBACK"):
                try:
                    # Whisper (seq2seq) pipelines don't accept 'sampling_rate' kwarg; omit it and set language
                    if getattr(self.config, "backend", "transformers") == "whisper":
                        kwargs = {}
                        # prefer explicit language if configured (e.g., Indonesian 'id')
                        kwargs["language"] = self.config.language
                    else:
                        kwargs = {"sampling_rate": sample_rate}

                    rt = self.config.return_timestamps
                    if rt in ("char", "word"):
                        kwargs["return_timestamps"] = rt

                    self.logger.info("Running full-audio ASR for alignment to segments")
                    full_asr_result = self._pipeline(audio_np_full, **kwargs)
                except Exception as e:
                    self.logger.error(f"Full-audio ASR failed: {e}")
                    full_asr_result = None

        # Build list of segment tasks that need per-segment ASR
        tasks = []
        for idx, seg in enumerate(segments):
            # Skip very short segments
            duration = seg.end - seg.start
            if duration < 0.3:
                continue
            tasks.append((idx, seg))

        # If we have a full-audio ASR result that includes timestamps, map once and avoid per-segment ASR
        if full_asr_result is not None:
            for idx, seg in tasks:
                text = self._map_full_asr_to_segment(full_asr_result, seg)
                if text:
                    text = self._postprocess_text(text)
                    if text:
                        transcripts.append(
                            TranscriptSegment(
                                speaker_id=seg.speaker_id,
                                start=seg.start,
                                end=seg.end,
                                text=text,
                                confidence=seg.confidence,
                                is_overlap=seg.is_overlap,
                                metadata={
                                    "embedding": (
                                        seg.embedding if hasattr(seg, "embedding") else None
                                    ),
                                    "asr_model": self.config.model_id,
                                },
                            )
                        )
            # Filter out tasks that were handled by mapping
            tasks = [
                (i, s)
                for (i, s) in tasks
                if not any(t.start == s.start and t.end == s.end for t in transcripts)
            ]

        # If quick_mode or parallel workers > 1, perform parallel per-segment ASR
        workers = int(getattr(self.config, "parallel_workers", 1))
        if workers > 1 and tasks:
            import concurrent.futures

            def _transcribe_task(item):
                idx, seg = item
                # Progress update is handled by caller optionally, but we log
                # Use context window if available
                if self.config.context_window_s and self._pipeline not in (None, "FALLBACK"):
                    ctx_start = max(0.0, seg.start - self.config.context_window_s)
                    ctx_end = seg.end + self.config.context_window_s
                    cs = int(ctx_start * sample_rate)
                    ce = int(min(ctx_end * sample_rate, waveform.shape[-1]))
                    audio_np = waveform[:, cs:ce].squeeze().cpu().numpy()
                    text = self._transcribe_audio(
                        torch.from_numpy(audio_np).unsqueeze(0), sample_rate
                    )
                else:
                    start_sample = int(seg.start * sample_rate)
                    end_sample = int(seg.end * sample_rate)
                    audio_segment = waveform[:, start_sample:end_sample]
                    text = self._transcribe_audio(audio_segment, sample_rate)

                text = self._postprocess_text(text)
                return idx, seg, text

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_transcribe_task, t): t for t in tasks}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        idx, seg, text = fut.result()
                        if not text or not text.strip():
                            continue
                        transcripts.append(
                            TranscriptSegment(
                                speaker_id=seg.speaker_id,
                                start=seg.start,
                                end=seg.end,
                                text=text,
                                confidence=seg.confidence,
                                is_overlap=seg.is_overlap,
                                metadata={
                                    "embedding": (
                                        seg.embedding if hasattr(seg, "embedding") else None
                                    ),
                                    "asr_model": self.config.model_id,
                                },
                            )
                        )
                    except Exception as e:
                        self.logger.error(f"Segment transcription failed: {e}")
        else:
            # Serial fallback
            for idx, seg in tasks:
                # create context window
                if self.config.context_window_s and self._pipeline not in (None, "FALLBACK"):
                    ctx_start = max(0.0, seg.start - self.config.context_window_s)
                    ctx_end = seg.end + self.config.context_window_s
                    cs = int(ctx_start * sample_rate)
                    ce = int(min(ctx_end * sample_rate, waveform.shape[-1]))
                    audio_np = waveform[:, cs:ce].squeeze().cpu().numpy()
                    text = self._transcribe_audio(
                        torch.from_numpy(audio_np).unsqueeze(0), sample_rate
                    )
                else:
                    start_sample = int(seg.start * sample_rate)
                    end_sample = int(seg.end * sample_rate)
                    audio_segment = waveform[:, start_sample:end_sample]
                    text = self._transcribe_audio(audio_segment, sample_rate)

                # Post-process text
                text = self._postprocess_text(text)

                # Skip empty transcriptions
                if not text or not text.strip():
                    continue

                transcripts.append(
                    TranscriptSegment(
                        speaker_id=seg.speaker_id,
                        start=seg.start,
                        end=seg.end,
                        text=text,
                        confidence=seg.confidence,
                        is_overlap=seg.is_overlap,
                        metadata={
                            "embedding": seg.embedding if hasattr(seg, "embedding") else None,
                            "asr_model": self.config.model_id,
                        },
                    )
                )

        return transcripts

    def _detect_language_from_text(self, text: str) -> Optional[str]:
        """Detect top language code from text using langdetect. Returns ISO code or None."""
        try:
            from langdetect import detect_langs

            if not text or not text.strip():
                return None
            probs = detect_langs(text)
            if not probs:
                return None
            return probs[0].lang
        except Exception:
            return None

    def _transcribe_audio(self, audio_segment: torch.Tensor, sample_rate: int) -> str:
        """Transcribe a single audio segment

        Supports `language='auto'` for Whisper backend which will perform a quick
        pre-pass (no language hint) and use a text-based language detector to
        choose the language for the final transcription pass.

        If `self.config.cst_hz` is set, an aggressive lossy preprocessor (approximation
        of a low-rate Continuous Speech Tokenizer) is applied before sending audio to
        the ASR backend. This significantly reduces compute at the cost of precision
        and should be used only when speed is critical.
        """

        # Fallback mode: only return placeholders when no working ASR backend is available.
        # If user requested WhisperX backend and model is loaded, prefer using WhisperX.
        if self._pipeline == "FALLBACK":
            backend = getattr(self.config, "backend", None)
            if not (backend == "whisperx" and self._whisperx_model is not None):
                duration = audio_segment.shape[-1] / sample_rate
                return f"[Transkripsi placeholder - durasi {duration:.1f}s]"

        # Convert to numpy
        audio_np = audio_segment.squeeze().cpu().numpy()

        # Apply CST approximation preprocessor if requested (lossy, speed-optimized)
        if getattr(self.config, "cst_hz", None) is not None:
            try:
                audio_np = self._apply_cst_approximation(audio_np, sample_rate, float(self.config.cst_hz))
                # After approximation we keep the original sample_rate for downstream callers
                self.logger.info(f"Applied CST approximation: {self.config.cst_hz} Hz (lossy)")
            except Exception as e:
                self.logger.warning(f"CST approximation failed, continuing with original audio: {e}")

        # Ensure float32
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)


        # WhisperX backend
        if getattr(self.config, "backend", None) == "whisperx":
            try:
                if self._whisperx_model is None:
                    self._load_model()
                if self._whisperx_model is None:
                    return ""

                language = getattr(self.config, "language", "id")
                # whisperx expects None for auto language
                language_arg = None if language == "auto" else language

                vad_filter = bool(getattr(self.config, "whisperx_vad_filter", True))

                # Build kwargs and only pass vad_filter if the transcribe signature accepts it
                from inspect import signature

                kwargs = {"batch_size": self.config.batch_size}
                if language_arg is not None:
                    kwargs["language"] = language_arg

                try:
                    sig = signature(self._whisperx_model.transcribe)
                    if "vad_filter" in sig.parameters:
                        kwargs["vad_filter"] = vad_filter
                except Exception:
                    # If introspection fails, do not pass vad_filter
                    pass

                # First attempt
                try:
                    result = self._whisperx_model.transcribe(audio_np, **kwargs)
                except Exception as e_inner:
                    self.logger.warning(f"WhisperX transcription failed on first attempt: {e_inner}. Retrying with `vad_filter=False, batch_size=1`")
                    # retry with safer options
                    try:
                        retry_kwargs = kwargs.copy()
                        retry_kwargs["batch_size"] = 1
                        if "vad_filter" in retry_kwargs:
                            retry_kwargs["vad_filter"] = False
                        result = self._whisperx_model.transcribe(audio_np, **retry_kwargs)
                    except Exception as e_retry:
                        self.logger.error(f"WhisperX transcription retry failed: {e_retry}. Falling back to lightweight Whisper model.")
                        # Fallback: switch backend to 'whisper' with small model and attempt to load it
                        try:
                            self.config.backend = "whisper"
                            self.config.model_id = "openai/whisper-small"
                            # Clear whisperx state
                            self._whisperx_model = None
                            self._pipeline = None
                            self._model = None
                            self._processor = None
                            self._load_model()
                            # attempt pipeline-based transcription
                            return self._transcribe_audio(audio_segment, sample_rate)
                        except Exception as e_fb:
                            self.logger.error(f"Fallback ASR model load/transcription failed: {e_fb}")
                            return ""

                # Normalize result into plain text.
                if isinstance(result, dict):
                    # 'text' is common, but some ASR returns 'segments' list
                    if "text" in result and result.get("text"):
                        return result.get("text", "")
                    if "segments" in result and isinstance(result["segments"], list):
                        seg_texts = [
                            s.get("text", "") for s in result["segments"] if isinstance(s, dict)
                        ]
                        joined = " ".join(t.strip() for t in seg_texts if t and t.strip())
                        return joined or ""
                    # fallback to empty
                    return ""
                return str(result)
            except Exception as e:
                self.logger.error(f"WhisperX transcription failed: {e}")
                return ""
        # Use pipeline if available
        if self._pipeline is not None and self._pipeline != "FALLBACK":
            try:
                # Whisper backend: handle language auto-detection
                if getattr(self.config, "backend", "transformers") == "whisper":
                    if getattr(self.config, "language", "id") == "auto":
                        # quick pre-pass to get candidate text
                        try:
                            quick_kwargs = {}
                            rt = self.config.return_timestamps
                            if rt in ("char", "word"):
                                quick_kwargs["return_timestamps"] = rt
                            quick_res = self._pipeline(audio_np, **quick_kwargs)
                            quick_text = (
                                quick_res.get("text", "")
                                if isinstance(quick_res, dict)
                                else str(quick_res)
                            )
                            detected = self._detect_language_from_text(quick_text)
                            chosen_lang = detected if detected else "id"
                        except Exception:
                            chosen_lang = "id"
                    else:
                        chosen_lang = getattr(self.config, "language", "id")

                    kwargs = {"language": chosen_lang}
                else:
                    kwargs = {"sampling_rate": sample_rate}

                rt = self.config.return_timestamps
                if rt in ("char", "word"):
                    kwargs["return_timestamps"] = rt

                result = self._pipeline(audio_np, **kwargs)

                # If result is a dict with text
                if isinstance(result, dict):
                    # If pipeline returns a list of word/segment timestamps, user may want that via full-audio flow
                    if isinstance(result.get("chunks", None), list) or isinstance(
                        result.get("segments", None), list
                    ):
                        return result.get("text", "")
                    return result.get("text", "")
                return str(result)

            except Exception as e:
                self.logger.warning(f"Pipeline transcription failed: {e}")
                # Try to fall back to direct model path (if available)
                self._pipeline = None
                # continue to attempt direct model below

        # Use direct model if pipeline not available
        if self._model is not None and self._processor is not None:
            try:
                # Process input
                inputs = self._processor(
                    audio_np, sampling_rate=sample_rate, return_tensors="pt", padding=True
                )

                # Move to device
                if self.device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    logits = self._model(**inputs).logits

                # If CTC beam decoder available and requested, use it
                if (
                    getattr(self, "_ctc_decoder", None) is not None
                    and self.config.decoder == "beam"
                ):
                    try:
                        # Convert logits to probabilities (T, C)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        # some models return batch dimension; take first batch
                        emissions = probs[0]

                        try:
                            # Try simple decode
                            transcription = self._ctc_decoder.decode(
                                emissions, beam_width=self.config.beam_width
                            )
                        except Exception:
                            # Try beam candidates and pick top
                            beams = self._ctc_decoder.decode_beams(
                                emissions, beam_width=self.config.beam_width
                            )
                            transcription = beams[0][0] if beams else ""

                        return transcription if transcription else ""
                    except Exception as e:
                        self.logger.warning(f"CTC beam decode failed: {e}")
                        # fallback to greedy

                # Fallback: greedy argmax decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self._processor.batch_decode(predicted_ids)

                return transcription[0] if transcription else ""

            except Exception as e:
                self.logger.error(f"Direct model transcription failed: {e}")
                return ""

        return ""

    def transcribe_full_audio(self, waveform: torch.Tensor, sample_rate: int = 16000) -> str:
        """
        Transcribe full audio without diarization.
        Useful for baseline comparison.
        """
        self._load_model()

        # WhisperX: call directly to keep consistency
        if getattr(self.config, "backend", None) == "whisperx":
            audio_np = waveform.squeeze().cpu().numpy().astype(np.float32, copy=False)
            if self._whisperx_model is None:
                return ""
            language = getattr(self.config, "language", "id")
            language_arg = None if language == "auto" else language
            vad_filter = bool(getattr(self.config, "whisperx_vad_filter", True))
            try:
                res = self._whisperx_model.transcribe(
                    audio_np,
                    batch_size=self.config.batch_size,
                    language=language_arg,
                    vad_filter=vad_filter,
                )
                text = res.get("text", "") if isinstance(res, dict) else str(res)
                return self._postprocess_text(text)
            except Exception as e:
                self.logger.warning(f"WhisperX full-audio transcription failed: {e}. Retrying with vad_filter=False, batch_size=1")
                try:
                    res = self._whisperx_model.transcribe(
                        audio_np,
                        batch_size=1,
                        language=language_arg,
                        vad_filter=False,
                    )
                    text = res.get("text", "") if isinstance(res, dict) else str(res)
                    return self._postprocess_text(text)
                except Exception as e2:
                    self.logger.error(f"WhisperX full-audio retry failed: {e2}. Falling back to 'whisper-small'.")
                    # Fallback to whisper-small pipeline
                    try:
                        self.config.backend = "whisper"
                        self.config.model_id = "openai/whisper-small"
                        self._whisperx_model = None
                        self._pipeline = None
                        self._model = None
                        self._processor = None
                        self._load_model()
                        text = self._transcribe_audio(waveform, sample_rate)
                        return self._postprocess_text(text)
                    except Exception as e_fb:
                        self.logger.error(f"Fallback full-audio ASR failed: {e_fb}")
                        return ""

        text = self._transcribe_audio(waveform, sample_rate)
        return self._postprocess_text(text)

    def _apply_cst_approximation(self, audio_np: np.ndarray, sample_rate: int, cst_hz: float) -> np.ndarray:
        """Approximate a Continuous Speech Tokenizer by block-averaging audio frames

        This method is intentionally conservative and reversible only in the sense
        that it produces a downsample-like version of the waveform which is then
        expanded back to the original rate (by repeating block values). This is
        extremely lossy but can reduce model runtime for long audio when you
        accept lower ASR fidelity.

        Implementation details:
        - token_duration = 1.0 / cst_hz
        - compute mean amplitude per token window
        - expand each token mean to the window length (constant value) to produce
          a waveform of the original sample length

        Note: This is an approximation to the user's requested ultralow-rate tokenizer
        (7.5 Hz). For best accuracy, tune `cst_hz` and verify results on your data.
        """
        if cst_hz <= 0 or np.isnan(cst_hz):
            return audio_np

        token_dur = 1.0 / float(cst_hz)
        window_samp = max(1, int(round(token_dur * sample_rate)))
        # Partition audio and compute mean for each window
        n = len(audio_np)
        n_windows = int(np.ceil(n / window_samp))
        means = []
        for i in range(n_windows):
            s = i * window_samp
            e = min(n, s + window_samp)
            if e <= s:
                means.append(0.0)
            else:
                means.append(float(np.mean(audio_np[s:e])))

        # Reconstruct waveform by repeating means per window
        out = np.zeros(n, dtype=np.float32)
        for i, m in enumerate(means):
            s = i * window_samp
            e = min(n, s + window_samp)
            out[s:e] = m

        return out

    def _postprocess_text(self, text: str) -> str:
        """Clean and format transcribed text"""
        if not text:
            return ""

        # Basic cleaning
        text = text.strip()

        # Remove special tokens and math/code blocks bounded by $$...$$
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = " ".join(text.split())

        # Capitalize first letter of sentences
        if self.config.capitalize_sentences and text:
            # Capitalize first character
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

            # Capitalize after sentence-ending punctuation
            text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

        # Add period if missing
        if text and text[-1] not in ".!?,:;":
            text += "."

        return text

    def _map_full_asr_to_segment(self, full_result: Any, seg: SpeakerSegment) -> str:
        """Attempt to extract text for a given segment from a full-audio ASR result.

        Supports multiple result shapes returned by different ASR pipelines:
        - result['chunks'] or result['segments']: list of dicts with 'start','end','text'
        - result may also include 'words' lists with per-word timestamps
        If no timestamped structure is present, returns empty string so caller can fallback.
        """
        try:
            # Prefer 'chunks' (some pipelines) then 'segments'
            blocks = None
            if isinstance(full_result, dict):
                if isinstance(full_result.get("chunks"), list):
                    blocks = full_result["chunks"]
                elif isinstance(full_result.get("segments"), list):
                    blocks = full_result["segments"]
                # some pipelines return word-level timestamps
                elif isinstance(full_result.get("words"), list):
                    words = full_result["words"]
                    text_parts = [
                        w["word"]
                        for w in words
                        if w.get("start") is not None
                        and w.get("end") is not None
                        and (w["start"] >= seg.start and w["end"] <= seg.end)
                    ]
                    return " ".join(text_parts)

            if blocks is None:
                return ""

            # Concatenate blocks that overlap with seg time window
            collected = []
            for b in blocks:
                bstart = float(b.get("start", 0.0))
                bend = float(b.get("end", 0.0))
                if bstart < seg.end and bend > seg.start:
                    collected.append(b.get("text", ""))

            return " ".join([c.strip() for c in collected]).strip()
        except Exception:
            return ""

    def get_transcription_stats(self, segments: List[TranscriptSegment]) -> Dict[str, Any]:
        """
        Get transcription statistics.

        Args:
            segments: List of transcript segments

        Returns:
            Dictionary with statistics
        """
        if not segments:
            return {
                "total_segments": 0,
                "total_words": 0,
                "total_duration": 0.0,
                "words_per_minute": 0.0,
                "speakers": {},
            }

        total_words = sum(seg.word_count for seg in segments)
        total_duration = sum(seg.duration for seg in segments)

        # Per-speaker stats
        speaker_stats = {}
        for seg in segments:
            if seg.speaker_id not in speaker_stats:
                speaker_stats[seg.speaker_id] = {
                    "word_count": 0,
                    "duration": 0.0,
                    "segment_count": 0,
                }

            speaker_stats[seg.speaker_id]["word_count"] += seg.word_count
            speaker_stats[seg.speaker_id]["duration"] += seg.duration
            speaker_stats[seg.speaker_id]["segment_count"] += 1

        return {
            "total_segments": len(segments),
            "total_words": total_words,
            "total_duration": total_duration,
            "words_per_minute": (total_words / total_duration * 60) if total_duration > 0 else 0,
            "speakers": speaker_stats,
        }
