"""
Main Pipeline Module
====================
Orchestrates all components for end-to-end meeting transcription.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.audio_processor import AudioConfig, AudioProcessor
from src.diarization import DiarizationConfig, SpeakerDiarizer, SpeakerSegment
from src.document_generator import DocumentGenerator, MeetingMetadata
from src.evaluator import EvaluationResult, Evaluator
from src.summarizer import BERTSummarizer, MeetingSummary, SummarizationConfig
from src.transcriber import ASRConfig, ASRTranscriber, TranscriptSegment

# Optional speechbrain adapter
try:
    from src.transcriber_speechbrain import (  # type: ignore
        SpeechBrainASRConfig,
        SpeechBrainTranscriber,
    )
except Exception:
    SpeechBrainTranscriber = None
    SpeechBrainASRConfig = None
from src.utils import (
    Timer,
    ensure_dir,
    format_duration,
    sanitize_filename,
    save_json,
    setup_logger,
)


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline"""

    # Paths
    models_dir: str = "./models"
    output_dir: str = "./data/output"
    cache_dir: str = "./cache"

    # Audio settings
    sample_rate: int = 16000

    # Diarization settings
    num_speakers: Optional[int] = None
    min_speech_duration: float = 0.3
    # Target speaker enforcement (convenience wrapper for DiarizationConfig.target_num_speakers)
    target_speakers: Optional[int] = None

    # ASR settings
    asr_model_id: str = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
    asr_backend: str = "whisper"  # whisper|transformers|whisperx|speechbrain
    asr_language: str = "id"
    whisperx_compute_type: str = "auto"
    whisperx_vad_filter: bool = True

    # Summarization settings
    num_summary_sentences: int = 5

    # Device
    device: str = "auto"

    # Flags
    save_intermediate: bool = True
    verbose: bool = True

    # Performance options
    fast_mode: bool = False  # reduce accuracy for speed
    quick_asr: bool = False  # use lightweight ASR where possible
    embedding_cache: bool = True  # cache diarization embeddings to disk

    # Preset mode (deployment = recommended default for production: WhisperX large-v3-turbo int8)
    preset: str = "deployment"  # choices: deployment|balanced|fast|accurate

    # Allow explicit override for ASR parallel workers (None = auto)
    asr_parallel_workers: Optional[int] = None  # override for per-segment ASR parallelism

    # Optional speaker mapping & diarization tuning
    speaker_map_path: Optional[str] = None
    tune_diarization: bool = False

    # Target speaker convenience
    target_speakers: Optional[int] = None

    def __post_init__(self):
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create directories
        ensure_dir(self.models_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.cache_dir)


@dataclass
class PipelineResult:
    """Complete result from pipeline processing"""

    # Input info
    audio_path: str
    audio_duration: float

    # Processing info
    num_speakers: int
    num_segments: int
    total_words: int
    processing_time: float

    # Outputs
    segments: List[Dict[str, Any]]
    transcript_text: str
    summary: Dict[str, Any]
    document_path: str

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "audio_path": self.audio_path,
            "audio_duration": self.audio_duration,
            "num_speakers": self.num_speakers,
            "num_segments": self.num_segments,
            "total_words": self.total_words,
            "processing_time": self.processing_time,
            "transcript_text": self.transcript_text,
            "summary": self.summary,
            "document_path": self.document_path,
            "metadata": self.metadata,
        }

    def save(self, filepath: str):
        """Save result to JSON file"""
        save_json(self.to_dict(), filepath)


class MeetingTranscriberPipeline:
    """
    End-to-end pipeline for automatic meeting transcription.

    Pipeline Flow:
        1. Audio Loading & Preprocessing
        2. Speaker Diarization (VAD + Embedding + Clustering)
        3. ASR Transcription (per speaker segment)
        4. BERT Summarization (extractive)
        5. Document Generation (.docx)

    Attributes:
        config: PipelineConfig object

    Example:
        >>> pipeline = MeetingTranscriberPipeline()
        >>> result = pipeline.process("meeting.wav", title="Team Meeting")
        >>> print(f"Document saved: {result.document_path}")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: PipelineConfig object (uses defaults if None)
        """
        self.config = config or PipelineConfig()

        # Setup logger
        self.logger = setup_logger(
            "MeetingTranscriber",
            log_file=(
                os.path.join(self.config.cache_dir, "pipeline.log")
                if self.config.save_intermediate
                else None
            ),
        )

        # Component placeholders (lazy loading)
        self._audio_processor = None
        self._diarizer = None
        self._transcriber = None
        self._summarizer = None
        self._doc_generator = None
        self._evaluator = None

        # Processing state
        self._waveform = None
        self._sample_rate = None
        self._diarization_segments = None
        self._transcript_segments = None
        self._summary = None

        if self.config.verbose:
            self._log(f"Pipeline initialized with device: {self.config.device}")

    # =========================================================================
    # Properties (Lazy Loading)
    # =========================================================================

    @property
    def audio_processor(self) -> AudioProcessor:
        """Get audio processor (lazy loaded)"""
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor(
                AudioConfig(sample_rate=self.config.sample_rate, mono=True, normalize=True)
            )
        return self._audio_processor

    @property
    def diarizer(self) -> SpeakerDiarizer:
        """Get diarizer (lazy loaded)"""
        if self._diarizer is None:
            dz_cfg = DiarizationConfig(
                min_speech_duration=self.config.min_speech_duration,
                device=self.config.device,
            )
            # If pipeline has target_speakers configured, propagate to diarizer config
            if getattr(self.config, "target_speakers", None) is not None:
                dz_cfg.target_num_speakers = int(self.config.target_speakers)
            self._diarizer = SpeakerDiarizer(config=dz_cfg, models_dir=self.config.models_dir)
        return self._diarizer

    @property
    def transcriber(self) -> ASRTranscriber:
        """Get transcriber (lazy loaded)"""
        if self._transcriber is None:
            # Instantiate ASR transcriber; if configured to use SpeechBrain backend prefer adapter
            asr_cfg = ASRConfig(
                model_id=self.config.asr_model_id,
                device=self.config.device,
                backend=getattr(self.config, "asr_backend", "whisper"),
                language=getattr(self.config, "asr_language", "id"),
                whisperx_compute_type=getattr(self.config, "whisperx_compute_type", "auto"),
                whisperx_vad_filter=bool(getattr(self.config, "whisperx_vad_filter", True)),
            )

            # Apply preset defaults (deployment/balanced/fast/accurate)
            preset = getattr(self.config, "preset", None)
            if preset == "deployment":
                # Deployment preset: prefer WhisperX large-v3-turbo (int8 on CPU), full-audio mapping, tuned parallelism
                asr_cfg.backend = "whisperx"

                # If user did not explicitly provide a WhisperX-compatible model (e.g. the
                # configured model contains 'wav2vec' or is an existing TF checkpoint),
                # override to a known WhisperX-compatible model id. This avoids trying to
                # load a Transformers checkpoint with WhisperX which expects CTranslate2 format
                # (contains 'model.bin').
                user_model = getattr(self.config, "asr_model_id", "") or ""
                user_model_l = user_model.lower()
                if (
                    (not user_model_l)
                    or ("wav2vec" in user_model_l)
                    or user_model_l.startswith("models/")
                ):
                    asr_cfg.model_id = "large-v3-turbo"
                    self._log(
                        "Preset 'deployment' selected: overriding ASR model to 'large-v3-turbo' for WhisperX compatibility."
                    )
                else:
                    asr_cfg.model_id = user_model

                asr_cfg.use_full_audio_for_segments = True
                asr_cfg.whisperx_compute_type = (
                    getattr(self.config, "whisperx_compute_type", "int8") or "int8"
                )
                try:
                    import os

                    asr_cfg.parallel_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
                except Exception:
                    pass
            elif getattr(self.config, "quick_asr", False):
                # Enable quick ASR settings if pipeline asked for it
                try:
                    # Use Whisper small model for seq2seq speed, unless user forced whisperx
                    if getattr(self.config, "asr_backend", "whisperx") == "whisperx":
                        asr_cfg.model_id = "openai/whisper-small"
                        asr_cfg.backend = "whisper"
                    else:
                        asr_cfg.model_id = "openai/whisper-small"
                    # Use full-audio single-pass mapping for speed
                    asr_cfg.use_full_audio_for_segments = True
                    # Increase parallel workers conservatively
                    import os

                    asr_cfg.parallel_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
                except Exception:
                    pass

            # Allow explicit override from pipeline config
            if getattr(self.config, "asr_parallel_workers", None) is not None:
                try:
                    asr_cfg.parallel_workers = int(self.config.asr_parallel_workers)
                except Exception:
                    pass

            # Allow explicit override from pipeline config
            if getattr(self.config, "asr_parallel_workers", None) is not None:
                try:
                    asr_cfg.parallel_workers = int(self.config.asr_parallel_workers)
                except Exception:
                    pass
            if (
                getattr(self.config, "asr_backend", None) == "speechbrain"
                and SpeechBrainTranscriber is not None
            ):
                # Create SpeechBrain adapter and wrap it with existing ASRTranscriber interface by setting backend
                self._transcriber = ASRTranscriber(
                    config=asr_cfg, models_dir=self.config.models_dir
                )
                self._transcriber.config.backend = "speechbrain"
            else:
                self._transcriber = ASRTranscriber(
                    config=asr_cfg,
                    models_dir=self.config.models_dir,
                )
        return self._transcriber

    @property
    def summarizer(self) -> BERTSummarizer:
        """Get summarizer (lazy loaded)"""
        if self._summarizer is None:
            self._summarizer = BERTSummarizer(
                config=SummarizationConfig(num_sentences=self.config.num_summary_sentences)
            )
        return self._summarizer

    @property
    def doc_generator(self) -> DocumentGenerator:
        """Get document generator (lazy loaded)"""
        if self._doc_generator is None:
            self._doc_generator = DocumentGenerator(output_dir=self.config.output_dir)
        return self._doc_generator

    @property
    def evaluator(self) -> Evaluator:
        """Get evaluator (lazy loaded)"""
        if self._evaluator is None:
            self._evaluator = Evaluator(output_dir=self.config.output_dir)
        return self._evaluator

    # =========================================================================
    # Main Processing Methods
    # =========================================================================

    def process(
        self,
        audio_path: str,
        title: str = "Notulensi Rapat",
        date: Optional[str] = None,
        location: str = "",
        num_speakers: Optional[int] = None,
        output_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        """
        Process audio file through complete pipeline.

        Args:
            audio_path: Path to audio file
            title: Meeting title for document
            date: Meeting date (default: today)
            location: Meeting location/platform
            num_speakers: Known number of speakers (auto-detect if None)
            output_filename: Output .docx filename (auto-generated if None)
            progress_callback: Callback function(step_name, current, total)

        Returns:
            PipelineResult with all outputs and metadata
        """
        start_time = time.time()

        def update_progress(step: str, current: int, total: int):
            if progress_callback:
                progress_callback(step, current, total)
            if self.config.verbose:
                self._log(f"Step {current}/{total}: {step}")

        self._log("=" * 60)
        self._log(f"Processing: {audio_path}")
        self._log("=" * 60)

        # =====================================================================
        # Step 1: Load and preprocess audio
        # =====================================================================
        update_progress("Loading audio", 1, 5)

        with Timer("Audio loading"):
            self._waveform, self._sample_rate = self.audio_processor.load_audio(audio_path)

        duration = self.audio_processor.get_duration(self._waveform, self._sample_rate)
        self._log(f"Audio loaded: {format_duration(duration)} ({duration:.2f}s)")

        # Validate audio duration
        max_duration_minutes = getattr(self.config, "max_duration_minutes", 60)
        max_duration_seconds = max_duration_minutes * 60
        if duration > max_duration_seconds:
            error_msg = (
                f"Audio duration ({duration:.1f}s) exceeds maximum allowed duration "
                f"({max_duration_seconds}s / {max_duration_minutes} minutes). "
                "Please split the audio or increase max_duration_minutes in config."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # =====================================================================
        # Step 2: Speaker diarization (optionally tune hyperparameters first)
        # =====================================================================
        update_progress("Speaker diarization", 2, 5)

        # Optional automatic tuning step
        if getattr(self.config, "tune_diarization", False):
            self._log("Tuning diarization hyperparameters...")
            try:
                self.diarizer.auto_tune(
                    self._waveform, self._sample_rate, num_speakers=num_speakers
                )
            except Exception as e:
                self._log(f"Diarization tuning failed (continuing with defaults): {e}")

        with Timer("Diarization"):
            # Pass cache directory and audio id so diarizer can cache embeddings
            self._diarization_segments = self.diarizer.process(
                self._waveform,
                self._sample_rate,
                num_speakers=num_speakers or self.config.num_speakers,
                cache_dir=self.config.cache_dir,
                audio_id=Path(audio_path).stem,
                fast_mode=self.config.fast_mode,
            )

        unique_speakers = set(seg.speaker_id for seg in self._diarization_segments)
        self._log(
            f"Found {len(unique_speakers)} speakers, {len(self._diarization_segments)} segments"
        )

        # =====================================================================
        # Step 3: ASR transcription
        # =====================================================================
        update_progress("Transcribing speech", 3, 5)

        with Timer("Transcription"):
            self._transcript_segments = self.transcriber.transcribe_segments(
                self._waveform, self._diarization_segments, self._sample_rate
            )

        total_words = sum(seg.word_count for seg in self._transcript_segments)
        self._log(f"Transcribed {len(self._transcript_segments)} segments, ~{total_words} words")

        # =====================================================================
        # Step 4: BERT summarization
        # =====================================================================
        update_progress("Generating summary", 4, 5)

        # If a manual speaker map was provided via config, apply it so summarizer sees mapped names
        if getattr(self.config, "speaker_map_path", None):
            try:
                speaker_map = self._load_speaker_map(self.config.speaker_map_path)
                self._apply_speaker_map(speaker_map)
            except Exception as e:
                self._log(f"Failed to load/apply speaker map: {e}")

        with Timer("Summarization"):
            self._summary = self.summarizer.summarize(self._transcript_segments)

        self._log(f"Generated summary with {len(self._summary.key_points)} key points")

        # =====================================================================
        # Step 5: Generate document
        # =====================================================================
        update_progress("Generating document", 5, 5)

        # Prepare metadata
        participants = list(unique_speakers)
        # If speaker map provided, map participants accordingly
        if getattr(self.config, "speaker_map_path", None):
            try:
                speaker_map = self._load_speaker_map(self.config.speaker_map_path)
                participants = [speaker_map.get(p, p) for p in participants]
            except Exception:
                pass

        metadata = MeetingMetadata(
            title=title,
            date=date or datetime.now().strftime("%d %B %Y"),
            time=datetime.now().strftime("%H:%M"),
            location=location,
            duration=format_duration(duration),
            participants=participants,
        )

        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = sanitize_filename(title)[:30]
            output_filename = f"notulensi_{safe_title}_{timestamp}.docx"

        with Timer("Document generation"):
            doc_path = self.doc_generator.generate(
                metadata=metadata,
                summary=self._summary,
                transcript=self._transcript_segments,
                output_filename=output_filename,
            )

        self._log(f"Document saved: {doc_path}")

        # =====================================================================
        # Save intermediate results
        # =====================================================================
        if self.config.save_intermediate:
            self._save_intermediate_results(audio_path, metadata)

        # Save speaker map alongside results if provided
        if getattr(self.config, "speaker_map_path", None):
            try:
                speaker_map = self._load_speaker_map(self.config.speaker_map_path)
                save_json(
                    speaker_map,
                    Path(self.config.cache_dir) / f"{Path(audio_path).stem}_speaker_map.json",
                )
            except Exception:
                pass

        # =====================================================================
        # Build result
        # =====================================================================
        processing_time = time.time() - start_time

        result = PipelineResult(
            audio_path=audio_path,
            audio_duration=duration,
            num_speakers=len(unique_speakers),
            num_segments=len(self._transcript_segments),
            total_words=total_words,
            processing_time=processing_time,
            segments=[seg.to_dict() for seg in self._transcript_segments],
            transcript_text=self.get_transcript_text(),
            summary=self._summary.to_dict(),
            document_path=doc_path,
            metadata={
                "title": title,
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "location": location,
                "device": self.config.device,
                "asr_model": self.config.asr_model_id,
            },
        )

        self._log("=" * 60)
        self._log(f"Processing complete! Total time: {format_duration(processing_time)}")
        self._log(f"Output: {doc_path}")
        self._log("=" * 60)

        return result

    # =========================================================================
    # Individual Step Methods
    # =========================================================================

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file"""
        self._waveform, self._sample_rate = self.audio_processor.load_audio(audio_path)
        return self._waveform, self._sample_rate

    def run_diarization(self, num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        """Run diarization on loaded audio"""
        if self._waveform is None:
            raise ValueError("Audio not loaded. Call load_audio() first.")

        self._diarization_segments = self.diarizer.process(
            self._waveform, self._sample_rate, num_speakers=num_speakers
        )
        return self._diarization_segments

    def run_transcription(self) -> List[TranscriptSegment]:
        """Run ASR on diarized segments"""
        if self._diarization_segments is None:
            raise ValueError("Diarization not done. Call run_diarization() first.")

        self._transcript_segments = self.transcriber.transcribe_segments(
            self._waveform, self._diarization_segments, self._sample_rate
        )
        return self._transcript_segments

    def run_summarization(self) -> MeetingSummary:
        """Generate summary from transcript"""
        if self._transcript_segments is None:
            raise ValueError("Transcription not done. Call run_transcription() first.")

        self._summary = self.summarizer.summarize(self._transcript_segments)
        return self._summary

    def generate_document(
        self, metadata: MeetingMetadata, output_filename: str = "notulensi.docx"
    ) -> str:
        """Generate .docx document"""
        if self._transcript_segments is None or self._summary is None:
            raise ValueError("Transcript and summary required.")

        return self.doc_generator.generate(
            metadata=metadata,
            summary=self._summary,
            transcript=self._transcript_segments,
            output_filename=output_filename,
        )

    # =========================================================================
    # Evaluation Methods
    # =========================================================================

    def evaluate(
        self,
        reference_transcript: Optional[str] = None,
        reference_diarization: Optional[List[Tuple[str, float, float]]] = None,
        sample_name: str = "sample",
        condition: str = "unknown",
    ) -> EvaluationResult:
        """
        Evaluate pipeline output against ground truth.

        Args:
            reference_transcript: Ground truth transcript text
            reference_diarization: Ground truth diarization [(speaker, start, end), ...]
            sample_name: Name for this sample
            condition: Test condition name

        Returns:
            EvaluationResult with WER and DER
        """
        wer_result = None
        der_result = None

        # Calculate WER if reference transcript provided
        if reference_transcript and self._transcript_segments:
            hypothesis = self.get_transcript_text()
            wer_result = self.evaluator.calculate_wer(reference_transcript, hypothesis)
            self._log(f"WER: {wer_result.wer:.4f} ({wer_result.wer*100:.2f}%)")

            # Calculate DER if reference diarization provided
        if reference_diarization and self._diarization_segments:
            hypothesis_diarization = [
                (seg.speaker_id, seg.start, seg.end) for seg in self._diarization_segments
            ]
            der_result = self.evaluator.calculate_der(reference_diarization, hypothesis_diarization)
            self._log(f"DER: {der_result.der:.4f} ({der_result.der*100:.2f}%)")

        # If reference diarization not provided but reference transcript contains speaker labels,
        # attempt to build a reference diarization by aligning the labeled transcript to the
        # pipeline's transcript segments. This often improves DER accuracy when GT RTTM is missing.
        if not reference_diarization and reference_transcript and self._diarization_segments:
            # Heuristic detection: presence of 'Name:' lines
            if ":" in reference_transcript and any(
                line.strip().endswith(":") or ":" in line
                for line in reference_transcript.splitlines()[:20]
            ):
                try:
                    from src.utils import (
                        align_reference_to_segments,
                        parse_speaker_labeled_text,
                    )

                    utterances = parse_speaker_labeled_text(reference_transcript)
                    if utterances:
                        hyp_segs = self._transcript_segments or []
                        # Build reference diarization from alignment
                        derived_ref = align_reference_to_segments(utterances, hyp_segs)
                        if derived_ref:
                            hypothesis_diarization = [
                                (seg.speaker_id, seg.start, seg.end)
                                for seg in self._diarization_segments
                            ]
                            der_result = self.evaluator.calculate_der(
                                derived_ref, hypothesis_diarization
                            )
                            self._log(
                                f"Derived RTTM used for DER (from speaker-labeled transcript). DER: {der_result.der:.4f} ({der_result.der*100:.2f}%)"
                            )
                except Exception as e:
                    self._log(f"Auto-alignment for RTTM failed: {e}")
                    pass

        return EvaluationResult(
            sample_name=sample_name,
            condition=condition,
            wer_result=wer_result,
            der_result=der_result,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_transcript_text(self) -> str:
        """Get full transcript as plain text"""
        if self._transcript_segments is None:
            return ""
        return " ".join(seg.text for seg in self._transcript_segments if seg.text)

    def get_formatted_transcript(self) -> str:
        """Get transcript with speaker labels and timestamps"""
        if self._transcript_segments is None:
            return ""

        lines = []
        for seg in self._transcript_segments:
            timestamp = format_duration(seg.start)
            lines.append(f"[{timestamp}] {seg.speaker_id}: {seg.text}")

        return "\n".join(lines)

    def get_speaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per speaker"""
        if self._transcript_segments is None:
            return {}

        stats = {}
        for seg in self._transcript_segments:
            if seg.speaker_id not in stats:
                stats[seg.speaker_id] = {"word_count": 0, "duration": 0.0, "segment_count": 0}

            stats[seg.speaker_id]["word_count"] += seg.word_count
            stats[seg.speaker_id]["duration"] += seg.duration
            stats[seg.speaker_id]["segment_count"] += 1

        return stats

    def clear_state(self):
        """Clear internal state for fresh processing"""
        self._waveform = None
        self._sample_rate = None
        self._diarization_segments = None
        self._transcript_segments = None
        self._summary = None

    def _log(self, message: str):
        """Log message"""
        if self.config.verbose:
            print(f"[Pipeline] {message}")
        self.logger.info(message)

    def _load_speaker_map(self, path: str) -> dict:
        """Load a speaker map from JSON or YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Speaker map file not found: {path}")
        try:
            import json

            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("Speaker map must be a JSON object mapping labels to names")
            return data
        except Exception:
            try:
                import yaml

                with open(p, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    raise ValueError("Speaker map must be a mapping in YAML/JSON format")
                return data
            except Exception as e:
                raise ValueError(f"Failed to parse speaker map: {e}")

    def _apply_speaker_map(self, mapping: dict):
        """Apply speaker mapping to transcript segments and summary action items.

        This replaces `seg.speaker_id` with the provided name and stores the original id in
        `seg.metadata['original_speaker_id']` for traceability.
        """
        if not mapping:
            return

        # Update transcript segments if they exist
        if getattr(self, "_transcript_segments", None):
            for seg in self._transcript_segments:
                orig = seg.speaker_id
                mapped = mapping.get(orig)
                if mapped and mapped != orig:
                    seg.metadata["original_speaker_id"] = orig
                    seg.speaker_id = mapped

        # Update action item owners in summary
        try:
            for ai in self._summary.action_items or []:
                owner = ai.get("owner")
                if owner and owner in mapping:
                    ai["owner"] = mapping[owner]
        except Exception:
            pass

        # Finally update diarization segments as well (if present)
        try:
            self._log(f"Applying speaker mapping to diarization segments: {mapping}")
            for dseg in self._diarization_segments or []:
                orig = dseg.speaker_id
                mapped = mapping.get(orig)
                self._log(f"Segment {orig} -> mapped: {mapped}")
                if mapped and mapped != orig:
                    dseg.metadata["original_speaker_id"] = orig
                    dseg.speaker_id = mapped
            self._log(
                f"Post-map speaker ids: {[d.speaker_id for d in (self._diarization_segments or [])]}"
            )
        except Exception as e:
            self._log(f"Error applying speaker map to diarization segments: {e}")
            pass

    def _save_intermediate_results(self, audio_path: str, metadata: MeetingMetadata):
        """Save intermediate results to JSON"""
        base_name = Path(audio_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "audio_path": audio_path,
            "timestamp": timestamp,
            "metadata": {
                "title": metadata.title,
                "date": metadata.date,
                "duration": metadata.duration,
            },
            "config": {
                "sample_rate": self.config.sample_rate,
                "asr_model": self.config.asr_model_id,
                "device": self.config.device,
            },
            "diarization": [
                {
                    "speaker_id": seg.speaker_id,
                    "start": seg.start,
                    "end": seg.end,
                    "is_overlap": seg.is_overlap,
                }
                for seg in (self._diarization_segments or [])
            ],
            "transcript": [seg.to_dict() for seg in (self._transcript_segments or [])],
            "summary": self._summary.to_dict() if self._summary else None,
        }

        output_path = Path(self.config.cache_dir) / f"{base_name}_{timestamp}_results.json"
        save_json(results, output_path)

        self._log(f"Intermediate results saved: {output_path}")

    # ------------------------------------------------------------------
    # Convenience methods for interactive flows (UI, Streamlit)
    # ------------------------------------------------------------------
    def run_diarization(self, audio_path: str) -> dict:
        """Run loading + diarization steps and return a dict with summary info.

        Returns: {"audio_duration": float, "num_windows": int, "num_speech_regions": int, "unique_speakers": [..], "segments": [..]}
        """
        # Load audio
        self._waveform, self._sample_rate = self.audio_processor.load_audio(audio_path)
        duration = self.audio_processor.get_duration(self._waveform, self._sample_rate)

        # Run diarization
        self._diarization_segments = self.diarizer.process(
            self._waveform,
            self._sample_rate,
            num_speakers=None,
            cache_dir=self.config.cache_dir,
            audio_id=Path(audio_path).stem,
            fast_mode=self.config.fast_mode,
        )

        unique_speakers = sorted(list(set(seg.speaker_id for seg in self._diarization_segments)))

        return {
            "audio_duration": duration,
            "num_segments": len(self._diarization_segments),
            "unique_speakers": unique_speakers,
            "segments": [
                {"speaker_id": s.speaker_id, "start": s.start, "end": s.end}
                for s in self._diarization_segments
            ],
        }

    def apply_speaker_map(
        self, mapping: dict, save_to_cache: bool = False, audio_id: Optional[str] = None
    ):
        """Apply a manual speaker mapping to internal state and optionally save the map to cache.

        mapping: dict mapping original speaker id -> desired display name
        """
        self._apply_speaker_map(mapping)
        if save_to_cache and audio_id:
            try:
                save_json(mapping, Path(self.config.cache_dir) / f"{audio_id}_speaker_map.json")
            except Exception:
                pass

    def continue_from_diarization(
        self,
        title: str = "Notulensi Rapat",
        date: Optional[str] = None,
        location: str = "",
        output_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        """Continue processing from the current _waveform and _diarization_segments.

        Runs ASR, summarization, and document generation using existing in-memory diarization.
        """
        if (
            getattr(self, "_waveform", None) is None
            or getattr(self, "_diarization_segments", None) is None
        ):
            raise RuntimeError(
                "Diarization state not found. Run run_diarization(audio_path) first."
            )

        update_progress = lambda step, cur, total: (
            progress_callback(step, cur, total) if progress_callback else None
        )

        # Step 3: ASR
        update_progress("Transcribing speech", 3, 5)
        with Timer("Transcription"):
            self._transcript_segments = self.transcriber.transcribe_segments(
                self._waveform, self._diarization_segments, self._sample_rate
            )

        total_words = sum(seg.word_count for seg in self._transcript_segments)
        self._log(f"Transcribed {len(self._transcript_segments)} segments, ~{total_words} words")

        # Apply speaker map if configured
        if getattr(self.config, "speaker_map_path", None):
            try:
                speaker_map = self._load_speaker_map(self.config.speaker_map_path)
                self._apply_speaker_map(speaker_map)
            except Exception as e:
                self._log(f"Failed to load/apply speaker map: {e}")

        # Step 4: Summarization
        update_progress("Generating summary", 4, 5)
        with Timer("Summarization"):
            self._summary = self.summarizer.summarize(self._transcript_segments)

        self._log(f"Generated summary with {len(self._summary.key_points)} key points")

        # Step 5: Document generation
        update_progress("Generating document", 5, 5)

        participants = list(set(seg.speaker_id for seg in self._diarization_segments))
        if getattr(self.config, "speaker_map_path", None):
            try:
                speaker_map = self._load_speaker_map(self.config.speaker_map_path)
                participants = [speaker_map.get(p, p) for p in participants]
            except Exception:
                pass

        metadata = MeetingMetadata(
            title=title,
            date=date or datetime.now().strftime("%d %B %Y"),
            time=datetime.now().strftime("%H:%M"),
            location=location,
            duration=format_duration(
                self.audio_processor.get_duration(self._waveform, self._sample_rate)
            ),
            participants=participants,
        )

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = sanitize_filename(title)[:30]
            output_filename = f"notulensi_{safe_title}_{timestamp}.docx"

        with Timer("Document generation"):
            doc_path = self.doc_generator.generate(
                metadata=metadata,
                summary=self._summary,
                transcript=self._transcript_segments,
                output_filename=output_filename,
            )

        self._log(f"Document saved: {doc_path}")

        # Save intermediate results
        if self.config.save_intermediate:
            self._save_intermediate_results(output_filename, metadata)

        processing_time = 0.0
        result = PipelineResult(
            audio_path=output_filename,
            audio_duration=self.audio_processor.get_duration(self._waveform, self._sample_rate),
            num_speakers=len(set(seg.speaker_id for seg in self._diarization_segments)),
            num_segments=len(self._transcript_segments),
            total_words=total_words,
            processing_time=processing_time,
            segments=[seg.to_dict() for seg in (self._transcript_segments or [])],
            transcript_text="\n".join([s.text for s in (self._transcript_segments or [])]),
            summary=self._summary.to_dict() if self._summary else {},
            document_path=str(doc_path),
        )

        return result
