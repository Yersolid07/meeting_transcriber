"""
Meeting Transcriber - Sistem Notulensi Rapat Otomatis
=====================================================

Sistem end-to-end untuk mengubah rekaman audio rapat menjadi
dokumen notulensi terstruktur menggunakan SpeechBrain dan BERT.

Modules:
    - config: Konfigurasi sistem
    - audio_processor: Preprocessing audio
    - diarization: Speaker diarization
    - transcriber: ASR transcription
    - summarizer: BERT summarization
    - document_generator: Export ke .docx
    - evaluator: Metrik evaluasi (WER, DER)
    - pipeline: Main orchestrator
    - utils: Utility functions

Example:
    >>> from src.pipeline import MeetingTranscriberPipeline
    >>> pipeline = MeetingTranscriberPipeline()
    >>> result = pipeline.process("meeting.wav", title="Team Meeting")
    >>> print(result.document_path)
"""

__version__ = "1.0.0"
__author__ = "Nama Mahasiswa"
__email__ = "email@university.ac.id"

from src.audio_processor import AudioConfig, AudioProcessor
from src.config import Config, load_config
from src.diarization import DiarizationConfig, SpeakerDiarizer, SpeakerSegment
from src.document_generator import DocumentGenerator, MeetingMetadata
from src.evaluator import DERResult, Evaluator, WERResult
from src.pipeline import MeetingTranscriberPipeline, PipelineConfig, PipelineResult
from src.summarizer import BERTSummarizer, MeetingSummary, SummarizationConfig
from src.transcriber import ASRConfig, ASRTranscriber, TranscriptSegment

__all__ = [
    # Config
    "Config",
    "load_config",
    # Audio
    "AudioProcessor",
    "AudioConfig",
    # Diarization
    "SpeakerDiarizer",
    "DiarizationConfig",
    "SpeakerSegment",
    # ASR
    "ASRTranscriber",
    "ASRConfig",
    "TranscriptSegment",
    # Summarization
    "BERTSummarizer",
    "SummarizationConfig",
    "MeetingSummary",
    # Document
    "DocumentGenerator",
    "MeetingMetadata",
    # Evaluation
    "Evaluator",
    "WERResult",
    "DERResult",
    # Pipeline
    "MeetingTranscriberPipeline",
    "PipelineConfig",
    "PipelineResult",
]
