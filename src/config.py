"""
Configuration Module
====================
Handles loading and managing configuration for the entire system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""

    threshold: float = 0.5
    min_speech_duration: float = 0.3
    min_silence_duration: float = 0.3
    speech_pad_ms: int = 30


@dataclass
class SegmentationConfig:
    """Segmentation configuration"""

    window_duration: float = 1.5
    window_hop: float = 0.75
    min_segment_duration: float = 0.5


@dataclass
class EmbeddingConfig:
    """Speaker embedding configuration"""

    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192


@dataclass
class ClusteringConfig:
    """Clustering configuration"""

    method: str = "agglomerative"
    threshold: float = 0.7
    min_cluster_size: int = 2
    linkage: str = "average"


@dataclass
class AudioConfig:
    """Audio processing configuration"""

    sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True
    trim_silence: bool = False
    max_duration_minutes: int = 60


@dataclass
class DiarizationConfig:
    """Speaker diarization configuration"""

    vad: VADConfig = field(default_factory=VADConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    merge_gap_threshold: float = 0.5
    min_segment_duration: float = 0.3
    smooth_segments: bool = True

    # Embedding and collapse options
    use_speechbrain: bool = True
    allow_fallback: bool = False
    collapse_threshold: float = 0.15
    silhouette_collapse_threshold: float = 0.05


@dataclass
class ASRConfig:
    """ASR configuration"""

    model_id: str = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
    chunk_length_s: float = 30.0
    stride_length_s: float = 5.0
    batch_size: int = 4
    return_timestamps: Optional[str] = None
    # Valid values: None (no timestamps), or 'char' / 'word' for CTC timestamp modes
    capitalize_sentences: bool = True
    normalize_whitespace: bool = True


@dataclass
class SummarizationConfig:
    """Summarization configuration"""

    model_id: str = "indobenchmark/indobert-base-p1"
    sentence_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    num_sentences: int = 5
    min_sentence_length: int = 10
    max_sentence_length: int = 200
    position_weight: float = 0.1
    decision_keywords: List[str] = field(
        default_factory=lambda: [
            "diputuskan",
            "disepakati",
            "kesimpulan",
            "keputusan",
            "jadi",
            "maka",
            "sepakat",
            "setuju",
            "final",
        ]
    )
    action_keywords: List[str] = field(
        default_factory=lambda: [
            "akan",
            "harus",
            "perlu",
            "tolong",
            "mohon",
            "deadline",
            "target",
            "tugas",
            "tanggung jawab",
            "action item",
            "follow up",
            "tindak lanjut",
        ]
    )


@dataclass
class DocumentConfig:
    """Document generation configuration"""

    template: str = "default"
    title_font_size: int = 18
    heading_font_size: int = 14
    body_font_size: int = 11
    font_family: str = "Calibri"
    include_timestamps: bool = True
    include_speaker_colors: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""

    wer_lowercase: bool = True
    wer_remove_punctuation: bool = True
    der_collar: float = 0.25
    der_skip_overlap: bool = False


@dataclass
class PathsConfig:
    """Paths configuration"""

    models_dir: str = "./models"
    audio_dir: str = "./data/audio"
    ground_truth_dir: str = "./data/ground_truth"
    output_dir: str = "./data/output"
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"


@dataclass
class Config:
    """Main configuration class"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: str = "auto"
    verbose: bool = True

    def __post_init__(self):
        """Create directories if they don't exist"""
        for path_attr in [
            "models_dir",
            "audio_dir",
            "ground_truth_dir",
            "output_dir",
            "cache_dir",
            "logs_dir",
        ]:
            path = getattr(self.paths, path_attr)
            os.makedirs(path, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config object with loaded settings
    """
    config = Config()

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            # Update audio config
            if "audio" in yaml_config:
                for key, value in yaml_config["audio"].items():
                    if hasattr(config.audio, key):
                        setattr(config.audio, key, value)

            # Update ASR config
            if "asr" in yaml_config:
                for key, value in yaml_config["asr"].items():
                    if hasattr(config.asr, key):
                        setattr(config.asr, key, value)

            # Update summarization config
            if "summarization" in yaml_config:
                for key, value in yaml_config["summarization"].items():
                    if hasattr(config.summarization, key):
                        setattr(config.summarization, key, value)

            # Update paths config
            if "paths" in yaml_config:
                for key, value in yaml_config["paths"].items():
                    if hasattr(config.paths, key):
                        setattr(config.paths, key, value)

            # Update device
            if "hardware" in yaml_config and "device" in yaml_config["hardware"]:
                config.device = yaml_config["hardware"]["device"]

    return config


def save_config(config: Config, config_path: str = "config.yaml"):
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path to save config.yaml
    """
    # Convert dataclass to dict
    config_dict = {
        "audio": config.audio.__dict__,
        "asr": config.asr.__dict__,
        "summarization": {
            k: v for k, v in config.summarization.__dict__.items() if not k.endswith("_keywords")
        },
        "document": config.document.__dict__,
        "evaluation": config.evaluation.__dict__,
        "paths": config.paths.__dict__,
        "hardware": {"device": config.device},
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
