import os
import sys

import numpy as np
import pytest
import torch

# Ensure project root is on path when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diarization import SpeakerSegment
from src.transcriber import ASRConfig, ASRTranscriber


class DummyPipeline:
    def __call__(self, audio, **kwargs):
        # return a structured dict resembling some ASR pipelines
        return {
            "text": "Halo dunia ini adalah tes suara",
            "chunks": [
                {"start": 0.0, "end": 1.0, "text": "Halo dunia"},
                {"start": 1.0, "end": 3.0, "text": "ini adalah tes"},
                {"start": 3.0, "end": 5.0, "text": "suara"},
            ],
        }


def test_map_full_result_to_segments():
    cfg = ASRConfig()
    cfg.use_full_audio_for_segments = True
    trans = ASRTranscriber(config=cfg)

    # Inject dummy pipeline
    trans._pipeline = DummyPipeline()

    # fake waveform (6s)
    sr = 16000
    t = 6 * sr
    wave = torch.from_numpy(np.zeros((1, t), dtype=np.float32))

    segments = [
        SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=1.2),
        SpeakerSegment(speaker_id="SPEAKER_00", start=1.2, end=3.5),
        SpeakerSegment(speaker_id="SPEAKER_00", start=3.5, end=5.0),
    ]

    results = trans.transcribe_segments(wave, segments, sample_rate=sr)

    assert len(results) == 3
    assert "halo" in results[0].text.lower()
    assert "tes" in results[1].text.lower()
    assert "suara" in results[2].text.lower()
