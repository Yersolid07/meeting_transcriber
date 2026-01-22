import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from pathlib import Path

from src.diarization import DiarizationConfig, SpeakerDiarizer


def test_embedding_cache_and_batch(tmp_path, monkeypatch):
    # Create a fake waveform: 1 sec of random audio at 16k
    sr = 16000
    waveform = torch.randn(1, sr)

    # Create synthetic windows (10 windows of 0.1s)
    windows = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]

    sd = SpeakerDiarizer(config=DiarizationConfig())

    # Monkeypatch internal methods to avoid heavy model loads
    monkeypatch.setattr(sd, "_detect_speech", lambda w, sr: [(0.0, 1.0)])
    monkeypatch.setattr(sd, "_create_windows", lambda regions: windows)

    # Create a fake embedding model with encode_batch
    class FakeModel:
        def __init__(self):
            self.device = "cpu"

        def encode_batch(self, x):
            # x may be tensor or list; return deterministic output
            if isinstance(x, list):
                x = torch.stack([torch.tensor([1.0]) for _ in x])
            if isinstance(x, torch.Tensor):
                b = x.shape[0]
                return torch.zeros((b, 192)) + 0.1
            return np.zeros((len(x), 192)) + 0.1

    sd._embedding_model = FakeModel()

    cache_dir = str(tmp_path)
    audio_id = "test_audio"

    # First call should compute and save cache
    embs1 = sd._extract_embeddings(waveform, windows, sr, cache_dir=cache_dir, audio_id=audio_id)
    assert embs1.shape[0] == len(windows)

    # Modify the fake model to return different values to ensure cache is loaded
    sd._embedding_model = FakeModel()
    sd._embedding_model.encode_batch = lambda x: torch.ones((len(x), 192)) * 0.5

    embs2 = sd._extract_embeddings(waveform, windows, sr, cache_dir=cache_dir, audio_id=audio_id)

    # embs2 should equal embs1 (loaded from cache), not new 0.5 values
    assert np.allclose(embs1, embs2)
