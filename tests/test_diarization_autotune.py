import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np

from src.diarization import DiarizationConfig, SpeakerDiarizer


def test_auto_tune_picks_params_and_sets_target(monkeypatch):
    rng = np.random.RandomState(0)
    # Create embeddings with two clear clusters
    c1 = rng.randn(192) * 0.01
    c2 = rng.randn(192) * 0.01 + 3.0

    embs = []
    windows = []
    for _ in range(60):
        embs.append(c1 + 0.01 * rng.randn(192))
        windows.append((0.0, 1.0))
    for _ in range(60):
        embs.append(c2 + 0.01 * rng.randn(192))
        windows.append((1.0, 2.0))

    embs = np.stack(embs, axis=0)

    sd = SpeakerDiarizer(config=DiarizationConfig())

    # Monkeypatch the internal methods to return our synthetic windows/embeddings
    monkeypatch.setattr(sd, "_detect_speech", lambda w, sr: [(0.0, 2.0)])
    monkeypatch.setattr(sd, "_create_windows", lambda regions: windows)
    monkeypatch.setattr(sd, "_extract_embeddings", lambda waveform, windows, sr: embs)

    best = sd.auto_tune(None, 16000, num_speakers=2)

    assert sd.config.target_num_speakers == 2
    assert "clustering_threshold" in best
    # Check that clustering with selected params yields <=2 clusters after merging
    labels = sd._cluster_embeddings(embs)
    assert len(np.unique(labels)) <= 2, f"Expected <=2 clusters, got {len(np.unique(labels))}"
