import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np

from src.diarization import DiarizationConfig, SpeakerDiarizer


def test_target_num_speakers_merge():
    rng = np.random.RandomState(42)
    # Create 10 somewhat distinct centers to simulate fragmentation
    centers = [rng.randn(192) + (i * 0.5) for i in range(10)]

    embeddings = []
    for i, c in enumerate(centers):
        for _ in range(5):
            embeddings.append(c + 0.01 * rng.randn(192))

    embeddings = np.stack(embeddings, axis=0)

    cfg = DiarizationConfig()
    cfg.target_num_speakers = 2
    cfg.target_force_threshold = 1.0  # allow merges regardless
    dz = SpeakerDiarizer(config=cfg)

    labels = dz._cluster_embeddings(embeddings)
    unique = np.unique(labels)
    assert len(unique) <= 2, f"Expected <=2 clusters after target merge, got {len(unique)}"
