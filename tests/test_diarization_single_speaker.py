import os
import sys

import numpy as np

# Ensure repo src is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.diarization import DiarizationConfig, SpeakerDiarizer


def test_single_speaker_fallback():
    rng = np.random.RandomState(1)
    # create 8 very similar embeddings by building near-identical audio segments
    # We'll simulate by directly calling _cluster_embeddings with similar vectors
    base = rng.randn(192)
    embeddings = np.stack([base + 0.001 * rng.randn(192) for _ in range(12)], axis=0)

    cfg = DiarizationConfig()
    cfg.min_cluster_size = 2
    dz = SpeakerDiarizer(config=cfg)

    # Force fallback behaviour by setting the extractor and embedding model
    dz._embedding_model = "FALLBACK"
    dz._fallback_extractor = lambda seg, sr: base / (np.linalg.norm(base) + 1e-12)

    labels = dz._cluster_embeddings(embeddings)
    unique = np.unique(labels)

    # Similar embeddings should collapse to 1 cluster
    assert len(unique) == 1, f"Expected single cluster, got {len(unique)}"
