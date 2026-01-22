import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from src.diarization import DiarizationConfig, SpeakerDiarizer


def test_iterative_centroid_merge_reduces_fragments():
    rng = np.random.RandomState(123)
    # Create 6 centers, but make two pairs very close
    base = rng.randn(192)
    centers = [base + 0.001 * rng.randn(192) for _ in range(2)]
    centers += [rng.randn(192) for _ in range(4)]

    embeddings = []
    for i, c in enumerate(centers):
        # each center has 3 samples
        for _ in range(3):
            embeddings.append(c + 0.01 * rng.randn(192))

    embeddings = np.stack(embeddings, axis=0)

    cfg = DiarizationConfig()
    # set thresholds to force iterative merging of the close pair
    cfg.iterative_merge_threshold = 0.02
    cfg.iterative_merge_silhouette_threshold = -1.0
    cfg.iterative_merge_max_iters = 10

    dz = SpeakerDiarizer(config=cfg)
    labels = dz._cluster_embeddings(embeddings)
    unique = np.unique(labels)
    # Expect that at least one merge has happened -> fewer than 6 unique clusters
    assert len(unique) < 6, f"Expected fewer than 6 clusters after iterative merge, got {len(unique)}"
