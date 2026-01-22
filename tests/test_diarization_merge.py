import os
import sys

import numpy as np

# Ensure repo src is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.diarization import DiarizationConfig, SpeakerDiarizer


def test_cluster_merge_small_clusters():
    # Create synthetic embeddings with 8 clusters where many are small (<3)
    rng = np.random.RandomState(42)
    centers = [rng.randn(192) for _ in range(8)]
    # cluster sizes: [1,1,1,1,1,1,3,4] => total 12
    embeddings = []
    labels_expected_min_count = 3

    for i, center in enumerate(centers):
        count = 1
        if i == 6:
            count = 3
        elif i == 7:
            count = 4
        for _ in range(count):
            embeddings.append(center + 0.01 * rng.randn(192))

    embeddings = np.stack(embeddings, axis=0)

    cfg = DiarizationConfig()
    cfg.min_cluster_size = labels_expected_min_count
    dz = SpeakerDiarizer(config=cfg)

    labels = dz._cluster_embeddings(embeddings, num_speakers=8)

    unique, counts = np.unique(labels, return_counts=True)
    # After merging, no cluster should have size < min_cluster_size
    assert all(c >= labels_expected_min_count for c in counts)


def test_kmeans_fallback(monkeypatch):
    # Simulate extreme fragmentation by forcing AgglomerativeClustering to return unique labels
    rng = np.random.RandomState(0)
    embeddings = rng.randn(200, 192)

    cfg = DiarizationConfig()
    dz = SpeakerDiarizer(config=cfg)

    class FakeAgg:
        def fit_predict(self, X):
            return np.arange(len(X))

    # Patch the AgglomerativeClustering constructor to return our fake
    monkeypatch.setattr(
        "src.diarization.AgglomerativeClustering", lambda *args, **kwargs: FakeAgg()
    )

    labels = dz._cluster_embeddings(embeddings)
    unique_labels = np.unique(labels)
    # After fallback and merging we should have a reasonable number of clusters
    assert (
        len(unique_labels) <= 12
    ), f"Expected <=12 clusters after fallback, got {len(unique_labels)}"
