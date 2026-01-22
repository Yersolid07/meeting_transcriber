import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from src.diarization import DiarizationConfig, SpeakerDiarizer

# test_cluster_merge_small_clusters
rng = np.random.RandomState(42)
centers = [rng.randn(192) for _ in range(8)]
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
print('unique counts:', list(zip(unique.tolist(), counts.tolist())))
assert all(c >= labels_expected_min_count for c in counts)
print('test_cluster_merge_small_clusters passed')

# test_kmeans_fallback
rng = np.random.RandomState(0)
embeddings = rng.randn(200, 192)
cfg = DiarizationConfig()
dz = SpeakerDiarizer(config=cfg)
class FakeAgg:
    def fit_predict(self, X):
        return np.arange(len(X))

import src.diarization as dmod
orig = dmod.AgglomerativeClustering
try:
    dmod.AgglomerativeClustering = lambda *args, **kwargs: FakeAgg()
    labels = dz._cluster_embeddings(embeddings)
    unique_labels = np.unique(labels)
    print('num clusters after fallback:', len(unique_labels))
    assert len(unique_labels) <= 12
    print('test_kmeans_fallback passed')
finally:
    dmod.AgglomerativeClustering = orig
