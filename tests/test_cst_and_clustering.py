import numpy as np
from src.transcriber import ASRTranscriber
from src.diarization import SpeakerDiarizer
import torch


def test_cst_approximation_basic():
    t = ASRTranscriber()
    # generate 1 second of 16 kHz sine wave
    sr = 16000
    tlen = sr
    x = (0.1 * np.sin(2 * np.pi * 220.0 * np.arange(tlen) / sr)).astype(np.float32)
    y = t._apply_cst_approximation(x, sr, 7.5)
    assert y.shape == x.shape
    # The compressed signal should have less variance than original (lossy smoothing)
    assert np.var(y) <= np.var(x) + 1e-6


def test_cluster_method_override():
    dz = SpeakerDiarizer()
    # create synthetic embeddings for 4 clusters
    rng = np.random.RandomState(0)
    emb = np.vstack([rng.randn(10, 192) + i * 3.0 for i in range(4)])
    labels_a = dz._cluster_embeddings(emb, num_speakers=None, method_override="agglomerative")
    labels_b = dz._cluster_embeddings(emb, num_speakers=None, method_override="spectral")
    assert len(labels_a) == emb.shape[0]
    assert len(labels_b) == emb.shape[0]
