"""Speaker classifier scaffold for multi-task training and evaluation.

This module provides a small PyTorch `SpeakerClassifier` that maps embeddings
(or pooled encoder outputs) to speaker logits, plus helpers to build speaker
mappings from manifests.
"""

import json
from pathlib import Path
from typing import Dict, List

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


class SpeakerClassifier:
    """A light-weight wrapper that exposes an API-compatible classifier.

    If PyTorch is available, `SpeakerClassifier.model` is a `nn.Module`.
    Otherwise this is a placeholder to keep the dependency optional in tests.
    """

    def __init__(self, input_dim: int, num_speakers: int, dropout: float = 0.1):
        self.input_dim = input_dim
        self.num_speakers = num_speakers
        self.dropout = dropout
        if torch is not None and nn is not None:
            self.model = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, num_speakers),
            )
        else:
            self.model = None

    def forward(self, x):
        if self.model is None:
            raise RuntimeError("PyTorch not available for SpeakerClassifier")
        return self.model(x)


def build_speaker_map(manifest_paths: List[str]) -> Dict[str, int]:
    """Read JSONL manifest(s) and return a speaker->id mapping.

    The manifest format: each line is JSON with optional "speaker" key.
    Labels are returned in deterministic sorted order.
    """
    speakers = set()
    for p in manifest_paths:
        pth = Path(p)
        if not pth.exists():
            continue
        with open(pth, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                spk = obj.get("speaker")
                if spk is not None:
                    speakers.add(str(spk))
    sorted_spks = sorted(speakers)
    return {s: i for i, s in enumerate(sorted_spks)}
