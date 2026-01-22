import os
import sys

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diarization import SpeakerSegment
from src.transcriber import ASRConfig, ASRTranscriber


class DummyModel:
    def __init__(self):
        self.device = "cpu"

    def eval(self):
        pass

    def __call__(self, **kwargs):
        # return an object with logits of shape (batch, seq, vocab)
        batch = kwargs.get("input_values")
        T = 10
        V = 30
        logits = torch.randn((1, T, V), dtype=torch.float32)
        return type("X", (), {"logits": logits})


class DummyProcessor:
    def __init__(self):
        class Tok:
            def get_vocab(self):
                return {"a": 0, "b": 1, "c": 2}

        self.tokenizer = Tok()

    def __call__(self, audio, sampling_rate=None, return_tensors=None, padding=False):
        # Return a fake input dict compatible with DummyModel
        return {"input_values": torch.zeros((1, 160))}


def test_ctc_decoder_monkeypatched(monkeypatch):
    cfg = ASRConfig()
    cfg.decoder = "beam"
    cfg.beam_width = 5

    trans = ASRTranscriber(config=cfg)

    # inject dummy model and processor
    trans._model = DummyModel()
    trans._processor = DummyProcessor()

    # Create and attach fake decoder directly
    class FakeDecoder:
        def decode(self, emissions, beam_width=5):
            return "decoded beam"

    trans._ctc_decoder = FakeDecoder()

    # Create fake inputs
    sr = 16000
    wave = torch.zeros((1, sr), dtype=torch.float32)

    # Directly call _transcribe_audio path
    text = trans._transcribe_audio(wave, sr)
    assert "decoded" in text
