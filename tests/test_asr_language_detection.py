import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root on sys.path so `src` package imports resolve
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.transcriber import ASRConfig, ASRTranscriber


def test_auto_language_detection_monkeypatch(monkeypatch):
    cfg = ASRConfig()
    cfg.language = "auto"
    cfg.backend = "whisper"
    tr = ASRTranscriber(config=cfg)

    # Create a tiny waveform tensor (1s of silence)
    sr = 16000
    wav = np.zeros((1, sr), dtype=np.float32)
    wav_t = torch.from_numpy(wav)

    calls = {}

    def fake_pipeline(audio_np, **kwargs):
        # First call (quick pre-pass) will have no 'language' key
        if "language" not in kwargs:
            return {"text": "halo dunia"}
        # Final call should include chosen language
        calls["lang_used"] = kwargs.get("language")
        return {"text": f"FINAL ({kwargs.get('language')})"}

    monkeypatch.setattr(tr, "_pipeline", fake_pipeline)

    out = tr._transcribe_audio(wav_t, sr)
    assert "FINAL" in out
    # Language detector should pick some language code (e.g., 'id', 'in', 'so', etc.)
    assert calls.get("lang_used") is not None and isinstance(calls.get("lang_used"), str)
