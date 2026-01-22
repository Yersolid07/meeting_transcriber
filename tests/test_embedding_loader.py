import os
import sys
import types

import pytest

# Ensure project root is on path when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diarization import DiarizationConfig, SpeakerDiarizer


class DummyError(Exception):
    pass


def test_loader_raises_when_no_speechbrain(monkeypatch, tmp_path):
    # Simulate EncoderClassifier.from_hparams failing
    import importlib

    sd = SpeakerDiarizer(config=DiarizationConfig(), models_dir=str(tmp_path))

    def fake_from_hparams(*args, **kwargs):
        raise DummyError("simulated failure")

    # Ensure torchaudio shim exists before importing speechbrain
    import types

    fake_torchaudio = types.SimpleNamespace()
    fake_torchaudio.list_audio_backends = lambda: ["sox_io"]
    fake_torchaudio.get_audio_backend = lambda: "sox_io"
    sys.modules["torchaudio"] = fake_torchaudio

    # Create fake speechbrain modules to avoid importing the real package during the test
    import types as _types

    fake_speechbrain_pkg = _types.ModuleType("speechbrain")
    fake_inference_pkg = _types.ModuleType("speechbrain.inference")
    fake_speaker_mod = _types.ModuleType("speechbrain.inference.speaker")

    class DummyEnc:
        @staticmethod
        def from_hparams(*args, **kwargs):
            raise DummyError("simulated failure")

    fake_speaker_mod.EncoderClassifier = DummyEnc
    sys.modules["speechbrain"] = fake_speechbrain_pkg
    sys.modules["speechbrain.inference"] = fake_inference_pkg
    sys.modules["speechbrain.inference.speaker"] = fake_speaker_mod

    monkeypatch.setattr(
        fake_speaker_mod.EncoderClassifier, "from_hparams", staticmethod(fake_from_hparams)
    )

    # When allow_fallback is False (default), loading should raise RuntimeError
    with pytest.raises(RuntimeError):
        sd._load_embedding_model()


def test_loader_falls_back_when_allowed(monkeypatch, tmp_path):
    sd = SpeakerDiarizer(config=DiarizationConfig(), models_dir=str(tmp_path))

    def fake_from_hparams(*args, **kwargs):
        raise DummyError("simulated failure")

    fake_torchaudio = types.SimpleNamespace()
    fake_torchaudio.list_audio_backends = lambda: ["sox_io"]
    fake_torchaudio.get_audio_backend = lambda: "sox_io"
    sys.modules["torchaudio"] = fake_torchaudio

    # Create fake speechbrain modules to avoid importing the real package during the test
    import types as _types

    fake_speechbrain_pkg = _types.ModuleType("speechbrain")
    fake_inference_pkg = _types.ModuleType("speechbrain.inference")
    fake_speaker_mod = _types.ModuleType("speechbrain.inference.speaker")

    class DummyEnc:
        @staticmethod
        def from_hparams(*args, **kwargs):
            raise DummyError("simulated failure")

    fake_speaker_mod.EncoderClassifier = DummyEnc
    sys.modules["speechbrain"] = fake_speechbrain_pkg
    sys.modules["speechbrain.inference"] = fake_inference_pkg
    sys.modules["speechbrain.inference.speaker"] = fake_speaker_mod

    monkeypatch.setattr(
        fake_speaker_mod.EncoderClassifier, "from_hparams", staticmethod(fake_from_hparams)
    )

    # allow fallback
    sd.config.allow_fallback = True
    sd._load_embedding_model()
    assert sd._embedding_model == "FALLBACK"
    assert callable(sd._fallback_extractor)
