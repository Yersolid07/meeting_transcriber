import pytest
import torch

from src.transcriber_speechbrain import SpeechBrainASRConfig, SpeechBrainTranscriber


class DummyModel:
    def transcribe_batch(self, audio_list):
        return ["dummy transcription" for _ in audio_list]

    def transcribe_file(self, path):
        return "dummy transcription file"


@pytest.fixture(autouse=True)
def patch_speechbrain(monkeypatch):
    # Patch the import path used in the adapter to return a dummy model
    class DummyEncoderASR:
        @staticmethod
        def from_hparams(source, savedir=None):
            return DummyModel()

    class DummyEncoderDecoderASR:
        @staticmethod
        def from_hparams(source, savedir=None):
            return DummyModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "speechbrain.pretrained",
        __import__("types").SimpleNamespace(
            EncoderASR=DummyEncoderASR, EncoderDecoderASR=DummyEncoderDecoderASR
        ),
    )
    yield


def test_transcribe_full_audio(tmp_path):
    cfg = SpeechBrainASRConfig(model_id="dummy/model")
    t = SpeechBrainTranscriber(cfg, models_dir=str(tmp_path))

    # Create a 1s sine wave
    sr = 16000
    t_samples = int(1.0 * sr)
    waveform = torch.from_numpy(
        (0.1 * torch.sin(torch.linspace(0, 2 * 3.1415, t_samples))).numpy()
    ).unsqueeze(0)

    text = t.transcribe_full_audio(waveform, sample_rate=sr)
    assert text == "dummy transcription"


def test_transcribe_segments(tmp_path):
    cfg = SpeechBrainASRConfig(model_id="dummy/model")
    t = SpeechBrainTranscriber(cfg, models_dir=str(tmp_path))

    sr = 16000
    t_samples = int(2.0 * sr)
    waveform = torch.from_numpy(
        (0.1 * torch.sin(torch.linspace(0, 2 * 3.1415 * 2, t_samples))).numpy()
    ).unsqueeze(0)

    from src.diarization import SpeakerSegment

    segs = [
        SpeakerSegment(speaker_id="spk1", start=0.0, end=1.0, confidence=1.0, is_overlap=False),
        SpeakerSegment(speaker_id="spk2", start=1.0, end=2.0, confidence=1.0, is_overlap=False),
    ]

    res = t.transcribe_segments(waveform, segs, sample_rate=sr)
    assert len(res) == 2
    assert res[0].text == "dummy transcription"
    assert res[1].text == "dummy transcription"
