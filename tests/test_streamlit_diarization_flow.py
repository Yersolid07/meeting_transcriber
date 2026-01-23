import os
import tempfile

import torch

from src.pipeline import MeetingTranscriberPipeline, PipelineConfig


class DummyAudioProcessor:
    def load_audio(self, path):
        sr = 16000
        t_samples = int(1.0 * sr)
        return torch.zeros((1, t_samples)), sr

    def get_duration(self, waveform, sr):
        return waveform.shape[-1] / sr


class DummyDiarizer:
    def process(self, waveform, sample_rate, num_speakers=None, **kwargs):
        from src.diarization import SpeakerSegment

        return [
            SpeakerSegment(
                speaker_id="SPEAKER_00", start=0.0, end=0.5, confidence=1.0, is_overlap=False
            ),
            SpeakerSegment(
                speaker_id="SPEAKER_01", start=0.5, end=1.0, confidence=1.0, is_overlap=False
            ),
        ]


def test_diarization_mapping_flow(tmp_path, monkeypatch):
    cfg = PipelineConfig(
        models_dir=str(tmp_path), output_dir=str(tmp_path), save_intermediate=False
    )
    pipeline = MeetingTranscriberPipeline(cfg)

    # Patch processors
    pipeline._audio_processor = DummyAudioProcessor()
    pipeline._diarizer = DummyDiarizer()

    # Run diarization
    dz = pipeline.run_diarization(str(tmp_path / "dummy.wav"))
    assert dz["unique_speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    # Apply mapping
    mapping = {"SPEAKER_00": "Budi", "SPEAKER_01": "Ani"}
    pipeline.apply_speaker_map(mapping, save_to_cache=True, audio_id="dummy")

    # Ensure diarization segments were updated
    assert [s.speaker_id for s in pipeline._diarization_segments] == ["Budi", "Ani"]

    # Continue processing (stub transcriber to avoid heavy work)
    class DummyTranscriber:
        def transcribe_segments(self, waveform, segments, sample_rate=16000):
            from src.transcriber import TranscriptSegment

            return [
                TranscriptSegment(
                    speaker_id=s.speaker_id,
                    start=s.start,
                    end=s.end,
                    text=("hello" if s.speaker_id == "Budi" else "hi"),
                )
                for s in segments
            ]

    pipeline._transcriber = DummyTranscriber()

    res = pipeline.continue_from_diarization(title="Test")

    # Check mapping reflected in transcript segments
    assert any(seg["speaker_id"] == "Budi" for seg in res.segments)
    assert res.num_speakers == 2
    # summary presence
    assert isinstance(res.summary, dict) or hasattr(res, "summary")
