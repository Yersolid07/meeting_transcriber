import torch

from src.pipeline import MeetingTranscriberPipeline, PipelineConfig


def test_pipeline_uses_speechbrain_backend(tmp_path, monkeypatch):
    cfg = PipelineConfig(models_dir=str(tmp_path), output_dir=str(tmp_path))
    cfg.asr_model_id = "dummy/model"
    cfg.asr_backend = "speechbrain"

    pipeline = MeetingTranscriberPipeline(config=cfg)

    # Patch SpeechBrainTranscriber to avoid actual downloads
    class DummySB:
        def transcribe_full_audio(self, waveform, sample_rate=16000):
            return "dummy full"

        def transcribe_segments(self, waveform, segments, sample_rate=16000):
            from src.transcriber import TranscriptSegment

            return [
                TranscriptSegment(speaker_id=s.speaker_id, start=s.start, end=s.end, text="dummy")
                for s in segments
            ]

    monkeypatch.setattr(
        "src.transcriber_speechbrain.SpeechBrainTranscriber", lambda cfg, models_dir=None: DummySB()
    )

    # Prepare a small audio
    sr = 16000
    t_samples = int(1.0 * sr)
    waveform = torch.zeros((1, t_samples))

    # Monkeypatch the audio_processor to return this waveform directly
    class DummyAudioProcessor:
        def load_audio(self, path):
            return waveform, sr

        def get_duration(self, waveform, sr):
            return waveform.shape[-1] / sr

    pipeline._audio_processor = DummyAudioProcessor()

    # Monkeypatch diarizer to return two toy segments so transcription runs
    class DummyDiarizer:
        def process(self, waveform, sample_rate, num_speakers=None, **kwargs):
            from src.diarization import SpeakerSegment

            return [
                SpeakerSegment(
                    speaker_id="spk1", start=0.0, end=0.5, confidence=1.0, is_overlap=False
                ),
                SpeakerSegment(
                    speaker_id="spk2", start=0.5, end=1.0, confidence=1.0, is_overlap=False
                ),
            ]

    pipeline._diarizer = DummyDiarizer()

    res = pipeline.process(
        audio_path=str(tmp_path / "dummy.wav"),
        title="Test",
    )

    assert res is not None
    assert res.transcript_text is not None
    assert "dummy" in res.transcript_text.lower()
