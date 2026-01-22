import time
import os
import sys
import torch
# Ensure repo root is on sys.path for tests run in isolation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.transcriber import ASRTranscriber, ASRConfig, TranscriptSegment


def make_waveform(duration_s=2.0, sample_rate=16000):
    # stereo dummy wave (1, samples)
    samples = int(duration_s * sample_rate)
    return torch.zeros((1, samples)), sample_rate


class DummyTranscriber(ASRTranscriber):
    def __init__(self, cfg=None):
        super().__init__(cfg or ASRConfig())
        # Prevent heavy model loading
        self._pipeline = None

    def _load_model(self):
        # Override to avoid model initialization in tests
        return

    def _transcribe_audio(self, audio, sample_rate):
        # simulate expensive work
        time.sleep(0.12)
        return "dummy transcription"


class DummySeg:
    def __init__(self, start, end, speaker_id=0, confidence=0.95, is_overlap=False):
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.confidence = confidence
        self.is_overlap = is_overlap


def test_parallel_transcribe_speeds_up():
    waveform, sr = make_waveform(duration_s=5.0)

    # create 4 segments that will each cause a sleep
    segs = [DummySeg(i * 0.5, i * 0.5 + 0.4) for i in range(4)]

    # serial
    t1 = DummyTranscriber(ASRConfig(parallel_workers=1))
    start = time.perf_counter()
    out1 = t1.transcribe_segments(waveform, segs, sample_rate=sr)
    serial_time = time.perf_counter() - start

    # parallel
    t2 = DummyTranscriber(ASRConfig(parallel_workers=4))
    start = time.perf_counter()
    out2 = t2.transcribe_segments(waveform, segs, sample_rate=sr)
    parallel_time = time.perf_counter() - start

    assert len(out1) == len(out2) == 4
    assert parallel_time < serial_time
