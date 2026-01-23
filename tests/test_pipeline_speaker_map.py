import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.pipeline import MeetingTranscriberPipeline, PipelineConfig
from src.transcriber import TranscriptSegment


def test_apply_speaker_map_updates_segments_and_actions(tmp_path):
    cfg = PipelineConfig()
    pipeline = MeetingTranscriberPipeline(config=cfg)

    # Create dummy transcript segments with SPEAKER_00 and SPEAKER_01
    segs = [
        TranscriptSegment(
            speaker_id="SPEAKER_00", start=0.0, end=2.0, text="Halo, saya akan mengerjakan laporan."
        ),
        TranscriptSegment(speaker_id="SPEAKER_01", start=2.0, end=4.0, text="Baik, terima kasih."),
    ]

    pipeline._transcript_segments = segs
    pipeline._diarization_segments = []

    # Create a fake summary with action items
    pipeline._summary = type("S", (), {})()
    pipeline._summary.action_items = [{"owner": "SPEAKER_00", "task": "membuat laporan", "due": ""}]

    mapping = {"SPEAKER_00": "Budi", "SPEAKER_01": "Siti"}

    pipeline._apply_speaker_map(mapping)

    assert pipeline._transcript_segments[0].speaker_id == "Budi"
    assert pipeline._transcript_segments[0].metadata.get("original_speaker_id") == "SPEAKER_00"
    assert pipeline._summary.action_items[0]["owner"] == "Budi"
