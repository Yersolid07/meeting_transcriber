import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.summarizer import BERTSummarizer, SummarizationConfig
from src.transcriber import TranscriptSegment


def test_advanced_nlp_action_extraction_simple():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)

    # Create simple segments with explicit "Budi akan ..." pattern
    segs = [
        TranscriptSegment(
            speaker_id="SPEAKER_01",
            start=0.0,
            end=3.0,
            text="Budi akan menyiapkan laporan minggu depan.",
        ),
        TranscriptSegment(
            speaker_id="SPEAKER_02", start=3.0, end=6.0, text="Baik, itu akan diselesaikan."
        ),
    ]

    actions = s._extract_action_items(segs)
    # Expect at least one action item mentioning 'menyiapkan laporan'
    found = any("menyiapkan laporan" in a["task"].lower() for a in actions)
    assert found, f"Expected action about 'menyiapkan laporan', got {actions}"


def test_decision_extraction_keywords():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)

    sentences = [
        "Kita sepakat untuk menunda acara hingga bulan depan.",
        "Tidak ada keputusan tambahan.",
    ]
    decisions = s._extract_decisions(sentences)
    assert any("sepakat" in d.lower() or "menunda" in d.lower() for d in decisions)
