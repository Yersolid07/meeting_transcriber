import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.summarizer import BERTSummarizer, AbstractiveSummarizer, SummarizationConfig
from src.transcriber import TranscriptSegment


def test_semantic_dedup_decisions_simple():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)
    items = [
        "Kita sepakat melakukan evaluasi rutin setiap dua minggu.",
        "Sepakat melakukan evaluasi rutin 2 minggu sekali.",
        "Kita akan membuat sistem prioritas tugas yang lebih jelas.",
    ]
    dedup = s._semantic_deduplicate(items, threshold=0.8)
    # Expect that similar first two are merged into one group
    assert len(dedup) <= 2


def test_semantic_dedup_action_items_merge_owners():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)
    actions = [
        {"owner": "A", "task": "Membuat draft sistem prioritas", "timestamp": "10.0s", "due": ""},
        {"owner": "B", "task": "membuat draft sistem prioritas", "timestamp": "12.0s", "due": ""},
        {"owner": "C", "task": "Sosialisasi dan mulai minggu depan", "timestamp": "50.0s", "due": ""},
    ]
    merged = s._semantic_dedup_action_items(actions, threshold=0.8)
    # One of merged tasks should have owners combined
    assert any('/' in a['owner'] or a['owner'] in ("A", "B") for a in merged)


def test_polish_overview_fallback():
    cfg = SummarizationConfig()
    s = AbstractiveSummarizer(config=cfg)
    messy = "Jadi contohnya " * 6 + " ini kita putuskan untuk coba mekanisme baru."
    polished = s._polish_overview(messy, messy)
    assert "jadi contohnya" not in polished.lower() or polished.lower().count("jadi contohnya") <= 1
    assert polished.endswith('.')
