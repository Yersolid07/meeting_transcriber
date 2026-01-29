import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.summarizer import BERTSummarizer, SummarizationConfig, AbstractiveSummarizer
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


def test_decision_deduplication():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)

    sentences = ["Setuju lah", "Setuju", "Kita sepakat untuk melakukan evaluasi rutin."]
    decisions = s._extract_decisions(sentences)
    # Expect only one 'setuju' style decision after deduplication
    setuju_count = sum(1 for d in decisions if "setuju" in d.lower())
    assert setuju_count == 1, f"Expected only one 'setuju' decision, got {decisions}"


def test_abstractive_cleaning_helper():
    s = AbstractiveSummarizer()
    messy = "<extra_id_0>nya kurang rapi ... ) <extra_id_41> Kita terapkan jam kerja fleksibel <extra_id_0>"
    overview, key_points = s._clean_abstractive_output(messy, "Beberapa tugas memang selesai. Kita akan membuat sistem prioritas tugas.")
    assert "<extra_id" not in overview
    assert all("<extra_id" not in kp for kp in key_points)
    assert len(key_points) >= 1


def test_action_item_filtering_filler():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)

    segs = [
        TranscriptSegment(speaker_id="SPEAKER_01", start=0.0, end=1.0, text="Kita mulai rapatnya ya."),
        TranscriptSegment(speaker_id="SPEAKER_02", start=1.0, end=3.0, text="Budi akan menyiapkan laporan minggu depan."),
        TranscriptSegment(speaker_id="SPEAKER_01", start=3.0, end=4.0, text="Terima kasih."),
    ]

    actions = s._extract_action_items(segs)
    assert any("menyiapkan laporan" in a["task"].lower() for a in actions)
    assert not any("kita mulai" in a["task"].lower() for a in actions)
    assert not any("terima kasih" in a["task"].lower() for a in actions)


def test_action_item_filters_non_action_sentences():
    cfg = SummarizationConfig()
    s = BERTSummarizer(config=cfg)

    segs = [
        TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text="Beberapa tugas memang selesai."),
        TranscriptSegment(speaker_id="SPEAKER_01", start=1.0, end=3.0, text="Kita perlu sistem prioritas nanti.")
    ]

    actions = s._extract_action_items(segs)
    # The first sentence is descriptive and should not be an action
    assert not any("beberapa tugas memang selesai" in a["task"].lower() for a in actions)
    # The second one is borderline but contains 'perlu' (action keyword) and may be present
    assert any("sistem prioritas" in a["task"].lower() for a in actions)


def test_generate_comprehensive_summary_fallback():
    s = AbstractiveSummarizer()

    full_text = (
        "Rapat dimulai dengan pembukaan. Kita membahas target kuartal berikutnya. "
        "Budi akan menyiapkan laporan, dan Siti akan menyiapkan materi presentasi. Pada akhirnya, kita sepakat untuk melanjutkan proyek X."
    )
    key_points = [
        "Pembahasan target kuartal",
        "Budi menyiapkan laporan",
        "Siti menyiapkan materi presentasi",
    ]
    decisions = ["Sepakat melanjutkan proyek X"]
    action_items = [{"owner": "Budi", "task": "Menyiapkan laporan"}, {"owner": "Siti", "task": "Menyiapkan materi presentasi"}]
    topics = ["Target", "Laporan", "Proyek X"]

    overview, keywords = s.generate_comprehensive_summary(full_text, key_points, decisions, action_items, topics)

    assert isinstance(overview, str)
    assert overview
    # Fallback should return a narrative paragraph summarizing the meeting (not list headers)
    assert isinstance(overview, str)
    assert overview
    assert "Poin-Poin Penting" not in overview and "Keputusan" not in overview
    # Expect at least one sentence and some keywords
    assert len(overview.split('.')) >= 2
    assert isinstance(keywords, list)
    assert len(keywords) >= 1


def test_comprehensive_overview_integration_with_bertsummarizer():
    cfg = SummarizationConfig()
    cfg.comprehensive_overview = True
    s = BERTSummarizer(config=cfg)

    segs = [
        TranscriptSegment(speaker_id="S1", start=0.0, end=2.0, text="Rapat dimulai. Kita bahas target kuartal berikutnya."),
        TranscriptSegment(speaker_id="S2", start=2.0, end=5.0, text="Budi akan menyiapkan laporan dan Siti menyiapkan presentasi."),
        TranscriptSegment(speaker_id="S1", start=5.0, end=7.0, text="Akhirnya kita sepakat untuk melanjutkan proyek X."),
    ]

    summary = s.summarize(segs)
    assert isinstance(summary.overview, str)
    assert summary.overview
    # Should have keywords attribute populated
    assert getattr(summary, "keywords", [])
    # The comprehensive overview should either contain section headers or at least have keywords
    assert "Poin-Poin Penting" in summary.overview or "Keputusan" in summary.overview or getattr(summary, "keywords", [])
