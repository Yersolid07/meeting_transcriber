import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.summarizer import AbstractiveSummarizer, SummarizationConfig, BERTSummarizer
from src.transcriber import TranscriptSegment


def test_default_method_is_extractive():
    cfg = SummarizationConfig()
    assert cfg.method == "extractive", "Default summarization method should be 'extractive'"


def test_clean_abstractive_output():
    s = AbstractiveSummarizer()
    raw = "Hasil ringkasan <extra_id_0> ini.......!!!   "
    cleaned = s._clean_abstractive_text(raw)
    assert "<extra_id_" not in cleaned
    # No excessive punctuation runs
    assert not re.search(r"[!?]{2,}", cleaned)
    assert "   " not in cleaned
    assert cleaned.endswith('.') or cleaned.endswith('!') or cleaned.endswith('?')


def test_sanitize_for_prompt_and_repetition_detection():
    s = AbstractiveSummarizer()
    messy_kp = ["<extra_id_0>nya kurang rapi", "Jadi contohnya " * 5]
    sanitized = [s._sanitize_for_prompt(k) for k in messy_kp]
    assert all("<extra_id" not in k for k in sanitized)

    repetitive = "Setuju lah " * 20
    assert s._is_repetitive_text(repetitive)
    assert not s._is_repetitive_text("Ini ringkasan yang baik dan singkat.")


def test_delegation_to_abstractive_on_config_flag():
    cfg = SummarizationConfig(method="abstractive")
    b = BERTSummarizer(config=cfg)
    res = b.summarize([])
    assert "Tidak ada konten" in res.overview


def test_collapse_repetition_in_cleaning():
    s = AbstractiveSummarizer()
    messy = "Jadi contohnya " * 8 + " Kita putuskan untuk coba sistem prioritas."
    cleaned = s._clean_abstractive_text(messy)
    # The repeated phrase should occur at most once in cleaned text
    assert cleaned.lower().count("jadi contohnya") <= 1


def test_parse_structured_output_simple_yaml():
    s = AbstractiveSummarizer()
    raw_yaml = """
    overview: Ringkasan singkat tentang rapat.
    key_points:
      - Pembahasan target kuartal
      - Penjadwalan ulang acara
    keywords:
      - rapat
      - target
    """

    overview, keywords = s._parse_structured_output(raw_yaml, {"key_points": []})
    assert "Ringkasan singkat" in overview
    assert isinstance(keywords, list)
    assert "rapat" in keywords or "target" in keywords


def test_sanitize_removes_domains_and_boilerplate():
    s = AbstractiveSummarizer()
    messy = "Eksekutif.com.co.id.id Eksekutif.info.id.id Semoga artikel ini bermanfaat bagi anda semua Semoga bermanfaat Terima kasih."
    cleaned = s._sanitize_for_prompt(messy)
    assert cleaned is not None
    # Should remove domain-like tokens and common boilerplate
    assert "eksekutif.com" not in cleaned.lower()
    assert "semoga artikel" not in cleaned.lower()
    assert "terima kasih" not in cleaned.lower()
