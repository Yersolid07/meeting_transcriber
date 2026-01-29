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


def test_delegation_to_abstractive_on_config_flag():
    cfg = SummarizationConfig(method="abstractive")
    b = BERTSummarizer(config=cfg)
    res = b.summarize([])
    assert "Tidak ada konten" in res.overview
