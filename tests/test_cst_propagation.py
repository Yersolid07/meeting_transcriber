import os
import sys
import pytest

# Ensure local package imports work when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import PipelineConfig, MeetingTranscriberPipeline


def test_cst_propagation_sets_asr_config():
    cfg = PipelineConfig(cst_hz=25)
    pipeline = MeetingTranscriberPipeline(cfg)
    assert pipeline.config.cst_hz == 25
    assert pipeline.transcriber.config.cst_hz == 25.0


def test_cst_default_when_none_keeps_disabled():
    cfg = PipelineConfig(cst_hz=None)
    pipeline = MeetingTranscriberPipeline(cfg)
    # Default should be disabled (None) unless explicitly set
    assert pipeline.config.cst_hz is None
    assert pipeline.transcriber.config.cst_hz is None


def test_cst_zero_disables_approximation():
    cfg = PipelineConfig(cst_hz=0)
    pipeline = MeetingTranscriberPipeline(cfg)
    assert pipeline.config.cst_hz == 0
    assert pipeline.transcriber.config.cst_hz == 0.0
