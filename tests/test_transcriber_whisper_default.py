from src.transcriber import ASRConfig


def test_asrconfig_defaults_to_whisper():
    cfg = ASRConfig()
    assert cfg.backend == "whisper"
    assert "whisper" in cfg.model_id or "openai/whisper-base" == cfg.model_id
    assert hasattr(cfg, "language")
    assert cfg.language == "id"
