import importlib.util
import json
from pathlib import Path

# Import eval_speaker by path to avoid package install issues in tests
spec = importlib.util.spec_from_file_location(
    "eval_speaker",
    str(Path("scripts/eval_speaker.py").resolve()),
)
eval_speaker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_speaker)


def test_eval_speaker_tmp(tmp_path):
    preds = tmp_path / "preds.jsonl"
    sample = [
        {"wav": "a.wav", "ref": "hello", "hyp": "hello", "speaker": "spk1", "pred_spk": "spk1"},
        {"wav": "b.wav", "ref": "hi", "hyp": "hi", "speaker": "spk2", "pred_spk": "spk1"},
    ]
    with open(preds, "w", encoding="utf-8") as fh:
        for s in sample:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    outdir = tmp_path / "out"
    outdir.mkdir()
    eval_speaker.main([str(preds), "--outdir", str(outdir)])
    j = outdir / "metrics_summary_spk.json"
    c = outdir / "metrics_summary_spk.csv"
    assert j.exists()
    assert c.exists()
    with open(j, "r", encoding="utf-8") as fh:
        s = json.load(fh)
    assert "accuracy" in s
    assert s["total"] == 2
    assert abs(s["accuracy"] - 0.5) < 1e-6
