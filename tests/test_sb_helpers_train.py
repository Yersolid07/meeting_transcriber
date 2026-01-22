import json
from pathlib import Path

from recipes.asr import sb_helpers


def write_manifest(path: Path):
    sample = [
        {"wav": "a.wav", "text": "saya makan nasi", "speaker": "spk1"},
        {"wav": "b.wav", "text": "apa kabar", "speaker": "spk2"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in sample:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def test_full_train_smoke(tmp_path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    write_manifest(train)
    write_manifest(valid)
    out = tmp_path / "exp"
    res = sb_helpers.full_train({}, str(train), str(valid), out, epochs=2, use_speaker_head=True)
    assert "predictions" in res
    assert "metrics" in res
    assert (out / "predictions.jsonl").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "checkpoints" / "ckpt_epoch_1.pt").exists()
