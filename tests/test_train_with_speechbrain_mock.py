import importlib.util
import json
import sys
import types
from pathlib import Path

# Create a fake 'speechbrain' module with minimal API expected by the script
fake_sb = types.ModuleType("speechbrain")
# minimal submodules/classes if needed
# tests will not call real sb functions; the presence of the module is enough for the script
sys.modules["speechbrain"] = fake_sb

# import the train_speechbrain by path
spec = importlib.util.spec_from_file_location(
    "train_speechbrain",
    str(Path("recipes/asr/train_speechbrain.py").resolve()),
)
train_speechbrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_speechbrain)


def write_small_manifest(path: Path):
    sample = [
        {"wav": "a.wav", "text": "saya makan nasi", "speaker": "spk1"},
        {"wav": "b.wav", "text": "apa kabar", "speaker": "spk2"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in sample:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def test_train_with_mock_speechbrain(tmp_path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    write_small_manifest(train)
    write_small_manifest(valid)
    outdir = tmp_path / "exp"
    args = [
        "--train-manifest",
        str(train),
        "--valid-manifest",
        str(valid),
        "--outdir",
        str(outdir),
        "--run-name",
        "testsb",
        "--epochs",
        "2",
        "--use-speaker-head",
        "--mock",
    ]
    rc = train_speechbrain.main(args)
    assert rc == 0
    run_dir = outdir / "testsb"
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoints" / "ckpt_epoch_1.pt").exists()
    # verify speaker accuracy appears in metrics
    with open(run_dir / "metrics.csv", "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "speaker_acc" in text


# cleanup fake module
del sys.modules["speechbrain"]
