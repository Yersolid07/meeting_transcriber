import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location(
    "preprocess_icorpus", Path(__file__).parents[1] / "scripts" / "preprocess_icorpus.py"
)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
build_manifest = preprocess.build_manifest


def create_dummy_wav(path: Path, duration=1.0, sr=16000):
    samples = int(duration * sr)
    data = np.zeros((samples,), dtype="float32")
    sf.write(str(path), data, sr)


def test_build_manifest_creates_jsonl(tmp_path):
    base = tmp_path / "icorpus"
    wav = base / "WAV"
    txt = base / "TXT"
    wav.mkdir(parents=True)
    txt.mkdir(parents=True)

    # create dummy wav and txt
    wavfile = wav / "TEST001.wav"
    create_dummy_wav(wavfile)

    txtfile = txt / "TEST001.txt"
    txtfile.write_text("[0.000,1.000]\tSPK1\tmale\thalo dunia\n", encoding="utf-8")

    out_manifest = tmp_path / "out.jsonl"

    # run manifest builder
    build_manifest(str(base), str(out_manifest))

    assert out_manifest.exists()
    lines = list(out_manifest.read_text(encoding="utf-8").splitlines())
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["text"] == "halo dunia"
    assert rec["speaker_id"] == "SPK1"
