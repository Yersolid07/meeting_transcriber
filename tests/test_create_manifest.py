import importlib.util
import json
from pathlib import Path

# Import scripts by path to make tests independent of package installation
spec = importlib.util.spec_from_file_location(
    "create_manifest", (Path(__file__).parents[1] / "scripts" / "create_manifest.py").resolve()
)
create_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_manifest)

spec2 = importlib.util.spec_from_file_location(
    "generate_sample_audio",
    (Path(__file__).parents[1] / "scripts" / "generate_sample_audio.py").resolve(),
)
generate_sample_audio = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(generate_sample_audio)


def test_make_manifest(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # generate sample wav
    wav_path = audio_dir / "sample.wav"
    generate_sample_audio.generate(wav_path, duration_s=0.5, sr=16000)

    # create transcripts csv
    csv = tmp_path / "trans.csv"
    csv.write_text("filename,wav_path,text\nsample.wav,sample.wav,hello world\n")

    out = tmp_path / "manifest.json"

    create_manifest.make_manifest(audio_dir, csv, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["text"] == "hello world"
    assert data[0]["wav"].endswith("sample.wav")
