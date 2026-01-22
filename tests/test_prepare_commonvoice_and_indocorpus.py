import importlib.util
import json
from pathlib import Path

# load prepare scripts directly from files
spec = importlib.util.spec_from_file_location(
    "prepare_commonvoice",
    (Path(__file__).parents[1] / "scripts" / "prepare_commonvoice.py").resolve(),
)
prepare_commonvoice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_commonvoice)

spec2 = importlib.util.spec_from_file_location(
    "prepare_indocorpus",
    (Path(__file__).parents[1] / "scripts" / "prepare_indocorpus.py").resolve(),
)
prepare_indocorpus = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(prepare_indocorpus)

# import generate by path
import importlib.util

spec_gen = importlib.util.spec_from_file_location(
    "generate_sample_audio",
    (Path(__file__).parents[1] / "scripts" / "generate_sample_audio.py").resolve(),
)
generate_mod = importlib.util.module_from_spec(spec_gen)
spec_gen.loader.exec_module(generate_mod)
generate = generate_mod.generate


def test_prepare_commonvoice(tmp_path):
    # Create a mini commonvoice-style structure
    cv_dir = tmp_path / "cv"
    clips = cv_dir / "clips"
    clips.mkdir(parents=True)

    # generate sample clips and a tsv
    wav = clips / "sample1.wav"
    generate(wav, duration_s=0.5)

    tsv = cv_dir / "train.tsv"
    tsv.write_text("client_id\tpath\tsentence\nabc\tsample1.wav\thalo dunia\n")

    out = tmp_path / "cv_manifest.json"
    prepare_commonvoice.prepare_commonvoice(cv_dir, "train", out)

    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["text"] == "halo dunia"


def test_prepare_indocorpus(tmp_path):
    # Create corpus structure
    corpus_dir = tmp_path / "indoc"
    wav_dir = corpus_dir / "WAV"
    txt_dir = corpus_dir / "TXT"
    wav_dir.mkdir(parents=True)
    txt_dir.mkdir(parents=True)

    # create a wav file
    wav = wav_dir / "G1_C1_0_S1.wav"
    generate(wav, duration_s=1.0)

    # create txt with two utterances
    txt = txt_dir / "G1_C1_0_S1.txt"
    txt.write_text("[0.00,0.50]\tS1\tM\thalo\n[0.50,1.00]\tS2\tF\tdunia\n")

    out_manifest = tmp_path / "indoc_manifest.json"
    out_rttm_dir = tmp_path / "rttm"

    prepare_indocorpus.prepare_indocorpus(corpus_dir, out_manifest, out_rttm_dir)

    man = json.loads(out_manifest.read_text())
    assert len(man) == 2
    assert man[0]["text"] == "halo"
    rttm_files = list(out_rttm_dir.glob("*.rttm"))
    assert len(rttm_files) == 1
