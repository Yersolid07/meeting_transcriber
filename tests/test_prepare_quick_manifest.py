import json
from pathlib import Path

from recipes.asr import prepare_quick_manifest


def test_prepare_quick_manifest(tmp_path):
    inp = tmp_path / "in.jsonl"
    lines = [{"a": i} for i in range(10)]
    with open(inp, "w", encoding="utf-8") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")
    out = tmp_path / "out.jsonl"
    prepare_quick_manifest.main(["--input", str(inp), "--output", str(out), "--max-examples", "3"])
    assert out.exists()
    with open(out, "r", encoding="utf-8") as fh:
        out_lines = [l.strip() for l in fh if l.strip()]
    assert len(out_lines) == 3
