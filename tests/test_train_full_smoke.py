import importlib.util
import json
import shutil
from pathlib import Path

# Import scripts by path to avoid requiring package installation in tests
spec_tf = importlib.util.spec_from_file_location(
    "train_full", str(Path("recipes/asr/train_full.py").resolve())
)
train_full = importlib.util.module_from_spec(spec_tf)
spec_tf.loader.exec_module(train_full)

spec_eval = importlib.util.spec_from_file_location(
    "eval_asr", str(Path("scripts/eval_asr.py").resolve())
)
eval_asr = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(eval_asr)


def test_train_full_small_data_smoke(tmp_path):
    outdir = tmp_path / "experiments"
    args = ["--small-data", "--run-name", "ci_smoke", "--outdir", str(outdir), "--epochs", "2"]
    # run dry-run smoke
    rc = train_full.main(args)
    assert rc == 0
    run_dir = outdir / "ci_smoke"
    assert run_dir.exists()
    preds = run_dir / "predictions.jsonl"
    metrics = run_dir / "metrics.csv"
    ckpt_dir = run_dir / "checkpoints"
    assert preds.exists()
    assert metrics.exists()
    assert ckpt_dir.exists()

    # run evaluation
    eval_out = tmp_path / "eval_out"
    eval_out.mkdir()
    eval_args = [str(preds), "--outdir", str(eval_out)]
    eval_asr.main(eval_args)
    sum_json = eval_out / "metrics_summary.json"
    sum_csv = eval_out / "metrics_summary.csv"
    assert sum_json.exists()
    assert sum_csv.exists()
    with open(sum_json, "r", encoding="utf-8") as fh:
        s = json.load(fh)
    assert s  # not empty
