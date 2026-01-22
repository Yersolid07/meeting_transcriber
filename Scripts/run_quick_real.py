import importlib.util
import sys
from pathlib import Path

# ensure project root is on sys.path so intra-repo imports work
sys.path.insert(0, str(Path(".").resolve()))

spec = importlib.util.spec_from_file_location(
    "sb_helpers", str(Path("recipes/asr/sb_helpers.py").resolve())
)
sb_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb_helpers)
res = sb_helpers.full_train(
    {},
    "data/manifests/train_small.jsonl",
    "data/manifests/valid_small.jsonl",
    Path("experiments/quick_real"),
    epochs=2,
    use_speaker_head=True,
    max_examples=2,
)
print("Done:", res)
