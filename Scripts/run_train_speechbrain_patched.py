import torchaudio

# Ensure SpeechBrain's torchaudio check succeeds by providing a list_audio_backends function
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import importlib.util
import sys
from pathlib import Path

# Ensure repo root is importable for intra-repo imports (e.g., src)
sys.path.insert(0, str(Path(".").resolve()))
spec = importlib.util.spec_from_file_location(
    "train_speechbrain", str(Path("recipes/asr/train_speechbrain.py").resolve())
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Forward CLI args to the module's main function
sys.exit(mod.main(sys.argv[1:]))
