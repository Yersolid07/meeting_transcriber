ASR Finetuning recipe (minimal)

This folder contains minimal artifacts to get started finetuning an ASR model using SpeechBrain-style hyperparameters.

Steps to create a small dev manifest and run a smoke training:

1. Generate a sample audio file for dev:

   python scripts/generate_sample_audio.py --out data/audio/sample_1s.wav

2. Create a manifest from the CSV example:

   python scripts/create_manifest.py --audio-dir data/audio --transcripts recipes/asr/dev_manifest_example.csv --out data/manifests/dev.json

3. Inspect `recipes/asr/hyperparams.yaml` and point `train_annotation` / `valid_annotation` to your manifests.

4. To run real training you'll need SpeechBrain installed; see `recipes/asr/train.py` for how to start.

Notes:
- This recipe is intentionally small and safe for initial hyperparameter development. For substantial training, follow SpeechBrain's full templates and use their hyperparams YAML structure.
