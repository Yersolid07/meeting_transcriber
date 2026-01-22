"""
Minimal ASR finetuning entrypoint (SpeechBrain-compatible).

This script expects SpeechBrain to be installed. It loads a HyperPyYAML file and
invokes a simple training recipe. It's intentionally minimal: replace or extend
with a full SpeechBrain recipe if you plan to run large-scale training.
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Train ASR with SpeechBrain-style hyperparams")
    parser.add_argument("hparams", type=str, help="Path to hyperparams.yaml")
    args = parser.parse_args()

    try:
        # Import lazily so this file can be present without speechbrain installed
        from hyperpyyaml import load_hyperpyyaml
        from speechbrain.utils.checkpoints import Checkpointer
        from speechbrain.utils.data_utils import download_file
        from speechbrain.utils.run_opts import RunOpts

        # Actual training utilities come from SpeechBrain templates; we keep this simple
    except Exception as e:
        print("Error: SpeechBrain and hyperpyyaml must be installed to run training.")
        print("Install with: pip install 'speechbrain[all]' hyperpyyaml")
        print(f"Detail: {e}")
        sys.exit(1)

    if not os.path.exists(args.hparams):
        print(f"Error: hyperparams file not found: {args.hparams}")
        sys.exit(1)

    with open(args.hparams, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides={})

    print(
        "Loaded hyperparams. This script is a starter; link it to a full SpeechBrain recipe for production."
    )
    print("Suggested next steps:")
    print(
        " - Hook the hyperparams into a SpeechBrain Brain-based training loop (see SpeechBrain templates)"
    )
    print(
        " - Prepare manifests with scripts/create_manifest.py and point `train_annotation`/`valid_annotation` accordingly"
    )


if __name__ == "__main__":
    main()
