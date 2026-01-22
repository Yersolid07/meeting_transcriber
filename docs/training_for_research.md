# Training for research: ASR + Speaker differentiation 📊🔬

This guide explains how the `recipes/asr/train_full.py` entrypoint supports reproducible experiments for both improving ASR word recognition and speaker differentiation.

## Research goals
- Improve ASR word recognition (metrics: WER, CER) ✅
- Improve speaker differentiation / identification (metrics: speaker ID accuracy, EER, DER when RTTMs exist) ✅
- Produce per-utterance artifacts for downstream analysis (per-utterance hypothesis, timestamps, speaker ids, confidence) ✅

## What the training entrypoint produces (artifacts)
- experiments/<run_name>/predictions.jsonl — per-utterance object: {wav, start, end, ref, hyp, speaker, epoch?}
- experiments/<run_name>/metrics.csv — epoch-level metrics (epoch, split, wer, cer, loss)
- experiments/<run_name>/checkpoints/ — saved model checkpoints
- experiments/<run_name>/tensorboard/ — TensorBoard logs for scalars and metrics
- evaluation summary in JSON and CSV via `scripts/eval_asr.py`

## Running a reproducible experiment (smoke)
- Dry-run (fast smoke test, no SpeechBrain):
  python recipes/asr/train_full.py --small-data --run-name my_experiment

- SpeechBrain-based training (mock mode for CI and a real run):
  - Mocked (CI-friendly) run that exercises the SpeechBrain integration without requiring the package:
    python recipes/asr/train_speechbrain.py --train-manifest data/train.jsonl --valid-manifest data/valid.jsonl --run-name exp1 --epochs 5 --use-speaker-head --mock
  - Full (real) run with SpeechBrain installed:
    python recipes/asr/train_speechbrain.py --hparams path/to/hparams.yaml --train-manifest data/train.jsonl --valid-manifest data/valid.jsonl --run-name exp1 --epochs 20 --use-speaker-head
### Quick runs and model selection guidance
- Use manifest sampling for fast experiments (recommended for research iterations):
  - Create a smaller manifest with `recipes/asr/prepare_quick_manifest.py`:

    python recipes/asr/prepare_quick_manifest.py --input data/manifests/train.json --output data/manifests/train_small.jsonl --max-examples 200

  - Or use `--sample-frac 0.1` to use 10% of the data.
- When running training, pass `--max-examples` or `--sample-frac` to `train_speechbrain.py` to restrict examples used per manifest:

    python recipes/asr/train_speechbrain.py --train-manifest data/manifests/train_small.jsonl --valid-manifest data/manifests/dev_small.jsonl --run-name quick_test --epochs 5 --use-speaker-head --max-examples 200

- Model choice (recommendations):
  - Small/faster models for experimentation: `wav2vec2-base`, `facebook/wav2vec2-base-960h`, `openai/whisper-small` (use for noisy/meeting audio), or other small checkpoints.
  - For longer experiments and final runs: larger models (base/large) may give better WER but require more compute.
  - Always start with a small subset and a quick model; once hyperparams converge, scale up.
Note: the mocked mode produces identical research artifacts (predictions, metrics, checkpoints) so downstream analysis and scripts can be validated in CI without heavy dependencies.

## Research-oriented config knobs (suggested hyperparams)
- `epochs` : number of epochs
- `lr` : learning rate
- `batch_size` : batch size
- `use_speaker_head` : boolean. If true, the training loop should add a speaker-classifier head trained with cross-entropy on `speaker` labels in manifests.
- `speaker_loss_weight` : float. Weight for speaker loss in a multi-task objective.

## Evaluations to add (recommended)
- WER and CER (already supported via `scripts/eval_asr.py`)
- Speaker accuracy / confusion (supported via `scripts/eval_speaker.py`) ✅
- DER (requires RTTM — produced by `scripts/prepare_indocorpus.py`)
- Speaker verification EER using trial lists (use SpeechBrain's speaker-eval utilities), and speaker verification pipelines for EER computation

## Notes for implementers
- The included `train_full.py` is a scaffold with a robust dry-run mode and placeholders for a full SpeechBrain recipe implementation. Use the scaffold to integrate a Brain subclass, dataset creation, and optimizer/encoder details.
- Always write `predictions.jsonl` and `metrics.csv` in the experiment folder to enable consistent downstream analysis and plotting.

---
**Tip:** For reproducible research, log the exact git commit, conda/pip environment and full hyperparams file into `experiments/<run>/metadata.json`.
