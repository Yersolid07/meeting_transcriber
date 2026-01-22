# Integration with SpeechBrain

## Purpose
Map SpeechBrain components (ASR, VAD, speaker-embedding) to `meeting_transcriber` modules, provide configuration examples, Windows/HF recommendations, and an implementation checklist.

## Overview (high-level)
- SpeechBrain ASR (EncoderASR / EncoderDecoderASR) → optional backend for `src/transcriber.py`
- SpeechBrain VAD (inference.VAD) → optional backend for `src/diarization.py` (VAD stage)
- SpeechBrain speaker embeddings (spkrec-ecapa) → can replace or complement current embedding loader in `src/diarization.py`

## Config snippets (example)
```yaml
asr:
  backend: speechbrain  # or 'hf' (default)
  model_id: speechbrain/hub-model-id

diariazation:
  vad_backend: speechbrain  # or 'energy'(default)
  embedding:
    model_id: speechbrain/spkrec-ecapa-voxceleb
```

## Windows / HF hub notes
- Windows can raise WinError 1314 when HF hub attempts to create symlinks. Recommend setting:

```bash
export HF_HUB_DISABLE_SYMLINKS=1  # or set in CI environment variables
```

- Implement fallback strategies in embedding loader: try `snapshot_download` → if symlink error, try `hf_hub_download` or copy; final fallback: deterministic MFCC embedding.

## Implementation checklist
1. Add `docs/integration_with_speechbrain.md` (this doc) ✅
2. Add optional dependency (extras_require) and note in README
3. Add SpeechBrain ASR wrapper (config-driven), lazy-load models and mock-friendly loader
4. Add optional SpeechBrain VAD wrapper and integrate into diarization
5. Unit tests that mock model downloads (avoid heavy downloads in CI)
6. Add CI matrix entry for Windows with `HF_HUB_DISABLE_SYMLINKS=1`

## Test ideas
- Mock `speechbrain.pretrained.EncoderASR.from_hparams` to return a dummy object that has `transcribe_file` returning deterministic text
- Test mapping of full-ASR output to diarization segments
- Test that embedding loader falls back to MFCC when HF symlink errors are raised

## Notes / Recommendations
- Keep SpeechBrain integration optional to avoid increasing base install size.
- Ensure code paths have small, fast unit tests and separate integration tests for manual runs (or separate CI job with cached models).

---

If you'd like, I can now implement the SpeechBrain ASR wrapper and add the unit tests for it (I already marked the corresponding todo as next).

**Training for research:** see `docs/training_for_research.md` for guidance on producing experiment artifacts, metrics (WER, CER, DER), and recommended hyperparams for speaker-differentiation studies.
