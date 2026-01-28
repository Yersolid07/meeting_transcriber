# Evaluation Metadata & Reproducibility Guidelines

This document describes what metadata and hyperparameters are recorded for each automated evaluation run and where to find the recorded values.

## Goals
- Ensure each evaluation report contains enough contextual information to reproduce the run.
- Provide a central document explaining where metadata is stored and what fields mean.

## Where metadata is captured
Each evaluation run (via `main.py --evaluate`) now includes a `metadata` block in the evaluation report. The fields recorded include (non-exhaustive):

- asr_backend: ASR backend used (whisper|whisperx|transformers|speechbrain)
- asr_model_id: ASR model identifier (HF id or local path)
- asr_language: language setting used for ASR
- asr_use_full_audio_for_segments: whether full-audio mapping to segments was used
- asr_whisperx_compute_type: WhisperX compute type (if used)
- asr_whisperx_vad_filter: WhisperX VAD filter setting
- asr_parallel_workers: per-segment ASR parallelism

- diarizer_vad_threshold: VAD energy threshold
- diarizer_min_speech_duration: Minimum speech duration for VAD
- diarizer_segment_window: Window size for embedding windows
- diarizer_segment_hop: Hop used for windows
- diarizer_clustering_method: clustering algorithm used
- diarizer_clustering_threshold: clustering distance threshold
- diarizer_min_cluster_size: minimum cluster size before merging
- diarizer_target_num_speakers: enforced target speaker count (if any)
- diarization_tune_result: dict of parameters selected by `auto_tune()` (if requested)
- tune_diarization_requested: boolean whether tuning was requested in pipeline config

- reference_transcript_provided: whether a reference transcript file was provided
- reference_diarization_provided: whether a reference RTTM was provided
- used_derived_rttm: whether a RTTM was derived from a speaker-labeled transcript

## Reproducibility tips
- To reproduce a run, keep a note of the exact command used (args printed by the CLI) and the evaluation report saved under `data/output/`.
- If tuning was run (`--tune-diarization`), consult the `diarization_tune_result` entry in the metadata to see chosen parameter values.

## Where to record benchmark runs
- Add entries to `docs/benchmarks/` (create a markdown file per benchmark) recording the evaluation report filename, the command used, and short interpretation.

## Example
An evaluation report will include a section similar to:

3. CONFIGURATION & HYPERPARAMETERS
--------------------------------------------------
   - asr_backend: whisper
   - asr_model_id: openai/whisper-small
   - diarization_tune_result:
       - clustering_threshold: 0.7
       - min_cluster_size: 2
   - tune_diarization_requested: True


----

## Summary Evaluation (ROUGE / BERTScore)

- New optional CLI flag `--reference-summary` accepts a plain text file containing a human-written summary for the audio file under evaluation.
- When provided, the pipeline will compute ROUGE (1/2/L) and BERTScore (P/R/F1) between the reference summary and the generated meeting summary and include aggregated values in the evaluation report.
- Note: the `evaluate` metrics require additional dependencies (e.g., `rouge_score`, `absl`, and `bertscore` related packages). If those packages are not installed, summary metric computation will gracefully fall back and metrics will be empty — the evaluation will still complete.

Keep this document short and update when new configs or tuning options are added.