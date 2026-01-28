"""Simple ASR benchmark utilities

Usage examples:
  # Synthetic benchmark (fast, no models required)
  python scripts/benchmark_asr.py --synthetic

  # Real full-pipeline benchmark (may download models if not present)
  python scripts/benchmark_asr.py --sample-audio data/audio/kondisi_multispeaker/rapatsingkat.mp3 --runs 2

The script measures wall-clock time for serial vs parallel per-segment ASR using a lightweight synthetic workload
and optionally runs real pipeline runs to compare default vs `--quick-asr`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure local package imports work when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
except Exception:
    torch = None
    print("Warning: PyTorch not available, synthetic benchmark will adapt to run without torch.")

from src.pipeline import MeetingTranscriberPipeline, PipelineConfig
from src.transcriber import ASRConfig, ASRTranscriber


class DummyTranscriber(ASRTranscriber):
    def __init__(self, cfg=None):
        super().__init__(cfg or ASRConfig())
        self._pipeline = None

    def _load_model(self):
        return

    def _transcribe_audio(self, audio, sample_rate):
        # Simulate moderately expensive work
        time.sleep(0.12)
        return "dummy"


def synthetic_per_segment_benchmark():
    print("Running synthetic per-segment benchmark...")
    # Create fake waveform and segments
    sr = 16000
    duration_s = 5.0
    samples = int(duration_s * sr)
    # Support fallback to numpy if torch is not available (CI robustness)
    if torch is None:
        import numpy as np

        waveform = np.zeros((1, samples))
    else:
        waveform = torch.zeros((1, samples))

    class Seg:
        def __init__(self, start, end, speaker_id=0):
            self.start = start
            self.end = end
            self.speaker_id = speaker_id
            self.confidence = 1.0
            self.is_overlap = False

    segments = [Seg(i * 0.5, i * 0.5 + 0.4) for i in range(8)]

    # Serial
    t1 = DummyTranscriber(ASRConfig(parallel_workers=1))
    # If torch is not available, run a fallback synthetic benchmark that does not rely on ASRTranscriber internals
    if torch is None:
        # Simple synthetic timing: simulate serial vs parallel sleep-based execution
        start = time.perf_counter()
        for _ in range(8):
            time.sleep(0.12)
        serial_time = time.perf_counter() - start

        start = time.perf_counter()
        import concurrent.futures

        def _sleep_task():
            time.sleep(0.12)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(_sleep_task) for _ in range(8)]
            for f in concurrent.futures.as_completed(futures):
                _ = f.result()
        parallel_time = time.perf_counter() - start

        print(f"Serial time  : {serial_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup      : {serial_time / parallel_time:.2f}x")

        return {"serial_time": serial_time, "parallel_time": parallel_time, "speedup": serial_time / parallel_time}
    start = time.perf_counter()
    out1 = t1.transcribe_segments(waveform, segments, sample_rate=sr)
    serial_time = time.perf_counter() - start

    # Parallel
    t2 = DummyTranscriber(ASRConfig(parallel_workers=4))
    start = time.perf_counter()
    out2 = t2.transcribe_segments(waveform, segments, sample_rate=sr)
    parallel_time = time.perf_counter() - start

    print(f"Serial time  : {serial_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Speedup      : {serial_time / parallel_time:.2f}x")

    return {
        "serial_time": serial_time,
        "parallel_time": parallel_time,
        "speedup": serial_time / parallel_time,
    }


def real_pipeline_benchmark(
    sample_audio: str,
    runs: int = 1,
    quick_backend: str = None,
    quick_model: str = None,
    quick_whisperx_compute_type: str = "int8",
    quick_parallel_workers: int = None,
):
    print("Running real pipeline benchmark (may download models, this can take time)")

    if not os.path.exists(sample_audio):
        print(f"Sample audio not found: {sample_audio}")
        return

    # Default pipeline
    cfg_default = PipelineConfig()
    pipeline_default = MeetingTranscriberPipeline(cfg_default)

    # Quick ASR pipeline (allow forcing backend/model)
    if quick_backend:
        # Force the backend/model and avoid the default quick_asr model remapping
        cfg_quick = PipelineConfig(
            quick_asr=False,
            asr_backend=quick_backend,
            asr_model_id=quick_model or PipelineConfig().asr_model_id,
            whisperx_compute_type=quick_whisperx_compute_type or "int8",
            asr_parallel_workers=quick_parallel_workers,
        )
    else:
        cfg_quick = PipelineConfig(quick_asr=True, asr_parallel_workers=quick_parallel_workers)
    pipeline_quick = MeetingTranscriberPipeline(cfg_quick)

    times_default = []
    times_quick = []

    for i in range(runs):
        print(f"Run {i+1}/{runs} — default...")
        start = time.perf_counter()
        pipeline_default.process(audio_path=sample_audio, title=f"Benchmark Default Run {i+1}")
        t = time.perf_counter() - start
        times_default.append(t)
        print(f"  default run time: {t:.2f}s")

        print(f"Run {i+1}/{runs} — quick-asr...")
        start = time.perf_counter()
        pipeline_quick.process(audio_path=sample_audio, title=f"Benchmark Quick Run {i+1}")
        t = time.perf_counter() - start
        times_quick.append(t)
        print(f"  quick-asr run time: {t:.2f}s")

    if times_default:
        print(f"Default avg: {sum(times_default)/len(times_default):.2f}s")
    if times_quick:
        print(f"Quick avg  : {sum(times_quick)/len(times_quick):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="ASR benchmark utilities")
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic per-segment benchmark (fast)"
    )
    parser.add_argument(
        "--sample-audio",
        type=str,
        default=None,
        help="Path to sample audio for real pipeline benchmark",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for real benchmark")
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON results for synthetic benchmark",
    )
    parser.add_argument(
        "--quick-backend",
        type=str,
        default=None,
        help="Force quick pipeline ASR backend (e.g., whisperx, whisper, transformers, speechbrain)",
    )
    parser.add_argument(
        "--quick-model",
        type=str,
        default=None,
        help="Force quick pipeline ASR model id (e.g., large-v3-turbo)",
    )
    parser.add_argument(
        "--quick-whisperx-compute-type",
        type=str,
        default="int8",
        help="WhisperX compute type for forced quick pipeline (int8|float16|int8_float16)",
    )
    parser.add_argument(
        "--quick-parallel-workers",
        type=int,
        default=None,
        help="Override parallel workers for quick pipeline (int)",
    )

    args = parser.parse_args()

    if args.synthetic:
        res = synthetic_per_segment_benchmark()
        if args.json_output:
            os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
            with open(args.json_output, "w") as fh:
                json.dump(res, fh)
            print(f"Written synthetic benchmark JSON to: {args.json_output}")

    if args.sample_audio:
        real_pipeline_benchmark(
            args.sample_audio,
            runs=args.runs,
            quick_backend=args.quick_backend,
            quick_model=args.quick_model,
            quick_whisperx_compute_type=args.quick_whisperx_compute_type,
            quick_parallel_workers=args.quick_parallel_workers,
        )


if __name__ == "__main__":
    main()
