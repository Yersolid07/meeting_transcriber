#!/usr/bin/env python3
"""
Main Entry Point - Meeting Transcriber System
==============================================

Automatic Meeting Minutes Generation using SpeechBrain + BERT

Usage:
    # Basic transcription
    python main.py --audio meeting.wav --title "Team Meeting"

    # With evaluation
    python main.py --audio meeting.wav --evaluate --reference transcript.txt

    # Batch processing
    python main.py --batch ./audio_folder/ --output ./results/

    # Specify number of speakers
    python main.py --audio meeting.wav --speakers 4
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from src.evaluator import EvaluationResult, Evaluator
from src.pipeline import MeetingTranscriberPipeline, PipelineConfig, PipelineResult
from src.utils import (
    format_duration,
    list_audio_files,
    parse_rttm_file,
    parse_transcript_file,
    validate_audio_file,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sistem Notulensi Rapat Otomatis (SpeechBrain + BERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
==================

  # Transkripsi dasar
  python main.py --audio rapat.wav

  # Dengan detail rapat
  python main.py --audio rapat.wav --title "Rapat Sprint" --speakers 4 --location "Zoom"

  # Dengan evaluasi WER
  python main.py --audio rapat.wav --evaluate --reference transkrip_manual.txt

  # Batch processing
  python main.py --batch ./folder_audio/ --output ./hasil/

Untuk dokumentasi lengkap, lihat README.md
        """,
    )

    # Input arguments
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--audio", "-a", type=str, help="Path ke file audio (.wav, .mp3, .m4a)"
    )
    input_group.add_argument(
        "--batch", "-b", type=str, help="Direktori berisi file audio untuk batch processing"
    )

    # Meeting metadata
    meta_group = parser.add_argument_group("Meeting Metadata")
    meta_group.add_argument(
        "--title",
        "-t",
        type=str,
        default="Notulensi Rapat",
        help="Judul rapat (default: 'Notulensi Rapat')",
    )
    meta_group.add_argument(
        "--date", "-d", type=str, default=None, help="Tanggal rapat (default: hari ini)"
    )
    meta_group.add_argument("--location", "-l", type=str, default="", help="Lokasi/platform rapat")
    meta_group.add_argument(
        "--speakers",
        "-s",
        type=int,
        default=None,
        help="Jumlah speaker (opsional, auto-detect jika tidak disebut)",
    )

    meta_group.add_argument(
        "--speaker-map",
        type=str,
        default=None,
        help='Path ke JSON/YAML file yang memetakan speaker label (SPEAKER_00) ke nama (mis: {"SPEAKER_00": "Budi"})',
    )

    meta_group.add_argument(
        "--tune-diarization",
        action="store_true",
        help="Jalankan tuning hyperparameter diarization sebelum clustering (tries several settings)",
    )

    meta_group.add_argument(
        "--target-speakers",
        type=int,
        default=None,
        help="Target jumlah speaker untuk dipaksakan (opsional). Jika diset, pipeline akan mencoba merge cluster hingga jumlah ini.",
    )

    # Performance and tuning flags
    misc_group = parser.add_argument_group("Performance")
    misc_group.add_argument(
        "--fast",
        action="store_true",
        help="Aktifkan modus cepat (mengorbankan sedikit akurasi demi kinerja)",
    )
    misc_group.add_argument(
        "--preset",
        type=str,
        choices=["deployment", "balanced", "fast", "accurate"],
        default="deployment",
        help="Preset pipeline yang merekomendasikan konfigurasi (default: deployment - prefer 'large-v3-turbo')",
    )
    misc_group.add_argument(
        "--quick-asr",
        action="store_true",
        help="Gunakan backend ASR lebih ringan/cepat (model kecil) jika memungkinkan (opsional override)",
    )
    misc_group.add_argument(
        "--prefer-whisper-small",
        action="store_true",
        help="Paksa penggunaan `openai/whisper-small` untuk ASR (lebih cepat, lebih ringan)",
    )
    misc_group.add_argument(
        "--cst-hz",
        type=float,
        default=None,
        help="(opsional) Approximate Continuous Speech Tokenizer token rate in Hz (e.g., 7.5). Applies lossy compression preprocessor for speed.",
    )
    misc_group.add_argument(
        "--diarization-compare",
        action="store_true",
        help="Jalankan perbandingan metode diarization (agglomerative vs spectral) selama evaluasi",
    )
    misc_group.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Override jumlah worker paralel untuk per-segment ASR (default: auto berdasarkan CPU atau preset)",
    )
    misc_group.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Nonaktifkan cache embeddings di disk (default: aktif)",
    )

    # Output settings
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/output",
        help="Direktori output (default: ./data/output)",
    )
    output_group.add_argument(
        "--filename",
        "-f",
        type=str,
        default=None,
        help="Nama file output (auto-generate jika tidak disebut)",
    )

    # Evaluation
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--evaluate", "-e", action="store_true", help="Aktifkan mode evaluasi")
    eval_group.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Path ke file reference transcript untuk WER",
    )
    eval_group.add_argument(
        "--reference-rttm", type=str, default=None, help="Path ke file RTTM untuk DER"
    )
    eval_group.add_argument(
        "--reference-summary",
        type=str,
        default=None,
        help="Path ke file reference summary untuk evaluasi ringkasan (ROUGE/BERTScore)",
    )
    eval_group.add_argument(
        "--condition",
        type=str,
        default="unknown",
        help="Nama kondisi untuk evaluasi (misal: bersih, noisy)",
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--asr-model",
        type=str,
        default="large-v3-turbo",
        help="ASR model (HF model id / alias / path folder model lokal). Default: large-v3-turbo for better accuracy.",
    )
    model_group.add_argument(
        "--asr-backend",
        type=str,
        default="whisper",
        choices=["whisperx", "whisper", "transformers", "speechbrain"],
        help="Backend ASR (default: whisper)",
    )
    model_group.add_argument(
        "--asr-language",
        type=str,
        default="id",
        help="Kode bahasa (mis: id, en, auto). Untuk WhisperX: 'auto' = autodetect.",
    )
    model_group.add_argument(
        "--whisperx-compute-type",
        type=str,
        default="auto",
        help="WhisperX compute_type (auto|float16|int8|int8_float16). Default auto.",
    )
    model_group.add_argument(
        "--whisperx-no-vad-filter",
        action="store_true",
        help="Matikan VAD filter WhisperX (kadang berguna untuk audio sangat pendek/aneh).",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device untuk inferensi (default: auto)",
    )

    # Misc
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--verbose", "-v", action="store_true", default=True, help="Output verbose"
    )
    misc_group.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    misc_group.add_argument(
        "--no-save-intermediate", action="store_true", help="Jangan simpan hasil intermediate"
    )

    return parser.parse_args()


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███╗   ███╗███████╗███████╗████████╗██╗███╗   ██╗ ██████╗     ║
║   ████╗ ████║██╔════╝██╔════╝╚══██╔══╝██║████╗  ██║██╔════╝     ║
║   ██╔████╔██║█████╗  █████╗     ██║   ██║██╔██╗ ██║██║  ███╗    ║
║   ██║╚██╔╝██║██╔══╝  ██╔══╝     ██║   ██║██║╚██╗██║██║   ██║    ║
║   ██║ ╚═╝ ██║███████╗███████╗   ██║   ██║██║ ╚████║╚██████╔╝    ║
║   ╚═╝     ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝     ║
║                                                                  ║
║          TRANSCRIBER - Notulensi Rapat Otomatis                 ║
║                  SpeechBrain + BERT Pipeline                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def process_single_audio(args, pipeline: MeetingTranscriberPipeline) -> PipelineResult:
    """Process a single audio file"""

    # Validate audio file
    validate_audio_file(args.audio)

    print(f"\n{'='*60}")
    print(f"Processing: {args.audio}")
    print(f"{'='*60}")

    # Run pipeline
    result = pipeline.process(
        audio_path=args.audio,
        title=args.title,
        date=args.date,
        location=args.location,
        num_speakers=args.speakers,
        output_filename=args.filename,
    )

    # Print summary
    print_result_summary(result)

    # Run evaluation if requested
    if args.evaluate:
        run_evaluation(args, pipeline, result)

    return result


def process_batch(args, pipeline: MeetingTranscriberPipeline) -> List[PipelineResult]:
    """Process multiple audio files in a directory"""

    batch_dir = Path(args.batch)

    if not batch_dir.is_dir():
        print(f"Error: Direktori tidak ditemukan: {args.batch}")
        sys.exit(1)

    # Find audio files
    audio_files = list_audio_files(batch_dir)

    if not audio_files:
        print(f"Tidak ada file audio ditemukan di: {args.batch}")
        sys.exit(1)

    print(f"\nDitemukan {len(audio_files)} file audio untuk diproses")
    print("-" * 60)

    results = []
    failed = []

    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_path.name}")

        try:
            # Generate title from filename
            title = audio_path.stem.replace("_", " ").replace("-", " ").title()

            result = pipeline.process(
                audio_path=str(audio_path),
                title=title,
                date=args.date,
                location=args.location,
                num_speakers=args.speakers,
            )
            results.append(result)

            # Clear state for next file
            pipeline.clear_state()

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            failed.append((audio_path.name, str(e)))
            continue

    # Print batch summary
    print_batch_summary(results, failed, audio_files)

    return results


def run_evaluation(args, pipeline: MeetingTranscriberPipeline, result: PipelineResult):
    """Run evaluation with reference files"""

    print(f"\n{'='*60}")
    print("EVALUASI")
    print(f"{'='*60}")

    reference_transcript = None
    reference_diarization = None

    # Load reference transcript
    if args.reference:
        if not os.path.exists(args.reference):
            print(f"Warning: File reference tidak ditemukan: {args.reference}")
        else:
            reference_transcript = parse_transcript_file(args.reference)
            print(f"Reference transcript loaded: {len(reference_transcript.split())} words")

    # Load reference diarization
    if args.reference_rttm:
        if not os.path.exists(args.reference_rttm):
            print(f"Warning: File RTTM tidak ditemukan: {args.reference_rttm}")
        else:
            reference_diarization = parse_rttm_file(args.reference_rttm)
            print(f"Reference diarization loaded: {len(reference_diarization)} segments")
    else:
        # If user didn't provide an RTTM, try to find a *_vibevoice.rttm for the sample
        try:
            audio_stem = Path(args.audio).stem
            cand = Path("data/ground_truth") / f"{audio_stem}_vibevoice.rttm"
            if cand.exists():
                reference_diarization = parse_rttm_file(str(cand))
                print(f"Reference RTTM auto-loaded: {cand} ({len(reference_diarization)} segments)")
        except Exception:
            pass

    # Load reference summary (optional)
    reference_summary = None
    if getattr(args, "reference_summary", None):
        if not os.path.exists(args.reference_summary):
            print(f"Warning: File reference summary tidak ditemukan: {args.reference_summary}")
        else:
            try:
                reference_summary = Path(args.reference_summary).read_text(encoding="utf-8")
                print(f"Reference summary loaded (len={len(reference_summary.split())} words)")
            except Exception as e:
                print(f"Warning: gagal membaca file summary: {e}")

    # Run evaluation
    eval_result = pipeline.evaluate(
        reference_transcript=reference_transcript,
        reference_diarization=reference_diarization,
        reference_summary=reference_summary,
        sample_name=Path(args.audio).stem,
        condition=args.condition,
    )

    # Print evaluation results
    print_evaluation_results(eval_result)

    # Generate and save report
    evaluator = Evaluator(output_dir=args.output)

    wer_results = [eval_result.wer_result] if eval_result.wer_result else []
    der_results = [eval_result.der_result] if eval_result.der_result else []

    # Pass evaluation metadata for reproducibility & documentation
    report = evaluator.generate_evaluation_report(
        wer_results=wer_results,
        der_results=der_results,
        summary_results=[eval_result.summary_result] if eval_result.summary_result else None,
        sample_names=[eval_result.sample_name],
        condition_name=args.condition,
        metadata=eval_result.metadata,
    )

    # Save report
    report_path = evaluator.save_report(
        report,
        f"evaluation_{eval_result.sample_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    print(f"\nEvaluation report saved: {report_path}")


def print_result_summary(result: PipelineResult):
    """Print processing result summary"""
    print(f"\n{'='*60}")
    print("HASIL PEMROSESAN")
    print(f"{'='*60}")
    print(f"  Audio Duration    : {format_duration(result.audio_duration)}")
    print(f"  Speakers Found    : {result.num_speakers}")
    print(f"  Total Segments    : {result.num_segments}")
    print(f"  Total Words       : {result.total_words}")
    print(f"  Processing Time   : {format_duration(result.processing_time)}")
    print(f"  Output Document   : {result.document_path}")
    print(f"{'='*60}")


def print_evaluation_results(eval_result: EvaluationResult):
    """Print evaluation results"""
    print("\n--- Hasil Evaluasi ---")

    if eval_result.wer_result:
        wer = eval_result.wer_result
        print("\nWord Error Rate (WER):")
        print(f"  WER           : {wer.wer:.4f} ({wer.wer*100:.2f}%)")
        print(f"  Substitutions : {wer.substitutions}")
        print(f"  Deletions     : {wer.deletions}")
        print(f"  Insertions    : {wer.insertions}")
        print(f"  Correct       : {wer.hits}")

    if eval_result.der_result:
        der = eval_result.der_result
        print("\nDiarization Error Rate (DER):")
        print(f"  DER               : {der.der:.4f} ({der.der*100:.2f}%)")
        print(f"  Missed Speech     : {der.missed_speech:.4f} ({der.missed_speech*100:.2f}%)")
        print(f"  False Alarm       : {der.false_alarm:.4f} ({der.false_alarm*100:.2f}%)")
        print(
            f"  Speaker Confusion : {der.speaker_confusion:.4f} ({der.speaker_confusion*100:.2f}%)"
        )

    # Summary metrics (if available)
    if eval_result.summary_result:
        s = eval_result.summary_result
        print("\nRingkasan (Summary) Evaluation:")
        try:
            print(f"  ROUGE-1 F1    : {s.rouge.get('rouge1_f', 0.0):.4f}")
            print(f"  ROUGE-2 F1    : {s.rouge.get('rouge2_f', 0.0):.4f}")
            print(f"  ROUGE-L F1    : {s.rouge.get('rougel_f', 0.0):.4f}")
            print(f"  BERTScore F1  : {s.bertscore.get('bertscore_f1', 0.0):.4f}")
        except Exception as e:
            print(f"  (failed to print summary metrics: {e})")


def print_batch_summary(
    results: List[PipelineResult], failed: List[tuple], total_files: List[Path]
):
    """Print batch processing summary"""
    print(f"\n{'='*60}")
    print("RINGKASAN BATCH PROCESSING")
    print(f"{'='*60}")
    print(f"  Total files       : {len(total_files)}")
    print(f"  Successful        : {len(results)}")
    print(f"  Failed            : {len(failed)}")

    if results:
        total_duration = sum(r.audio_duration for r in results)
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / len(results)

        print(f"  Total audio       : {format_duration(total_duration)}")
        print(f"  Total proc. time  : {format_duration(total_time)}")
        print(f"  Avg time/file     : {format_duration(avg_time)}")

    if failed:
        print("\n  Failed files:")
        for filename, error in failed:
            print(f"    - {filename}: {error[:50]}...")

    print(f"{'='*60}")


def main():
    """Main entry point"""
    args = parse_args()

    # Handle quiet mode
    verbose = not args.quiet and args.verbose

    if verbose:
        print_banner()

    # Validate input
    if not args.audio and not args.batch:
        print("Error: Harap tentukan --audio atau --batch")
        print("Gunakan --help untuk informasi penggunaan")
        sys.exit(1)

    # Determine device
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("\nDevice: {}".format(device))
        print("ASR Backend: {}".format(args.asr_backend))
        print("ASR Model: {}".format(args.asr_model))
        print("ASR Language: {}".format(args.asr_language))
        print("Output Dir: {}".format(args.output))

    # Initialize pipeline
    config = PipelineConfig(
        output_dir=args.output,
        asr_model_id=args.asr_model,
        asr_backend=args.asr_backend,
        asr_language=args.asr_language,
        whisperx_compute_type=args.whisperx_compute_type,
        whisperx_vad_filter=not args.whisperx_no_vad_filter,
        device=device,
        verbose=verbose,
        save_intermediate=not args.no_save_intermediate,
        fast_mode=args.fast,
        quick_asr=args.quick_asr,
        prefer_whisper_small=args.prefer_whisper_small,
        cst_hz=args.cst_hz,
        diarization_compare=args.diarization_compare,
        embedding_cache=not args.no_embedding_cache,
        target_speakers=args.target_speakers,
        # New flags
        asr_parallel_workers=args.parallel_workers,
        speaker_map_path=args.speaker_map,
        tune_diarization=args.tune_diarization,
        num_speakers=args.speakers,
        preset=args.preset,
    )

    pipeline = MeetingTranscriberPipeline(config)

    # Run processing
    try:
        if args.batch:
            process_batch(args, pipeline)
        else:
            process_single_audio(args, pipeline)

        print("\nSelesai!")

    except KeyboardInterrupt:
        print("\n\nProses dibatalkan oleh user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
