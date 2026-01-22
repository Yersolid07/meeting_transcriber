"""
Evaluation Module
=================
Implements WER, DER, and other metrics for thesis validation.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from jiwer import cer, mer, process_words, wer, wil

    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("[Evaluator] Warning: jiwer not installed. WER calculation will use fallback.")


@dataclass
class WERResult:
    """Word Error Rate evaluation result"""

    wer: float
    mer: float = 0.0  # Match Error Rate
    wil: float = 0.0  # Word Information Lost
    cer: float = 0.0  # Character Error Rate
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    reference_length: int = 0
    hypothesis_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "wer": self.wer,
            "mer": self.mer,
            "wil": self.wil,
            "cer": self.cer,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "reference_length": self.reference_length,
            "hypothesis_length": self.hypothesis_length,
        }


@dataclass
class DERResult:
    """Diarization Error Rate evaluation result"""

    der: float
    missed_speech: float = 0.0
    false_alarm: float = 0.0
    speaker_confusion: float = 0.0
    total_duration: float = 0.0
    num_speakers_ref: int = 0
    num_speakers_hyp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "der": self.der,
            "missed_speech": self.missed_speech,
            "false_alarm": self.false_alarm,
            "speaker_confusion": self.speaker_confusion,
            "total_duration": self.total_duration,
            "num_speakers_ref": self.num_speakers_ref,
            "num_speakers_hyp": self.num_speakers_hyp,
        }


@dataclass
class EvaluationResult:
    """Combined evaluation result"""

    sample_name: str
    condition: str
    wer_result: Optional[WERResult] = None
    der_result: Optional[DERResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """
    Evaluation metrics calculator for ASR and Diarization.

    Provides:
        - WER (Word Error Rate) for ASR evaluation
        - DER (Diarization Error Rate) for speaker diarization evaluation
        - Report generation for thesis documentation

    Example:
        >>> evaluator = Evaluator()
        >>> wer_result = evaluator.calculate_wer(reference, hypothesis)
        >>> print(f"WER: {wer_result.wer:.2%}")
    """

    def __init__(self, output_dir: str = "./data/output"):
        """
        Initialize Evaluator.

        Args:
            output_dir: Directory for evaluation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Text Preprocessing
    # =========================================================================

    @staticmethod
    def preprocess_text(
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        normalize_whitespace: bool = True,
        remove_filler_words: bool = False,
    ) -> str:
        """
        Preprocess text for fair WER comparison.

        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            normalize_whitespace: Normalize whitespace
            remove_filler_words: Remove filler words (eh, um, etc.)

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Lowercase
        if lowercase:
            text = text.lower()

        # Remove punctuation
        if remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        # Remove filler words (common in Indonesian)
        if remove_filler_words:
            filler_words = ["eh", "em", "um", "uh", "ah", "hmm", "eee", "anu"]
            pattern = r"\b(" + "|".join(filler_words) + r")\b"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Normalize whitespace
        if normalize_whitespace:
            text = " ".join(text.split())

        return text.strip()

    # =========================================================================
    # WER Calculation
    # =========================================================================

    def calculate_wer(self, reference: str, hypothesis: str, preprocess: bool = True) -> WERResult:
        """
        Calculate Word Error Rate and related metrics.

        WER = (S + D + I) / N
        where:
            S = Substitutions
            D = Deletions
            I = Insertions
            N = Total words in reference

        Args:
            reference: Ground truth text
            hypothesis: ASR output text
            preprocess: Apply text preprocessing

        Returns:
            WERResult with detailed metrics
        """
        # Preprocess
        if preprocess:
            reference = self.preprocess_text(reference)
            hypothesis = self.preprocess_text(hypothesis)

        # Handle empty cases
        if not reference:
            return WERResult(
                wer=1.0 if hypothesis else 0.0,
                reference_length=0,
                hypothesis_length=len(hypothesis.split()) if hypothesis else 0,
            )

        if not hypothesis:
            return WERResult(
                wer=1.0,
                deletions=len(reference.split()),
                reference_length=len(reference.split()),
                hypothesis_length=0,
            )

        # Use jiwer if available
        if JIWER_AVAILABLE:
            try:
                wer_score = wer(reference, hypothesis)
                mer_score = mer(reference, hypothesis)
                wil_score = wil(reference, hypothesis)
                cer_score = cer(reference, hypothesis)

                # Get detailed breakdown
                output = process_words(reference, hypothesis)

                return WERResult(
                    wer=wer_score,
                    mer=mer_score,
                    wil=wil_score,
                    cer=cer_score,
                    substitutions=output.substitutions,
                    deletions=output.deletions,
                    insertions=output.insertions,
                    hits=output.hits,
                    reference_length=len(reference.split()),
                    hypothesis_length=len(hypothesis.split()),
                )
            except Exception as e:
                print(f"[Evaluator] jiwer calculation failed: {e}")

        # Fallback: manual calculation using edit distance
        return self._calculate_wer_manual(reference, hypothesis)

    def _calculate_wer_manual(self, reference: str, hypothesis: str) -> WERResult:
        """Calculate WER using manual edit distance (fallback)"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Dynamic programming for edit distance
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # Deletion
                        dp[i][j - 1] + 1,  # Insertion
                        dp[i - 1][j - 1] + 1,  # Substitution
                    )

        # Backtrack to count operations
        i, j = m, n
        substitutions = deletions = insertions = hits = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                hits += 1
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1

        total_errors = substitutions + deletions + insertions
        wer_score = total_errors / len(ref_words) if ref_words else 0.0

        return WERResult(
            wer=wer_score,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            hits=hits,
            reference_length=len(ref_words),
            hypothesis_length=len(hyp_words),
        )

    def calculate_wer_batch(
        self, references: List[str], hypotheses: List[str], preprocess: bool = True
    ) -> Tuple[float, List[WERResult]]:
        """
        Calculate WER for multiple pairs and return aggregate.

        Args:
            references: List of reference texts
            hypotheses: List of hypothesis texts
            preprocess: Apply preprocessing

        Returns:
            Tuple of (weighted average WER, list of individual results)
        """
        if len(references) != len(hypotheses):
            raise ValueError("Reference and hypothesis lists must have same length")

        results = []
        for ref, hyp in zip(references, hypotheses):
            result = self.calculate_wer(ref, hyp, preprocess)
            results.append(result)

        # Calculate weighted average WER
        total_ref_words = sum(r.reference_length for r in results)
        total_errors = sum(r.substitutions + r.deletions + r.insertions for r in results)

        avg_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

        return avg_wer, results

    # =========================================================================
    # DER Calculation
    # =========================================================================

    def calculate_der(
        self,
        reference_segments: List[Tuple[str, float, float]],
        hypothesis_segments: List[Tuple[str, float, float]],
        collar: float = 0.25,
    ) -> DERResult:
        """
        Calculate Diarization Error Rate.

        DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Duration

        Args:
            reference_segments: Ground truth [(speaker_id, start, end), ...]
            hypothesis_segments: System output [(speaker_id, start, end), ...]
            collar: Forgiveness collar in seconds (standard: 0.25s)

        Returns:
            DERResult with detailed breakdown
        """
        if not reference_segments:
            return DERResult(
                der=0.0,
                total_duration=0.0,
                num_speakers_ref=0,
                num_speakers_hyp=(
                    len(set(s[0] for s in hypothesis_segments)) if hypothesis_segments else 0
                ),
            )

        # Get unique speakers
        ref_speakers = set(s[0] for s in reference_segments)
        hyp_speakers = set(s[0] for s in hypothesis_segments) if hypothesis_segments else set()

        # Calculate total reference duration
        total_ref_duration = sum(end - start for _, start, end in reference_segments)

        if total_ref_duration == 0:
            return DERResult(
                der=0.0,
                total_duration=0.0,
                num_speakers_ref=len(ref_speakers),
                num_speakers_hyp=len(hyp_speakers),
            )

        # Frame-based evaluation
        resolution = 0.01  # 10ms resolution

        # Get time range
        all_starts = [s[1] for s in reference_segments + (hypothesis_segments or [])]
        all_ends = [s[2] for s in reference_segments + (hypothesis_segments or [])]

        min_time = min(all_starts) if all_starts else 0
        max_time = max(all_ends) if all_ends else 0

        # Initialize counters
        missed_speech = 0.0
        false_alarm = 0.0
        speaker_confusion = 0.0

        # Frame-by-frame evaluation
        t = min_time
        while t < max_time:
            t_mid = t + resolution / 2

            # Get reference speakers at time t
            ref_spk_at_t = set()
            for speaker, start, end in reference_segments:
                # Apply collar
                if (start + collar) <= t_mid < (end - collar):
                    ref_spk_at_t.add(speaker)

            # Get hypothesis speakers at time t
            hyp_spk_at_t = set()
            if hypothesis_segments:
                for speaker, start, end in hypothesis_segments:
                    if start <= t_mid < end:
                        hyp_spk_at_t.add(speaker)

            # Count errors
            if ref_spk_at_t and not hyp_spk_at_t:
                # Missed speech: reference has speech, hypothesis doesn't
                missed_speech += resolution
            elif hyp_spk_at_t and not ref_spk_at_t:
                # False alarm: hypothesis has speech, reference doesn't
                false_alarm += resolution
            elif ref_spk_at_t and hyp_spk_at_t:
                # Both have speech - check for speaker confusion
                # Simplified: if number of speakers differs, count as confusion
                ref_count = len(ref_spk_at_t)
                hyp_count = len(hyp_spk_at_t)

                if ref_count != hyp_count:
                    # Partial confusion
                    confusion_ratio = abs(ref_count - hyp_count) / max(ref_count, hyp_count)
                    speaker_confusion += resolution * confusion_ratio

            t += resolution

        # Calculate DER
        total_error = missed_speech + false_alarm + speaker_confusion
        der = total_error / total_ref_duration

        return DERResult(
            der=min(der, 1.0),  # Cap at 100%
            missed_speech=missed_speech / total_ref_duration,
            false_alarm=false_alarm / total_ref_duration,
            speaker_confusion=speaker_confusion / total_ref_duration,
            total_duration=total_ref_duration,
            num_speakers_ref=len(ref_speakers),
            num_speakers_hyp=len(hyp_speakers),
        )

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_evaluation_report(
        self,
        wer_results: List[WERResult],
        der_results: Optional[List[DERResult]] = None,
        sample_names: Optional[List[str]] = None,
        condition_name: str = "Unknown",
    ) -> str:
        """
        Generate formatted evaluation report for thesis.

        Args:
            wer_results: List of WER results
            der_results: List of DER results (optional)
            sample_names: Names for each sample
            condition_name: Name of test condition

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("LAPORAN EVALUASI SISTEM NOTULENSI RAPAT OTOMATIS")
        lines.append(f"Kondisi: {condition_name}")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # WER Summary
        lines.append("1. EVALUASI ASR (Word Error Rate)")
        lines.append("-" * 50)

        if wer_results:
            wer_values = [r.wer for r in wer_results]
            avg_wer = np.mean(wer_values)
            std_wer = np.std(wer_values)
            min_wer = np.min(wer_values)
            max_wer = np.max(wer_values)

            total_subs = sum(r.substitutions for r in wer_results)
            total_dels = sum(r.deletions for r in wer_results)
            total_ins = sum(r.insertions for r in wer_results)
            total_hits = sum(r.hits for r in wer_results)

            lines.append(f"   Jumlah sampel      : {len(wer_results)}")
            lines.append(f"   WER rata-rata      : {avg_wer:.4f} ({avg_wer*100:.2f}%)")
            lines.append(f"   Standar deviasi    : {std_wer:.4f}")
            lines.append(f"   WER minimum        : {min_wer:.4f} ({min_wer*100:.2f}%)")
            lines.append(f"   WER maksimum       : {max_wer:.4f} ({max_wer*100:.2f}%)")
            lines.append("")
            lines.append("   Detail Error Total:")
            lines.append(f"   - Substitutions    : {total_subs}")
            lines.append(f"   - Deletions        : {total_dels}")
            lines.append(f"   - Insertions       : {total_ins}")
            lines.append(f"   - Correct (Hits)   : {total_hits}")

            # Per-sample details
            if sample_names and len(sample_names) == len(wer_results):
                lines.append("")
                lines.append("   Detail per Sampel:")
                for name, result in zip(sample_names, wer_results):
                    lines.append(f"   - {name}: WER = {result.wer:.4f} ({result.wer*100:.2f}%)")
        else:
            lines.append("   Tidak ada data WER untuk dievaluasi.")

        lines.append("")

        # DER Summary
        lines.append("2. EVALUASI DIARIZATION (Diarization Error Rate)")
        lines.append("-" * 50)

        if der_results:
            der_values = [r.der for r in der_results]
            avg_der = np.mean(der_values)
            std_der = np.std(der_values)

            avg_missed = np.mean([r.missed_speech for r in der_results])
            avg_fa = np.mean([r.false_alarm for r in der_results])
            avg_conf = np.mean([r.speaker_confusion for r in der_results])

            lines.append(f"   Jumlah sampel      : {len(der_results)}")
            lines.append(f"   DER rata-rata      : {avg_der:.4f} ({avg_der*100:.2f}%)")
            lines.append(f"   Standar deviasi    : {std_der:.4f}")
            lines.append("")
            lines.append("   Komponen Error (rata-rata):")
            lines.append(f"   - Missed Speech    : {avg_missed:.4f} ({avg_missed*100:.2f}%)")
            lines.append(f"   - False Alarm      : {avg_fa:.4f} ({avg_fa*100:.2f}%)")
            lines.append(f"   - Speaker Confusion: {avg_conf:.4f} ({avg_conf*100:.2f}%)")

            # Per-sample details
            if sample_names and len(sample_names) == len(der_results):
                lines.append("")
                lines.append("   Detail per Sampel:")
                for name, result in zip(sample_names, der_results):
                    lines.append(f"   - {name}: DER = {result.der:.4f} ({result.der*100:.2f}%)")
        else:
            lines.append("   Tidak ada data DER untuk dievaluasi.")

        lines.append("")
        lines.append("=" * 70)
        lines.append("Catatan:")
        lines.append(
            "- Evaluasi WER menggunakan preprocessing standar (lowercase, hapus tanda baca)"
        )
        lines.append("- Evaluasi DER menggunakan collar forgiveness 0.25 detik")
        lines.append("=" * 70)

        return "\n".join(lines)

    def export_results_to_csv(
        self, results: List[EvaluationResult], output_filename: str = "evaluation_results.csv"
    ) -> str:
        """
        Export evaluation results to CSV for thesis appendix.

        Args:
            results: List of EvaluationResult objects
            output_filename: Output CSV filename

        Returns:
            Path to saved CSV file
        """
        output_path = self.output_dir / output_filename

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Sample",
                    "Condition",
                    "WER",
                    "MER",
                    "WIL",
                    "CER",
                    "Substitutions",
                    "Deletions",
                    "Insertions",
                    "Hits",
                    "Ref_Words",
                    "Hyp_Words",
                    "DER",
                    "Missed_Speech",
                    "False_Alarm",
                    "Speaker_Confusion",
                    "Duration_Sec",
                    "Num_Speakers_Ref",
                    "Num_Speakers_Hyp",
                ]
            )

            # Data rows
            for result in results:
                wer = result.wer_result
                der = result.der_result

                row = [
                    result.sample_name,
                    result.condition,
                    # WER metrics
                    f"{wer.wer:.4f}" if wer else "",
                    f"{wer.mer:.4f}" if wer else "",
                    f"{wer.wil:.4f}" if wer else "",
                    f"{wer.cer:.4f}" if wer else "",
                    wer.substitutions if wer else "",
                    wer.deletions if wer else "",
                    wer.insertions if wer else "",
                    wer.hits if wer else "",
                    wer.reference_length if wer else "",
                    wer.hypothesis_length if wer else "",
                    # DER metrics
                    f"{der.der:.4f}" if der else "",
                    f"{der.missed_speech:.4f}" if der else "",
                    f"{der.false_alarm:.4f}" if der else "",
                    f"{der.speaker_confusion:.4f}" if der else "",
                    f"{der.total_duration:.2f}" if der else "",
                    der.num_speakers_ref if der else "",
                    der.num_speakers_hyp if der else "",
                ]

                writer.writerow(row)

        return str(output_path)

    def generate_summary_table(
        self, results_by_condition: Dict[str, List[EvaluationResult]]
    ) -> str:
        """
        Generate summary table comparing results across conditions.

        Args:
            results_by_condition: Dict mapping condition name to list of results

        Returns:
            Formatted table string
        """
        lines = []
        lines.append("")
        lines.append("TABEL RINGKASAN EVALUASI PER KONDISI")
        lines.append("=" * 80)
        lines.append("")

        # Header
        header = (
            f"{'Kondisi':<20} {'N':>5} {'WER Mean':>10} {'WER Std':>10} "
            f"{'DER Mean':>10} {'DER Std':>10}"
        )
        lines.append(header)
        lines.append("-" * 80)

        # Data rows
        for condition, results in results_by_condition.items():
            n = len(results)

            # WER stats
            wer_values = [r.wer_result.wer for r in results if r.wer_result]
            wer_mean = np.mean(wer_values) if wer_values else 0
            wer_std = np.std(wer_values) if wer_values else 0

            # DER stats
            der_values = [r.der_result.der for r in results if r.der_result]
            der_mean = np.mean(der_values) if der_values else 0
            der_std = np.std(der_values) if der_values else 0

            row = (
                f"{condition:<20} {n:>5} {wer_mean:>10.4f} {wer_std:>10.4f} "
                f"{der_mean:>10.4f} {der_std:>10.4f}"
            )
            lines.append(row)

        lines.append("-" * 80)
        lines.append("")

        return "\n".join(lines)

    def save_report(self, report: str, filename: str = "evaluation_report.txt") -> str:
        """Save evaluation report to file"""
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return str(output_path)
