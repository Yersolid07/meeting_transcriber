#!/usr/bin/env python3
"""
Script untuk menjalankan evaluasi lengkap pada semua kondisi.
Menghasilkan tabel dan grafik untuk skripsi.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project modules are imported lazily inside functions to avoid executing code at import time

# =============================================================================
# Configuration
# =============================================================================

CONDITIONS = {
    "bersih": {
        "audio_dir": "./data/audio/kondisi_bersih",
        "gt_dir": "./data/ground_truth/kondisi_bersih",
        "description": "Audio bersih, ruangan tenang, 2-3 speaker",
        "expected_wer": 0.15,
        "expected_der": 0.15,
    },
    "noisy": {
        "audio_dir": "./data/audio/kondisi_noisy",
        "gt_dir": "./data/ground_truth/kondisi_noisy",
        "description": "Audio dengan background noise (AC, ambient)",
        "expected_wer": 0.25,
        "expected_der": 0.25,
    },
    "overlap": {
        "audio_dir": "./data/audio/kondisi_overlap",
        "gt_dir": "./data/ground_truth/kondisi_overlap",
        "description": "Audio dengan overlapping speech, diskusi aktif",
        "expected_wer": 0.35,
        "expected_der": 0.40,
    },
    "multispeaker": {
        "audio_dir": "./data/audio/kondisi_multispeaker",
        "gt_dir": "./data/ground_truth/kondisi_multispeaker",
        "description": "Audio dengan 4-6 speaker berbeda",
        "expected_wer": 0.25,
        "expected_der": 0.35,
    },
}


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_single_audio(
    pipeline, audio_path: Path, gt_transcript_path: Path, gt_rttm_path: Path, condition: str
):
    """Evaluate a single audio file"""

    sample_name = audio_path.stem

    # Clear previous state
    pipeline.clear_state()

    # Load and process audio
    pipeline.load_audio(str(audio_path))
    pipeline.run_diarization()
    pipeline.run_transcription()

    # Load ground truth
    reference_transcript = None
    reference_diarization = None

    # Lazy import parsing utilities
    from src.utils import parse_rttm_file, parse_transcript_file

    if gt_transcript_path.exists():
        reference_transcript = parse_transcript_file(gt_transcript_path)

    if gt_rttm_path.exists():
        reference_diarization = parse_rttm_file(gt_rttm_path)

    # Evaluate
    result = pipeline.evaluate(
        reference_transcript=reference_transcript,
        reference_diarization=reference_diarization,
        sample_name=sample_name,
        condition=condition,
    )

    return result


def evaluate_condition(pipeline, condition_name: str, config: dict, verbose: bool = True):
    """Evaluate all samples in a condition"""

    audio_dir = Path(config["audio_dir"])
    gt_dir = Path(config["gt_dir"])

    results = []

    if not audio_dir.exists():
        if verbose:
            print(f"  ⚠️ Audio directory not found: {audio_dir}")
        return results

    # Lazy import utility to list audio files
    from src.utils import list_audio_files

    audio_files = list_audio_files(audio_dir)

    if not audio_files:
        if verbose:
            print("  ⚠️ No audio files found")
        return results

    if verbose:
        print(f"\n  Found {len(audio_files)} audio files")

    for i, audio_path in enumerate(audio_files, 1):
        sample_name = audio_path.stem

        if verbose:
            print(f"    [{i}/{len(audio_files)}] {sample_name}...", end=" ")

        try:
            gt_transcript_path = gt_dir / f"{sample_name}.txt"
            gt_rttm_path = gt_dir / f"{sample_name}.rttm"

            result = evaluate_single_audio(
                pipeline, audio_path, gt_transcript_path, gt_rttm_path, condition_name
            )

            results.append(result)

            if verbose:
                wer_str = f"WER={result.wer_result.wer:.3f}" if result.wer_result else "WER=N/A"
                der_str = f"DER={result.der_result.der:.3f}" if result.der_result else "DER=N/A"
                print(f"✓ {wer_str}, {der_str}")

        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            continue

    return results


def run_full_evaluation(output_dir: str = "./data/output/evaluation", verbose: bool = True):
    """Run evaluation on all conditions"""

    output_dir = Path(output_dir)
    # lazy import ensure_dir and other utilities
    from src.utils import ensure_dir

    ensure_dir(output_dir)

    # Initialize pipeline
    if verbose:
        print("=" * 70)
        print("EVALUASI SISTEM NOTULENSI RAPAT OTOMATIS")
        print("=" * 70)
        print("\nInitializing pipeline...")

    # Lazy import pipeline classes
    from src.pipeline import MeetingTranscriberPipeline, PipelineConfig

    pipeline = MeetingTranscriberPipeline(PipelineConfig(verbose=False, save_intermediate=False))

    # Run evaluation for each condition
    all_results = {}

    for condition_name, config in CONDITIONS.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Kondisi: {condition_name.upper()}")
            print(f"Deskripsi: {config['description']}")
            print(f"{'='*50}")

        from src.utils import Timer

        with Timer(f"Evaluation - {condition_name}"):
            results = evaluate_condition(pipeline, condition_name, config, verbose)

        all_results[condition_name] = results

    return all_results


# =============================================================================
# Results Processing
# =============================================================================


def create_results_dataframe(all_results: Dict[str, List[object]]) -> pd.DataFrame:
    """Create DataFrame from evaluation results"""

    rows = []

    for condition, results in all_results.items():
        for result in results:
            row = {
                "condition": condition,
                "sample": result.sample_name,
                # WER metrics
                "wer": result.wer_result.wer if result.wer_result else None,
                "mer": result.wer_result.mer if result.wer_result else None,
                "wil": result.wer_result.wil if result.wer_result else None,
                "substitutions": result.wer_result.substitutions if result.wer_result else None,
                "deletions": result.wer_result.deletions if result.wer_result else None,
                "insertions": result.wer_result.insertions if result.wer_result else None,
                "hits": result.wer_result.hits if result.wer_result else None,
                "ref_words": result.wer_result.reference_length if result.wer_result else None,
                "hyp_words": result.wer_result.hypothesis_length if result.wer_result else None,
                # DER metrics
                "der": result.der_result.der if result.der_result else None,
                "missed_speech": result.der_result.missed_speech if result.der_result else None,
                "false_alarm": result.der_result.false_alarm if result.der_result else None,
                "speaker_confusion": (
                    result.der_result.speaker_confusion if result.der_result else None
                ),
                "duration": result.der_result.total_duration if result.der_result else None,
                "num_speakers_ref": (
                    result.der_result.num_speakers_ref if result.der_result else None
                ),
                "num_speakers_hyp": (
                    result.der_result.num_speakers_hyp if result.der_result else None
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate summary statistics for thesis tables"""

    summaries = {}

    # WER summary
    wer_summary = (
        df.groupby("condition")["wer"]
        .agg([("N", "count"), ("Mean", "mean"), ("Std", "std"), ("Min", "min"), ("Max", "max")])
        .round(4)
    )
    summaries["wer"] = wer_summary

    # DER summary
    der_summary = (
        df.groupby("condition")["der"]
        .agg([("N", "count"), ("Mean", "mean"), ("Std", "std"), ("Min", "min"), ("Max", "max")])
        .round(4)
    )
    summaries["der"] = der_summary

    # DER components
    der_components = (
        df.groupby("condition")[["missed_speech", "false_alarm", "speaker_confusion"]]
        .mean()
        .round(4)
    )
    summaries["der_components"] = der_components

    # Error breakdown
    error_breakdown = df.groupby("condition")[
        ["substitutions", "deletions", "insertions", "hits"]
    ].sum()
    summaries["error_breakdown"] = error_breakdown

    return summaries


# =============================================================================
# Visualization Functions
# =============================================================================


def create_boxplot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create boxplot comparison of WER and DER"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = sns.color_palette("Set2", 4)

    # WER Boxplot
    ax1 = axes[0]
    df_wer = df.dropna(subset=["wer"])

    if not df_wer.empty:
        order = ["bersih", "noisy", "overlap", "multispeaker"]
        existing_order = [c for c in order if c in df_wer["condition"].unique()]

        sns.boxplot(
            data=df_wer, x="condition", y="wer", order=existing_order, ax=ax1, palette=colors
        )

        # Add expected values
        for i, cond in enumerate(existing_order):
            if cond in CONDITIONS:
                expected = CONDITIONS[cond]["expected_wer"]
                ax1.hlines(
                    expected,
                    i - 0.4,
                    i + 0.4,
                    colors="red",
                    linestyles="--",
                    alpha=0.7,
                    linewidth=2,
                )

        ax1.set_title("Word Error Rate (WER) per Kondisi", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Kondisi", fontsize=12)
        ax1.set_ylabel("WER", fontsize=12)
        ax1.set_ylim([0, min(1.0, df_wer["wer"].max() * 1.3)])

    # DER Boxplot
    ax2 = axes[1]
    df_der = df.dropna(subset=["der"])

    if not df_der.empty:
        order = ["bersih", "noisy", "overlap", "multispeaker"]
        existing_order = [c for c in order if c in df_der["condition"].unique()]

        sns.boxplot(
            data=df_der, x="condition", y="der", order=existing_order, ax=ax2, palette=colors
        )

        # Add expected values
        for i, cond in enumerate(existing_order):
            if cond in CONDITIONS:
                expected = CONDITIONS[cond]["expected_der"]
                ax2.hlines(
                    expected,
                    i - 0.4,
                    i + 0.4,
                    colors="red",
                    linestyles="--",
                    alpha=0.7,
                    linewidth=2,
                )

        ax2.set_title("Diarization Error Rate (DER) per Kondisi", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Kondisi", fontsize=12)
        ax2.set_ylabel("DER", fontsize=12)
        ax2.set_ylim([0, min(1.0, df_der["der"].max() * 1.3)])

    plt.tight_layout()

    # Save
    output_path = output_dir / "fig_boxplot_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig_boxplot_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def create_bar_chart(df: pd.DataFrame, output_dir: Path):
    """Create bar chart with error bars"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = ["bersih", "noisy", "overlap", "multispeaker"]
    colors = sns.color_palette("Set2", 4)

    # WER Bar Chart
    ax1 = axes[0]
    wer_data = df.groupby("condition")["wer"].agg(["mean", "std"]).reindex(order).dropna()

    if not wer_data.empty:
        x = np.arange(len(wer_data))
        bars = ax1.bar(
            x,
            wer_data["mean"],
            yerr=wer_data["std"],
            capsize=5,
            color=colors[: len(wer_data)],
            edgecolor="black",
            linewidth=1.2,
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(wer_data.index, fontsize=11)
        ax1.set_title("Mean WER per Kondisi", fontsize=14, fontweight="bold")
        ax1.set_ylabel("WER", fontsize=12)
        ax1.set_ylim([0, wer_data["mean"].max() * 1.4])

        # Add value labels
        for bar, (idx, row) in zip(bars, wer_data.iterrows()):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + row["std"] + 0.01,
                f'{row["mean"]:.3f}',
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # DER Bar Chart
    ax2 = axes[1]
    der_data = df.groupby("condition")["der"].agg(["mean", "std"]).reindex(order).dropna()

    if not der_data.empty:
        x = np.arange(len(der_data))
        bars = ax2.bar(
            x,
            der_data["mean"],
            yerr=der_data["std"],
            capsize=5,
            color=colors[: len(der_data)],
            edgecolor="black",
            linewidth=1.2,
        )

        ax2.set_xticks(x)
        ax2.set_xticklabels(der_data.index, fontsize=11)
        ax2.set_title("Mean DER per Kondisi", fontsize=14, fontweight="bold")
        ax2.set_ylabel("DER", fontsize=12)
        ax2.set_ylim([0, der_data["mean"].max() * 1.4])

        # Add value labels
        for bar, (idx, row) in zip(bars, der_data.iterrows()):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + row["std"] + 0.01,
                f'{row["mean"]:.3f}',
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save
    output_path = output_dir / "fig_barchart_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig_barchart_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def create_der_components_chart(df: pd.DataFrame, output_dir: Path):
    """Create stacked bar chart for DER components"""

    fig, ax = plt.subplots(figsize=(12, 6))

    order = ["bersih", "noisy", "overlap", "multispeaker"]

    der_components = (
        df.groupby("condition")[["missed_speech", "false_alarm", "speaker_confusion"]]
        .mean()
        .reindex(order)
        .dropna()
    )

    if not der_components.empty:
        x = np.arange(len(der_components))
        width = 0.6

        # Stacked bars
        bottom = np.zeros(len(der_components))
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        labels = ["Missed Speech", "False Alarm", "Speaker Confusion"]

        for col, color, label in zip(der_components.columns, colors, labels):
            values = der_components[col].values
            ax.bar(x, values, width, bottom=bottom, label=label, color=color, edgecolor="black")
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(der_components.index, fontsize=12)
        ax.set_title("Komponen DER per Kondisi", fontsize=14, fontweight="bold")
        ax.set_ylabel("Error Rate", fontsize=12)
        ax.legend(loc="upper left", fontsize=11)
        ax.set_ylim([0, bottom.max() * 1.2])

        # Add total DER labels
        for i, total in enumerate(bottom):
            ax.text(
                i,
                total + 0.01,
                f"{total:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save
    output_path = output_dir / "fig_der_components.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig_der_components.pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def create_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of results"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = ["bersih", "noisy", "overlap", "multispeaker"]

    # WER heatmap
    ax1 = axes[0]
    wer_pivot = (
        df.pivot_table(values="wer", index="condition", columns="sample", aggfunc="first")
        .reindex(order)
        .dropna(how="all")
    )

    if not wer_pivot.empty:
        sns.heatmap(
            wer_pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax1, cbar_kws={"label": "WER"}
        )
        ax1.set_title("WER per Sample dan Kondisi", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Sample", fontsize=12)
        ax1.set_ylabel("Kondisi", fontsize=12)

    # DER heatmap
    ax2 = axes[1]
    der_pivot = (
        df.pivot_table(values="der", index="condition", columns="sample", aggfunc="first")
        .reindex(order)
        .dropna(how="all")
    )

    if not der_pivot.empty:
        sns.heatmap(
            der_pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax2, cbar_kws={"label": "DER"}
        )
        ax2.set_title("DER per Sample dan Kondisi", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Sample", fontsize=12)
        ax2.set_ylabel("Kondisi", fontsize=12)

    plt.tight_layout()

    # Save
    output_path = output_dir / "fig_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig_heatmap.pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================


def generate_latex_tables(summaries: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate LaTeX tables for thesis"""

    latex_content = []

    # WER Table
    latex_content.append(
        """
% Tabel 4.1: Hasil Evaluasi WER
\\begin{table}[htbp]
\\centering
\\caption{Hasil Evaluasi Word Error Rate (WER) pada Berbagai Kondisi}
\\label{tab:wer_results}
\\begin{tabular}{lccccc}
\\hline
\\textbf{Kondisi} & \\textbf{N} & \\textbf{Mean} & \\textbf{Std} & \\
\\textbf{Min} & \\textbf{Max} \\\\\
\\hline
"""
    )

    if "wer" in summaries and not summaries["wer"].empty:
        for condition, row in summaries["wer"].iterrows():
            pattern = "{} & {} & {:.4f} & {:.4f} & {:.4f} " "& {:.4f} \\\\\n"
            row_str = pattern.format(
                condition.capitalize(),
                int(row["N"]),
                row["Mean"],
                row["Std"],
                row["Min"],
                row["Max"],
            )
            latex_content.append(row_str)

    latex_content.append(
        """\\hline
\\end{tabular}
\\end{table}
"""
    )

    # DER Table
    latex_content.append(
        """
% Tabel 4.2: Hasil Evaluasi DER
\\begin{table}[htbp]
\\centering
\\caption{Hasil Evaluasi Diarization Error Rate (DER) pada Berbagai Kondisi}
\\label{tab:der_results}
\\begin{tabular}{lccccc}
\\hline
\\textbf{Kondisi} & \\textbf{N} & \\textbf{Mean} & \\textbf{Std} & \\
\\textbf{Min} & \\textbf{Max} \\\\\
\\hline
"""
    )

    if "der" in summaries and not summaries["der"].empty:
        for condition, row in summaries["der"].iterrows():
            pattern = "{} & {} & {:.4f} & {:.4f} & {:.4f} " "& {:.4f} \\\\\n"
            row_str = pattern.format(
                condition.capitalize(),
                int(row["N"]),
                row["Mean"],
                row["Std"],
                row["Min"],
                row["Max"],
            )
            latex_content.append(row_str)

    latex_content.append(
        """\\hline
\\end{tabular}
\\end{table}
"""
    )

    # Save
    output_path = output_dir / "tables_latex.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(latex_content)

    print(f"  ✓ Saved: {output_path}")


def generate_markdown_tables(summaries: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate Markdown tables for thesis"""

    md_content = []

    md_content.append("# Tabel Hasil Evaluasi\n\n")

    # WER Table
    md_content.append("## Tabel 4.1: Hasil Evaluasi WER\n\n")
    md_content.append("| Kondisi | N | Mean | Std | Min | Max |\n")
    md_content.append("|---------|---|------|-----|-----|-----|\n")

    if "wer" in summaries and not summaries["wer"].empty:
        for condition, row in summaries["wer"].iterrows():
            md_content.append(
                f"| {condition.capitalize()} | {int(row['N'])} | {row['Mean']:.4f} | "
                f"{row['Std']:.4f} | {row['Min']:.4f} | {row['Max']:.4f} |\n"
            )

    md_content.append("\n")

    # DER Table
    md_content.append("## Tabel 4.2: Hasil Evaluasi DER\n\n")
    md_content.append("| Kondisi | N | Mean | Std | Min | Max |\n")
    md_content.append("|---------|---|------|-----|-----|-----|\n")

    if "der" in summaries and not summaries["der"].empty:
        for condition, row in summaries["der"].iterrows():
            md_content.append(
                f"| {condition.capitalize()} | {int(row['N'])} | {row['Mean']:.4f} | "
                f"{row['Std']:.4f} | {row['Min']:.4f} | {row['Max']:.4f} |\n"
            )

    md_content.append("\n")

    # DER Components Table
    md_content.append("## Tabel 4.3: Komponen DER per Kondisi\n\n")
    md_content.append("| Kondisi | Missed Speech | False Alarm | Speaker Confusion |\n")
    md_content.append("|---------|---------------|-------------|-------------------|\n")

    if "der_components" in summaries and not summaries["der_components"].empty:
        for condition, row in summaries["der_components"].iterrows():
            md_content.append(
                f"| {condition.capitalize()} | {row['missed_speech']:.4f} | "
                f"{row['false_alarm']:.4f} | {row['speaker_confusion']:.4f} |\n"
            )

    # Save
    output_path = output_dir / "tables_markdown.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(md_content)

    print(f"  ✓ Saved: {output_path}")


def generate_full_report(df: pd.DataFrame, summaries: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate comprehensive evaluation report"""

    lines = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 80)
    lines.append("LAPORAN EVALUASI SISTEM NOTULENSI RAPAT OTOMATIS")
    lines.append("=" * 80)
    lines.append(f"\nTimestamp: {timestamp}")
    lines.append(f"Total samples evaluated: {len(df)}")
    lines.append(f"Conditions tested: {', '.join(df['condition'].unique())}")
    lines.append("\n")

    # Overall summary
    lines.append("-" * 80)
    lines.append("RINGKASAN KESELURUHAN")
    lines.append("-" * 80)

    overall_wer = df["wer"].dropna()
    overall_der = df["der"].dropna()

    if len(overall_wer) > 0:
        lines.append("\nWER Keseluruhan:")
        lines.append(f"  Mean: {overall_wer.mean():.4f} ({overall_wer.mean()*100:.2f}%)")
        lines.append(f"  Std:  {overall_wer.std():.4f}")
        lines.append(f"  Min:  {overall_wer.min():.4f}")
        lines.append(f"  Max:  {overall_wer.max():.4f}")

    if len(overall_der) > 0:
        lines.append("\nDER Keseluruhan:")
        lines.append(f"  Mean: {overall_der.mean():.4f} ({overall_der.mean()*100:.2f}%)")
        lines.append(f"  Std:  {overall_der.std():.4f}")
        lines.append(f"  Min:  {overall_der.min():.4f}")
        lines.append(f"  Max:  {overall_der.max():.4f}")

    lines.append("\n")

    # Per-condition details
    lines.append("-" * 80)
    lines.append("DETAIL PER KONDISI")
    lines.append("-" * 80)

    for condition in CONDITIONS.keys():
        cond_data = df[df["condition"] == condition]

        if len(cond_data) == 0:
            continue

        lines.append(f"\n[{condition.upper()}]")
        lines.append(f"  Deskripsi: {CONDITIONS[condition]['description']}")
        lines.append(f"  Expected WER: {CONDITIONS[condition]['expected_wer']}")
        lines.append(f"  Expected DER: {CONDITIONS[condition]['expected_der']}")
        lines.append(f"  Samples: {len(cond_data)}")

        cond_wer = cond_data["wer"].dropna()
        cond_der = cond_data["der"].dropna()

        if len(cond_wer) > 0:
            lines.append("\n  WER Results:")
            lines.append(f"    Mean: {cond_wer.mean():.4f}")
            lines.append(f"    Std:  {cond_wer.std():.4f}")

            # Check if meets expectation
            if cond_wer.mean() <= CONDITIONS[condition]["expected_wer"]:
                lines.append("    Status: ✓ MEETS EXPECTATION")
            else:
                lines.append("    Status: ✗ ABOVE EXPECTATION")

        if len(cond_der) > 0:
            lines.append("\n  DER Results:")
            lines.append(f"    Mean: {cond_der.mean():.4f}")
            lines.append(f"    Std:  {cond_der.std():.4f}")

            if cond_der.mean() <= CONDITIONS[condition]["expected_der"]:
                lines.append("    Status: ✓ MEETS EXPECTATION")
            else:
                lines.append("    Status: ✗ ABOVE EXPECTATION")

    lines.append("\n")

    # Conclusions
    lines.append("-" * 80)
    lines.append("KESIMPULAN")
    lines.append("-" * 80)

    # Find best and worst conditions
    wer_by_cond = df.groupby("condition")["wer"].mean()
    der_by_cond = df.groupby("condition")["der"].mean()

    # Conclusions (lanjutan)
    if len(wer_by_cond) > 0:
        best_wer_cond = wer_by_cond.idxmin()
        worst_wer_cond = wer_by_cond.idxmax()
        lines.append("\n1. Performa ASR (WER):")
        lines.append(
            f"   - Kondisi terbaik: {best_wer_cond} (WER = {wer_by_cond[best_wer_cond]:.4f})"
        )
        lines.append(
            f"   - Kondisi terburuk: {worst_wer_cond} (WER = {wer_by_cond[worst_wer_cond]:.4f})"
        )

    if len(der_by_cond) > 0:
        best_der_cond = der_by_cond.idxmin()
        worst_der_cond = der_by_cond.idxmax()
        lines.append("\n2. Performa Diarization (DER):")
        lines.append(
            f"   - Kondisi terbaik: {best_der_cond} (DER = {der_by_cond[best_der_cond]:.4f})"
        )
        lines.append(
            f"   - Kondisi terburuk: {worst_der_cond} (DER = {der_by_cond[worst_der_cond]:.4f})"
        )

    lines.append("\n3. Rekomendasi:")
    lines.append("   - Sistem direkomendasikan untuk audio dengan kondisi bersih")
    lines.append("   - Hindari penggunaan pada kondisi dengan banyak overlapping speech")
    lines.append("   - Gunakan microphone berkualitas untuk hasil optimal")

    lines.append("\n")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Save
    output_path = output_dir / "evaluation_report_full.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# Main Function
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run full evaluation on all conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/output/evaluation",
        help="Output directory for results",
    )

    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip processing, only generate reports from existing CSV",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to existing CSV file (with --skip-processing)",
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output_dir = Path(args.output)
    # import utility functions lazily
    from src.utils import ensure_dir, save_json

    ensure_dir(output_dir)
    verbose = not args.quiet

    # Run evaluation or load existing
    if args.skip_processing and args.csv_path:
        if verbose:
            print(f"Loading existing results from: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
    else:
        # Run full evaluation
        all_results = run_full_evaluation(str(output_dir), verbose)

        # Create DataFrame
        df = create_results_dataframe(all_results)

        # Save CSV
        csv_path = output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n✓ Results saved to: {csv_path}")

    if df.empty:
        print("\n⚠️ No evaluation results to process")
        return

    # Generate summaries
    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING REPORTS AND VISUALIZATIONS")
        print("=" * 70)

    summaries = generate_summary_statistics(df)

    # Print summary tables
    if verbose:
        print("\n--- WER Summary ---")
        print(summaries["wer"].to_string())
        print("\n--- DER Summary ---")
        print(summaries["der"].to_string())

    # Generate visualizations
    if verbose:
        print("\nGenerating visualizations...")

    create_boxplot_comparison(df, output_dir)
    create_bar_chart(df, output_dir)
    create_der_components_chart(df, output_dir)
    create_heatmap(df, output_dir)

    # Generate reports
    if verbose:
        print("\nGenerating reports...")

    generate_latex_tables(summaries, output_dir)
    generate_markdown_tables(summaries, output_dir)
    generate_full_report(df, summaries, output_dir)

    # Save summaries as JSON
    summaries_dict = {k: v.to_dict() for k, v in summaries.items()}
    save_json(summaries_dict, output_dir / "evaluation_summaries.json")

    if verbose:
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print("\nGenerated files:")
        for f in sorted(output_dir.glob("*")):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
