"""
Utility Functions Module
========================
Helper functions used across the system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, List, Optional, Union

# =============================================================================
# Logging Setup
# =============================================================================


def setup_logger(
    name: str = "MeetingTranscriber", level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup and return a logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Timing Utilities
# =============================================================================


def timer(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[Timer] {func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, name: str = "Block"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"[Timer] {self.name} took {self.elapsed:.2f} seconds")


# =============================================================================
# File Utilities
# =============================================================================


def get_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')

    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_audio_files(
    directory: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    List all audio files in directory.

    Args:
        directory: Directory to search
        extensions: List of extensions to include (default: common audio formats)

    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"]

    directory = Path(directory)
    audio_files = []

    for ext in extensions:
        audio_files.extend(directory.glob(f"*{ext}"))
        audio_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(audio_files)


def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filename"""
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


# =============================================================================
# JSON Utilities
# =============================================================================


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Text Utilities
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 0:
        return "0:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    """Format timestamp for document display"""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean text: normalize whitespace, remove special chars"""
    if not text:
        return ""

    # Normalize whitespace
    text = " ".join(text.split())

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    return text.strip()


# =============================================================================
# Progress Utilities
# =============================================================================


class ProgressTracker:
    """Simple progress tracker for long operations"""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        self._print_progress()

    def _print_progress(self):
        """Print progress bar"""
        percent = self.current / self.total * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time

        # Estimate remaining time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = format_duration(eta)
        else:
            eta_str = "?"

        bar_length = 30
        filled = int(bar_length * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"\r[{bar}] {percent:5.1f}% ({self.current}/{self.total}) ETA: {eta_str}  ",
            end="",
            flush=True,
        )

        if self.current >= self.total:
            print()  # New line at completion

    def finish(self):
        """Mark progress as complete"""
        self.current = self.total
        self._print_progress()

        elapsed = time.time() - self.start_time
        print(f"[{self.description}] Completed in {format_duration(elapsed)}")


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_audio_file(filepath: Union[str, Path]) -> bool:
    """
    Validate that file exists and is a supported audio format.

    Args:
        filepath: Path to audio file

    Returns:
        True if valid, raises exception otherwise
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}

    if filepath.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: {filepath.suffix}. "
            f"Supported: {', '.join(supported_formats)}"
        )

    return True


def validate_ground_truth_file(filepath: Union[str, Path]) -> bool:
    """
    Validate ground truth file format.

    Args:
        filepath: Path to ground truth file

    Returns:
        True if valid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")

    supported_formats = {".txt", ".json", ".rttm"}

    if filepath.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported ground truth format: {filepath.suffix}. "
            f"Supported: {', '.join(supported_formats)}"
        )

    return True


# =============================================================================
# Ground Truth Parsing
# =============================================================================


def parse_transcript_file(filepath: Union[str, Path]) -> str:
    """
    Parse transcript file (plain text).

    Args:
        filepath: Path to transcript file

    Returns:
        Transcript text
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_rttm_file(filepath: Union[str, Path]) -> List[tuple]:
    """
    Parse RTTM (Rich Transcription Time Marked) file for diarization ground truth.

    RTTM format:
    SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    Args:
        filepath: Path to RTTM file

    Returns:
        List of (speaker_id, start, end) tuples
    """
    segments = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]

                segments.append((speaker_id, start, start + duration))

    return segments


def create_ground_truth_template(
    output_path: Union[str, Path], audio_duration: float, num_speakers: int = 2
):
    """
    Create template ground truth files for annotation.

    Args:
        output_path: Output directory
        audio_duration: Duration of audio in seconds
        num_speakers: Expected number of speakers
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create transcript template
    transcript_template = """# Ground Truth Transcript
# Instruksi: Tulis transkripsi lengkap audio di bawah ini
# Hapus baris komentar (yang dimulai dengan #) sebelum evaluasi

[Tulis transkripsi di sini...]
"""

    with open(output_path / "transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript_template)

    # Create RTTM template
    rttm_template = f"""# Ground Truth Diarization (RTTM Format)
# Format: SPEAKER <file_id> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
#
# Contoh:
# SPEAKER audio 1 0.0 5.5 <NA> <NA> SPEAKER_00 <NA> <NA>
# SPEAKER audio 1 5.5 3.2 <NA> <NA> SPEAKER_01 <NA> <NA>
#
# Audio duration: {audio_duration:.2f} seconds
# Expected speakers: {num_speakers}
#
# Tambahkan baris SPEAKER di bawah:

"""

    with open(output_path / "diarization.rttm", "w", encoding="utf-8") as f:
        f.write(rttm_template)

    print(f"Ground truth templates created in: {output_path}")
