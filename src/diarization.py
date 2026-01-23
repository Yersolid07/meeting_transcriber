"""
Speaker Diarization Module
==========================
Implements VAD + Speaker Embedding + Clustering pipeline for speaker diarization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import setup_logger


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization"""

    # VAD settings
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.3
    min_silence_duration: float = 0.3

    # Segmentation settings
    segment_window: float = 1.5
    segment_hop: float = 0.75

    # Clustering settings
    clustering_method: str = "agglomerative"
    clustering_threshold: float = 0.7
    min_cluster_size: int = 2
    max_speakers: Optional[int] = None

    # Post-processing
    merge_gap_threshold: float = 0.5
    min_segment_duration: float = 0.3

    # Model settings
    embedding_model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    use_speechbrain: bool = True  # prefer SpeechBrain embeddings
    allow_fallback: bool = False  # if False, raise an error when SpeechBrain cannot be loaded

    # Collapse heuristics
    collapse_threshold: float = 0.15
    silhouette_collapse_threshold: float = 0.05

    # Iterative merging (centroid-based)
    iterative_merge_threshold: float = 0.15
    iterative_merge_silhouette_threshold: float = 0.0
    iterative_merge_max_iters: int = 10

    # Performance tuning
    embedding_batch_size: int = 32
    embedding_cache: bool = True  # write/load embedding arrays to cache_dir
    use_fast_embedding: bool = False  # use MFCC deterministic embeddings for speed

    # Optional: target speaker count - if set, clusters will be greedily merged to meet target
    target_num_speakers: Optional[int] = None
    target_force_threshold: float = (
        1.0  # 1.0 => allow merges regardless of distance; lower = more conservative
    )

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and metadata"""

    speaker_id: str
    start: float
    end: float
    confidence: float = 1.0
    is_overlap: bool = False
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get segment duration in seconds"""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "speaker_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "is_overlap": self.is_overlap,
            "duration": self.duration,
        }


class SpeakerDiarizer:
    """
    Speaker Diarization using SpeechBrain ECAPA-TDNN embeddings.

    Pipeline:
        1. Voice Activity Detection (VAD)
        2. Audio segmentation into windows
        3. Speaker embedding extraction (ECAPA-TDNN)
        4. Clustering to assign speaker labels
        5. Post-processing (merging, smoothing)

    Attributes:
        config: DiarizationConfig object

    Example:
        >>> diarizer = SpeakerDiarizer()
        >>> segments = diarizer.process(waveform, sample_rate=16000, num_speakers=4)
        >>> for seg in segments:
        ...     print(f"{seg.speaker_id}: {seg.start:.2f}s - {seg.end:.2f}s")
    """

    def __init__(self, config: Optional[DiarizationConfig] = None, models_dir: str = "./models"):
        """
        Initialize SpeakerDiarizer.

        Args:
            config: DiarizationConfig object
            models_dir: Directory to cache downloaded models
        """
        self.config = config or DiarizationConfig()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.device = self.config.device

        # Setup logger
        self.logger = setup_logger("SpeakerDiarizer")

        # Model placeholders (lazy loading)
        self._embedding_model = None
        self._vad_model = None
        self._embedding_model_is_speechbrain = False

    def _load_embedding_model(self):
        """Lazy load speaker embedding model

        This function will attempt to patch missing torchaudio APIs (e.g., list_audio_backends)
        so that SpeechBrain imports cleanly on environments with older torchaudio builds.
        """
        if self._embedding_model is None:
            # Shim torchaudio compatibility if needed (some torchaudio versions lack list_audio_backends)
            try:
                import importlib

                if importlib.util.find_spec("torchaudio"):
                    import torchaudio

                    if not hasattr(torchaudio, "list_audio_backends"):

                        def _list_audio_backends():
                            # best-effort guess of available backends; not exhaustive
                            backends = []
                            try:
                                # prefer sox_io and soundfile as common options
                                backends.append("sox_io")
                            except Exception:
                                pass
                            try:
                                backends.append("soundfile")
                            except Exception:
                                pass
                            if not backends:
                                backends = ["sox_io"]
                            return backends

                        torchaudio.list_audio_backends = _list_audio_backends

                    if not hasattr(torchaudio, "get_audio_backend"):
                        torchaudio.get_audio_backend = lambda: torchaudio.list_audio_backends()[0]
            except Exception:
                # best-effort only, don't prevent embedding loading attempt
                pass

            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self.logger.info(f"Loading embedding model: {self.config.embedding_model_id}")

                import os

                # Prefer to disable HF symlinks up-front on Windows to prevent permission errors
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

                # Try a robust direct download into a local models directory to avoid symlinks entirely
                dest_dir = str(self.models_dir / self.config.embedding_model_id.replace("/", "_"))
                try:
                    from huggingface_hub import snapshot_download

                    self.logger.info(
                        f"Attempting to snapshot_download model to local dir {dest_dir} (no symlinks)"
                    )
                    os.makedirs(dest_dir, exist_ok=True)
                    snapshot_download(
                        repo_id=self.config.embedding_model_id,
                        local_dir=dest_dir,
                        local_dir_use_symlinks=False,
                    )
                    # Try to load from the locally downloaded snapshot
                    try:
                        self._embedding_model = EncoderClassifier.from_hparams(
                            source=dest_dir,
                            savedir=dest_dir,
                            run_opts={"device": self.device},
                        )
                        self.logger.info("Embedding model loaded successfully from local snapshot")
                        # mark that we used speechbrain
                        self._embedding_model_is_speechbrain = True
                        return
                    except Exception as e_local:
                        self.logger.warning(f"Local snapshot load failed: {e_local}")
                except Exception:
                    # snapshot_download not available or failed; continue with other strategies
                    pass

                try:
                    # First try: load directly from hf cache (no savedir) - this typically avoids writing symlinks
                    self._embedding_model = EncoderClassifier.from_hparams(
                        source=self.config.embedding_model_id,
                        run_opts={"device": self.device},
                    )
                    self.logger.info("Embedding model loaded successfully (from HF cache)")
                    self._embedding_model_is_speechbrain = True
                    return
                except Exception as e:
                    err_msg = str(e)

                    # Detect Windows symlink permission error and retry with savedir + disabled symlink env
                    if (
                        ("A required privilege" in err_msg)
                        or ("symlink" in err_msg.lower())
                        or getattr(e, "winerror", None) == 1314
                    ):
                        try:
                            os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
                            self.logger.warning(
                                "Detected symlink/permission issue; retrying model load with HF_HUB_DISABLE_SYMLINKS=1 and specifying savedir"
                            )
                            self._embedding_model = EncoderClassifier.from_hparams(
                                source=self.config.embedding_model_id,
                                savedir=str(self.models_dir / "spkrec-ecapa"),
                                run_opts={"device": self.device},
                            )
                            self.logger.info(
                                "Embedding model loaded successfully (after disabling symlinks)"
                            )
                            self._embedding_model_is_speechbrain = True
                            return
                        except Exception:
                            # Try monkeypatching SB fetch to use COPY
                            try:
                                import speechbrain.utils.fetching as sbfetch

                                orig_fetch = sbfetch.fetch

                                def _fetch_copy(*args, **kwargs):
                                    kwargs.setdefault("local_strategy", sbfetch.LocalStrategy.COPY)
                                    return orig_fetch(*args, **kwargs)

                                sbfetch.fetch = _fetch_copy
                                self.logger.info(
                                    "Retrying model load with SpeechBrain fetch set to COPY strategy"
                                )
                                self._embedding_model = EncoderClassifier.from_hparams(
                                    source=self.config.embedding_model_id,
                                    savedir=str(self.models_dir / "spkrec-ecapa"),
                                    run_opts={"device": self.device},
                                )
                                self.logger.info(
                                    "Embedding model loaded successfully (after switching fetch strategy)"
                                )
                                self._embedding_model_is_speechbrain = True
                                return
                            except Exception as e3:
                                err_msg = str(e3)
                            finally:
                                try:
                                    sbfetch.fetch = orig_fetch
                                except Exception:
                                    pass

                    self.logger.error(f"Failed to load SpeechBrain embedding model: {err_msg}")

                    # Try to salvage by copying an existing cached snapshot or downloading directly into dest_dir
                    try:
                        import re
                        import shutil

                        m = re.search(r"'([^']+)'\s*->\s*'([^']+)'", err_msg)
                        if m:
                            src_file = m.group(1)
                            src_dir = os.path.dirname(src_file)
                            self.logger.info(
                                f"Attempting to copy cached snapshot from {src_dir} to {dest_dir}"
                            )
                            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

                            # Retry loading from the local copied directory
                            try:
                                self._embedding_model = EncoderClassifier.from_hparams(
                                    source=dest_dir,
                                    savedir=dest_dir,
                                    run_opts={"device": self.device},
                                )
                                self.logger.info(
                                    "Embedding model loaded successfully (after copying cached snapshot)"
                                )
                                self._embedding_model_is_speechbrain = True
                                return
                            except Exception as e4:
                                err_msg = str(e4)

                        # As a last resort, try to download model files directly into dest_dir using huggingface_hub APIs
                        from huggingface_hub import hf_hub_download, list_repo_files

                        self.logger.info(
                            f"Attempting direct HF download into {dest_dir} to avoid symlinks"
                        )
                        os.makedirs(dest_dir, exist_ok=True)
                        files = list_repo_files(self.config.embedding_model_id)
                        for fname in files:
                            if fname.endswith("/"):
                                continue
                            hf_hub_download(
                                repo_id=self.config.embedding_model_id,
                                filename=fname,
                                local_dir=dest_dir,
                                local_dir_use_symlinks=False,
                            )

                        # Retry loading now that files are locally present
                        self._embedding_model = EncoderClassifier.from_hparams(
                            source=dest_dir,
                            savedir=dest_dir,
                            run_opts={"device": self.device},
                        )
                        self.logger.info(
                            "Embedding model loaded successfully (after direct HF download)"
                        )
                        self._embedding_model_is_speechbrain = True
                        return
                    except Exception as e5:
                        err_msg = str(e5)

                    self.logger.warning(
                        "Common fixes: install a compatible torchaudio (matching your PyTorch), and install 'soundfile' or enable 'sox_io' backend."
                    )

                    # If user allows fallback, provide MFCC fallback; otherwise raise an error to enforce SpeechBrain usage
                    if getattr(self.config, "allow_fallback", False):
                        self.logger.warning(
                            "Falling back to MFCC-based deterministic embeddings (will be less accurate)."
                        )
                        self._embedding_model = "FALLBACK"
                        self._fallback_extractor = self._mfcc_embedding
                        return
                    else:
                        raise RuntimeError(
                            "Failed to load SpeechBrain embedding model and 'allow_fallback' is False. "
                            "Ensure torchaudio and speechbrain are installed, or set 'allow_fallback=True' in DiarizationConfig."
                        )
            except Exception:
                # Import of SpeechBrain failed entirely; honor allow_fallback setting
                self.logger.warning(
                    "Could not import SpeechBrain; checking 'allow_fallback' setting"
                )
                if getattr(self.config, "allow_fallback", False):
                    self.logger.warning(
                        "Falling back to MFCC-based deterministic embeddings (allow_fallback=True)"
                    )
                    self._embedding_model = "FALLBACK"
                    self._fallback_extractor = self._mfcc_embedding
                else:
                    raise RuntimeError(
                        "Failed to import or initialize SpeechBrain embedding model and 'allow_fallback' is False. "
                        "Install SpeechBrain or set 'allow_fallback=True' in DiarizationConfig to allow deterministic fallback."
                    )

    def _mfcc_embedding(
        self, segment_np: np.ndarray, sample_rate: int, target_dim: int = 192
    ) -> np.ndarray:
        """Compute a deterministic embedding from audio segment using MFCCs.

        Falls back to simple waveform statistics if librosa is not available.
        Returns a fixed-size vector of length `target_dim`.
        """
        try:
            import librosa

            mfcc = librosa.feature.mfcc(y=segment_np, sr=sample_rate, n_mfcc=40)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            vec = np.concatenate([mfcc_mean, mfcc_std])
        except Exception:
            # Minimal deterministic fallback: use downsampled waveform statistics + spectral centroid approximation
            vec = []
            vec.append(np.mean(segment_np))
            vec.append(np.std(segment_np))
            # simple spectral centroid proxy
            freqs = np.fft.rfftfreq(len(segment_np), d=1.0 / sample_rate)
            spec = np.abs(np.fft.rfft(segment_np))
            if spec.sum() > 0:
                centroid = float((freqs * spec).sum() / spec.sum()) / (sample_rate / 2)
            else:
                centroid = 0.0
            vec.append(centroid)
            vec = np.array(vec, dtype=float)

        # Pad or trim to target_dim
        if len(vec) < target_dim:
            padded = np.zeros(target_dim, dtype=float)
            padded[: len(vec)] = vec
            vec = padded
        elif len(vec) > target_dim:
            vec = vec[:target_dim]

        # normalize
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def process(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        cache_dir: Optional[str] = None,
        audio_id: Optional[str] = None,
        fast_mode: bool = False,
    ) -> List[SpeakerSegment]:
        """
        Main diarization pipeline.

        Args:
            waveform: Audio waveform [1, T]
            sample_rate: Audio sample rate
            num_speakers: Known number of speakers (auto-detect if None)

        Returns:
            List of SpeakerSegment with speaker assignments
        """
        self._load_embedding_model()

        # Step 1: Voice Activity Detection
        speech_regions = self._detect_speech(waveform, sample_rate)

        if not speech_regions:
            self.logger.warning("No speech detected in audio")
            return []

        self.logger.info(f"Detected {len(speech_regions)} speech regions")

        # Step 2: Create analysis windows
        windows = self._create_windows(speech_regions)

        if not windows:
            self.logger.warning("No valid windows created")
            return []

        self.logger.info(f"Created {len(windows)} analysis windows")

        # Step 3: Extract speaker embeddings
        embeddings = self._extract_embeddings(waveform, windows, sample_rate)

        self.logger.info(f"Extracted embeddings with shape: {embeddings.shape}")

        # Step 4: Cluster embeddings
        labels = self._cluster_embeddings(
            embeddings, num_speakers=num_speakers or self.config.max_speakers
        )

        num_speakers_found = len(set(labels))
        self.logger.info(f"Found {num_speakers_found} speakers")

        # Step 5: Create segments from windows and labels
        raw_segments = self._create_segments(windows, labels, embeddings)

        # Step 6: Post-processing
        processed_segments = self._postprocess_segments(raw_segments)

        # Step 7: Detect overlapping speech
        processed_segments = self._detect_overlaps(processed_segments)

        self.logger.info(f"Final: {len(processed_segments)} segments")

        return processed_segments

    def auto_tune(
        self, waveform: torch.Tensor, sample_rate: int = 16000, num_speakers: Optional[int] = None
    ) -> dict:
        """Auto-tune clustering-related hyperparameters by searching simple parameter grid.

        This method extracts embeddings and tries different clustering thresholds and
        minimum cluster sizes, scoring candidates by silhouette score (and closeness
        to `num_speakers` if provided). The best parameter set is applied to
        `self.config` and returned for inspection.
        """
        # Quick extraction path
        speech_regions = self._detect_speech(waveform, sample_rate)
        if not speech_regions:
            self.logger.warning("Auto-tune: no speech regions detected; aborting tuning")
            return {}

        windows = self._create_windows(speech_regions)
        if not windows:
            self.logger.warning("Auto-tune: no analysis windows created; aborting tuning")
            return {}

        embeddings = self._extract_embeddings(waveform, windows, sample_rate)
        if embeddings is None or len(embeddings) < 4:
            self.logger.warning("Auto-tune: insufficient embeddings for tuning; aborting tuning")
            return {}

        # Parameter grid (coarse)
        clustering_thresholds = [0.95, 0.85, 0.7, 0.5, 0.3, 0.15]
        min_cluster_sizes = [1, 2, 3, 4]

        best_score = -1e9
        best_params = {
            "clustering_threshold": self.config.clustering_threshold,
            "min_cluster_size": self.config.min_cluster_size,
            "iterative_merge_threshold": self.config.iterative_merge_threshold,
        }

        # Save original values to restore if needed
        orig_threshold = self.config.clustering_threshold
        orig_min_size = self.config.min_cluster_size
        orig_iter_thresh = self.config.iterative_merge_threshold

        try:
            for thr in clustering_thresholds:
                for msize in min_cluster_sizes:
                    # Temporarily set
                    self.config.clustering_threshold = thr
                    self.config.min_cluster_size = msize

                    try:
                        labels = self._cluster_embeddings(embeddings, num_speakers=None)
                        k = len(np.unique(labels))
                        if k <= 1:
                            sil = 0.0
                        else:
                            try:
                                sil = silhouette_score(embeddings, labels, metric="cosine")
                            except Exception:
                                sil = 0.0

                        # Scoring: prefer higher silhouette and closeness to desired num_speakers
                        score = sil
                        if num_speakers is not None:
                            score -= 0.1 * abs(k - num_speakers)
                        # small penalty for many clusters
                        score -= 0.02 * k

                        self.logger.debug(
                            f"Auto-tune candidate: thr={thr}, min_size={msize} -> k={k}, sil={sil:.4f}, score={score:.4f}"
                        )

                        if score > best_score:
                            best_score = score
                            best_params = {
                                "clustering_threshold": thr,
                                "min_cluster_size": msize,
                                "achieved_k": k,
                                "silhouette": sil,
                            }
                    except Exception as e:
                        self.logger.debug(f"Auto-tune candidate failed: {e}")
                        continue

            # Apply best params
            self.config.clustering_threshold = float(
                best_params.get("clustering_threshold", orig_threshold)
            )
            self.config.min_cluster_size = int(best_params.get("min_cluster_size", orig_min_size))
            # If a desired num_speakers was provided, set target merge accordingly
            if num_speakers is not None:
                self.config.target_num_speakers = int(num_speakers)

            self.logger.info(f"Auto-tune selected: {best_params}")
            return best_params
        finally:
            # nothing to restore; we've intentionally applied best params
            pass

    def _detect_speech(self, waveform: torch.Tensor, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect speech regions using energy-based VAD.

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate

        Returns:
            List of (start, end) tuples for speech regions
        """
        waveform_np = waveform.squeeze().cpu().numpy()

        # Frame parameters
        frame_length_ms = 25  # 25ms frames
        hop_length_ms = 10  # 10ms hop

        frame_length = int(frame_length_ms * sample_rate / 1000)
        hop_length = int(hop_length_ms * sample_rate / 1000)

        # Calculate energy per frame
        num_frames = max(1, 1 + (len(waveform_np) - frame_length) // hop_length)
        energies = np.zeros(num_frames)

        for i in range(num_frames):
            start_idx = i * hop_length
            end_idx = min(start_idx + frame_length, len(waveform_np))
            frame = waveform_np[start_idx:end_idx]

            if len(frame) > 0:
                energies[i] = np.sqrt(np.mean(frame**2) + 1e-10)

        # Compute adaptive threshold
        if len(energies) > 0:
            energy_sorted = np.sort(energies)
            # Use 30th percentile as noise floor estimate
            noise_floor = energy_sorted[int(0.3 * len(energy_sorted))]
            threshold = noise_floor + self.config.vad_threshold * np.std(energies)
        else:
            threshold = self.config.vad_threshold

        # Find speech regions
        is_speech = energies > threshold

        # Apply morphological operations to smooth
        # (simple dilation and erosion using convolution)
        kernel_size = max(1, int(self.config.min_speech_duration * 1000 / hop_length_ms))

        if kernel_size > 1 and len(is_speech) > kernel_size:
            # Simple smoothing
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(is_speech.astype(float), kernel, mode="same")
            is_speech = smoothed > 0.5

        # Convert to time regions
        regions = []
        in_speech = False
        speech_start = 0.0

        for i, speech in enumerate(is_speech):
            time = i * hop_length / sample_rate

            if speech and not in_speech:
                speech_start = time
                in_speech = True
            elif not speech and in_speech:
                duration = time - speech_start
                if duration >= self.config.min_speech_duration:
                    regions.append((speech_start, time))
                in_speech = False

        # Handle last region
        if in_speech:
            end_time = len(waveform_np) / sample_rate
            duration = end_time - speech_start
            if duration >= self.config.min_speech_duration:
                regions.append((speech_start, end_time))

        # Merge nearby regions
        regions = self._merge_nearby_regions(regions, self.config.min_silence_duration)

        return regions

    def _merge_nearby_regions(
        self, regions: List[Tuple[float, float]], min_gap: float
    ) -> List[Tuple[float, float]]:
        """Merge regions that are close together"""
        if not regions:
            return []

        merged = [regions[0]]

        for start, end in regions[1:]:
            last_start, last_end = merged[-1]

            if start - last_end <= min_gap:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))

        return merged

    def _create_windows(
        self, speech_regions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Create sliding windows over speech regions for embedding extraction"""
        windows = []

        for region_start, region_end in speech_regions:
            t = region_start

            while t < region_end:
                window_end = min(t + self.config.segment_window, region_end)

                # Only include windows with sufficient duration
                if (window_end - t) >= self.config.min_segment_duration:
                    # Avoid creating too many tiny windows across short recordings
                    if (region_end - region_start) < (self.config.segment_window * 2):
                        # for short regions, use a single window covering the region
                        windows.append((region_start, region_end))
                        break
                    windows.append((t, window_end))

                t += self.config.segment_hop

        return windows

    def _extract_embeddings(
        self,
        waveform: torch.Tensor,
        windows: List[Tuple[float, float]],
        sample_rate: int,
        cache_dir: Optional[str] = None,
        audio_id: Optional[str] = None,
        fast_mode: bool = False,
    ) -> np.ndarray:
        """Extract speaker embeddings for each window.

        Optimizations implemented:
        - Disk cache (if enabled in config and cache_dir provided)
        - Batch extraction using model's batch API when available
        - Fast MFCC embedding path when `use_fast_embedding` is True
        """
        # Try disk cache first
        if (
            cache_dir
            and audio_id
            and self.config.embedding_cache
            and getattr(self.config, "embedding_cache", True)
        ):
            try:
                import os

                cache_path = Path(cache_dir) / f"{audio_id}_embeddings.npy"
                if cache_path.exists():
                    arr = np.load(str(cache_path))
                    if arr.shape[0] == len(windows):
                        self.logger.info(f"Loaded embeddings from cache: {cache_path}")
                        return arr
            except Exception:
                pass

        n = len(windows)
        embeddings = [None] * n

        # If fallback or user requested fast embedding, compute MFCC-based embeddings vectorized
        if (
            (self._embedding_model == "FALLBACK" or self._embedding_model is None)
            or getattr(self.config, "use_fast_embedding", False)
            or fast_mode
        ):
            for i, (start, end) in enumerate(windows):
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment = waveform[:, start_sample:end_sample]
                try:
                    seg_np = segment.squeeze().cpu().numpy()
                    emb = self._fallback_extractor(seg_np, sample_rate)
                except Exception:
                    seg_np = segment.squeeze().cpu().numpy()
                    emb = self._mfcc_embedding(seg_np, sample_rate)
                embeddings[i] = emb

            embeddings = np.stack(embeddings, axis=0)

            # Save to cache
            try:
                if cache_dir and audio_id and self.config.embedding_cache:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    np.save(str(Path(cache_dir) / f"{audio_id}_embeddings.npy"), embeddings)
            except Exception:
                pass

            return embeddings

        # Otherwise use model batch encoding when available
        batch_size = max(1, int(getattr(self.config, "embedding_batch_size", 32)))

        # Prepare segment numpy arrays
        segs = []
        seg_indices = []
        for i, (start, end) in enumerate(windows):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment = waveform[:, start_sample:end_sample]
            segs.append(segment)
            seg_indices.append(i)

        # Try batch processing
        try:
            # If model supports encode_batch on a list or stacked tensor, process in chunks
            for i in range(0, len(segs), batch_size):
                batch = segs[i : i + batch_size]
                # Stack into a tensor batch
                try:
                    batch_tensor = torch.stack(
                        [b.squeeze(0) if b.dim() == 2 else b for b in batch], dim=0
                    )
                except Exception:
                    # Some models expect list of tensors; keep as list
                    batch_tensor = batch

                with torch.no_grad():
                    try:
                        # Move to model device if available
                        if hasattr(self._embedding_model, "device") and isinstance(
                            batch_tensor, torch.Tensor
                        ):
                            batch_tensor = batch_tensor.to(self._embedding_model.device)

                        out = None
                        # Try the most common batch API names
                        if hasattr(self._embedding_model, "encode_batch"):
                            out = self._embedding_model.encode_batch(batch_tensor)
                        elif hasattr(self._embedding_model, "encode"):
                            out = self._embedding_model.encode(batch_tensor)
                        else:
                            # fallback: try to call on each separately
                            out = [self._embedding_model.encode_batch(x) for x in batch]

                        # Normalize outputs into numpy array
                        if isinstance(out, torch.Tensor):
                            out_np = out.cpu().numpy()
                        elif isinstance(out, list):
                            out_np = np.stack(
                                [
                                    (
                                        o.squeeze().cpu().numpy()
                                        if isinstance(o, torch.Tensor)
                                        else np.array(o)
                                    )
                                    for o in out
                                ],
                                axis=0,
                            )
                        else:
                            out_np = np.array(out)

                        # assign back to embeddings
                        for j, idx in enumerate(range(i, i + out_np.shape[0])):
                            embeddings[idx] = out_np[j]

                    except Exception as e:
                        # fallback to per-segment extraction for this batch
                        self.logger.debug(f"Batch embedding failed, falling back per-segment: {e}")
                        for bb_idx, seg in enumerate(batch):
                            try:
                                with torch.no_grad():
                                    if hasattr(self._embedding_model, "device") and isinstance(
                                        seg, torch.Tensor
                                    ):
                                        seg = seg.to(self._embedding_model.device)
                                    emb = self._embedding_model.encode_batch(seg)
                                    emb = emb.squeeze().cpu().numpy()
                            except Exception:
                                emb = np.random.randn(192).astype(np.float32)
                            embeddings[i + bb_idx] = emb

            embeddings = np.stack(embeddings, axis=0)

            # Save to cache
            try:
                if cache_dir and audio_id and self.config.embedding_cache:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    np.save(str(Path(cache_dir) / f"{audio_id}_embeddings.npy"), embeddings)
            except Exception:
                pass

            return embeddings

        except Exception as e:
            self.logger.warning(f"Batch embedding extraction failed: {e}")
            # final fallback: single extraction loop
            embeddings = []
            for start, end in windows:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment = waveform[:, start_sample:end_sample]
                try:
                    with torch.no_grad():
                        if hasattr(self._embedding_model, "device"):
                            segment = segment.to(self._embedding_model.device)
                        emb = self._embedding_model.encode_batch(segment)
                        emb = emb.squeeze().cpu().numpy()
                except Exception:
                    emb = np.random.randn(192).astype(np.float32)
                embeddings.append(emb)

            embeddings = np.stack(embeddings, axis=0)
            return embeddings

    def _cluster_embeddings(
        self, embeddings: np.ndarray, num_speakers: Optional[int] = None
    ) -> np.ndarray:
        """Cluster embeddings to assign speaker labels, with small-cluster merging."""
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)

        # Support both nested (Config.diarization.clustering) and flat config shapes
        if hasattr(self.config, "clustering"):
            method = self.config.clustering.method
            threshold = self.config.clustering.threshold
            linkage = self.config.clustering.linkage
            min_size_cfg = getattr(
                self.config.clustering,
                "min_cluster_size",
                getattr(self.config, "min_cluster_size", 2),
            )
            max_speakers_cfg = getattr(self.config, "max_speakers", None)
        else:
            method = getattr(self.config, "clustering_method", "spectral")
            threshold = getattr(self.config, "clustering_threshold", 0.7)
            linkage = getattr(self.config, "clustering_linkage", "average")
            min_size_cfg = getattr(self.config, "min_cluster_size", 2)
            max_speakers_cfg = getattr(self.config, "max_speakers", None)

        if method == "agglomerative":
            if num_speakers is not None:
                clustering = AgglomerativeClustering(
                    n_clusters=num_speakers, metric="cosine", linkage=linkage
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    metric="cosine",
                    linkage=linkage,
                )

        elif method == "spectral":
            n_clusters = num_speakers or min(8, len(embeddings) // 2)
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=min(10, len(embeddings) - 1),
            )

        elif method == "kmeans":
            n_clusters = num_speakers or min(8, len(embeddings) // 2)
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        try:
            labels = clustering.fit_predict(embeddings_norm)
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            labels = np.array([i % 2 for i in range(len(embeddings))])

        # Debug: cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        sizes = dict(zip(unique.tolist(), counts.tolist()))
        self.logger.debug(f"Initial clusters: {len(unique)}, sizes: {sizes}")

        # Global check: if all embeddings are very similar, collapse directly to 1 speaker
        try:
            # First, perform a row-normalized (per-embedding) cosine check on raw embeddings
            row_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
            n_sample = min(200, len(row_norm))
            idx = np.linspace(0, len(row_norm) - 1, n_sample).astype(int)
            sub = row_norm[idx]
            sims = np.dot(sub, sub.T)
            sims = np.clip(sims, -1.0, 1.0)
            dists = 1.0 - sims
            mean_row_dist = (
                float(np.mean(dists[np.triu_indices_from(dists, k=1)])) if n_sample > 1 else 1.0
            )
            global_row_threshold = getattr(self.config, "global_collapse_threshold", 0.03)
            # Be more permissive for short recordings (few windows)
            if len(embeddings) < 40:
                global_row_threshold = max(global_row_threshold, 0.08)
            if mean_row_dist < global_row_threshold:
                self.logger.info(
                    f"Row-normalized embeddings too similar (mean dist={mean_row_dist:.6f}), collapsing to 1 speaker"
                )
                return np.zeros(len(embeddings), dtype=int)

            # Next, check on scaled embeddings (existing logic)
            n_sample = min(200, len(embeddings_norm))
            idx = np.linspace(0, len(embeddings_norm) - 1, n_sample).astype(int)
            sub = embeddings_norm[idx]
            sims = np.dot(sub, sub.T)
            sims = np.clip(sims, -1.0, 1.0)
            dists = 1.0 - sims
            mean_global_dist = (
                float(np.mean(dists[np.triu_indices_from(dists, k=1)])) if n_sample > 1 else 1.0
            )
            global_collapse_threshold = getattr(self.config, "global_collapse_threshold", 0.03)
            if mean_global_dist < global_collapse_threshold:
                self.logger.info(
                    f"Global embeddings too similar (mean dist={mean_global_dist:.4f}), collapsing to 1 speaker"
                )
                return np.zeros(len(embeddings), dtype=int)

            # Additional small-variance heuristic: if feature-wise std is tiny, collapse as well
            mean_std = float(np.mean(np.std(embeddings_norm, axis=0)))
            std_threshold = getattr(self.config, "global_std_threshold", 1e-2)
            if mean_std < std_threshold:
                self.logger.info(
                    f"Embeddings have tiny variance (mean std={mean_std:.6f}), collapsing to 1 speaker"
                )
                return np.zeros(len(embeddings), dtype=int)
        except Exception:
            pass

        # If centroids are very close to each other, this is likely a single-speaker recording.
        # Compute mean pairwise centroid cosine distance; if below a threshold, collapse to 1 cluster.
        try:
            labels_unique = np.unique(labels)
            centroids = [embeddings_norm[labels == l].mean(axis=0) for l in labels_unique]
            if len(centroids) > 1:
                pair_dists = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        a = centroids[i] / (np.linalg.norm(centroids[i]) + 1e-12)
                        b = centroids[j] / (np.linalg.norm(centroids[j]) + 1e-12)
                        pair_dists.append(1.0 - float(np.dot(a, b)))
                mean_pair_dist = float(np.mean(pair_dists)) if pair_dists else 1.0
            else:
                mean_pair_dist = 1.0

            collapse_threshold = getattr(self.config, "collapse_threshold", 0.15)
            if mean_pair_dist < collapse_threshold:
                self.logger.info(
                    f"Centroids too similar (mean dist={mean_pair_dist:.3f}), collapsing to 1 speaker"
                )
                labels = np.zeros_like(labels)

            # If SpeechBrain embeddings are used and clusters have a very low silhouette score,
            # it's likely that the recording is single-speaker and clustering is over-fragmenting.
            try:
                if getattr(self.config, "use_speechbrain", True) and getattr(
                    self, "_embedding_model_is_speechbrain", False
                ):
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 1:
                        try:
                            score = silhouette_score(embeddings_norm, labels, metric="cosine")
                            if score < getattr(self.config, "silhouette_collapse_threshold", 0.05):
                                self.logger.info(
                                    f"Low silhouette score ({score:.4f}) detected with SpeechBrain embeddings; collapsing to 1 speaker"
                                )
                                return np.zeros(len(embeddings), dtype=int)
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass

        # Merge clusters smaller than min_cluster_size
        min_size = min_size_cfg
        if min_size and min_size > 1:
            changed = True
            while changed:
                changed = False
                labels_unique, label_counts = np.unique(labels, return_counts=True)
                small_labels = [l for l, c in zip(labels_unique, label_counts) if c < min_size]
                if not small_labels:
                    break

                # compute centroids for existing labels
                centroids = {l: embeddings_norm[labels == l].mean(axis=0) for l in labels_unique}

                for sl in small_labels:
                    candidates = [l for l in labels_unique if l != sl]
                    if not candidates:
                        continue

                    # find nearest centroid (cosine distance)
                    def cosine_dist(a, b):
                        a_norm = a / (np.linalg.norm(a) + 1e-12)
                        b_norm = b / (np.linalg.norm(b) + 1e-12)
                        return 1.0 - float(np.dot(a_norm, b_norm))

                    distances = [(c, cosine_dist(centroids[sl], centroids[c])) for c in candidates]
                    nearest = min(distances, key=lambda x: x[1])[0]

                    # reassign labels
                    labels[labels == sl] = nearest
                    changed = True

        # Final cluster sizes
        unique2, counts2 = np.unique(labels, return_counts=True)
        sizes2 = dict(zip(unique2.tolist(), counts2.tolist()))
        self.logger.debug(f"Clusters after merge: {len(unique2)}, sizes: {sizes2}")

        # Additional centroid-based merging: merge clusters whose centroids are very close
        try:
            labels_unique = np.unique(labels)
            centroids = {l: embeddings_norm[labels == l].mean(axis=0) for l in labels_unique}
            # compute pairwise centroid distances
            pairs = []
            for i, a in enumerate(labels_unique):
                for j, b in enumerate(labels_unique):
                    if j <= i:
                        continue
                    dist = 1.0 - float(
                        np.dot(
                            centroids[a] / (np.linalg.norm(centroids[a]) + 1e-12),
                            centroids[b] / (np.linalg.norm(centroids[b]) + 1e-12),
                        )
                    )
                    pairs.append((dist, a, b))

            # merge pairs with distance < threshold
            pairs.sort()
            merged = False
            for dist, a, b in pairs:
                if dist < threshold:
                    # merge b into a
                    labels[labels == b] = a
                    merged = True

            if merged:
                labels_unique2, counts2 = np.unique(labels, return_counts=True)
                sizes2 = dict(zip(labels_unique2.tolist(), counts2.tolist()))
                self.logger.debug(
                    f"Clusters after centroid-merge: {len(labels_unique2)}, sizes: {sizes2}"
                )

            # Iterative silhouette-guided merging: try merging closest centroid pairs while it improves or meets configured criteria
            try:
                iterative_thresh = getattr(self.config, "iterative_merge_threshold", threshold)
                silhouette_min = getattr(self.config, "iterative_merge_silhouette_threshold", 0.0)
                max_merge_iters = getattr(self.config, "iterative_merge_max_iters", 10)

                def compute_centroids(curr_labels):
                    uniq = np.unique(curr_labels)
                    return {l: embeddings_norm[curr_labels == l].mean(axis=0) for l in uniq}

                def pairwise_min_pair(centroids_dict):
                    uniq = list(centroids_dict.keys())
                    best = (1.0, None, None)
                    for i, a in enumerate(uniq):
                        for j in range(i + 1, len(uniq)):
                            b = uniq[j]
                            a_c = centroids_dict[a] / (np.linalg.norm(centroids_dict[a]) + 1e-12)
                            b_c = centroids_dict[b] / (np.linalg.norm(centroids_dict[b]) + 1e-12)
                            dist = 1.0 - float(np.dot(a_c, b_c))
                            if dist < best[0]:
                                best = (dist, a, b)
                    return best

                curr_labels = labels.copy()
                prev_score = None
                try:
                    if len(np.unique(curr_labels)) > 1:
                        prev_score = silhouette_score(embeddings_norm, curr_labels, metric="cosine")
                except Exception:
                    prev_score = None

                iters = 0
                while iters < max_merge_iters:
                    iters += 1
                    cent = compute_centroids(curr_labels)
                    if len(cent) <= 1:
                        break
                    min_dist, a, b = pairwise_min_pair(cent)
                    if min_dist >= iterative_thresh:
                        break

                    # simulate merge and evaluate silhouette
                    next_labels = curr_labels.copy()
                    next_labels[next_labels == b] = a

                    try:
                        if len(np.unique(next_labels)) > 1:
                            next_score = silhouette_score(
                                embeddings_norm, next_labels, metric="cosine"
                            )
                        else:
                            next_score = 1.0
                    except Exception:
                        next_score = None

                    accept = False
                    if next_score is not None:
                        if prev_score is None:
                            # accept merges that meet a minimum silhouette threshold
                            if next_score >= silhouette_min:
                                accept = True
                        else:
                            # accept if silhouette improves by a small margin or stays acceptable
                            if next_score >= prev_score or next_score >= silhouette_min:
                                accept = True

                    if accept:
                        curr_labels = next_labels
                        prev_score = next_score
                        labels = curr_labels.copy()
                        # continue iterating
                    else:
                        break

                if iters > 1:
                    labels_unique2, counts2 = np.unique(labels, return_counts=True)
                    sizes2 = dict(zip(labels_unique2.tolist(), counts2.tolist()))
                    self.logger.debug(
                        f"Clusters after iterative-merge (iters={iters}): {len(labels_unique2)}, sizes: {sizes2}"
                    )

                # If user requested a target speaker count, greedily merge closest centroid pairs until we meet it
                try:
                    target_k = getattr(self.config, "target_num_speakers", None)
                    force_thresh = float(getattr(self.config, "target_force_threshold", 1.0))
                    if target_k is not None:
                        curr_labels = labels.copy()

                        def compute_centroids(curr):
                            uniq = np.unique(curr)
                            return {l: embeddings_norm[curr == l].mean(axis=0) for l in uniq}

                        merged_iters = 0
                        while len(np.unique(curr_labels)) > target_k:
                            cent = compute_centroids(curr_labels)
                            if len(cent) <= 1:
                                break
                            # find closest pair
                            uniq = list(cent.keys())
                            best = (1.0, None, None)
                            for i, a in enumerate(uniq):
                                for j in range(i + 1, len(uniq)):
                                    b = uniq[j]
                                    a_c = cent[a] / (np.linalg.norm(cent[a]) + 1e-12)
                                    b_c = cent[b] / (np.linalg.norm(cent[b]) + 1e-12)
                                    dist = 1.0 - float(np.dot(a_c, b_c))
                                    if dist < best[0]:
                                        best = (dist, a, b)

                            min_dist, a, b = best
                            # if min_dist is too large and force_thresh < 1.0, break
                            if min_dist > force_thresh and force_thresh < 1.0:
                                self.logger.warning(
                                    f"Stopping target-merge early: nearest cluster dist {min_dist:.3f} > force_thresh {force_thresh}"
                                )
                                break

                            # merge b into a
                            curr_labels[curr_labels == b] = a
                            merged_iters += 1
                            # safety to avoid infinite loops
                            if merged_iters > 1000:
                                break

                        if merged_iters:
                            labels = curr_labels.copy()
                            labels_unique2, counts2 = np.unique(labels, return_counts=True)
                            sizes2 = dict(zip(labels_unique2.tolist(), counts2.tolist()))
                            self.logger.info(
                                f"Clusters after target-merge (target={target_k}, iters={merged_iters}): {len(labels_unique2)}, sizes: {sizes2}"
                            )
                except Exception:
                    pass

            except Exception:
                # don't let merging errors break the pipeline
                pass

            # Heuristic fallback: if still too fragmented, run KMeans with estimated speaker count
            n_clusters_found = len(np.unique(labels))
            max_allowed = 20
            if n_clusters_found > max_allowed:
                est_k = min(12, max(2, int(len(embeddings) / 80)))
                self.logger.warning(
                    f"Too many clusters ({n_clusters_found}), falling back to KMeans with k={est_k}"
                )
                try:
                    km = KMeans(n_clusters=est_k, random_state=42, n_init=10)
                    labels = km.fit_predict(embeddings_norm)
                    # Re-merge small clusters after KMeans
                    labels_unique2, counts2 = np.unique(labels, return_counts=True)
                    sizes2 = dict(zip(labels_unique2.tolist(), counts2.tolist()))
                    self.logger.info(
                        f"Clusters after KMeans fallback: {len(labels_unique2)}, sizes: {sizes2}"
                    )
                except Exception as e:
                    self.logger.error(f"KMeans fallback failed: {e}")
        except Exception:
            pass

        return labels

    def _create_segments(
        self, windows: List[Tuple[float, float]], labels: np.ndarray, embeddings: np.ndarray
    ) -> List[SpeakerSegment]:
        """Create SpeakerSegment objects from windows and labels"""
        segments = []

        for (start, end), label, emb in zip(windows, labels, embeddings):
            segments.append(
                SpeakerSegment(
                    speaker_id=f"SPEAKER_{label:02d}",
                    start=start,
                    end=end,
                    confidence=1.0,
                    embedding=emb,
                )
            )

        # If we used the fallback extractor, update segment embeddings to the deterministic MFCC embeddings
        if getattr(self, "_fallback_extractor", None) is not None:
            try:
                for i, seg in enumerate(segments):
                    # reuse windows to create a deterministic embedding
                    s, e = windows[i]
                    # external code expects embeddings array, but ensure segment.embedding is deterministic
                    if (
                        segments[i].embedding is None
                        or isinstance(self._embedding_model, str)
                        and self._embedding_model == "FALLBACK"
                    ):
                        # compute on-demand using fallback extractor
                        seg_np = self._extract_waveform_segment(windows[i])
                        segments[i].embedding = self._fallback_extractor(seg_np, sample_rate)
            except Exception:
                pass

        return segments

    def _postprocess_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Post-process segments: merge adjacent, filter short"""
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x.start)

        # Merge adjacent segments from same speaker
        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]
            gap = seg.start - last.end

            if seg.speaker_id == last.speaker_id and gap <= self.config.merge_gap_threshold:
                # Merge: extend last segment
                last.end = max(last.end, seg.end)
                last.confidence = (last.confidence + seg.confidence) / 2
            else:
                merged.append(seg)

        # Smoothing: fix short isolated segments between identical speakers
        smoothed = merged
        if len(smoothed) >= 3:
            changed = False
            for i in range(1, len(smoothed) - 1):
                seg = smoothed[i]
                prev = smoothed[i - 1]
                nxt = smoothed[i + 1]
                threshold = max(1.0, self.config.min_segment_duration)
                if seg.duration < threshold and prev.speaker_id == nxt.speaker_id:
                    seg.speaker_id = prev.speaker_id
                    changed = True

            if changed:
                # merge again after smoothing
                merged2 = [smoothed[0]]
                for seg in smoothed[1:]:
                    last = merged2[-1]
                    gap = seg.start - last.end
                    if seg.speaker_id == last.speaker_id and gap <= self.config.merge_gap_threshold:
                        last.end = max(last.end, seg.end)
                        last.confidence = (last.confidence + seg.confidence) / 2
                    else:
                        merged2.append(seg)
                merged = merged2

        # Filter short segments
        filtered = [seg for seg in merged if seg.duration >= self.config.min_segment_duration]

        return filtered

    def _merge_segments(
        self, segments: List[SpeakerSegment], max_gap: float = 0.5
    ) -> List[SpeakerSegment]:
        """Compatibility helper: merge adjacent segments from same speaker within max_gap"""
        if not segments:
            return []

        segments = sorted(segments, key=lambda x: x.start)
        merged_list = [segments[0]]

        for seg in segments[1:]:
            last = merged_list[-1]
            gap = seg.start - last.end
            if seg.speaker_id == last.speaker_id and gap <= max_gap:
                # Merge: extend last segment
                last.end = max(last.end, seg.end)
                last.confidence = (last.confidence + seg.confidence) / 2
            else:
                merged_list.append(seg)

        return merged_list

    def _detect_overlaps(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Mark segments that overlap with other speakers"""
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i != j and seg1.speaker_id != seg2.speaker_id:
                    # Check for time overlap
                    overlap_start = max(seg1.start, seg2.start)
                    overlap_end = min(seg1.end, seg2.end)

                    if overlap_start < overlap_end:
                        seg1.is_overlap = True
                        seg2.is_overlap = True

        return segments

    def get_speaker_stats(self, segments: List[SpeakerSegment]) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each speaker.

        Returns:
            Dict mapping speaker_id to stats (total_duration, num_segments, etc.)
        """
        stats = {}

        for seg in segments:
            if seg.speaker_id not in stats:
                stats[seg.speaker_id] = {
                    "total_duration": 0.0,
                    "num_segments": 0,
                    "avg_segment_duration": 0.0,
                    "overlap_duration": 0.0,
                }

            stats[seg.speaker_id]["total_duration"] += seg.duration
            stats[seg.speaker_id]["num_segments"] += 1

            if seg.is_overlap:
                stats[seg.speaker_id]["overlap_duration"] += seg.duration

        # Calculate averages
        for speaker_id in stats:
            num_segs = stats[speaker_id]["num_segments"]
            if num_segs > 0:
                stats[speaker_id]["avg_segment_duration"] = (
                    stats[speaker_id]["total_duration"] / num_segs
                )

        return stats
