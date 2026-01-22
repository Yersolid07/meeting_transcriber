import os
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import MeetingTranscriberPipeline, PipelineConfig


st.set_page_config(page_title="Meeting Transcriber", layout="wide")

st.title("Meeting Transcriber — Demo")
st.markdown("Upload an audio file or pick a sample to generate transcript, summary and downloadable DOCX.")

# Sample audio chooser
AUDIO_DIR = Path.cwd() / "data" / "audio"
# Build safe sample list: prefer paths relative to cwd, but fall back to absolute paths if not possible
SAMPLES = []
for p in AUDIO_DIR.rglob("*.mp3"):
    try:
        SAMPLES.append(str(p.relative_to(Path.cwd())))
    except ValueError:
        # Path is not under cwd (different drive or external mount), use absolute path
        SAMPLES.append(str(p.resolve()))

with st.sidebar:
    st.header("Settings")
    # Detect deployment target (e.g., set STREAMLIT_DEPLOY_TARGET=community on Streamlit Cloud)
    deploy_target = os.getenv("STREAMLIT_DEPLOY_TARGET", "")
    # Community Cloud has no GPU and limited CPU/time; default to 'fast' preset there
    default_index = 0
    default_quick_asr = False
    if deploy_target.lower() == "community":
        default_index = 2  # 'fast'
        default_quick_asr = True
        st.info("Running in Streamlit Community mode: using fast preset and quick ASR for responsiveness.")

    preset = st.selectbox("Preset", ["deployment", "balanced", "fast", "accurate"], index=default_index)
    quick_asr = st.checkbox("Quick ASR (override)", value=default_quick_asr)
    parallel_workers = st.number_input("Parallel workers (0 = auto)", min_value=0, max_value=16, value=0)
    sample_choice = st.selectbox("Pick sample audio (optional)", ["None"] + SAMPLES)

uploaded_file = st.file_uploader("Upload audio (.wav, .mp3, .m4a)")

# Determine audio path
audio_path = None
if uploaded_file is not None:
    tmpdir = tempfile.gettempdir()
    tmp_path = Path(tmpdir) / uploaded_file.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    audio_path = str(tmp_path)
elif sample_choice and sample_choice != "None":
    audio_path = sample_choice

if not audio_path:
    st.info("Upload an audio file or pick a sample from the sidebar to begin.")

# Interactive flow: run diarization first and allow manual mapping
# Clear existing session state if user changed audio selection
if "diarization_done" in st.session_state and st.session_state.get("audio_path") != audio_path:
    # Keep only unrelated session keys
    for k in ["diarization_done", "pipeline", "dz_res", "sample_segments", "snippet_transcripts", "result", "mapping"]:
        if k in st.session_state:
            del st.session_state[k]

if st.button("Run diarization only"):
    if not audio_path:
        st.error("Please provide audio first.")
    else:
        cfg = PipelineConfig(preset=preset, quick_asr=quick_asr)
        if parallel_workers and parallel_workers > 0:
            cfg.asr_parallel_workers = int(parallel_workers)
        pipeline = MeetingTranscriberPipeline(cfg)

        with st.spinner("Running diarization..."):
            try:
                dz_res = pipeline.run_diarization(audio_path)
                st.success("Diarization complete")
            except Exception as e:
                st.error(f"Diarization failed: {e}")
                raise

        # Persist state so interactive widgets survive reruns
        st.session_state["diarization_done"] = True
        st.session_state["pipeline"] = pipeline
        st.session_state["dz_res"] = dz_res
        st.session_state["audio_path"] = audio_path

# If we already have diarization state (either just-run or from previous interaction), show mapping UI
if st.session_state.get("diarization_done") and audio_path:
    pipeline = st.session_state["pipeline"]
    dz_res = st.session_state["dz_res"]

    st.write(f"Detected {len(dz_res['unique_speakers'])} speakers and {dz_res['num_segments']} segments")

    # Playable sample and quick per-speaker snippets so user can listen/read before mapping
    st.subheader("Sample snippets (listen + read before mapping)")

    # Try to reuse cached sample snippets if present
    sample_segments = st.session_state.get("sample_segments") or {}
    snippet_transcripts = st.session_state.get("snippet_transcripts") or {}

    if not sample_segments:
        try:
            dsegs = pipeline._diarization_segments or []
            for spk in dz_res["unique_speakers"]:
                cand = [s for s in dsegs if s.speaker_id == spk]
                if not cand:
                    continue
                best = max(cand, key=lambda x: x.duration)
                cap_end = min(best.end, best.start + 10.0)
                from src.diarization import SpeakerSegment

                sample_segments[spk] = SpeakerSegment(
                    speaker_id=best.speaker_id,
                    start=best.start,
                    end=cap_end,
                    confidence=best.confidence,
                    is_overlap=best.is_overlap,
                    metadata=best.metadata.copy() if getattr(best, "metadata", None) else {},
                )
            st.session_state["sample_segments"] = sample_segments
        except Exception as e:
            st.warning(f"Could not prepare sample segments: {e}")
            sample_segments = {}

    # Run quick per-segment ASR for the sample snippets (avoid full-audio mapping for speed)
    if not snippet_transcripts and sample_segments:
        try:
            transcriber = pipeline.transcriber
            orig_full_audio = getattr(transcriber.config, "use_full_audio_for_segments", False)
            transcriber.config.use_full_audio_for_segments = False
            orig_workers = getattr(transcriber.config, "parallel_workers", 1)
            transcriber.config.parallel_workers = 1

            transcripts = transcriber.transcribe_segments(
                pipeline._waveform, list(sample_segments.values()), pipeline._sample_rate
            )
            for t in transcripts:
                snippet_transcripts[t.speaker_id] = t.text

            transcriber.config.use_full_audio_for_segments = orig_full_audio
            transcriber.config.parallel_workers = orig_workers

            st.session_state["snippet_transcripts"] = snippet_transcripts
        except Exception as e:
            st.warning(f"Quick snippet transcription failed: {e}")

    # Display snippets in columns with audio player + short transcript
    import soundfile as sf
    import tempfile

    mapping = st.session_state.get("mapping") or {}
    st.subheader("Manual speaker mapping")
    audio_id = Path(audio_path).stem
    for spk in dz_res["unique_speakers"]:
        with st.expander(f"Speaker: {spk}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                seg = sample_segments.get(spk)
                if seg is not None:
                    try:
                        sr = pipeline._sample_rate
                        start_sample = int(seg.start * sr)
                        end_sample = int(seg.end * sr)
                        audio_np = pipeline._waveform[:, start_sample:end_sample].squeeze().cpu().numpy()
                        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        sf.write(tmpf.name, audio_np, sr)
                        st.audio(tmpf.name)
                    except Exception as e:
                        st.warning(f"Could not prepare audio snippet: {e}")
                else:
                    st.write("No sample segment available for this speaker")
            with col2:
                st.write("**Sample transcript:**")
                st.write(snippet_transcripts.get(spk, "(no transcription available)"))
                key = f"map_{audio_id}_{spk}"
                # Preserve user input across reruns by using session state keys
                default_val = mapping.get(spk, spk)
                mapping_val = st.text_input(f"Map {spk} to name", value=default_val, key=key)
                mapping[spk] = mapping_val

    st.session_state["mapping"] = mapping

    if st.button("Apply mapping and continue processing"):
        pipeline.apply_speaker_map(mapping, save_to_cache=True, audio_id=audio_id)
        with st.spinner("Running full processing..."):
            try:
                res = pipeline.continue_from_diarization(title="Streamlit run")
                st.session_state["result"] = res
                st.success("Processing complete")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                raise

    # If result available, display
    if st.session_state.get("result"):
        res = st.session_state["result"]
        st.subheader("Summary")
        st.json(res.summary or {})

        st.subheader("Transcript (first 5000 characters)")
        st.text(res.transcript_text[:5000])

        if res.document_path and os.path.exists(res.document_path):
            with open(res.document_path, "rb") as fh:
                doc_bytes = fh.read()
            st.download_button("Download .docx", data=doc_bytes, file_name=Path(res.document_path).name)

        st.write("---")
        st.write("Processing metadata:")
        st.write({
            "Audio duration": res.audio_duration,
            "Speakers found": res.num_speakers,
            "Segments": res.num_segments,
            "Total words": res.total_words,
            "Processing time (s)": res.processing_time,
        })

        st.balloons()

    # Allow clearing state
    if st.button("Clear diarization state"):
        for k in ["diarization_done", "pipeline", "dz_res", "sample_segments", "snippet_transcripts", "result", "mapping"]:
            if k in st.session_state:
                del st.session_state[k]
