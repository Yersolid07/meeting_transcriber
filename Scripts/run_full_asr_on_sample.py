import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import MeetingTranscriberPipeline, PipelineConfig


def run_sample(audio_path: str):
    pc = PipelineConfig()
    pc.models_dir = "./models"
    pc.output_dir = "./data/output"
    pc.cache_dir = "./cache"
    pc.verbose = True

    pipeline = MeetingTranscriberPipeline(config=pc)

    # Enable full-audio ASR mapping and run Wav2Vec2 with CTC beam decoding (no LM)
    pipeline.transcriber.config.use_full_audio_for_segments = True
    pipeline.transcriber.config.return_timestamps = "word"
    pipeline.transcriber.config.model_id = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
    pipeline.transcriber.config.backend = "transformers"
    pipeline.transcriber.config.decoder = "beam"
    pipeline.transcriber.config.beam_width = 8
    pipeline.transcriber.config.use_lm = False
    print(
        "[Runner] Using ASR model:",
        pipeline.transcriber.config.model_id,
        "backend:",
        pipeline.transcriber.config.backend,
        "decoder: beam",
    )

    result = pipeline.process(audio_path, title="Full-audio ASR Test")

    # Print transcript segments
    for seg in result.segments:
        print(f"{seg['speaker_id']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}")

    # Save result summary
    out = Path(pc.output_dir) / (Path(audio_path).stem + "_full_asr_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print("Saved result to:", out)


if __name__ == "__main__":
    run_sample("data/audio/kondisi_bersih/samplesyermia.wav")
