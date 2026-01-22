#!/usr/bin/env python3
"""Evaluate a Whisper checkpoint on a manifest (JSONL) and compute WER.

Usage:
    python scripts/evaluate_whisper.py --manifest data/manifests/icorpus.jsonl --model models/whisper-base-finetuned-quick --samples 200 --exclude-first 100
"""
import argparse
import json
import random
from pathlib import Path

import evaluate
import soundfile as sf
import torch
from transformers import pipeline


def load_manifest(manifest_path: str):
    m = Path(manifest_path)
    lines = [json.loads(l) for l in m.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines


def normalize_text(t: str) -> str:
    # lowercase and strip extra whitespace; remove trailing punctuation for fairer WER
    import re

    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = " ".join(t.split())
    return t


def evaluate_model(
    manifest_path: str, model_path: str, samples: int = 200, exclude_first: int = 0, seed: int = 42
):
    data = load_manifest(manifest_path)
    if exclude_first and len(data) > exclude_first:
        pool = data[exclude_first:]
    else:
        pool = data

    if len(pool) == 0:
        raise ValueError("No examples available for evaluation")

    random.seed(seed)
    sample = random.sample(pool, min(samples, len(pool)))

    device_id = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(
        f"[Eval] Loading processor from 'openai/whisper-base' and model from {model_path} on device {device}"
    )
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    except Exception:
        # fallback: try loading base and then load state dict if only weights present
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        try:
            import torch as _torch

            state = _torch.load(Path(model_path) / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state)
        except Exception:
            pass

    model.to(device)
    model.eval()

    wer_metric = evaluate.load("wer")

    hypos = []
    refs = []
    out_lines = []

    for rec in sample:
        try:
            audio_path = rec.get("audio_filepath")
            start = rec.get("start", None)
            end = rec.get("end", None)
            speech, sr = sf.read(audio_path)
            # slice if timestamps present
            if start is not None and end is not None:
                s = int(start * sr)
                e = int(end * sr)
                speech = speech[s:e]
        except Exception as e:
            print(f"[Eval] Could not read {audio_path}: {e}")
            continue

        try:
            # Run processor + model generation
            inputs = processor.feature_extractor(
                speech, sampling_rate=sr, return_tensors="pt"
            ).input_features
            inputs = inputs.to(device)
            with torch.no_grad():
                gen = model.generate(inputs)
            text = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"[Eval] ASR failed on {audio_path}: {e}")
            text = ""

        ref = rec.get("text", "")
        nref = normalize_text(ref)
        nhyp = normalize_text(text)

        hypos.append(nhyp)
        refs.append(nref)
        out_lines.append({"audio": audio_path, "ref": ref, "hyp": text, "nref": nref, "nhyp": nhyp})

    wer = wer_metric.compute(predictions=hypos, references=refs)
    print(f"[Eval] Samples evaluated: {len(hypos)} WER: {wer:.4f}")

    # Save detailed outputs
    outp = Path("logs")
    outp.mkdir(parents=True, exist_ok=True)
    outf = outp / f"eval_whisper_{Path(model_path).name}.jsonl"
    with outf.open("w", encoding="utf-8") as fw:
        for l in out_lines:
            fw.write(json.dumps(l, ensure_ascii=False) + "\n")

    return wer, out_lines


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/icorpus.jsonl")
    p.add_argument("--model", default="models/whisper-base-finetuned-quick")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--exclude-first", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wer, lines = evaluate_model(
        args.manifest,
        args.model,
        samples=args.samples,
        exclude_first=args.exclude_first,
        seed=args.seed,
    )
    print(f"Final WER: {wer:.4f}")
