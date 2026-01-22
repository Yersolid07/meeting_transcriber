#!/usr/bin/env python3
"""Lightweight finetuning scaffold for Whisper-base using Hugging Face Trainer
Designed for small quick experiments; supports streaming dataset from JSONL manifest.
"""
import argparse
import os
from pathlib import Path

import evaluate
import numpy as np
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


def load_manifest(manifest_path: str):
    ds = load_dataset("json", data_files=manifest_path, split="train")
    return ds


def prepare_dataset(ds, processor, max_length_seconds=30.0, quick_limit=None):
    """Prepare dataset by loading audio directly with soundfile and extracting features with the processor.
    This avoids requiring the datasets' Audio feature which may need torchcodec on some platforms.
    """

    def map_to_features(batch):
        import soundfile as sf

        # Try known audio fields in priority
        if (
            batch.get("audio")
            and isinstance(batch.get("audio"), dict)
            and batch["audio"].get("path")
        ):
            src = batch["audio"]["path"]
            speech, sr = sf.read(src)
        elif batch.get("audio_filepath"):
            src = batch["audio_filepath"]
            speech, sr = sf.read(src)
        else:
            raise ValueError("No audio source found in example")

        # Ensure numpy float32
        import numpy as _np

        if speech.dtype != _np.float32:
            speech = speech.astype(_np.float32)

        # Whisper expects 'input_features' key (float arrays)
        batch["input_features"] = processor.feature_extractor(
            speech, sampling_rate=sr
        ).input_features[0]

        # prepare labels (text)
        # Keep tokenizer outputs (encodings dict) for the collator to pad properly
        labels = processor.tokenizer(batch.get("text", ""))
        batch["labels"] = labels
        return batch

    # Optionally limit for quick tests
    if quick_limit:
        ds = ds.select(range(min(len(ds), quick_limit)))

    ds = ds.map(map_to_features, remove_columns=ds.column_names, num_proc=1)
    return ds


def compute_metrics(pred):
    metric = evaluate.load("wer")
    preds = pred.predictions.argmax(-1) if pred.predictions.ndim == 3 else pred.predictions
    # Tokenizer decode
    # This is a simple placeholder; proper decoding with processor.tokenizer needed
    # We'll leave more thorough eval for later
    return {"wer": 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/icorpus.jsonl")
    p.add_argument("--output", default="models/whisper-base-finetuned")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="Quick run on a small subset")
    args = p.parse_args()

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    ds = load_manifest(args.manifest)

    # If dataset lacks an 'audio' column, add by loading file paths
    if "audio" not in ds.column_names and "audio_filepath" in ds.column_names:

        def add_audio(example):
            return {"audio": {"path": example["audio_filepath"]}}

        ds = ds.map(add_audio)

    ds = prepare_dataset(ds, processor, quick_limit=100 if args.quick else None)

    import torch

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=2,
        predict_with_generate=False,
    )

    # Custom data collator to pad input feature arrays and label encodings
    import torch
    from transformers import DataCollatorForSeq2Seq

    def data_collator(features: list):
        # Pad input_features (numpy arrays) to the max length
        import numpy as _np

        input_feats = [_np.asarray(f["input_features"]) for f in features]
        # support 1D or 2D features (time, feat_dim)
        max_len = max(arr.shape[0] for arr in input_feats)
        feat_dim = input_feats[0].shape[1] if input_feats[0].ndim == 2 else 1
        batch_inputs = _np.zeros((len(input_feats), max_len, feat_dim), dtype=_np.float32)
        for i, arr in enumerate(input_feats):
            if arr.ndim == 1:
                batch_inputs[i, : arr.shape[0], 0] = arr
            else:
                batch_inputs[i, : arr.shape[0], :] = arr
        input_tensor = torch.from_numpy(batch_inputs)
        # Pad labels using tokenizer
        labels = [f["labels"] for f in features]
        tokenized = processor.tokenizer.pad(labels, return_tensors="pt")
        labels_ids = tokenized["input_ids"]
        # replace pad token id with -100 for loss masking
        labels_ids[labels_ids == processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_tensor, "labels": labels_ids}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
