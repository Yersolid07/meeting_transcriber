"""Finetune Wav2Vec2 (facebook/wav2vec2-base-960h) on a manifest JSONL dataset.

Usage example:
  python scripts/finetune_wav2vec2.py --train-manifest data/manifests/icorpus_train_2000.jsonl \
      --valid-manifest data/manifests/icorpus_valid_500.jsonl --run-name sb_w2v2_icorpus_2k500 --epochs 1 --batch-size 8

This is a minimal, research-oriented finite script producing:
  - experiments/<run_name>/checkpoints/ckpt_epoch_{epoch}.pt
  - experiments/<run_name>/predictions_epoch_{epoch}.jsonl
  - experiments/<run_name>/metrics.csv

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import evaluate
import torch
import torchaudio
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        max_examples: int = None,
        target_sr: int = 16000,
        max_audio_seconds: int = 15,
    ):
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_examples and i >= max_examples:
                    break
                obj = json.loads(line)
                # support common fields: 'audio', 'audio_filepath', 'wav', 'text'
                audio_path = obj.get("audio") or obj.get("audio_filepath") or obj.get("wav")
                text = obj.get("text") or obj.get("transcript") or obj.get("label")
                if not audio_path or text is None:
                    continue
                self.samples.append({"path": audio_path, "text": text})
        self.target_sr = target_sr
        self.max_audio_seconds = max_audio_seconds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Use soundfile to avoid torchaudio torchcodec dependency
        import soundfile as sf

        waveform, sr = sf.read(s["path"], dtype="float32")
        # waveform shape: (n,) or (n, channels)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != self.target_sr:
            from torchaudio import transforms

            resampler = transforms.Resample(sr, self.target_sr)
            waveform = resampler(torch.tensor(waveform)).numpy()
        # trim or pad long audio to avoid huge memory usage on CPU during tests
        max_len = int(self.target_sr * self.max_audio_seconds)
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
        return {"audio": waveform, "sampling_rate": self.target_sr, "text": s["text"], "id": idx}


def collate_fn(batch: List[Dict], processor: Wav2Vec2Processor):
    audios = [b["audio"] for b in batch]
    texts = [b["text"] for b in batch]
    ids = [b["id"] for b in batch]
    # processor will pad to longest
    inputs = processor(
        audios, sampling_rate=batch[0]["sampling_rate"], return_tensors="pt", padding=True
    )
    # Tokenize labels using the tokenizer directly (avoid deprecated as_target_processor)
    labels = processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
    # replace pad token id's with -100 for CTC loss
    labels_mask = labels != processor.tokenizer.pad_token_id
    labels[~labels_mask] = -100
    input_values = inputs["input_values"]
    attention_mask = inputs.get("attention_mask", None)
    return input_values, attention_mask, labels, texts, ids


def decode_preds(pred_ids, processor: Wav2Vec2Processor):
    # pred_ids: (batch, seq_len)
    # replace -100 etc. just use argmaxed token ids
    return processor.batch_decode(pred_ids)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    model_name = args.model_name
    outdir = Path("experiments") / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print("Loading model and processor:", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device)

    train_ds = ManifestDataset(
        args.train_manifest, max_examples=args.max_train, max_audio_seconds=args.max_audio_seconds
    )
    valid_ds = ManifestDataset(
        args.valid_manifest, max_examples=args.max_valid, max_audio_seconds=args.max_audio_seconds
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    wer_metric = evaluate.load("wer")

    metrics_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for input_values, attention_mask, labels, texts, ids in train_loader:
            input_values = input_values.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / (steps if steps else 1.0)
        print(f"Epoch {epoch} train loss: {avg_loss:.4f}")

        # validation
        model.eval()
        preds = []
        refs = []
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for input_values, attention_mask, labels, texts, ids in valid_loader:
                input_values = input_values.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                if attention_mask is not None:
                    outputs = model(input_values, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(input_values, labels=labels)
                val_loss += outputs.loss.item()
                vsteps += 1
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)
                decoded = decode_preds(pred_ids.cpu().numpy(), processor)
                preds.extend(decoded)
                refs.extend(texts)
        avg_val_loss = val_loss / (vsteps if vsteps else 1.0)
        wer = wer_metric.compute(predictions=preds, references=refs)
        print(f"Epoch {epoch} valid loss: {avg_val_loss:.4f}, WER: {wer:.4f}")

        # save checkpoint
        ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )

        # save predictions
        pred_file = outdir / f"predictions_epoch_{epoch}.jsonl"
        with open(pred_file, "w", encoding="utf-8") as pf:
            for i, (r, p) in enumerate(zip(refs, preds)):
                pf.write(
                    json.dumps({"id": i, "reference": r, "prediction": p}, ensure_ascii=False)
                    + "\n"
                )

        metrics_rows.append({"epoch": epoch, "split": "valid", "wer": wer, "loss": avg_val_loss})

        # append metrics CSV
        metrics_csv = outdir / "metrics.csv"
        write_mode = "w" if epoch == 1 else "a"
        with open(metrics_csv, write_mode, encoding="utf-8") as mfh:
            if write_mode == "w":
                mfh.write("epoch,split,wer,loss\n")
            for row in metrics_rows[-1:]:
                mfh.write(f"{row['epoch']},{row['split']},{row['wer']},{row['loss']}\n")

    print("Training finished. Artifacts under", outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--run-name", default="wav2vec2_run")
    parser.add_argument("--model-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument(
        "--max-audio-seconds",
        type=int,
        default=15,
        help="Trim audio longer than this (seconds) to avoid huge memory during CPU runs",
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
