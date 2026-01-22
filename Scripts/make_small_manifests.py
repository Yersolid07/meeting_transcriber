import json
from pathlib import Path


def make():
    p1 = Path("data/manifests")
    p1.mkdir(parents=True, exist_ok=True)
    train = p1 / "train_small.jsonl"
    valid = p1 / "valid_small.jsonl"
    train_examples = [
        {
            "wav": "data/sample_audio/utt1.wav",
            "text": "saya makan nasi",
            "speaker": "spk1",
            "duration": 1.0,
        }
    ]
    valid_examples = [
        {
            "wav": "data/sample_audio/utt2.wav",
            "text": "apa kabar semua",
            "speaker": "spk2",
            "duration": 1.2,
        }
    ]
    with open(train, "w", encoding="utf-8") as fh:
        for e in train_examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(valid, "w", encoding="utf-8") as fh:
        for e in valid_examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("Wrote", train, valid)


if __name__ == "__main__":
    make()
