import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.transcriber import TranscriptSegment
from src.summarizer import BERTSummarizer

p = "cache/rapatsingkat_20260129_151744_results.json"
with open(p, encoding='utf-8') as f:
    data = json.load(f)

segs = [TranscriptSegment(s['speaker_id'], s['start'], s['end'], s['text']) for s in data.get('transcript', [])]

s = BERTSummarizer()
summary = s.summarize(segs)
print(summary.to_json())
