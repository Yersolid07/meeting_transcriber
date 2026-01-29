import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.summarizer import AbstractiveSummarizer, SummarizationConfig

p = "cache/rapatsingkat_20260129_171834_results.json"
if not os.path.exists(p):
    print("Cache file not found:", p)
    sys.exit(1)

with open(p, encoding='utf-8') as f:
    data = json.load(f)

full_text = " ".join([s.get('text','') for s in data.get('transcript', [])])
key_points = data.get('summary', {}).get('key_points', []) or []
decisions = data.get('summary', {}).get('decisions', []) or []
actions = data.get('summary', {}).get('action_items', []) or []
topics = data.get('summary', {}).get('topics', []) or []

cfg = SummarizationConfig()
# Request a long comprehensive overview
cfg.comprehensive_overview = True
cfg.comprehensive_max_length = 700
cfg.polish_overview = True

s = AbstractiveSummarizer(config=cfg)
# Use the generation function that includes sanitization, retries, and fallback
overview, keywords = s.generate_comprehensive_summary(full_text, key_points, decisions, actions, topics)

print('\n=== COMPREHENSIVE OVERVIEW ===\n')
print(overview)
print('\n=== KEYWORDS ===\n')
print(keywords)
