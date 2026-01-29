import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.summarizer import AbstractiveSummarizer, SummarizationConfig

p = "cache/rapatsingkat_20260129_151744_results.json"
with open(p, encoding='utf-8') as f:
    data = json.load(f)

full_text = " ".join([s.get('text','') for s in data.get('transcript', [])])
key_points = data.get('summary', {}).get('key_points', []) or []
decisions = data.get('summary', {}).get('decisions', []) or []
actions = data.get('summary', {}).get('action_items', []) or []
topics = data.get('summary', {}).get('topics', []) or []

cfg = SummarizationConfig()
# ensure we use same settings
cfg.comprehensive_overview = True
s = AbstractiveSummarizer(config=cfg)
s._load_model()
if s._pipeline is None:
    print('No pipeline available; cannot debug model output here.')
    sys.exit(0)

# Recreate the prompt exactly as in generate_comprehensive_summary
prompt_parts = [
    "Anda adalah asisten yang menulis ringkasan rapat yang komprehensif dan terstruktur.",
    "Output harus dalam format YAML dengan kunci: overview, key_points (list), decisions (list), action_items (list of {owner, task, due}), keywords (list).",
    "Berikan overview naratif yang jelas, serta daftar poin penting, keputusan, dan tindak lanjut.",
    "Topik yang dibahas:",
    ", ".join(topics) if topics else "-",
    "Poin-poin penting:\n" + "\n".join([f"- {p}" for p in key_points]) if key_points else "",
    "Keputusan:\n" + "\n".join([f"- {d}" for d in decisions]) if decisions else "",
    "Tindak lanjut (Action Items):\n" + "\n".join([f"- [{a.get('owner','TBD')}] {a.get('task','')}" for a in actions]) if actions else "",
    "Mohon hasilkan YAML yang valid."
]
prompt = "\n\n".join([p for p in prompt_parts if p])

print('\n=== PROMPT START ===\n')
print(prompt[:4000])
print('\n=== PROMPT END (truncated) ===\n')

print('\nCalling pipeline... (this may be slow)')
out = s._pipeline(prompt, max_length=getattr(cfg, 'comprehensive_max_length', 512), min_length=60, truncation=True, do_sample=False)
print('\n=== RAW MODEL OUTPUT ===\n')
print(out)

# Also show cleaned/parsing results
raw = out[0].get('summary_text','').strip()
print('\n=== RAW TEXT ===\n')
print(raw)
cleaned = s._collapse_repeated_phrases(raw)
print('\n=== COLLAPSED ===\n')
print(cleaned)
cleaned2 = s._clean_abstractive_text(cleaned)
print('\n=== CLEANED TEXT ===\n')
print(cleaned2)

try:
    ov, kws = s._parse_structured_output(cleaned2, {'key_points': key_points, 'decisions': decisions, 'action_items': actions})
    print('\n=== PARSED OVERVIEW ===\n', ov)
    print('\n=== PARSED KEYWORDS ===\n', kws)
except Exception as e:
    print('Parsing failed:', e)

print('\nDone')
