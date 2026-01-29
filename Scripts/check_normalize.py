import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.summarizer import AbstractiveSummarizer, SummarizationConfig

cfg = SummarizationConfig()
s = AbstractiveSummarizer(cfg)
key_points = ["Pembahasan target kuartal", "Budi menyiapkan laporan", "Siti menyiapkan materi presentasi"]
decisions = ["Sepakat melanjutkan proyek X"]
actions = [{"owner": "Budi", "task": "Menyiapkan laporan"}, {"owner": "Siti", "task": "Menyiapkan materi presentasi"}]
parts = []
if key_points:
    parts.append("Poin-Poin Penting:\n" + "\n".join([f"- {p}" for p in key_points]))
if decisions:
    parts.append("Keputusan:\n" + "\n".join([f"- {d}" for d in decisions]))
if actions:
    parts.append("Action Items:\n" + "\n".join([f"- [{a.get('owner','TBD')}] {a.get('task','')}" for a in actions]))
assembled = "\n\n".join(parts)
print('ASSEMBLED RAW:\n', assembled)
print('\nNORMALIZED:\n', repr(s._normalize_overview_text(assembled)))
