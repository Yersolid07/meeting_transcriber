import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.summarizer import AbstractiveSummarizer, SummarizationConfig

full_text = (
    "Rapat dimulai dengan pembukaan. Kita membahas target kuartal berikutnya. "
    "Budi akan menyiapkan laporan, dan Siti akan menyiapkan materi presentasi. Pada akhirnya, kita sepakat untuk melanjutkan proyek X."
)
key_points = [
    "Pembahasan target kuartal",
    "Budi menyiapkan laporan",
    "Siti menyiapkan materi presentasi",
]
decisions = ["Sepakat melanjutkan proyek X"]
action_items = [{"owner": "Budi", "task": "Menyiapkan laporan"}, {"owner": "Siti", "task": "Menyiapkan materi presentasi"}]
topics = ["Target", "Laporan", "Proyek X"]

cfg = SummarizationConfig()
s = AbstractiveSummarizer(cfg)
# Call and print results
ov, kws = s.generate_comprehensive_summary(full_text, key_points, decisions, action_items, topics)
print('OVERVIEW:', repr(ov))
print('KEYWORDS:', kws)

# Also call with pipeline disabled to show fallback
s._pipeline = None
ov2, kws2 = s.generate_comprehensive_summary(full_text, key_points, decisions, action_items, topics)
print('FALLBACK OVERVIEW:', repr(ov2))
print('FALLBACK KEYWORDS:', kws2)
