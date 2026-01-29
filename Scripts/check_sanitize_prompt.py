import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.summarizer import AbstractiveSummarizer, SummarizationConfig
s = AbstractiveSummarizer(SummarizationConfig())
kp=["Pembahasan target kuartal","Budi menyiapkan laporan","Siti menyiapkan materi presentasi"]
print('Sanitized key points:', [s._sanitize_for_prompt(k) for k in kp])
print('Sanitized decisions:', [s._sanitize_for_prompt(d) for d in ["Sepakat melanjutkan proyek X"]])
print('Sanitized actions:', [s._sanitize_for_prompt(a.get('task')) for a in [{"task":"Menyiapkan laporan"},{"task":"Menyiapkan materi presentasi"}]])
