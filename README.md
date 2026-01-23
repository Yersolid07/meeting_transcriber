# 🎙️ Meeting Transcriber - Sistem Notulensi Rapat Otomatis

Sistem end-to-end untuk mengubah rekaman audio rapat menjadi dokumen notulensi terstruktur menggunakan **SpeechBrain** (ASR + Diarization) dan **BERT** (Summarization).

## 📋 Fitur

- ✅ **Speaker Diarization** - Identifikasi "siapa berbicara kapan"
- ✅ **Speech-to-Text** - Transkripsi audio ke teks (Bahasa Indonesia)
- ✅ **Extractive Summarization** - Ringkasan poin-poin penting
- ✅ **Document Generation** - Export ke format Word (.docx)
- ✅ **Evaluation Metrics** - WER dan DER untuk validasi

## 🚀 Quick Start

### 1. Instalasi

```bash
# Clone repository
git clone https://github.com/Yersolid07/meeting_transcriber.git
cd meeting-transcriber

# Install dependencies
pip install -r requirements.txt

```

## ⚠️ Windows / SpeechBrain Troubleshooting

SpeechBrain's model extraction may attempt to create symlinks on Windows which can fail with a permission error (WinError 1314). To avoid this in CI or local shells, set:

```powershell
# Persistently (PowerShell):
setx HF_HUB_DISABLE_SYMLINKS 1

# Temporarily for current session:
$env:HF_HUB_DISABLE_SYMLINKS = '1'
```

- Ensure `torchaudio` and `speechbrain` are installed and compatible with your PyTorch build.
- The pipeline includes robust fallback strategies when SpeechBrain model loading encounters symlink or permission errors — it will attempt snapshot download, copy strategies, or final deterministic MFCC fallback when `allow_fallback: true` in `config.yaml`.

Example (config.yaml):

```yaml
diarization:
  allow_fallback: true
  embedding:
    model_id: "speechbrain/spkrec-ecapa-voxceleb"
```

If you still see failures, consider running Python as Administrator or enabling Developer Mode on Windows.

## ⚡ Performance & CLI flags

Kami menyederhanakan pengaturan dengan menambahkan konsep **preset**. Jika Anda tidak ingin mengonfigurasi banyak flag, gunakan `--preset` untuk memilih konfigurasi yang sudah dipertimbangkan antara akurasi dan kecepatan:

- `deployment` (default): Rekomendasi produksi — **WhisperX `large-v3-turbo`** dengan `int8` compute pada CPU, single-pass full-audio mapping, dan parallel workers yang disesuaikan. Ini memberikan kombinasi terbaik dari akurasi dan waktu pemrosesan yang dapat diterima di lingkungan produksi.
- `balanced`: Trade-off seimbang (sedikit lebih konservatif dari `deployment`).
- `fast`: Mode sangat cepat (mengorbankan akurasi), cocok untuk preview atau low-cost runs.
- `accurate`: Mode prioritas akurasi (gunakan model besar dan presisi tinggi, lebih lambat).

Flag lama masih tersedia untuk override (`--quick-asr`, `--parallel-workers`, dan lain-lain), tetapi `--preset deployment` adalah rekomendasi default untuk deploy ke Streamlit.

Contoh singkat (sederhana):

```bash
# Gunakan preset deployment (default)
python main.py --audio rapat.wav

# Jika mau eksplisit:
python main.py --audio rapat.wav --preset deployment
```

## ⚙️ Benchmark singkat (cara cepat menjalankan benchmark lokal)

Kami menambahkan skrip benchmarking ringan `scripts/benchmark_asr.py` yang menjalankan dua pengukuran cepat:

1. Synthetic per-segment benchmark (tersedia secara default, cepat, tidak perlu model) — menunjukkan peningkatan wall-clock saat `parallel_workers` > 1.
2. Opsi full run dengan audio contoh jika Anda ingin membandingkan `--quick-asr` vs default (perhatian: dapat mengunduh model besar jika belum tersedia).

Contoh menjalankan benchmark:

```bash
# Synthetic benchmark (cepat)
python scripts/benchmark_asr.py --synthetic --json-output results/benchmark_synthetic.json

# Jika Anda punya sample short.wav dan model sudah di-cache, jalankan perbandingan real
python scripts/benchmark_asr.py --sample-audio data/audio/kondisi_multispeaker/rapatsingkat.mp3 --runs 2
```

### CI integration

A GitHub Actions job `CI - Tests + Benchmark` will run the **synthetic** benchmark on every pull request and upload the JSON result and log as artifacts (`benchmark-synthetic`). This gives quick feedback on parallel speedups for changes that might affect ASR or diarization performance.

## Streamlit Deployment

We included a minimal Streamlit app `streamlit_app.py` for quick demo and deployment. Basic usage:

- Run locally:

```bash
# Install streamlit if not present
pip install streamlit

# Run local app
streamlit run streamlit_app.py
```

- Docker (recommended for cloud deploy):

```bash
# Build image
docker build -t meeting-transcriber:latest .

# Run container
docker run -p 8501:8501 meeting-transcriber:latest
```

Notes:
- The app supports `--preset` selections from the sidebar; `deployment` (WhisperX large-v3-turbo int8) is the default pipeline recommendation.
- For production, mount a persistent `models/` volume and increase memory/cpu allocation for faster runs.

### Deploy to Streamlit Community Cloud (quick demo)

If you want a public demo or a quick internal preview, Streamlit Community Cloud is the fastest path. Keep in mind it offers no GPU and has strict runtime limits, so use the `fast` preset (or the sidebar `fast`) there.

1. Push your repository to GitHub.
2. Go to https://streamlit.io/cloud and connect your GitHub account.
3. Create a new app, pick your repo and the `streamlit_app.py` file.
4. Set an environment variable `STREAMLIT_DEPLOY_TARGET=community` in the Streamlit app settings so the app will default to `fast` + quick-ASR for responsiveness.
5. If you want to change model/defaults later, edit the code or set environment variables and push; Streamlit Cloud will redeploy on push.

Tips:
- Use lightweight models (e.g., `openai/whisper-small`) if you expect many users or want fast previews.
- Do not enable `deployment`/WhisperX `large-v3-turbo` unless you have a GPU-backed host (Cloud Run with GPU or VM with GPU).

### Iterating models and redeploy

- To change model, update `PipelineConfig` defaults or change `preset` in `streamlit_app.py`, commit and push.
- For experiments/finetuning, you can store models in Hugging Face or MLflow and update the code to fetch them at startup.
- When you are ready for production-grade performance (GPU/scale), follow the production plan in the section above and migrate workers to a GPU-enabled deployment.

