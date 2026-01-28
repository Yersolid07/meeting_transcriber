# 🎙️ Meeting Transcriber - Sistem Notulensi Rapat Otomatis

Sistem end-to-end untuk mengubah rekaman audio rapat menjadi dokumen notulensi terstruktur menggunakan **Whisper** / **SpeechBrain** (ASR + Diarization) dan **BERT** (Summarization).

**Versi:** 1.0.0  
**Autor:** Yermia Turangan  
**Bahasa Mendukung:** 🇮🇩 Bahasa Indonesia

## � Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Quick Start](#-quick-start)
- [Panduan Lengkap](#-panduan-lengkap)
  - [CLI Commands](#command-line-interface-cli)
  - [Web Interface](#streamlit-web-interface)
  - [Makefile](#makefile-commands)
- [Konfigurasi](#-konfigurasi)
- [Output Format](#-output-format)
- [Evaluation](#-evaluation)
- [Troubleshooting](#-troubleshooting)
- [Performance Benchmark](#-performance-benchmark)
- [Directory Structure](#-directory-structure)
- [Supported Formats](#-supported-audio-formats)
- [Use Cases](#-use-cases)
- [Quality Assurance](#-quality-assurance)
- [FAQ](#-troubleshooting--faq)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support--contact)

## �📋 Fitur Utama

- ✅ **Speaker Diarization** - Identifikasi "siapa berbicara kapan" dengan clustering otomatis
- ✅ **Speech-to-Text (ASR)** - Transkripsi audio ke teks dengan Whisper/SpeechBrain (Bahasa Indonesia)
- ✅ **Extractive Summarization** - Ringkasan poin-poin penting menggunakan BERT
- ✅ **Decision & Action Items** - Ekstraksi keputusan dan tindak lanjut otomatis
- ✅ **Document Generation** - Export ke format Word (.docx) yang siap pakai
- ✅ **Evaluation Metrics** - WER (Word Error Rate) dan DER (Diarization Error Rate)
- ✅ **Interactive Streamlit UI** - Web interface untuk preview dan mapping speaker
- ✅ **Batch Processing** - Proses multiple file sekaligus
- ✅ **Performance Presets** - Pilihan antara kecepatan vs akurasi

## 🚀 Quick Start

### 1. Instalasi

```bash
# Clone repository (jika belum)
git clone https://github.com/Yersolid07/meeting_transcriber.git
cd meeting_transcriber

# Install dependencies
pip install -r requirements.txt

# (Windows only) Disable symlink requirement untuk SpeechBrain
$env:HF_HUB_DISABLE_SYMLINKS = '1'
# Atau set permanent: setx HF_HUB_DISABLE_SYMLINKS 1
```

### 2. Penggunaan Dasar

```bash
# Transkripsi file audio single
python main.py --audio rapat.wav

# Dengan informasi rapat
python main.py --audio rapat.wav --title "Rapat Sprint Q1" --speakers 3 --location "Zoom"

# Output ke folder spesifik
python main.py --audio rapat.wav --output ./hasil/
```

### 3. Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
# Buka http://localhost:8501
```

## 📖 Panduan Lengkap

### Command Line Interface (CLI)

#### Transkripsi Dasar

```bash
# Single file
python main.py --audio rapat.wav

# Dengan output directory spesifik
python main.py --audio rapat.wav --output ./hasil/

# Dengan nama file custom
python main.py --audio rapat.wav --filename my_transcript
```

#### Dengan Metadata Rapat

```bash
# Lengkap dengan informasi
python main.py --audio rapat.wav \
  --title "Rapat Sprint Q1" \
  --date "2026-01-27" \
  --location "Zoom" \
  --speakers 4
```

#### Batch Processing

```bash
# Proses semua file audio di folder
python main.py --batch ./audio_folder/ --output ./hasil/

# Dengan metadata sama untuk semua
python main.py --batch ./audio_folder/ --title "Rapat Mingguan"
```

#### Mapping Speaker Otomatis

```bash
# Gunakan JSON file untuk mapping speaker ke nama
python main.py --audio rapat.wav --speaker-map speaker_names.json

# Format speaker_names.json:
# {
#   "SPEAKER_00": "Budi",
#   "SPEAKER_01": "Ani",
#   "SPEAKER_02": "Citra"
# }
```

#### Evaluasi & Benchmark

```bash
# Dengan reference transcript untuk WER
python main.py --audio rapat.wav \
  --evaluate \
  --reference transkrip_manual.txt

# Dengan RTTM reference untuk Diarization Error Rate (DER)
python main.py --audio rapat.wav \
  --evaluate \
  --reference-rttm reference.rttm

# Bandingkan multiple metode diarization
python main.py --audio rapat.wav \
  --diarization-compare
```

#### Performance & Tuning

**Presets (untuk kemudahan):**

```bash
# deployment: Whisper large-v3-turbo + parallel, production-ready
python main.py --audio rapat.wav --preset deployment

# balanced: Trade-off antara akurasi & kecepatan
python main.py --audio rapat.wav --preset balanced

# fast: Whisper small, sangat cepat (untuk preview)
python main.py --audio rapat.wav --preset fast

# accurate: Model besar, prioritas akurasi
python main.py --audio rapat.wav --preset accurate
```

**Fine-tuning diarization:**

```bash
# Jalankan hyperparameter tuning sebelum clustering
python main.py --audio rapat.wav --tune-diarization

# Paksa jumlah speaker tertentu (merge clusters)
python main.py --audio rapat.wav --target-speakers 3

# Kontrol jumlah worker paralel
python main.py --audio rapat.wav --parallel-workers 4
```

**Model-specific overrides:**

```bash
# Paksa Whisper small (lebih cepat)
python main.py --audio rapat.wav --prefer-whisper-small

# Quick ASR mode
python main.py --audio rapat.wav --quick-asr
```

### Streamlit Web Interface

```bash
# Start web UI
streamlit run streamlit_app.py

# Open browser di http://localhost:8501
```

**Fitur:**

- Upload file audio atau pilih dari sample
- Preview diarization result
- Manual speaker mapping (interactive)
- Real-time transcript & summary
- Download .docx report

### Makefile Commands

```bash
# Setup & Install
make install              # Install dependencies
make install-dev         # Install dengan development tools
make setup-dirs          # Create directory structure

# Running
make run AUDIO=path/to/audio.wav        # Transcribe single file
make run-batch DIR=path/to/folder       # Batch processing

# Evaluation & Testing
make evaluate            # Run full evaluation
make create-gt           # Create ground truth templates
make test                # Run unit tests
make test-cov            # Run tests dengan coverage report
make benchmark           # Run performance benchmark

# Utilities
make clean               # Clean generated files
make lint                # Run linter (flake8)
make format              # Format code (black, isort)

# Help
make help                # Show all available commands
```

## ⚙️ Konfigurasi

### config.yaml - Pengaturan Utama

#### Audio Processing

```yaml
audio:
  sample_rate: 16000 # Resample ke 16kHz
  mono: true # Konversi ke mono
  normalize: true # Normalize amplitude
  trim_silence: false # Auto-trim silence
  max_duration_minutes: 60 # Maksimal durasi audio
```

#### Speaker Diarization

```yaml
diarization:
  # Voice Activity Detection
  vad:
    threshold: 0.5
    min_speech_duration: 0.3 # Durasi minimum speech (detik)
    min_silence_duration: 0.3 # Durasi minimum silence

  # Clustering
  clustering:
    method: 'spectral' # agglomerative | spectral | kmeans
    threshold: 0.7 # Speaker separation threshold
    min_cluster_size: 2

  # Speaker Embedding
  embedding:
    model_id: 'speechbrain/spkrec-ecapa-voxceleb'
    embedding_dim: 192
```

#### ASR (Speech Recognition)

```yaml
asr:
  model_id: 'whisper/whisper-base'
  chunk_length_s: 30 # Chunk size (detik)
  stride_length_s: 5 # Overlap (detik)
  batch_size: 4
  return_timestamps: false

  # Post-processing
  text_postprocessing:
    capitalize_sentences: true
    normalize_whitespace: true
    add_punctuation: false
```

#### Summarization

```yaml
summarization:
  model_id: 'indobenchmark/indobert-base-p1'

  extractive:
    num_sentences: 5 # Jumlah sentence untuk summary
    min_sentence_length: 10
    max_sentence_length: 200
    position_weight: 0.1 # Boost awal/akhir
    similarity_threshold: 0.3

  # Keywords untuk ekstraksi
  keywords:
    decisions:
      - 'diputuskan'
      - 'disepakati'
      - 'kesimpulan'
    action_items:
      - 'akan'
      - 'harus'
```

## 📊 Output Format

### Struktur File Output

```
data/output/
├── notulensi_rapat_20260127_120000/
│   ├── transcript.txt           # Teks lengkap dengan speaker label
│   ├── summary.txt              # Ringkasan ekstraksi
│   ├── decisions.txt            # Keputusan yang diambil
│   ├── action_items.txt         # Tindak lanjut
│   ├── report.docx              # Dokumen Word final
│   ├── diarization.rttm         # RTTM format (untuk evaluasi)
│   └── metadata.json            # Metadata & config
```

### Format Transcript

```
SPEAKER_00 [00:00 - 00:15]: Assalamu'alaikum, hari ini kita akan membahas...
SPEAKER_01 [00:16 - 00:45]: Terima kasih. Saya ingin menambahkan...
SPEAKER_00 [00:46 - 01:20]: Baik, mari kita lanjut ke agenda berikutnya...
```

### Format .docx Report

Dokumen Word yang dihasilkan berisi:

- Header dengan judul, tanggal, lokasi, peserta
- Transkripsi lengkap per speaker
- Ringkasan poin penting
- Keputusan yang diambil
- Tindak lanjut (Action Items)
- Metadata & waktu proses

## 🔍 Evaluation

### Word Error Rate (WER)

```bash
# Bandingkan hasil ASR dengan transkrip manual
python main.py --audio rapat.wav \
  --evaluate \
  --reference reference_transcript.txt

# Format reference_transcript.txt:
# Plain text tanpa speaker labels
```

**Output Evaluasi:**

```
WER (Word Error Rate): 15.3%
CER (Character Error Rate): 8.2%
Matched: 850/1000 kata
```

### Diarization Error Rate (DER)

```bash
# Gunakan RTTM reference file
python main.py --audio rapat.wav \
  --evaluate \
  --reference-rttm reference.rttm

# Format RTTM:
# SPEAKER rapat.wav 1 0.5 3.2 <NA> <NA> SPEAKER_00 <NA> <NA>
# SPEAKER rapat.wav 1 3.9 2.1 <NA> <NA> SPEAKER_01 <NA> <NA>
```

**Output:**

```
DER (Diarization Error Rate): 12.5%
  - False Alarm: 3.2%
  - Missed Detection: 6.1%
  - Speaker Error: 3.2%
```

## 🐛 Troubleshooting

### Windows - SpeechBrain Symlink Error

**Error:** `WinError 1314` atau permission denied saat loading model

**Solusi:**

```powershell
# Temporary (sesi saat ini)
$env:HF_HUB_DISABLE_SYMLINKS = '1'

# Permanent (semua sesi)
setx HF_HUB_DISABLE_SYMLINKS 1

# Verify
$env:HF_HUB_DISABLE_SYMLINKS
```

### CUDA / GPU Issues

Jika menggunakan GPU:

```bash
# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU (jika ada error)
python main.py --audio rapat.wav --device cpu
```

### Memory Issues

Untuk file audio besar atau sistem dengan RAM terbatas:

```bash
# Reduce batch size
python main.py --audio rapat.wav --asr-batch-size 1

# Use fast preset
python main.py --audio rapat.wav --preset fast

# Enable disk cache untuk embedding
# (default sudah aktif, disable dengan --no-embedding-cache jika ada error)
```

### Model Download Errors

```bash
# Set HuggingFace cache directory
$env:HF_HOME = "C:\huggingface_cache"

# Atau download model secara manual sebelumnya
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base')"
```

## 📦 Dependency Requirements

**Core Libraries:**

- `torch >= 2.0.0` - Deep learning framework
- `torchaudio >= 2.0.0` - Audio processing
- `transformers >= 4.30.0` - BERT models
- `speechbrain >= 0.5.15` - Speaker embedding & VAD

**Audio Processing:**

- `librosa` - Audio analysis
- `soundfile` - WAV I/O
- `pydub` - Format conversion

**NLP & ML:**

- `sentence-transformers` - Sentence embeddings
- `scikit-learn` - Clustering algorithms
- `jiwer` - WER calculation

**Output & UI:**

- `python-docx` - Word document generation
- `streamlit` - Web interface
- `pandas` - Data handling

## 🏃 Performance Benchmark

Waktu pemrosesan (contoh dengan audio 10 menit):

| Preset     | ASR Model        | Waktu (min) | Akurasi | Memory |
| ---------- | ---------------- | ----------- | ------- | ------ |
| fast       | whisper-small    | 3-4         | ~80%    | 2GB    |
| balanced   | whisper-small    | 4-5         | ~85%    | 2.5GB  |
| deployment | whisper-large-v3 | 8-12        | ~92%    | 6GB    |
| accurate   | whisper-large-v3 | 12-15       | ~95%    | 8GB    |

_Perkiraan waktu pada CPU modern (Intel i7/Ryzen 7)_

## 📝 Directory Structure

```
meeting_transcriber/
├── src/                       # Source code
│   ├── pipeline.py           # Main orchestrator
│   ├── transcriber.py        # ASR pipeline
│   ├── diarization.py        # Speaker diarization
│   ├── summarizer.py         # Text summarization
│   ├── document_generator.py # DOCX export
│   ├── evaluator.py          # Evaluation metrics
│   └── utils.py              # Utility functions
├── data/
│   ├── audio/                # Audio samples
│   ├── output/               # Generated transcripts
│   └── speaker_maps/         # Speaker mapping files
├── tests/                    # Unit tests
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── main.py                   # CLI entry point
├── streamlit_app.py          # Web UI
└── README.md                 # This file
```

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

1. Push your repository ke GitHub.
2. Go to https://streamlit.io/cloud dan connect GitHub account.
3. Create a new app, pick your repo dan `streamlit_app.py` file.
4. Set environment variable `STREAMLIT_DEPLOY_TARGET=community` di Streamlit app settings.
5. Jika mau change model/defaults, edit code dan push; Streamlit Cloud akan redeploy otomatis.

Tips:

- Gunakan lightweight models (e.g., `openai/whisper-small`) untuk banyak users.
- Jangan enable `deployment`/WhisperX `large-v3-turbo` tanpa GPU.

## 📚 Contoh Penggunaan

### Skenario 1: Rapat Pagi (5-10 menit)

```bash
# Cepat & casual preview
python main.py --audio rapat_pagi.wav --preset fast --title "Rapat Pagi Tim"
```

### Skenario 2: Rapat Formal (1 jam) dengan Evaluasi

```bash
# Akurat & lengkap, dengan evaluasi
python main.py --audio rapat_formal.wav \
  --preset deployment \
  --title "Rapat Board Q1" \
  --speakers 6 \
  --date "2026-01-27" \
  --location "Kantor Pusat" \
  --evaluate \
  --reference reference_transcript.txt
```

### Skenario 3: Batch Processing Multiple Rapat

```bash
# Proses folder berisi 10+ file audio
python main.py --batch ./audio_meetings/ \
  --output ./results/ \
  --preset balanced \
  --parallel-workers 4
```

### Skenario 4: Production Deployment (Streamlit)

```bash
# Deploy web UI untuk tim
streamlit run streamlit_app.py --server.port 8501
# Akses http://your-server:8501
```

## 🎯 Use Cases

| Use Case         | Command                                              | Preset     | Notes               |
| ---------------- | ---------------------------------------------------- | ---------- | ------------------- |
| Quick Preview    | `python main.py --audio x.wav`                       | fast       | Instant result      |
| Formal Minutes   | `python main.py --audio x.wav --evaluate`            | deployment | High accuracy       |
| Batch Processing | `python main.py --batch dir/`                        | balanced   | Parallel processing |
| Archive/History  | `python main.py --audio x.wav --date 2026-01-27`     | deployment | With metadata       |
| Research/Eval    | `python main.py --audio x.wav --diarization-compare` | accurate   | Compare methods     |
| Web Interface    | `streamlit run streamlit_app.py`                     | -          | Interactive UI      |

## ✅ Quality Assurance

### Testing

```bash
# Run unit tests
make test

# Run dengan coverage
make test-cov

# Run specific test file
pytest tests/test_pipeline.py -v
```

### Code Quality

```bash
# Lint code (flake8)
make lint

# Auto-format code
make format

# Check type hints
mypy src/ --ignore-missing-imports
```

## 📊 Supported Audio Formats

| Format | Extension | Support        | Notes                      |
| ------ | --------- | -------------- | -------------------------- |
| WAV    | .wav      | ✅ Recommended | Uncompressed, best quality |
| MP3    | .mp3      | ✅ Supported   | Lossy, widely used         |
| M4A    | .m4a      | ✅ Supported   | AAC codec, common          |
| OGG    | .ogg      | ✅ Supported   | Vorbis codec               |
| FLAC   | .flac     | ✅ Supported   | Lossless compression       |
| WebM   | .webm     | ✅ Supported   | VP9/Opus                   |

## 🌐 Supported Languages

Saat ini sistem dioptimalkan untuk:

- **🇮🇩 Bahasa Indonesia** (primary)
- **🇬🇧 English** (secondary, via Whisper multilingual)

Model Whisper mendukung 99+ bahasa. Untuk bahasa lain, ubah model di `config.yaml`:

```yaml
asr:
  model_id: 'openai/whisper-large-v3' # Multilingual
```

## 🔐 Privacy & Security

- ✅ Offline processing - tidak perlu koneksi internet untuk processing
- ✅ Local models - models di-cache di local machine
- ✅ No cloud uploads - semua data tetap di local
- ✅ Audio tidak disimpan - hanya transcript yang disimpan

**Security considerations:**

```bash
# Disable embedding cache (jika ada privacy concern)
python main.py --audio rapat.wav --no-embedding-cache

# Output ke folder encrypted (recommended)
python main.py --audio rapat.wav --output /path/to/encrypted/
```

## 📞 Troubleshooting & FAQ

### Q: Bagaimana jika hasil transkripsi tidak akurat?

**A:** Coba beberapa hal:

1. Naik ke preset `accurate` atau `deployment`
   ```bash
   python main.py --audio rapat.wav --preset accurate
   ```
2. Pastikan audio berkualitas baik (clear speech, minimal background noise)
3. Jika ada accent tertentu, pertimbangkan fine-tune model

### Q: Proses terlalu lambat, bagaimana mempercepat?

**A:** Gunakan `fast` preset atau options:

```bash
# Tercepat (trade-off akurasi)
python main.py --audio rapat.wav --preset fast

# Atau manual tuning
python main.py --audio rapat.wav --quick-asr --parallel-workers 4
```

### Q: Error "CUDA out of memory"

**A:** Kurangi batch size atau gunakan CPU:

```bash
# Force CPU
python main.py --audio rapat.wav --device cpu

# Atau reduce batch
python main.py --audio rapat.wav --asr-batch-size 1
```

### Q: Speaker diarization salah (mixing speakers)

**A:** Coba tune hyperparameter:

```bash
# Automatic tuning
python main.py --audio rapat.wav --tune-diarization

# Atau paksa jumlah speaker
python main.py --audio rapat.wav --target-speakers 3

# Atau adjust threshold di config.yaml
diarization:
  clustering:
    threshold: 0.8  # Increase untuk lebih strict
```

### Q: Bagaimana jika audio punya noise tinggi?

**A:** Preprocessing dengan librosa:

```python
import librosa
import soundfile as sf

# Load audio
y, sr = librosa.load('noisy.wav')

# Reduce noise (simple method)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_db = librosa.power_to_db(S, ref=np.max)
# ... apply noise reduction ...

# Save cleaned
sf.write('cleaned.wav', y, sr)
```

Atau gunakan tool khusus seperti Audacity atau FFmpeg:

```bash
# Use FFmpeg noise reduction
ffmpeg -i noisy.wav -af anlmdn=s=0.002:om=o -q:a 9 cleaned.wav
```

### Q: Bagaimana format speaker map JSON?

**A:** Buat file `speaker_map.json`:

```json
{
  "SPEAKER_00": "Budi Santoso",
  "SPEAKER_01": "Ani Wijaya",
  "SPEAKER_02": "Citra Dewi",
  "SPEAKER_03": "Deni Supriyadi"
}
```

Kemudian:

```bash
python main.py --audio rapat.wav --speaker-map speaker_map.json
```

### Q: Bagaimana me-deploy ke production?

**A:** Gunakan Docker:

```dockerfile
# Dockerfile sudah included
docker build -t meeting-transcriber:latest .
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  meeting-transcriber:latest
```

Atau Kubernetes:

```bash
# Create deployment
kubectl create deployment meeting-transcriber --image=meeting-transcriber:latest
kubectl expose deployment meeting-transcriber --type=LoadBalancer --port=8501
```

### Q: Model mana yang recommended?

**A:**

- **Untuk kecepatan:** `openai/whisper-small` (350M parameters)
- **Balanced:** `openai/whisper-base` (140M parameters)
- **Akurasi:** `openai/whisper-large-v3` (1.5B parameters)

Update di `config.yaml`:

```yaml
asr:
  model_id: 'openai/whisper-small' # Ganti model
```

## 📋 Checklist Setup

- [ ] Clone repository
- [ ] Install Python 3.9+
- [ ] `pip install -r requirements.txt`
- [ ] Set `HF_HUB_DISABLE_SYMLINKS=1` (Windows)
- [ ] Download sample audio atau siapkan file audio
- [ ] Test dengan: `python main.py --audio test.wav`
- [ ] Verifikasi output di `data/output/`
- [ ] (Optional) Setup Streamlit: `streamlit run streamlit_app.py`
- [ ] (Optional) Create speaker map JSON jika perlu
- [ ] Ready to use!

## 🤝 Contributing

Kontribusi welcome! Berikut cara:

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/meeting_transcriber.git
cd meeting_transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate di Windows

# Install dev dependencies
make install-dev

# Run tests before submitting PR
make test
make lint
```

## 📄 License

MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📧 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/Yersolid07/meeting_transcriber/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Yersolid07/meeting_transcriber/discussions)
- **Email:** support@example.com

## 🎓 Citation

Jika menggunakan sistem ini dalam research, silakan cite:

```bibtex
@software{meeting_transcriber2026,
  author = {Turangan, Yermia},
  title = {Meeting Transcriber: Automated Meeting Minutes System},
  year = {2026},
  url = {https://github.com/Yersolid07/meeting_transcriber}
}
```

## 📚 Related Works & Resources

- **Whisper ASR:** https://github.com/openai/whisper
- **SpeechBrain:** https://github.com/speechbrain/speechbrain
- **Transformers:** https://huggingface.co/transformers/
- **Streamlit:** https://streamlit.io/

## 🎉 Acknowledgments

Terima kasih kepada:

- OpenAI untuk Whisper
- SpeechBrain team
- Hugging Face community
- Contributors dan testers

---

**Terakhir diupdate:** 27 Januari 2026  
**Status:** ✅ Production Ready
