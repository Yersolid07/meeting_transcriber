# Review Sistem Pipeline + Panduan Integrasi Model Fine-Tuned

Dokumen ini disusun untuk membantu Anda (dan LLM lain) memahami arsitektur sistem serta titik integrasi model hasil fine-tuning (ASR/summarization/diarization) secara **praktis dan implementatif**.

---

## 1) Ringkasan Arsitektur Pipeline

Alur utama saat `main.py` dieksekusi:

1. CLI membaca argumen (audio path, backend ASR, preset, opsi evaluasi, dsb)
2. `MeetingTranscriberPipeline` mengorkestrasi semua komponen
3. `AudioProcessor` memuat + normalisasi audio
4. `SpeakerDiarizer` melakukan diarization (VAD → embedding → clustering)
5. `ASRTranscriber` mentranskripsi tiap segmen speaker
6. `BERTSummarizer` membentuk ringkasan terstruktur
7. `DocumentGenerator` menyimpan hasil ke `.docx`
8. `Evaluator` (opsional) menghitung WER/DER/CER

Referensi kode:
- `main.py` (CLI entrypoint): `parse_args()`, mode single/batch, evaluasi
- `src/pipeline.py`: `MeetingTranscriberPipeline`, `PipelineConfig`, `PipelineResult`
- `src/audio_processor.py`, `src/diarization.py`, `src/transcriber.py`, `src/summarizer.py`, `src/document_generator.py`, `src/evaluator.py`

---

## 2) Titik Integrasi Model (Paling Penting)

### A. Integrasi model ASR fine-tuned

Lokasi utama: `src/transcriber.py` (`ASRConfig`, `ASRTranscriber._load_model`).

Strategi integrasi:
1. Simpan model fine-tuned di Hugging Face repo atau local path.
2. Set `model_id` ke model hasil fine-tune.
3. Pilih backend sesuai tipe model:
   - Whisper seq2seq: `backend="whisper"` atau `backend="whisperx"`
   - Wav2Vec2 CTC: `backend="transformers"`
   - SpeechBrain: `backend="speechbrain"`

Contoh CLI:

```bash
python main.py \
  --audio data/sample_audio/utt1.wav \
  --preset balanced \
  --quick-asr \
  --output ./data/output
```

Contoh override dari Python (disarankan untuk integrasi programatik):

```python
from src.pipeline import MeetingTranscriberPipeline, PipelineConfig

cfg = PipelineConfig(
    asr_backend="transformers",                     # atau "whisper"/"whisperx"/"speechbrain"
    asr_model_id="your-org/wav2vec2-id-finetuned", # boleh local path
    asr_language="id",
)

pipeline = MeetingTranscriberPipeline(cfg)
result = pipeline.process("data/sample_audio/utt1.wav", title="Uji Integrasi")
print(result.document_path)
```

---

### B. Integrasi model diarization/embedding fine-tuned

Lokasi utama: `src/diarization.py` (`DiarizationConfig.embedding_model_id`, `_load_embedding_model`).

Langkah:
1. Ubah `embedding_model_id` ke model embedding speaker Anda.
2. Pastikan format output embedding kompatibel (vector per-segmen).
3. Validasi threshold clustering (karena distribusi embedding baru bisa berubah).

Contoh konfigurasi Python:

```python
from src.pipeline import PipelineConfig

cfg = PipelineConfig()
# akses lewat pipeline -> DiarizationConfig saat inisialisasi komponen
# (jika perlu, extend PipelineConfig agar menerima override embedding_model_id)
```

> Rekomendasi engineering: tambahkan field `diarization_embedding_model_id` di `PipelineConfig`
> lalu pass ke `DiarizationConfig(embedding_model_id=...)` saat pipeline membangun diarizer.

---

### C. Integrasi model summarization fine-tuned

Lokasi utama: `src/summarizer.py` (`SummarizationConfig`).

Dua pola:
1. **Extractive**: ganti `sentence_model_id` (mis. sentence-transformer fine-tuned domain rapat)
2. **Abstractive**: set `method="abstractive"` dan `abstractive_model_id` ke model fine-tuned Anda

Contoh:

```python
from src.pipeline import MeetingTranscriberPipeline, PipelineConfig

cfg = PipelineConfig(
    summarization_method="abstractive",
    abstractive_model_id="your-org/mt5-rapat-finetuned",
    sentence_model_id="your-org/sentence-embed-rapat"
)

pipeline = MeetingTranscriberPipeline(cfg)
```

---

## 3) Template Prompt untuk LLM Lain (siap pakai)

Gunakan prompt berikut jika Anda ingin LLM lain membantu integrasi:

```text
Saya punya repo meeting_transcriber dengan modul utama:
- src/pipeline.py (MeetingTranscriberPipeline, PipelineConfig)
- src/transcriber.py (ASRConfig, ASRTranscriber)
- src/diarization.py (DiarizationConfig, SpeakerDiarizer)
- src/summarizer.py (SummarizationConfig)

Tugas Anda:
1) Integrasikan model fine-tuned saya:
   - ASR: <isi model_id/backend>
   - Diarization embedding: <isi model_id>
   - Summarization: <isi model_id + method>
2) Ubah kode seminimal mungkin, backward-compatible.
3) Tambahkan validasi config + fallback jika model gagal load.
4) Berikan patch per file + alasan desain.
5) Tambahkan checklist uji: smoke test CLI, evaluasi WER/DER, dan sanity output DOCX.

Constraint:
- Jangan ubah kontrak output PipelineResult.
- Pertahankan kompatibilitas CLI main.py.
- Sertakan contoh command run.
```

---

## 4) Checklist Integrasi (Praktis)

1. **Pastikan dependency model sesuai**
   - Whisper/Transformers/SpeechBrain terpasang
2. **Set model ID/path**
   - via `PipelineConfig` atau argument CLI
3. **Smoke test cepat**
   - jalankan 1 file audio pendek
4. **Evaluasi kualitas**
   - WER (dengan reference transcript), DER (dengan RTTM)
5. **Validasi output dokumen**
   - periksa section summary/decisions/action items
6. **Fallback plan**
   - jika load gagal, kembali ke model default yang stabil

---

## 5) Gap & Rekomendasi Refactor (agar mudah diintegrasikan LLM)

Agar LLM lain lebih mudah melakukan patch otomatis, sangat disarankan:

1. **Expose semua model-id sebagai field `PipelineConfig`**
   - termasuk `diarization_embedding_model_id`
2. **Sediakan fungsi factory tunggal**
   - `build_diarizer_config()`, `build_asr_config()`, `build_summarizer_config()`
3. **Tambahkan class adapter per backend**
   - `AsrBackendAdapter`, `SummarizerAdapter`
4. **Standardisasi error handling model load**
   - return message yang eksplisit dan actionable
5. **Tambahkan 3 test integrasi ringan**
   - ASR custom model id, summarizer abstractive custom id, diarizer custom embedding id

---

## 6) Snippet Implementasi Refactor (Contoh yang Bisa Dipakai LLM)

Contoh ide patch kecil pada pipeline (pseudo-kode, bukan drop-in langsung):

```python
@dataclass
class PipelineConfig:
    asr_model_id: str = "openai/whisper-small"
    asr_backend: str = "whisper"
    diarization_embedding_model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    summarization_method: str = "extractive"
    sentence_model_id: str | None = None
    abstractive_model_id: str | None = None
```

```python
diar_cfg = DiarizationConfig(
    embedding_model_id=self.config.diarization_embedding_model_id,
    # parameter lain tetap
)
```

```python
sum_cfg = SummarizationConfig(
    method=self.config.summarization_method,
    sentence_model_id=self.config.sentence_model_id or DEFAULT_SENTENCE_MODEL,
    abstractive_model_id=self.config.abstractive_model_id or DEFAULT_ABSTRACTIVE_MODEL,
)
```

---

## 7) Daftar Diagram Pendukung

Diagram yang sudah tersedia untuk Anda masukkan ke dokumen skripsi:
- `docs/diagrams/system_context.mmd`
- `docs/diagrams/pipeline_flow.mmd`
- `docs/diagrams/component_diagram.mmd`
- `docs/diagrams/class_diagram.mmd`
- `docs/diagrams/sequence_transcription.mmd`
- `docs/diagrams/activity_evaluation.mmd`
- `docs/diagrams/deployment.mmd`
- `docs/diagrams/data_schema.mmd`
- `docs/diagrams/benchmark_chart.mmd`

---

## 8) Ringkasan untuk Pengguna Non-Teknis

Sistem ini sudah modular dan cukup siap untuk menerima model fine-tuned baru, terutama pada tiga titik: **ASR**, **speaker embedding diarization**, dan **summarization**. Untuk mempercepat integrasi oleh LLM lain, fokuskan instruksi pada file `src/pipeline.py`, `src/transcriber.py`, `src/diarization.py`, dan `src/summarizer.py`.
