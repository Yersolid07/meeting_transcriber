import sys
from pathlib import Path
# Ensure project root is on path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.summarizer import AbstractiveSummarizer, SummarizationConfig
from src.transcriber import TranscriptSegment
import re

raw = '''SPEAKER_00 [00:02–00:03]: Oke, before Lee.
SPEAKER_02 [00:04–00:06]: Kita mulai rapatnya ya.
SPEAKER_02 [00:07–00:10]: Topik hari ini adalah evaluasi pola kerja tim.
SPEAKER_00 [00:10–00:12]: Dan dampaknya ke produktivitas.
SPEAKER_00 [00:13–00:17]: Beberapa minggu terakhir aku lihat ada penurunan di beberapa target.
SPEAKER_03 [00:18–00:22]: Iya, aku juga sempat perhatikan. Beberapa tugas memang selesai.
SPEAKER_03 [00:22–00:25]: Tapi waktunya agak molor dari yang direncanakan.
SPEAKER_45 [00:27–00:27]: Nah ini.
SPEAKER_02 [00:27–00:29]: Itu yang mau kita bahas.
SPEAKER_02 [00:30–00:30]: Menurut Mol,
SPEAKER_00 [00:31–00:32]: Penyebab utamanya apa.
SPEAKER_00 [00:33–00:36]: Apakah dari beban kerja jadwal.
SPEAKER_00 [00:36–00:37]: Atau cara koordinasi.
SPEAKER_03 [00:39–00:41]: Kalau dari pengamatanku ya, kombinasi sih ya.
SPEAKER_03 [00:42–00:43]: Beban kerja sebenarnya mesti masuk akal.
SPEAKER_03 [00:44–00:46]: Tapi cara pembagian waktunya kurang rapi.
SPEAKER_00 [00:47–00:49]: Maksudnya kurang rapi gimana?
SPEAKER_03 [00:50–00:55]: Iya, jadi contohnya ada beberapa orang yang pegang banyak tugas kecil sekaligus.
SPEAKER_03 [00:55–01:01]: Secara total mungkin ringan, tapi karena banyaknya pindah fokus jadinya lama selesai.
SPEAKER_45 [01:02–01:02]: Oh, dia.
SPEAKER_02 [01:03–01:05]: Jadi masalahnya bukan di jumlah tugas.
SPEAKER_02 [01:06–01:08]: Tapi di jenis dan pola pengerjaannya.
SPEAKER_03 [01:09–01:10]: Iya, betul.
SPEAKER_03 [01:11–01:16]: Terus ada juga yang ngerjain tugas tanpa prioritas jelas Jadi yang penting.
SPEAKER_03 [01:16–01:18]: Jadi tugas-tugas yang penting malah ketunda.
SPEAKER_04 [01:20–01:22]: Oke, itu masuk akal.
SPEAKER_00 [01:22–01:25]: Selama ini kita memang belum punya aturan. [OVERLAP]
SPEAKER_02 [01:24–01:27]: Kita memang belum punya aturan prioritas yang tertulis. [OVERLAP]
SPEAKER_02 [01:28–01:30]: Biasanya cuma disampaikan secara lesan.
SPEAKER_11 [01:32–01:32]: Nah.
SPEAKER_22 [01:32–01:34]: Itu mungkin yang bikin beda persepsi.
SPEAKER_03 [01:34–01:37]: Persepsi ya, ada yang anggap semua tugas sama pentingnya.
SPEAKER_04 [01:38–01:41]: Kalau begitu, salah satu solusinya... [OVERLAP]
SPEAKER_00 [01:40–01:44]: Salah satu solusinya kita perlu sistem prioritas yang jelas ya. [OVERLAP]
SPEAKER_02 [01:44–01:46]: Misalnya tugas harian mingguan. [OVERLAP]
SPEAKER_00 [01:45–01:47]: Tugas harian, mingguan dan yang sifatnya mendesak. [OVERLAP]
SPEAKER_02 [01:47–01:48]: Dan yang sifatnya mendesak. [OVERLAP]
SPEAKER_14 [01:49–01:50]: Setujuh.
SPEAKER_03 [01:51–01:54]: Tapi kita juga perlu hati-hati biar sistemnya nggak terlalu ribet.
SPEAKER_02 [01:55–01:59]: Iya, jangan sampai malah nambah beban administrasi.
SPEAKER_03 [02:00–02:02]: Terus aku juga mau bahas soal jam kerja.
SPEAKER_03 [02:03–02:07]: Ada beberapa orang yang kelihatan produktifnya naik kalau kerjanya agak teksif.
SPEAKER_00 [02:09–02:10]: Ini nyambung sama pembahasan kita. [OVERLAP]
SPEAKER_02 [02:10–02:12]: Nyambung sama pembahasan kita sebelumnya soal zaman. [OVERLAP]
SPEAKER_34 [02:11–02:12]: Kita sebelumnya soal jam masuk ya. [OVERLAP]
SPEAKER_03 [02:14–02:21]: Iya, beberapa orang lebih fokus kalau mulai agak siang. Tapi ada juga yang justru lebih produktif di pagi hari.
SPEAKER_02 [02:22–02:25]: Kalau menurutmu sistem fleksibel itu lebih banyak untuk. [OVERLAP]
SPEAKER_00 [02:25–02:26]: Flexible itu lebih banyak untung atau ruginya. [OVERLAP]
SPEAKER_03 [02:28–02:29]: Kalau menurut aku ya.
SPEAKER_22 [02:29–02:31]: Lebih banyak untungnya, asal.
SPEAKER_03 [02:31–02:34]: Asal ada aturan jelas kalau terlalu bebas nanti malah susah koordinasinya.
SPEAKER_02 [02:37–02:39]: Hmm, koordinasi ini penting.
SPEAKER_04 [02:40–02:42]: Apalagi buat kerja tim.
SPEAKER_00 [02:42–02:44]: Jangan sampai orang susah dihubungi.
SPEAKER_03 [02:47–02:53]: Mungkin kita bisa tentukan jam inti, misalnya jam 10 sampai jam 3.
SPEAKER_03 [02:53–02:55]: Dan di jam itu semua wajib aktif.
SPEAKER_42 [02:57–02:57]: Oke.
SPEAKER_04 [02:57–02:58]: Itu ide bagus.
SPEAKER_04 [02:59–03:01]: Jadi di luar jam inti boleh fleksibel,
SPEAKER_00 [03:02–03:03]: Tapi jam inti tetap sama.
SPEAKER_03 [03:04–03:06]: Iya biar meeting dan diskusi tetap.
SPEAKER_32 [03:06–03:07]: Tetap jalan.
SPEAKER_02 [03:09–03:10]: Terus soal komunikasi.
SPEAKER_00 [03:11–03:16]: Aku dapat beberapa masukan kalau informasi kadang telat sampai ke anggota tim.
SPEAKER_03 [03:18–03:23]: Iya, aku juga dengar. Kadang info disampaikan di suatu tempat, tapi nggak semua orang baca.
SPEAKER_00 [03:24–03:25]: Berarti.
SPEAKER_02 [03:26–03:27]: Kita perlu satu saluran komunikasi. [OVERLAP]
SPEAKER_00 [03:26–03:29]: Saluran komunikasi utama yang wajib dipantau. [OVERLAP]
SPEAKER_14 [03:31–03:31]: Setuju.
SPEAKER_03 [03:32–03:36]: Mungkin cukup satu grup utama dan informasi penting jangan disebar di tempat lain.
SPEAKER_00 [03:38–03:40]: Nanti kita tegaskan itu sebagai aturan.
SPEAKER_14 [03:42–03:43]: Ngomong-ngomong soal productivity?
SPEAKER_03 [03:44–03:46]: Gimana dengan evaluasi kerja?
SPEAKER_03 [03:47–03:48]: Selama ini kan kita cuma lihat hasil akhir.
SPEAKER_00 [03:50–03:53]: Iya, kita jarang bahas prosesnya.
SPEAKER_03 [03:54–03:57]: Padahal kadang prosesnya yang bermasalah, bukan orangnya.
SPEAKER_00 [03:59–03:59]: Setuju.
SPEAKER_02 [04:00–04:02]: Mungkin kita perlu evaluasi rutin, misalnya. [OVERLAP]
SPEAKER_00 [04:01–04:04]: Evaluasi rutin, misalnya 2 minggu sekali, [OVERLAP]
SPEAKER_02 [04:04–04:05]: Tapi singkat aja.
SPEAKER_03 [04:06–04:08]: Kayak check-in ringan ya.
SPEAKER_03 [04:09–04:10]: Bukan evaluasi formal.
SPEAKER_02 [04:11–04:14]: Iya, cuma bahas hambatan dan kebutuhan aja.
SPEAKER_03 [04:16–04:19]: Menurut aku sih itu bagus ya, biar masalah ketahuan lebih cepat.
SPEAKER_42 [04:21–04:22]: Oke kita.
SPEAKER_00 [04:22–04:23]: Kita catat itu.
SPEAKER_02 [04:24–04:25]: Evaluasi rutin dua minggu sekali.
SPEAKER_02 [04:26–04:28]: Durasinya maksimal 30 menit.
SPEAKER_03 [04:30–04:33]: Terus soal pembagian tugas, apakah perlu diubah juga?
SPEAKER_02 [04:35–04:37]: Aku rasa perlu sedikit penjelasan.
SPEAKER_00 [04:37–04:38]: Sedikit penyesuaian.
SPEAKER_04 [04:39–04:42]: Beberapa orang kelihatannya kebanyakan tugas koordinasi. [OVERLAP]
SPEAKER_02 [04:41–04:42]: Kebanyakan tugas koordinasi. [OVERLAP]
SPEAKER_03 [04:44–04:45]: Iya, itu bikin mereka kurang fokus.
SPEAKER_11 [04:46–04:47]: Pugas utama.
SPEAKER_00 [04:49–04:51]: Mungkin tugas koordinasi bisa diputar.
SPEAKER_02 [04:52–04:53]: Gak selalu orang yang sama.
SPEAKER_03 [04:54–04:56]: Jadi bebannya lebih merata ya.
SPEAKER_42 [04:58–04:58]: Oke.
SPEAKER_02 [04:59–05:02]: Jadi sejauh ini poin yang kita bahas ada beberapa.
SPEAKER_00 [05:03–05:04]: Sistem prioritas tugas.
SPEAKER_00 [05:05–05:07]: Jam kerja fleksibel dengan jam inti.
SPEAKER_00 [05:07–05:09]: Seluruh komunikasi utama.
SPEAKER_02 [05:10–05:12]: Evaluasi rutin, dan pembagian tugas.
SPEAKER_22 [05:15–05:15]: Cukup banyak juga ya.
SPEAKER_00 [05:16–05:19]: Makanya perlu dirapikan jadi keputusan yang jelas. [OVERLAP]
SPEAKER_02 [05:18–05:19]: Jadi keputusan yang jelas. [OVERLAP]
SPEAKER_03 [05:21–05:22]: Untuk sistem prioritas?
SPEAKER_03 [05:23–05:24]: Siapa yang nanti bikin aturannya?
SPEAKER_00 [05:25–05:28]: Aku bisa buat draft awalnya, nanti kamu bantu review.
SPEAKER_03 [05:30–05:30]: Ya, untuk.
SPEAKER_03 [05:31–05:34]: Untuk jam kerja fleksibel dan jam inti, kapan kita mulai terapkan.
SPEAKER_00 [05:35–05:37]: Menurutku jangan langsung.
SPEAKER_02 [05:37–05:42]: Kita sosialisasi dulu minggu ini lalu mulai minggu depan.
SPEAKER_03 [05:43–05:47]: Oke, oke. Komunikasi utama juga sekalian diubungkan ya.
SPEAKER_00 [05:48–05:49]: Iya, satu paket.
SPEAKER_03 [05:51–05:53]: Untuk evaluasi dua mingguan jadwalnya tetap.
SPEAKER_00 [05:55–05:56]: Kita tindukan hari tetap.
SPEAKER_02 [05:57–05:59]: Tapi jamnya bisa menyesuaikan.
SPEAKER_03 [06:00–06:01]: Oke, sip.
SPEAKER_03 [06:01–06:06]: Terus soal pembagian tugas kita bahas detailnya di rapat terpisah atau sekarang.
SPEAKER_00 [06:08–06:12]: Sekarang cukup garis besarnya dulu. Nah. [OVERLAP]
SPEAKER_45 [06:11–06:12]: Dulu, nah. [OVERLAP]
SPEAKER_02 [06:12–06:14]: Nah detailnya nanti kita sesuai.
SPEAKER_00 [06:14–06:16]: Nanti kita susun setelah lihat beban masing-masing.
SPEAKER_00 [06:19–06:22]: Jadi aku rangkum ya hasil rapat hari ini.
SPEAKER_03 [06:23–06:24]: Iya silahkan.
SPEAKER_00 [06:25–06:27]: Pertama kita akan membuat sistem. [OVERLAP]
SPEAKER_04 [06:26–06:28]: Kita akan membuat sistem prioritas tugas yang lebih jelas. [OVERLAP]
SPEAKER_00 [06:27–06:29]: Sistem prioritas tugas yang lebih jelas. [OVERLAP]
SPEAKER_00 [06:30–06:33]: Kedua kita terapkan jam kerja fleksibel dengan jam. [OVERLAP]
SPEAKER_02 [06:32–06:33]: Kerja fleksibel dengan jam inti. [OVERLAP]
SPEAKER_00 [06:34–06:35]: Ketiga.
SPEAKER_00 [06:35–06:38]: Kita tetapkan satu saluran komunikasi utama.
SPEAKER_00 [06:39–06:42]: Keempat, kita adakan evaluasi kerja dua minggu sekali.
SPEAKER_02 [06:43–06:46]: Kelima, pembagian tugas akan dievaluasi supaya lebih seimbang.
SPEAKER_03 [06:49–06:51]: Oke, rangkumannya sudah sesuai ya.
SPEAKER_00 [06:52–06:53]: Untuk dinerbanjot,
SPEAKER_00 [06:54–06:56]: Aku bertanggung jawab bikin draft sistem prioritas.
SPEAKER_00 [06:57–06:58]: Dan pengumuman resmi.
SPEAKER_02 [06:59–07:02]: Kamu bantu review dan susun jadwal evaluasi ya.
SPEAKER_03 [07:03–07:04]: Siap aku kerjakan minggu ini ya.
SPEAKER_02 [07:05–07:07]: Oke, kalau begitu rapat kita coba.
SPEAKER_00 [07:08–07:08]: Cukupkan sampai disini.
SPEAKER_00 [07:09–07:11]: Terima kasih atas masukannya, Beverly.
SPEAKER_03 [07:12–07:14]: Terima kasih juga ya Rumya.
'''

lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
segments = []
for l in lines:
    m = re.match(r"(SPEAKER_[0-9A-Z]+) \[([0-9:]+)–([0-9:]+)\]:\s*(.*)$", l)
    if m:
        sp, t0, t1, text = m.groups()
        def ts(s):
            parts = s.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        start = ts(t0)
        end = ts(t1)
        segments.append(TranscriptSegment(speaker_id=sp, start=start, end=end, text=text))

cfg = SummarizationConfig()
cfg.method = 'extractive'
# Use extractive (BERT embeddings) summarizer
from src.summarizer import BERTSummarizer
summ = BERTSummarizer(cfg).summarize(segments)
print('--- Overview ---')
print(summ.overview)
print('\n--- Key Points ---')
for p in summ.key_points:
    print('-', p)
print('\n--- Decisions ---')
for d in summ.decisions:
    print('-', d)
print('\n--- Action Items ---')
for a in summ.action_items:
    print('-', a)
print('\n--- Topics ---')
print(', '.join(summ.topics))
