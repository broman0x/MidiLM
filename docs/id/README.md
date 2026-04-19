# Dokumentasi MidiLM (Indonesia)

MidiLM adalah model Transformer yang dikhususkan untuk menciptakan musik dan nyanyian.

## Cara Kerjanya
1. **Transkripsi Audio**: Anda memberikan file MP3, lalu sistem menggunakan AI untuk mengambil nada-nada MIDI-nya.
2. **Tokenisasi**: Musik dan Lirik diubah menjadi kode teks atau "token".
3. **Training**: Model belajar menebak apa nada atau kata berikutnya (misal: jika ada nada "DO" lalu "RE", AI belajar polanya).
4. **Inference**: Anda memberikan perintah seperti [LANG: ID] Lagu sedih, dan AI akan menciptakan urutan nada dan lirik baru.
5. **Rekonstruksi**: Kode tersebut diubah kembali menjadi file MIDI, yang bisa didengarkan sebagai MP3 menggunakan SoundFont.

## Kebutuhan Sistem
- Python 3.12
- PyTorch (untuk Otak AI)
- TensorFlow (untuk mengubah MP3 ke MIDI)
- FluidSynth (untuk mendengarkan hasil musik)
