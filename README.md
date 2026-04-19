# MidiLM

MidiLM adalah model Transformer yang dirancang untuk memprediksi urutan nada MIDI dan lirik. Proyek ini menggabungkan teknik pemrosesan bahasa alami (NLP) dengan data musik untuk menghasilkan melodi dan nyanyian.

## Cara Kerja
Sistem ini bekerja dengan mengubah data musik menjadi urutan token (teks). 
1. **MP3 ke MIDI**: Menggunakan AI Basic Pitch untuk mengekstraksi nada dari file audio.
2. **Tokenisasi**: Mengubah nada dan lirik menjadi angka (token) agar bisa diproses model.
3. **Prediksi**: Model belajar memprediksi nada atau kata berikutnya berdasarkan data yang sudah ada.
4. **Rekonstruksi**: Mengubah kembali hasil prediksi menjadi file musik (MIDI/MP3).

## Fitur Saat Ini
- Transkripsi otomatis dari file audio (MP3/WAV/OGG).
- Dukungan lirik untuk lagu Bahasa Indonesia dan Inggris.
- Penyesuaian nada (Augmentasi) otomatis saat penyiapan data.

## Keterbatasan (Realitas)
- **Kualitas MIDI**: Hasil transkripsi dari MP3 tidak selalu 100% akurat. Musik yang terlalu kompleks (banyak instrumen) akan menghasilkan MIDI yang berantakan.
- **Ekstraksi Lirik**: Sistem saat ini tidak bisa mengambil lirik langsung dari MP3. Lirik harus dimasukkan secara manual dalam file MIDI jika ingin AI belajar bernyanyi.
- **Hardware**: Membutuhkan GPU (minimal Tesla T4 di Colab) untuk proses training dan generasi yang cepat.

## Struktur Folder
- `model/`: Arsitektur Transformer.
- `tokenizer/`: Logika pengubahan musik ke token.
- `dataset/`: Skrip konversi audio dan penyiapan data.
- `train/`: Skrip pelatihan model.
- `inference/`: Skrip pembuatan musik baru.
- `converter/`: Skrip pengubah token kembali ke musik.

## Penggunaan
Prosedur lengkap penggunaan tersedia di dalam notebook `MidiLM_Colab.ipynb`. Silakan ikuti langkah-langkah di sana mulai dari instalasi dependensi hingga tahap pembuatan lagu.
