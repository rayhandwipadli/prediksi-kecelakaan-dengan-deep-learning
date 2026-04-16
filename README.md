# Dashboard Prediksi Kecelakaan Berbasis AI

Aplikasi berbasis **Streamlit** untuk analisis dan prediksi kecelakaan lalu lintas menggunakan pendekatan **Data Analytics** dan **Deep Learning (LSTM)**.

Project ini menggabungkan:

* Visualisasi data interaktif
* Prediksi kecelakaan berbasis AI
* Analisis risiko kecelakaan
* Rekomendasi otomatis berbasis data

---

## Fitur Utama

### 1. Dashboard Interaktif

* Filter berdasarkan **daerah**
* Menampilkan:

  * Total kecelakaan
  * Korban meninggal
  * Total korban luka

---

### 2. Visualisasi Data

* Bar chart penyebab kecelakaan
* Pie chart:

  * Distribusi jenis kelamin
  * Pengaruh alkohol
* Tren kecelakaan bulanan

---

### 3. Prediksi AI (LSTM)

Menggunakan model **Long Short-Term Memory (LSTM)** untuk:

* Menganalisis data historis
* Memprediksi jumlah kecelakaan **12 bulan ke depan**
* Menampilkan grafik aktual vs prediksi

---

### 4. Narasi AI Otomatis

Sistem menghasilkan insight otomatis seperti:

* Prediksi puncak kecelakaan
* Analisis tren
* Interpretasi hasil model

---

### 5. Rekomendasi AI

Rekomendasi berbasis data meliputi:

* Infrastruktur jalan
* Cuaca & waktu kejadian
* Perilaku pengemudi
* Jenis kendaraan dominan
* Penyebab utama kecelakaan

---

### 6. Confidence Level AI

* Mengukur tingkat keyakinan model
* Berdasarkan pola dominan:

  * Kondisi jalan
  * Cuaca
  * Waktu
  * Alkohol

---

## Teknologi yang Digunakan

* Python
* Streamlit
* Pandas & NumPy
* Plotly & Matplotlib
* Scikit-learn
* TensorFlow / Keras (LSTM)

---

## Struktur Project

```
project_kecelakaan/
│
├── app.py
├── requirements.txt
├── data/
│   └── data_kecelakaan_new.csv
├── pages/
└── README.md
```

---

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

---

## Cara Kerja Model AI

1. Data kecelakaan diolah menjadi time series bulanan
2. Data dinormalisasi menggunakan **MinMaxScaler**
3. Dibentuk sequence data untuk LSTM
4. Model dilatih menggunakan:

   * 2 layer LSTM
   * EarlyStopping
5. Model memprediksi 12 bulan ke depan
6. Hasil dikembalikan ke skala asli dan divisualisasikan

---

## Tujuan Project

Project ini dibuat untuk:

* Membantu analisis risiko kecelakaan
* Memberikan insight berbasis data
* Mendukung pengambilan keputusan
* Menjadi prototype sistem **Smart Traffic Analytics**

---

## Author

**Rayhan Dwi Padli**
Data Analyst & Programmer

---

## Catatan

* Model bergantung pada kualitas data
* Prediksi bersifat estimasi, bukan kepastian
* Disarankan untuk penggunaan analisis & eksplorasi

---

## Future Improvement

* Integrasi dengan data real-time
* Penambahan model AI lain (XGBoost, Prophet)
* Mapping lokasi kecelakaan (GIS)
* Deployment ke web production

---

⭐ Jika project ini membantu, jangan lupa kasih star!
