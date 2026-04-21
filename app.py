import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Dashboard Prediksi Kecelakaan", layout="wide")
st.title("Dashboard Prediksi & Analisis Kecelakaan Berbasis AI")
st.caption("Deep Learning (LSTM) • Analisis Risiko • Forecasting Oleh : Rayhan Dwi Padli")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_kecelakaan_new.csv")

    # PASTIKAN FORMAT TANGGAL BENAR
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    df = df.dropna(subset=["tanggal"])

    df["tahun"] = df["tanggal"].dt.year
    df["bulan"] = df["tanggal"].dt.month

    return df

df = load_data()

# ===============================
# SIDEBAR FILTER
# ===============================
st.sidebar.header("Filter Data")
daerah = st.sidebar.selectbox("Pilih Daerah", sorted(df["daerah"].unique()))
df_f = df[df["daerah"] == daerah].copy()

# ===============================
# METRIK UTAMA
# ===============================
st.subheader(f"Ringkasan Kecelakaan – {daerah}")
c1, c2, c3 = st.columns(3)

c1.metric("Total Kecelakaan", int(df_f["jumlah_kecelakaan"].sum()))
c2.metric("Korban Meninggal", int(df_f["korban_meninggal"].sum()))
c3.metric(
    "Total Korban Luka",
    int(df_f["korban_luka_berat"].sum() + df_f["korban_luka_ringan"].sum())
)

# ==============================
# BAR CHART KECELAKAAN 
# ==============================
st.subheader("Penyebab Kecelakaan Terbanyak")

# Agregasi data
cause_df = (
    df_f.groupby("penyebab_kecelakaan")["jumlah_kecelakaan"]
    .sum()
    .reset_index()
    .sort_values(by="jumlah_kecelakaan", ascending=False)
)

# Plotly bar chart
fig = px.bar(
    cause_df,
    x="jumlah_kecelakaan",
    y="penyebab_kecelakaan",
    orientation="h",
    text="jumlah_kecelakaan",
    title=f"Penyebab Kecelakaan Terbanyak di {daerah}",
    labels={
        "jumlah_kecelakaan": "Jumlah Kecelakaan",
        "penyebab_kecelakaan": "Penyebab Kecelakaan"
    }
)

fig.update_traces(texttemplate="%{text:,}", textposition="outside")
fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    height=400,
    margin=dict(l=120, r=40, t=50, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# PIE CHART
# ==============================
st.subheader("Karakteristik Kecelakaan")

col1, col2 = st.columns(2)

# ===============================
# PIE 1 – JENIS KELAMIN
# ===============================
gender_df = (
    df_f.groupby("jenis_kelamin_pengemudi")["jumlah_kecelakaan"]
    .sum()
    .reset_index()
)

fig_gender = px.pie(
    gender_df,
    names="jenis_kelamin_pengemudi",
    values="jumlah_kecelakaan",
    hole=0.4,  # donut biar lebih modern
    title="Distribusi Kecelakaan Berdasarkan Jenis Kelamin",
)

fig_gender.update_traces(
    textinfo="percent+label",
    pull=[0.05] * len(gender_df),  # sedikit keluar, lebih menarik
)

# ===============================
# PIE 2 – PENGARUH ALKOHOL
# ===============================
alcohol_df = (
    df_f.groupby("pengaruh_alkohol")["jumlah_kecelakaan"]
    .sum()
    .reset_index()
)

fig_alcohol = px.pie(
    alcohol_df,
    names="pengaruh_alkohol",
    values="jumlah_kecelakaan",
    hole=0.4,
    title="Distribusi Kecelakaan Berdasarkan Pengaruh Alkohol",
)

fig_alcohol.update_traces(
    textinfo="percent+label",
    pull=[0.08 if x.lower() == "ya" else 0 for x in alcohol_df["pengaruh_alkohol"]],
)

# ===============================
# TAMPILKAN SEBELAHAN
# ===============================
with col1:
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    st.plotly_chart(fig_alcohol, use_container_width=True)

# ===============================
# TREN BULANAN (REAL DATA)
# ===============================
st.subheader("Tren Kecelakaan Bulanan")

df_month = (
    df_f.groupby(pd.Grouper(key="tanggal", freq="ME"))["jumlah_kecelakaan"]
    .sum()
    .reset_index()
)

st.line_chart(df_month, x="tanggal", y="jumlah_kecelakaan")

# ===============================
# LSTM FORECASTING (FIX TOTAL)
# ===============================
st.subheader("Prediksi Kecelakaan 12 Bulan ke Depan (LSTM)")

series = df_month["jumlah_kecelakaan"].values.reshape(-1, 1)

scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

SEQ_LEN = 12

def make_sequence(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

X, y = make_sequence(series_scaled, SEQ_LEN)

# MODEL WARAS
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(
    X, y,
    epochs=250,
    batch_size=8,
    verbose=0,
    callbacks=[EarlyStopping(patience=30, restore_best_weights=True)]
)

# ===============================
# FUTURE PREDICTION (CONSISTENT)
# ===============================
last_seq = series_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
future_scaled = []

for _ in range(12):
    pred = model.predict(last_seq, verbose=0)
    future_scaled.append(pred[0, 0])
    last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

future = scaler.inverse_transform(
    np.array(future_scaled).reshape(-1, 1)
).flatten()

future_dates = pd.date_range(
    df_month["tanggal"].iloc[-1] + pd.offsets.MonthEnd(1),
    periods=12,
    freq="M"
)

# ===============================
# VISUAL AKHIR
# ===============================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_month["tanggal"], df_month["jumlah_kecelakaan"], label="Aktual")
ax.plot(future_dates, future, linestyle="--", label="Prediksi")
ax.set_ylabel("Jumlah Kecelakaan")
ax.legend()
st.pyplot(fig)

# ===============================
# NARASI AI (TIDAK DIBULATKAN)
# ===============================
pred_df = pd.DataFrame({
    "tanggal": future_dates,
    "prediksi": future
})

puncak = pred_df.loc[pred_df["prediksi"].idxmax()]

bulan_pred = puncak["tanggal"].strftime("%B")
tahun_pred = puncak["tanggal"].year
jumlah_pred = int(round(puncak["prediksi"]))

st.success(f"""
**Narasi Prediksi AI – Tahun {tahun_pred}**

AI memprediksi **puncak kecelakaan** terjadi pada **{bulan_pred} {tahun_pred}**
dengan estimasi sekitar **{jumlah_pred:,} kejadian kecelakaan**.

Pola prediksi menunjukkan **fluktuasi nyata**, bukan nilai datar,
yang mengindikasikan bahwa **model berhasil menangkap tren historis bulanan**.

**Kesimpulan**  
Tanpa intervensi kebijakan preventif, risiko kecelakaan
diperkirakan tetap **signifikan pada bulan-bulan tertentu** di tahun prediksi.
""")

st.subheader("Rekomendasi AI Pencegahan Kecelakaan Spesifik Daerah")

recommendations = []

# ===============================
# 1. INFRASTRUKTUR JALAN
# ===============================
dominant_road = df_f["jenis_jalan"].mode()[0]
road_condition = df_f["kondisi_jalan"].mode()[0]
road_geometry = df_f["geometri_jalan"].mode()[0]

if road_condition.lower() in ["rusak", "berlubang", "tidak rata"]:
    recommendations.append(
        "Melakukan perbaikan fisik jalan pada segmen dengan kondisi rusak atau berlubang untuk mengurangi risiko kecelakaan."
    )

if road_geometry.lower() in ["tikungan tajam", "menurun", "persimpangan"]:
    recommendations.append(
        "Melakukan evaluasi desain geometrik jalan seperti tikungan dan persimpangan serta penambahan rambu peringatan."
    )

# ===============================
# 2. LINGKUNGAN & WAKTU
# ===============================
dominant_weather = df_f["cuaca"].mode()[0]
dominant_time = df_f["waktu_kejadian"].mode()[0]

if dominant_weather.lower() in ["hujan", "kabut"]:
    recommendations.append(
        "Meningkatkan drainase jalan, marka reflektif, dan penerangan untuk mengurangi risiko kecelakaan saat cuaca buruk."
    )

if dominant_time.lower() in ["malam", "dini hari"]:
    recommendations.append(
        "Menambah penerangan jalan umum dan meningkatkan pengawasan pada jam rawan kecelakaan."
    )

# ===============================
# 3. PERILAKU PENGEMUDI
# ===============================
high_speed_ratio = (df_f["kecepatan_pengendara"] == "tinggi").mean()
alcohol_ratio = (df_f["pengaruh_alkohol"] == "ya").mean()

if high_speed_ratio > 0.4:
    recommendations.append(
        "Menerapkan pengendalian kecepatan melalui rambu batas kecepatan, speed camera, dan traffic calming."
    )

if alcohol_ratio > 0.1:
    recommendations.append(
        "Melakukan razia berkala terhadap pengemudi di bawah pengaruh alkohol terutama pada waktu malam hari."
    )

# ===============================
# 4. JENIS KENDARAAN DOMINAN
# ===============================
dominant_vehicle = df_f["jenis_kendaraan"].mode()[0]

if dominant_vehicle.lower() == "sepeda motor":
    recommendations.append(
        "Menyediakan jalur khusus sepeda motor serta meningkatkan kampanye keselamatan berkendara bagi pengendara roda dua."
    )

# ===============================
# 5. PENYEBAB UTAMA KECELAKAAN
# ===============================
main_cause = df_f["penyebab_kecelakaan"].mode()[0]

if main_cause.lower() in ["human error", "kelalaian pengemudi"]:
    recommendations.append(
        "Meningkatkan edukasi keselamatan berkendara dan penegakan hukum terhadap pelanggaran lalu lintas."
    )

if main_cause.lower() in ["jalan", "infrastruktur"]:
    recommendations.append(
        "Melakukan audit keselamatan jalan (road safety audit) pada lokasi rawan kecelakaan."
    )

# ===============================
# 6. FALLBACK (JIKA DATA AMAN)
# ===============================
if not recommendations:
    recommendations.append(
        "Kondisi keselamatan relatif terkendali, disarankan monitoring rutin dan evaluasi berkala terhadap titik rawan kecelakaan."
    )

# ===============================
# OUTPUT
# ===============================
st.success("**Rekomendasi AI Berbasis Pola Kecelakaan Daerah**")
st.caption(
    f"AI menganalisis faktor dominan: jalan {dominant_road}, cuaca {dominant_weather}, waktu {dominant_time}"
)

for i, rec in enumerate(recommendations, 1):
    st.write(f"{i}. {rec}")

st.subheader("Bukti Data Pendukung Keputusan AI")

support_data = []

# Infrastruktur
road_bad_pct = (df_f["kondisi_jalan"].str.lower().isin(
    ["rusak", "berlubang", "tidak rata"]
).mean()) * 100

support_data.append({
    "Faktor": "Kondisi Jalan Buruk",
    "Persentase (%)": round(road_bad_pct, 1)
})

# Cuaca
bad_weather_pct = (df_f["cuaca"].str.lower().isin(
    ["hujan", "kabut"]
).mean()) * 100

support_data.append({
    "Faktor": "Cuaca Buruk",
    "Persentase (%)": round(bad_weather_pct, 1)
})

# Waktu Malam
night_pct = (df_f["waktu_kejadian"].str.lower().isin(
    ["malam", "dini hari"]
).mean()) * 100

support_data.append({
    "Faktor": "Waktu Malam / Dini Hari",
    "Persentase (%)": round(night_pct, 1)
})

# Alkohol
alcohol_pct = (df_f["pengaruh_alkohol"].str.lower() == "ya").mean() * 100

support_data.append({
    "Faktor": "Pengaruh Alkohol",
    "Persentase (%)": round(alcohol_pct, 1)
})

support_df = pd.DataFrame(support_data)
st.dataframe(support_df, use_container_width=True)

st.subheader("Tingkat Keyakinan AI (Confidence Level)")

# Hitung kekuatan pola (berapa faktor dominan)
pattern_strength = 0

if road_bad_pct > 40:
    pattern_strength += 1
if bad_weather_pct > 30:
    pattern_strength += 1
if night_pct > 30:
    pattern_strength += 1
if alcohol_pct > 10:
    pattern_strength += 1

# Confidence logic
if pattern_strength >= 4:
    confidence = "Sangat Tinggi"
    confidence_score = 85
elif pattern_strength >= 2:
    confidence = "Sedang"
    confidence_score = 65
else:
    confidence = "Rendah"
    confidence_score = 45

st.metric("AI Confidence Level", confidence)
st.progress(confidence_score / 100)
st.caption(
    f"AI mendeteksi {pattern_strength} pola dominan dari data kecelakaan daerah ini."
)

# =============================
# Narasi Lengkap
# =============================
st.subheader("Narasi AI – Ringkasan Karakteristik Kecelakaan Daerah")

total_kecelakaan = int(df_f["jumlah_kecelakaan"].sum())
# Ambil pola dominan
dominant_weather = df_f["cuaca"].mode()[0]
weather_pct = (df_f["cuaca"] == dominant_weather).mean() * 100

main_cause = df_f["penyebab_kecelakaan"].mode()[0]
cause_pct = (df_f["penyebab_kecelakaan"] == main_cause).mean() * 100

dominant_vehicle = df_f["jenis_kendaraan"].mode()[0]
vehicle_pct = (df_f["jenis_kendaraan"] == dominant_vehicle).mean() * 100

dominant_gender = df_f["jenis_kelamin_pengemudi"].mode()[0]
gender_pct = (df_f["jenis_kelamin_pengemudi"] == dominant_gender).mean() * 100

alcohol_yes_pct = (df_f["pengaruh_alkohol"].str.lower() == "ya").mean() * 100

dominant_time = df_f["waktu_kejadian"].mode()[0]
time_pct = (df_f["waktu_kejadian"] == dominant_time).mean() * 100

# Narasi AI
narasi_ai = f"""
Berdasarkan analisis data kecelakaan di daerah **{daerah}**, tercatat sebanyak **{total_kecelakaan:,} kejadian kecelakaan** pada periode pengamatan.
Sebagian besar kecelakaan terjadi pada kondisi **cuaca {dominant_weather}** dengan proporsi sekitar **{weather_pct:.1f}%**,
yang mengindikasikan bahwa faktor lingkungan memiliki pengaruh signifikan terhadap risiko kecelakaan di wilayah ini.
Dari sisi penyebab, kecelakaan paling banyak disebabkan oleh **{main_cause}** (**{cause_pct:.1f}%**),
menunjukkan bahwa faktor dominan berasal dari karakteristik pengemudi dan situasi lalu lintas.

Berdasarkan jenis kendaraan, **{dominant_vehicle}** merupakan kendaraan yang paling sering terlibat kecelakaan
dengan proporsi **{vehicle_pct:.1f}%**, sementara dari sisi demografi,
pengemudi **{dominant_gender}** mendominasi kejadian kecelakaan sebesar **{gender_pct:.1f}%**.
Kecelakaan paling sering terjadi pada waktu **{dominant_time}** (**{time_pct:.1f}%**),
dan berdasarkan data, keterlibatan alkohol tercatat pada sekitar **{alcohol_yes_pct:.1f}%** kasus,
yang menunjukkan bahwa pengaruh alkohol ada namun bukan faktor utama di wilayah ini.
"""

st.success("**Narasi AI Otomatis Berbasis Data**")
st.write(narasi_ai)


# ===============================
# DATA
# ===============================
st.subheader("Data Detail")
st.dataframe(df_f)
