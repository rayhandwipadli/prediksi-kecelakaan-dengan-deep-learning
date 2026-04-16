import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Prediksi Santunan AI", layout="wide")

st.title("Prediksi Beban Santunan Jasa Raharja Berbasis AI")
st.caption(
    "Prediksi santunan per kategori (meninggal, luka berat, luka ringan) "
    "berdasarkan data historis dan aturan bisnis Jasa Raharja"
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_kecelakaan_new.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal"])
    return df

df = load_data()

# ===============================
# SIDEBAR FILTER
# ===============================
st.sidebar.header("Filter Data")

daerah_list = ["Semua Daerah"] + sorted(df["daerah"].dropna().unique())
daerah = st.sidebar.selectbox("Pilih Daerah", daerah_list)

if daerah == "Semua Daerah":
    df_f = df.copy()
    label_daerah = "Seluruh Daerah"
else:
    df_f = df[df["daerah"] == daerah].copy()
    label_daerah = daerah

# ===============================
# BUSINESS RULE (WAJIB)
# ===============================
df_f = df_f[df_f["eligible_santunan"] == "ya"]

# ===============================
# TARIF RESMI (PMK 2017)
# ===============================
TARIF_MENINGGAL = 50_000_000
TARIF_LUKA_BERAT = 50_000_000
TARIF_LUKA_RINGAN = 20_000_000

kategori = {
    "Meninggal Dunia": ("korban_meninggal", TARIF_MENINGGAL),
    "Luka Berat": ("korban_luka_berat", TARIF_LUKA_BERAT),
    "Luka Ringan / Perawatan": ("korban_luka_ringan", TARIF_LUKA_RINGAN),
}

st.subheader(f"Prediksi Santunan per Kategori – {label_daerah}")

hasil_prediksi = {}

# ===============================
# LOOP AI PER KATEGORI
# ===============================
for nama, (kolom, tarif) in kategori.items():

    monthly = (
        df_f
        .groupby(pd.Grouper(key="tanggal", freq="M"))[kolom]
        .sum()
        .reset_index()
    )

    nilai = monthly[kolom].values.reshape(-1, 1) * tarif

    # cek data cukup
    if len(nilai) < 24:
        st.warning(f"Data {nama} belum cukup untuk prediksi AI.")
        continue

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(nilai)

    SEQ = 12
    X, y = [], []
    for i in range(len(scaled) - SEQ):
        X.append(scaled[i:i + SEQ])
        y.append(scaled[i + SEQ])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(SEQ, 1)),
        LSTM(16),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X, y,
        epochs=150,
        batch_size=8,
        verbose=0,
        callbacks=[EarlyStopping(patience=20, restore_best_weights=True)]
    )

    last_seq = scaled[-SEQ:].reshape(1, SEQ, 1)
    future_scaled = []

    for _ in range(12):
        pred = model.predict(last_seq, verbose=0)
        future_scaled.append(pred[0, 0])
        last_seq = np.append(
            last_seq[:, 1:, :],
            pred.reshape(1, 1, 1),
            axis=1
        )

    future = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).sum()

    hasil_prediksi[nama] = int(future)

    st.metric(
        f"Prediksi Santunan {nama}",
        f"Rp {int(future):,.0f}"
    )

# ===============================
# TOTAL
# ===============================
if hasil_prediksi:
    total = sum(hasil_prediksi.values())
    st.success(f"**Total Prediksi Santunan Tahun Depan: Rp {total:,.0f}**")

# ===============================
# DISCLAIMER
# ===============================
st.warning("""
**Catatan Penting**

Prediksi ini merupakan estimasi agregat berbasis AI.
Perhitungan hanya mencakup kasus yang **eligible santunan**
sesuai aturan bisnis Jasa Raharja.

Kategori santunan disesuaikan **sepenuhnya dengan data yang tersedia**,
tanpa asumsi tambahan terkait kondisi medis korban.
""")
