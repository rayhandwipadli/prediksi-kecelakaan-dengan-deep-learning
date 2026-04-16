import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Evaluasi Dampak Penyuluhan",
    layout="wide"
)

st.title("Evaluasi Dampak Penyuluhan / Intervensi Keselamatan")
st.caption(
    "Analisis before–after untuk mengevaluasi dampak kegiatan pencegahan "
    "terhadap tingkat kecelakaan lalu lintas"
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
st.sidebar.header("Pengaturan Analisis")

# Filter daerah
daerah_list = ["Semua Daerah"] + sorted(df["daerah"].dropna().unique())
daerah = st.sidebar.selectbox("Pilih Daerah", daerah_list)

if daerah == "Semua Daerah":
    df_f = df.copy()
    label_daerah = "Seluruh Daerah"
else:
    df_f = df[df["daerah"] == daerah].copy()
    label_daerah = daerah

# Input tanggal intervensi
st.sidebar.subheader("Periode Intervensi")
start_event = st.sidebar.date_input("Tanggal Mulai Kegiatan")
end_event = st.sidebar.date_input("Tanggal Selesai Kegiatan")

window = st.sidebar.slider(
    "Rentang Hari Sebelum & Sesudah",
    min_value=7,
    max_value=30,
    value=14,
    step=1
)

# ===============================
# VALIDASI INPUT
# ===============================
if start_event > end_event:
    st.error("Tanggal mulai tidak boleh lebih besar dari tanggal selesai.")
    st.stop()

start_event = pd.to_datetime(start_event)
end_event = pd.to_datetime(end_event)

before_start = start_event - pd.Timedelta(days=window)
after_end = end_event + pd.Timedelta(days=window)

# ===============================
# LABEL PERIODE
# ===============================
df_before = df_f[
    (df_f["tanggal"] >= before_start) &
    (df_f["tanggal"] < start_event)
]

df_during = df_f[
    (df_f["tanggal"] >= start_event) &
    (df_f["tanggal"] <= end_event)
]

df_after = df_f[
    (df_f["tanggal"] > end_event) &
    (df_f["tanggal"] <= after_end)
]

# ===============================
# METRIK UTAMA
# ===============================
def summary(df_phase):
    total = df_phase["jumlah_kecelakaan"].sum()
    days = df_phase["tanggal"].nunique()
    avg = total / days if days > 0 else 0
    return int(total), round(avg, 2)

before_total, before_avg = summary(df_before)
during_total, during_avg = summary(df_during)
after_total, after_avg = summary(df_after)

# ===============================
# TAMPILKAN METRIK
# ===============================
st.subheader(f"Hasil Evaluasi – {label_daerah}")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Sebelum Intervensi",
    f"{before_total} kejadian",
    f"{before_avg} / hari"
)

c2.metric(
    "Saat Intervensi",
    f"{during_total} kejadian",
    f"{during_avg} / hari"
)

c3.metric(
    "Sesudah Intervensi",
    f"{after_total} kejadian",
    f"{after_avg} / hari"
)

# ===============================
# PERBANDINGAN BEFORE VS AFTER
# ===============================
st.subheader("Perbandingan Sebelum vs Sesudah")

if before_total > 0:
    diff_total = after_total - before_total
    diff_pct = (diff_total / before_total) * 100
else:
    diff_pct = 0

st.write(
    f"""
**Total kecelakaan sebelum intervensi:** {before_total}  
**Total kecelakaan sesudah intervensi:** {after_total}  

Perubahan total kecelakaan: **{diff_pct:.1f}%**
"""
)

# ===============================
# GRAFIK
# ===============================
st.subheader("Visualisasi Perbandingan")

plot_df = pd.DataFrame({
    "Periode": ["Sebelum", "Saat", "Sesudah"],
    "Total Kecelakaan": [before_total, during_total, after_total]
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(plot_df["Periode"], plot_df["Total Kecelakaan"])
ax.set_ylabel("Jumlah Kecelakaan")
ax.set_title("Perbandingan Kecelakaan per Periode")
st.pyplot(fig)

# ===============================
# NARASI AI
# ===============================
st.subheader("Narasi Analisis Otomatis")

if diff_pct < 0:
    narasi = (
        f"Terjadi **penurunan kecelakaan sebesar {abs(diff_pct):.1f}%** "
        "pada periode sesudah intervensi dibandingkan sebelum kegiatan dilakukan. "
        "Hal ini mengindikasikan bahwa intervensi keselamatan berpotensi memberikan "
        "dampak positif terhadap penurunan risiko kecelakaan."
    )
elif diff_pct > 0:
    narasi = (
        f"Terjadi **peningkatan kecelakaan sebesar {diff_pct:.1f}%** "
        "pada periode sesudah intervensi dibandingkan periode sebelumnya. "
        "Hal ini menunjukkan bahwa dampak intervensi belum optimal atau dipengaruhi "
        "faktor eksternal lain."
    )
else:
    narasi = (
        "Tidak terdapat perubahan signifikan antara periode sebelum dan sesudah "
        "intervensi keselamatan."
    )

st.success(narasi)

# ===============================
# DISCLAIMER
# ===============================
st.warning(
    "Analisis ini dilakukan berdasarkan periode waktu pelaksanaan kegiatan "
    "yang ditentukan pengguna dan tidak menggunakan data partisipasi langsung "
    "kegiatan penyuluhan atau razia."
)

# ===============================
# DATA DETAIL
# ===============================
with st.expander("Lihat Data per Periode"):
    st.write("### Sebelum Intervensi")
    st.dataframe(df_before)

    st.write("### Saat Intervensi")
    st.dataframe(df_during)

    st.write("### Sesudah Intervensi")
    st.dataframe(df_after)