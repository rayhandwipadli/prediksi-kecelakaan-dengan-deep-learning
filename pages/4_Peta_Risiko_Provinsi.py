import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Peta Risiko Kota", layout="wide")
st.title("Peta Risiko Kecelakaan per Kota")
st.caption("Visualisasi spasial tingkat kecelakaan berdasarkan kota")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_kecelakaan_new.csv")
    df["daerah"] = df["daerah"].str.strip()
    return df

df = load_data()

# ===============================
# FILTER TAHUN
# ===============================
tahun_list = sorted(df["tahun"].unique())
selected_year = st.selectbox("Pilih Tahun", tahun_list)

df = df[df["tahun"] == selected_year]

# ===============================
# AGREGASI PER KOTA
# ===============================
agg = (
    df.groupby("daerah")["jumlah_kecelakaan"]
    .sum()
    .reset_index()
)

# ===============================
# KOORDINAT KOTA (HARDCODE)
# ===============================
city_coords = {
    "Semarang": [-6.9667, 110.4167],
    "Bandung": [-6.9175, 107.6191],
    "Bekasi": [-6.2383, 106.9756],
    "Jakarta Barat": [-6.1674, 106.7637],
    "Jakarta Selatan": [-6.2615, 106.8106],
    "Jakarta Timur": [-6.2250, 106.9004],
    "Makassar": [-5.1477, 119.4327],
    "Medan": [3.5952, 98.6722],
    "Palembang": [-2.9761, 104.7754],
    "Surabaya": [-7.2575, 112.7521],
    "Yogyakarta": [-7.7956, 110.3695]
}

# ===============================
# BUAT MAP
# ===============================
m = folium.Map(location=[-2.5, 118], zoom_start=5)

# Tentukan scaling ukuran lingkaran
max_value = agg["jumlah_kecelakaan"].max()

for _, row in agg.iterrows():
    city = row["daerah"]
    value = row["jumlah_kecelakaan"]

    if city in city_coords:
        lat, lon = city_coords[city]

        radius = (value / max_value) * 40  # skala ukuran

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=f"""
            <b>{city}</b><br>
            Total Kecelakaan: {int(value):,}<br>
            Tahun: {selected_year}
            """,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.6,
        ).add_to(m)

# ===============================
# TAMPILKAN MAP
# ===============================
st_folium(m, width=1000, height=600)

# ===============================
# RANKING KOTA
# ===============================
st.subheader("Ranking Kota Risiko Tertinggi")

ranking = agg.sort_values(by="jumlah_kecelakaan", ascending=False)
st.dataframe(ranking, use_container_width=True)
