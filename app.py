import komputasi
import pandas as pd
import streamlit as st
from PIL import Image

# Set judul dan ikon
st.set_page_config(page_title="Apriori Toko Handari", page_icon="images/basket.png")

# Menampilkan judul dan deskripsi
st.title('Data Mining Apriori')
st.write('OPTIMASI STOK BARANG DENGAN ANALISIS PENJUALAN MENGGUNAKAN METODE ASSOCIATION RULE MINING DI TOKO HANDARI')

# Menampilkan gambar
image = Image.open('images/image.jpg')
st.image(image)

# Memuat dataset
df = None
dataset_file = st.file_uploader("Upload Dataset Anda", type=['csv'])

# Menangani kesalahan saat memuat dataset
if dataset_file is None:
    st.warning('Mohon upload dataset Anda!')
    st.stop()

try:
    df = pd.read_csv(dataset_file)
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
    st.stop()

# Mendapatkan nama kolom
if df is not None and not df.empty:
    pembeli, tanggal, produk = df.columns[0], df.columns[1], df.columns[2]

    # Memanggil fungsi untuk prapemrosesan data
    df = komputasi.data_summary(df, pembeli, tanggal, produk)

    # Memanggil fungsi untuk melakukan Association Rule Mining menggunakan Apriori
    komputasi.MBA(df, pembeli, produk)
else:
    st.warning("Dataset kosong atau tidak valid. Mohon unggah dataset yang valid.")
