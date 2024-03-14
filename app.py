import komputasi
import pandas as pd
import streamlit as st
from PIL import Image

# Set judul dan ikon
st.set_page_config(page_title="Apriori Toko Handari", page_icon="images/basket.png", layout="wide")

# Menampilkan judul dan deskripsi
st.title('Data Mining Apriori')
st.write('OPTIMASI STOK BARANG DENGAN ANALISIS PENJUALAN MENGGUNAKAN METODE ASSOCIATION RULE MINING DI TOKO HANDARI')

# Menampilkan gambar
image = Image.open('images/image.jpg')
st.image(image)

# Memuat dataset
df = None
dataset_file = st.file_uploader("Upload Dataset Anda", type=['csv'])
st.write('Contoh format dataset : ')
st.write('-  [ID,DATE,ITEM]')
st.write('- https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset?datasetId=877335&sortBy=voteCount')
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
    try:
        pembeli, tanggal, produk = df.columns[0], df.columns[1], df.columns[2]

        # Memanggil fungsi untuk prapemrosesan data
        df = komputasi.data_summary(df, pembeli, tanggal, produk)

        # Memanggil fungsi untuk melakukan Association Rule Mining menggunakan Apriori
        komputasi.MBA(df, pembeli, produk)
    except IndexError:
        st.warning("Indeks di luar batas. Periksa bahwa dataset memiliki setidaknya tiga kolom.")
    except ValueError:
        st.warning("Terjadi kesalahan saat prapemrosesan data. Pastikan format data yang sesuai.")
else:
    st.warning("Dataset kosong atau tidak valid. Mohon unggah dataset yang valid.")
