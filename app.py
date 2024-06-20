import pandas as pd
import streamlit as st
from PIL import Image
from komputasi import data_summary, MBA


st.set_page_config(page_title="Apriori Toko Handari", page_icon="images/basket.png", layout="wide")

st.markdown("""<style>
        .big-font { font-size: 30px !important; font-weight: bold; }
        body {
            zoom: 125%;
        }
    </style>""", unsafe_allow_html=True)

st.title('Data Mining Apriori')
st.write('ANALISIS DATA PENJUALAN')


image = Image.open('images/image1.jpg')
st.image(image)


dataset_file = st.file_uploader("Upload Dataset Anda", type=['csv'])
# st.write('Contoh format dataset : ')
# st.write('- ID,DATE,ITEM')
# st.write('- [Kaggle Groceries Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset?datasetId=877335&sortBy=voteCount)')


if dataset_file is None:
    st.warning('Mohon upload dataset Anda!')
    st.stop()

try:
    df = pd.read_csv(dataset_file)
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
    st.stop()


if df is not None and not df.empty:
    try:
        pembeli, tanggal, produk = df.columns[0], df.columns[1], df.columns[2]

        # Memanggil fungsi untuk prapemrosesan data
        df = data_summary(df, pembeli, tanggal, produk)

        # Memanggil fungsi untuk melakukan Association Rule Mining menggunakan Apriori
        MBA(df, pembeli, produk)
    except IndexError:
        st.warning("Indeks di luar batas. Periksa bahwa dataset memiliki setidaknya tiga kolom.")
    except ValueError:
        st.warning("Terjadi kesalahan saat prapemrosesan data. Pastikan format data yang sesuai.")
else:
    st.warning("Dataset kosong atau tidak valid. Mohon unggah dataset yang valid.")
