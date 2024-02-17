# Import Library
import komputasi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

col1, col2 = st.columns(2)
col1.title('Data Mining Apriori')
col1.write('OPTIMASI STOK BARANG DENGAN ANALISIS PENJUALAN MENGGUNAKAN METODE ASSOCIATION RULE MINING DI TOKO HANDARI')


image = Image.open('images/image.jpg')
st.image(image)
df = None
dataset_file = st.file_uploader("Upload Dataset Anda", type=['csv'])
try:
    df = pd.read_csv(dataset_file)
except:
    st.warning('Mohon upload dataset Anda!')
    st.stop()

# Get Cols Names
pembeli = df.columns[0]
tanggal = df.columns[1]
produk = df.columns[2]

# Data Mining
df = komputasi.data_summary(df, pembeli, tanggal, produk)

# MBA using Apriori
komputasi.MBA(df, pembeli, produk)