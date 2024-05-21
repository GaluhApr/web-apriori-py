from datetime import date
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
    scaler = MinMaxScaler()
    df[['Tanggal', 'Bulan', 'Tahun']] = scaler.fit_transform(df[['Tanggal', 'Bulan', 'Tahun']])
    return df

def preprocess_data(df, tanggal, sep, dateformat):
    df = prep_date(df, tanggal, sep, dateformat)
    df = normalize_data(df)
    return df

def prep_date(df, tanggal, sep, dateformat):
    if dateformat == 'ddmmyy':
        df['Tanggal'] = df[tanggal].apply(lambda x: int(x.split(sep)[0]))
        df['Bulan'] = df[tanggal].apply(lambda x: int(x.split(sep)[1]))
        df['Tahun'] = df[tanggal].apply(lambda x: int(x.split(sep)[2]))
    elif dateformat == 'mmddyy':
        df['Tanggal'] = df[tanggal].apply(lambda x: int(x.split(sep)[1]))
        df['Bulan'] = df[tanggal].apply(lambda x: int(x.split(sep)[0]))
        df['Tahun'] = df[tanggal].apply(lambda x: int(x.split(sep)[2]))
    elif dateformat == 'yymmdd':
        df['Tanggal'] = df[tanggal].apply(lambda x: int(x.split(sep)[2]))
        df['Bulan'] = df[tanggal].apply(lambda x: int(x.split(sep)[1]))
        df['Tahun'] = df[tanggal].apply(lambda x: int(x.split(sep)[0]))
    return df

def dataset_settings(df, pembeli, tanggal, produk):
    c1, c2 = st.columns((2, 1))
    year_list = ['Semua']
    year_list = np.append(year_list, df['Tahun'].unique())
    by_year = c1.selectbox('Tahun ', (year_list))
    if by_year != 'Semua':
        df = df[df['Tahun'] == int(by_year)]
        by_month = c2.slider('Bulan', 1, 12, (1, 12))
        df = df[df['Bulan'].between(int(by_month[0]), int(by_month[1]), inclusive="both")]
    return df

def show_transaction_info(df, produk, pembeli):
    try:
        col1, col2 = st.columns(2)
        st.subheader(f'Informasi Transaksi:')
        total_produk = df[produk].nunique()
        total_transaksi = df[pembeli].nunique()
        total_barang_terjual = df[produk].sum()  # Menghitung jumlah total barang terjual
        total_frekuensi_produk = len(df)  # Menghitung frekuensi total dari semua produk
        col1.info(f'Produk terjual     : {total_produk}')
        col2.info(f'Total transaksi  : {total_transaksi}')
        col2.info(f'Frekuensi total produk terjual  : {total_frekuensi_produk}')  # Menampilkan frekuensi total produk terjual
        sort = col1.radio('Tentukan kategori produk', ('Terlaris', 'Kurang Laris'))
        jumlah = col2.slider('Tentukan jumlah produk', 0, total_produk, 10)
        if sort == 'Terlaris':
            most_sold = df[produk].value_counts().head(jumlah)
        else:
            most_sold = df[produk].value_counts().tail(jumlah)
            most_sold = most_sold.sort_values(ascending=True)
        if not most_sold.empty:
            c1, c2 = st.columns((2, 1))
            plt.figure(figsize=(8, 4)) 
            most_sold.plot(kind='bar')
            plt.title('Grafik Penjualan')
            c1.pyplot(plt)
            c2.write(most_sold)
        else:
            st.warning("Tidak ada data yang sesuai dengan kriteria yang dipilih.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menampilkan informasi transaksi: {str(e)}")
        
def data_summary(df, pembeli, tanggal, produk):
    st.header('Ringkasan Dataset')
    col1, col2 = st.columns(2)
    sep_option = col1.radio('Tentukan separator tanggal', options=[('-', 'Dash'), ('/', 'Slash')])
    sep = sep_option[0]
    dateformat = col2.radio('Tentukan format urutan tanggal', ('ddmmyy', 'mmddyy', 'yymmdd'))
    try:
        df = prep_date(df, tanggal, sep, dateformat)
    except ValueError:
        st.warning('Format tanggal tidak sesuai! Silakan cek kembali dan pastikan format yang benar.')
        st.stop()
    except IndexError:
        st.warning('Separator tanggal salah! Silakan cek kembali dan pastikan pemisah yang benar.')
        st.stop()
    st.write('Setelan Tampilan Dataset:')
    df = dataset_settings(df, pembeli, tanggal, produk)
    st.dataframe(df.sort_values(by=['Tahun', 'Bulan', 'Tanggal'], ascending=True))
    show_transaction_info(df, produk, pembeli)
    return df

def prep_frozenset(rules):
    temp = re.sub(r'frozenset\({', '', str(rules))
    temp = re.sub(r'}\)', '', temp)
    return temp

def MBA(df, pembeli, produk):
    st.header('Association Rule Mining Menggunakan Apriori')
    if st.button("Mulai Perhitungan Asosiasi"):
        start_time = time.time()  
        transaction_list = []
        for i in df[pembeli].unique():
            tlist = list(set(df[df[pembeli]==i][produk]))
            if len(tlist)>0:
                transaction_list.append(tlist)
        te = TransactionEncoder()
        te_ary = te.fit(transaction_list).transform(transaction_list)
        df2 = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df2, min_support=0.02, use_colnames=True)   #nilai support yang digunakan
        try:
            rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.01) 
            # Ganti min_threshold sesuai dengan nilai confidence yang diinginkan
        except ValueError as e:
            st.error(f"Terjadi kesalahan saat menghasilkan aturan asosiasi: {str(e)}")
            st.stop()
        end_time = time.time()  
        processing_time = end_time - start_time  
        col1, col2 = st.columns(2)
        col1.subheader('Hasil Rules (Pola Pembelian Pelanggan)')
        st.write('Total rules yang dihasilkan :', len(rules))
        col1.write(f'Waktu yang dibutuhkan untuk memproses rule: {processing_time:.2f} detik')
        if len(rules) == 0:  # Tidak ada aturan yang dihasilkan
            st.write("Tidak ada aturan yang dihasilkan.")
        else:
            antecedents = rules['antecedents'].apply(prep_frozenset)
            consequents = rules['consequents'].apply(prep_frozenset)
            matrix = {
                'antecedents': antecedents,
                'consequents': consequents,
                'support': rules['support'],
                'confidence': rules['confidence'],
                'lift ratio': rules['lift'],
                'contribution': rules['support'] * rules['confidence']
            }
            matrix = pd.DataFrame(matrix)
            matrix.reset_index(drop=True, inplace=True)
            matrix.index += 1
            col1.write(matrix)
            col2.subheader('Keterangan')
            col2.write("- Support = Seberapa sering sebuah rules tersebut muncul dalam data,")
            col2.write("- Confidence = Seberapa sering rules tersebut dikatakan benar")
            col2.write("- Lift Ratio = Ukuran Kekuatan hubungan antara dua item")
            col2.write("- Contribution = Kontribusi setiap rules terhadap peningkatan lift secara keseluruhan")
            
            # Tampilkan rekomendasi stok barang untuk dibeli
            col1, col2 = st.columns(2)
            col1.subheader("Rekomendasi stok barang untuk dibeli (contribution) :")
            recommended_products = []
            recommended_products_contribution = {}
            for antecedent, contribution in zip(matrix['antecedents'], matrix['contribution']):
                antecedent_list = antecedent.split(', ')
                for item in antecedent_list:
                    if item not in recommended_products_contribution:
                        recommended_products_contribution[item] = contribution
                    else:
                        recommended_products_contribution[item] += contribution
                recommended_products.extend(antecedent_list)
            recommended_products = list(set(recommended_products))  # Hapus duplikat
            recommended_products_sorted = sorted(recommended_products, key=lambda x: (recommended_products_contribution[x], matrix[matrix['antecedents'].apply(lambda y: x in y)]['lift ratio'].values[0]), reverse=True)
            for idx, item in enumerate(recommended_products_sorted, start=1):
                col1.write(f"{idx}. <font color='red'>{item}</font> ({recommended_products_contribution[item]})", unsafe_allow_html=True)
            # Tampilkan informasi tentang produk yang paling laris terjual dalam bentuk tabel
            most_sold = df[produk].value_counts()
            if not most_sold.empty:
                col2.subheader("Jumlah Produk Terjual")
                col2.dataframe(most_sold, width=400)  
            else:
                st.warning("Tidak ada data yang sesuai dengan kriteria yang dipilih.")

            for a, c, supp, conf, lift in sorted(zip(matrix['antecedents'], matrix['consequents'], matrix['support'], matrix['confidence'], matrix['lift ratio']), key=lambda x: x[4], reverse=True):
                st.info(f'Jika customer membeli {a}, maka customer juga membeli {c}')
                st.write('Support : {:.3f}'.format(supp))
                st.write('Confidence : {:.3f}'.format(conf))
                st.write('Lift Ratio : {:.3f}'.format(lift))
                st.write('Contribution : {:.3f}'.format(supp * conf))
                st.write('')