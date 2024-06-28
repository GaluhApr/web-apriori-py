from datetime import date
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time
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
    by_year = c1.selectbox('Pilih Tahun ', (year_list))
    if by_year != 'Semua':
        df = df[df['Tahun'] == int(by_year)]
        month_list = np.arange(1, 13)  # Daftar bulan dari 1 sampai 12
        by_months = c2.multiselect('Pilih Bulan', month_list)
        if by_months:
            df = df[df['Bulan'].isin(by_months)]
    return df

def show_transaction_info(df, produk, pembeli):
    try:
        st.subheader(f'Informasi Transaksi:')
        col1, col2 = st.columns(2)
        total_produk = df[produk].nunique()
        total_transaksi = df[pembeli].nunique()
        total_barang_terjual = df[produk].sum()  #menghitung jumlah total barang terjual
        total_frekuensi_produk = len(df)  #menghitung frekuensi total dari semua produk
        col1.info(f'Produk terjual     : {total_produk} Jenis')
        col2.info(f'Total transaksi  : {total_transaksi} Transaksi')
        col2.info(f'Frekuensi total produk terjual  : {total_frekuensi_produk} Produk Terjual')  #menampilkan frekuensi total produk terjual
        sort = col1.radio('Tentukan kategori produk', ('Terlaris', 'Kurang Laris'))
        jumlah_options = list(range(1, total_produk + 1))  # Membuat daftar pilihan jumlah produk
        default_index = 9 if total_produk >= 10 else 0  # Default index untuk 10, atau 0 jika kurang dari 10 produk
        jumlah = col2.selectbox('Tentukan jumlah produk yang ingin ditampilkan', jumlah_options, index=default_index)
        if sort == 'Terlaris':
            most_sold = df[produk].value_counts().head(jumlah)
        else:
            most_sold = df[produk].value_counts().tail(jumlah)
            most_sold = most_sold.sort_values(ascending=True)
        if not most_sold.empty:
            c1, c2 = st.columns([3, 1])  # Mengubah proporsi kolom
            plt.figure(figsize=(10, 6))  # Meningkatkan ukuran grafik
            plt.title('Grafik Penjualan', fontsize=20)
            plt.xlabel('Produk', fontsize=14)   
            plt.ylabel('Jumlah', fontsize=14)
            sns.barplot(data=most_sold)
            plt.xticks(rotation=90)  # Menjadikan label vertikal
            c1.pyplot(plt)
            c2.write(most_sold)
            
        else:
            st.warning("Tidak ada data yang sesuai dengan kriteria yang dipilih.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menampilkan informasi transaksi: {str(e)}")

def data_summary(df, pembeli, tanggal, produk):
    st.markdown('<p class="big-font">Setelan Dataset</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    sep_option = col1.radio('Tentukan separator tanggal', options=[('-', 'Dash'), ('/', 'Slash')])
    sep = sep_option[0]
    dateformat = col2.radio('Tentukan format urutan tanggal', ('ddmmyy', 'mmddyy', 'yymmdd'))
    try:
        df = prep_date(df, tanggal, sep, dateformat)
    except ValueError:
        st.warning('Format Atau Separator tanggal salah! Silakan cek kembali dan pastikan pemisah yang benar.')
        st.stop()
    except IndexError:
        st.warning('Format Atau Separator tanggal salah! Silakan cek kembali dan pastikan pemisah yang benar.')
        st.stop()
    df = dataset_settings(df, pembeli, tanggal, produk)
    st.dataframe(df.sort_values(by=['Tahun', 'Bulan', 'Tanggal'], ascending=True), use_container_width=True)
    show_transaction_info(df, produk, pembeli)
    return df

def prep_frozenset(rules):
    temp = re.sub(r'frozenset\({', '', str(rules))
    temp = re.sub(r'}\)', '', temp)
    return temp

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

def MBA(df, pembeli, produk):
    st.header('Association Rule Mining Menggunakan Apriori')
    
    # Input untuk menyesuaikan minimum support dan confidence
    min_support = st.number_input("Masukkan minimum support:", min_value=0.0, max_value=1.0, format="%.3f")
    min_confidence = st.number_input("Masukkan minimum confidence:", min_value=0.0, max_value=1.0, format="%.3f")
    
    if st.button("Mulai Perhitungan Asosiasi"):
        start_time = time.time()  
        
        # Ubah data transaksi menjadi format yang sesuai untuk apriori
        transaction_list = df.groupby(pembeli)[produk].apply(list).reset_index()
        transaction_list = transaction_list[produk].apply(lambda x: pd.Series(1, index=x)).fillna(0).astype(int)
        
        # Hitung frekuensi itemset menggunakan apriori
        frequent_itemsets = apriori(transaction_list, min_support=min_support, use_colnames=True)
        
        # Buat aturan asosiasi dari frequent itemsets
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        end_time = time.time()  
        processing_time = end_time - start_time  
        
        col1, col2 = st.columns(2)
        col1.subheader('Hasil Rules (Aturan)')
        st.write('Total rules yang dihasilkan :', len(rules))
        col1.write(f'Waktu yang dibutuhkan untuk memproses rule: {processing_time:.2f} detik')
        
        if len(rules) == 0:
            st.write("Tidak ada aturan yang dihasilkan.")
        else:
            # Tampilkan aturan asosiasi
            rules.reset_index(drop=True, inplace=True)
            rules.index += 1
            col1.write(rules, use_container_width=True)
            
            col2.subheader('Keterangan')
            col2.write("- Support = Seberapa sering sebuah rules tersebut muncul dalam data")
            col2.write("- Confidence = Seberapa sering rules tersebut dikatakan benar")
            col2.write("- Lift Ratio = Ukuran Kekuatan hubungan antara dua item")
            
            # Menampilkan rekomendasi stok barang untuk dibeli
            col1, col2 = st.columns(2)
            col1.subheader("Rekomendasi barang untuk dibeli:")
            recommended_products = []
            recommended_products_contribution = {}
            
            # Ambil semua item dari antecedents dan consequents dari setiap aturan asosiasi
            for antecedent, consequent, contribution in zip(rules['antecedents'], rules['consequents'], rules['confidence']):
                antecedent_list = list(antecedent)
                consequent_list = list(consequent)
                items = antecedent_list + consequent_list
                
                # Hitung kontribusi masing-masing item
                for item in items:
                    if item not in recommended_products_contribution:
                        recommended_products_contribution[item] = contribution
                    else:
                        recommended_products_contribution[item] += contribution
                recommended_products.extend(items)
            
            # Hapus duplikat item
            recommended_products = list(set(recommended_products))  
            
            # Urutkan item berdasarkan kontribusi
            recommended_products_sorted = sorted(recommended_products, key=lambda x: recommended_products_contribution[x], reverse=True)
            
            # Tampilkan rekomendasi stok barang
            for idx, item in enumerate(recommended_products_sorted, start=1):
                col1.write(f"{idx}. <font color='red'>{item}</font>", unsafe_allow_html=True)
            
            # Menampilkan informasi tentang produk yang paling laris terjual dalam bentuk tabel
            most_sold = df[produk].value_counts()
            if not most_sold.empty:
                col2.subheader("Jumlah Produk Terjual")
                col2.dataframe(most_sold, width=400, use_container_width=True)  
            else:
                st.warning("Tidak ada data yang sesuai dengan kriteria yang dipilih.")
            

            st.subheader('Rekomendasi Pembelian Barang:')
            for a, c, supp, conf, lift, contrib in sorted(zip(rules['antecedents'], rules['consequents'], rules['support'], rules['confidence'], rules['lift'], rules['confidence']), key=lambda x: x[4], reverse=True):
                st.info(f'Jika melakukan pembelian barang {a}, maka juga lakukan pembelian barang {c}')
                st.write('Support : {:.4f}'.format(supp))
                st.write('Confidence : {:.4f}'.format(conf))
                st.write('Lift Ratio : {:.4f}'.format(lift))
                st.write('Contribution : {:.4f}'.format(contrib))
                st.write('')

            st.markdown('<br><br>', unsafe_allow_html=True)  # Menambahkan spasi vertikal

# Fungsi tambahan untuk mengonversi frozenset ke string yang dapat dibaca
def prep_frozenset(fs):
    return ', '.join(list(fs))

# Contoh penggunaan fungsi MBA dengan DataFrame df
# df = pd.read_csv('path_to_your_transaction_data.csv')
# MBA(df, 'customer_id', 'product')
