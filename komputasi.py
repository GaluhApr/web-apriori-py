from datetime import date
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

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
    col1, col2 = st.columns(2)
    st.subheader(f'Informasi Transaksi:')
    total_produk = df[produk].nunique()
    total_transaksi = df[pembeli].nunique()
    col1.info(f'Total produk     : {total_produk}')
    col2.info(f'Total transaksi  : {total_transaksi}')
    sort = col1.radio('Tentukan kategori produk', ('Terlaris', 'Kurang Laris'))
    jumlah = col2.slider('Tentukan jumlah produk', 0, total_produk, 5)
    if sort == 'Terlaris':
        most_sold = df[produk].value_counts().head(jumlah)
    else:
        most_sold = df[produk].value_counts().tail(jumlah)
        most_sold = most_sold.sort_values(ascending=True)
    c1, c2 = st.columns((2, 1))
    most_sold.plot(kind='bar')
    plt.title('Jumlah Produk Terjual')
    c1.pyplot(plt)
    c2.write(most_sold)

def data_summary(df, pembeli, tanggal, produk):
    st.header('Ringkasan Dataset')
    col1, col2 = st.columns(2)
    sep = col1.radio('Tentukan separator tanggal', ('-', '/'))
    dateformat = col2.radio('Tentukan format tanggal', ('ddmmyy', 'mmddyy', 'yymmdd'))
    try:
        df = prep_date(df, tanggal, sep, dateformat)
    except:
        st.warning('Separator tanggal salah!')
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

def MBA(df, pembeli, produk, min_support, min_confidence):
    st.header('Association Rule Mining Menggunakan Apriori')
    transaction_list = []
    for i in df[pembeli].unique():
        tlist = list(set(df[df[pembeli]==i][produk]))
        if len(tlist) > 0:
            transaction_list.append(tlist)

    te = TransactionEncoder()
    te_ary = te.fit(transaction_list).transform(transaction_list)
    df2 = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df2, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_confidence)

    if rules.empty:
        st.warning("Tidak ada aturan yang dihasilkan dengan nilai support dan confidence yang diberikan.")
    else:
        st.subheader('Hasil Rules')
        antecedents = rules['antecedents'].apply(prep_frozenset)
        consequents = rules['consequents'].apply(prep_frozenset)
        matrix = {
            'antecedents': antecedents,
            'consequents': consequents,
            'support': rules['support'],
            'confidence': rules['confidence'],
            'lift': rules['lift'],
        }
        matrix = pd.DataFrame(matrix)
        n_rules = st.number_input('Tentukan jumlah rules yang diinginkan : ', 1, len(matrix), 1)
        matrix = matrix.sort_values(['lift', 'confidence', 'support'], ascending=False).head(n_rules)
        
        st.write('- Support merupakan perbandingan jumlah transaksi A dan B dengan total semua transaksi')
        st.write('- Confidence merupakan perbandingan jumlah transaksi A dan B dengan total transaksi A')
        st.write('- Lift merupakan ukuran kekuatan rules "Jika customer membeli A, maka membeli B"')
        
        # Menambahkan rekomendasi stok barang yang harus dibeli
        recommended_products = set()
        for antecedent in matrix['antecedents']:
            recommended_products |= set(antecedent.split(', '))
        recommended_products = list(recommended_products)
        st.write("Rekomendasi stok barang yang harus dibeli:")
        st.write(recommended_products)
        
        for a, c, supp, conf, lift in zip(matrix['antecedents'], matrix['consequents'], matrix['support'], matrix['confidence'], matrix['lift']):
            st.info(f'Jika customer membeli {a}, maka ia membeli {c}')
            st.write('Support : {:.3f}'.format(supp))
            st.write('Confidence : {:.3f}'.format(conf))
            st.write('Lift : {:.3f}'.format(lift))
            st.write('')
