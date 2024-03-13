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

def prep_frozenset(rules):
    temp = re.sub(r'frozenset\({', '', str(rules))
    temp = re.sub(r'}\)', '', temp)
    return temp

def MBA(df, pembeli, produk):
    st.header('Association Rule Mining Menggunakan Apriori')
    transaction_list = []
    for i in df[pembeli].unique():
        tlist = list(set(df[df[pembeli]==i][produk]))
        if len(tlist)>0:
            transaction_list.append(tlist)

    te = TransactionEncoder()
    te_ary = te.fit(transaction_list).transform(transaction_list)
    df2 = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df2, min_support=0.01, use_colnames=True)   #nilai support yang digunakan
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.5) #nilai confidence yang digunakan

    st.subheader('Hasil Rules')
    antecedents = rules['antecedents'].apply(prep_frozenset)
    consequents = rules['consequents'].apply(prep_frozenset)
    matrix = {
        'antecedents':antecedents,
        'consequents': consequents,
        'support':rules['support'],
        'confidence':rules['confidence'],
        'lift':rules['lift'],
    }
    matrix = pd.DataFrame(matrix)
    show_all_rules = st.button("Tampilkan Seluruh Rules")  # Tombol untuk menampilkan seluruh rules
    
    if show_all_rules:
        st.button("Tutup")
        st.write(matrix)
    else:
        n_rules = st.number_input('Tentukan jumlah rules yang diinginkan : ', 1, len(rules['antecedents']), 1)
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

# UI code for Streamlit
def main():
    st.title('Analisis Produk Terlaris dengan Association Rule Mining')
    
    # Load Data
    st.sidebar.title('Data Settings')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
    
    pembeli_column = st.sidebar.selectbox('Select column for Buyer', df.columns)
    tanggal_column = st.sidebar.selectbox('Select column for Date', df.columns)
    produk_column = st.sidebar.selectbox('Select column for Product', df.columns)
    
    # Data Summary
    data_summary(df, pembeli_column, tanggal_column, produk_column)
    
    # Association Rule Mining
    MBA(df, pembeli_column, produk_column)

if __name__ == '__main__':
    main()
