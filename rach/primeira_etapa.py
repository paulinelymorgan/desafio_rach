import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import datetime
# %matplotlib inline

# Warnings
# import warnings
# warnings.filterwarnings('ignore')

# Styles
plt.style.use('ggplot')
sns.set_style('whitegrid')

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 11
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
# plt.rcParams['patch.force_edgecolor'] = True

# Text Preprocessing
import nltk
# nltk.download("all")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

import spacy
# nlp = spacy.load("en")

### carregando dados
endereco_arquivo = "/Users/paulinelymorgan/git/desafio_rach/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

gerarGraficos = False

### infomacoes da base
print()
base.info()
### infomacoes da base



#### plotar grafico da classificacao em formato de pizza
print("\nquantidade classificacoes: ")
print(base['IsSpam'].value_counts())

# grafico

if gerarGraficos:
    base["IsSpam"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = False, label= 'classes')
    plt.ylabel("COMUNS vs SPAMS")
    plt.legend(["COMUNS", "SPAMS"])
    plt.title("Mensagens COMUNS vs SPAMS")
    plt.show()

#### plotar grafico da classificacao em formato de pizza











# agrupado_mes_q3_2 = questao3.groupby(['Month/Year', 'Word_Count'])
# print(agrupado_mes_q3_2.size())
# print(agrupado_mes_q3_2.size().mean())

# print(questao3_2)
# testando = agrupado_mes_q3.groupby(['Month/Year', 'Word_Count']).agg({'text':'size', 'Word_Count':'mean'})\
#     .rename(columns={'text':'count','sent':'mean_sent'})\
#     .reset_index()

# testando = questao3.groupby(['Month/Year', 'Word_Count'])\
#     .agg({'Date':'size', 'Common_Word_Count':'mean'})\
#     .rename(columns={'Date':'Size', 'Common_Word_Count':'mean_'})\
#     .reset_index()

# testando = questao3.groupby(['Month/Year'])\
#     .agg({'Date':'size', 'Common_Word_Count':'mean'})\
#     .rename(columns={'Date':'Size', 'Common_Word_Count':'mean_'})\
#     .reset_index()

# agrega com o tamanho
# classificacoes_por_mes_q3 = agrupado_mes_q3.size()
# testando = testando


# print(testando)
# print(classificacoes_por_mes_q3)
# print(classificacoes_por_mes_q3.get_values())

# classificacoes_por_mes_q3.to_pickle("file_name.txt")
# import h5py
# from pandas import HDFStore,DataFrame
# store = HDFStore('store.h5')
# store = pd.HDFStore('q3.h5')
# store['classificacoes_por_mes_q3'] = classificacoes_por_mes_q3  # save it
# store['classificacoes_por_mes_q3']  # load it