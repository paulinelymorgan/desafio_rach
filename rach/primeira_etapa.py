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
endereco_arquivo = "C:/Users/Ana Raquel/Senior/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

gerarGraficos = False

### infomacoes da base
print()
base.info()
### infomacoes da base

def grafico_barras(minhaBase, titulo, x_label, y_label, legenda, opacity):

    ocorrencias = [x[1] for x in minhaBase]
    palavras = [x[0] for x in minhaBase]
    # fig, ax = plt.subplots()
    plt.subplots()
    index = np.arange(len(minhaBase))
    bar_width = 0.25
    plt.bar(index, ocorrencias, bar_width, alpha=opacity, color='b', label=legenda)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(titulo)
    plt.xticks(index + bar_width, palavras)
    plt.legend()

    plt.tight_layout()
    plt.show()


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


#### QUESTAO 1 - palavras mais frequentes
qntPalavras = 25
palavras = Counter(" ".join(base["Full_Text"]).split()).most_common(qntPalavras)
print("\n{} palavras mais frequentes: ".format(qntPalavras))
print(palavras)

# grafico de barras

if gerarGraficos:
    grafico_barras(palavras, '{} Palavras mais frequentes'.format(qntPalavras),
        'Palavras', 'Ocorrencias', 'Ocorrencias', 0.5)

#### QUESTAO 1 - palavras mais frequentes


#### QUESTAO 2 - palavras mais frequentes por mes
print("questao 2: ")

# extraindo apenas os tres parametros
questao2 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Text': base.Full_Text, 'IsSpam': base.IsSpam})

# criando um novo parametro (mes/ano)
questao2['Month/Year'] = questao2['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))
# print(questao2)

# agrupa pelos dois parametros (o novo gerado e 'IsSpam')
agrupado_mes_q2 = questao2.groupby(['Month/Year', 'IsSpam'])

# agrega com o tamanho
classificacoes_por_mes_q2 = agrupado_mes_q2.size()
print(classificacoes_por_mes_q2)

#### QUESTAO 2 - palavras mais frequentes por mes


#### QUESTAO 3 - varias metricas
print("\nquestao 3: ")

# extraindo apenas os dois parametros
questao3 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Word_Count': base.Word_Count})
# questao3_2 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Word_Count': base.Word_Count, '': base.Word})
# questao3_2 = questao3

# criando um novo parametro (mes/ano)
questao3['Month/Year'] = questao3['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))
# questao3_2['Month/Year'] = questao3_2['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))

# agrupa pelos dois parametros (o novo gerado e 'Word_Count')
print(questao3)
agrupado_mes_q3 = questao3.groupby(['Month/Year'])

print()

print("\nsize: {}".format(agrupado_mes_q3.size()))
print("\nmax: {}".format(agrupado_mes_q3.max().rename(columns={'Word_Count':'MAX'})))
print("\nmin: {}".format(agrupado_mes_q3.min().rename(columns={'Word_Count':'MIN'})))
print("\nmean: {}".format(agrupado_mes_q3.mean().rename(columns={'Word_Count':'MEAN'})))
print("\nmed: {}".format(agrupado_mes_q3.median().rename(columns={'Word_Count':'MED'})))
print("\nstd: {}".format(agrupado_mes_q3.std().rename(columns={'Word_Count':'STD'})))
print("\nvar: {}".format(agrupado_mes_q3.var().rename(columns={'Word_Count':'VAR'})))


#### QUESTAO 3 - varias metricas


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
