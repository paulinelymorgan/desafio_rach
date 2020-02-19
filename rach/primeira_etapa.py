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
questao2 = pd.DataFrame({'Timestamp': pd.to_datetime(base.Date), 'Texto': base.Full_Text, 'IsSpam': base.IsSpam})

# criando um novo parametro (mes/ano)
questao2['Month/Year'] = questao2['Timestamp'].apply(lambda x: "%d/%d" % (x.month, x.year))
# print(questao2)

# agrupa pelos dois parametros (o novo gerado e 'IsSpam')
agrupado_mes_q2 = questao2.groupby(['Month/Year', 'IsSpam'])

# agrega com o tamanho
classificacoes_por_mes_q2 = agrupado_mes_q2.size()
print(classificacoes_por_mes_q2)

#### QUESTAO 2 - palavras mais frequentes por mes


#### QUESTAO 3 - palavras mais frequentes por mes
print("questao 3: ")

# extraindo apenas os dois parametros
questao3 = pd.DataFrame({'Timestamp': pd.to_datetime(base.Date), 'Word_Count': base.Word_Count})

# criando um novo parametro (mes/ano)
questao3['Month/Year'] = questao3['Timestamp'].apply(lambda x: "%d/%d" % (x.month, x.year))
# print(questao3)

# agrupa pelos dois parametros (o novo gerado e 'Word_Count')
agrupado_mes_q3 = questao3.groupby(['Month/Year', 'Word_Count'])

# agrega com o tamanho
classificacoes_por_mes_q3 = agrupado_mes_q3.size()
print(classificacoes_por_mes_q3)