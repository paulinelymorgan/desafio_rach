import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
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

base = pd.read_csv('/Users/paulinelymorgan/Google Drive/comRaquel/desafio/sms_senior.csv', encoding = "ISO-8859-1")
# df = pd.read_csv('/Users/paulinelymorgan/Google Drive/comRaquel/desafio/sms_senior.csv', encoding = "latin")

#### infomacoes da base
# base.info()
#### infomacoes da base

#### plotar grafico da classificacao em formato de pizza
print("\nquantidade classificacoes: ")
print(base['IsSpam'].value_counts())

# grafico
base["IsSpam"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = False, label= 'classes')
plt.ylabel("COMUNS vs SPAMS")
plt.legend(["COMUNS", "SPAMS"])
plt.title("Mensagens COMUNS vs SPAMS")
plt.show()
#### plotar grafico da classificacao em formato de pizza

#### palavras mais frequentes
qntPalavras = 20
palavras = Counter(" ".join(base["Full_Text"]).split()).most_common(qntPalavras)
print("\n{} palavras mais frequentes: ".format(qntPalavras))
print(palavras)

# grafico
tamanho = len(palavras)
ocorrencias = [x[1] for x in palavras]
palavras = [x[0] for x in palavras]
fig, ax = plt.subplots()
index = np.arange(tamanho)
bar_width = 0.25
opacity = 0.4

plt.bar(index, ocorrencias, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Ocorrencias')

plt.xlabel('Palavras')
plt.ylabel('Ocorrencias')
plt.title('{} Palavras mais frequentes'.format(qntPalavras))
plt.xticks(index + bar_width, palavras)
plt.legend()

plt.tight_layout()
plt.show()
#### palavras mais frequentes


#Separando por mês


