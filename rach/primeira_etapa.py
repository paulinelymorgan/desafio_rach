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

base = pd.read_csv('C:/Users/Ana Raquel/Senior/desafio/sms_senior.csv', encoding = "ISO-8859-1")
# df = pd.read_csv('C:/Users/Ana Raquel/Senior/desafio/sms_senior.csv', encoding = "latin")

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

#base.loc[x, 'Date']

def defineMounth(data):
    jan = '01'
    fev = '02'
    mar = '03'
    abr = '04'
    mai = '05'
    jun = '06'
    jul = '07'
    ago = '08'
    sete = '09'
    out = '10'
    nov = '11'
    dez = '12'

    if data[5] == jan[0] and data[6] == jan[1] :
        return 'Janeiro'
    elif data[5] == fev[0] and data[6] == fev[1]:
        return 'Fevereiro'
    elif data[5] == mar[0] and data[6] == mar[1]:
        return 'Março'
    elif data[5] == abr[0] and data[6] == abr[1]:
        return 'Abril'
    elif data[5] == mai[0] and data[6] == mai[1]:
        return 'Maio'
    elif data[5] == jun[0] and data[6] == jun[1]:
        return 'Junho'
    elif data[5] == jul[0] and data[6] == jul[1]:
        return 'Julho'
    elif data[5] == ago[0] and data[6] == ago[1]:
        return 'Agosto'
    elif data[5] == sete[0] and data[6] == sete[1]:
        return 'Setembro'
    elif data[5] == out[0] and data[6] == out[1]:
        return 'Outubro'
    elif data[5] == nov[0] and data[6] == nov[1]:
        return 'Novembro'
    elif data[5] == dez[0] and data[6] == dez[1]:
        return 'Dezembro'
    else:
        return 'data inválida'

base['Mounth'] = base['Date'].apply(defineMounth)

#print(base. tail())
     
