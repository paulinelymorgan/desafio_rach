import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

### carregando dados
endereco_arquivo = "/Users/paulinelymorgan/git/desafio_rach/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

gerarGraficos = True

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