import pandas as pd
import matplotlib.pyplot as plt

### carregando dados
endereco_arquivo = "/Users/paulinelymorgan/git/desafio_rach/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

#### QUESTAO 2 - classificacao por mes
print("questao 2: ")

# extraindo apenas os tres parametros
questao2 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Text': base.Full_Text, 'IsSpam': base.IsSpam})

# criando um novo parametro (mes/ano)
questao2['Month/Year'] = questao2['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))
# print(questao2)

# agrupa pelos dois parametros (o novo gerado e 'IsSpam')
agrupado_mes_q2 = questao2.groupby(['Month/Year', 'IsSpam'])

# agrega com o tamanho (.size) e jah plota o grafico (.plot)
agrupado_mes_q2.size().plot(kind = 'pie', figsize = (6, 6), autopct = '%1.1f%%', label=' ', legend=True)
plt.title("Mensagens COMUNS vs SPAMS / por mes")
# plt.legend(["JAN", "JAN", "FEV", "FEV", "MAR", "MAR"])
plt.show()
# print(classificacoes_por_mes_q2)
# print(classificacoes_por_mes_q2.values)

#### QUESTAO 2 - classificacao por mes
