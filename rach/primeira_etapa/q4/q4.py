import pandas as pd
import matplotlib.pyplot as plt

### carregando dados
endereco_arquivo = "/Users/paulinelymorgan/git/desafio_rach/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
base_comum = base.loc[base['IsSpam'] == 'comum']
### carregando dados

#### QUESTAO 4 - dia
print("questao 4: ")

# extraindo apenas os dois parametros da base, com classificacao 'comum'
questao4 = pd.DataFrame({'Date': pd.to_datetime(base_comum.Date), 'IsSpam': base_comum.IsSpam})

# criando dois novos parametros
questao4['Day/Month'] = questao4['Date'].apply(lambda x: "%d/%d" % (x.day, x.month))
questao4['Month'] = questao4['Date'].apply(lambda x: "%d" % (x.month))

# separa por mes
comum_mes_1 = questao4.loc[(questao4['Month'] == '1')]
comum_mes_2 = questao4.loc[(questao4['Month'] == '2')]
comum_mes_3 = questao4.loc[(questao4['Month'] == '3')]

# agrupa pelos dois parametros
agrupado_mes_1 = comum_mes_1.groupby(['Day/Month', 'IsSpam'])
agrupado_mes_2 = comum_mes_2.groupby(['Day/Month', 'IsSpam'])
agrupado_mes_3 = comum_mes_3.groupby(['Day/Month', 'IsSpam'])

# imprime o dia que teve a maior qnt de spam no mes
# se quiser a lista ordenada por mes, apenas tirar o metodo index
print("maior numero de msg comum de jan: {}".format(agrupado_mes_1.size().sort_values().index[-1]))
# print("maior numero de msg comum de jan: {}".format(agrupado_mes_1.size().sort_values()))
print("maior numero de msg comum de fev: {}".format(agrupado_mes_2.size().sort_values().index[-1]))
# print("maior numero de msg comum de fev: {}".format(agrupado_mes_2.size().sort_values()))
print("maior numero de msg comum de mar: {}".format(agrupado_mes_3.size().sort_values().index[-1]))
# print("maior numero de msg comum de mar: {}".format(agrupado_mes_3.size().sort_values()))

#### QUESTAO 4 - dia
