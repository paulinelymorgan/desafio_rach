import pandas as pd

### carregando dados
endereco_arquivo = "/Users/paulinelymorgan/git/desafio_rach/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

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