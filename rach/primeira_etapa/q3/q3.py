import pandas as pd

### carregando dados
endereco_arquivo = "/Users/paulinelymorgan/git/desafio_rach/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

#### QUESTAO 3 - varias metricas
print("\nquestao 3: ")

# extraindo apenas os dois parametros
questao3 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Word_Count': base.Word_Count})

# criando um novo parametro (mes/ano)
questao3['Month/Year'] = questao3['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))

# agrupa pelos dois parametros (o novo gerado e 'Word_Count')
print(questao3)
agrupado_mes_q3 = questao3.groupby(['Month/Year'])

print()

print("\nmax: {}".format(agrupado_mes_q3.max().rename(columns={'Word_Count':'MAX'})))
print("\nmin: {}".format(agrupado_mes_q3.min().rename(columns={'Word_Count':'MIN'})))
print("\nmean: {}".format(agrupado_mes_q3.mean().rename(columns={'Word_Count':'MEAN'})))
print("\nmedian: {}".format(agrupado_mes_q3.median().rename(columns={'Word_Count':'MEDIAN'})))
print("\nstd: {}".format(agrupado_mes_q3.std().rename(columns={'Word_Count':'STD'})))
print("\nvar: {}".format(agrupado_mes_q3.var().rename(columns={'Word_Count':'VARIANCE'})))
print("\nsize: {}".format(agrupado_mes_q3.size()))

#### QUESTAO 3 - varias metricas
