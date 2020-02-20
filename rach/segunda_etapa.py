import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import datetime
import sklearn
from sklearn import preprocessing
import re
# %matplotlib inline

# Warnings
# import warnings
# warnings.filterwarnings('ignore')

# Text Preprocessing
import nltk
#nltk.download("all")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

import spacy
# nlp = spacy.load("en")

### carregando dados
endereco_arquivo = 'C:/Users/Ana Raquel/Senior/desafio/sms_senior.csv'
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

example = """  ***** CONGRATlations **** You won 2 tIckETs to Hamilton in 
NYC http://www.hamiltonbroadway.com/J?NaIOl/event   wORtH over $500.00...CALL 
555-477-8914 or send message to: hamilton@freetix.com to get ticket !! !  """

### infomacoes da base
#print()
#base.info()
### infomacoes da base

y = base['IsSpam'].value_counts()
print(y)

le = preprocessing.LabelEncoder()
y_enc = le.fit_transform(y)

raw_text = base['Full_Text']

stop_words=set(stopwords.words("english"))

processed = raw_text.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b','emailaddr')
processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)','httpaddr')
processed = processed.str.replace(r'£|\$', 'moneysymb')    
processed = processed.str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr')    
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')

processed = processed.str.lower()

stop_words = nltk.corpus.stopwords.words('english')

processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words))
)

porter = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(
    porter.stem(term) for term in x.split())
)



def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', cleaned)
    cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
    cleaned = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
    return ' '.join(
        porter.stem(term) 
        for term in cleaned.split()
        if term not in set(stop_words)
    )

print(preprocess_text(example))


print((processed == raw_text.apply(preprocess_text)).all())

