
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import json
import re
import string
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import gensim


# read csv
#df = pd.read_csv('./wddata/yelp_dataset/review.csv', sep = ',', header = 0)


# read json
def init_ds(json):
    ds = {}
    keys = json.keys()
    for k in keys:
        ds[k] = []
    return ds, keys


def read_json(file):
    dataset = {}
    keys = []
    with open(file) as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count == 0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])
        return pd.DataFrame(dataset)


df = read_json('./wddata/yelp_data/yelp_dataset/review.json')


df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df.date.dt.year


# function which returns the length of text
def length(text):
    return len(text)


df['before_length'] = df['text'].apply(length)


# Convert Accented Characters
# remove accented characters from text, e.g. café
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


df['text'] = df['text'].apply(lambda x: remove_accented_chars(x))


# Remove Number
def remove_num(text):
    text = re.sub('[0-9]', '', text)
    return text


df['text'] = df['text'].apply(lambda x: remove_num(x))


# Expand Contractions
# expand shortened words, e.g. don't to do not
import gensim.downloader as api
from pycontractions import Contractions
# Choose model accordingly for contractions function
model = api.load('glove-twitter-25')
# model = api.load('glove-twitter-100')
# model = api.load('word2vec-google-news-300')
cont = Contractions(kv_model=model)
cont.load_models()


def expand_contractions(text):
    text = list(cont.expand_texts([text], precise=True))[0]
    return text


df['text'] = df['text'].apply(lambda x: expand_contractions(x))


# Remove Punctuation
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct


df['text'] = df['text'].apply(lambda x: remove_punctuation(x))


# Lowercase
def lower(text):
    text = [word.lower() for word in text.split()]
    return " ".join(text)


df['text'] = df['text'].apply(lambda x: lower(x))


# Remove 2-gram
def remove_2n(text):
    regex = re.compile(r'\w{3,}')
    text = regex.findall(text)
    return " ".join(text)


df['text'] = df['text'].apply(lambda x: remove_2n(x))


# Lemmatization
lem = WordNetLemmatizer()


def lemmating(text):
    text = [lem.lemmatize(word, 'v') for word in text.split()]
    return " ".join(text)


df['text'] = df['text'].apply(lemmating)


# Stemming
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")  # 需定義語言
# stemmer = LancasterStemmer()


def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)


df['text'] = df['text'].apply(stemming)


# Remove Stopwords
sw = stopwords.words('english')


def remove_stopwords(text):
    text = [word for word in text.split() if word not in stopwords.words('english')]
    return " ".join(text)


df['text'] = df['text'].apply(remove_stopwords)


# Text Length -  After Text Clear
df['after_clear_length'] = df['text'].apply(length)


df.to_pickle('./wddata/yelp_dataset/preprocess_review.pkl')
df.to_csv('./wddata/yelp_dataset/preprocess_review.csv', index=False)
