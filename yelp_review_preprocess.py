
print('Load Package')

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


#print('-Read CSV-')
# read csv
#df = pd.read_csv('./wddata/yelp_dataset/review.csv', sep = ',', header = 0)


print('-Read Json-')
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

print('Read Json OK!')


print('-Before - text length-')
# function which returns the length of text
def length(text):
    return len(text)

df['before_length'] = df['text'].apply(length)

print('Before - text length OK!')


print('-Convert Accented Characters-')
# Convert Accented Characters
# remove accented characters from text, e.g. café
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

df['text'] = df['text'].apply(lambda x: remove_accented_chars(x))

print('Convert Accented Characters OK!')


print('-Remove Number-')
# Remove Number
def remove_num(text):
    text = re.sub('[0-9]', '', text)
    return text

df['text'] = df['text'].apply(lambda x: remove_num(x))

print('Remove Number OK!')


print('-Expand Contractions-')
'''
# Expand Contractions
# expand shortened words, e.g. don't to do not
import gensim.downloader as api
from pycontractions import Contractions
# Choose model accordingly for contractions function

print('load model')

model = api.load('glove-twitter-25')
# model = api.load('glove-twitter-100')
# model = api.load('word2vec-google-news-300')

print('model setting')
cont = Contractions(kv_model=model)
cont.load_models()

print('model setting OK!')

print('def Expand Contractions function')

def expand_contractions(text):
    text = list(cont.expand_texts([text], precise=True))[0]
    return text
'''

from contractions import CONTRACTION_MAP

print('-Def Expand Contractions function-')


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format(
        '|'.join(contraction_mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    return re.sub("'", "", expanded_text)

print('-Expand Contractions Do-')
df['text'] = df['text'].apply(lambda x: expand_contractions(x))

print('Expand Contractions OK!')


print('-Remove Punctuation-')
# Remove Punctuation
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

df['text'] = df['text'].apply(lambda x: remove_punctuation(x))

print('Remove Punctuation OK!')


print('-Lowercase-')
# Lowercase
def lower(text):
    text = [word.lower() for word in text.split()]
    return " ".join(text)

df['text'] = df['text'].apply(lambda x: lower(x))

print('Lowercase OK!')


print('-Remove 2-gram-')
# Remove 2-gram
def remove_2n(text):
    regex = re.compile(r'\w{3,}')
    text = regex.findall(text)
    return " ".join(text)

df['text'] = df['text'].apply(lambda x: remove_2n(x))

print('Remove 2-gram OK!')


print('-Lemmatization-')
# Lemmatization
lem = WordNetLemmatizer()

def lemmating(text):
    text = [lem.lemmatize(word, 'v') for word in text.split()]
    return " ".join(text)

df['text'] = df['text'].apply(lemmating)

print('Lemmatization OK!')


print('-Stemming-')
# Stemming
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")  # 需定義語言
# stemmer = LancasterStemmer()

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

df['text'] = df['text'].apply(stemming)

print('Stemming OK!')


'''
print('-Remove Stopwords-')
# Remove Stopwords
sw = stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text.split() if word not in stopwords.words('english')]
    return " ".join(text)

df['text'] = df['text'].apply(remove_stopwords)

print('Remove Stopwords OK!')
'''

print('-After - text length-')
# Text Length -  After Text Clear
df['after_clear_length'] = df['text'].apply(length)

print('After - text length OK!')


print('-Save File-')
df.to_pickle('/home/jasonch/mycode/school/w2vd/wddata/yelp_data/preprocess_review_nostopwd.pkl')
df.to_csv('/home/jasonch/mycode/school/w2vd/wddata/yelp_data/preprocess_review_nostopwd.csv', index = False)

print('Save File OK!')

import sys

sys.exit()