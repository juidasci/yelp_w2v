
print('Load Package')

import numpy as np
import pandas as pd
import json
import string
import gensim

print('Read File')
# Read File
# text_preprocess csv
df = pd.read_csv('./wddata/yelp_data/preprocess_review.csv', sep=',', header=0)
# df.head(10)


# text_preprocess pickle
#df = pd.read_pickle('./wddata/yelp_dataset/preprocess_review.pkl', sep = ',', header = 0)
# df.head(10)

print('Read OK!')

print('-Coversion format-')
# Coversion format


def read_questions(row, column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))


print('Coversion format Do')
documents = []
for index, row in df.iterrows():
    documents.append(read_questions(row, 'text'))

# documents[0:3]

print('Coversion format OK!')


print('Load gensim package')

# word2vec
from gensim.models import word2vec
import random

print('CBOW')

# CBOW
size = 300
mincount = 2
win = 4
meth = 0   # sg=0 CBOW ; sg=1 skip-gram
workers = 10

print('Build model')
# build word2vec
model_c = word2vec.Word2Vec(
    size=size, min_count=mincount, window=win, sg=meth, workers=workers)

# word and ID mapping
model_c.build_vocab(documents)

print('Train model')
# train word2vec model
model_c.train(sentences=documents, total_examples=len(
    documents), epochs=model_c.iter)

# train word2vec model ; shuffle data every epoch
# for i in range(20):
#    random.shuffle(documents)
#    model_c.train(documents, total_examples = len(documents), epochs = 1)

print('CBOW OK!')

print('Save model')
# save model
model_c.save("./models/yelp_cbow")
print('Save CBOW model OK!')


import sys

sys.exit()

# exit()
