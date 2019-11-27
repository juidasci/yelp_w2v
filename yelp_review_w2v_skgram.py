
print('Load Package')

import numpy as np
import pandas as pd
import json
import string
import gensim

print('Read File')
# Read File
# text_preprocess csv
df = pd.read_csv(
    './w2vd/wddata/yelp_data/preprocess_review.csv', sep=',', header=0)
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

print('Skip-gram')

# Skip-gram
size = 300
mincount = 2
win = 4
meth = 1   # sg=0 CBOW ; sg=1 skip-gram
workers = 10

print('Build model')
# build word2vec
model_s = word2vec.Word2Vec(
    size=size, min_count=mincount, window=win, sg=meth, workers=workers)

# word and ID mapping
model_s.build_vocab(documents)

print('Train model')
# train word2vec model
model_s.train(sentences=documents, total_examples=len(
    documents), epochs=model_s.iter)

# train word2vec model ; shuffle data every epoch
# for i in range(20):
#    random.shuffle(documents)
#    model_s.train(documents, total_examples = len(documents), epochs = 1)

print('Skip-gram OK!')

print('Save model')
# save model
model_s.save("./models/yelp_skipgram")
print('Save Skip-gram model OK!')


import sys

sys.exit()
