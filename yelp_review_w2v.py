
import numpy as np
import pandas as pd
import json
import string
import nltk
import gensim

print('Read File')
# Read File
# text_preprocess csv
df = pd.read_csv('./wddata/yelp_dataset/preprocess_review.csv', sep = ',', header = 0)
#df.head(10)


# text_preprocess pickle
#df = pd.read_pickle('./wddata/yelp_dataset/preprocess_review.pkl', sep = ',', header = 0)
#df.head(10)

print('Read OK!')

# Coversion format
def read_questions(row, column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))

documents = []
for index, row in df.iterrows():
    documents.append(read_questions(row, 'text')

#documents[0:3]

print('Coversion format OK!')

print('Load gensim package')


# word2vec
from gensim.models import word2vec
import random

print('CBOW')

# CBOW
size = 250
mincount = 3
win = 4
meth = 0   # sg=0 CBOW ; sg=1 skip-gram
workers = 10

# build word2vec
model_c = word2vec.Word2Vec(size = size, min_count = mincount, window = win, sg = meth, workers = workers)

# 建立字與 ID 的 mapping
model_c.build_vocab(documents) 

# train word2vec model
model_c.train(sentences = documents, total_examples = len(documents), epochs = model_c.iter)

# train word2vec model ; shuffle data every epoch
#for i in range(20):
#    random.shuffle(documents)
#    model_c.train(documents, total_examples = len(documents), epochs = 1)


## save model
model_c.save("./models/CBOW_yelp")
print('Save CBOW model OK!')

print('Skip-gram')

# Skip-gram
size = 250
mincount = 3
win = 4
meth = 1   # sg=0 CBOW ; sg=1 skip-gram


# build word2vec
model_s = word2vec.Word2Vec(size = size, min_count = mincount, window = win, sg = meth)

# 建立字與 ID 的 mapping
model_s.build_vocab(documents)

# train word2vec model ; shuffle data every epoch
for i in range(20):
    random.shuffle(documents)
    model_s.train(documents, total_examples = len(documents), epochs = 1)

print('Skip-gram OK!')

## save model
model_s.save("./models/skipgram_yelp")
print('Save Skip-gram model OK!')