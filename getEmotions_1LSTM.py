#loading libraries for LR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from keras.regularizers import L1L2
#from sklearn import cross_validation
# Create the model
from keras.layers import Dropout

#loading libraries for scikit learn, nlp, db, plot and matrix.
import sqlite3
import pdb
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree

from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer


# using the SQLite Table to read data.
con = sqlite3.connect('./final.sqlite') 

#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
final = pd.read_sql_query("""
SELECT *
FROM Reviews
""", con) 

print(final.head(2))

# To randomly sample the data and sort based on time before doing train/ test split.

num_points = 100000

# used to format headings 
bold = '\033[1m'
end = '\033[0m'

# you can use random_state for reproducibility
# sampled_final = final.sample(n=num_points, random_state=2)

#Sorting data according to Time in ascending order
sorted_final = final.sort_values('Time', axis=0, 
                ascending=True, inplace=False, kind='quicksort', na_position='first')

final = sorted_final.tail(num_points)

# fetching the outcome class 
y = final['Score'] 

def class2num(response):
    if (response == 'positive'):
        return 1
    else:
        return 0

y_bin = list(map(class2num, y))


# Build vocabulary and make sorted Word-Frequency  table

import re
import collections

words = []

# Take in all the reviews
reviewText = final['CleanedText'].values

# Create a list of all words in the db
# iterate for each review/sentence
for sent in reviewText: 
    sent = str(sent, 'utf-8')
    sent = re.sub("[^\w]", " ",  sent).split()

    for word in sent:
        words.append(word)

# to create a dict of word:frequency
counter=collections.Counter(words)

# the keys in the dictionary contains unique words.
# set of unique words represents the vocabulary
vocab = counter.keys()

# sort the words based on frequency
sortedWords_by_frequency = sorted(
        counter.items(), key=lambda kv: kv[1], reverse=True)

# print(sortedWords_by_frequency)

# select 'n' words having highest frequency
top_words_count = 50000
top_words = sortedWords_by_frequency[:top_words_count]

# Encode each review based on indices and split into test/train

# Doing indexing of top words and storing it in a dictionary
top_words_dict = {}
index = 1
for word_freq in top_words:
    top_words_dict[word_freq[0]] =  index
    index = index + 1

import json

with open('top_words_dict.txt', 'w') as file:
     file.write(json.dumps(top_words_dict))

# Convert reviews as list of indices of words
indexedReview = []
for idx, sent in enumerate(reviewText): 
    sent = str(sent, 'utf-8')
    sent = re.sub("[^\w]", " ",  sent).split()
        
    wordIndices = []
    for word in sent:
        wordIndex = top_words_dict.get(word, -1)
        if (wordIndex > 0):
            wordIndices.append(wordIndex)
        
    indexedReview.append(wordIndices)
indexedReview = indexedReview


# Split the encoded Amazon reviews to train/test in LSTM
X_train, X_test, y_train, y_test = train_test_split(
                    indexedReview, y_bin, test_size=0.3, random_state=42)



# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)

# Truncate and/or pad input sequences
max_review_length = 600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_test)

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

def trainModel(model):

    print(X_train.size)
    history = model.fit(numpy.array(X_train), numpy.array(y_train), validation_split=0.33, epochs=30, batch_size=64)

    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # Final evaluation of the model
    scores = model.evaluate(numpy.array(X_test), numpy.array(y_test), verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save('getEmotions_1LSTM.h5')

# Create the model
# regularizers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

embedding_vector_length = 32
model1 = Sequential()
model1.add(Embedding(top_words_count, embedding_vector_length, input_length=max_review_length))
model1.add(Dropout(0.75))
model1.add(LSTM(100, bias_regularizer=L1L2(l1=0.0, l2=0.05)))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
#Refer: https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model

trainModel(model=model1)
