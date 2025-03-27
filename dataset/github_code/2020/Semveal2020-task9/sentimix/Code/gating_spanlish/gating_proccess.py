from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict


def parse(f):
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    # print(lines)
    X_train = []
    Y_train = []
    linelen=[]
    labels = ['negative', 'neutral', 'positive']
    i=1
    # Processes individual lines
    for line in lines:
        i=i+1
        # Seperator for the current dataset. Currently '\t'.
        line = line.split('\t')
        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = line[0].split()
        linelen.append(len(tokenized_lines))
        X_train.append(tokenized_lines)
        if line[1] == labels[0]:
            Y_train.append(0)
        elif line[1] == labels[1]:
            Y_train.append(1)
        elif line[1] == labels[2]:
            Y_train.append(2)
        else:
            print(i)

    # Converts Y_train to a numpy array
    Y_train = np.asarray(Y_train)
    print(len(X_train))
    print(Y_train.shape[0])
    maxlen=np.max(linelen)
    print('max sentence length:',np.max(linelen))
    assert (len(X_train) == Y_train.shape[0])

    return [X_train, Y_train,maxlen]


def parsetest(f):
    # Reads the files and splits data into individual lines
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    X_test = []
    id_test = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split('\t')
        tokenized_lines = line[1].split()
        X_test.append(tokenized_lines)
        id_test.append(line[0])

    return [X_test, id_test]



def build_data_train_test(data_train, data_test, data_dev):
    """
    Loads data and process data into index
    """
    vocab = defaultdict(float)

    # Pre-process train data set
    for i in range(len(data_train)):
        rev = data_train[i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1

    for i in range(len(data_test)):
        rev = data_test[i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1

    for i in range(len(data_dev)):
        rev = data_dev[i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
    print('kkkkkkkkk',vocab)
    return vocab


def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    logging.info('unk words: %d' % (unk_words))
    return word_vecs



def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []

    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x



def make_idx_data(X_train, X_test, X_dev, word_idx_map, maxlen=50):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_train, x_test, x_dev = [], [], []
    for line in X_train:
        sent = get_idx_from_sent(line, word_idx_map)
        x_train.append(sent)
    for line in X_test:
        sent = get_idx_from_sent(line, word_idx_map)
        x_test.append(sent)
    for line in X_dev:
        sent = get_idx_from_sent(line, word_idx_map)
        x_dev.append(sent)

    x_train = sequence.pad_sequences(np.array(x_train), maxlen=maxlen)
    print("X_train:", x_train.shape)
    x_test = sequence.pad_sequences(np.array(x_test), maxlen=maxlen)
    x_dev = sequence.pad_sequences(np.array(x_dev), maxlen=maxlen)

    return [x_train, x_test, x_dev]

if __name__ == '__main__':
    ftrain = open("E:/Sub-word-LSTM(sentimix)/dataprocess/spanlish/Data/train_user_hashtag_hindi_nochar.tsv", 'r',
                  encoding='UTF-8')
    train = parse(ftrain)
    X_train = train[0]
    X_train = np.asarray(X_train)
    y_train = train[1]
    maxlen = train[2]

    fdev = open("E:/Sub-word-LSTM(sentimix)/dataprocess/spanlish/Data/dev_user_hashtag_hindi_nochar.tsv", 'r',
                  encoding='UTF-8')
    dev = parse(fdev)
    X_dev = dev[0]
    X_dev = np.asarray(X_dev)
    y_dev = dev[1]
    maxlendev = dev[2]

    ftest = open("E:/Sub-word-LSTM(sentimix)/dataprocess/spanlish/Data/test_user_hashtag_hindi_nochar.tsv", 'r',
                encoding='UTF-8')

    test = parsetest(ftest)
    X_test = test[0]
    X_test = np.asarray(X_test)
    yid_test = test[1]

    vocab = build_data_train_test(X_train, X_test,X_dev)
    vocsize = len(vocab)
    print('vocabsize:', vocsize)

    model_file = os.path.join('../vector', 'glove_model.txt')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    w2v = load_bin_vec(model, vocab)
    print('word embeddings loaded!')
    print('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=model.vector_size)
    x_train, x_dev, x_test = make_idx_data(X_train, X_dev, X_test, word_idx_map, maxlen=30)

    pickle_file = os.path.join('../pickle/span/gating_wordembed.pickle3')
    pickle.dump([W, word_idx_map, vocab, x_train, x_test, x_dev, y_train, y_dev], open(pickle_file, 'wb'))