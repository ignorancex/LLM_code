from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import tensorflow as tf
import logging
from keras.utils import to_categorical
import pickle
import numpy as np
import keras
from datetime import datetime
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding,Convolution1D,MaxPooling1D, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, TimeDistributed
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split



# Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 0
LABEL_COLUMN = 1
LABELS = ['negative', 'neutral', 'positive']  # 0 -> Negative, 1-> Neutral, 2-> Positive
# LABELS = ['0', '1', '2']  # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num = {}
mapping_num2char = {}
MAXLEN = 180
char_embedding_size = 300
# LSTM Model Parameters
# Embedding
MAX_FEATURES = 0
embedding_size = 128
# Convolution
filter_length = 3
nb_filter = 128
pool_length = 3
# LSTM
lstm_output_size = 128
# Training
batch_size = 128
number_of_epochs = 50
numclasses = 3
test_size = 0.2


from keras.utils import np_utils

maxlen = 50
batch_size = 128
nb_epoch = 50
hidden_dim = 128

kernel_size = 3
nb_filter = 60

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def parse(seperator, datacol, labelcol, labels):
    """
    Purpose -> Data I/O
    Input   -> Data file containing sentences and labels along with the global variables
    Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
    """
    # Reads the files and splits data into individual lines
    f = open('E:/Sub-word-LSTM(sentimix)/dataprocess/hindi/data/pre/train_nouser_hashtag_hindi.tsv', 'r', encoding='UTF-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]

    X_train = []
    Y_train = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator)
        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = line[datacol].split()
        # print(tokenized_lines)

        # Creates character lists
        char_list = []
        for words in tokenized_lines:
            for char in words:
                char_list.append(char)
            char_list.append(' ')
        X_train.append(char_list)

        # Appends labels
        if line[labelcol] == labels[0]:
            Y_train.append(0)
        if line[labelcol] == labels[1]:
            Y_train.append(1)
        if line[labelcol] == labels[2]:
            Y_train.append(2)

    # Converts Y_train to a numpy array
    Y_train = np.asarray(Y_train)

    assert (len(X_train) == Y_train.shape[0])
    print(X_train)
    return [X_train, Y_train]


def convert_char2num(mapping_n2c, mapping_c2n, trainwords, maxlen):
    """
    Purpose -> Convert characters to integers, a unique value for every character
    Input   -> Training data (In list of lists format) along with global variables
    Output  -> Converted training data along with global variables
    """
    allchars = []
    maxchar = []
    errors = 0

    # Creates a list of all characters present in the dataset
    for line in trainwords:
        maxchar.append(len(line))

        try:
            allchars = set(allchars + line)
            allchars = list(allchars)
        except:
            errors += 1

    # print(maxchar)
    print('maxchar:', max(maxchar))
    print('erros', errors)  # Debugging
    print(allchars)  # Debugging

    # Creates character dictionaries for the characters
    charno = 0
    for char in allchars:
        mapping_char2num[char] = charno
        mapping_num2char[charno] = char
        charno += 1

    assert (len(allchars) == charno)  # Checks

    # Converts the data from characters to numbers using dictionaries
    X_train = []

    for line in trainwords:

        char_list = []
        for letter in line:
            char_list.append(mapping_char2num[letter])
        # print(no) -- Debugs the number mappings
        X_train.append(char_list)
    print('X:', X_train)

    # print(mapping_char2num)
    # print(mapping_num2char)
    # Pads the X_train to get a uniform vector
    # TODO: Automate the selection instead of manual input
    X_train = sequence.pad_sequences(X_train[:], maxlen=maxlen)
    # print(X_train)
    return [X_train, mapping_num2char, mapping_char2num, charno]



if __name__ == '__main__':

    out = parse( SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    # print('out:', out)
    # outtest = parse(Masterdir, testfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    X_train = out[0]
    y_train = out[1]
    print('Parsing complete!')

    print('Creating character dictionaries and format conversion in progess...')
    out1 = convert_char2num(mapping_num2char, mapping_char2num, X_train, MAXLEN)
    mapping_num2char = out1[1]
    mapping_char2num = out1[2]
    MAX_FEATURES = out1[3]
    X_train = np.asarray(out1[0])
    y_train = np.asarray(y_train).flatten()

    print('Splitting data into train and test...')
    char_train, char_test, chary_train, chary_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1)


    pickle_file = os.path.join('./pickle/hindi/align_hi_en_nouser_hastag.pickle3')

    W, word_idx_map, vocab,wordx_train,wordx_test,wordy_train,wordy_test = pickle.load(open(pickle_file, 'rb'))

    max_features = W.shape[0]
    num_features = W.shape[1]  # 400

    # Keras Model
    # char_embedding
    char_input = Input(shape=[MAXLEN,], dtype='int32', name='char_input')  # (None, 36, 15)
    char_embed = Embedding(input_dim=MAX_FEATURES, output_dim=char_embedding_size,
                           embeddings_initializer='lecun_uniform',
                           input_length=[MAXLEN], mask_zero=False, name='char_embedding')(char_input)
    convolution = Convolution1D(filters=nb_filter,
                                filter_length=filter_length,
                                padding='valid',
                                activation='relu',
                                strides=1
                                )(char_embed)
    maxpooling = MaxPooling1D(pool_length=pool_length)(convolution)
    lstm = LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(maxpooling)
    char_lstm1 = LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=False)(lstm)

    # word_embedding
    word_input = Input(shape=(50,), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=False,
                         weights=[W], trainable=False)(word_input)
    embedded = Dropout(0.25)(embedded)
    bilstm = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25,return_sequences=True)) (embedded)
    word_bilstm1 = Bidirectional(LSTM(hidden_dim//2,recurrent_dropout=0.25))(bilstm)

    #concat
    x = keras.layers.Concatenate(axis=1)([char_lstm1, word_bilstm1])
    # GRU
    # hidden = GRU(hidden_dim, recurrent_dropout=0.25)(embedded)

    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=[word_input,char_input], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = LossHistory()
    es = EarlyStopping(monitor='val_loss', patience=10)

    y_test1 = to_categorical(wordy_test)
    y_train = to_categorical(y_train)
    model.fit([wordx_train,char_train], y_train, validation_data=[wordx_test,char_test],
              batch_size=batch_size, epochs=nb_epoch, verbose=1,
              callbacks=[history,es])
    history.loss_plot('epoch')

    y_pred1 = model.predict([wordx_test,char_test], batch_size=batch_size)
    print('y_pred:',y_pred1)
    y_pred = np.argmax(y_pred1, axis=1)


    def accuracy(original, predicted):
        print("F1_macro score is: " + str(f1_score(original, predicted, average='macro')))
        print("F1_weighted score is: " + str(f1_score(original, predicted, average='weighted')))
        print("recall score is: " + str(recall_score(original, predicted, average='macro')))
        scores = confusion_matrix(original, predicted)
        print(scores)
        print(np.trace(scores) / float(np.sum(scores)))

    accuracy(y_test, y_pred)