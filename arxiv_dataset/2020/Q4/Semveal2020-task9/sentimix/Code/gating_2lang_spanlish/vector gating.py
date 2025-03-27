import numpy as np
import keras
import os
import pandas as pd
import logging
import h5py
import tensorflow as tf
import pickle
import gensim
from copy import deepcopy
from keras.utils import to_categorical
from collections import defaultdict
from keras.models import Model
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import metrics
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Dense, Input, Lambda, merge, dot, Subtract
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, Flatten, CuDNNGRU, Input, SpatialDropout1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import matplotlib.pyplot as plt
from Capsule_net import Capsule
from Attention_layer import AttentionM
from datetime import datetime
from time import time
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score, recall_score

from keras.utils import np_utils
from MyNormalizer import token


Masterdir = 'E:/Sub-word-LSTM(sentimix)/'
Modeldir = 'Pretrained_Models/'
Featuredir = 'Features/'
train_file = 'dataprocess/spanlish/Data/train_user_hashtag_hindi_nochar.tsv'
dev_file = 'dataprocess/spanlish/Data/dev_user_hashtag_hindi_nochar.tsv'
test_file = 'dataprocess/spanlish/Data/test_user_hashtag_hindi_nochar.tsv'
exp_details = 'coed-mix(word)'

SEPERATOR = '\t'
DATA_COLUMN = 0
DATAtest_COLUMN = 1
LABEL_COLUMN = 1
id_COLUMN = 0
LABELS = ['negative', 'neutral', 'positive']
labels = [0, 1, 2]  # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num = {}
mapping_num2char = {}
wordMAXLEN = 30
charMAXLEN = 10

# LSTM Model Parameters
# Embedding
MAX_FEATURES = 0
char_embedding_size = 150

# Training
batch_size = 128
number_of_epochs = 20
numclasses = 3


########################################################
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


class AttentionM(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix.

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM()(enc)

    """

    def __init__(self, **kwargs):
        super(AttentionM, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionM, self).get_config()


def parse(Masterdir, filename, seperator, datacol, labelcol, labels):
    """
    Purpose -> Data I/O
    Input   -> Data file containing sentences and labels along with the global variables
    Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
    """
    # Reads the files and splits data into individual lines
    f = open(Masterdir + filename, 'r', encoding='UTF-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    print(lines)

    X_train = []
    Y_train = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator)
        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = token(line[datacol])

        X_train.append(tokenized_lines)
        # print(X_train)

        # Appends labels
        if line[labelcol] == labels[0]:
            Y_train.append(0)
        if line[labelcol] == labels[1]:
            Y_train.append(1)
        if line[labelcol] == labels[2]:
            Y_train.append(2)

    # Converts Y_train to a numpy array
    Y_train = np.asarray(Y_train)
    # print(Y_train)

    assert (len(X_train) == Y_train.shape[0])

    return [X_train, Y_train]


def parsetest(Masterdir, filename, seperator, datacol, idlcol):

    # Reads the files and splits data into individual lines
    f = open(Masterdir + filename, 'r', encoding='UTF-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    print(lines)

    X_test = []
    id_test = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator)
        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = token(line[datacol])
        X_test.append(tokenized_lines)
        id_test.append(line[idlcol])

    return [X_test, id_test]



def char2num(mapping_n2c, mapping_c2n, trainwords, wordmaxlen, charmaxlen):
    """
    Purpose -> Convert characters to integers, a unique value for every character
    Input   -> Training data (In list of lists format) along with global variables
    Output  -> Converted training data along with global variables
    """

    errors = 0
    maxlen_char_word = 0
    X_train = []
    for line in trainwords:
        char_list = []
        for word in line:
            charlist = list(word)
            if len(charlist) > maxlen_char_word:
                maxlen_char_word = len(char_list)
            char_list.append(charlist)
            # char_list.append(' ')
        # print(char_list)
        # print(char_list) - Debugs the character list created
        X_train.append(char_list)
    print('maxlen_char_word:', maxlen_char_word)
    # print(X_train)
    # Creates a list of all characters present in the dataset
    allchars = []
    for line in X_train:

        for word in line:

            # for char in word:
            # print(char)
            try:
                allchars = set(allchars + word)
                # print('kj',allchars)
                allchars = list(allchars)
            except:
                errors += 1

    print(errors)  # Debugging
    print('allcha', allchars)  # Debugging

    # Creates character dictionaries for the characters
    charno = 0
    for char in allchars:
        mapping_char2num[char] = charno
        mapping_num2char[charno] = char
        charno += 1

    assert (len(allchars) == charno)  # Checks

    # Converts the data from characters to numbers using dictionaries
    X_train1 = []

    for line in X_train:
        linelist = []
        for word in line:
            char_list = []
            for char in word:
                char_list.append(mapping_char2num[char])
            # char_list=sequence.pad_sequences(np.array(char_list),maxlen=wordmaxlen)
            linelist.append(char_list)

        X_train1.append(linelist)
    print('XX',X_train1)
     # pad_char_all = np.zeros(((17000,30,10)))
    pad_char_all=[]
    for i,line in enumerate(X_train1):
        while len(line) < 30:
            line.insert(0, [])
        while len(line) >30:
            line=line[:30]
        pad_senc =pad_sequences(line, maxlen=10)
        pad_senc=pad_senc.tolist()
        s=[]
        for w in pad_senc:
            x = []
            for c in w:
                x.append(c)
            s.append(x)

        # pad_char_all[i]=pad_senc
        pad_char_all.append(s)
    i=0
    for line in pad_char_all:
        i=i+1
        if len(line)!=30:
            print("**************")
            print(i)



    pad_char_all = np.asarray(pad_char_all)
    print(pad_char_all.shape)

    charno = charno + 1
    maxlen_char=10
    return [pad_char_all, mapping_num2char, mapping_char2num, charno, maxlen_char]


def build_data_train_test(data_train, data_test):
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


def make_idx_data(X_train, X_dev, X_test, word_idx_map, maxlen=30):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_train, x_test, x_dev = [], [], []
    for line in X_train:
        sent = get_idx_from_sent(line, word_idx_map)
        x_train.append(sent)
    for line in X_dev:
        sent = get_idx_from_sent(line, word_idx_map)
        x_dev.append(sent)
    for line in X_test:
        sent = get_idx_from_sent(line, word_idx_map)
        x_test.append(sent)

    x_train = sequence.pad_sequences(np.array(x_train), maxlen=maxlen)
    print("X_train:", x_train.shape)
    x_dev = sequence.pad_sequences(np.array(x_dev), maxlen=maxlen)
    x_test = sequence.pad_sequences(np.array(x_test), maxlen=maxlen)

    return [x_train, x_dev, x_test]


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


def save_model(Masterdir, filename, model):
    """
    Purpose -> Saves Keras model files to the given directory
    Input   -> Directory and experiment details to be saved and trained model file
    Output  -> Nil
    """
    # Referred from:- http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model.save_weights(Masterdir + Modeldir + 'LSTM_' + filename + '_weights.h5')
    json_string = model.to_json()
    f = open(Masterdir + Modeldir + 'LSTM_' + filename + '_architecture.json', 'w')
    f.write(json_string)
    f.close()



if __name__ == '__main__':
    train = parse(Masterdir, train_file, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    dev = parse(Masterdir, dev_file, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    test = parsetest(Masterdir, test_file, SEPERATOR, DATAtest_COLUMN, id_COLUMN)
    # print(test[0],test[1])
    X_train = train[0]
    X_train = np.asarray(X_train)
    y_train = train[1]

    print('Creating character dictionaries and format conversion in progess...')
    train = char2num(mapping_num2char, mapping_char2num, X_train, wordMAXLEN, charMAXLEN)
    mapping_num2char = train[1]
    mapping_char2num = train[2]
    MAX_FEATURES = train[3]
    maxlen_char_word = train[4]
    X_chartrain = train[0]
    y_chartrain = np.asarray(y_train).flatten()

    X_dev = dev[0]
    X_dev = np.asarray(X_dev)
    y_dev = dev[1]

    X_test = test[0]
    id_test = test[1]
    dev = char2num(mapping_num2char, mapping_char2num, X_dev, wordMAXLEN, charMAXLEN)
    X_chardev = dev[0]
    y_chardev = np.asarray(y_dev).flatten()

    test = char2num(mapping_num2char, mapping_char2num, X_test, wordMAXLEN, charMAXLEN)

    X_chartest = test[0]
    # y_chartest = y_chardev
    print(X_chartrain)


    pickle_file = os.path.join('../pickle/span/hi_es_user_hastag_glove_wik_gating_2lang.pickle3')

    W, word_idx_map, vocab, x_train, x_test, x_dev, y_train, y_dev ,yid_test= pickle.load(open(pickle_file, 'rb'))
    max_features = W.shape[0]

    'Creating charembedding...'
    char_input = Input(shape=[wordMAXLEN, maxlen_char_word], dtype='int32', name='char_input')  # (None, 36, 15)
    char_embed = Embedding(input_dim=MAX_FEATURES, output_dim=char_embedding_size,
                           embeddings_initializer='lecun_uniform',
                           input_length=[None, maxlen_char_word], mask_zero=False, name='char_embedding')(char_input)

    s = char_embed.shape  # (?,36,15,150)
    print(s)
    char_embed = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], char_embedding_size)))(char_embed)
    print('char_embedding', char_embed)
    fwd_state = LSTM(150)(char_embed)
    print(fwd_state)
    bwd_state = LSTM(150, go_backwards=True)(char_embed)
    char_embed = Concatenate(axis=1)([fwd_state, bwd_state])
    print('char_lstm:', char_embed.shape)
    print(s[1])
    char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 150]))(char_embed)
    print('char_lstm(reshape):', char_embed.shape)  # (?,36,300)

    'word_embedding'
    word_embed_dim = 300
    word_input = Input(shape=[wordMAXLEN], dtype='int32', name='input')
    embed = Embedding(input_dim=max_features, output_dim=word_embed_dim, input_length=wordMAXLEN,
                      weights=[W], mask_zero=False, name='embedding', trainable=False)(word_input)
    print(embed.shape)
    bilstm = Bidirectional(LSTM(150, recurrent_dropout=0.25,return_sequences=True))(embed)

    'vector gating'
    linembed = Dense(300, activation='sigmoid')(bilstm)
    print('lineembed', linembed)
    # gating = Lambda(lambda x,y: (1.0 - y)*x+y*x)(embed,linembed)
    # gating=(1.0 - linembed)*embed+linembed*embed
    # gating=MyLayer(300)(linembed,embed,char_embed)
    t = Lambda(lambda x: K.ones_like(x, dtype='float32'))(linembed)
    merged1 = merge([linembed, char_embed], name='merged1', mode='mul')
    sub = Subtract()([t, linembed])
    merged2 = merge([embed, sub], name='merged2', mode='mul')
    gating = merge([merged1, merged2], name='gating', mode='sum')

    # gating = (1.0 - linembed) *embed +linembed * char_embed

    enc = Bidirectional(LSTM(150, recurrent_dropout=0.25,return_sequences=True))(gating)
    att = AttentionM()(enc)
    # fc = Dense(64, activation="relu")(att)
    # dropout = Dropout(0.25)(fc)
    output = Dense(3, activation='softmax')(att)

    model = Model(inputs=[word_input, char_input], outputs=output, name='output')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'],
                  )
    print(model.summary())

    history = LossHistory()

    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    es = EarlyStopping(monitor='val_loss', patience=2)
    mc = ModelCheckpoint(log_dir + '\\span_gating_'+'-{epoch:02d}-ACC{val_acc:.4f}.h5',
                         monitor='val_loss', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)
    print(X_chartrain.shape)
    print(x_train.shape)
    print(X_chardev.shape )
    print(x_dev.shape)
    print()

    y_chartrain = to_categorical(y_chartrain)
    y_chardev = to_categorical(y_chardev)
    model.fit([x_train,X_chartrain ], y_chartrain,
              batch_size=batch_size,
              shuffle=True,
              nb_epoch=number_of_epochs,
              validation_data=([x_dev,X_chardev ], y_chardev),
               callbacks=[history, es, mc, tb])
    history.loss_plot('epoch')

    numclasses = 3




    print('Saving experiment...')
    save_model(Masterdir, exp_details, model)
    # model.save("../Pretrained_models/vector_gating_hindi.h5")

    y_pred = model.predict([ x_test,X_chartest], batch_size=batch_size)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)


    def accuracy(original, predicted):
        print("F1 score is: " + str(f1_score(original, predicted, average='macro')))
        print("recall score is: " + str(recall_score(original, predicted, average='macro')))
        scores = confusion_matrix(original, predicted)
        print(scores)
        print(np.trace(scores) / float(np.sum(scores)))

    # accuracy(y_chartest, y_pred)
    # target_names = ['class 0', 'class 1','class 2']
    # print(metrics.classification_report(y_chartest, y_pred, target_names=target_names))



result_file_name = './answer1.txt'
answer=[]
for i in range(len(y_pred)):
    if y_pred[i] == labels[0]:
        answer.append(LABELS[0])
    if y_pred[i] == labels[1]:
        answer.append(LABELS[1])
    if y_pred[i] == labels[2]:
        answer.append(LABELS[2])

print(answer)
print(id_test)
def write_as_test(result_file_name,answer, id_test):
    with open(result_file_name, 'w', encoding='utf8') as my_file:
        my_file.write('Uid'+","+'Sentiment\n')
        for i, text in enumerate(answer):
            my_file.write('%s,%s\n'% (id_test[i], text))

write_as_test(result_file_name,answer,id_test)