import numpy as np
import keras
import os
import pandas as pd
import logging
import h5py
import tensorflow as tf
import pickle
from keras.layers.wrappers import Bidirectional, TimeDistributed
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score, recall_score

from keras.utils import np_utils
from MyNormalizer import token

################# GLOBAL VARIABLES #####################
# Filenames
# TODO: Add to coding conventions that directories are to always end with '/'
Masterdir = 'E:/Sub-word-LSTM(sentimix)/'
Datadir = 'Data/'
Modeldir = 'Pretrained_Models/'
Featuredir = 'Features/'
inputdatasetfilename = 'train.txt'
# testfilename='test.txt'
exp_details = 'coed-mix(word)_gating2'
filename = 'duallstm_128_subword'

# Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 0
LABEL_COLUMN = 1
LABELS = ['0', '1', '2']  # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num = {}
mapping_num2char = {}
wordMAXLEN = 30
charMAXLEN = 10

# LSTM Model Parameters
# Embedding
MAX_FEATURES = 0
char_embedding_size = 150
# Convolution
filter_length = 4
nb_filter = 128
pool_length = 4
# LSTM
lstm_output_size = 128
# Training
batch_size = 256
number_of_epochs = 100
numclasses = 3
test_size = 0.2
nclasses = 3

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


def parse(Masterdir, filename, seperator, datacol, labelcol, labels):
    """
    Purpose -> Data I/O
    Input   -> Data file containing sentences and labels along with the global variables
    Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
    """
    # Reads the files and splits data into individual lines
    f = open(Masterdir + Datadir + filename, 'r', encoding='UTF-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    # print(lines)

    X_train = []
    Y_train = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator)
        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = token(line[datacol])
        # print(tokenized_lines)

        # Creates character lists
        # char_list = []
        # sentence = []
        # for words in tokenized_lines:
        #     for char in words:
        #         char_list.append(char)
        #     sentence.append(char_list)
        # print(sentence)
        # # print(char_list) - Debugs the character list created
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
                # print(mapping_char2num[char])
                char_list.append(mapping_char2num[char])
            # print(char_list)
            # char_list=sequence.pad_sequences(np.array(char_list),maxlen=wordmaxlen)
            linelist.append(char_list)
        # linelist=sequence.pad_sequences(linelist,maxlen=charmaxlen)
        # print('kkkk')
        # print(linelist)
        # print(no) -- Debugs the number mappings
        X_train1.append(linelist)
    print('XX',X_train1)
     # pad_char_all = np.zeros(((17000,30,10)))
    pad_char_all=[]
    for i,line in enumerate(X_train1):
        while len(line) < 30:
            line.insert(0, [])
        while len(line) >30:
            line=line[:30]
        # print(len(line))
        # print('line',line)
        pad_senc =pad_sequences(line, maxlen=10)
        pad_senc=pad_senc.tolist()
        s=[]
        for w in pad_senc:
            x = []
            for c in w:
                x.append(c)
            s.append(x)
        # print(pad_senc)
        # print('padsen',pad_senc.shape)
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
    # print(mapping_char2num)
    # print(mapping_num2char)
    # Pads the X_train to get a uniform vector
    # TODO: Automate the selection instead of manual input

    # X_train1 = sequence.pad_sequences(np.array(X_train1), maxlen=wordmaxlen)
    # print('kkkk', pad_char_all)
    # # print(np.array(pad_char_all).shape)
    # print('xxxx')
    charno = charno + 1
    maxlen_char=10
    return [pad_char_all, mapping_num2char, mapping_char2num, charno, maxlen_char]


def build_data_train_test(data_train, data_test, train_ratio=0.8):
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


def make_idx_data(X_train, X_test, word_idx_map, maxlen=30):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_train, x_test = [], []
    for line in X_train:
        sent = get_idx_from_sent(line, word_idx_map)
        x_train.append(sent)
    # for line in X_dev:
    #     sent = get_idx_from_sent(line, word_idx_map)
    #     x_dev.append(sent)
    for line in X_test:
        sent = get_idx_from_sent(line, word_idx_map)
        x_test.append(sent)

    x_train = sequence.pad_sequences(np.array(x_train), maxlen=maxlen)
    # print("X_train:", x_train.shape)
    # x_dev = sequence.pad_sequences(np.array(x_dev), maxlen=maxlen)
    x_test = sequence.pad_sequences(np.array(x_test), maxlen=maxlen)

    return [x_train, x_test]


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


def save_model(Masterdir, exp_details, model):
    """
    Purpose -> Saves Keras model files to the given directory
    Input   -> Directory and experiment details to be saved and trained model file
    Output  -> Nil
    """
    # Referred from:- http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model.save_weights(Masterdir + Modeldir + exp_details + '_weights.h5')
    json_string = model.to_json()
    f = open(Masterdir + Modeldir +  exp_details + '_architecture.json', 'w')
    f.write(json_string)
    f.close()


def get_activations(model, layer, X_batch):
    """
    Purpose -> Obtains outputs from any layer in Keras
    Input   -> Trained model, layer from which output needs to be extracted & files to be given as input
    Output  -> Features from that layer
    """
    # Referred from:- TODO: Enter the forum link from where I got this
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output, ])
    activations = get_activations([X_batch, 0])
    return activations


def evaluate_model(X_test,x_chartrain, y_test, model, batch_size, numclasses):
    """
    Purpose -> Evaluate any model on the testing data
    Input   -> Testing data and labels, trained model and global variables
    Output  -> Nil
    """
    # Convert y_test to one-hot encoding
    y_test = np_utils.to_categorical(y_test, numclasses)
    # Evaluate the accuracies
    score, acc = model.evaluate([X_test, x_chartrain],y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def save_data(Masterdir, filename, X_train, X_test, y_train, y_test):
    """
    Purpose -> Saves train, test data along with labels and features in the respective directories in the folder
    Input   -> Train and test data, labels and features along with the directory and experiment details to be mentioned
    Output  -> Nil
    """
    h5f = h5py.File(Masterdir + Datadir + 'Xtrain_' + filename + '.h5', 'w')
    h5f.create_dataset('dataset', data=X_train)
    h5f.close()

    h5f = h5py.File(Masterdir + Datadir + 'Xtest_' + filename + '.h5', 'w')
    h5f.create_dataset('dataset', data=X_test)
    h5f.close()

    output = open(Masterdir + Datadir + 'Ytrain_' + filename + '.pkl', 'wb')
    pickle.dump([y_train], output)
    output.close()

    output = open(Masterdir + Datadir + 'Ytest_' + filename + '.pkl', 'wb')
    pickle.dump([y_test], output)
    output.close()

    '''h5f = h5py.File(Masterdir+Featuredir+'features_train_'+filename+'.h5', 'w')
    h5f.create_dataset('dataset', data=features_train)
    h5f.close()

    h5f = h5py.File(Masterdir+Featuredir+'features_test_'+filename+'.h5', 'w')
    h5f.create_dataset('dataset', data=features_test)
    h5f.close()'''


from keras.engine.topology import Layer


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def call(self, x, a, b, c):
        # assert isinstance(x, list)
        # a, b,c = x
        return (1.0 - a) * b + a * c

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


if __name__ == '__main__':
    """
    Master function
    """
    print('Starting RNN Engine...\nModel: Char-level LSTM.\nParsing data files...')
    out = parse(Masterdir, inputdatasetfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    # print('out:', out)
    # outtest = parse(Masterdir, testfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    X_train = out[0]
    X_train = np.asarray(X_train)
    y_train = out[1]
    # print(X_train)
    # print(y_train)
    # X_test = outtest[0]
    # y_test = outtest[1]
    print('Parsing complete!')

    print('Creating character dictionaries and format conversion in progess...')
    out = char2num(mapping_num2char, mapping_char2num, X_train, wordMAXLEN, charMAXLEN)
    # outtest=convert_char2num(mapping_num2char,mapping_char2num,X_test,MAXLEN)
    # print(out[0])
    # print('kkkkkkkkk')

    mapping_num2char = out[1]
    mapping_char2num = out[2]
    MAX_FEATURES = out[3]
    maxlen_char_word = out[4]
    X_chartrain = out[0]
    # print(X_chartrain)
    # print(X_chartrain.shape)
    # print('ooooo',X_chartrain)
    y_chartrain = np.asarray(y_train).flatten()
    X_chartrain, X_chartest, y_chartrain, y_chartest = train_test_split(X_chartrain, y_chartrain, test_size=0.2,
                                                                        random_state=42)
    # X_chartrain, X_chardev, y_chartrain, y_chardev = train_test_split(X_chartrain, y_chartrain, test_size=0.2,
    #                                                                   random_state=42)

    # X_test = np.asarray(outtest[0])
    # y_test = np.asarray(y_test).flatten()
    print('Complete!')
    print('Splitting data into train and test...')
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    vocab = build_data_train_test(X_train, X_test)
    vocsize = len(vocab)
    print('vocabsize:', vocsize)
    # X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(X_train)
    print('X_train shape:', X_train.shape)
    # print('X_dev shape:', X_dev.shape)

    model_file = os.path.join('vector', 'glove_model.txt')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    w2v = load_bin_vec(model, vocab)
    print('word embeddings loaded!')
    print('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=model.vector_size)
    x_train, x_test = make_idx_data(X_train, X_test, word_idx_map, maxlen=30)
    print('x_train',x_train)
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
    fwd_state = GRU(150)(char_embed)
    print(fwd_state)
    bwd_state = GRU(150, go_backwards=True)(char_embed)
    char_embed = Concatenate(axis=1)([fwd_state, bwd_state])
    print('char_lstm:', char_embed.shape)
    print(s[1])
    char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 150]))(char_embed)
    char_embed = Dropout(0.1, name='char_embed_dropout')(char_embed)
    print('char_lstm(reshape):', char_embed.shape)  # (?,36,300)

    'word_embedding'
    word_embed_dim = 300
    word_input = Input(shape=[wordMAXLEN], dtype='int32', name='input')
    embed = Embedding(input_dim=max_features, output_dim=word_embed_dim, input_length=wordMAXLEN,
                      weights=[W], mask_zero=False, name='embedding', trainable=False)(word_input)
    print(embed.shape)


    'attention is used to concate char2word'


    merged1 = merge([embed,char_embed], mode='sum')
    tanh = Activation('tanh')(merged1)
    W_tanh = Dense(300)(tanh)
    a = Activation('sigmoid')(W_tanh)

    t = Lambda(lambda x: K.ones_like(x, dtype='float32'))(a)

    merged2 = merge([a, embed], mode='mul')
    sub = Subtract()([t, a])
    merged3 = merge([sub, char_embed], mode='mul')
    x_wave = merge([merged2, merged3], mode='sum')

    # (None, 36, 5)
    auxc = Bidirectional(LSTM(150), merge_mode='concat')(
    x_wave)
    auxc=Dense(3,activation='softmax')(auxc)

    # 双向GRU
    bi_gru = Bidirectional(GRU(150, name='gru'), merge_mode='concat')(
        x_wave)  # (None, None, 128)
    bi_gru = Dropout(0.1)(bi_gru)

    # 主分类器

    mainc = Dense(nclasses, activation='softmax')(bi_gru)  # (None, 36, 4)


    # 将辅助分类器和主分类器相加，作为模型最终输出
    final_output = merge([auxc, mainc], mode='sum')














    # 'vector gating'
    # linembed = Dense(300, activation='sigmoid')(embed)
    # print('lineembed', linembed)
    # # gating = Lambda(lambda x,y: (1.0 - y)*x+y*x)(embed,linembed)
    # # gating=(1.0 - linembed)*embed+linembed*embed
    # # gating=MyLayer(300)(linembed,embed,char_embed)
    # t = Lambda(lambda x: K.ones_like(x, dtype='float32'))(linembed)
    # merged1 = merge([linembed, char_embed], name='merged1', mode='mul')
    # sub = Subtract()([t, linembed])
    # merged2 = merge([embed, sub], name='merged2', mode='mul')
    # gating = merge([merged1, merged2], name='gating', mode='sum')
    #
    # # gating = (1.0 - linembed) *embed +linembed * char_embed
    #
    # enc = Bidirectional(LSTM(150, recurrent_dropout=0.25))(gating)
    # fc = Dense(64, activation="relu")(enc)
    # dropout = Dropout(0.25)(fc)
    # output = Dense(3, activation='softmax')(dropout)

    model = Model(inputs=[word_input, char_input], outputs=final_output, name='output')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'],
                  )
    print(model.summary())

    history = LossHistory()

    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    es = EarlyStopping(monitor='val_loss', patience=5,min_delta=0.001)
    # mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
    #                      monitor='val_loss', save_best_only=True)
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0)
    print(X_chartrain.shape)
    print(x_train.shape)
    # print(X_chardev.shape )
    # print(x_dev.shape)
    # print()

    y_chartrain = to_categorical(y_chartrain)
    y_chartest = to_categorical(y_chartest)
    model.fit([x_train,X_chartrain ], y_chartrain,
              batch_size=batch_size,
              shuffle=True,
              nb_epoch=number_of_epochs,
              validation_data=([x_test,X_chartest ], y_chartest),
               callbacks=[history, es])
    history.loss_plot('epoch')

    numclasses = 3

    # print('Evaluating model...')
    # evaluate_model(x_test,X_chartest , y_chartest, model, batch_size, numclasses)


    print('Saving experiment...')
    save_model(Masterdir, exp_details, model)
    # save_data(Masterdir, exp_details, X_train, X_test, y_train, y_test)
    print('Saved! Experiment finished!')

    y_pred1 = model.predict([ x_test,X_chartest], batch_size=batch_size)
    y_pred = np.argmax(y_pred1, axis=1)
    print(y_pred)


    def accuracy(original, predicted):
        print("F1 score is: " + str(f1_score(original, predicted, average='macro')))
        print("recall score is: " + str(recall_score(original, predicted, average='macro')))
        scores = confusion_matrix(original, predicted)
        print(scores)
        print(np.trace(scores) / float(np.sum(scores)))


    accuracy(y_test, y_pred)

    result_output = pd.DataFrame(data={"sentiment": y_pred})
    result_output.to_csv("../result/dual-lstm.csv", index=False, quoting=3)
