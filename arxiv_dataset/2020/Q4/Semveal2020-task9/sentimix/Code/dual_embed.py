import numpy as np
import keras
import os
import pandas as pd
import h5py
import pickle
from copy import deepcopy
from keras.models import Model
from sklearn.metrics import confusion_matrix
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Dropout, Activation
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
exp_details = 'coed-mix(word)'
filename = 'duallstm_128_subword'

# Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 0
LABEL_COLUMN = 1
LABELS = ['0', '1', '2']  # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num = {}
mapping_num2char = {}
MAXLEN = 200

# LSTM Model Parameters
# Embedding
MAX_FEATURES = 0
embedding_size = 128
# Convolution
filter_length = 4
nb_filter = 128
pool_length = 4
# LSTM
lstm_output_size = 128
# Training
batch_size = 128
number_of_epochs = 200
numclasses = 3
test_size = 0.2


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
    print(lines)

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
        char_list = []
        for words in tokenized_lines:
            for char in words:
                char_list.append(char)
            char_list.append(' ')
        # print(char_list)
        # print(char_list) - Debugs the character list created
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



def word(Masterdir, filename, seperator, datacol, labelcol, labels):
    """
    Purpose -> Data I/O
    Input   -> Data file containing sentences and labels along with the global variables
    Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
    """
    # Reads the files and splits data into individual lines
    f = open(Masterdir + Datadir + filename, 'r', encoding='UTF-8')
    lines = f.read().lower()
    lines = lines.lower().split('\n')[:-1]
    #print(lines)

    X_train = []
    Y_train = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator)
        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = token(line[datacol])
        #print(tokenized_lines)

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
    #print(Y_train)

    assert (len(X_train) == Y_train.shape[0])

    return [X_train, Y_train]


def convert_char2num(mapping_n2c, mapping_c2n, trainwords, maxlen):
    """
    Purpose -> Convert characters to integers, a unique value for every character
    Input   -> Training data (In list of lists format) along with global variables
    Output  -> Converted training data along with global variables
    """
    allchars = []
    errors = 0.

    # Creates a list of all characters present in the dataset
    for line in trainwords:
        try:
            allchars = set(allchars + line)
            allchars = list(allchars)
        except:
            errors += 1

    print(errors)  # Debugging
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

    print(mapping_char2num)
    print(mapping_num2char)
    # Pads the X_train to get a uniform vector
    # TODO: Automate the selection instead of manual input
    X_train = sequence.pad_sequences(X_train[:], maxlen=maxlen)
    # print(X_train)
    return [X_train, mapping_num2char, mapping_char2num, charno]


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


def RNN(X_train, y_train, args):
    """
    Purpose -> Define and train the proposed LSTM network
    Input   -> Data, Labels and model hyperparameters
    Output  -> Trained LSTM network
    """
    # Sets the model hyperparameters
    # Embedding hyperparameters
    max_features = args[0]
    maxlen = args[1]
    embedding_size = args[2]
    # Convolution hyperparameters
    filter_length = args[3]
    nb_filter = args[4]
    pool_length = args[5]
    # LSTM hyperparameters
    lstm_output_size = args[6]
    # Training hyperparameters
    batch_size = args[7]
    nb_epoch = args[8]
    numclasses = args[9]
    test_size = args[10]

    # Format conversion for y_train for compatibility with Keras
    y_train = np_utils.to_categorical(y_train, numclasses)
    print(y_train)
    # Train & Validation data splitting
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    # Build the sequential model
    # Model Architecture is:
    # Input -> Embedding -> Conv1D+Maxpool1D -> LSTM -> LSTM -> FC-1 -> Softmaxloss
    print('Build model...')
    start = time()
    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)

    es = EarlyStopping(monitor='val_loss', patience=20)
    mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
                         monitor='val_loss', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

    sequence = Input(shape=(maxlen,), dtype='int32')

    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True, weights=[W], trainable=False) (sequence)
    embedded = Embedding(input_dim=max_features, output_dim=embedding_size, input_length=maxlen,
                         trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)
    convolution = Convolution1D(filters=nb_filter,
                                filter_length=filter_length,
                                padding='valid',
                                activation='relu',
                                strides=1
                                )(embedded)
    maxpooling = MaxPooling1D(pool_length=pool_length)(convolution)
    lstm = LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(maxpooling)
    lstm1 = LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=False)(lstm)
    enc = Bidirectional(GRU(lstm_output_size // 2, recurrent_dropout=0.25, return_sequences=True))(maxpooling)
    att = AttentionM()(enc)
    x = keras.layers.Concatenate(axis=1)([lstm1, att])
    fc1 = Dense(128, activation="relu")(x)
    fc2 = Dense(64, activation="relu")(fc1)
    fc3 = Dense(32, activation="relu")(fc2)
    fc4 = Dense(16, activation="relu")(fc3)
    fc4_dropout = Dropout(0.25)(fc4)
    output = Dense(3, activation='softmax')(fc4_dropout)
    model = Model(inputs=sequence, outputs=output)

    '''model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))
    model.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 32
    model.add(Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True))
    model.add(Flatten())


    model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(Bidirectional(LSTM(lstm_output_size//2, recurrent_dropout=0.25, return_sequences=False)))

    #model.add(AttentionM())
    model.add(Dropout(0.25))
    model.add(Dense(numclasses,activation='softmax'))'''

    # Optimizer is Adamax along with categorical crossentropy loss
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'],
                  )
    print(model.summary())
    history = LossHistory()

    print('Train...')
    # Trains model for 50 epochs with shuffling after every epoch for training data and validates on validation data
    model.fit(X_train, y_train,
              batch_size=batch_size,
              shuffle=True,
              nb_epoch=nb_epoch,
              validation_data=(X_valid, y_valid),
              callbacks=[history, es, mc, tb])

    history.loss_plot('epoch')
    return model


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


def evaluate_model(X_test, y_test, model, batch_size, numclasses):
    """
    Purpose -> Evaluate any model on the testing data
    Input   -> Testing data and labels, trained model and global variables
    Output  -> Nil
    """
    # Convert y_test to one-hot encoding
    y_test = np_utils.to_categorical(y_test, numclasses)
    # Evaluate the accuracies
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
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


if __name__ == '__main__':
    """
    Master function
    """
    print('Starting RNN Engine...\nModel: Char-level LSTM.\nParsing data files...')
    charout = parse(Masterdir, inputdatasetfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    print('out:', charout )
    # outtest = parse(Masterdir, testfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    X_chartrain = charout[0]
    y_chartrain = charout[1]
    wordout = word(Masterdir, inputdatasetfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    X_wordtrain=wordout[0]
    print(' X_wordtrain:',X_wordtrain)

    print('Creating character dictionaries and format conversion in progess...')
    out = convert_char2num(mapping_num2char, mapping_char2num, X_chartrain, MAXLEN)
    # outtest=convert_char2num(mapping_num2char,mapping_char2num,X_test,MAXLEN)
    print(out)
    mapping_num2char = out[1]
    mapping_char2num = out[2]
    MAX_FEATURES = out[3]
    X_train = np.asarray(out[0])
    y_train = np.asarray(y_train).flatten()
    # X_test = np.asarray(outtest[0])
    # y_test = np.asarray(y_test).flatten()
    print('Complete!')

    print('Splitting data into train and test...')
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(X_train)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Creating LSTM Network...')
    model = RNN(deepcopy(X_train), deepcopy(y_train), [MAX_FEATURES, MAXLEN, embedding_size, \
                                                       filter_length, nb_filter, pool_length, lstm_output_size,
                                                       batch_size, \
                                                       number_of_epochs, numclasses, test_size])

    print('Evaluating model...')
    evaluate_model(X_test, deepcopy(y_test), model, batch_size, numclasses)

    '''print('Feature extraction pipeline running...')
    activations = get_activations(model, 4, X_train)
    features_train = np.asarray(activations)
    activations = get_activations(model, 4, X_test)
    features_test = np.asarray(activations)
    print('Features extracted!')
    '''
    print('Saving experiment...')
    save_model(Masterdir, exp_details, model)
    save_data(Masterdir, exp_details, X_train, X_test, y_train, y_test)
    print('Saved! Experiment finished!')

    y_pred = model.predict(X_test, batch_size=batch_size)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
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