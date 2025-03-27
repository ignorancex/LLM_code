
import numpy as np
import json, argparse, os
import re
import io
import pickle

from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding,MaxPooling1D,Convolution1D,CuDNNGRU, Bidirectional, GRU, Input, Flatten, SpatialDropout1D, LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from Capsule_net import Capsule
from Attention_layer import AttentionM


# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""

NUM_FOLDS = 10                   # Value of K in K-fold Cross Validation
NUM_CLASSES = 3                 # Number of classes - Happy, Sad, Angry, Others
# MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = 50         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = 128                  # The batch size to be chosen for training the model.
LSTM_DIM = 128                    # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.25                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = 10                  # Number of epochs to train a model for
emb_mask_zero = False
emb_trainable = False

kernel_size = 3
nb_filter = 60


def lstmModel(embeddingMatrix):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                embeddingMatrix.shape[1],
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                mask_zero=emb_mask_zero,
                                trainable=emb_trainable)(sequence)
    embedded = Dropout(0.25)(embeddingLayer)

    bilstm1 = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True))(embedded)
    bilstm2 = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))(bilstm1)

    fc1 = Dense(128, activation="relu")(bilstm2)
    fc2_dropout = Dropout(0.3)(fc1)

    output = Dense(NUM_CLASSES, activation='softmax')(fc2_dropout)
    # rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model, 'lstmModel'


def cnnlstmModel(embeddingMatrix):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                embeddingMatrix.shape[1],
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                mask_zero=emb_mask_zero,
                                trainable=emb_trainable)(sequence)
    embedded = Dropout(0.25)(embeddingLayer)
    convolution = Convolution1D(filters=nb_filter,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1
                                )(embedded)
    maxpooling = MaxPooling1D(pool_size=2)(convolution)

    bilstm = Bidirectional(LSTM(LSTM_DIM // 2, recurrent_dropout=0.25, return_sequences=True))(maxpooling)
    bilstm1 = Bidirectional(LSTM(LSTM_DIM // 2, recurrent_dropout=0.25, return_sequences=False))(bilstm)
    # att = AttentionM()(bilstm1)

    output = Dense(3, activation='softmax')(bilstm1)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, 'cnnlstmModel'


def capsulnetModel(embeddingMatrix):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 32
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(embeddingMatrix.shape[0],
                                embeddingMatrix.shape[1],
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                mask_zero=emb_mask_zero,
                                trainable=emb_trainable)

    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(LSTM(LSTM_DIM//2, return_sequences=True))(embedded_sequences)
    x = Bidirectional(LSTM(LSTM_DIM // 2, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True, kernel_size=(3, 1))(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.2)(capsule)

    output = Dense(NUM_CLASSES, activation='softmax')(capsule)
    model = Model(inputs=sequence_input, outputs=output)

    # rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model, 'capsulnetModel'


def attentionModel(embeddingMatrix):
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               embeddingMatrix.shape[1],
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               mask_zero=emb_mask_zero,
                               trainable=emb_trainable)(sequence)
    enc = Bidirectional(GRU(LSTM_DIM, dropout=DROPOUT, return_sequences=True))(embeddingLayer)
    enc = Bidirectional(GRU(LSTM_DIM, dropout=DROPOUT, return_sequences=True))(enc)
    att = AttentionM()(enc)
    fc1 = Dense(128, activation="relu")(att)
    fc2_dropout = Dropout(0.25)(fc1)
    output = Dense(NUM_CLASSES, activation='softmax')(fc2_dropout)
    model = Model(inputs=sequence, outputs=output)
    # rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])

    return model, 'attention'


def get_stacking(clf, data, labels, x_dev, x_test, n_folds=5, name=None):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num, dev_num = data.shape[0], x_test.shape[0], x_dev.shape[0]
    second_level_train_set = np.zeros((train_num, NUM_CLASSES))
    dev_result = np.zeros((dev_num, NUM_CLASSES))
    test_result = np.zeros((test_num, NUM_CLASSES))
    test_nfolds_sets = []
    dev_nfolds_sets = []
    print('model:',name)
    for k in range(NUM_FOLDS):
        print('-'*80)
        print('Fold %d/%d' %(k+1, NUM_FOLDS))
        validationSize = int(len(data)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k + 1)

        xTrain = np.vstack((data[:index1], data[index2:]))
        yTrain = np.vstack((labels[:index1], labels[index2:]))
        xVal = data[index1:index2]
        yVal = labels[index1:index2]
        print("Building model...")
        early_stopping = EarlyStopping(monitor='val_acc', patience=10)
        clf.fit(xTrain, yTrain, validation_data=[xVal, yVal], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                callbacks=[early_stopping])
        path = './model/{0}'.format(name)
        if not os.path.exists(path):
            os.makedirs(path)
        clf.save('./model/%s/bi-%s-model-fold-%d.h5' % (name, name, k))

        second_level_train_set[index1:index2] = clf.predict(xVal)
        dev_nfolds_sets.append(clf.predict(x_dev))
        test_nfolds_sets.append(clf.predict(x_test))

    for item in test_nfolds_sets:
        test_result += item
    test_result = test_result / n_folds

    for item in dev_nfolds_sets:
        dev_result += item
    dev_result = dev_result / n_folds

    # second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, dev_result, test_result

def accuracy(original, predicted):
        print("F1_macro score is: " + str(f1_score(original, predicted, average='macro')))
        print("F1_weighted score is: " + str(f1_score(original, predicted, average='weighted')))
        print("recall score is: " + str(recall_score(original, predicted, average='macro')))


def main():

    print('loading data...')
    pickle_file = os.path.join('./pickle/hindi/hi_en_nouser_hastag_glove_wik.pickle3')

    embeddingMatrix, word_idx_map, vocab, x_train, x_test, y_train, y_test = pickle.load(open(pickle_file, 'rb'))
    y_train = to_categorical(y_train)
    print(embeddingMatrix.shape)
    print(x_train.shape,y_train.shape)
    print(x_test.shape,y_test.shape)
    train_sets = []
    dev_sets = []
    test_sets = []
    print('K-fold start')
    print('-'*60)
    for clf, name in [attentionModel(embeddingMatrix), capsulnetModel(embeddingMatrix), lstmModel(embeddingMatrix), cnnlstmModel(embeddingMatrix)]:
        train_set, dev_set, test_set = get_stacking(clf, x_train, y_train, x_test, x_test, name=name)
        train_sets.append(train_set)
        dev_sets.append(dev_set)
        test_sets.append(test_set)

    meta_train = np.concatenate([result_set.reshape(-1, 3) for result_set in train_sets], axis=1)
    meta_dev = np.concatenate([dev_result_set.reshape(-1, 3) for dev_result_set in dev_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, 3) for y_test_set in test_sets], axis=1)
    path = './pickle/stacking_local.pickle'
    pickle.dump([meta_train,y_train, meta_dev, meta_test,y_test], open(path, 'wb'))

    svc = SVC(kernel='sigmoid', C=3)
    svc.fit(meta_train, np.array(y_train.argmax(axis=1)))
    predictions = svc.predict(meta_test)
    accuracy(y_test, predictions )





if __name__ == '__main__':
    main()
