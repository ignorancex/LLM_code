
from keras.layers import Embedding, LSTM, Bidirectional
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout, GRU
import os
import numpy as np
import pickle
from Capsule_net import Capsule
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Input, merge
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from Attention_layer import AttentionM
from vote_classifier import VotingClassifier
from keras.utils import to_categorical


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
nb_filter = 128





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
    return model


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
    return model


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
    return model


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
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])


    return model

def accuracy(original, predicted):
        print("F1_macro score is: " + str(f1_score(original, predicted, average='macro')))
        print("F1_weighted score is: " + str(f1_score(original, predicted, average='weighted')))
        print("recall score is: " + str(recall_score(original, predicted, average='macro')))

pickle_file = os.path.join('./pickle/hindi/hi_en_nouser_hastag_glove_wik.pickle3')
embeddingMatrix, word_idx_map, vocab, x_train, x_test, y_train, y_test = pickle.load(open(pickle_file, 'rb'))

y_train = to_categorical(y_train)
# Use scikit-learn to grid search the batch size and epochs
clf1 = KerasClassifier(build_fn=attentionModel(embeddingMatrix), verbose=2, epochs=10, batch_size=BATCH_SIZE)
clf2 = KerasClassifier(build_fn=capsulnetModel(embeddingMatrix), verbose=2, epochs=10, batch_size=BATCH_SIZE)
clf3 = KerasClassifier(build_fn=cnnlstmModel(embeddingMatrix), verbose=2, epochs=10, batch_size=BATCH_SIZE)
clf4 = KerasClassifier(build_fn=lstmModel(embeddingMatrix), verbose=2, epochs=10, batch_size=BATCH_SIZE)

eclf1 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3),('clf4',clf4)], voting='soft')

eclf1.fit(x_train, y_train)

y_pred1 = eclf1.predict(x_test)
y_pred = np.argmax(y_pred1, axis=1)
accuracy(y_test, y_pred)


