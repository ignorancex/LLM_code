## E3_LSE.py
import numpy as np
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics

## Designing an Auto-Encoder-Classifier model
def E3_LSE(input_size,output_size, LV=5, lmd=0.5):
    # Encoder Network
    enc_input = Input(shape=(input_size,), name='enc_input')
    enc_l1 = Dense(50, activation='relu', name='encoder_layer1')(enc_input)
    enc_l1 = BatchNormalization()(enc_l1)
    enc_l1 = Dropout(rate = 0.3)(enc_l1)

    enc_l2 = Dense(25, activation='relu', name='encoder_layer2')(enc_l1)
    enc_l2 = BatchNormalization()(enc_l2)
    enc_l2 = Dropout(rate = 0.3)(enc_l2)

    enc_l3 = Dense(25, activation='relu', name='encoder_layer3')(enc_l2)
    enc_l3 = BatchNormalization()(enc_l3)
    enc_l3 = Dropout(rate = 0.3)(enc_l3)

    enc_l4 = Dense(10, activation='relu', name='encoder_layer4')(enc_l3)
    enc_l4 = BatchNormalization()(enc_l4)
    enc_l4 = Dropout(rate = 0.3)(enc_l4)

    encoder_output = Dense(LV, activation='sigmoid', name='encoder_output')(enc_l4)

    # Classifier Network
    class_l1 = Dense(10, activation='relu', name='class_layer1')(encoder_output)
    class_l2 = Dense(10, activation='relu', name='class_layer2')(class_l1)
    class_l3 = Dense(10, activation='relu', name='class_layer3')(class_l2)
    class_output = Dense(2, activation='softmax', name='class_output')(class_l3)
    # class_output = Dense(2, activation='softmax', name='class_output')(encoder_output)

    # Decoder Network1
    dec_l1 = Dense(10, activation='relu', name='decoder_layer1')(encoder_output)
    dec_l1 = BatchNormalization()(dec_l1)
    dec_l1 = Dropout(rate = 0.3)(dec_l1)

    dec_l2 = Dense(25, activation='relu', name='decoder_layer2')(dec_l1)
    dec_l2 = BatchNormalization()(dec_l2)
    dec_l2 = Dropout(rate = 0.3)(dec_l2)

    dec_l3 = Dense(25, activation='relu', name='decoder_layer3')(dec_l2)
    dec_l3 = BatchNormalization()(dec_l3)
    dec_l3 = Dropout(rate = 0.3)(dec_l3)

    dec_l4 = Dense(50, activation='relu', name='decoder_layer4')(dec_l3)
    dec_l4 = BatchNormalization()(dec_l4)
    dec_l4 = Dropout(rate = 0.3)(dec_l4)

    decoder_output = Dense(output_size, activation='sigmoid', name='decoder_output')(dec_l4)
    model = Model(inputs=[enc_input], outputs=[class_output, decoder_output])

    # Compiling model
    model.compile(optimizer='rmsprop',
                  loss={'class_output': 'binary_crossentropy', 'decoder_output': 'mean_squared_error'},
                  loss_weights={'class_output': 1 - lmd, 'decoder_output': lmd},
                  metrics=[metrics.categorical_accuracy])
    # Here I used rmsprops optimizer with default values, two objective functions are optimized
    # using  weight factors [1 for classifier and 0.1 for decoder loss]
    return model


## Designing a DNN model
def DNN(input_size):
    # Layer
    L_input = Input(shape=(input_size,), name='L_input')
    class_l1 = Dense(10, activation='relu', name='class_layer1')(L_input)
    class_l2 = Dense(10, activation='relu', name='class_layer2')(class_l1)
    class_l3 = Dense(10, activation='relu', name='class_layer3')(class_l2)
    class_output = Dense(2, activation='softmax', name='class_output')(class_l3)

    model = Model(inputs=[enc_input], outputs=[class_output])

    # Compiling model
    model.compile(optimizer='rmsprop',
                  loss={'class_output': 'binary_crossentropy'},
                  loss_weights={'class_output': 1},
                  metrics=[metrics.categorical_accuracy])
    return model