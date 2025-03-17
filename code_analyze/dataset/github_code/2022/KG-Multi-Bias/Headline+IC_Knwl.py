import os
import csv
import numpy as np
import pandas
import random

import shutil

from keras.models import *

from keras import metrics, Sequential
from keras.layers import *
from keras import optimizers

from keras.preprocessing import text
from keras.utils import to_categorical
from keras.preprocessing import sequence

from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import keras

from keras.models import *
from keras.models import Model
from keras.preprocessing import text

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score,f1_score
from imblearn.metrics import geometric_mean_score

import warnings
from sklearn.metrics import classification_report,accuracy_score

from sentence_transformers import SentenceTransformer# Set a seed value
seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value) 

import matplotlib.pyplot as plt

def text_vectorization(data):

    # lst_models = ['distiluse-base-multilingual-cased-v2','paraphrase-multilingual-MiniLM-L12-v2','paraphrase-multilingual-mpnet-base-v2']

    model = SentenceTransformer('ml-distiluse-base-multilingual-cased-v2') 
    model_sum = SentenceTransformer('ml-distiluse-base-multilingual-cased-v2')

    colnames = ['outlet','title_ml','title_en','lang','kg_en','kg_ml','bias']
    df = pandas.read_csv('data/' + data + '.csv', names=colnames, sep='\t')

    lst_title = df.title_ml.tolist()
    lst_summary = df.kg_ml.tolist()
    
    #Vectorization..
    embeddings_title = model.encode(lst_title) 
    np.save(data + '_title.npy', embeddings_title)

    embeddings_summary = model_sum.encode(lst_summary) 
    np.save(data + '_summary.npy', embeddings_summary)

    
def train_save_model(model_name):

    dic = 3

    #------------------------------------------------------------------------------------------------------------------------------
    # calculate the length of the files..

    #subtract 1 if headers are present..
    num_train = len(open('data/train.csv', 'r').readlines())
    num_valid = len(open('data/valid.csv', 'r').readlines())
    num_test = len(open('data/test.csv', 'r').readlines())

    print('\nDataset statistics : ' + '  num_train : ' + str(num_train) + ',  num_valid  : ' + str(num_valid) + ',  num_test  : ' + str(num_test) + '\n')
    #-------------------------------------------------------------------------------------------------------
    # model building..

    print('\nBuilding model...\n')

    encode_title = Input(shape=(512,))
    encode_summary = Input(shape=(512,))


    gate_model = Concatenate()([encode_title, encode_summary])

    gate_model = Dense(3, activation='softmax')(gate_model)

    gate_model = Model(inputs=[encode_title,encode_summary], outputs=gate_model)
    gate_model.summary()
    
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'models/'+ model_name +'/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping]

    
    if os.path.isfile('models/'+ model_name +'/MODEL.h5') == False:

        colnames = ['outlet','title_ml','title_en','lang','kg_en','kg_ml','bias']
        df_train = pandas.read_csv('data/train.csv', names=colnames, sep='\t')
        df_valid = pandas.read_csv('data/valid.csv', names=colnames, sep='\t')
        df_test = pandas.read_csv('data/test.csv', names=colnames, sep='\t')

        train_bias = df_train.bias.tolist()
        train_bias_list = []

        for item in train_bias:
            if item == 'LC':
                train_bias_list.append(0)
            elif item == 'LB':
                train_bias_list.append(1)
            else:
                train_bias_list.append(2)

        valid_bias = df_valid.bias.tolist()
        valid_bias_list = []

        for item in valid_bias:
            if item == 'LC':
                valid_bias_list.append(0)
            elif item == 'LB':
                valid_bias_list.append(1)
            else:
                valid_bias_list.append(2)

        trainans = to_categorical(train_bias_list, 3)
        validans = to_categorical(valid_bias_list, 3)

        trainque_feature = np.load('embeddings/ml-distiluse-base-multilingual-cased-v2/train_title.npy')
        validque_feature = np.load('embeddings/ml-distiluse-base-multilingual-cased-v2/valid_title.npy')
        testque_feature = np.load('embeddings/ml-distiluse-base-multilingual-cased-v2/test_title.npy')

        trainsum_feature = np.load('embeddings/ml-distiluse-base-multilingual-cased-v2/train_summary.npy')
        validsum_feature = np.load('embeddings/ml-distiluse-base-multilingual-cased-v2/valid_summary.npy')
        testsum_feature = np.load('embeddings/ml-distiluse-base-multilingual-cased-v2/test_summary.npy')

        history = gate_model.fit([trainque_feature,trainsum_feature], trainans, epochs=50, batch_size=1024, validation_data=([validque_feature,validsum_feature], validans), callbacks=callbacks_list, verbose=1)

        fig = plt.figure()
        fig.set_dpi(300)    
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
        fig.savefig('bert-11-no_atten-loss.png')

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('models/'+ model_name +'/MODEL.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('models/'+ model_name +'/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('models/'+ model_name +'/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('models/'+ model_name +'/MODEL.h5', by_name=True) 

    print('\n\nGenerating answers...') 
    ans = gate_model.predict([testque_feature,testsum_feature])

    fp = open('models/'+ model_name +'/test.ans', 'w')

    for h in range(num_test):
        if np.argmax(ans[h]) == 0:
            fp.write('LC\n')
        elif np.argmax(ans[h]) == 1:
            fp.write('LB\n')
        else:
            fp.write('RC\n')

    fp.close()

def evaluate(model_name):

    warnings.filterwarnings("ignore", category=UserWarning)

    languages = ['slv','fin','swe','ron','ces']
    
    f_test = open('data/test.csv')

    lines_test = f_test.readlines()

    true_ans_test = []
    lang_list = []

    for line in lines_test:
        bias = line.split('\t')[9].strip()
        true_ans_test.append(bias)
        lang_name = line.split('\t')[3].strip()
        lang_list.append(lang_name)

    f = open('models/'+ model_name +'/test.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()

    print(jaccard_score(true_ans_test, pred_ans,average='macro')) 
    print(confusion_matrix(true_ans_test, pred_ans))

    print('\n\n')

    for ln in languages:

        true_ = []
        pred_ = []
       

        for i in range(0,len(true_ans_test)):
            if lang_list[i] == ln:
                true_.append(true_ans_test[i])
                pred_.append(pred_ans[i])


        print('\n--------Language: '+ ln + '--------\n')

        print(set(true_))
        print(set(pred_))

        print(jaccard_score(true_, pred_,average='micro'))
        print(f1_score(true_, pred_,average='micro'))
        print(classification_report(true_, pred_))

def main():

     try:
         print('\n\nTurning text into vectors...')

         if os.path.isfile('test_summary.npy') == False:
            
             text_vectorization('test')
             print('\nTest Vectorization complete...\n\n')
             text_vectorization('valid')
             print('\nValid Vectorization complete...\n\n')
             text_vectorization('train')
             print('\nTrain Vectorization complete...\n\n')
        
         print('\nVectorization complete...\n\n')
     except:
         pass


    if os.path.exists('models/model-en-11/') == False:
        os.mkdir('models/model-en-11/')
        train_save_model("model-en-11")
        evaluate("model-en-11")
        shutil.rmtree('models/model-en-11/', ignore_errors=True)

if __name__ == "__main__":
    main()
