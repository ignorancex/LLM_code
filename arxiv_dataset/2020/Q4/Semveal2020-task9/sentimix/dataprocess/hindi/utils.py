import os
import sys
import logging
import time
import emoji

import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

def process_text(train_file):
    ret_texts = []
    ret_labels = []
    ret_sid = []
    length = []

    with open(train_file, 'r', encoding='utf8') as my_file:
        text = ''
        label = ''
        s_id = ''

        http_flag = False
        for line in my_file.readlines():
            line = line.replace('\n', '').split()

            if (len(line) == 3):
                label = line[2]
                s_id = line[1]

            if (len(line) == 2):
                text = text + line[0]

                if (line[0].startswith('http')):
                    http_flag = True
                    continue
                elif (http_flag):
                    continue
                elif (http_flag and len(line[0]) == 10):
                    http_flag = False
                    text = text + ' '
                else:
                    text = text + ' '

            if len(line) == 0:
                ret_texts.append(text)
                ret_labels.append(label)
                ret_sid.append(s_id)
                length.append(len(text.split(' ')))
                text = ''
                label = ''
                s_id = ''
                http_flag = False

        print(np.max(length))
    return ret_texts, ret_labels, ret_sid


def process_test():
    test_file = 'E:/Sub-word-LSTM(sentimix)/dataprocess/hindi/data/Hindi_test_unalbelled_conll_updated.txt'
    data = pd.read_table(test_file, header=None,quoting=3,keep_default_na=False)
    meta_index = list(data[data[0] == 'meta'].index)
    id_list = list(data.loc[meta_index, 1])
    sentence_list = []
    for i in range(len(meta_index)):
        if i < len(meta_index) - 1:
            words = list(data.loc[meta_index[i] + 1:meta_index[i + 1] - 1][0])
        else:
            words = list(data.loc[meta_index[i] + 1:][0])
        if 'https' in words:
            https_index = words.index('https')
            sentence1 = ' '.join(str(s) for s in words[:https_index])
            sentence2 = ''.join(str(s) for s in words[https_index:])
            sentence = sentence1 + ' ' + sentence2
            sentence_list.append(sentence)
        else:
            sentence = ' '.join(str(s) for s in words)
            sentence_list.append(sentence)
    okdata = {'uid': id_list,
              'sentence': sentence_list}
    df = pd.DataFrame(okdata)
    df.to_csv('E:/Sub-word-LSTM(sentimix)/dataprocess/hindi/data/get_test_f1.txt', sep='\t', index=False)


def write_file(file_name, texts, labels):
    with open(file_name, 'w', encoding='utf8') as my_file:
        my_file.write('sentence\tlabel\n')
        for i,text in enumerate(texts):
            my_file.write('%s\t%s\n' % (text,labels[i]))

def write_as_test(file_name, texts, sids):
    with open(file_name, 'w', encoding='utf8') as my_file:
        my_file.write('uid\tsetence\n')
        for i, text in enumerate(texts):
            my_file.write('%s\t%s\n' % (sids[i], text))

