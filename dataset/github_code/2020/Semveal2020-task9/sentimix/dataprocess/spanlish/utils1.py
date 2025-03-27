import os
import sys
import logging
import time
import emoji
import random
import numpy as np

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from googletrans import Translator
translator = Translator()

def tokenize(tweet):
    # try:
    # tweet = emoji.demojize(tweet.lower())
    tokens = tweet.split()
    new_tokens = []
    for token in tokens:
        # if token.startswith('@'):
        #     continue
        # elif token.startswith('#'):
        #     continue
        if token.startswith('http'):
            continue
        else:
            new_tokens.append(token)

    return ' '.join(new_tokens)
    # except:
    #     return 'NC'

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

            if(len(line) == 3):
                label = line[2]
                s_id = line[1]

            if(len(line) == 2):
                if line[1] == 'ne':
                    text= text + "user "
                elif line[1] == 'unk':
                    continue
                else:
                    # if(line[0] == '@'):
                    #     continue
                    # elif(line[0] == '#'):
                    #     continue
                    text = text + line[0]
                    if(line[0].startswith('http')):
                        http_flag = True
                        continue
                    elif(http_flag):
                        continue
                    elif(http_flag and len(line[0]) == 10):
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
            # print(ret_texts)
                

        print(np.max(length))
    return ret_texts, ret_labels, ret_sid

def write_file(file_name, texts, labels):
    with open(file_name, 'w', encoding='utf8') as my_file:
        my_file.write('sentence\tlabel\n')
        for i,text in enumerate(texts):
            print(i)
            my_file.write('%s\t%s\n' % (text,labels[i]))

def write_as_test(file_name, texts, sids):
    with open(file_name, 'w', encoding='utf8') as my_file:
        my_file.write('sentence\tlabel\n')
        for i, text in enumerate(texts):
            my_file.write('%s\t%s\n' % (sids[i], text))

def clean_str(texts):
    ret_texts = []

    for i, text in enumerate(texts):
        new_text = tokenize(text)
        try:
            # trans_text = translator.translate(new_text, dest='en')
            # emoji_text = emoji.emojize(new_tex).lower()

            ret_texts.append(new_text.lower())
            # print(emoji_text)

        except:
            ret_texts.append(new_text.lower())
            # print(new_text.lower())


        # time.sleep(10)
        #

    return ret_texts