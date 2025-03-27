import editdistance

import pandas as pd
import regex as re
import sys
from config import ConfigManager
from data_utils import clean_text, read_dataset
import glob
import nltk
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import os
from tqdm import tqdm
import random
import numpy as np


sys.path.insert(0, './converter')

from converter_drinovski import Converter

def get_chars(all_words, all_gs_words):
    chars = set()

    for sentence, gs_sentence in zip(all_words, all_gs_words):
        for w, gs in zip(sentence, gs_sentence):
            chars = chars.union(set(w)).union(set(gs))

    l = sorted(list(chars))

    d = {c: i for i, c in enumerate(l)}
    ind = {i: c for i, c in enumerate(l)}

    return l, d, ind


def create_confusion_matrix(config):
    nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    train_data_files = glob.glob(config.get_train_dataset() + '/*.txt') # Get all files from the configured directory
    _ = glob.glob(config.get_test_dataset() + '/*.txt')

    _, train_aligned_words, train_gs_words, train_labels = read_dataset(train_data_files, sentence_tokenizer, False)
    _, test_aligned_words, test_gs_words, test_labels = read_dataset(train_data_files, sentence_tokenizer, False)

    all_words = train_aligned_words + test_aligned_words
    all_gs_words = train_gs_words + test_gs_words
    all_labels = train_labels + test_labels

    c, d, ind = get_chars(all_words, all_gs_words)
    lenc = len(c)

    cmatrix = [[0]*lenc for i in range(lenc)]

    for sentence, gs_sentence, lbls in zip(all_words, all_gs_words, all_labels):
        for w, gs, l in zip(sentence, gs_sentence, lbls):

            if l == 1:
                for o, g in zip(w, gs):
                    if o != g:
                        cmatrix[d[o]][d[g]] += 1

    return cmatrix, d, ind

def get_random_char(word, d):
    c = random.choice(word)

    return c if c in d else get_random_char(word, d)

def add_noise(word, cmatrix, d, ind):
    new_word = word

    c = get_random_char(word, d)

    total = sum(cmatrix[d[c]])
    if total > 0:
        new_c = [x / total for x in cmatrix[d[c]]]
        randomElement = np.random.choice(range(len(new_c)), p=new_c)
        new_word = new_word.replace(c, ind[randomElement], 1)

    return new_word

def generate_data(converter, config):
    noise_prob = 0.15 # noise 15% of tokens
    cmatrix, d, ind = create_confusion_matrix(config)

    files = glob.glob(config.get_main_path() + "/data/synthetic_source/*.txt")

    for i, file in enumerate(files):
        raw_text = ''
        with open(file, "r", encoding="utf-8-sig") as f:
            raw_text = f.read()

        sentences = sent_tokenize(raw_text)
        new_sentences = []
        converted_sentences = []
        total_words = 0

        for sentence in tqdm(sentences):
            sentence = sentence.lower()

            csentence = converter.convert_text(sentence)
            total_words += len(csentence)

            if len(csentence) > 0:           
                converted_sentences.append(' '.join(csentence))
        
        for sentence in converted_sentences:
            edited_sentence = sentence
            words = word_tokenize(sentence)

            for word in words:
                if word in string.punctuation:
                    continue
                
                if random.uniform(0, 1) <= noise_prob:
                    noised_word = add_noise(word, cmatrix, d, ind)
                    edited_sentence = edited_sentence.replace(word, noised_word, 1)
 
            new_sentences.append(edited_sentence)

        f = open(config.get_synthetic_dataset() + "/train_data_new_" + os.path.basename(f.name).replace(' ', '_'), "w", encoding="utf-8-sig")

        for ocr_sentence, edited_sentence in zip(converted_sentences, new_sentences):
            f.write(clean_text(ocr_sentence) + '\n')
            f.write(clean_text(edited_sentence) + '\n')

        f.close()

def main():
    config = ConfigManager()
    converter = Converter(config)
    generate_data(converter, config)
    
if __name__ == '__main__':
    main()