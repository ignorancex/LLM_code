#coding=utf8

import sentencepiece as spm
import json
import os
import sys

def train_vocab(inputfile, model_prefix, vocab_size, user_syms=[]):
    spm.SentencePieceTrainer.train(input=inputfile,
                                    model_prefix=model_prefix,
                                    vocab_size=vocab_size,
                                    user_defined_symbols=user_syms)

def read_one_line(line):
    flist = line.strip().split('\t')
    cluster_id = flist[0]
    summ_url = flist[1]
    pairs = flist[2]
    pair_obj = json.loads(pairs)

    sentences = []
    for pid in pair_obj:
        pair = pair_obj[pid]
        document = pair['[DOCUMENT]']
        source = pair['[SORUCE]'].replace(':80', '')
        if source.find('www.newser.com') > -1:
            summary = pair['[TITLE]'] + pair['[SUMMARY]']
        else:
            summary = pair['[SUMMARY]']
        for sentence in document:
            sentences.append(sentence.lower())
        for sentence in summary:
            sentences.append(sentence.lower())
    return sentences

def preprocess(oringinal_data, outputfile):
    fpout = open(outputfile, 'w')
    with open(oringinal_data) as f:
        line = f.read().strip()
        json_obj = json.loads(line)
        for line in json_obj:
            sentences = read_one_line(line)
            fpout.write('\n'.join(sentences)+'\n')
    fpout.close()

if __name__ == '__main__':
    oringinal_data = '/scratch/xxu/multi-multi/raw_data/train.json'
    inputfile = './tmp.input'
    model_prefix = 'multi'
    vocab_size = 10000
    user_syms = ['<S>', '<T>', '</S>', '<P>', '<PAD>', '<Q>', '<SUPP_START>', '<supp_start>']
    preprocess(oringinal_data, inputfile)
    train_vocab(inputfile, model_prefix, vocab_size, user_syms)
