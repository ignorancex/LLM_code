#encoding=utf-8

import os
import sys
import time
import json
import random

INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
MAX_DOCS_IN_CLUSTER = 5
MAX_LEN_SUMM = 200
MAX_LEN_DOC = 500
source_domain = 'nytimes'
#target_domain = 'nytimes'
#source_domain = 'washingtonpost'
#target_domain = 'washingtonpost'
#source_domain = 'theguardian'
#source_domain = 'cnn'
#target_domain = 'cnn'
#target_domain = 'bbc'
#source_domain = 'theguardian'
target_domain = 'theguardian'
source_domain = 'dailymail'
#target_domain = 'dailymail'

def cut_by_length(doc, max_length):
    sents = doc.split('\t')
    new_doc = []; length = 0
    for line in sents:
        flist = line.split()
        length += len(flist)
        if length < max_length:
            new_doc.append(line)
    return '\t'.join(new_doc)

def load_clusters():
    res_pairs = []
    res_source_single = []
    res_target_single = []
    for i, line in enumerate(open(INPUT_FILE)):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        source_domain_summary = ''
        target_domain_summary = ''
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            document = cut_by_length(document, MAX_LEN_DOC).replace('\t', ' ')
            summary = cut_by_length(summary, MAX_LEN_SUMM).replace('\t', ' ')
            if source.find(source_domain) > -1:
                source_domain_summary = summary
            if source.find(target_domain) > -1:
                target_domain_summary = summary
        if source_domain_summary != '' and target_domain_summary != '':
            res_pairs.append((source_domain_summary, target_domain_summary))
        elif source_domain_summary != '':
            res_source_single.append(source_domain_summary)
        elif target_domain_summary != '':
            res_target_single.append(target_domain_summary)
    return res_pairs, res_source_single, res_target_single

def analysis_novelty(train_filename, test_filename, ngram, min_count=0):
    from collections import Counter
    def get_ngram(filename):
        doc_grams = []
        for line in open(filename):
            document = line.strip().split()
            for i in range(len(document)-ngram+1):
                doc_grams.append(' '.join(document[i:i+ngram]))
        return doc_grams
    train_ngrams = Counter(get_ngram(train_filename))
    train_ngrams = set([item for item in train_ngrams if train_ngrams[item]>=min_count])
    test_ngrams = get_ngram(test_filename)
    count = 0; novel = 0
    for gram in test_ngrams:
        if gram not in train_ngrams:
            novel += 1
        count += 1
    return novel, count

def ngram_novelty(train_filename, test_filename):
    test_novel_num_uni = []; test_count_num_uni = []
    test_novel_num_bi = []; test_count_num_bi = []
    test_novel_num_tri = []; test_count_num_tri = []
    test_novel_num_four = []; test_count_num_four = []
    novel, count = analysis_novelty(train_filename, test_filename, 1)
    test_novel_num_uni.append(novel)
    test_count_num_uni.append(count)
    novel, count = analysis_novelty(train_filename, test_filename, 2)
    test_novel_num_bi.append(novel)
    test_count_num_bi.append(count)
    novel, count = analysis_novelty(train_filename, test_filename, 3)
    test_novel_num_tri.append(novel)
    test_count_num_tri.append(count)
    novel, count = analysis_novelty(train_filename, test_filename, 4)
    test_novel_num_four.append(novel)
    test_count_num_four.append(count)
    print ('Novelty uni-gram', sum(test_novel_num_uni)/sum(test_count_num_uni))
    print ('Novelty bi-gram', sum(test_novel_num_bi)/sum(test_count_num_bi))
    print ('Novelty tri-gram', sum(test_novel_num_tri)/sum(test_count_num_tri))
    print ('Novelty four-gram', sum(test_novel_num_four)/sum(test_count_num_four))

if __name__ == '__main__':
    res_pairs, res_source_single, res_target_single = load_clusters()
    random.shuffle(res_source_single)
    random.shuffle(res_target_single)
    random.shuffle(res_pairs)

    fpout = open('check_lm_train.txt', 'w')
    fpout.write('\n'.join(res_source_single))
    fpout.close()

    fpout = open('check_lm_test_same.txt', 'w')
    fpout.write('\n'.join([pair[0] for pair in res_pairs]))
    fpout.close()

    fpout = open('check_lm_test_cross.txt', 'w')
    fpout.write('\n'.join([pair[1] for pair in res_pairs]))
    fpout.close()

    # Same resource
    train_filename = 'check_lm_train.txt'
    test_filename = 'check_lm_test_same.txt'
    print ("Same Resource:")
    ngram_novelty(train_filename, test_filename)

    # Different resource
    train_filename = 'check_lm_train.txt'
    test_filename = 'check_lm_test_cross.txt'
    print ("Cross Resource:")
    ngram_novelty(train_filename, test_filename)
