#coding=utf8

MAX_DOCS_IN_CLUSTER = 5
MIN_SENTENCE_LENGTH = 3
MIN_LENGTH = 3
MAX_LEN_SUMM = 200
MAX_LEN_DOC = 500
MAX_LEN_SUP = 500
TEST_NUM = 6000
DEV_NUM = 5000
INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
RAW_DATA_DIR = '/scratch/xxu/multi-multi/raw_data/'
CLS_TOK = '<s>'
SEP_TOK = '</s>'

import sys
import json
import random
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def split():
    data = [line.strip() for line in open(INPUT_FILE)]
    random.shuffle(data)
    data_test = data[:TEST_NUM]
    data_dev = data[TEST_NUM:(TEST_NUM+DEV_NUM)]
    data_train = data[(TEST_NUM+DEV_NUM):]

    train_path = RAW_DATA_DIR + 'train.json'
    fpout = open(train_path, 'w')
    fpout.write(json.dumps(data_train))
    fpout.close()
    dev_path = RAW_DATA_DIR + 'dev.json'
    fpout = open(dev_path, 'w')
    fpout.write(json.dumps(data_dev))
    fpout.close()
    test_path = RAW_DATA_DIR + 'test.json'
    fpout = open(test_path, 'w')
    fpout.write(json.dumps(data_test))
    fpout.close()

def _preprocess(doc, max_length, high_freq_sent, tokenize=False):
    sents = doc.lower().split('\t')
    new_doc = []; length = 0
    for line in sents:
        if line in high_freq_sent:
            continue
        if line.find('newser') > -1:
            continue
        flist = line.split()
        if len(flist) <= MIN_SENTENCE_LENGTH:
            continue
        if tokenize:
            inputs = tokenizer(line)
            length += len(inputs['input_ids'])
        else:
            length += len(flist)
        if length > max_length:
            break
        new_doc.append(line)
    return '\t'.join(new_doc)

def _preprocess_high_freq():
    path = RAW_DATA_DIR + '/train.json'
    src_dict = {}; tgt_dict = {}
    with open(path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)

        for line in json_obj:
            flist = line.strip().split('\t')
            cluster_id = flist[0]
            summ_url = flist[1]
            pairs = flist[2]
            pair_obj = json.loads(pairs)

            for pid in pair_obj:
                pair = pair_obj[pid]
                for sent in pair['[DOCUMENT]']:
                    sent_lower = sent.lower()
                    if sent_lower not in src_dict:
                        src_dict[sent_lower] = 1
                    else:
                        src_dict[sent_lower] += 1
                source = pair['[SORUCE]'].replace(':80', '')
                if source.find('www.newser.com') > -1:
                    summary = pair['[TITLE]'] + pair['[SUMMARY]']
                else:
                    summary = pair['[SUMMARY]']
                for sent in summary:
                    sent_lower = sent.lower()
                    if sent_lower not in tgt_dict:
                        tgt_dict[sent_lower] = 1
                    else:
                        tgt_dict[sent_lower] += 1
        high_freq_src = set()
        high_freq_tgt = set()
        for sent in src_dict:
            if src_dict[sent] > 50:
                high_freq_src.add(sent)
        for sent in tgt_dict:
            if tgt_dict[sent] > 50:
                high_freq_tgt.add(sent)
        return high_freq_src, high_freq_tgt


def _extract_pairs(set_name, high_freq_src, high_freq_tgt):
    if set_name == 'train':
        path = RAW_DATA_DIR + '/train.json'
        src_path = RAW_DATA_DIR + '/multi_train_src.jsonl'
        tgt_path = RAW_DATA_DIR + '/multi_train_tgt.jsonl'
    elif set_name == 'dev':
        path = RAW_DATA_DIR + '/dev.json'
        src_path = RAW_DATA_DIR + '/multi_dev_src.jsonl'
        tgt_path = RAW_DATA_DIR + '/multi_dev_tgt.jsonl'
    elif set_name == 'test':
        path = RAW_DATA_DIR + '/test.json'
        src_path = RAW_DATA_DIR + '/multi_test_src.jsonl'
        tgt_path = RAW_DATA_DIR + '/multi_test_tgt.jsonl'

    fpout_src = open(src_path, 'w')
    fpout_tgt = open(tgt_path, 'w')
    with open(path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)

        for line in json_obj:
            flist = line.strip().split('\t')
            cluster_id = flist[0]
            summ_url = flist[1]
            pairs = flist[2]
            pair_obj = json.loads(pairs)

            docs = []; summs = []
            for pid in pair_obj:
                pair = pair_obj[pid]
                document = '\t'.join(pair['[DOCUMENT]'])
                source = pair['[SORUCE]'].replace(':80', '')
                if source.find('www.newser.com') > -1:
                    summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
                else:
                    summary = '\t'.join(pair['[SUMMARY]'])
                document = _preprocess(document, MAX_LEN_DOC, high_freq_src)
                summary = _preprocess(summary, MAX_LEN_SUMM, high_freq_tgt)
                if len(document.split()) > MIN_LENGTH and len(summary.split()) > MIN_LENGTH:
                    docs.append(document)
                    summs.append(summary)
                if len(docs) == MAX_DOCS_IN_CLUSTER:
                    break
            fpout_src.write('\n'.join(docs)+'\n')
            fpout_tgt.write('\n'.join(summs)+'\n')

        fpout_src.close()
        fpout_tgt.close()

def extract_pairs():
    high_freq_src, high_freq_tgt = _preprocess_high_freq()
    _extract_pairs('train', high_freq_src, high_freq_tgt)
    _extract_pairs('dev', high_freq_src, high_freq_tgt)
    _extract_pairs('test', high_freq_src, high_freq_tgt)

def _multi_in_single_out(set_name, high_freq_src, high_freq_tgt):
    if set_name == 'train':
        path = RAW_DATA_DIR + '/train.json'
        src_path = RAW_DATA_DIR + '/multi_train_src.jsonl'
        tgt_path = RAW_DATA_DIR + '/multi_train_tgt.jsonl'
    elif set_name == 'dev':
        path = RAW_DATA_DIR + '/dev.json'
        src_path = RAW_DATA_DIR + '/multi_dev_src.jsonl'
        tgt_path = RAW_DATA_DIR + '/multi_dev_tgt.jsonl'
    elif set_name == 'test':
        path = RAW_DATA_DIR + '/test.json'
        src_path = RAW_DATA_DIR + '/multi_test_src.jsonl'
        tgt_path = RAW_DATA_DIR + '/multi_test_tgt.jsonl'

    fpout_src = open(src_path, 'w')
    fpout_tgt = open(tgt_path, 'w')
    with open(path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)

        for line in json_obj:
            flist = line.strip().split('\t')
            cluster_id = flist[0]
            summ_url = flist[1]
            pairs = flist[2]
            pair_obj = json.loads(pairs)

            docs = []; summs = []; sups = []
            sup_doc_num = len(pair_obj)-1
            max_sup_len = MAX_LEN_SUP / sup_doc_num
            for pid in pair_obj:
                pair = pair_obj[pid]
                document = '\t'.join(pair['[DOCUMENT]'])
                source = pair['[SORUCE]'].replace(':80', '')
                if source.find('www.newser.com') > -1:
                    summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
                else:
                    summary = '\t'.join(pair['[SUMMARY]'])
                document = _preprocess(document, MAX_LEN_DOC, high_freq_src, tokenize=True)
                sup_document = _preprocess(document, max_sup_len, high_freq_src, tokenize=True)
                summary = _preprocess(summary, MAX_LEN_SUMM, high_freq_tgt)
                if len(document.split()) > MIN_LENGTH and len(summary.split()) > MIN_LENGTH:
                    docs.append(document)
                    summs.append(summary)
                    sups.append(sup_document)
                if len(docs) == MAX_DOCS_IN_CLUSTER:
                    break
            if len(docs) < 2:
                continue
            for i in range(len(docs)):
                src, tgt = _prepare_sup_docs(docs, summs, sups, i)
                fpout_src.write(src+'\n')
                fpout_tgt.write(tgt+'\n')

        fpout_src.close()
        fpout_tgt.close()

def _prepare_sup_docs(docs, summs, sup_docs, idx):
    sups = []
    for i in range(len(docs)):
        if i == idx:
            continue
        sups.append(SEP_TOK+' '+sup_docs[i])
    return CLS_TOK + ' ' + docs[idx] + '\t' + '\t'.join(sups), summs[idx]

def multi_in_single_out():
    high_freq_src, high_freq_tgt = _preprocess_high_freq()
    _multi_in_single_out('train', high_freq_src, high_freq_tgt)
    _multi_in_single_out('dev', high_freq_src, high_freq_tgt)
    _multi_in_single_out('test', high_freq_src, high_freq_tgt)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('need parameters: split/extract_pairs')
        exit(1)
    if sys.argv[1] == 'split':
        split()
    elif sys.argv[1] == 'extract_pairs':
        extract_pairs()
    elif sys.argv[1] == 'multi_pairs':
        multi_in_single_out()
