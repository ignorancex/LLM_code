#coding=utf8

INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi.jsonl'
OUTPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
#OUTPUT_FILE = '/scratch/xxu/multi-multi/multi_summary_clean.jsonl'

DOC_MIN_LEN = 50
SENT_MIN_LEN = 8
CLUSTER_MIN_SIZE = 2
THREAD_NUM = 28

import json
import multiprocessing
from nltk import word_tokenize
import spacy
nlp_seg = spacy.load("en_core_web_sm")

#from allennlp.predictors import Predictor
#predictor = Predictor.from_path("https://allennlp.s3.amazonaws.com/models/ner-model-2020.02.10.tar.gz")

def one_line(line):
    flist = line.strip().split('\t')
    cluster_id = flist[0]
    summ_url = flist[1]
    pairs = flist[2]
    pair_obj = json.loads(pairs)

    filtered_pairs = {}
    for pid in pair_obj:
        pair = pair_obj[pid]

        if '[DOCUMENT]' not in pair:
            continue
        if '[SUMMARY]' not in pair:
            continue
        if '[TITLE]' not in pair:
            continue
        if '[URL]' not in pair:
            continue
        document = pair['[DOCUMENT]']
        summary = pair['[SUMMARY]']
        title = pair['[TITLE]']
        url = pair['[URL]']
        if document == None or summary == None or url == None or title == None:
            continue

        doc_tokens = document.split()
        summ_tokens = summary.split()
        title_tokens = title.split()

        if len(doc_tokens) < DOC_MIN_LEN:
            continue
        if len(summ_tokens) < SENT_MIN_LEN:
            continue
        if summ_tokens[0].lower() in ['he', 'she', 'he\'s', 'she\'s', 'his', 'her']:
            continue

        timestamp = url.split('http')[1].split('/')[-2][0:8]
        url_domain = url.split('http')[2].split('/')[2]
        
        '''
        results = predictor.predict(sentence=summary)
        idx = 0
        while idx < len(results["tags"]):
            if results["tags"][idx] != 'O':
                break
            idx += 1
        if idx == len(results["tags"]):
            print (summary)
            continue
        '''
        
        document = ' '.join(word_tokenize(document.replace('\n', ' ')))
        summary = ' '.join(word_tokenize(summary.replace('\n', ' ')))
        title = ' '.join(word_tokenize(title.replace('\n', ' ')))
        docs = [str(d) for d in list(nlp_seg(document).sents)]
        summs = [str(s) for s in list(nlp_seg(summary).sents)]
        tits = [str(s) for s in list(nlp_seg(title).sents)]

        filtered_pairs[pid] = {'[DOCUMENT]': docs, '[SUMMARY]': summs, '[TITLE]':tits, '[URL]': url, '[DATE]': timestamp, '[SORUCE]': url_domain}
        #filtered_pairs[pid] = {'[SUMMARY]': summary, '[URL]': url, '[DATE]': timestamp}

    if len(filtered_pairs) < CLUSTER_MIN_SIZE:
        return ''

    string = cluster_id + '\t' + summ_url + '\t' + json.dumps(filtered_pairs)
    #string = cluster_id + '\t' + summ_url + '\t' + json.dumps(filtered_pairs) + '\n'

    return string

if __name__ == '__main__':

    fpout = open(OUTPUT_FILE, 'w')
    pool = multiprocessing.Pool(processes=THREAD_NUM)
    batch = []

    for line in open(INPUT_FILE):
        if len(batch) == THREAD_NUM:
            utterances = pool.map(one_line, batch)
            for res in utterances:
                if res == '':
                    continue
                fpout.write(res + '\n')
            del batch[:]
        batch.append(line.strip())

    if len(batch) > 0:
        utterances = pool.map(one_line, batch)
        for res in utterances:
            if res == '':
                continue
            fpout.write(res + '\n')

    fpout.close()
