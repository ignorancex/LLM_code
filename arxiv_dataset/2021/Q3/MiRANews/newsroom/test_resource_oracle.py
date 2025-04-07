#coding=utf8

import os
import json
import random
from test_rouge import report_rouge

ORACLE_LEN = 3
MAX_DOC_LEN = 8000
MAX_GT_LEN = 500
SAMPLE_NUM = 1000
SYSTEM_COMMAND = 'extoracle [SOURCE] [TARGET] -method greedy -output [ORACLE] -length [LEN]'

target_sources = set([line.strip() for line in open('test_resource_lead.txt')])

pairs_dict = {}
for key in target_sources:
    pairs_dict[key] = []

def _read_one_line(line):
    flist = line.strip().split('\t')
    cluster_id = flist[0]
    summ_url = flist[1]
    pairs = flist[2]
    pair_obj = json.loads(pairs)
    
    golds = []; docs = []; sources = []
    for pid in pair_obj:
        pair = pair_obj[pid]
        document = pair['[DOCUMENT]']
        source = pair['[SORUCE]'].replace(':80', '')
        if source.find('www.newser.com') > -1:
            summary = pair['[TITLE]'] + pair['[SUMMARY]']
        else:
            summary = pair['[SUMMARY]']

        if len(document) < ORACLE_LEN:
            continue

        source = '.'.join(source.split('.')[:2])
        if source not in target_sources:
            continue
        sources.append(source)

        doc = ' '.join(' '.join(document).lower().split()[:MAX_DOC_LEN])
        docs.append(doc)

        summary = ' '.join(summary).lower()
        golds.append(summary)

    return golds, docs, sources

def process(input_path):
    with open(input_path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)
        for line in json_obj:
            golds, docs, sources = _read_one_line(line)
            for i in range(len(golds)):
                gold = golds[i]
                doc = docs[i]
                source = sources[i]
                pairs_dict[source].append((gold, doc))

def process_report(gold_path, doc_path, ora_path):
    for source in pairs_dict:
        fpout_gold = open(gold_path, 'w')
        print (source, len(pairs_dict[source]))
        fpout_doc = open(doc_path, 'w')
        for item in random.sample(pairs_dict[source], SAMPLE_NUM):
            fpout_gold.write(item[0]+'\n')
            fpout_doc.write(item[1]+'\n')
        fpout_gold.close()
        fpout_doc.close()

        system_command = SYSTEM_COMMAND.replace('[SOURCE]', doc_path)\
                                        .replace('[TARGET]', gold_path)\
                                        .replace('[ORACLE]', ora_path)\
                                        .replace('[LEN]', str(ORACLE_LEN))
        os.system(system_command)
        report_rouge(gold_path, ora_path)

if __name__ == '__main__':
    input_path = '/scratch/xxu/multi-multi/raw_data/train.json'
    tmp_gold_path = './tmp/gold.txt'
    tmp_doc_path = './tmp/doc.txt'
    tmp_ora_path = './tmp/ora.txt'
    process(input_path)
    process_report(tmp_gold_path, tmp_doc_path, tmp_ora_path)
