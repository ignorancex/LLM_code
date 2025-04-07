#coding=utf8

import os, sys
import json
from test_rouge import report_rouge

ORACLE_LEN = 3
MAX_DOC_LEN = 8000
MAX_GT_LEN = 500
SYSTEM_COMMAND = 'extoracle [SOURCE] [TARGET] -method greedy -output [ORACLE] -length [LEN]'

def _read_one_line(line):
    flist = line.strip().split('\t')
    cluster_id = flist[0]
    summ_url = flist[1]
    pairs = flist[2]
    pair_obj = json.loads(pairs)
    
    docs = []; golds = []
    for pid in pair_obj:
        pair = pair_obj[pid]
        document = pair['[DOCUMENT]']
        source = pair['[SORUCE]'].replace(':80', '')
        if source.find('www.newser.com') > -1:
            summary = pair['[TITLE]'] + pair['[SUMMARY]']
        else:
            summary = pair['[SUMMARY]']

        doc = ' '.join(' '.join(document).lower().split()[:MAX_DOC_LEN])
        docs.append(doc)

        summary = ' '.join(' '.join(summary).lower().split()[:MAX_GT_LEN])
        golds.append(summary)

    return docs, golds

def process(input_path, gold_path, ora_path, doc_path, task):
    fpout_gold = open(gold_path, 'w')
    fpout_ref = open(ora_path, 'w')
    fpout_doc = open(doc_path, 'w')
    with open(input_path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)
        for line in json_obj:
            docs, golds = _read_one_line(line)
            if task == 'single':
                for i in range(len(golds)):
                    gold = golds[i]
                    doc = docs[i]
                    fpout_gold.write(gold+'\n')
                    fpout_doc.write(doc+'\n')
            if task == 'assist':
                for i in range(len(golds)):
                    gold = golds[i]
                    doc = []
                    for j in range(len(golds)):
                        if i != j:
                            doc.append(docs[j])
                    doc = ' '.join(doc)
                    fpout_gold.write(gold+'\n')
                    fpout_doc.write(doc+'\n')
            if task == 'full':
                for i in range(len(golds)):
                    gold = golds[i]
                    doc = ' '.join(docs)
                    fpout_gold.write(gold+'\n')
                    fpout_doc.write(doc+'\n')
        fpout_gold.close()
        fpout_doc.close()
        gold_path, ora_path, doc_path
        system_command = SYSTEM_COMMAND.replace('[SOURCE]', doc_path)\
                                        .replace('[TARGET]', gold_path)\
                                        .replace('[ORACLE]', ora_path)\
                                        .replace('[LEN]', str(ORACLE_LEN))
        os.system(system_command)
        return

if __name__ == '__main__':
    input_path = '/scratch/xxu/multi-multi/raw_data/test.json'
    tmp_gold_path = './tmp/gold.txt'
    tmp_ora_path = './tmp/ora.txt'
    tmp_doc_path = './tmp/doc.txt'
    task = sys.argv[1]
    if task == 'single':
        process(input_path, tmp_gold_path, tmp_ora_path, tmp_doc_path, task)
    elif task == 'assist':
        process(input_path, tmp_gold_path, tmp_ora_path, tmp_doc_path, task)
    elif task == 'full':
        process(input_path, tmp_gold_path, tmp_ora_path, tmp_doc_path, task)
    report_rouge(tmp_gold_path, tmp_ora_path)
