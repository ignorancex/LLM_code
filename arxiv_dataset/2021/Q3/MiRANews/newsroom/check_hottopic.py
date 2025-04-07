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
source_domain = 'NONE'
target_domain = 'NONE'
#source_domain = 'washingtonpost'
#target_domain = 'nytimes'
#target_domain = 'cnn'
#source_domain='theguardian'
#target_domain='bbc'
#topic = "obama"
#topic = "hillary"
#topic = "nuclear"
#topic = "trump"
topic = "brexit"
#topic = "north korea"

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
        summaires = []; domains = []; topic_related = False
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            document = cut_by_length(document, MAX_LEN_DOC).replace('\t', ' ').lower()
            summary = cut_by_length(summary, MAX_LEN_SUMM).replace('\t', ' ').lower()
            if source.find(source_domain) > -1:
                source_domain_summary = summary
            elif source.find(target_domain) > -1:
                target_domain_summary = summary
            else:
                summaires.append(summary)
                domains.append(source)
            if summary.find(topic) > -1:
                topic_related = True
        if source_domain_summary != '' and target_domain_summary != '':
            res_pairs.append((source_domain_summary, target_domain_summary))
            if source_domain_summary.find(topic) > -1 or target_domain_summary.find(topic) > -1:
                print ('['+source_domain+']:', source_domain_summary)
                print ('['+target_domain+']:', target_domain_summary)
                print ('')
        elif topic_related:
            for i, summ in enumerate(summaires):
                print ('['+domains[i]+']:', summ)
            print ('')

if __name__ == '__main__':
    load_clusters()
