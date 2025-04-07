#coding=utf8

import json
import random
from test_rouge import report_rouge

LEAD_SENT_NUM = 2
SAMPLE_NUM = 1000

target_sources = set([line.strip() for line in open('test_resource_lead.txt')])
print (target_sources)

pairs_dict = {}
for key in target_sources:
    pairs_dict[key] = []

def _read_one_line(line):
    flist = line.strip().split('\t')
    cluster_id = flist[0]
    summ_url = flist[1]
    pairs = flist[2]
    pair_obj = json.loads(pairs)
    
    golds = []; gens = []; sources = []
    for pid in pair_obj:
        pair = pair_obj[pid]
        document = pair['[DOCUMENT]']
        source = pair['[SORUCE]'].replace(':80', '')
        if source.find('www.newser.com') > -1:
            summary = pair['[TITLE]'] + pair['[SUMMARY]']
        else:
            summary = pair['[SUMMARY]']

        if len(document) < LEAD_SENT_NUM:
            continue

        source = '.'.join(source.split('.')[:2])

        if source not in target_sources:
            continue

        lead = ' '.join(document[:LEAD_SENT_NUM]).lower()
        gens.append(lead)

        summary = ' '.join(summary).lower()
        golds.append(summary)

        sources.append(source)

    return golds, gens, sources

def process(input_path):
    with open(input_path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)
        for line in json_obj:
            golds, gens, sources = _read_one_line(line)
            for i in range(len(golds)):
                gold = golds[i]
                gen = gens[i]
                source = sources[i]
                pairs_dict[source].append((gold, gen))

def process_report(gold_path, gen_path):
    for source in pairs_dict:
        fpout_gold = open(gold_path, 'w')
        print (source, len(pairs_dict[source]))
        fpout_gen = open(gen_path, 'w')
        for item in random.sample(pairs_dict[source], SAMPLE_NUM):
            fpout_gold.write(item[0]+'\n')
            fpout_gen.write(item[1]+'\n')
        fpout_gold.close()
        fpout_gen.close()
        report_rouge(gold_path, gen_path)

if __name__ == '__main__':
    input_path = '/scratch/xxu/multi-multi/raw_data/train.json'
    tmp_gold_path = './tmp/gold.txt'
    tmp_gen_path = './tmp/gen.txt'
    process(input_path)
    process_report(tmp_gold_path, tmp_gen_path)
