#coding=utf8

import json
from test_rouge import report_rouge

LEAD_SENT_NUM = 2

def _read_one_line(line):
    flist = line.strip().split('\t')
    cluster_id = flist[0]
    summ_url = flist[1]
    pairs = flist[2]
    pair_obj = json.loads(pairs)
    
    golds = []; refs = []
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

        lead = ' '.join(document[:LEAD_SENT_NUM]).lower()
        refs.append(lead)

        summary = ' '.join(summary).lower()
        golds.append(summary)

    return golds, refs

def process(input_path, gold_path, ref_path):
    fpout_gold = open(gold_path, 'w')
    fpout_ref = open(ref_path, 'w')
    with open(input_path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)
        for line in json_obj:
            golds, refs = _read_one_line(line)
            for i in range(len(golds)):
                gold = golds[i]
                ref = refs[i]
                fpout_gold.write(gold+'\n')
                fpout_ref.write(ref+'\n')
        fpout_gold.close()
        fpout_ref.close()

if __name__ == '__main__':
    input_path = '/scratch/xxu/multi-multi/raw_data/test.json'
    tmp_gold_path = './tmp/gold.txt'
    tmp_ref_path = './tmp/ref.txt'
    process(input_path, tmp_gold_path, tmp_ref_path)
    report_rouge(tmp_gold_path, tmp_ref_path)
