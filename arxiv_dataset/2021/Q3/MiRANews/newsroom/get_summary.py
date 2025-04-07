#coding=utf8

OUTPUT_FILE = '/scratch/xxu/multi-multi/multi_summary_clean.jsonl'
INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'

if __name__ == '__main__':
    import json

    fpout = open(OUTPUT_FILE, 'w')

    for line in open(INPUT_FILE):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        new_pair = {}
        for pid in pair_obj:
            pair = pair_obj[pid]
            summary = ' '.join(pair['[SUMMARY]'])
            source = pair['[SORUCE]'].replace(':80', '')
            new_pair[pid] = {'[SUMMARY]':summary, '[SORUCE]':source}

        fpout.write(cluster_id + '\t' + summ_url + '\t' + json.dumps(new_pair) + '\n\n')

    fpout.close()
