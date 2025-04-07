from newsroom import jsonl
import json

URL2CLUSTER_DICT = 'input.url_map'
MULTI_NEWS = '../Multi-News/data/summaries.txt'
INPUT_FILE = '/scratch/xxu/multi-multi/input.dataset'
OUPUT_FILE = '/scratch/xxu/multi-multi/multi_multi.jsonl'
#OUPUT_FILE = '/scratch/xxu/multi-multi/multi_summary.jsonl'

def clean_document(text):
    return '\n'.join([item for item in text.split('\n') if item != ''])

def load_url2cluster():
    with open(URL2CLUSTER_DICT) as f:
        line = f.read().strip()
        map_dict = json.loads(line)
        return map_dict

def load_multi_news_url():
    multi_news_urls = {}
    for line in open(MULTI_NEWS):
        flist = line.strip().split('\t')
        url = flist[0]
        uid = flist[1].replace('summaries/', '')
        multi_news_urls[uid] = url
    return multi_news_urls

if __name__ == '__main__':
    map_dict = load_url2cluster()
    multi_news_urls = load_multi_news_url()
    clusters = {}
    with jsonl.open(INPUT_FILE, gzip = True) as train_file:
        for entry in train_file:
            document = clean_document(entry["text"])
            summary = entry["summary"]
            title = entry["title"]
            url = entry["archive"]
            uid = map_dict[url]
            cid = uid.replace('articles/', '').split('-')[0]
            if cid not in clusters:
                clusters[cid] = {}
                if cid in multi_news_urls:
                    clusters[cid]['multi_news_url'] = multi_news_urls[cid]
                else:
                    clusters[cid]['multi_news_url'] = 'NONE'
                clusters[cid]['doc_summ_cluster'] = {}
            clusters[cid]['doc_summ_cluster'][uid] = {}
            clusters[cid]['doc_summ_cluster'][uid]['[DOCUMENT]'] = document
            clusters[cid]['doc_summ_cluster'][uid]['[SUMMARY]'] = summary
            clusters[cid]['doc_summ_cluster'][uid]['[TITLE]'] = title
            clusters[cid]['doc_summ_cluster'][uid]['[URL]'] = url

    fpout = open(OUPUT_FILE, 'w')
    cluster_size = 0
    statistics = {}
    for cid in clusters:
        line = cid + '\t' + clusters[cid]['multi_news_url']
        clst = json.dumps(clusters[cid]['doc_summ_cluster'])
        line += '\t' + clst
        fpout.write(line + '\n')
        cluster_size += len(clusters[cid]['doc_summ_cluster'])
        if len(clusters[cid]['doc_summ_cluster']) not in statistics:
            statistics[len(clusters[cid]['doc_summ_cluster'])] = 1
        else:
            statistics[len(clusters[cid]['doc_summ_cluster'])] += 1
    fpout.close()

    print ('Number of clusters: ', len(clusters))
    print ('Ave size of clusters:', cluster_size / len(clusters))
    print (statistics)
