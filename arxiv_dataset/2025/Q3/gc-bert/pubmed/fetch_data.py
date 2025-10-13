import pandas as pd
import metapub
import os
import json
from multiprocessing import Pool
import time
from functools import partial

SAVE_PATH = './data/articles/'

def read_nodes(path='data/Pubmed-Diabetes.NODE.paper.tab'):
    nodes = {}
    with open(path, 'r') as f:
        for line in f.readlines()[2:]:
            pubmed_id, label = line.split('\t')[:2]
            nodes[pubmed_id] = label[-1]
    nodes = pd.Series(nodes).reset_index().rename(columns={'index': 'pubmed_id', 0: 'label'})
    return nodes

def fetched(save_path=SAVE_PATH):
    return [i.split('.')[0] for i in os.listdir(save_path)]

def fetch_article(pmid, fetcher, save_path=SAVE_PATH):
    start_time = time.time()
    article = fetcher.article_by_pmid(pmid)
    article_dict = article.to_dict()
    article_dict['author_list'] = [i.affiliations for i in article_dict['author_list']]
    del article_dict['content'], article_dict['xml']
    article_dict['history'] = {k: str(v) for k,v in article_dict['history'].items()}
    json.dump(article_dict, open(f'{save_path}/{pmid}.json', 'w'))
    end_time = time.time()
    if end_time - start_time < 1:
        time.sleep(1 - (end_time - start_time) + 0.3)
    print(f'Fetched article {pmid}')
    return article_dict

def fetch():
    fetcher = metapub.PubMedFetcher(cachedir=os.getcwd() + '/.cache')
    nodes = read_nodes()
    to_fetch = list(set(nodes.pubmed_id.tolist()) - set(fetched()))
    with Pool(8) as p:
        p.map(
            partial(fetch_article, fetcher=fetcher), 
            to_fetch
        )

if __name__ == '__main__':
    fetch()
