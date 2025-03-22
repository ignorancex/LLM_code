import os
import json
from sklearn.model_selection import train_test_split




seed = 42

with open('outputs/clustered_queries.json', 'r') as f:
    data = json.load(f)


train_queries = {}
dev_queries = {}
with open('data/MS-MARCO/queries.train.tsv', 'r') as f:
    for line in f:
        query_id, query = line.strip().split('\t')
        train_queries[query] = query_id

with open('data/MS-MARCO/queries.dev.tsv', 'r') as f:
    for line in f:
        query_id, query = line.strip().split('\t')
        dev_queries[query] = query_id

id_name_mapping = {
    0: 'geography',
    2: 'science',
    5: 'history',
    8: 'health',
    11: 'technology',
    15: 'finance',
    23: 'entertainment',
}

# new_queries = {}


for cluster_id, queries in data.items():
    if int(cluster_id) not in id_name_mapping:
        continue
    cluster_name = id_name_mapping[int(cluster_id)]
    # new_queries[cluster_name] = queries

    train_query_list = []
    dev_query_list = []
    for query in queries:
        if query in train_queries:
            qid = train_queries[query]
            train_query_list.append((qid, query))
        elif query in dev_queries:
            qid = dev_queries[query]
            dev_query_list.append((qid, query))
    

    print(f'{cluster_name}: train {len(train_query_list)}, dev {len(dev_query_list)}')


    if not os.path.exists(f'data/MS-MARCO/topic_queries/{cluster_name}'):
        os.mkdir(f'data/MS-MARCO/topic_queries/{cluster_name}')
    with open(f'data/MS-MARCO/topic_queries/{cluster_name}/queries.train.tsv', 'w') as f:
        for qid, query in train_query_list:
            f.write(f'{qid}\t{query}\n')
    with open(f'data/MS-MARCO/topic_queries/{cluster_name}/queries.dev.tsv', 'w') as f:
        for qid, query in dev_query_list:
            f.write(f'{qid}\t{query}\n')