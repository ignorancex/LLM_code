queries = []
with open('data/MS-MARCO/queries.train.tsv', 'r') as f:
    for line in f:
        qid, query = line.strip().split('\t')
        queries.append((qid, query))


num_train_queries = len(queries)
print('train queries num: ', num_train_queries)



with open('data/MS-MARCO/queries.dev.tsv', 'r') as f:
    for line in f:
        qid, query = line.strip().split('\t')
        queries.append((qid, query))


print('dev queries num: ', len(queries) - num_train_queries)
print('all queries num: ', len(queries))


with open('data/MS-MARCO/queries.all.tsv', 'w') as f:
    for qid, query in queries:
        f.write(f'{qid}\t{query}\n')