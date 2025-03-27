import re
import json
import torch
import numpy as np
from .dataset import TextPreprocess


def get_rel2desc(filename):
    rel2desc = json.load(open('../data/rel2desc.json'))
    rel2desc = rel2desc[filename.split('/')[-1]]
    return rel2desc


# given single file, construct corresponding graph of terms and its dictionary and query set
def load_data(filename='../data/datasets/cl.obo', use_text_preprocesser=False, return_triples=False):
    """
    args:
        use text preprocesser: decide whether we process the data wtih lowercasing and removing punctuations

    returns:
        name_array: array of all the terms' names. no repeated element, in the manner of lexicographic order

        query_id_array: array of (query, id), later we split the query_set into train and test dataset;sorted by ids

        mention2id: map all mentions(names and synonyms of all terms) to ids, the name and synonyms with same term have the same id

        graph


    some basic process rules:
    1.To avoid overlapping, we just abandon the synonyms which are totally same as their names
    2. Considering that some names appear twice or more, We abandon correspoding synonyms
    3. Some synonyms have more than one corresponding term, we just take the first time counts
    """
    text_processer = TextPreprocess()
    name_list = []  # record of all terms, rememeber some elements are repeated
    name_array = []
    query_id_array = []
    mention2id = {}

    edges = []
    triples = []

    with open(file=filename, mode='r', encoding='utf-8') as f:
        check_new_term = False
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[:6] == '[Term]':  # starts with a [Term]
                check_new_term = True
                continue
            if line[:1] == '\n':  # ends with a '\n'
                check_new_term = False
                continue
            if check_new_term == True:
                if line[:5] == 'name:':
                    name_list.append(text_processer.run(line[6:-1]) if use_text_preprocesser else line[6:-1])

        name_count = {}

        # record the count of names in raw file
        for i, name in enumerate(name_list):
            name_count[name] = name_list.count(name)

        # build a mapping function of name2id, considering that some names appear twice or more, we remove the duplication and sort them
        name_array = sorted(list(set(name_list)))

        for i, name in enumerate(name_array):
            mention2id[name] = i

        # temporary variables for every term
        # construct a scipy csr matrix of edges and collect synonym pairs
        check_new_term = False
        check_new_name = False  # remember that not every term has a name and we just take the terms with name count. Good news: names' locations are relatively above
        name = ""
        iter_name = iter(name_list)

        for i, line in enumerate(lines):
            if line[:6] == '[Term]':  # starts with a [Term] and ends with an '\n'
                check_new_term = True
                continue
            if line[:5] == 'name:':
                check_new_name = True
                if check_new_term == True:
                    name = next(iter_name)
                continue
            if line[:1] == '\n':  # signal the end of current term
                check_new_term = False
                check_new_name = False
                continue

            if check_new_term == True and check_new_name == True:
                # construct term graph
                if line[:5] == 'is_a:':
                    entry = line.split(" ")
                    if '!' in entry:  # some father_nodes are not divided by '!' and we abandon them
                        father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if father_node in name_array:  # some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[father_node], mention2id[name]))
                if line[:16] == 'intersection_of:':
                    entry = line.split(" ")
                    if '!' in entry:  # some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:  # some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node], mention2id[name]))

                if line[:13] == 'relationship:':
                    entry = line.split(" ")
                    if '!' in entry:  # some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:  # some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node], mention2id[name]))

                # collect synonyms and to dictionary set and query set
                if line[:8] == 'synonym:' and name_count[
                    name] == 1:  # anandon the situations that name appears more than once
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    synonym = text_processer.run(line[start_pos:end_pos]) if use_text_preprocesser else line[
                                                                                                        start_pos:end_pos]
                    if synonym == name: continue  # filter these mentions that are literally equal to the node's name, make sure there is no verlap
                    if synonym in mention2id.keys(): continue  # only take the first time synonyms appears counts
                    id = mention2id[name]
                    mention2id[synonym] = id
                    query_id_array.append((synonym, id))

                rel2desc = get_rel2desc(filename)
                for r in rel2desc:
                    if re.match('^[^:]+: {} '.format(r), line):
                        if '!' in entry:
                            node = " ".join(entry[entry.index('!') + 1:])[:-1]
                            if node in mention2id:
                                triples.append((mention2id[name], r, mention2id[node]))
                if re.match('^is_a: ', line):
                    if '!' in entry:
                        node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if node in mention2id:
                            triples.append((mention2id[name], 'is_a', mention2id[node]))

        query_id_array = sorted(query_id_array, key=lambda x: x[1])
        triples = sorted(list(set(triples)))

        print('mention num', len(mention2id.items()))
        print('names num', len(name_array))
        print('query num', len(query_id_array))
        print('edge num', len(list(set(edges))))
        print('triple num', len(triples))

        values = [1] * (2 * len(edges))
        rows = [i for (i, j) in edges] + [j for (i, j) in edges]  # construct undirected graph
        cols = [j for (i, j) in edges] + [i for (i, j) in edges]
        edge_index = torch.LongTensor([rows, cols])  # undirected graph edge index
        # graph = sparse.coo_matrix((values, (rows, cols)), shape = (len(name_array), len(name_array)))
        # n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        # #print(n_components)

        ret = np.array(name_array), np.array(query_id_array), mention2id, edge_index, triples
        if return_triples:
            return ret
        else:
            return ret[:-1]