
import json
# import wget
import os
import scipy
import  tqdm
import random
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import json
#
# import wget
import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import re
import torch
from string import punctuation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy import sparse
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

class TextPreprocess():
    """
    Text Preprocess module
    Support lowercase, removing punctuation, typo correction
    """

    def __init__(self,
                 lowercase=True,
                 remove_punctuation=True,
                 ignore_punctuations="",
                 typo_path=None):
        """
        Parameters
        ==========
        typo_path : str
            path of known typo dictionary
        """
        self.lowercase = lowercase
        self.typo_path = typo_path
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        for ig_punc in ignore_punctuations:
            self.punctuation = self.punctuation.replace(ig_punc, "")
        self.rmv_puncts_regex = re.compile(r'[\s{}]+'.format(re.escape(self.punctuation)))

        if typo_path:
            self.typo2correction = self.load_typo2correction(typo_path)
        else:
            self.typo2correction = {}

    def load_typo2correction(self, typo_path):
        typo2correction = {}
        with open(typo_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip()
                tokens = s.split("||")
                value = "" if len(tokens) == 1 else tokens[1]
                typo2correction[tokens[0]] = value

        return typo2correction

    def remove_punctuation(self, phrase):
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = ' '.join(phrase).strip()

        return phrase

    def correct_spelling(self, phrase):
        phrase_tokens = phrase.split()
        phrase = ""

        for phrase_token in phrase_tokens:
            if phrase_token in self.typo2correction.keys():
                phrase_token = self.typo2correction[phrase_token]
            phrase += phrase_token + " "

        phrase = phrase.strip()
        return phrase

    def run(self, text):
        if self.lowercase:
            text = text.lower()

        if self.typo_path:
            text = self.correct_spelling(text)

        if self.rmv_puncts:
            text = self.remove_punctuation(text)

        text = text.strip()

        return text


# catch all suitable datasets on the website
def get_all_data(filename='../data/ontologies.jsonld',data_dir='../data/datasets'):
    specific_problem_ids = ['rs', 'fix', 'eo',
                            'envo']  # for some unkonwn reasons, rs.obo, fix.obo and eo.obo can not be downloaded;and envo has a strange problem
    urls = []
    ids = []
    with open(filename, mode='r', encoding='utf-8') as f:
        content = json.load(f)['ontologies']
        for i, entry in enumerate(content):
            id = entry['id']
            # every entry has an id, and we only need to consider the urls which are normalized as {id}.obo
            if 'products' in entry.keys():
                products = entry['products']

                for product in products:
                    if product['id'] == id + '.obo' and id not in specific_problem_ids:
                        urls.append(product['ontology_purl'])
                        ids.append(id)

    # download relative files to data_dir, finnally we get 95 files
    # print(ids)
    for i, (id, url) in enumerate(zip(ids, urls)):
        # print(id)
        filename = id + '.obo'
        file = wget.download(url=url, out=os.path.join(data_dir, filename))


# given single file, construct corresponding graph of terms and its dictionary and query set
def load_data(filename='../data/datasets/cl.obo', use_text_preprocesser=False):
    """
    args:
        use text preprocesser: decide whether we process the data wtih lowercasing and removing punctuations

    returns:
        name_array: array of all the terms' names. no repeated element, in the manner of lexicographic order
        query_id_array: array of (query,id), later we split the query_set into train and test dataset;sorted by ids
        mention2id: map all mentions(names and synonyms of all terms) to ids, the name and synonyms with same term have the same id

        graph

    some basic process rules:
    1.To oavoid overlapping, we just abandon the synonyms which are totally same as their names
    2. Considering that some names appear twice or more, We abandon correspoding synonyms
    3.Some synonyms have more than one corresponding term, we just take the first time counts
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
                    synonym = text_processer.run(line[start_pos:end_pos]) if use_text_preprocesser else line[start_pos:end_pos]
                    if synonym == name: continue  # filter these mentions that are literally equal to the node's name,make sure there is no verlap
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

        print('mention num', len(mention2id.items()))
        print('names num', len(name_array))
        print('query num', len(query_id_array))
        print('edge num', len(list(set(edges))))
        print('triple num', len(triples))
        values = [1] * (2 * len(edges))
        rows = [i for (i, j) in edges] + [j for (i, j) in edges]  # construct undirected graph
        cols = [j for (i, j) in edges] + [i for (i, j) in edges]
        edge_index = torch.LongTensor([rows, cols])  # undirected graph edge index
        graph = sparse.coo_matrix((values, (rows, cols)), shape=(len(name_array), len(name_array)))
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        # print(n_components)
        return np.array(name_array), np.array(query_id_array), mention2id, edge_index,edges,triples
# SZ: Need edges for further processing
def get_rel2desc(filename):
    rel2desc = json.load(open('./rel2desc.json'))
    rel2desc = rel2desc[filename.split('/')[-1]]
    return rel2desc
# split train,eval and test data for one file that corresponds to the queries
def data_split(query_id_array, is_unseen=True, test_size=0.33,seed=0):
    """
    args:
    is_unseen:if is_unseen==true, then the ids in training pairs and testing pairs will not overlap
    returns:
    train,valid,test datasets
    """
    setup_seed(seed=seed)
    # notice that we have sorted all the queries according to ids
    # as a result, we could remove all the (mention,concept) pairs with the same concept in an easy manner
    mentions = [mention for (mention, id) in query_id_array]
    ids = [id for (mention, id) in query_id_array]
    # random split
    if is_unseen == False:
        queries_train, queries_test = train_test_split(mentions, test_size=test_size)  # have already set up seed
        queries_valid, queries_test = train_test_split(queries_test, test_size=0.5)
        return np.array(queries_train), np.array(queries_valid), np.array(queries_test)

    # random split, and the concepts in train set and test set will not overlap
    else:
        queries_train, queries_test, ids_train, ids_test = mentions.copy(), [], ids.copy(), []

        left_ids = sorted(list(set(ids)))
        while len(queries_test) < len(mentions) * test_size:
            id = random.sample(left_ids, 1)[0]

            start_index, end_index = ids.index(id), len(ids) - 1 - list(reversed(ids)).index(
                id)  # the start index and the end index of the same concept

            for K in range(start_index, end_index + 1):
                queries_test.append(mentions[K])
                queries_train.remove(mentions[K])
                ids_test.append(id)
                ids_train.remove(id)

            left_ids.remove(id)
        queries_valid, queries_test = train_test_split(queries_test, test_size=0.5)

        return queries_train,queries_valid,queries_test

        # check overlap
        # for concept in concepts_test:
        #    if concept in concepts_train:
        #        print(concept)


# generate negative samples if needed
def construct_positive_and_negative_pairs(concept_list, synonym_pairs, neg_posi_rate):
    """
    returns: positive pairs and negative pairs.And the number of negative samples is neg_posi_rate more than synonym pairs(positive samples)
    """
    negative_pairs = []
    for i, (mention, _) in enumerate(synonym_pairs):
        for _ in range(neg_posi_rate):
            concept = random.sample(concept_list, 1)[0]
            while (mention, concept) in synonym_pairs or (mention, concept) in negative_pairs:  # avoid overlapping
                concept = random.sample(concept_list, 1)[0]
            negative_pairs.append((mention, concept))
    return synonym_pairs, negative_pairs
