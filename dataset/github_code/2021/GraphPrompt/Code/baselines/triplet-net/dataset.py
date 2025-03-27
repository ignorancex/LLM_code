
import json
from posixpath import join
from numpy.core.numeric import full_like
from transformers.models import bert
import wget
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
from transformers import BertTokenizer

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
            self.punctuation = self.punctuation.replace(ig_punc,"")
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

    def remove_punctuation(self,phrase):
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

#catch all suitable datasets on the website
def get_all_data(filename='../data/ontologies.jsonld'):
    specific_problem_ids=['rs','fix','eo','envo']# for some unkonwn reasons, rs.obo, fix.obo and eo.obo can not be downloaded;and envo has a strange problem
    urls = []
    ids = []
    with open(filename,mode='r',encoding='utf-8') as f:
        content = json.load(f)['ontologies']
        for i,entry in enumerate(content):
            id = entry['id']
            #every entry has an id, and we only need to consider the urls which are normalized as {id}.obo
            if 'products' in entry.keys():
                products = entry['products']
                
                for product in products:
                    if product['id']==id + '.obo' and id not in specific_problem_ids:
                        urls.append(product['ontology_purl'])
                        ids.append(id)
    
    #download relative files to data_dir, finnally we get 95 files
    #print(ids)
    data_dir = '../data/datasets'
    for i,(id,url) in  enumerate(zip(ids,urls)):
        #print(id)
        filename = id+'.obo'
        file = wget.download(url=url,out= os.path.join(data_dir,filename))

#given single file, construct corresponding graph of terms and its dictionary and query set
def load_data(filename='../data/datasets/cl.obo', use_text_preprocesser = False):
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
    name_list = []#record of all terms, rememeber some elements are repeated
    name_array = []
    query_id_array = []
    mention2id = {}
    
    edges=[] 

    with open(file=filename,mode='r',encoding='utf-8') as f:
        check_new_term = False
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] 
                check_new_term = True
                continue
            if line[:1]=='\n':#ends with a '\n'
                check_new_term = False
                continue
            if check_new_term == True:
                if line[:5]=='name:':
                    name_list.append(text_processer.run(line[6:-1])if use_text_preprocesser else line[6:-1])
        
        name_count = {}
        
        #record the count of names in raw file
        for i,name in enumerate(name_list):
            name_count[name] = name_list.count(name)
        
        #build a mapping function of name2id, considering that some names appear twice or more, we remove the duplication and sort them
        name_array = sorted(list(set(name_list)))

        for i,name in enumerate(name_array):
            mention2id[name] = i
        
        #temporary variables for every term
        #construct a scipy csr matrix of edges and collect synonym pairs
        check_new_term = False
        check_new_name = False#remember that not every term has a name and we just take the terms with name count. Good news: names' locations are relatively above
        name = ""
        iter_name = iter(name_list)

        for i,line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with an '\n'
                check_new_term = True
                continue
            if line[:5]=='name:':
                check_new_name = True
                if check_new_term == True:
                    name = next(iter_name)
                continue
            if line[:1]=='\n':# signal the end of current term
                check_new_term = False
                check_new_name = False
                continue

            if check_new_term == True and check_new_name == True:
                #construct term graph
                if line[:5]=='is_a:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if father_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[father_node],mention2id[name]))
                if line[:16]=='intersection_of:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node],mention2id[name]))
                
                if line[:13]=='relationship:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node],mention2id[name]))
                
                # collect synonyms and to dictionary set and query set
                if line[:8]=='synonym:' and name_count[name] == 1: #anandon the situations that name appears more than once
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    synonym = text_processer.run(line[start_pos:end_pos]) if use_text_preprocesser else line[start_pos:end_pos]
                    if synonym==name:continue#filter these mentions that are literally equal to the node's name,make sure there is no verlap
                    if synonym in mention2id.keys():continue# only take the first time synonyms appears counts
                    id = mention2id[name]
                    mention2id[synonym] = id
                    query_id_array.append((synonym,id))
        
        query_id_array = sorted(query_id_array,key = lambda x:x[1])
        
        
        print('---entity_num %d ---query_num %d ---edge_num %d' %(len(name_array),len(query_id_array),len(list(set(edges)))))
        
        values=[1]*(2*len(edges))
        rows = [i for (i,j) in edges] + [j for (i,j) in edges]# construct undirected graph
        cols = [j for (i,j) in edges] + [i for (i,j) in edges]
        edge_index = torch.LongTensor([rows,cols])# undirected graph edge index

        graph = sparse.coo_matrix((values,(rows,cols)), shape = (len(name_array),len(name_array)))
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        #print(n_components)

        return np.array(name_array),np.array(query_id_array),mention2id,edge_index



def get_rel2desc(filename):
    rel2desc = json.load(open('./rel2desc.json'))
    rel2desc = rel2desc[filename.split('/')[-1]]
    return rel2desc

def simple_load_data(filename='../data/datasets/cl.obo', use_text_preprocesser = False, return_triples=False, collect_triple = True):
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
    name_list = [] #record of all terms, rememeber some elements are repeated
    name_array = []
    query_id_array = []
    mention2id = {}
    
    edges=[] 
    triples = []

    with open(file=filename, mode='r', encoding='utf-8') as f:
        check_new_term = False
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] 
                check_new_term = True
                continue
            if line[:1]=='\n':#ends with a '\n'
                check_new_term = False
                continue
            if check_new_term == True:
                if line[:5]=='name:':
                    name_list.append(text_processer.run(line[6:-1])if use_text_preprocesser else line[6:-1])
        
        name_count = {}
        
        #record the count of names in raw file
        for i, name in enumerate(name_list):
            name_count[name] = name_list.count(name)
        
        #build a mapping function of name2id, considering that some names appear twice or more, we remove the duplication and sort them
        name_array = sorted(list(set(name_list)))

        for i, name in enumerate(name_array):
            mention2id[name] = i
        
        #temporary variables for every term
        #construct a scipy csr matrix of edges and collect synonym pairs
        check_new_term = False
        check_new_name = False#remember that not every term has a name and we just take the terms with name count. Good news: names' locations are relatively above
        name = ""
        iter_name = iter(name_list)

        for i, line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with an '\n'
                check_new_term = True
                continue
            if line[:5]=='name:':
                check_new_name = True
                if check_new_term == True:
                    name = next(iter_name)
                continue
            if line[:1]=='\n':# signal the end of current term
                check_new_term = False
                check_new_name = False
                continue

            if check_new_term == True and check_new_name == True:
                #construct term graph
                if line[:5]=='is_a:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if father_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[father_node], mention2id[name]))
                if line[:16]=='intersection_of:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node], mention2id[name]))
                
                if line[:13]=='relationship:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node], mention2id[name]))
                
                # collect synonyms and to dictionary set and query set
                if line[:8]=='synonym:' and name_count[name] == 1: #anandon the situations that name appears more than once
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    synonym = text_processer.run(line[start_pos:end_pos]) if use_text_preprocesser else line[start_pos:end_pos]
                    if synonym==name:continue#filter these mentions that are literally equal to the node's name, make sure there is no verlap
                    if synonym in mention2id.keys():continue# only take the first time synonyms appears counts
                    id = mention2id[name]
                    mention2id[synonym] = id
                    query_id_array.append((synonym, id))

                if collect_triple:
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
        
        query_id_array = sorted(query_id_array, key = lambda x:x[1])
        triples = sorted(list(set(triples)))
        
        print('mention num', len(mention2id.items()))
        print('names num', len(name_array))
        print('query num', len(query_id_array))
        print('edge num', len(list(set(edges))))
        print('triple num', len(triples))
        
        rows = [i for (i, j) in edges] + [j for (i, j) in edges]# construct undirected graph
        cols = [j for (i, j) in edges] + [i for (i, j) in edges]
        edge_index = torch.LongTensor([rows, cols])# undirected graph edge index, tensor of 2*num_edge


        
        # graph = sparse.coo_matrix((values, (rows, cols)), shape = (len(name_array), len(name_array)))
        # n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        # #print(n_components)


        rel_dict={h:[] for h,r,t in triples}
        for h,r,t in triples:
            rel_dict[h].append((r,t))

        ret = np.array(name_array), np.array(query_id_array), mention2id, edge_index, triples

        

        #print(triples[0])
        if return_triples:
            return ret
        else:
            return ret[:-1]




#split train,eval and test data for one file that corresponds to the queries
def data_split(query_id_array,is_unseen,test_size,seed):
    """
    args:
    is_unseen:if is_unseen==true, then the ids in training pairs and testing pairs will not overlap 
    returns:
    train,valid,test datasets
    """
    #notice that we have sorted all the queries according to ids
    #as a result, we could remove all the (mention,concept) pairs with the same concept in an easy manner 
    setup_seed(seed=seed)
    mentions = [mention for (mention,id) in query_id_array] 
    ids = [id for (mention,id) in query_id_array]
    #random split
    if is_unseen == False:
        queries_train,queries_test = train_test_split(mentions,test_size=test_size)#have already set up seed 
        queries_valid,queries_test = train_test_split(queries_test,test_size=0.5)
        return np.array(queries_train),np.array(queries_valid),np.array(queries_test)
    
    #random split, and the concepts in train set and test set will not overlap
    else:
        queries_train,queries_test,ids_train,ids_test=mentions.copy(),[],ids.copy(),[]
        
        left_ids = sorted(list(set(ids)))
        while len(queries_test) < len(mentions) * test_size:
            id = random.sample(left_ids,1)[0]
                
            start_index,end_index = ids.index(id), len(ids)-1 -  list(reversed(ids)).index(id)#the start index and the end index of the same concept

            for K in range(start_index,end_index+1):
                queries_test.append(mentions[K])
                queries_train.remove(mentions[K])
                ids_test.append(id)
                ids_train.remove(id)
                
            left_ids.remove(id)
        queries_valid,queries_test = train_test_split(queries_test, test_size=0.5)
        return np.array(queries_train),np.array(queries_valid),np.array(queries_test)

            #check overlap
            #for concept in concepts_test:
            #    if concept in concepts_train:
            #        print(concept)
         

#generate negative samples if needed
def construct_positive_and_negative_pairs(concept_list,synonym_pairs,neg_posi_rate):
    """
    returns: positive pairs and negative pairs.And the number of negative samples is neg_posi_rate more than synonym pairs(positive samples)
    """
    negative_pairs = []
    for i,(mention,_) in enumerate(synonym_pairs):
        for _ in range(neg_posi_rate):
            concept = random.sample(concept_list,1)[0]
            while (mention,concept) in synonym_pairs or (mention,concept) in negative_pairs:#avoid overlapping
                concept = random.sample(concept_list,1)[0]
            negative_pairs.append((mention,concept))
    return synonym_pairs,negative_pairs


class Mention_Dataset(Dataset):
    def __init__(self,mention_array,tokenizer):
        super(Mention_Dataset,self).__init__()
        self.mention_array  = mention_array
        self.tokenizer = tokenizer
    def __getitem__(self, index):
        tokens = self.tokenizer(self.mention_array[index], add_special_tokens=True, max_length = 24, padding='max_length',truncation=True,return_attention_mask = True, return_tensors='pt')
        input_ids = torch.squeeze(tokens['input_ids'])
        attention_mask = torch.squeeze(tokens['attention_mask'])
        return input_ids,attention_mask
    def __len__(self):
        return len(self.mention_array)


class Biosyn_Dataset(Dataset):
    def __init__(self,name_array,query_array,mention2id,top_k,sparse_encoder,bert_encoder,names_sparse_embedding,names_bert_embedding,bert_ratio,tokenizer):
        """
        args:
            name_arrayy: all the name of nodes in a sorted order; str of list
            query_array: all the query mentions; str of list
            top_k: the number of candidates
            mention2id: map names and queries to ids; generate labels
            sparse_score_matrix: tensor of shape(num_query, num_name)
            bert_score_matrix: tensor of shape(num_query, num_name)

        """
        super(Biosyn_Dataset,self).__init__()
        self.name_array = name_array
        self.query_array = query_array
        self.mention2id = mention2id
        self.top_k = top_k

        self.sparse_encoder = sparse_encoder
        self.bert_encoder = bert_encoder# still on the device
        self.names_sparse_embedding = names_sparse_embedding.cuda()
        self.names_bert_embedding = names_bert_embedding.cuda()# tensor of shape(num_query, num_names)
        
        self.bert_ratio = bert_ratio
        self.n_bert = int(self.top_k * self.bert_ratio)
        self.n_sparse = self.top_k - self.n_bert
        self.tokenizer = tokenizer

    # use score matrix to get candidate indices, return a tensor of shape(self.top_k,)
    def get_candidates_indices(self,query_sparse_embedding,query_bert_embedding):

        candidates_indices = torch.LongTensor(size=(self.top_k,)).cuda()
        sparse_score = (torch.matmul(torch.reshape(query_sparse_embedding,shape=(1,-1)),self.names_sparse_embedding.transpose(0,1))).squeeze()
        _,sparse_indices = torch.sort(sparse_score,descending=True)
        bert_score = (torch.matmul(torch.reshape(query_bert_embedding,shape=(1,-1)),self.names_bert_embedding.transpose(0,1))).squeeze()
        _,bert_indices = torch.sort(bert_score,descending=True)

        candidates_indices[:self.n_sparse] = sparse_indices[:self.n_sparse]
        j = 0
        for i in range(self.n_sparse,self.top_k):
            while bert_indices[j] in candidates_indices[:self.n_sparse]:
                j+=1
            candidates_indices[i] = bert_indices[j]
            j+=1
        #print(candidates_indices)
        assert(len(torch.unique(candidates_indices))==len(candidates_indices))# assert no overlap
        return candidates_indices.to('cpu'),sparse_score.to('cpu')
    
    def __getitem__(self, index):
        """
        returns:
            ids,masks and sparse_scores of candidates indices(for later predictioon)
        """
        query = self.query_array[index]
        query_tokens = self.tokenizer(query,add_special_tokens=True, max_length = 24, padding='max_length',truncation=True,return_attention_mask = True, return_tensors='pt')
        query_ids,query_attention_mask = torch.squeeze(query_tokens['input_ids']).cuda(),torch.squeeze(query_tokens['attention_mask']).cuda()

        query_bert_embedding = self.bert_encoder(query_ids.unsqueeze(0),query_attention_mask.unsqueeze(0)).last_hidden_state[:,0,:]# still on device
        query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform([query]).toarray()).cuda()
        
        candidates_indices,sparse_score = self.get_candidates_indices(query_sparse_embedding,query_bert_embedding)
        candidates_sparse_score = sparse_score[candidates_indices]
        candidates_names = self.name_array[candidates_indices]

        candidates_names_ids, candidates_names_attention_mask=[],[]
        for name in candidates_names:
            name_tokens = self.tokenizer(name,add_special_tokens=True, max_length = 24, padding='max_length',truncation=True,return_attention_mask = True, return_tensors='pt')
            name_ids,name_attention_mask = torch.squeeze(name_tokens['input_ids']),torch.squeeze(name_tokens['attention_mask'])
            candidates_names_ids.append(name_ids)
            candidates_names_attention_mask.append(name_attention_mask)
        
        candidates_names_ids = torch.stack(candidates_names_ids,dim=0)# tensor of shape(top_k, max_len)
        candidates_names_attention_mask = torch.stack(candidates_names_attention_mask,dim=0)# tensor of shape(top_k, max_len)

        labels = torch.LongTensor([self.mention2id[query]==self.mention2id[name] for name in candidates_names])

        assert(labels.shape==torch.Size([self.top_k]))

        return query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score,labels

    def __len__(self):
        return len(self.query_array)


# do not need to iter with epoches
class Graph_Dataset(Dataset):
    def __init__(self,query_array,mention2id,tokenizer):
        super(Graph_Dataset,self).__init__()
        self.query_array = query_array
        self.mention2id = mention2id
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        query = self.query_array[index]
        query_tokens = self.tokenizer(query,add_special_tokens=True, max_length = 24, padding='max_length',truncation=True,return_attention_mask = True, return_tensors='pt')
        query_ids,query_attention_mask = torch.squeeze(query_tokens['input_ids']).cuda(),torch.squeeze(query_tokens['attention_mask']).cuda()
        query_index = torch.LongTensor([self.mention2id[query]])#这里的query index就是query的label
        return query_ids,query_attention_mask,query_index,query
    
    def __len__(self):
        return len(self.query_array)


def count_datasets(dir = '../data/datasets'):
    filenames=os.listdir(dir)
    file_sizes = {filename:os.path.getsize(os.path.join(dir,filename)) for filename in filenames}
    file_sizes = sorted(file_sizes.items() ,key=lambda x: x[1], reverse=False)
    print(file_sizes)

    for (filename,size) in file_sizes:
        print(filename,'   with size%d' %size)
        load_data(os.path.join(dir,filename))

def collect_mention():
    dir = '../data/datasets'
    files = os.listdir(dir)
    for filename in files:
        print(filename)
        name_array, query_id_array, mention2id, edge_index, triples =simple_load_data(os.path.join(dir,filename),use_text_preprocesser=False,return_triples=True,collect_triple=False)
        mention_set = sorted(mention2id.keys())
        mention_set = [i +'\n' for i in mention_set]
        output_file = os.path.join('../data/mention',filename+'.txt')
        if len(name_array)>0:
            with open(output_file,mode='w',encoding='utf-8') as f:
                f.writelines(mention_set)
            

if __name__ == '__main__':
    collect_mention()