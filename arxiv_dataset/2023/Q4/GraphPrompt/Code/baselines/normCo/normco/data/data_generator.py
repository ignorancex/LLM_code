import argparse
import re
import string
import numpy as np
import pickle
import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
# from gensim.utils import tokenize
# from gensim.models.keyedvectors import KeyedVectors

from ..utils.text_processing import word_tokenize
from ..utils.text_processing import word_tokenize
from ..utils.text_processing import clean_text
from ..utils.text_processing import load_dict_from_vocab_file
from ..utils.text_processing import tokens_to_ids
from ..data.data_utils import *
# from..data.data_utils import load_text_batch
# from ..model.reader_utils import text_to_batch
# from torch.nn import embeddings
import networkx as nx
from networkx import Graph,DiGraph
stop_words = set(stopwords.words('english'))


'''
concept2id dict --> to replace concept to disease
'''

class HierarchyGraph:
    def __init__(self,relations):
        self.graph = self._build_graph(relations)
    def _build_graph(self,relations):
        graph = nx.Graph()
        for pair in relations:
            graph.add_edge(pair[0],pair[-1])
        return graph
    def get_neighbor_idx(self,target,max_depth=10,max_nodes=20,search_method='bfs'):
        ls_neighbors = []
        if target in self.graph:
            if search_method=='bfs':
                # get the end node for each edge
                ls_neighbors = [i[1] for i in nx.bfs_edges(self.graph,target,max_depth)]
                #prune the list of neighbors
                ls_neighbors = ls_neighbors[:min(max_nodes,len(ls_neighbors))]
            elif search_method=='dfs':
                # get the end node for each edge
                ls_neighbors = [i[1] for i in nx.dfs_edges(self.graph,target,max_depth)]
                #prune the list of neighbors
                ls_neighbors = ls_neighbors[:min(max_nodes,len(ls_neighbors))]

        ls_neighbors = [target] + ls_neighbors
        return ls_neighbors



class DataGenerator:
    def __init__(self,args):
        self.args = args
    # def createVocabAndEmbeddings(self,concept_dict, vocabFileName, embeddingInitFileName, conceptInitFileName,
    #                              pretrained_embeddings, use_unk_concept=True):
    def create_vocab(self,mention_id_dict,need_concept_embedding=False,pretrained_embeddings=None, use_unk_concept=True):
        print("CREATING VOCABULARY...\n")
        vocab = set()
        if need_concept_embedding:
            for (mention,concept) in mention_id_dict.keys():
                mention = " ".join([t for t in word_tokenize(mention) if t not in stop_words])
                mention_tokens = set(word_tokenize(clean_text(mention, removePunct=True, lower=False)))
                vocab.update(mention_tokens)
                vocab.update(set([t.lower() for t in mention_tokens]))
                concept = " ".join([t for t in word_tokenize(concept) if t not in stop_words])
                concept_tokens = set(word_tokenize(clean_text(concept, removePunct=True, lower=False)))
                vocab.update(concept_tokens)
                vocab.update(set([t.lower() for t in concept_tokens]))
        else:
            for mention in mention_id_dict.keys():
                mention = " ".join([t for t in word_tokenize(mention) if t not in stop_words])
                mention_tokens = set(word_tokenize(clean_text(mention, removePunct=True, lower=False)))
                vocab.update(mention_tokens)
                vocab.update(set([t.lower() for t in mention_tokens]))

        print("RESOLVING DUPLICATE VOCAB TOKENS...\n")
        # Resolve same word different case
        network = defaultdict(set)
        for v in vocab:
            head_word = v.lower()
            network[head_word].add(v)
            network[head_word].add(v.lower())

        duplicates = {}
        for n in network:
            if len(network[n]) > 1:
                duplicates[n] = network[n]
        # wv = KeyedVectors.load_word2vec_format(pretrained_embeddings, binary=True)
        # mean = np.mean(wv.syn0)
        # var = np.var(wv.syn0)
        # vocab = set()
        for d in duplicates:
            vocab = vocab - network[d]
            # just use lowercase version
            vocab.add(d)
        vocab = set([k.lower() for k in vocab])
        vocab.update({'<pad>','<unk>'})
        vocab = sorted(vocab)
        vocab_dict = {}
        for i in range(len(vocab)):
            vocab_dict[vocab[i]]=i
        return vocab_dict
    def save_data(self,dir,words,lens,ids,seqlens):
        np.savez(dir, words=np.expand_dims(words, 1),
                 lens=np.expand_dims(lens, 1),
                 ids=np.expand_dims(ids, 1), seq_lens=np.expand_dims(seqlens, 1))

    def build_graph(self,tree):
        concept_graph = HierarchyGraph(tree)
        return concept_graph
    def group_mentions(self,mention2id):
        id2mention_dict = {}
        for mention in mention2id.keys():
            concept_id = mention2id[mention]
            if concept_id in id2mention_dict:
                id2mention_dict[concept_id].append(mention)
            else:id2mention_dict[concept_id] = [mention]
        return id2mention_dict
    def get_related_concept_ls_unseen(self,concept,train_concept,mentions,train_concept_graph,full_concept_graph,id2mentions,mention2id,n_context_cutoff=6):
        res = {}
        isolated_nodes = {}

        for mode in concept:
            print(mode)
            discarded = 0
            preserved = 0
            available_concepts = set(concept[mode])
            if mode == "train":
                for mention in mention2id:
                        # pair is of format (mention,concept)
                    id = mention2id[mention]
                    if id in available_concepts:
                        context_ids = set(full_concept_graph.get_neighbor_idx(id, self.args.max_depth, self.args.max_nodes,
                                                                             self.args.search_method)[1:])
                        ls_mentions = [mention]
                        ls_ids = [id]
                        for context_id in context_ids:
                            available = set(id2mentions[context_id]) #.intersection(set(mentions[mode]))
                            if len(available) > 0:
                                ls_mentions.append(random.choice(list(available)))
                                ls_ids.append(context_id)
                        if len(ls_mentions) < n_context_cutoff:
                            # res[mention] = [None,None]
                            isolated_nodes[mention] = id
                            discarded += 1
                        else:
                            preserved += 1
                            ls_mentions = ls_mentions[:n_context_cutoff]
                            ls_ids = ls_ids[:n_context_cutoff]
                            res[mention] = [ls_mentions, ls_ids]
            else:
                for mention in mention2id:
                    id = mention2id[mention]
                    if id in (available_concepts - train_concept):
                        context_ids = set(full_concept_graph.get_neighbor_idx(id, self.args.max_depth, self.args.max_nodes,
                                                                         self.args.search_method)[1:])
                        ls_mentions = [mention]
                        ls_ids = [id]
                        for context_id in context_ids:
                            ls_mentions.append(random.choice(id2mentions[context_id]))
                            ls_ids.append(context_id)
                        if len(ls_mentions)<n_context_cutoff:
                                # res[mention] = [None,None]
                            isolated_nodes[mention]=id
                            discarded+=1
                        else:
                            preserved+=1
                            ls_mentions = ls_mentions[:n_context_cutoff]
                            ls_ids = ls_ids[:n_context_cutoff]
                            res[mention]=[ls_mentions,ls_ids]
                    # mention2id_set = mention2id_set - context_ids
            print("number of discarded entities in {} in coherence database: {}".format(mode,discarded))

            print("number of preserved entities in {} in coherence database: {}".format(mode, preserved))

        return res,isolated_nodes

    def get_related_concept_ls_seen(self,concept,train_concept,mentions,train_concept_graph,full_concept_graph,id2mentions,mention2id,n_context_cutoff=6):
        res = {}
        isolated_nodes = {}

        for mode in concept:
            print(mode)
            discarded = 0
            preserved = 0
            if mode == "train":
                for mention in mentions[mode]:
                        # pair is of format (mention,concept)
                    id = mention2id[mention]
                    context_ids = set(full_concept_graph.get_neighbor_idx(id, self.args.max_depth, self.args.max_nodes,
                                                                         self.args.search_method))
                    ls_mentions = []
                    ls_ids = []
                    for context_id in context_ids:
                        available = set(id2mentions[context_id]).intersection(set(mentions[mode]))
                        if len(available)>0:
                            ls_mentions.extend(available)
                            ls_ids.extend([context_id]*len(available))
                    if len(ls_mentions) < n_context_cutoff:
                        # res[mention] = [None,None]
                        isolated_nodes[mention] = id
                        discarded += 1
                    else:
                        preserved += 1
                        ls_mentions = ls_mentions[:n_context_cutoff]
                        ls_ids = ls_ids[:n_context_cutoff]
                        res[mention] = [ls_mentions, ls_ids]
            else:
                for mention in mentions[mode]:
                    id = mention2id[mention]
                    context_ids = set(full_concept_graph.get_neighbor_idx(id, self.args.max_depth, self.args.max_nodes,
                                                                     self.args.search_method))
                    ls_mentions = []
                    ls_ids = []
                    for context_id in context_ids:
                        ls_mentions.extend(id2mentions[context_id])
                        ls_ids.extend([context_id]*len(id2mentions[context_id]))
                    if len(ls_mentions)<3:
                            # res[mention] = [None,None]
                        isolated_nodes[mention]=id
                        discarded+=1
                    else:
                        preserved+=1
                        ls_mentions = ls_mentions[:3]
                        ls_ids = ls_ids[:3]
                        res[mention]=[ls_mentions,ls_ids]
                    # mention2id_set = mention2id_set - context_ids
            print("number of discarded entities in {} in coherence database: {}".format(mode,discarded))

            print("number of preserved entities in {} in coherence database: {}".format(mode, preserved))

        return res,isolated_nodes


    def gen_data_dict(self,concept_ids,mentions,coherence_data,isolated_nodes,vocab):
        data_dicts = {}
        for k in concept_ids.keys():
            concept_ls = concept_ids[k]
            mention_ls = mentions[k]
            #mentions only data (isolated points)
            if len(isolated_nodes)>0:
                mwords, mlens, mids, mseqlens = load_text_batch([[mention_ls[i],[concept_ls[i]]] for i in range(len(concept_ls)) if mention_ls[i] in isolated_nodes],vocab,6)
                m_dict = {
                "words":np.expand_dims(mwords,1),
                "lens":np.expand_dims(mlens,1),
                "ids":np.expand_dims(mids,1),
                'seq_lens':np.expand_dims(np.asarray([[1] for i in range(mseqlens)]), 1)

                    }
            else:
                m_dict = None
            #data with context information
            hwords, hlens, hids, hseqlens = [],[],[],[]
            for mention in mention_ls:
                if mention not in isolated_nodes:
                    neighbor_mentions,neighbor_concept_ids = coherence_data[mention]
                    if neighbor_mentions is not None:
                        wd,lens,ids,seqlens = load_text_batch([[neighbor_mentions[i],[neighbor_concept_ids[i]]] for i in range(len(neighbor_concept_ids))],vocab,6)
                        hwords.append(wd)
                        hlens.append(lens)
                        hids.append(ids)
                        hseqlens.append(seqlens)
            h_dict={"words": np.asarray(hwords),
                "lens": np.asarray(hlens),
                "ids": np.asarray(hids),
                'seq_lens': np.asarray(hseqlens)}
            if self.args.save_only:
                split_type = "unseen" if self.args.is_unseen else "seen"
                print("./data/{}/{}_{}.pkl".format(split_type, self.args.dataset_nm, k))
                with open("./data/{}/{}_{}.pkl".format(split_type,self.args.dataset_nm,k), "wb") as tf:
                    d = {'mentions':m_dict,'hierarchy':h_dict}
                    pickle.dump(d, tf)
                data_dicts[k] = "../data/{}/{}_{}.pkl".format(split_type,self.args.dataset_nm,k)
                del d
            else:
                data_dicts[k] = {'mentions':m_dict,'hierarchy':h_dict}

        return data_dicts

    #paired_data:
    def prepare_data(self,concepts,mentions,paired_data,tree,mention2id):
        '''
        main method for data generation
        workflow:
        construct dictionary (map between concept and id)
        generate vocabulary
        gen mentions data
        gen dictionary data
        gen coherence data
        '''

        vocab = self.create_vocab(mention2id,False,None,True)
        #input: mention2id dict, output: conceptid_to_mentions dict
        concept2mentions = self.group_mentions(mention2id)
        #tree: pairs of indicies of concepts

        full_concept_graph = self.build_graph(tree)
        train_concepts = set(concepts['train'])
        if self.args.is_unseen:
            train_tree = [i for i in tree if (i[0] in train_concepts) and (i[-1] in train_concepts)]
            print(train_concepts)
            train_concept_graph = self.build_graph(train_tree)
        # graph becomes all ids
        # construct id2mention dictionary --> get available mentions from data to construct the coherence model
            related_mentions_dict,isolated_nodes = \
                self.get_related_concept_ls_unseen(concepts,train_concepts,mentions,train_concept_graph,full_concept_graph,concept2mentions,mention2id,n_context_cutoff=self.args.sequence_len)
        else:
            related_mentions_dict, isolated_nodes = \
                self.get_related_concept_ls_seen(concepts,train_concepts, mentions, full_concept_graph, full_concept_graph,
                                            concept2mentions, mention2id, n_context_cutoff=self.args.sequence_len)
        #concepts list of ids used

        data_dicts = self.gen_data_dict(concepts,mentions,related_mentions_dict,isolated_nodes,vocab)

        return data_dicts,vocab



