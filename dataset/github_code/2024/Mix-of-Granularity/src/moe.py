'''
Implement the Router class for MoE mechanism

Zijie 8 April 2024
'''

# Import libraries
from sentence_transformers import SentenceTransformer
import sys
import os
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import random

from config import config
os.environ['HF_ENDPOINT'] = config['hf_endpoint']
os.environ['HF_HUB_URL'] = config['hf_hub_url']

medrag_path = config['medrag_path']
sys.path.insert(0, medrag_path)
from src.medrag import MedRAG
import json
from torch.utils.data import DataLoader



from utils import (
                   load_rawdata,
                   collate_fn,
                    )


# Define the Router class
class Router(nn.Module):
    '''
    An instance of the Router Class takes a query as input, and output a vector. Each dimension of the output vector represents the weight of the document with different granualarity levels.
    
    For example, let's say we have 3 granularity levels. The output vector is [0.2, 0.3, 0.5]. This means that the document with the highest granularity level has the highest weight (0.5), and the document with the lowest granularity level has the lowest weight (0.2). The weights are normalized to sum up to 1.
    
    The final score of the document is the weighted sum of the scores from different granularity levels.
    '''
    def __init__(self, query, output_dim=4, device='cuda',exp_option=None):
        super(Router, self).__init__()
        self.query = query
        self.output_dim = output_dim
        self.device = device
        self.exp_option = exp_option
        
        self.query_encoder = SentenceTransformer('stsb-roberta-large', cache_folder = '***/pt_models/')
        sample_query_embedding = self.query_encoder.encode(self.query)
        
        # The query_embedding goes through an MLP to get the weight vector
        # Define the MLP in the middle
        def MLP(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 2*input_dim),
                nn.ReLU(),
                nn.Linear(2*input_dim, 8*input_dim),
                nn.ReLU(),
                nn.Linear(8*input_dim, 32*input_dim),
                nn.ReLU(),
                nn.Linear(32*input_dim, 4*input_dim),
                nn.ReLU(),
                nn.Linear(4*input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.Softmax(dim = -1)
            )
            
        # backwards compartible code for exp140's evaluation
        def MLP_140(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.Softmax(dim = -1)
            )
        
        if self.exp_option == 'exp140':
            self.mlp = MLP_140(sample_query_embedding.shape[0], self.output_dim)
            
        else:
            self.mlp = MLP(sample_query_embedding.shape[0], self.output_dim)
    
        self.mlp = self.mlp.to(self.device)
        
    def run(self, query):
        # Run the Router model
        # The output is the weight vector
        query_embedding = torch.tensor(self.query_encoder.encode(query))
        query_embedding = query_embedding.to(self.device)
        weight_vector = self.mlp(query_embedding).to(self.device)
        return weight_vector
    
class simLoss(nn.Module):
    def __init__(self, sim_option='tfidf'):
        super(simLoss, self).__init__()
        self.query_encoder = SentenceTransformer('stsb-roberta-large', cache_folder = '***/pt_models/')
        self.vectorizer = TfidfVectorizer()
        self.sim_option = sim_option #["tfidf", "roberta"]

    def forward(self, snippet, context):
        loss_total = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        loss_counter = 0
        if "tfidf" in self.sim_option.lower():
            # version with tfidf
            tfidf_matrix = self.vectorizer.fit_transform([snippet, context])
            similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            
            loss = 1 - similarity_score
            loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
            
            loss_total = loss_total + loss
            loss_counter += 1
            
        if "roberta" in self.sim_option.lower():
            # verison with reboerta
            snippet_embedding = self.query_encoder.encode(snippet).reshape(1, -1)
            context_embedding = self.query_encoder.encode(context).reshape(1, -1)
            similarity_score = cosine_similarity(snippet_embedding, context_embedding)[0][0]
            
            loss = 1 - similarity_score
            loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
            
            loss_total = loss_total + loss
            loss_counter += 1
            
        if "hitrate" in self.sim_option.lower():
            # version with hitrate
            snippet_tokens = set(snippet.split())
            context_tokens = set(context.split())
            intersection = snippet_tokens.intersection(context_tokens)
            
            ratio = len(intersection) / len(context_tokens)
            loss = max(0.01, min(1 - ratio, 0.99))
            loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
            
            loss_total = loss_total + loss
            loss_counter += 1

            
        final_loss = loss_total/loss_counter
        return final_loss # return the average loss
    
    
    
class softLabel():
    def __init__(self, retrieval_result_path=None, sim_option=None):
        self.retrieval_result_path = retrieval_result_path
        self.sim_option = sim_option
        self.query_encoder = SentenceTransformer('stsb-roberta-large', cache_folder = '***/pt_models/')
        self.vectorizer = TfidfVectorizer()
        
    def build(self, medmcqa_path=None, bioasq_path=None, pubmedqa_path=None, medqa_path=None, mmlu_path=None):
        dataset_totrain = []
        if medmcqa_path is not None:
            dataset_totrain.append("medmcqa")
        if bioasq_path is not None:
            dataset_totrain.append("bioasq")
        if pubmedqa_path is not None:
            dataset_totrain.append("pubmedqa")
        if medqa_path is not None:
            dataset_totrain.append("medqa")
        if mmlu_path is not None:
            dataset_totrain.append("mmlu")
        cache_file_path = os.path.join(self.retrieval_result_path, f'cache_soft_labels_{self.sim_option}_{dataset_totrain}.json')
        if os.path.exists(cache_file_path):
            return cache_file_path
        
        else:
            # build the soft labels
            retrieval_result_path = self.retrieval_result_path
            if 'tfidf' in self.sim_option.lower():
                vectorizer = self.vectorizer
            
                router_dataset = load_rawdata(medmcqa_path=medmcqa_path, bioasq_path=bioasq_path, pubmedqa_path=pubmedqa_path, medqa_path=medqa_path, mmlu_path=mmlu_path, retrieval_result_path = retrieval_result_path, sim_option=self.sim_option)
            
                data_loader = DataLoader(router_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
                
                
                json_list = []
                for batch_idx, batch in tqdm(enumerate(data_loader), total = len(router_dataset), desc='Building soft labels with tfidf'):
                    questions, labels, snippets, scores = batch
                    
                    corpora_snippets = snippets[0][0]
                    top_snippet_list = []
                    for snip_list in corpora_snippets:
                        top_snippet_list.append(snip_list[0])
                        
                    sim_score_list = [0] * len(top_snippet_list)
                    
                    for i in range(len(top_snippet_list)):
                        t_snip = top_snippet_list[i]
                        if t_snip != "NO_TEXT_RETRIEVED":
                            t_snip = t_snip['contents']
                            tfidf_matrix = vectorizer.fit_transform([t_snip, labels[0]])
                            similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                            sim_score_list[i] = similarity_score
                            
                    def transform_sim_scores(sim_score_list):
                        #random index for all 0
                        if all(score == 0 for score in sim_score_list):
                            indices = list(range(len(sim_score_list)))
                            random_indices = random.sample(indices,2)
                            max_index = random_indices[0]
                            second_max_index = random_indices[1]
                        else:
                            max_value = max(sim_score_list)
                            max_index = sim_score_list.index(max_value)
                            second_max_value = max(sim_score_list[:max_index] + sim_score_list[max_index+1:])
                            second_max_index = sim_score_list.index(second_max_value)
                        new_list = [0] * len(sim_score_list)
                        new_list[max_index] = 0.8
                        new_list[second_max_index] = 0.2
                        return new_list

                    
                    soft_label = transform_sim_scores(sim_score_list)
                    j_object = {
                        "question": questions[0],
                        "labels": labels[0],
                        "retrieved_snippets": snippets[0],
                        "scores": scores[0],
                        "soft_label": soft_label
                    }
                    json_list.append(j_object)
                
                # after all the iteration, write back to the retrieval_result_path
                with open(cache_file_path, 'w') as f:
                    json.dump(json_list, f)
                    
            if 'roberta' in self.sim_option.lower():
                query_encoder = self.query_encoder
                
                router_dataset = load_rawdata(medmcqa_path=medmcqa_path, bioasq_path=bioasq_path, pubmedqa_path=pubmedqa_path, medqa_path=medqa_path, mmlu_path=mmlu_path, retrieval_result_path = retrieval_result_path, sim_option=self.sim_option)
            
                data_loader = DataLoader(router_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
                
                
                json_list = []
                for batch_idx, batch in tqdm(enumerate(data_loader), total = len(router_dataset), desc='Building soft labels with roberta'):
                    questions, labels, snippets, scores = batch
                    
                    corpora_snippets = snippets[0][0]
                    top_snippet_list = []
                    for snip_list in corpora_snippets:
                        top_snippet_list.append(snip_list[0])
                        
                    sim_score_list = [0] * len(top_snippet_list)
                    
                    for i in range(len(top_snippet_list)):
                        t_snip = top_snippet_list[i]
                        if t_snip != "NO_TEXT_RETRIEVED":
                            t_snip = t_snip['contents']                     
                            snippet_embedding = query_encoder.encode(t_snip).reshape(1, -1)
                            context_embedding = query_encoder.encode(labels[0]).reshape(1, -1)
                            similarity_score = cosine_similarity(snippet_embedding, context_embedding)[0][0]
                            sim_score_list[i] = similarity_score
                    
                    def transform_sim_scores(sim_score_list):
                        #random index for all 0
                        if all(score == 0 for score in sim_score_list):
                            indices = list(range(len(sim_score_list)))
                            random_indices = random.sample(indices,2)
                            max_index = random_indices[0]
                            second_max_index = random_indices[1]
                        else:
                            max_value = max(sim_score_list)
                            max_index = sim_score_list.index(max_value)
                            second_max_value = max(sim_score_list[:max_index] + sim_score_list[max_index+1:])
                            second_max_index = sim_score_list.index(second_max_value)
                        new_list = [0] * len(sim_score_list)
                        new_list[max_index] = 0.8
                        new_list[second_max_index] = 0.2
                        return new_list
                        
                    soft_label = transform_sim_scores(sim_score_list)
                    j_object = {
                        "question": questions[0],
                        "labels": labels[0],
                        "retrieved_snippets": snippets[0],
                        "scores": scores[0],
                        "soft_label": soft_label
                    }
                    json_list.append(j_object)
                
                # after all the iteration, write back to the retrieval_result_path
                with open(cache_file_path, 'w') as f:
                    json.dump(json_list, f)
                
            if 'hitrate' in self.sim_option.lower():
                vectorizer = TfidfVectorizer()
            
                router_dataset = load_rawdata(medmcqa_path=medmcqa_path, bioasq_path=bioasq_path, pubmedqa_path=pubmedqa_path, medqa_path=medqa_path, mmlu_path=mmlu_path, retrieval_result_path = retrieval_result_path, sim_option=self.sim_option)
            
                data_loader = DataLoader(router_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
                
                
                json_list = []
                for batch_idx, batch in tqdm(enumerate(data_loader), total = len(router_dataset), desc='Building soft labels with hitrate'):
                    questions, labels, snippets, scores = batch
                    
                    corpora_snippets = snippets[0][0]
                    top_snippet_list = []
                    for snip_list in corpora_snippets:
                        top_snippet_list.append(snip_list[0])
                        
                    sim_score_list = [0] * len(top_snippet_list)
                    
                    for i in range(len(top_snippet_list)):
                        t_snip = top_snippet_list[i]
                        if t_snip != "NO_TEXT_RETRIEVED":
                            t_snip = t_snip['contents']
                            snippet_tokens = set(t_snip.split())
                            context_tokens = set(labels[0].split())
                            intersection = snippet_tokens.intersection(context_tokens)
                            similarity_score = len(intersection) / len(context_tokens)
                            sim_score_list[i] = similarity_score
                            
                    def transform_sim_scores(sim_score_list):
                        #random index for all 0
                        if all(score == 0 for score in sim_score_list):
                            indices = list(range(len(sim_score_list)))
                            random_indices = random.sample(indices,2)
                            max_index = random_indices[0]
                            second_max_index = random_indices[1]
                        else:
                            max_value = max(sim_score_list)
                            max_index = sim_score_list.index(max_value)
                            second_max_value = max(sim_score_list[:max_index] + sim_score_list[max_index+1:])
                            second_max_index = sim_score_list.index(second_max_value)
                        new_list = [0] * len(sim_score_list)
                        new_list[max_index] = 0.8
                        new_list[second_max_index] = 0.2
                        return new_list
                    
                    soft_label = transform_sim_scores(sim_score_list)
                    j_object = {
                        "question": questions[0],
                        "labels": labels[0],
                        "retrieved_snippets": snippets[0],
                        "scores": scores[0],
                        "soft_label": soft_label
                    }
                    json_list.append(j_object)
                
                # after all the iteration, write back to the retrieval_result_path
                with open(cache_file_path, 'w') as f:
                    json.dump(json_list, f)
                