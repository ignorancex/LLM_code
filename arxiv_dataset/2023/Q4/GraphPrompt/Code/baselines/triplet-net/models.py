from numpy.core.numeric import indices
import torch
import torch.nn as nn
from transformers import *
import os
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class Biosyn_Model(nn.Module):
    def __init__(self,model_path,initial_sparse_weight,device):
        super(Biosyn_Model,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.to(device)
        self.sparse_weight = nn.Parameter(torch.empty(1).cuda(0))
        self.sparse_weight.data.fill_(initial_sparse_weight)
        
    
    def forward(self,query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score):
        """
        args:
            candidates_names_ids: batch * top_k * max_len
            candidates_names_attention_mask: batch * top_k * max_len
            candidates_sparse_score: batch * top_k
        """
        query_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        candidiate_names_graph_embedding = []
        for i in range(candidates_names_ids.shape[1]):
            ids = candidates_names_ids[:,i,:]
            attention_mask = candidates_names_attention_mask[:,i,:]
            cls_embedding = self.bert_encoder(ids,attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            candidiate_names_graph_embedding.append(cls_embedding)
        candidiate_names_graph_embedding = torch.stack(candidiate_names_graph_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)

        query_embedding = torch.unsqueeze(query_embedding,dim=1)#batch * 1 *hidden_size
        bert_score = torch.bmm(query_embedding, candidiate_names_graph_embedding.transpose(dim0=1,dim1=2)).squeeze()# batch * top_k

        score = bert_score + candidates_sparse_score * self.sparse_weight
        return score

class Graphsage_Model(torch.nn.Module):
    def __init__(self,feature_size,hidden_size,output_size,model_path,initial_sparse_weight,device):
        super(Graphsage_Model,self).__init__()

        #load bert encoder
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.to(device)

        self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
        self.sparse_weight.data.fill_(initial_sparse_weight)

        self.sage1 = GCNConv(feature_size,hidden_size).to(device)
        self.sage2 = GCNConv(hidden_size,output_size).to(device)

        self.score_network = nn.Linear(in_features=768*2,out_features=1).to(device)


    @torch.no_grad()# with pretrained and fine tuned bert model, we could get candidates with about 80 accu_k
    def candidates_retrieve(self,query_ids,query_attention_mask,sparse_score,names_bert_embedding,top_k):
        query_bert_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]# shape of (batch,hidden_size)
        bert_score = torch.matmul(query_bert_embedding,torch.transpose(names_bert_embedding,dim0=0,dim1=1))
        score = self.sparse_weight * sparse_score + bert_score
        sorted_bert_score,candidates_indices =torch.sort(score,descending=True)# descending
        return sorted_bert_score[:,:top_k],candidates_indices[:,:top_k]#tensors of shape(batch,top_k)



    # decide the candidates set in forward set;here we simply decide the candidates by the sum of sparse scores and dense scores
    # in order to choose enouth positive samples, we put the positive samples into candidates set artificially
    def forward(self,query_ids,query_attention_mask,sparse_score,names_bert_embedding,query_indices,edge_index,top_k,is_training=True):
        query_bert_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]# shape of (batch,hidden_size)
        
        names_graph_embedding = self.sage1(names_bert_embedding,edge_index)
        names_graph_embedding = F.relu(names_graph_embedding)
        names_graph_embedding = F.dropout(names_graph_embedding)
        names_graph_embedding = self.sage2(names_graph_embedding,edge_index)# shape of (N, hidden_size)
        sorted_bert_score,candidates_indices = self.candidates_retrieve(
            query_ids=query_ids,query_attention_mask=query_attention_mask,
            sparse_score=sparse_score,names_bert_embedding=names_bert_embedding,top_k=top_k)# tensors of shape(batch,top_k)
        batch_size = query_ids.shape[0]

        if is_training:# put the ground truth index into candidate indices
            query_indices = torch.squeeze(query_indices)
            print('-'*5+'query_indices'+'-'*5)
            print(query_indices)
            print(candidates_indices)
            for i,query_index in enumerate(query_indices):
                if query_index not in candidates_indices[i]:
                    candidates_indices[i][-1] = query_index# replace the last index
        score = []
        for i in range(top_k):
            ith_indices = candidates_indices[:,i]# the ith index for every query, tensor of shape(batch,)
            ith_candidate_graph_embedding = names_graph_embedding[ith_indices]# tensor of shape(batch,hidden)
            ith_score_embedding = torch.cat((ith_candidate_graph_embedding,query_bert_embedding),dim=1)#tensor of shape(batch,hidden*2)
            ith_score = F.relu(self.score_network(ith_score_embedding))#tensor of shape(batch,1)
            score.append(ith_score)
        score = torch.cat(score,dim=1)#tensor fo shape(batch,top_k)# rememebr that the first index is not the predicted result
        
        return score,candidates_indices
        

class Bert_Candidate_Generator(nn.Module):
    def __init__(self,model_path,initial_sparse_weight,device):
        super(Bert_Candidate_Generator,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.to(device)
        self.sparse_weight = nn.Parameter(torch.empty(1).cuda(0))
        self.sparse_weight.data.fill_(initial_sparse_weight)
    
    def load_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.sparse_weight = torch.load(os.path.join(model_path,'sparse_weight.pth'))

    def forward(self,query_ids,query_attention_mask,candidates_ids,candidates_attention_mask,candidates_sparse_score):
        """
        args:
            candidates_names_ids: batch * top_k * max_len
            candidates_names_attention_mask: batch * top_k * max_len
            candidates_sparse_score: batch * top_k
        """
        query_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        candidiate_names_graph_embedding = []
        for i in range(candidates_ids.shape[1]):#top_k
            ids = candidates_ids[:,i,:]
            attention_mask = candidates_attention_mask[:,i,:]
            cls_embedding = self.bert_encoder(ids,attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            candidiate_names_graph_embedding.append(cls_embedding)
        candidiate_names_graph_embedding = torch.stack(candidiate_names_graph_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)

        query_embedding = torch.unsqueeze(query_embedding,dim=1)#batch * 1 *hidden_size
        bert_score = torch.bmm(query_embedding, candidiate_names_graph_embedding.transpose(dim0=1,dim1=2)).squeeze()# batch * top_k

        score = bert_score + candidates_sparse_score * self.sparse_weight
        return score



class SimpleEmbedding(nn.Module):
    def __init__(self):
        super(SimpleEmbedding, self).__init__()
        self.layer1 = nn.Conv1d(1, 1, kernel_size=200 + 1 - 128)
        self.layer1 = nn.Conv1d(1, 1, kernel_size=3, padding=29)
        self.BN = nn.BatchNorm1d(200)
        self.layer2 = nn.PReLU()
        # self.layer3 = nn.MaxPool1d(3, stride=1, padding=1)
        self.layer3 = nn.MaxPool1d(2, stride=2, padding=0)
        # self.layer = nn.Sequential(nn.Conv1d(1, 1, kernel_size=200+1-128), nn.PReLU(), nn.MaxPool1d(2, stride=2, padding=0))

    def forward(self, x):
        # print(x.shape)
        # x = self.BN(x)
        x = torch.unsqueeze(x, 0)
        #y = torch.split(x, 1, dim=1)
        x = torch.transpose(x, 0, 1)
        x = self.layer1(x)
        # print(x.shape)
        # x = torch.transpose(x, 1, 2)
        # print(x.shape)
        # x = self.BN(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)



        
class Bert_Cross_Encoder(nn.Module):
    def __init__(self,model_path,device):
        super(Bert_Cross_Encoder,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.to(device)
        self.linear = nn.Linear(in_features=768,out_features=1).to(device)
    
    def load_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.linear.load_state_dict(torch.load(os.path.join(model_path,'linear.pth')))

    # return corss encoder scores among all candidates(tensor of shape(batch,top_k))
    def forward(self,pair_ids,pair_attn_mask):
        """
        args:
            pair_ids: tensor of shape(batch,top_k,max_len)
            pair_attn_mask: tensor of shape(batch,top_k,max_len)
        """
        score = []
        top_k = pair_ids.shape[1]
        for k in range(top_k):
            ids = pair_ids[:,k,:]
            attn_mask = pair_attn_mask[:,k,:]
            cls_embedding = self.bert_encoder(ids,attn_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            cls_embedding = F.dropout(input=cls_embedding,p=0.5)
            score_k = self.linear(cls_embedding)# tensor of shape(batch,1)
            score.append(score_k)
        score = torch.cat(score,dim=1)#tensor of shape(batch,top_k)
        return score





