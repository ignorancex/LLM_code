from typing import ForwardRef
import torch
import torch.nn as nn
import os
import sys

class RankingCNN(nn.Module):
    def __init__(self, num_filters = 50,emb_dim=200,word_context_size=2, batch_size = 24) -> None:
        super().__init__()
        # input is concated word embeddings of size d * n_words
        self.conv_m  =  nn.Conv2d(1,num_filters,(emb_dim,word_context_size))
        self.conv_y  =  nn.Conv2d(1,num_filters,(emb_dim,word_context_size))
        self.M = nn.parameter.Parameter(torch.rand(batch_size, emb_dim,emb_dim), requires_grad=True)
        self.fc = nn.Linear(2*emb_dim+1, 2*emb_dim+1)
    
    def forward(self, emb_m, emb_y):
        v_m = self.conv_m(emb_m)
        v_y = self.conv_y(emb_y)

        v_m = torch.max(v_m,3).values
        v_y = torch.max(v_y,3).values
   
        v_sim = torch.bmm(torch.transpose(v_m,1,2),torch.bmm(self.M,v_y))
     
        v_mor = 0
        v_joint = torch.cat([v_m,v_y,v_sim],dim=1)
        out = self.fc(v_joint)

        return 0


        
