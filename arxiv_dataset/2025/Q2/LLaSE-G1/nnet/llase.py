
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Union, Optional
from transformers import LlamaConfig, LlamaForCausalLM

NUM_AUDIO_TOKENS = 65536 # Codebook size

class LLM_AR(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int
    ):
        super().__init__()
        self.d_model = d_model

        self.audio_linear_y = nn.Linear(1024, d_model)
        self.audio_linear_x = nn.Linear(1024, d_model)

        self.Llama_config = LlamaConfig(
            hidden_size=d_model*2,
            intermediate_size=d_model * 4,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            dropout_rate=0.1,
            attention_dropout=0.1,
            is_decoder=True,
            use_cache=True
        )

        self.llama= LlamaForCausalLM(config=self.Llama_config)
        self.predict_layer_x = nn.Linear(2*d_model, NUM_AUDIO_TOKENS)
        self.predict_layer_y = nn.Linear(2*d_model, NUM_AUDIO_TOKENS)

    def forward(
        self,
        y: torch.Tensor,
        x: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        # y = y.transpose(1,2) # if codec input use this transpose

        if x is None:
            x = torch.zeros_like(y)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
            x = x.expand_as(y)
            

        y_emb = self.audio_linear_y(y)  # [B, T, D] 
        x_emb = self.audio_linear_x(x)  # [B, T, D]

        if x_emb.shape[1] < y_emb.shape[1]:
            pad_length = y_emb.shape[1] - x_emb.shape[1]
            x_emb= F.pad(x_emb, (0, 0, 0, pad_length), mode='constant', value=0)
            
        if y_emb.shape[1] < x_emb.shape[1]:
            pad_length = x_emb.shape[1] - y_emb.shape[1]
            y_emb= F.pad(y_emb, (0, 0, 0, pad_length), mode='constant', value=0)
            
        y_emb = torch.concat([x_emb, y_emb], dim = -1) # [B, T_y, D*2]
         
        outputs = self.llama(inputs_embeds = y_emb, output_hidden_states=True)

        dec = outputs.hidden_states[-1] # [B, T_y, D*2]
        
        logits_y = self.predict_layer_y(dec)  # [B, T, NUM_AUDIO_TOKENS]
        logits_x = self.predict_layer_x(dec)

        logits_y = logits_y.transpose(-1, -2)  # [B, NUM_AUDIO_TOKENS, T]
        logits_x = logits_x.transpose(-1, -2)

        return logits_y, logits_x

if __name__=="__main__":
    # Simple test
    model = LLM_AR(d_model=1024, nhead=8, num_layers=16)
    ce_loss = nn.CrossEntropyLoss()

    y = torch.randn([1,199,1024])
    x = torch.randn([1,99,1024])
    label = torch.from_numpy(np.random.randint(0, 300, size=[2,1,199])) 

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total Params: {total_params}")

    logits = model(y)
    print(logits[0].shape)
    print(logits[1].shape)

    logits = model(y,x)
    print(logits[0].shape)
    print(logits[1].shape)
    
    logits = model(y,y)
    print(logits[0].shape)
    print(logits[1].shape)
