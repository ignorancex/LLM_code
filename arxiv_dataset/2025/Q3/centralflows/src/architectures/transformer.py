import torch.nn as nn
import torch
from dataclasses import dataclass

from .utils import replace_layernorm, TrainableLayerNorm
from .base import Architecture
import math


@dataclass
class Transformer(Architecture):
    layers : int = 4
    d_model : int = 32
    nhead: int = 4
    replace_ln: bool = True
    weight_multiplier: float = 0.0

    def create(self, input_shape, output_dim):    
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        net = TransformerLM( 
            output_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.layers,
            dim_feedforward=4*self.d_model,
            dropout=0.0,
            max_seq_length=input_shape[0])
    
        if self.replace_ln:
            replace_layernorm(net, TrainableLayerNorm)
        
        with torch.no_grad():
            net.output_layer.weight *= self.weight_multiplier
        
        return net
    
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        dropout=0.0,
        max_seq_length=1024
    ):
        super().__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Create decoder layer that will be used multiple times
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        
        # Create the decoder with multiple layers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final linear layer to project to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
        self.d_model = d_model
        
    def _init_parameters(self):
        """Initialize the parameters using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, size):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, mask=None):
        """
        Args:
            src: Source sequence [batch_size, seq_len]
            mask: Optional mask for the sequence
        """
        # Create mask if not provided (for casual/autoregressive attention)
        if mask is None:
            mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Embed tokens and add positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through decoder
        # Note: For a decoder-only transformer, we pass the same sequence as both memory and tgt
        output = self.decoder(x, x, tgt_mask=mask)
        
        # Project to vocabulary size
        return self.output_layer(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    