import torch
import torch.nn as nn
import math
from einops import rearrange
import sys

from TimeTransformer.encoder import Encoder, CrossAttention_Encoder, AdaIN_Encoder
from TimeTransformer.decoder import Decoder
from TimeTransformer.utils import generate_original_PE, generate_regular_PE
import TimeTransformer.causal_convolution_layer as causal_convolution_layer
import torch.nn.functional as F

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



class Transformer1(nn.Module):
    '''
    good transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_condition_emb_1:int,
                 d_condition_emb_2:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 spatialloc: list,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 ifkg: bool = True,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.condition_emb_1_dim = d_condition_emb_1
        self.condition_emb_2_dim = d_condition_emb_2
        self.channels = d_input
        self.ifkg = ifkg
        self.spatialloc = spatialloc
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.condition_emb_1_mlp = nn.Sequential(
            nn.Linear(self.condition_emb_1_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.condition_emb_2_mlp = nn.Sequential(
            nn.Linear(self.condition_emb_2_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timelinear = nn.Linear(self.condition_emb_2_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(d_model, d_model) 
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition_emb_1: torch.Tensor, condition_emb_2: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) 

        
        condition_emb_2 = self.timelinear(condition_emb_2)
        condition_emb_2 = condition_emb_2.unsqueeze(1)
        condition_emb_2 = torch.repeat_interleave(condition_emb_2, self.layernum, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])

        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)
        encoding.add_(condition_emb_2)
        if self.ifkg:
            condition_emb_1 = self.condition_emb_1_mlp(condition_emb_1)
            condition_emb_1 = condition_emb_1.unsqueeze(1)        
            condition_emb_1 = torch.repeat_interleave(condition_emb_1, 160, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])
            encoding[:, self.spatialloc[0]:self.spatialloc[1], :].add_(condition_emb_1)
        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)   # torch.Size([8, 64])  
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        output = self._linear(encoding)

        return output.permute(0,2,1)

   
class Transformer2(nn.Module):
    '''
    Conditions are added to each layer of transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_condition_emb_1:int,
                 d_condition_emb_2:int,
                 N: int,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,  
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.condition_emb_1_dim = d_condition_emb_1
        self.condition_emb_2_dim = d_condition_emb_2
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout) 
            for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model) # channel -> 512
        self._linear = nn.Linear(d_model, d_output) # 512 -> channel

        # 选一个位置编码函数
        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.condition_emb_1_mlp = nn.Sequential(  
            nn.Linear(self.condition_emb_1_dim, d_model), # [128 * 512]
        ) 
        
        self.condition_emb_2_mlp = nn.Sequential( 
            nn.Linear(self.condition_emb_2_dim, d_model), # [128 * 512]
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, condition_emb_1: torch.Tensor, condition_emb_2: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) # x shape:  torch.Size([1, 18, 1024])
        # [num , 模型参数量 / channel , channel]

        condition_emb_1 = self.condition_emb_1_mlp(condition_emb_1)
        condition_emb_1 = condition_emb_1.unsqueeze(1)        
        condition_emb_1 = torch.repeat_interleave(condition_emb_1, self.layernum, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])
        # [num , 模型参数量 / channel , 512]
        
        condition_emb_2 = self.condition_emb_2_mlp(condition_emb_2)
        condition_emb_2 = condition_emb_2.unsqueeze(1)
        condition_emb_2 = torch.repeat_interleave(condition_emb_2, self.layernum, dim=1)   # regionemb.shape torch.Size([1, 6000, 64])
        # [num , 模型参数量 / channel , 512]

        step = self.step_mlp(t)  
        step = step.unsqueeze(1) 
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2) # [num , 模型参数量 / channel , 512]
        encoding.add_(step_emb)
        
        condition = condition_emb_1 + condition_emb_2

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding.add_(condition)  # each layer
            encoding = layer(encoding)

        output = self._linear(encoding) # [num , 模型参数量 / channel , channel]
        
        return output.permute(0,2,1) # [num , 模型参数量 / channel , channel]
    
class Transformer3(nn.Module):
    '''
    After the conditions are aggregated, they are added to each layer of transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_condition_emb_1:int,
                 d_condition_emb_2:int,
                 N: int,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.condition_emb_1_dim = d_condition_emb_1
        self.condition_emb_2_dim = d_condition_emb_2
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout) 
            for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.condition_emb_1_linear = nn.Linear(self.condition_emb_1_dim, d_model)
        
        self.condition_emb_2_linear = nn.Linear(self.condition_emb_2_dim, d_model)
        
        self.forQueryFunc = nn.Sequential(  
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, condition_emb_1: torch.Tensor, condition_emb_2: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # x.shape:  [64, 64, 265] <- [batchsize, channel, length]
        xEmb = self._embedding(x.permute(0,2,1))                # xEmb.shape [64, 265, 256]

        condition_emb_1 = self.condition_emb_1_linear(condition_emb_1)                        # [64, 256]
        condition_emb_1 = condition_emb_1.unsqueeze(2)                              # [64, 256, 1]
        
        condition_emb_2 = self.condition_emb_2_linear(condition_emb_2)                  # [64, 256]
        condition_emb_2 = condition_emb_2.unsqueeze(2)                          # [64, 256, 1]
        
        condition_emb = torch.cat((condition_emb_1, condition_emb_2), 2)                 # condition_emb [64, 256, 2]
        
        xQuery = self.forQueryFunc(xEmb)                        # xQuery [64, 265, 256]
        
        # [64, 265, 256] * [64, 256, 2] -> [64, 265, 2]        
        score = torch.bmm(xQuery, condition_emb)                       # score.shape [64, 265, 2]
        score = F.softmax(score, dim = 2)
        
        # [64, 265, 2] * [64, 256, 2] -> [64, 265, 256]
        condition = torch.bmm(score, torch.transpose(condition_emb, 1, 2))  # condition: [64, 265, 256]  


        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # prepare embedding into encoder
        encoding = xEmb
        encoding = encoding + step_emb
        
        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)   # torch.Size([8, 64]) 
            encoding.add_(positional_encoding)  

        # Encoder stack
        for layer in self.layers_encoding:
            encoding = encoding + condition  
            encoding = layer(encoding)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
      
class Transformer4(nn.Module):
    '''
        cross attention
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_condition_emb_1:int,
                 d_condition_emb_2:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,  
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.condition_emb_1_dim = d_condition_emb_1
        self.condition_emb_2_dim = d_condition_emb_2
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([CrossAttention_Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.condition_emb_1_linear = nn.Linear(self.condition_emb_1_dim, d_model)
        
        self.condition_emb_2_linear = nn.Linear(self.condition_emb_2_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, condition_emb_1: torch.Tensor, condition_emb_2: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) # x shape:  torch.Size([1, 18, 1024])

        condition_emb_1 = self.condition_emb_1_linear(condition_emb_1)                        # [64, 256]
        condition_emb_1 = condition_emb_1.unsqueeze(2)                              # [64, 256, 1]
        
        condition_emb_2 = self.condition_emb_2_linear(condition_emb_2)                  # [64, 256]
        condition_emb_2 = condition_emb_2.unsqueeze(2)                          # [64, 256, 1]
        
        condition_emb = torch.cat((condition_emb_1, condition_emb_2), 2)                 # condition_emb [64, 256, 2]

        step = self.step_mlp(t)  
        step = step.unsqueeze(1) 
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)   
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, condition_emb)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
    
class Transformer5(nn.Module):
    '''
        Adaptive LayerNorm
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_condition_emb_1:int,
                 d_condition_emb_2:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.condition_emb_1_dim = d_condition_emb_1
        self.condition_emb_2_dim = d_condition_emb_2
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([AdaIN_Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.condition_emb_1_linear = nn.Linear(self.condition_emb_1_dim, d_model)
        
        self.condition_emb_2_linear = nn.Linear(self.condition_emb_2_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, condition_emb_1: torch.Tensor, condition_emb_2: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape) #torch.Size([4, 128, 759]) 
        x2 = x.permute(0,2,1) # x shape:  torch.Size([1, 18, 1024])
        
        condition_emb = torch.cat((condition_emb_1, condition_emb_2), 1)            

        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, condition_emb)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)    


