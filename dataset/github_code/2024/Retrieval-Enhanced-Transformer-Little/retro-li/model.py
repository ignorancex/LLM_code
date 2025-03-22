import math
import numpy as np
from typing import Set
import torch
from torch import nn
from rmsnorm_torch import RMSNorm
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import yaml
from collections import OrderedDict

try:
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.simulator.configs import (
        InferenceRPUConfig,
        WeightNoiseType,
        WeightClipType,
        WeightModifierType,
    )
    from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
    from aihwkit.simulator.rpu_base import cuda
except ImportError:
    print("AIHWKIT not available")




class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.theta = nn.Parameter(1. / (base ** (torch.arange(0, d, 2).float() / d)), requires_grad=False)
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, n_heads, d = x.shape
        d_2 = d // 2
        seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)
        idx_theta = torch.einsum('n,d->nd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        neg_half_x = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
        rx = (x * idx_theta2.cos()[None, :, None, :]) + (neg_half_x * idx_theta2.sin()[None, :, None, :])
        return rx
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, is_causal: bool, dropout_rate):
        super().__init__()
        self.is_causal, self.n_heads,self.d_k, self.dropout_rate, self.scale, self.query, self.key, self.value, self.norm, self.softmax, self.rotary_pe, self.output = is_causal, n_heads, d_k, dropout_rate, 1 / math.sqrt(d_k), nn.Linear(d_model, n_heads * d_k), nn.Linear(d_model, n_heads * d_k), nn.Linear(d_model, n_heads * d_k), RMSNorm(d_model), nn.Softmax(dim=-1), RotaryPositionalEmbeddings(d_k), nn.Linear(n_heads * d_k, d_model)
    def mask_attention(self, attn: torch.Tensor):
        if not self.is_causal:
            return attn
        mask = torch.tril(attn.new_ones(attn.shape[-2:]))
        return attn.masked_fill(mask == 0, float('-inf'))
    def forward(self, h: torch.Tensor):
        h_res = h
        h = self.norm(h)
        mh_shape = (*h.shape[:-1], self.n_heads, self.d_k)
        q, k, v = self.query(h).view(mh_shape), self.key(h).view(mh_shape), self.value(h).view(mh_shape)
        q, k = self.rotary_pe(q), self.rotary_pe(k)
        attn = torch.einsum('bihd,bjhd->bhij', q, k)
        attn = attn * self.scale
        attn = self.mask_attention(attn)
        attn = self.softmax(attn)
        h = torch.einsum("bhij,bjhd->bihd", attn, v)
        h = h.reshape(*h.shape[:-2], -1)
        h = self.output(h)
        return h + h_res
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, dropout_rate):
        super().__init__()
        self.n_heads, self.d_k, self.dropout_rate, self.scale, self.query, self.key, self.value, self.norm, self.softmax, self.output = n_heads, d_k, dropout_rate, 1 / math.sqrt(d_k), nn.Linear(d_model, n_heads * d_k), nn.Linear(d_model, n_heads * d_k), nn.Linear(d_model, n_heads * d_k), RMSNorm(d_model), nn.Softmax(dim=-1), nn.Linear(n_heads * d_k, d_model)
    def forward(self, e: torch.Tensor, h: torch.Tensor):
        e_res = e
        e,q,k,v = self.norm(e), self.query(e).view(*e.shape[:-1], self.n_heads, self.d_k), self.key(h).view(*h.shape[:-1], self.n_heads, self.d_k), self.value(h).view(*h.shape[:-1], self.n_heads, self.d_k)
        attn = torch.einsum('bcnihd,bcjhd->bcnhij', q, k)
        attn = attn * self.scale
        attn = self.softmax(attn)
        e = torch.einsum("bcnhij,bcjhd->bcnihd", attn, v)
        e = e.reshape(*e.shape[:-2], -1)
        e = self.output(e)
        return e + e_res
class ChunkedCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, chunk_len: int, dropout_rate):
        super().__init__()
        self.chunk_len, self.n_heads, self.d_k, self.dropout_rate, self.scale, self.query, self.key, self.value, self.norm, self.softmax, self.output = chunk_len, n_heads, d_k, dropout_rate, 1 / math.sqrt(d_k), nn.Linear(d_model, n_heads * d_k), nn.Linear(d_model, n_heads * d_k), nn.Linear(d_model, n_heads * d_k), RMSNorm(d_model), nn.Softmax(dim=-1), nn.Linear(n_heads * d_k, d_model)
    def forward(self, h: torch.Tensor, e: torch.Tensor):
        batch_size, chunks, neighbors, neighbor_len, d_model = e.shape
        if chunks == 0:
            return h
        h_res = h
        h = h[:, self.chunk_len - 1:]
        h = self.norm(h)
        if h.shape[1] < chunks * self.chunk_len:
            h = torch.cat((h, h.new_zeros(batch_size, chunks * self.chunk_len - h.shape[1], d_model)), dim=1)
        h = h.reshape(batch_size, chunks, self.chunk_len, d_model)
        q,k,v = self.query(h).view(*h.shape[:-1], self.n_heads, self.d_k), self.key(e).view(*e.shape[:-1], self.n_heads, self.d_k), self.value(e).view(*e.shape[:-1], self.n_heads, self.d_k)
        attn = torch.einsum('bcihd,bcnjhd->bchinj', q, k)
        attn = attn * self.scale
        attn = self.softmax(attn.view(*attn.shape[:-2], -1)).view(attn.shape)
        h = torch.einsum("bchinj,bcnjhd->bcihd", attn, v)
        h = h.reshape(batch_size, chunks * self.chunk_len, -1)
        h = self.output(h)
        h = torch.cat((h.new_zeros(batch_size, self.chunk_len - 1, d_model), h), dim=1)
        return h[:, :h_res.shape[1]] + h_res
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.norm = RMSNorm(d_model)
    def forward(self, h: torch.Tensor):
        h_res = h
        h = self.norm(h)
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        return h + h_res
class NearestNeighborEncoder(nn.Module):
    def __init__(self, chunk_len: int, n_layers: int, ca_layers: Set[int],d_model: int, n_heads: int, d_k: int, d_ff: int, retro_flag):
        super().__init__()
        self.ca_layers = ca_layers
        self.chunk_len = chunk_len
        self.retro_flag = retro_flag
        self.ca = nn.ModuleList([CrossAttention(d_model, n_heads, d_k, 0) for _ in range(len(ca_layers))])
        self.attn = nn.ModuleList([SelfAttention(d_model, n_heads, d_k, is_causal=False, dropout_rate=0) for _ in range(n_layers)])
        self.ffw = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_layers)])
        self.norm_h = RMSNorm(d_model)
    def forward(self, e: torch.Tensor, h: torch.Tensor):
        batch_size, chunks, neighbors, neighbor_len, d_model = e.shape
        h_split = h[:, :self.chunk_len * chunks, :].reshape(batch_size, chunks, self.chunk_len, d_model)
        h_split = self.norm_h(h_split)
        p_ca = 0
        for p in range(len(self.attn)):
            e = self.attn[p](e.view(-1, neighbor_len, d_model)).view(e.shape)
            if self.retro_flag:
                if p in self.ca_layers:
                    e = self.ca[p_ca](e, h_split)
                    p_ca += 1
            e = self.ffw[p](e)
        return e
class RetroFittedGPT2(nn.Module):
    def __init__(self, ca_layers: Set[int], chunk_len: int,d_k: int, d_ff: int, encoder: NearestNeighborEncoder, retro_flag, gpt2_config = None, gpt2_finetuned_path=None, noisy_embed_sequence=False, noisy_embed_neighbors=False, noise_alpha=0, noise_coeff=None,is_aihwkit=0):
        super().__init__()
        if gpt2_finetuned_path:
            self.transformer = GPT2Model.from_pretrained(gpt2_finetuned_path) 
        else:
            self.transformer = GPT2Model.from_pretrained('gpt2')
        if gpt2_config == None:
            raise Error("Please specify the GPT2 configurations file.")
        else:
            self.config = gpt2_config
        self.transformer.init_weights()
        self.noisy_embed_sequence = noisy_embed_sequence
        self.noisy_embed_neighbors = noisy_embed_neighbors
        self.noise_alpha = noise_alpha
        self.noise_coeff = noise_coeff
        if noise_coeff:
            self.gauss = True
        else:
            self.gauss = False
        self.ca_layers = ca_layers
        self.encoder = encoder
        self.retro_flag = retro_flag
        self.chunk_len = chunk_len
        self.gpt2_finetuned_path = gpt2_finetuned_path
        self.cca = nn.ModuleList(
            [ChunkedCrossAttention(gpt2_config.n_embd, gpt2_config.n_head, d_k, chunk_len, 0.0) for _ in range(len(ca_layers))])
        self.ffw = nn.ModuleList([FeedForward(gpt2_config.n_embd, d_ff) for _ in range(len(self.transformer.h))])
        self.read = nn.Linear(gpt2_config.n_embd, gpt2_config.vocab_size)
        self.is_aihwkit = is_aihwkit
        if self.is_aihwkit:
            self.rpu_config = InferenceRPUConfig()
            self.rpu_config.forward.out_res = -1.0  # Turn off (output) ADC discretization.
            self.rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
            self.rpu_config.forward.w_noise = 0.02  # Short-term w-noise.

            self.rpu_config.clip.type = WeightClipType.FIXED_VALUE
            self.rpu_config.clip.fixed_value = 1.0
            self.rpu_config.modifier.pdrop = 0.03  # Drop connect.
            self.rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
            self.rpu_config.modifier.std_dev = 0.1
            self.rpu_config.modifier.rel_to_actual_wmax = True
            #retro specifc setting to avoid split into multi tile
            self.rpu_config.mapping.max_output_size = 768
            self.rpu_config.mapping.max_input_size = 768

            # Inference noise model.
            self.rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

            # drift compensation
            self.rpu_config.drift_compensation = GlobalDriftCompensation()

    def noised_embed_aihwkit(self, x, validate=False):
        torch.autograd.set_detect_anomaly(True)
        h = self.transformer.wte(x)
        noisy_h = torch.zeros([0, h.shape[-1]], dtype=h.dtype, device=h.device)
        for index in np.ndindex(tuple(h.shape[:-2])):
            retr_weights = torch.transpose(h[index].view(-1,h.shape[-1]),0,1)
            retrive_model = nn.Linear(retr_weights.shape[1],retr_weights.shape[0])
            retrive_model.weight.data = retr_weights.to("cpu")
            retrive_model.to(h.device)
            inputs_vecs = torch.eye(retr_weights.shape[1]).to(h.device)
            with torch.no_grad():
                retrive_model.bias.fill_(0.)
                model = convert_to_analog(retrive_model, self.rpu_config)
                model.eval()
                model.drift_analog_weights(0)
                dist = model(inputs_vecs)
                noisy_h = torch.cat((noisy_h,dist),0)
        return noisy_h.view(h.shape)
        
            
    def noised_embed_gauss(self, x, coeff, validate=False):
        h = self.transformer.wte(x)
        if not validate:
            mu = torch.mean(torch.abs(h).to(h.device),dtype=torch.float64).to(h.device)
            sigma = coeff*mu
            gaussian_noise = torch.empty(h.shape, device=h.device).normal_(mean=0,std=(sigma.to(h.dtype)))
            return h + gaussian_noise
        else:
            return h
    def noised_embed(self, x, noise_alpha, validate=False): #alpha is a hyperparameter that can / should be tuned
        h = self.transformer.wte(x)
        if not validate:
            dims = torch.tensor(h.size(-2) * h.size(-1)) #L x d
            mag_norm = noise_alpha/torch.sqrt(dims)
            return h + torch.zeros_like(h).uniform_(-mag_norm, mag_norm)
        else:
            return h
    def forward(self, x: torch.Tensor, ret: torch.Tensor, past=None, validate=False, x_attention_mask = None, ret_attention_mask = None,is_aihwkit=0):
        batch_size = x.shape[0]
        if self.noisy_embed_sequence:
            if self.gauss:
                h = self.noised_embed_gauss(x, validate=validate)
            else:
                h = self.noised_embed(x, noise_alpha=self.noise_alpha, validate=validate)
        else:
            h = self.transformer.wte(x)
        if self.retro_flag:
            if self.noisy_embed_neighbors:
                if self.is_aihwkit:
                    ret_emb = self.noised_embed_aihwkit(ret, validate=validate)

                else:
                    if self.gauss:
                        ret_emb = self.noised_embed_gauss(ret, coeff=self.noise_coeff, validate=validate)
                    else:
                        ret_emb = self.noised_embed(ret, noise_alpha=self.noise_alpha, validate=validate)
            else:
                ret_emb = self.transformer.wte(ret) #we still use the same embedding for both, so still wte.
        else:
            ret_emb = None
        p_ca = 0
        if past is None:
            past_length = 0
            past = [None] * len(self.transformer.h)
        for p, (block, layer_past) in enumerate(zip(self.transformer.h, past)):
            outputs = block(h,layer_past=layer_past,attention_mask=x_attention_mask,use_cache=self.config.use_cache,output_attentions=self.config.output_attentions)
            h, present = outputs[:2]
            if self.retro_flag:
                if self.ca_layers and p == min(self.ca_layers):
                    e = self.encoder(ret_emb, h)
                    e = self.transformer.ln_f(e)
                if p in self.ca_layers:
                    h = self.cca[p_ca](h, e)
                    p_ca += 1
            h = self.ffw[p](h)
        return self.read(h)
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, is_aihwkit=0):
        config_file_path = "/".join(pretrained_model_name_or_path.split("/")[:-1])
        with open(config_file_path+'/run.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        retrieval_on = True if (config_dict["config_json"]["retro"] == "On") else False
        if "noisy_embed" in config_dict["config_json"]:#this was the old way we added noise, no longer valid as it does not specify where the noise is added
            raise Exception("This way of adding noise is no longer valid.")
        noisy_embed_sequence = True if ("noisy_embed_sequence" in config_dict["config_json"] and config_dict["config_json"]["noisy_embed_sequence"]== "True") else False
        noisy_embed_neighbors = True if ("noisy_embed_neighbors" in config_dict["config_json"] and config_dict["config_json"]["noisy_embed_neighbors"]== "True") else False
        noise_alpha = int(config_dict["config_json"]["noise_alpha"]) if ("noise_alpha" in config_dict["config_json"]) else None
        noise_coeff = float(config_dict["config_json"]["noise_coeff"]) if ("noise_coeff" in config_dict["config_json"]) else None
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        gpt2_config = GPT2Config()
        nearest_neighbor_encoder = NearestNeighborEncoder(config_dict["config_json"]["chunk_len"], config_dict["config_json"]["n_encoder_layers"], config_dict["config_json"]["encoder_ca_layers"], gpt2_config.n_embd, config_dict["config_json"]["n_heads"], config_dict["config_json"]["d_k"], config_dict["config_json"]["d_ff"], retro_flag=retrieval_on)
        model = RetroFittedGPT2(ca_layers=config_dict["config_json"]["decoder_ca_layers"], chunk_len=config_dict["config_json"]["chunk_len"],d_k=config_dict["config_json"]["d_k"], d_ff=config_dict["config_json"]["d_ff"], encoder=nearest_neighbor_encoder,retro_flag=retrieval_on, gpt2_config=gpt2_config, noisy_embed_sequence=noisy_embed_sequence,noisy_embed_neighbors=noisy_embed_neighbors, noise_alpha=noise_alpha, noise_coeff=noise_coeff,is_aihwkit=is_aihwkit)
        f = open(pretrained_model_name_or_path, 'rb')
        state_dict = torch.load(f, map_location=torch.device(device))["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == "module.":
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)# load params
        return model
if __name__ == '__main__':
    pass