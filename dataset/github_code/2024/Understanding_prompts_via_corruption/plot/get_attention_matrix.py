import torch

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM

import surgeon_pytorch as surgeon
import matplotlib.pyplot as plt

import bitsandbytes
import bitsandbytes.autograd._functions

torch.set_grad_enabled(False)


@torch.no_grad()
def compute_attention_norms_opt(attention_weights, value_layer, dense, device=None):
    """
    Args:
        # attention_weights: (num_heads, s, s)  num_heads = all_head_size/head_size
        # value_layer: (s, h)
        # dense: nn.Linear (h, h)
    """
    orig_device = None
    if device is not None:
        orig_device = attention_weights.device
        attention_weights = attention_weights.to(device)
        value_layer = value_layer.to(device)
        dense = dense.to(device)

    seq_len, all_head_size = value_layer.shape
    num_heads = attention_weights.shape[0]
    head_size = int(all_head_size / num_heads)

    # [s, h] -> [h, s] -> [n, h/n, s] -> [n, s, h/n] -> [1, n, s, h/n]
    value_layer = value_layer.transpose(0, 1).reshape(-1, head_size, seq_len).transpose(1, 2).unsqueeze(0).contiguous()

    # value_layer is converted to (batch, seq_length, num_heads, 1, head_size)
    value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
    value_shape = value_layer.size()
    value_layer = value_layer.view(value_shape[:-1] + (1, value_shape[-1],))

    # dense weight is converted to (num_heads, head_size, all_head_size)
    dense_layer = dense
    dense = dense.weight

    if dense.dtype == torch.int8:
        raise NotImplementedError("Quantized models are not supported yet.")
        # self.weight.data = undo_layout(self.state.CxB, self.state.tile_indices)
        dense = bitsandbytes.autograd._functions.undo_layout(dense_layer.state.CxB, dense_layer.state.tile_indices)

    dense = dense.view(all_head_size, value_layer.shape[2], head_size)
    dense = dense.permute(1, 2, 0).contiguous()

    # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
    transformed_layer = value_layer.matmul(dense)
    transformed_shape = transformed_layer.size() #(batch, seq_length, num_heads, 1, all_head_size)
    transformed_layer = transformed_layer.view(transformed_shape[:-2] + (transformed_shape[-1],))
    transformed_layer = transformed_layer.permute(0, 2, 1, 3).contiguous() 
    transformed_shape = transformed_layer.size() #(batch, num_heads, seq_length, all_head_size)
    transformed_norm = torch.norm(transformed_layer, dim=-1)
   
    # Make weighted vectors αf(x) from transformed vectors (transformed_layer) and attention weights (attention_probs).
    weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_weights.unsqueeze(0), transformed_layer) #(batch, num_heads, seq_length, seq_length, all_head_size)
   
    # batch_size = 1 # moved it here, it was after this line
    # weighted_layer = torch.empty(batch_size, num_heads, seq_len, seq_len, all_head_size)

    # attention_weights = attention_weights.unsqueeze(0) # added by NS

    # for b in range(batch_size):
    #     for h in range(num_heads):
    #         weighted_layer[b, h] = torch.einsum('ks,sd->ksd', attention_weights[b, h], transformed_layer[b, h])

    weighted_norm = torch.norm(weighted_layer, dim=-1) #(batch, num_heads, seq_length, seq_length)

    # Sum each αf(x) over all heads: (batch, seq_length, seq_length, all_head_size)
    summed_weighted_layer = weighted_layer.sum(dim=1)

    # Calculate L2 norm of summed weighted vectors: (batch, seq_length, seq_length)
    summed_weighted_norm = torch.norm(summed_weighted_layer, dim=-1)

    if orig_device is not None:
        transformed_norm = transformed_norm.to(orig_device)
        weighted_norm = weighted_norm.to(orig_device)
        summed_weighted_norm = summed_weighted_norm.to(orig_device)

    del transformed_shape

    # outputs: ||f(x)||, ||αf(x)||, ||Σαf(x)||
    outputs = (
        transformed_norm.detach().cpu(),
        weighted_norm.detach().cpu(),
        summed_weighted_norm.detach().cpu(),
    )

    del transformed_layer, weighted_layer, summed_weighted_layer
    torch.cuda.empty_cache()

    return outputs



@torch.no_grad()
def gptj_split_heads(tensor, num_attention_heads, attn_head_size, rotary):
    """
    Splits hidden dim into attn_head_size and num_attention_heads
    """
    new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    if rotary:
        return tensor
    if len(tensor.shape) == 5:
        return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
    elif len(tensor.shape) == 4:
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    else:
        raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")


@torch.no_grad()
def compute_attention_norms_gptj(attention_weights, value_layer, dense, device=None):
    """
    Args:
    # attention_weights: (num_heads, s, s) num_heads = all_head_size/head_size
    # value_layer: (s, h)
    # dense: nn.Linear (h, h)
    """
    orig_device = None
    if device is not None:
        orig_device = attention_weights.device
        attention_weights = attention_weights.to(device)
        value_layer = value_layer.to(device)
        dense = dense.to(device)

    seq_len, all_head_size = value_layer.shape 
    num_heads = attention_weights.shape[0]
    head_size = int(all_head_size / num_heads)

    # (batch, head, seq_length, head_features)
    value_layer = gptj_split_heads(value_layer.unsqueeze(0), num_heads, head_size, rotary=False)

    # value_layer is converted to (seq_length, num_heads, 1, head_size)
    value_layer = value_layer.permute(0, 2, 1, 3).unsqueeze(3).contiguous()
    # torch.Size([6, 16, 1, 256])
    assert value_layer.shape == (1, seq_len, num_heads, 1, head_size), value_layer.shape

    # dense weight is converted to (num_heads, head_size, all_head_size)
    dense_layer = dense
    dense = dense.weight
    dense = dense.view(all_head_size, num_heads, head_size) 
    dense = dense.permute(1, 2, 0).contiguous()

    # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
    if dense.dtype != torch.int8:
        transformed_layer = value_layer.matmul(dense)
    else:
        raise NotImplementedError("Quantized models are not supported yet.")
        # bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        transformed_layer = bitsandbytes.matmul(value_layer.to(torch.float16), dense, bias=None, state=dense_layer.state)

    # value_layer.shape = (seq_length, num_heads, 1, head_size)
    # dense.shape = (num_heads, head_size, all_head_size)
    transformed_layer = transformed_layer.squeeze(-2).permute(0, 2, 1, 3)  #(batch, num_heads, seq_length, all_head_size)
    transformed_norm = torch.norm(transformed_layer, dim=-1)

    # Make weighted vectors αf(x) from transformed vectors (transformed_layer) and attention weights (attention_probs).
    weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_weights.unsqueeze(0), transformed_layer) # (batch, num_heads, seq_length, seq_length, all_head_size)
    # batch_size = 1 # moved it here, it was after this line
    # weighted_layer = torch.empty(batch_size, num_heads, seq_len, seq_len, all_head_size)

    # attention_weights = attention_weights.unsqueeze(0) # added by NS

    # for b in range(batch_size):
    #     for h in range(num_heads):
    #         weighted_layer[b, h] = torch.einsum('ks,sd->ksd', attention_weights[b, h], transformed_layer[b, h])

    # Now weighted_layer is the same shape as before, but the computation has been broken down

    weighted_norm = torch.norm(weighted_layer, dim=-1) #(batch, num_heads, seq_length, seq_length)

    # Sum each αf(x) over all heads: (batch, seq_length, seq_length, all_head_size)
    summed_weighted_layer = weighted_layer.sum(dim=1)  

    # Calculate L2 norm of summed weighted vectors: (batch, seq_length, seq_length)
    summed_weighted_norm = torch.norm(summed_weighted_layer, dim=-1)

    if orig_device is not None:
        transformed_norm = transformed_norm.to(orig_device)
        weighted_norm = weighted_norm.to(orig_device)
        summed_weighted_norm = summed_weighted_norm.to(orig_device)

    # outputs: ||f(x)||, ||αf(x)||, ||Σαf(x)|| 
    outputs = (
        transformed_norm.detach().cpu(),
        weighted_norm.detach().cpu(),
        summed_weighted_norm.detach().cpu(),
    )

    del transformed_layer, weighted_layer, summed_weighted_layer
    torch.cuda.empty_cache()

    return outputs

@torch.no_grad()
def get_attention_norms(model, layers, out, device=None, num_layers=None):
    '''
    get weighted attention norms || sum( alpha * f(x))  || where f(x) = (x Wv) W
    args:
        model: GPT2LMHeadModel or GPTJForCausalLM
        layers: output of pytorch_surgeon
        out: output of model
        num_layers: number of layers to compute attention norms for, by default compute for all layers
        device: torch.device, override default device
    '''
    attention_norms = []

    value_layer_name = "model.decoder.layers.{i}.self_attn.v_proj"
    if isinstance(model, GPTJForCausalLM):
        value_layer_name = "transformer.h.{i}.attn.v_proj"

    compute_attention_norms = compute_attention_norms_opt
    if isinstance(model, GPTJForCausalLM):
        compute_attention_norms = compute_attention_norms_gptj

    if num_layers is None:
        if isinstance(model, GPTJForCausalLM):
            num_layers = len(model.transformer.h)
        else:
            num_layers = len(model.model.decoder.layers) # model is model.model changed by NS

    for i in range(num_layers):
        _layer_name = value_layer_name.format(i=i)
        value_layer = layers[_layer_name].squeeze(0)  # [src_len, hidden_size] , this is V = xWv

        if isinstance(model, GPTJForCausalLM):
            layer_dense_matrix = model.transformer.h[i].attn.out_proj # [hidden_size, hidden_size], Wo weight matrix
        else:
            layer_dense_matrix = model.model.decoder.layers[i].self_attn.out_proj # [hidden_size, hidden_size], Wo weight matrix

        layer_attention_weights = out.attentions[i].squeeze(0)
        # norm_fx (batch, seq_length, seq_length, all_head_size)
        # norm_sum_afx (batch, seq_length, seq_length))
        norm_fx, norm_afx, norm_sum_afx = compute_attention_norms(layer_attention_weights, value_layer, layer_dense_matrix, device=device)
        attention_norms.append(norm_sum_afx) # average across heads

    attention_norms = torch.stack(attention_norms, dim=0) # [layers, b, s, s]
    attention_norms = attention_norms.squeeze() # [layers, s, s] as b is 1
    attention_norms_to_plot = attention_norms / attention_norms.sum(axis=-1, keepdims=True) # for decoder only models
    attention_norms_to_plot = attention_norms_to_plot ** 0.5 # increase contrast

    return attention_norms_to_plot


def get_attention_matrix_gptj(prompt, model, tokenizer):
    
    model.eval();

    # save_path = 'gptj_vis.pdf'
    # vertical_plot = False
    # generate_tokens = 0  # only use it without minimize_GPU_memory

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    minimize_GPU_memory = True  # only put the tensors that you need to the GPU
    # model = model.to(device) # added by NS

    text_to_visualize = prompt["Instance"]["input"]
    textid = prompt["id"]
    corruption_name = prompt["Corruption_name"]

    if minimize_GPU_memory:
        model = model.to(device)
    inputs = tokenizer(text_to_visualize, return_tensors="pt").to(device)

    all_model_layers = surgeon.get_layers(model)
    all_model_layers = {k: k for k in all_model_layers}
    disected_model = surgeon.Inspect(model, all_model_layers)

    num_layers = model.config.num_hidden_layers
    out, layers = disected_model(**inputs, output_attentions=True, output_hidden_states=True)

    if minimize_GPU_memory:
        model = model.to("cpu")
        del disected_model
        torch.cuda.empty_cache()
    
    attention_norms = get_attention_norms(model, layers, out, device=device)

    return attention_norms