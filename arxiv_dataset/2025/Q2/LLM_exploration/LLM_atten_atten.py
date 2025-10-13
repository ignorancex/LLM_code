from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch

import copy
import argparse
import numpy as np
import json
import scipy
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoraModel, PeftConfig, PeftModel
import os
import pickle
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import json
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from proj import FP
import random


from typing import List, Optional, Tuple, Union
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import LlamaModel,Cache,StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast





class CustomLlamaModel(LlamaModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        edge_mask=None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 调用父类的forward方法
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # # 打印 position_ids
        # print("Position IDs:", position_ids)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions,edge_mask=edge_mask
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        edge_mask=None
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            edge_mask=edge_mask
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        edge_mask=None,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            
            if edge_mask is not None and edge_mask.shape==causal_mask.shape:
                # print(causal_mask.shape)
                index1,index2,index3,index4=torch.where(edge_mask==1)
                
                causal_mask[index1,index2,index3,index4]=0
                
                index1,index2,index3,index4=torch.where(edge_mask==-6000)
                
                causal_mask[index1,index2,index3,index4]=-6000
                
                
                
                # causal_mask[index1,index2,index3,index4]=0
                
                # print(causal_mask.shape)
        return causal_mask
    
class GLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        edge_mask=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            edge_mask=edge_mask,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        return
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        edge_mask=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min
            

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=attention_mask.shape[1],
                    target_length=attention_mask.shape[1],
                    dtype=self.lm_head.weight.dtype,
                    device=attention_mask.device,
                    min_dtype=torch.finfo(self.lm_head.weight.dtype).min,
                    cache_position = torch.arange(0, 0 + attention_mask.shape[1], device=attention_mask.device),
                    batch_size=attention_mask.shape[0],
                    edge_mask=edge_mask,
                    **kwargs,
                )
            # print(attention_mask.shape)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                'edge_mask':edge_mask,
            }
        )
        
        return model_inputs


def get_total_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def get_first_and_second_order_neighbors(data, input_ids):
    # 将邻接矩阵转换为 SparseTensor，这里直接使用 data.adj_t，因为它已经是 SparseTensor 类型
    adj_matrix = data.adj_t
    
    # 创建一个从原始节点 ID 到新节点 ID 的映射
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    # 创建一个空字典来存储结果
    neighbors_dict = {}
    
    # 对于每一个输入节点
    for node_id in input_ids:
        # 将原始节点 ID 映射到新的节点 ID
        new_node_id = id_mapping[node_id.item()]
        
        # 获取该节点的所有邻居
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        
        # 获取一阶邻居的原始 ID
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        # 为每个一阶邻居获取二阶邻居
        for first_order_neighbor in first_order_neighbor_ids:
            # 获取该一阶邻居的所有邻居
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            
            # 过滤掉自己作为一阶邻居的情况
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            
            # 获取二阶邻居的原始 ID
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            
            # 添加二阶邻居到对应的一阶邻居下
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 将邻居列表添加到字典中
        neighbors_dict[node_id.item()] = first_order_neighbors
    return neighbors_dict

def random_get_first_and_second_order_neighbors_I(data, input_ids):
    # 将邻接矩阵转换为 SparseTensor，这里直接使用 data.adj_t，因为它已经是 SparseTensor 类型
    adj_matrix = data.adj_t
    
    # 创建一个从原始节点 ID 到新节点 ID 的映射
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    # 创建一个空字典来存储结果
    neighbors_dict = {}
    
    # 对于每一个输入节点
    for node_id in input_ids:
        # 将原始节点 ID 映射到新的节点 ID
        new_node_id = id_mapping[node_id.item()]
        
        # 获取该节点的所有邻居
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        
        # 获取一阶邻居的原始 ID
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        # 为每个一阶邻居获取二阶邻居
        for first_order_neighbor in first_order_neighbor_ids:
            # 获取该一阶邻居的所有邻居
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            
            # 过滤掉自己作为一阶邻居的情况
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            
            # 获取二阶邻居的原始 ID
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            
            # 添加二阶邻居到对应的一阶邻居下
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 在完成邻居获取后，尝试添加扰动
        if len(first_order_neighbors) > 1:
            # 随机选择两个一阶邻居
            selected_neighbors = random.sample(list(first_order_neighbors.keys()), 2)
            
            # 交换它们的二阶邻居列表
            first_order_neighbors[selected_neighbors[0]], first_order_neighbors[selected_neighbors[1]] = \
                first_order_neighbors[selected_neighbors[1]].copy(), first_order_neighbors[selected_neighbors[0]].copy()
        
        # 将邻居列表添加到字典中
        neighbors_dict[node_id.item()] = first_order_neighbors
    
    return neighbors_dict

def random_get_first_and_second_order_neighbors_II(data, input_ids):
    adj_matrix = data.adj_t
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    neighbors_dict = {}
    
    for node_id in input_ids:
        new_node_id = id_mapping[node_id.item()]
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        for first_order_neighbor in first_order_neighbor_ids:
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        iter_num=0
        while iter_num<=10:
            if len(first_order_neighbors) > 1:
                # 随机选择两个一阶邻居
                selected_neighbors = random.sample(list(first_order_neighbors.keys()), 2)
                # 为这两个邻居随机选择一些二阶邻居进行交换
                swap_size = min(len(first_order_neighbors[selected_neighbors[0]]), len(first_order_neighbors[selected_neighbors[1]]))
                if swap_size > 0:
                    swap_indices = random.sample(range(swap_size), swap_size)
                    for idx in swap_indices:
                        first_order_neighbors[selected_neighbors[0]][idx], first_order_neighbors[selected_neighbors[1]][idx] = \
                            first_order_neighbors[selected_neighbors[1]][idx], first_order_neighbors[selected_neighbors[0]][idx]
            iter_num+=1
        
        neighbors_dict[node_id.item()] = first_order_neighbors
    
    return neighbors_dict

def random_get_first_and_second_order_neighbors_III(data, input_ids):
    adj_matrix = data.adj_t
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    all_nodes = list(id_mapping.values())
    
    neighbors_dict = {}
    
    for node_id in input_ids:
        new_node_id = id_mapping[node_id.item()]
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        for first_order_neighbor in first_order_neighbor_ids:
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 将所有的一阶和二阶节点收集到一个列表中
        all_neighbors = list(first_order_neighbors.keys()) + [neighbor for sublist in first_order_neighbors.values() for neighbor in sublist]
        
        print(first_order_neighbors)
        
        # 打乱所有节点
        np.random.shuffle(all_neighbors)
        
        # 重新分配一阶和二阶节点
        new_first_order_neighbors = all_neighbors[:len(first_order_neighbors)]
        new_second_order_neighbors = all_neighbors[len(first_order_neighbors):]
        
        # 构建新的邻居字典
        new_neighbors_dict = {node: [] for node in new_first_order_neighbors}
        
        for i, node in enumerate(new_first_order_neighbors):
            new_neighbors_dict[node] = new_second_order_neighbors[i * len(first_order_neighbors):(i + 1) * len(first_order_neighbors)]
        
        neighbors_dict[node_id.item()] = new_neighbors_dict
    
    return neighbors_dict


def get_all_neighbors(data, input_ids):
    # 获取所有输入节点的一阶和二阶邻居
    all_neighbors = {}
    adj_matrix = data.adj_t
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    for node_id in input_ids:
        new_node_id = id_mapping[node_id.item()]
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        for first_order_neighbor in first_order_neighbor_ids:
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        all_neighbors[node_id.item()] = first_order_neighbors
    
    return all_neighbors

def random_get_first_and_second_order_neighbors_IV(data, input_ids):
    all_neighbors = get_all_neighbors(data, input_ids)  # 获取整个 batch 的邻居信息
    neighbors_dict = {}

    for center_node_id in input_ids:
        temp_neighbors = copy.deepcopy(all_neighbors[center_node_id.item()])
        other_center_nodes = set(input_ids) - {center_node_id}
        
        # 收集其他中心节点的所有一阶和二阶邻居作为不相关节点池
        unrelated_nodes_pool = []
        for other_node_id in other_center_nodes:
            unrelated_nodes_pool.extend(list(all_neighbors[other_node_id.item()].keys()))
            for second_order_neighbors in all_neighbors[other_node_id.item()].values():
                unrelated_nodes_pool.extend(second_order_neighbors)
        
        # 确保不相关节点池中的节点不在当前中心节点的邻居列表中
        unrelated_nodes_pool = list(set(unrelated_nodes_pool) - set(temp_neighbors.keys()) - set([node for sublist in temp_neighbors.values() for node in sublist]))
        
        iter_num=0
        while iter_num<=10:
            if unrelated_nodes_pool:  # 如果存在不相关节点，则进行替换
                # 随机选择一定数量的不相关节点
                num_unrelated_to_replace = min(len(unrelated_nodes_pool), len(temp_neighbors)) // 2  # 可根据需要调整比例

                unrelated_nodes_sample = random.sample(unrelated_nodes_pool, num_unrelated_to_replace)

                # 对于每一个选中的不相关节点，随机决定是替换一阶还是二阶邻居
                for unrelated_node in unrelated_nodes_sample:
                    if random.choice([True, False]):  # 替换一阶邻居
                        replaced_node = random.choice(list(temp_neighbors.keys()))
                        temp_neighbors[unrelated_node] = temp_neighbors.pop(replaced_node)
                    else:  # 替换二阶邻居
                        if temp_neighbors:
                            first_order_node, second_order_neighbors = random.choice(list(temp_neighbors.items()))
                            if second_order_neighbors:
                                replaced_node = random.choice(second_order_neighbors)
                                second_order_neighbors.remove(replaced_node)
                                second_order_neighbors.append(unrelated_node)
            iter_num+=1
        neighbors_dict[center_node_id.item()] = temp_neighbors
    
    return neighbors_dict



def get_args():
    parser = argparse.ArgumentParser(description="PyTorch PYG implementation")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # CPU/GPU
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    
    """LLM Config"""
    parser.add_argument('--backbone', type=str, default='./llama2-7b-hf')
    parser.add_argument('--tokenizer', type=str, default='AutoTokenizer')
    parser.add_argument('--max_text_length', type=int, default=4096)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=int, default=0.05)

    
    """LLM Training"""
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="8,8",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--perturbation",
        type=int,
        default=0,
        help="perturbation type",
    )
    parser.add_argument(
        "--bidi",
        type=bool,
        default=True,
        help="bidi?unbi",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k ,attention Window for graph visiable horizon",
    )
    
    
    """Dataset"""
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument("--dataset", type=str, default="pubmed", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--num_nodes", type=int, default="19717", help="the number of nodes")

    """Global """
    parser.add_argument("--train", type=bool, default="True", help="training ")
    parser.add_argument("--test", type=bool, default="False", help="testing ")
    args = parser.parse_args(args=[])

    
    return args

def pre_data(args):
    if args.dataset=='ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
        data=dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        train_loader = NeighborLoader(data, input_nodes=split_idx["train"],
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                      ,batch_size=args.batch_size, 
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=split_idx["valid"],
                                      batch_size=args.batch_size,
                                         num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                      , shuffle=False,num_workers=args.num_workers)
        test_loader = NeighborLoader(copy.copy(data), input_nodes=split_idx["test"],
                                     batch_size=args.batch_size,
                                    num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                     , shuffle=False,num_workers=args.num_workers)
        
    if args.dataset=='deezer-europe':
        deezer = scipy.io.loadmat(f'./deezer_europe/deezer-europe.mat')
        adj_t= SparseTensor(row=torch.tensor(deezer['A'].tocoo().row).to(torch.long), col=torch.tensor(deezer['A'].tocoo().col).to(torch.long),sparse_sizes=(len(deezer['label'][0]), len(deezer['label'][0])))
        data= Data(x=torch.tensor(deezer['features'].toarray()).to(torch.float32), adj_t=adj_t,y=torch.tensor(deezer['label']).squeeze())
        data.adj_t = data.adj_t.to_symmetric()
        # 获取节点总数
        num_nodes = len(data.y)
        # 定义数据集划分比例
        train_ratio = 0.5
        val_ratio = 0.25
        test_ratio = 0.25
        # 计算每种数据集包含的节点数
        num_train = int(num_nodes * train_ratio)
        num_val = int(num_nodes * val_ratio)
        num_test = num_nodes - num_train - num_val
        # 随机排列节点索引
        node_indices = torch.randperm(num_nodes)
        # 切分索引
        train_indices = node_indices[:num_train]
        val_indices = node_indices[num_train:num_train + num_val]
        test_indices = node_indices[num_train + num_val:]

        train_loader = NeighborLoader(data, input_nodes=train_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)    
    if args.dataset in ['roman_empire','amazon_ratings','questions']:
        
        file_path = f'./{args.dataset}/{args.dataset}_right.npz'
        if args.dataset=='amazon_ratings':
            file_path = f'./{args.dataset}/{args.dataset}_right_10.npz'
            
        data = np.load(file_path)
        
        # 切分索引
        train_indices = np.where(data['train_masks'][0])[0]
        val_indices = np.where(data['val_masks'][0])[0]
        test_indices = np.where(data['test_masks'][0])[0]
        
        
        # data = np.load('./roman_empire/roman_empire.npz')
        adj_t= SparseTensor(row=torch.tensor(data['edges']).t()[0].to(torch.long), col=torch.tensor(data['edges']).t()[1].to(torch.long),sparse_sizes=(len(data['node_labels']),len(data['node_labels']) ))
        data= Data(x=torch.tensor(data['node_features']), adj_t=adj_t,y=torch.tensor(data['node_labels']))
        data.adj_t = data.adj_t.to_symmetric()


        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    if args.dataset in ['pubmed']:
        file_path = f'./{args.dataset}/data.pt'
        data = torch.load(file_path)
        
        data.adj_t = data.adj_t.to_symmetric()
        
        data.y=torch.tensor(data.y)
        
        train_indices = data.train_id
        val_indices = data.val_id
        test_indices = data.test_id
            
        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
    if args.dataset in ['wikics']:
        file_path = f'./{args.dataset}/data_token_right_10.pt'
        
        data = torch.load(file_path)
        
        node_id = np.arange(data.num_nodes)
        
        
        random.shuffle(node_id)
        
        train_indices = np.sort(node_id[:int(data.num_nodes * 0.6)])
        val_indices = np.sort(
            node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
        test_indices = np.sort(node_id[int(data.num_nodes * 0.8):])
        
        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)

    
    return train_loader,valid_loader,test_loader,data

class Trainer():
    def __init__(self,args):
        self.args=args
        
        if args.dataset=='wikics':
            template={}
            template['train']="<User>: In paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>: {}"
            template['test']="<User>: In paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>:"
        if args.dataset=='roman_empire':
            template={}
            template['train']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that connect {}, what is the word {} syntactic role? <Assistant>: {}"
            template['test']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that connect {}, what is the word {} syntactic role? <Assistant>:"
        if args.dataset=='amazon_ratings':
            template={}
            template['train']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that connect {}, what is the product {} rating? <Assistant>: {}"
            template['test']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that connect {}, what is the product {} rating? <Assistant>:"
        if args.dataset=='pubmed':
            template={}
            template['train']="<User>: In medical paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of medical papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>: {}"
            template['test']="<User>: In medical paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of medical papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>:"
        
        self.template=template
        
        self.tokenizer = self.get_tokenizer()
        self.train_loader, self.valid_loader, self.test_loader,self.data=pre_data(self.args)
        
        self.model= self.get_model()
        
        self.optimizer, self.lr_scheduler=self.get_optimizer()
        
    def get_tokenizer(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.args.backbone, max_length=self.args.max_text_length,do_lower_case=self.args.do_lower_case)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token
        
        new_tokens=[ 'node_'+str(i) for i in range(self.args.num_nodes)]
        
        tokenizer.add_tokens(new_tokens)
        
        return tokenizer
    def get_optimizer(self):
        
        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print('Warmup ratio:', warmup_ratio)
        print("Warm up Iters: %d" % warmup_iters)
        
        if self.args.dataset in ['pubmed','amazon_ratings']:
            for param in self.model.model.model.embed_tokens.parameters():
                param.requires_grad = True
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    'lr': self.args.lr,
                },
                # 这个组包含了bias和LayerNorm的所有参数，不应用权重衰减
                {
                    "params":[p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    'lr': self.args.lr,
                }

        ]
        optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
        
        return optim, lr_scheduler
    
    def get_model(self):
        model = GLM.from_pretrained(
                                    self.args.backbone,
                                    load_in_8bit=True,
                                    torch_dtype=torch.float16,
                                    use_safetensors=False,
                                    device_map='cuda:0'
                                )
        
        # model_embed=model.model.embed_tokens.weight.data
        
        model.resize_token_embeddings(32000+self.args.num_nodes)
        
        model.model.embed_tokens.weight.data[-self.args.num_nodes:]=self.data.x
        
        model = prepare_model_for_kbit_training(model)
        
        
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=['q_proj','k_proj']
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
            
        )
        
        model= get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        
        return model
    
    def get_prompt(self,batch,is_training=True):
        #将label又数字id形式转化为文字
        if self.args.dataset=='ogbn-arxiv':
            dict_labelid2categeory=load_pickle('dict_labelid2arxivcategeory.pkl')
        if self.args.dataset=='deezer-europe':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='female'
            dict_labelid2categeory[1]='male'
        if self.args.dataset=='roman_empire':
            dict_labelid2categeory={}
            
            dict_labelid2categeory = {1: 'prepositional object',2: 'preposition',3: 'determiner',4: 'adjectival',5: 
                                    'conjunct',6: 'nominal subject',7: 'coordinating conjunction',0: 'root',
                                    8: 'direct object',9: 'adverbial',10: 'compound',11: 'auxiliary',
                                    12: 'appositional',13: 'passive auxiliary',14: 'passive nominal subject',15:
                                    'possession',16: 'relative clause',17: 'other'}
            
        if self.args.dataset=='amazon_ratings':
            
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='Very Positive'
            dict_labelid2categeory[1]='Positive'
            dict_labelid2categeory[2]='Neutral'
            dict_labelid2categeory[3]='Negative'
            dict_labelid2categeory[4]='Very Negative'
            
        if self.args.dataset=='questions':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='activate'
            dict_labelid2categeory[1]='no'
        
        if self.args.dataset=='wikics':
            dict_labelid2categeory={
            0: 'Computational linguistics',
            1: 'Databases',
            2: 'Operating systems',
            3: 'Computer architecture',
            4: 'Computer security',
            5: 'Internet protocols',
            6: 'Computer file systems',
            7: 'Distributed computing architecture',
            8: 'Web technology',
            9: 'Programming language topics'}
        if self.args.dataset=='pubmed':
            dict_labelid2categeory={
            0: 'Diabetes Mellitus, Experimental',
            1: 'Diabetes Mellitus Type 1',
            2: 'Diabetes Mellitus Type 2'}
        if self.args.perturbation==0:
            neighbors_dict=get_first_and_second_order_neighbors(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==1:
            neighbors_dict=random_get_first_and_second_order_neighbors_I(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==2:
            neighbors_dict=random_get_first_and_second_order_neighbors_II(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==3:
            neighbors_dict=random_get_first_and_second_order_neighbors_III(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==4:
            neighbors_dict=random_get_first_and_second_order_neighbors_IV(batch,batch.n_id[:batch.batch_size])
            
            
        batch_text=[]
        labels=[]
        for i,label in zip(neighbors_dict.keys(),batch.y[:batch.batch_size]):
            label=dict_labelid2categeory[label.item()]
            connect_text='['
            text=''
            for j in neighbors_dict[i].keys():
                connect_text+='node_'+str(j) + ' is connected [' + ','.join('node_'+str(item) for item in neighbors_dict[i][j]) + ' ] ,'
            connect_text=connect_text[:-1]+']'

            if is_training :
                text = self.template['train'].format('node_'+str(i),connect_text,'node_'+str(i),label)+'</s>'
            else:
                text = self.template['test'].format('node_'+str(i),connect_text,'node_'+str(i))
            batch_text.append(text)
            labels.append(label+'</s>')
        input_ids=self.tokenizer(batch_text,padding='longest',
                                 max_length=self.args.max_text_length,return_tensors="pt")['input_ids']
        attention_mask=self.tokenizer(batch_text,padding='longest',
                                      max_length=self.args.max_text_length,return_tensors="pt")['attention_mask']
        
        #去掉开头的字符
        label_ids=self.tokenizer(labels,padding='longest',
                                 max_length=self.args.max_text_length,return_tensors="pt")['input_ids']
        
        
        if is_training:
            
            label_ids[label_ids.eq(self.tokenizer.pad_token_id)]=-100
            label_ids[:,-1]=2
            label_ids[label_ids.eq(1)]=-100
            label_ids=torch.cat((torch.full((label_ids.size(0), input_ids.size(-1)-label_ids.size(-1)), -100),
                              label_ids),dim=-1)
        else:
            # 测试阶段可能不需要生成标签
            label_ids = label_ids
        
        # edge_mask_edge=[]
        # edge_mask_node=[]
        # for i in range(len(neighbors_dict)):
        #     nodes=torch.where((input_ids[i]<32000+22662)&(input_ids[i]>=32000))[0][1:-1].tolist()
        #     edges=torch.where(input_ids[i]>=32000+22662)[0].tolist()
        #     idx=[]
        #     idxx=[]
        #     index=0
        #     i=list(neighbors_dict.keys())[i]
        #     for j in neighbors_dict[i]:
        #         idx.append(nodes[index])
        #         if len(neighbors_dict[i][j])!=0:
        #             idxx.append(nodes[index+1:index+len(neighbors_dict[i][j])+1])
        #             index+=len(neighbors_dict[i][j])+1
        #     idx=[idx]+idxx
        #     edge_mask_node.append(idx)
        #     edge_mask_edge.append(edges)
        # edge_mask=prepare_4d_edge_mask(attention_mask,edge_mask_edge,edge_mask_node)
        
#         edge_mask_edge=[]
#         edge_mask_node=[]
#         for i in range(len(neighbors_dict)):
#             nodes=torch.where((input_ids[i]<32000+self.args.num_nodes)&(input_ids[i]>=32000))[0][1:-1].tolist()
#             first_node=torch.where((input_ids[i]<32000+self.args.num_nodes)&(input_ids[i]>=32000))[0][0].tolist()
#             edges=torch.where(input_ids[i]>=32000+self.args.num_nodes)[0].tolist()
#             idx=[]
#             idxx=[]
#             index=0
#             i=list(neighbors_dict.keys())[i]
#             for j in neighbors_dict[i]:
#                 idx.append(nodes[index])
#                 if len(neighbors_dict[i][j])!=0:
#                     idxx.append(nodes[index+1:index+len(neighbors_dict[i][j])+1])
#                     index+=len(neighbors_dict[i][j])+1
#             idx=[idx]+idxx+edges
#             edges.extend([first_node]+idx[0])
#             edge_mask_node.append(idx)
#             edge_mask_edge.append(edges)
#         edge_mask=prepare_4d_edge_mask(attention_mask,edge_mask_edge,edge_mask_node)
        
        idx_src_list=[]
        idx_dist_list=[]
        for i in range(len(neighbors_dict)):
            nodes=torch.where((input_ids[i]<32000+self.args.num_nodes)&(input_ids[i]>=32000))[0][:-1].tolist()
            first_node=torch.where((input_ids[i]<32000+self.args.num_nodes)&(input_ids[i]>=32000))[0][0].tolist()
            idx_src=[]
            idx_dist=[]
            
            index=1
            i=list(neighbors_dict.keys())[i]
            
            for j in neighbors_dict[i]:
                
                idx_src.append(nodes[index])
                if len(neighbors_dict[i][j])!=0:
                    idx_dist.append(nodes[index+1:index+len(neighbors_dict[i][j])+1])
                index+=len(neighbors_dict[i][j])+1
                
            idx_dist.append(idx_src.copy())
            idx_src.append(nodes[0])
            
            idx_src_list.append(idx_src)
            idx_dist_list.append(idx_dist)
        node_mask=((input_ids<32000+self.args.num_nodes)&(input_ids>=32000))
        
        edge_mask=prepare_4d_edge_mask(attention_mask,idx_src_list,idx_dist_list,node_mask,self.args.bidi,self.args.k)
        print(edge_mask.shape)
        
        # edge_mask=prepare_4d_edge_mask(attention_mask)
        
        return input_ids, attention_mask, label_ids,edge_mask
        
    def load_checkpoint(self, ckpt_path,proj_path):
        results = self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
        results = self.proj_model.load_state_dict(torch.load(proj_path), strict=True)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            print('Model loaded from ', proj_path)
            print(results)
    def train(self):
        
        self.model.train()
        
        pbar = tqdm(total=len(self.train_loader), ncols=275)
        for epoch in range(self.args.epoch):
            
            for step_i, batch in enumerate(self.train_loader):
                
                input_ids, attention_mask, labels, edge_mask=self.get_prompt(batch,True)
                
                attention_mask=attention_mask.to(self.args.device)
                
                labels=labels.to(self.args.device)
                
                input_ids=input_ids.to(self.args.device)
                
                output= self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    edge_mask=edge_mask)
                
                loss = output['loss']/ self.args.gradient_accumulation_steps
                
                
                loss.backward()
                
                
                if step_i % self.args.gradient_accumulation_steps == 0:
                    # 在训练循环中调用
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    
                    self.optimizer.step()  # Update
                    self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None
                if step_i % 1 == 0:
                    lr = self.lr_scheduler.get_lr()[0]
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'
                    desc_str += f' Loss:{loss:.3f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
        pbar.close()
            
        torch.save(self.model.state_dict(),"llmcom_{}_end_{}_{}_{}.pth".format(self.args.epoch,self.args.dataset,self.args.bidi,self.args.k))
                                         
    def test(self):
        for epoch in range(1):
            ckpt_path = "llmcom_1_end_{}.pth".format(self.args.dataset)

            self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
            
            self.model.eval()
            acc_list=[]
            for time in range(4):
                with torch.no_grad():
                    print('len of val_loader is {}'.format(len(self.test_loader)))
                    acc=0
                    for step_i, batch in tqdm(enumerate(self.test_loader)):

                        input_ids, attention_mask, labels,edge_mask=self.get_prompt(batch,False)

                        attention_mask=attention_mask.to(self.args.device)

                        input_ids=input_ids.to(self.args.device)

                        embeds=self.model.model.model.embed_tokens(input_ids).to(self.args.device)

                        output= self.model.generate(inputs_embeds=embeds,
                                                    attention_mask=attention_mask,max_new_tokens=20,num_beams=2,edge_mask=edge_mask)
                        output=self.tokenizer.batch_decode(output,skip_special_tokens=True)

                        labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
                        # print(output)
                        print(labels)
                        for i in range(len(output)):
                            if labels[i] == output[i]:
                               acc+=1
                        print(acc)
                    acc_list.append(acc)
            print(acc_list)
    def test_perturbation(self):
        for epoch in range(1):
            ckpt_path = "llmcom_1_end_{}.pth".format(self.args.dataset)

            self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
            
            self.model.eval()
            
            acc_list_perbation=[]
            for perturbation in range(1,5):
                print('test_perturbation is {}'.format(perturbation))
                self.args.perturbation=perturbation
                acc_list=[]
                for time in range(4):
                    with torch.no_grad():
                        print('len of val_loader is {}'.format(len(self.test_loader)))
                        acc=0
                        for step_i, batch in tqdm(enumerate(self.test_loader)):

                            input_ids, attention_mask, labels,edge_mask=self.get_prompt(batch,False)

                            attention_mask=attention_mask.to(self.args.device)

                            input_ids=input_ids.to(self.args.device)

                            embeds=self.model.model.model.embed_tokens(input_ids).to(self.args.device)

                            output= self.model.generate(inputs_embeds=embeds,
                                                        attention_mask=attention_mask,max_new_tokens=20,num_beams=2,edge_mask=edge_mask)
                            output=self.tokenizer.batch_decode(output,skip_special_tokens=True)

                            labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
                            # print(output)
                            print(labels)
                            for i in range(len(output)):
                                if labels[i] == output[i]:
                                   acc+=1
                            print(acc)
                        acc_list.append(acc)
                print(acc_list)
                acc_list_perbation.append(acc_list)
            print(acc_list_perbation)
    def test_window(self):
        for epoch in range(1):
            ckpt_path = "llmcom_1_end_{}_{}_{}.pth".format(self.args.dataset,self.args.bidi,self.args.k)

            self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
            
            self.model.eval()
            
            acc_list_k_bidi=[]
            
            for k in range(1,5):
                for bidi in [False,True]:
                    self.args.k=k
                    self.args.bidi=bidi
                    acc_list=[]
                    for time in range(2):
                        with torch.no_grad():
                            print('len of val_loader is {}'.format(len(self.test_loader)))
                            print(f'Start,k is {k},bidi is {bidi},time is {time}')
                            acc=0
                            for step_i, batch in tqdm(enumerate(self.test_loader)):

                                input_ids, attention_mask, labels,edge_mask=self.get_prompt(batch,False)

                                attention_mask=attention_mask.to(self.args.device)

                                input_ids=input_ids.to(self.args.device)

                                embeds=self.model.model.model.embed_tokens(input_ids).to(self.args.device)

                                output= self.model.generate(inputs_embeds=embeds,
                                                            attention_mask=attention_mask,max_new_tokens=20,num_beams=2,edge_mask=edge_mask)
                                output=self.tokenizer.batch_decode(output,skip_special_tokens=True)

                                labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
                                for i in range(len(output)):
                                    if labels[i] == output[i]:
                                       acc+=1
                                if step_i%100==0:
                                    print(f'{step_i} acc is {acc}')
                            print(f"k is {k},bidi is {bidi},time is {time}, acc is {acc}")
                            acc_list.append(acc)
                    print(f"k is {k},bidi is {bidi},time is {time}, acc_list is {acc_list}")
                    acc_list_k_bidi.append(acc_list)
            print(f"ALL is Done acc_list_k_bidi is {acc_list_k_bidi}")
            
            
def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_4d_edge_mask(attention_mask,idx_src_list=None,idx_dist_list=None,node_mask=None,bidi=None,k=None):
    dtype=attention_mask.dtype
    min_dtype=-1
    device=attention_mask.device
    cache_position=torch.arange(0, 0 + attention_mask.shape[1], device=attention_mask.device)
    sequence_length=attention_mask.shape[1]
    target_length=attention_mask.shape[1]
    batch_size=attention_mask.shape[0]
    causal_mask = torch.full(
    (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(sequence_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )
    causal_mask[causal_mask==0]=1
    causal_mask[causal_mask<0]=0
    if idx_src_list!=None and idx_dist_list!=None:
        for i in range(len(idx_dist_list)):
            for j in range(len(idx_dist_list[i])):
                for z in range(len(idx_dist_list[i][j])):
                    causal_mask[i,0,idx_dist_list[i][j][z],node_mask[i]]=0
                    
                    causal_mask[i,0,idx_dist_list[i][j][z],idx_src_list[i][j]]=1
                    
                    # 上三角
                    if bidi:
                        causal_mask[i,0,idx_src_list[i][j],idx_dist_list[i][j]]=1
                    
                    causal_mask[i,0,idx_dist_list[i][j][z],idx_dist_list[i][j][z]]=1
            
            # 获取 node_mask[i] 为 True 的索引
            indices = torch.where(node_mask[i])[0][:-1]
            row_indices, col_indices = torch.meshgrid(indices, indices)
            ed=causal_mask[i,0,row_indices,col_indices]
            
            if k==1:
                ed = ed+ed.t()
            if k==2:
                ed = torch.matmul(ed+ed.t(),ed+ed.t())
            if k==3:
                ed = torch.matmul(torch.matmul(ed+ed.t(),ed+ed.t()), ed+ed.t())
            if k==4:
                ed = torch.matmul(torch.matmul(ed+ed.t(),ed+ed.t()), torch.matmul(ed+ed.t(),ed+ed.t()))
            
            ed[ed>0]=1
            
            if not bidi:
                ed=torch.tril(ed)
            
            ed[ed == 0] = -6000
            causal_mask[i,0,row_indices,col_indices]=ed
    return causal_mask
        
        
        
def main():
    args=get_args()

    seed_value = 42
    
    set_random_seed(seed_value)
    trainer=Trainer(args)
    if args.train==True:   
        trainer.train()

        
    # set_random_seed(seed_value)
    # trainer=Trainer(args)
    # if args.test==True:   
    #     trainer.test()
    
    # set_random_seed(seed_value)
    # trainer=Trainer(args)
    # if args.test==True:   
    #     trainer.test_perturbation()
    
    # set_random_seed(seed_value)
    # trainer=Trainer(args)
    # if args.test==True:   
    #     trainer.test_window()

if __name__=='__main__':
    main()
