import torch
import torch.nn as nn
import numpy as np
from .llm.modeling_llama import _make_causal_mask, _expand_mask
from typing import List


class BeginBlock(nn.Module):
    def __init__(self, modal_tokenizer, eva_backbone, bridge, llm_layers, tokenMode):
        super().__init__()
        self.modal_tokenizer = modal_tokenizer
        self.eva_backbone = eva_backbone
        self.bridge = bridge
        self.llm_layers = llm_layers
        self.tokenMode = tokenMode
        
    def forward(self, src_input, text_embed, input_atts_pad):
        
        src_embed = self.eva_backbone(self.modal_tokenizer(src_input))
        if isinstance(src_embed, List):
            src_embed = src_embed[-1]
            
        if self.tokenMode == "no_cls":
            src_embed = src_embed[:, 1:]
        src2t_embed, src2t_atts = self.bridge(src_embed)
        
        inputs_embeds = torch.cat([src2t_embed, text_embed], dim=1)
        attention_mask = torch.cat([src2t_atts, input_atts_pad], dim=1)
        
        batch_size, seq_length, _ = inputs_embeds.shape
        
        past_key_values_length = 0
        
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        all_hidden_states = inputs_embeds.unsqueeze(0)
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                all_hidden_states[-1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            all_hidden_states = torch.cat([all_hidden_states, layer_outputs[0].unsqueeze(0)], dim=0)
        
        return all_hidden_states, attention_mask, position_ids
        
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    

class CoreBlock(nn.Module):
    def __init__(self, llm_layers):
        super().__init__()
        self.llm_layers = llm_layers
        
    def forward(self, all_hidden_states, attention_mask, position_ids):
        
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                all_hidden_states[-1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            all_hidden_states = torch.cat([all_hidden_states, layer_outputs[0].unsqueeze(0)], dim=0)
        
        return all_hidden_states, attention_mask, position_ids
    
    
class EndBlock(nn.Module):
    def __init__(self, llm_layers, norm, task_head, out_idxs):
        super().__init__()
        self.llm_layers = llm_layers
        self.norm = norm
        self.task_head = task_head
        self.out_idxs = out_idxs
        
    def forward(self, all_hidden_states, attention_mask, position_ids):
        
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                all_hidden_states[-1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            all_hidden_states = torch.cat([all_hidden_states, layer_outputs[0].unsqueeze(0)], dim=0)
        
        out_hidden = []
        for i in range(len(all_hidden_states)):
            if i not in self.out_idxs:
                continue
            out_hidden.append(self.norm(all_hidden_states[i]))
        
        pred = self.task_head(out_hidden)
        return pred
    

def build_ModelwLLM_Pipe(modal_tokenizer, eva_backbone, bridge, llm_backbone, 
                         task_head, out_idxs, parallel_list, EVA_tokenMode='no_cls'):
    beginBlock = BeginBlock(modal_tokenizer, eva_backbone, bridge, 
                            llm_backbone.model.layers[0:parallel_list[0]], EVA_tokenMode).to("cuda:0")
    coreBlocks = []
    core_num = len(parallel_list) - 2
    for i in range(core_num):
        coreBlocks.append(CoreBlock(
            llm_backbone.model.layers[parallel_list[i]:parallel_list[i+1]]
            ).to(f"cuda:{i+1}"))
    endBlock = EndBlock(llm_backbone.model.layers[parallel_list[-2]:], llm_backbone.model.norm,
                        task_head, out_idxs).to(f"cuda:{core_num+1}")
    return nn.Sequential(
        beginBlock,
        *coreBlocks,
        endBlock
    )
    
    
# class ModelwLLM(nn.Module):
#     def __init__(self, llm_tokenizer, llm_backbone, modal_tokenizer, 
#                  eva_backbone, bridge, task_head, out_idxs=None):
#         super().__init__()
#         self.llm_tokenizer = llm_tokenizer
#         self.llm_backbone = llm_backbone
#         self.modal_tokenizer = modal_tokenizer
#         self.eva_backbone = eva_backbone
#         self.bridge = bridge
#         self.task_head = task_head
#         self.out_idxs = out_idxs
    
#     def forward(self, src_input, text_embed, input_atts_pad, target=None, mode='train'):
        
#         hidden_states = self.forward_feature(src_input.to("cuda:0"), text_embed, input_atts_pad)
#         if mode == 'train':
#             loss = self.task_head(hidden_states, target.to(hidden_states[-1].device))
#             return loss
#         elif mode == 'test':
#             pred = self.task_head.pred(hidden_states)
#             return pred
    
#     def forward_feature(self, src_input, text_embed, input_atts_pad):

#         # TODO beginBlock immBlock endBlock
#         # TODO nn.Seq pipe
#         src_embed = self.eva_backbone(self.modal_tokenizer(src_input))[-1]
#         src_embed = src_embed[:, 1:]
#         src2t_embed, src2t_atts = self.bridge(src_embed)
        
#         inputs_embeds = torch.cat([src2t_embed, text_embed], dim=1)
#         attention_mask = torch.cat([src2t_atts, input_atts_pad], dim=1)
        
#         hidden_states = self.llm_backbone.forward_hiddenStates(inputs_embeds=inputs_embeds, 
#                                                                attention_mask=attention_mask,
#                                                                out_idxs=self.out_idxs)
#         return hidden_states
    
#     def tocuda_modelParallel(self, parallel_list):
#         self.parallel_list = parallel_list
        
#         self.modal_tokenizer.to('cuda:0')
#         self.eva_backbone.to('cuda:0')
#         self.bridge.to('cuda:0')
#         self.llm_backbone.model.embed_tokens.to('cuda:0')
        
#         cur_gpuid = 0
#         for i, layer in enumerate(self.llm_backbone.model.layers):
#             if i < self.parallel_list[cur_gpuid]:
#                 layer.to(f'cuda:{cur_gpuid}')
#             else:
#                 cur_gpuid += 1
#         self.llm_backbone.model.norm.to(f'cuda:{cur_gpuid}')
        
#         self.task_head.to(f'cuda:{cur_gpuid}')
        
#     ## 调用
#     # model = ModelwLLM(llm_tokenizer=llm_tokenizer, llm_backbone=llm_backbone, modal_tokenizer=modal_tokenizer, 
#     #                   eva_backbone=eva_backbone, bridge=bridge, task_head=task_head, out_idxs=args.out_idxs)

#     # model.tocuda_modelParallel(parallel_list)