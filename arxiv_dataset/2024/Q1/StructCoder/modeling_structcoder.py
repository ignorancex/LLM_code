from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5Block, T5LayerFF, T5Stack
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import ModelOutput
from typing import Optional
import torch.nn.functional as F


class StructCoderEncoderAttention(T5Attention):
    def __init__(self, config, dtype):
        super().__init__(config, has_relative_attention_bias=True)
        self.is_decoder = False
        self.dist_attn_a = nn.Parameter(torch.empty(1,self.n_heads,1,1), requires_grad=True)
        self.dist_attn_b = nn.Parameter(torch.empty(1,self.n_heads,1,1), requires_grad=True)
        nn.init.constant_(self.dist_attn_a, 0.1)
        nn.init.constant_(self.dist_attn_b, 0)
        self.dtype = dtype
        
    def forward(self, hidden_states, mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
               position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks
        
        bsz = hidden_states.size()[0]
        position_bias_is_none = position_bias is None
        
        # get queries
        query_states = self.q(hidden_states)  # b,L',h*dk
        query_states = query_states.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2) # b,h,L',dk

        # get keys
        key_states = self.k(hidden_states) # b,L',h*dk
        key_states = key_states.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2) # b,h,L',dk
        
        # get values
        value_states = self.v(hidden_states) # b,L',h*dk
        value_states = value_states.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2) # b,h,L',dk
        
        # content attention
        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # b,h,L',L'
        
        if position_bias_is_none:
            L_dfg = 0 if dfg_code_links is None else dfg_code_links.size()[1]
            L_ast = 0 if ast_code_links is None else ast_code_links.size()[1]
            L = hidden_states.size()[1]-L_dfg-L_ast
            device = hidden_states.device
            
            # add position bias for code-code and leaf-leaf
            position_bias = self.compute_bias(L, L, device=device) # 1,h,L,L
            if L_dfg>0:
                position_bias = torch.cat((position_bias, torch.zeros((1,self.n_heads,L,L_dfg), device=device)),dim=-1) #1,h,L,L+L_dfg
                position_bias = torch.cat((position_bias, torch.zeros((1,self.n_heads,L_dfg,L+L_dfg), device=device)),
                                          dim=-2) #1,h,L+L_dfg,L+L_dfg
            if L_ast>0:
                position_bias = torch.cat((position_bias, torch.zeros((1,self.n_heads,L+L_dfg,L_ast),device=device)), 
                                          dim=-1) # 1,h,L+L_dfg,L+L_dfg+L_ast
                leaf_leaf_sim_attn = self.dist_attn_a*ast_ast_sims[:,None,:,:] + self.dist_attn_b # b,h,L_ast,L_ast
                position_bias_bottom_block = torch.cat((torch.zeros((bsz,self.n_heads,L_ast,L+L_dfg), device=device),
                                                       leaf_leaf_sim_attn), dim=-1) # b,h,L_ast,L+L_dfg+L_ast
                position_bias = torch.cat((position_bias.repeat((bsz,1,1,1)), 
                                           position_bias_bottom_block), dim=-2) # b,h,L+L_dfg+L_ast,L+L_dfg+L_ast
                
            # mask: extra padded positions, dfg-ast, dfg-code, ast-code
            mask = 1-mask # 1 for masked
            mask_padded_positions = mask[:,None,:,None] + mask[:,None,None,:] # b,1,L',L'
            mask = torch.zeros((bsz,1,L,L), device=device)
            if L_dfg>0:
                dfg_code_links, dfg_dfg_links = 1-dfg_code_links, 1-dfg_dfg_links
                mask = torch.cat((mask, dfg_code_links.transpose(1,2)[:,None,:,:]), dim=-1) # b,1,L,L+L_dfg
                mask_2_12 = torch.cat((dfg_code_links[:,None,:,:], dfg_dfg_links[:,None,:,:]), dim=-1) # b,1,L_dfg,L+L_dfg
                mask = torch.cat((mask, mask_2_12),dim=-2) # b,1,L+L_dfg,L+L_dfg
            if L_ast>0:
                ast_code_links = 1-ast_code_links
                mask_12_3 = ast_code_links.transpose(1,2)[:,None,:,:] # b,1,L,L_ast
                if L_dfg>0:
                    mask_12_3 = torch.cat((mask_12_3, 
                                           torch.ones((bsz,1,L_dfg,L_ast),device=device)), dim=-2) # b,1,L+L_dfg,L_ast
                mask = torch.cat((mask,mask_12_3),dim=-1) # b,1,L+L_dfg,L+L_dfg+L_ast
                mask_3 = torch.cat((mask_12_3.transpose(2,3), 
                                    torch.zeros((bsz,1,L_ast,L_ast),device=device)), dim=-1) # b,1,L_ast,L+L_dfg+L_ast
                mask = torch.cat((mask, mask_3), dim=-2) # b,1,L',L'
            mask = mask+mask_padded_positions # b,1,L',L'
            # convert 0/1 to -inf/0
            mask = mask.to(dtype=self.dtype)  # fp16 compatibility
            mask = torch.clamp(mask, max=1) * torch.finfo(self.dtype).min
            
            position_bias = position_bias + mask # b,h,L',L'
        scores += position_bias # b,h,L',L'
        
        # softmax and dropout
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores) # b,h,L',L'
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # b,h,L',L' 

        # multiply attentions with values and add
        attn_output = torch.matmul(attn_weights, value_states) # b,h,L',dk
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim) # b,L',h*dk
        attn_output = self.o(attn_output) # b,L',d
        
        # outputs
        outputs = (attn_output, 
                   position_bias if position_bias_is_none else None,
                   attn_weights if output_attentions else None)
        return outputs
        

class StructCoderEncoderLayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, dtype):
        super().__init__(config, True)
        self.SelfAttention = StructCoderEncoderAttention(config, dtype)
        
    def forward(self, hidden_states, mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
               position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks
        
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask, 
                                              dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                                              position_bias, output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return (hidden_states,) + attention_output[1:] # (bL'd) (bhL'L') (bhL'L') 
       
        
class StructCoderEncoderBlock(T5Block):
    def __init__(self, config, dtype):
        super().__init__(config, True)
        self.layer = nn.ModuleList()
        self.layer.append(StructCoderEncoderLayerSelfAttention(config, dtype))
        self.layer.append(T5LayerFF(config))
        
    def forward(self, hidden_states, mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
               position_bias=None, output_attentions=False):
        # hidden_states -> b,L+L_dfg+L_ast,d
        # mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        # position_bias -> bsz, h, L', L' -> code-code and leaf-leaf positional attention and all masks
        
        self_attention_outputs = self.layer[0](hidden_states, mask, dfg_code_links, dfg_dfg_links, 
                                               ast_code_links, ast_ast_sims, 
                                               position_bias=position_bias, output_attentions=output_attentions)
        hidden_states = self_attention_outputs[0]
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            
        return (hidden_states,)+self_attention_outputs[1:] # (bL'd) (bhL'L') (bhL'L') 
    
    
class StructCoderEncoderStack(T5Stack):
    def __init__(self, config):
        super().__init__(config)
        self.block = nn.ModuleList([StructCoderEncoderBlock(config, self.dtype) for i in range(config.num_layers)])
    
    def forward(self, inputs_embeds, attention_mask, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims,
                output_attentions=False, output_hidden_states=False):
        # inputs_embeds -> b,L+L_dfg+L_ast,d
        # attention_mask -> b,L+L_dfg+L_ast -> 0 or 1
        # dfg_code_links -> bsz, L_dfg, L -> 0 or 1
        # dfg_dfg_links -> bsz, L_dfg, L_dfg -> 0 or 1
        # ast_code_links -> bsz, L_ast, L -> 0 or 1
        # ast_ast_sims -> bsz, L_ast, L_ast
        
        all_attentions = () if output_attentions else None
        position_bias = None
        hidden_states = self.dropout(inputs_embeds) # bL'd
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        
        for i, layer_module in enumerate(self.block):

            layer_outputs = layer_module(hidden_states, attention_mask, dfg_code_links, dfg_dfg_links, 
                                         ast_code_links, ast_ast_sims,
                                         position_bias=position_bias, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                
            # share the position biases between the layers - after the first layer store them
            if i==0:
                position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    
class StructCoderOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    dfg_loss: Optional[torch.FloatTensor] = None
    ast_loss: Optional[torch.FloatTensor] = None
    lm_logits: torch.FloatTensor = None
    dfg_logits: torch.FloatTensor = None
    ast_logits: torch.FloatTensor = None
    
    
class StructCoderForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, args):
        config = T5Config.from_pretrained('Salesforce/codet5-base')
        if args.model_size=='small':
            config.d_model, config.d_kv, config.d_ff = 256, 32, 1024
            config.num_layers, config.num_decoder_layers, config.num_heads = 5, 5, 8
        super().__init__(config)
        factor = config.initializer_factor
        
        # dfg encoder weights
        self.dfg_node_emb = nn.Parameter(torch.empty((1, 1, self.config.d_model)), 
                                         requires_grad=True)
        self.dfg_node_emb.data.normal_(mean=0.0, std=factor*1.0)
        # ast encoder weights
        self.ast_type_emb = nn.Embedding(args.num_node_types, self.config.d_model)
        self.ast_type_emb.weight.data.normal_(mean=0.0, std=factor*1.0)
        self.ast_depth_emb = nn.Parameter(torch.empty((1, 1, args.max_ast_depth, self.config.d_model)), 
                                          requires_grad=True)
        self.ast_depth_emb.data.normal_(mean=0.0, std=factor*1.0)
        # encoder
        self.encoder = StructCoderEncoderStack(config)
        
        # dfg output 
        self.dfg_bits = 16
        self.dfg_weight1 = nn.Linear(self.dfg_bits, 32, bias=False)
        self.dfg_weight2 = nn.Linear(self.dfg_bits, 32, bias=False)
        self.dfg_b1 = nn.Linear(self.dfg_bits, 1, bias=False)
        self.dfg_b2 = nn.Linear(self.dfg_bits, 1, bias=False)
        self.dfg_b3 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        for layer in [self.dfg_weight1, self.dfg_weight2, self.dfg_b1, self.dfg_b2]:
            layer.weight.data.normal_(mean=0.0, std=factor * 1.0)
        self.dfg_b3.data.normal_(mean=0.0, std=factor * 1.0)
        self.eps=1e-10
        
        # ast output
        self.ast_path_bits = 128
        self.ast_path_head = nn.Linear(self.ast_path_bits, args.max_ast_depth*args.num_node_types, bias=False)
        self.args = args
        
        # load pretrained weights
        if args.model_size!='small':
            args.logger.write('\nLoading StructCoder weights from CodeT5.')
            pt_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
            pt_model_dict = pt_model.state_dict()
            model_dict = self.state_dict()
            not_found = []
            for k in model_dict:
                if k in pt_model_dict:
                    model_dict[k] = pt_model_dict[k]
                else:
                    not_found.append(k)
            self.load_state_dict(model_dict)
            args.logger.write('Could not load weights "'+str(not_found)+'" from pretrained CodeT5.')
        
    # to support generate()
    def forward2(self, encoder_outputs, attention_mask, decoder_input_ids, past_key_values):
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                                        encoder_attention_mask=attention_mask,
                                        past_key_values=past_key_values,
                                        use_cache=self.config.use_cache
                                        )
        sequence_output = decoder_outputs[0] # b,L_out,d
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output) #b,L_out,V
        return Seq2SeqLMOutput(logits=lm_logits, 
                               past_key_values=decoder_outputs.past_key_values, 
                               decoder_hidden_states=decoder_outputs.hidden_states)
        
    # TODO: Add support for output_attentions, output_hidden_states
    def forward(self, inputs=None, outputs=None, 
                encoder_outputs=None, attention_mask=None, decoder_input_ids=None, 
                past_key_values=None, max_length=None, num_beams=None, **kwargs):
        # inputs['input_ids'] -> bsz, L
        # inputs['attention_mask'] -> bsz, L
        
        # inputs['num_dfg_nodes'] -> bsz
        # inputs['dfg_code_links'] -> bsz, L_dfg, L
        # inputs['dfg_dfg_links'] -> bsz, L_dfg, L_dfg
        
        # inputs['ast_code_links'] -> bsz, L_ast, L
        # inputs['ast_ast_sims'] -> bsz, L_ast, L_ast
        # inputs['ast_paths'] -> bsz, L_ast, max_depth -> -1 for masked positions
        
        # outputs['input_ids'] -> bsz, L_out -> first token indicates what to generate
        # outputs['attention_mask'] -> bsz, L_out -> 0/1
        # outputs['dfg_dfg_links'] -> bsz, L_out, L_out -> 0/1 or -1 for masked
        # outputs['ast_paths'] -> bsz, L_out, max_depth -> -1 for masked
        
        # remaining inputs are for forward2
        if encoder_outputs is not None:
            return self.forward2(encoder_outputs, attention_mask, decoder_input_ids, past_key_values)
        
        # 1. Input embedding
        device = inputs['input_ids'].device
        input_embeds = self.shared(inputs['input_ids']) # b,L,d
        attention_mask = inputs['attention_mask']
        L, L_dfg, L_ast = inputs['input_ids'].size()[1], 0, 0
        
        # add dfg nodes to input_embeds and attention_mask
        if inputs.get('num_dfg_nodes', None) is not None: 
            bsz, L_dfg, _ = inputs['dfg_code_links'].size()
            dfg_embeds = self.dfg_node_emb*torch.ones((bsz,L_dfg,1), device=device) # b,L_dfg,d 
            input_embeds = torch.cat((input_embeds, dfg_embeds), dim=1) # b,L+L_dfg,d
            dfg_attention_mask = (torch.arange(L_dfg, device=device)[None,:]<inputs['num_dfg_nodes'][:,None]).int() #b,L_dfg
            attention_mask = torch.cat((attention_mask, dfg_attention_mask), dim=-1) # b,L+L_dfg
        
        # add ast leaves to input_embeds and attention_mask
        if inputs.get('ast_paths',None) is not None:
            L_ast = inputs['ast_paths'].size()[1]
            ast_paths_mask = (inputs['ast_paths']>=0) #b,L_ast,max_depth
            ast_paths = torch.clip(inputs['ast_paths'],0) #b,L_ast,max_depth
            ast_leaf_embeds = self.ast_type_emb(ast_paths) + self.ast_depth_emb # b,L_ast,max_depth,d
            ast_leaf_embeds = (ast_leaf_embeds*ast_paths_mask[:,:,:,None]).sum(dim=2) # b,L_ast,d
            input_embeds = torch.cat((input_embeds, ast_leaf_embeds), dim=1) # b,L+L_dfg+L_ast,d
            ast_attention_mask = torch.clip(ast_paths_mask.sum(dim=-1),0,1) # b,L_ast
            attention_mask = torch.cat((attention_mask, ast_attention_mask), dim=-1) # b,L+L_dfg+L_ast
            
        # 2. StructCoder encoder
        encoder_outputs = self.encoder(input_embeds, attention_mask, inputs.get('dfg_code_links',None), 
                                       inputs.get('dfg_dfg_links',None), inputs.get('ast_code_links',None), 
                                       inputs.get('ast_ast_sims',None))
        
        # 3.0. Generate if outputs is None
        if outputs is None:
            return self.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask, 
                                 max_length=max_length, num_beams=num_beams, 
                                 decoder_start_token_id=self.config.bos_token_id, use_cache=True)

        # 3.1. StructCoder decoder
        decoder_outputs = self.decoder(input_ids=outputs['input_ids'],
                                        attention_mask=outputs['attention_mask'],
                                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                                        encoder_attention_mask=attention_mask,
                                        use_cache=self.config.use_cache,
                                        return_dict=self.config.use_return_dict,
                                        )
        sequence_output = decoder_outputs.last_hidden_state[:,:-1,:] # b,L_out-1,d
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
            
        # 4. Language modeling task
        lm_logits = self.lm_head(sequence_output) #b,L_out-1,V
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), (outputs['input_ids'][:,1:]).reshape(-1))
        
        # 5. DFG links prediction task
        if 'dfg_dfg_links' in outputs:
            dfg_hidden = sequence_output[:, :, :self.dfg_bits]
            hidden1 = self.dfg_weight1(dfg_hidden) # b,L-1,d'
            hidden2 = self.dfg_weight2(dfg_hidden) # b,L-1,d'
            dfg_logits = torch.bmm(hidden1, hidden2.permute([0, 2, 1]).contiguous()) + self.dfg_b1(dfg_hidden) + \
                                             self.dfg_b2(dfg_hidden).permute(0,2,1) + self.dfg_b3 # b,L-1,L-1
            dfg_loss = F.binary_cross_entropy_with_logits(dfg_logits, outputs['dfg_dfg_links'][:,1:,1:], 
                                               reduction='none') # -log p(correct_class)
            is_pos = outputs['dfg_dfg_links'][:,1:,1:]==1
            is_neg = outputs['dfg_dfg_links'][:,1:,1:]==0
            dfg_loss = (dfg_loss*is_pos).sum() / (2*torch.clamp(is_pos.sum(),min=1)) \
                                + (dfg_loss*is_neg).sum() / (2*torch.clamp(is_neg.sum(),min=1))
        else:
            dfg_logits, dfg_loss = None, None
        
        # 6. AST paths prediction task
        if 'ast_paths' in outputs:
            ast_hidden = sequence_output[:, :, self.dfg_bits:self.dfg_bits+self.ast_path_bits]
            ast_logits = self.ast_path_head(ast_hidden) # b,L-1,max_depth*num_node_types
            ast_logits = ast_logits.view(-1, ast_logits.size()[1], self.args.max_ast_depth, self.args.num_node_types)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            ast_loss = loss_fct(ast_logits.view(-1, ast_logits.size(-1)), 
                                (outputs['ast_paths'][:,1:,:]).reshape(-1).long())
        else:
            ast_logits, ast_loss = None, None

        return StructCoderOutput(
                    lm_logits=lm_logits,
                    lm_loss=lm_loss,
                    dfg_logits=dfg_logits,
                    dfg_loss=dfg_loss,
                    ast_logits=ast_logits,
                    ast_loss=ast_loss
                )