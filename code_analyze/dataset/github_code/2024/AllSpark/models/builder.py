import os
from functools import partial

import torch
import torch.nn as nn

from .bridge import PerceiverResampler
from .EVA import VisionTransformer
from .llm.modeling_llama import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification
from .llm.tokenization_llama import LlamaTokenizer
from .modal_backbone import ViT_b
from .modal_tokenizer import HSItokenizer, Visual2DTokenizer
from .task_head import LinearClsHead, MultiLabelLinearClsHead


#############
# Task Head #
#############
def build_taskHead(taskHead_cfg):
    
    print('### TaskHead building')
    if taskHead_cfg['name'] == "LinearClsHead":
        task_head = LinearClsHead(taskHead_cfg['num_classes'], taskHead_cfg['in_channels'])
    elif taskHead_cfg['name'] == "MultiLabelLinearClsHead":
        task_head = MultiLabelLinearClsHead(taskHead_cfg['num_classes'], taskHead_cfg['in_channels'])  
        
    else:    
        raise NotImplementedError
    
    return task_head


#############
#    LLM    #
#############
def build_LLM(cfg):
    
    DEFAULT_PAD_TOKEN = "[PAD]"
    TOKEN_NONE_FLAG = "[NONE]"
    rpath = cfg['LLM_checkpoint']
    assert os.path.exists(rpath)
    
    ## build LLM Tokenizer
    print("### build LLM Tokenizer")
    use_left_pad = True

    num_new_tokens = 0

    tokenizer = LlamaTokenizer.from_pretrained(rpath)

    if tokenizer.pad_token is None:
        num_new_tokens = tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
            
    print("-" * 40)
    print("### Vocab Size: ", len(tokenizer), flush=True)

    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token is not None

    if tokenizer.bos_token is None:
        print("set bos_token to: ", TOKEN_NONE_FLAG, flush=True)
        tokenizer.bos_token = TOKEN_NONE_FLAG

    else:
        print("bos_token, ", tokenizer.bos_token)
        print("bos_token_id, ", tokenizer.bos_token_id)

    if use_left_pad:
        tokenizer.padding_side = "left"

    print("Left Pad: ", use_left_pad, flush=True)

    print("eos_token, ", tokenizer.eos_token)
    print("eos_token_id, ", tokenizer.eos_token_id)

    print("pad_token, ", tokenizer.pad_token)
    print("pad_token_id, ", tokenizer.pad_token_id)

    print("unk_token, ", tokenizer.unk_token, flush=True)
    print("unk_token_id, ", tokenizer.unk_token_id, flush=True)
    print("-" * 40)
    
    ## build LLM backbone
    print("### build LLM backbone")

    text_config = LlamaConfig.from_json_file(os.path.join(rpath, "config.json"))

    text_config.use_flash_attn = False

    text_config.use_adapter = True
    text_config.adapter_freq = 2
    text_config.freeze_params = True
    text_config.label_smoothing = 0.

    model = LlamaForCausalLM.from_pretrained(rpath, config=text_config)

    model.model.padding_idx = tokenizer.pad_token_id

    missing_keys = [n for n, _ in model.named_parameters() if 'adapter' in n]

    if num_new_tokens > 0:
        print("### LLM Vocab Size: ", model.config.vocab_size, flush=True)
        print("### num_new_tokens: ", num_new_tokens, flush=True)
        vocab_size = model.config.vocab_size + num_new_tokens
        assert vocab_size == len(tokenizer)

        model.resize_token_embeddings(vocab_size)
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    adapter_checkpoint = cfg['adapter_checkpoint']
    if adapter_checkpoint:
        print(f"### LLM load params from {adapter_checkpoint}")
        state_dict = torch.load(adapter_checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False) # verified

    print("### Freeze LLM", flush=True)
    for name, param in model.named_parameters():
        if 'adapter' in name:
            continue
        param.requires_grad = False

    return tokenizer, model

def build_LLMForSegCls(cfg):
    
    DEFAULT_PAD_TOKEN = "[PAD]"
    TOKEN_NONE_FLAG = "[NONE]"
    SEP_TOKEN = "</s>"
    CLS_TOKEN = "<s>"
    MASK_TOKEN = "<mask>"
    rpath = cfg['LLM_checkpoint']
    assert os.path.exists(rpath)
    
    ## build LLM Tokenizer
    print("### build LLM Tokenizer")
    use_left_pad = True

    num_new_tokens = 0

    tokenizer = LlamaTokenizer.from_pretrained(rpath)

    if tokenizer.pad_token is None:
        num_new_tokens = tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
                "sep_token": SEP_TOKEN,
                "mask_token": MASK_TOKEN,
                "cls_token": CLS_TOKEN
            }
        )
            
    print("-" * 40)
    print("### Vocab Size: ", len(tokenizer), flush=True)

    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token is not None
    assert tokenizer.sep_token is not None

    if tokenizer.bos_token is None:
        print("set bos_token to: ", TOKEN_NONE_FLAG, flush=True)
        tokenizer.bos_token = TOKEN_NONE_FLAG

    else:
        print("bos_token, ", tokenizer.bos_token)
        print("bos_token_id, ", tokenizer.bos_token_id)

    if use_left_pad:
        tokenizer.padding_side = "left"

    print("Left Pad: ", use_left_pad, flush=True)

    print("eos_token, ", tokenizer.eos_token)
    print("eos_token_id, ", tokenizer.eos_token_id)

    print("pad_token, ", tokenizer.pad_token)
    print("pad_token_id, ", tokenizer.pad_token_id)

    print("unk_token, ", tokenizer.unk_token, flush=True)
    print("unk_token_id, ", tokenizer.unk_token_id, flush=True)
    print("-" * 40)
    
    ## build LLM backbone
    print("### build LLM backbone")

    text_config = LlamaConfig.from_json_file(os.path.join(rpath, "config.json"))

    text_config.use_flash_attn = False

    text_config.use_adapter = True
    text_config.adapter_freq = 2
    text_config.freeze_params = True
    text_config.label_smoothing = 0.

    model = LlamaForSequenceClassification.from_pretrained(rpath, config=text_config)

    model.model.padding_idx = tokenizer.pad_token_id

    missing_keys = [n for n, _ in model.named_parameters() if 'adapter' in n]

    if num_new_tokens > 0:
        print("### LLM Vocab Size: ", model.config.vocab_size, flush=True)
        print("### num_new_tokens: ", num_new_tokens, flush=True)
        vocab_size = model.config.vocab_size + num_new_tokens
        assert vocab_size == len(tokenizer)

        model.resize_token_embeddings(vocab_size)
        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
    
    adapter_checkpoint = cfg['adapter_checkpoint']
    if adapter_checkpoint:
        print(f"### LLM load params from {adapter_checkpoint}")
        state_dict = torch.load(adapter_checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False) # verified

    print("### Freeze LLM", flush=True)
    for name, param in model.named_parameters():
        if 'adapter' in name:
            continue
        param.requires_grad = False

    return tokenizer, model


##################
# modal backbone #
##################
def build_modalBackbone(modalBackbone_cfg):
    if modalBackbone_cfg['name'] == "EVA":
        return build_evaBackbone(modalBackbone_cfg['params_path'], vision_feats_return_layer=modalBackbone_cfg['ret_layers'])
    elif modalBackbone_cfg['name'] == "vit_b":
        return ViT_b(modalBackbone_cfg['ret_layers'])
    
    return NotImplementedError

def build_evaBackbone(params_path, vision_feats_return_layer=-1):
    
    evaBackbone = VisionTransformer(
        embed_dim=1408,
        depth=40,
        num_heads=1408 // 88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=0.4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=False,
        vision_feats_return_layer=vision_feats_return_layer,
    )

    assert os.path.exists(params_path)
    state_dict = torch.load(params_path, map_location="cpu")

    evaBackbone.load_state_dict(state_dict, strict=True)
        
    for name, param in evaBackbone.named_parameters():
        param.requires_grad = False

    return evaBackbone


############
#  bridge  #
############
def build_bridge(bridge_cfg):
    print("### Building Bridge", flush=True)

    model = PerceiverResampler(bridge_cfg['src_dim'], 4096, depth=bridge_cfg['depth'], 
                               num_latents=bridge_cfg['num_latents'], dim_head=bridge_cfg['dim_head'], ff_mult=4)
    
    if bridge_cfg['bridge_checkpoint'] is not None:
        bridge_checkpoint = bridge_cfg['bridge_checkpoint']
        print(f"### LLM load params from {bridge_checkpoint}")
        state_dict = torch.load(bridge_checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    
    if bridge_cfg['freeze_params']:
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model


###################
# modal tokenizer #
###################
def build_modalTokenizer(modalTokenizer_cfg):
    if modalTokenizer_cfg['name'] == 'Visual2DTokenizer':    
        modal_tokenizer = build_visual2DTokenizer(modalTokenizer_cfg['img_size'], 
            embed_dim=modalTokenizer_cfg['embed_dim'], params_path=modalTokenizer_cfg['tokenizer_checkpoint'], 
            freeze_params=modalTokenizer_cfg['freeze_params'], in_channels=modalTokenizer_cfg['in_chans'])    
    elif modalTokenizer_cfg['name'] == 'HIStokenizer':
        modal_tokenizer = HSItokenizer(image_size=modalTokenizer_cfg['image_size'], 
            near_band=modalTokenizer_cfg['near_band'], num_patches=modalTokenizer_cfg['num_patches'], dim=modalTokenizer_cfg['dim'])
        
    else:
        raise NotImplementedError
    
    return modal_tokenizer


def interpolate_pos_embed(model, checkpoint_model, pos_embed_name="pos_embed"):
    if pos_embed_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_embed_name].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_embed_name] = new_pos_embed

def build_visual2DTokenizer(img_size, embed_dim, params_path, freeze_params, in_channels=3):
    
    tokenizer = Visual2DTokenizer(img_size=img_size, patch_size=14, embed_dim=embed_dim, in_chans=in_channels)
    
    if params_path:
        assert os.path.exists(params_path)
        state_dict = torch.load(params_path, map_location="cpu")
        interpolate_pos_embed(tokenizer, state_dict)
        tokenizer.load_state_dict(state_dict, strict=True)
        
    if freeze_params:
        for name, param in tokenizer.named_parameters():
            param.requires_grad = False
            
    return tokenizer