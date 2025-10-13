from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

def build_model(hp_args):
    if hp_args.use_flash_attn:
        print("Use Flash Attn!!!")
    args_load = {"pretrained_model_name_or_path": hp_args.init_ckpt, 
                 "use_flash_attention_2": hp_args.use_flash_attn, 
                 "attention_dropout": hp_args.dropout, 
                 "load_in_8bit": hp_args.eval_only}
                    

    if "70b" in hp_args.init_ckpt:
        args_load['pretraining_tp'] = 1
    load_class = AutoModelForCausalLM
    model = load_class.from_pretrained(**args_load)
    tokenizer = AutoTokenizer.from_pretrained(
            hp_args.init_ckpt,
            model_max_length=hp_args.max_seq_len,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model
