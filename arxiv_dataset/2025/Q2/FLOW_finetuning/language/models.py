import torch
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
)
from utils.data_utils import *

def create_model_tokenizer(num_labels, args):

    if 'roberta' in args.model:

        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.to(args.device)

    return model, tokenizer


def create_peft_model(model, args):

    if 'roberta' in args.model:

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.rslora,
            target_modules=["query", "value", "attention.output.dense", "output.dense"],
        )

    elif 't5' in args.model:

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.rslora,
            target_modules=["q", "v", "k", "o", "wi", "wo"],
        )

    model = get_peft_model(model, peft_config)

    model.to(args.device)

    return model, peft_config


def create_model_tokenizer_it(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map=None,
    ) 
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if hasattr(args, "tokenizer") else args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    #model.to(args.device)

    return model, tokenizer

def create_reweight_model(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.reweight_sample_model, 
        device_map=None,
    ) 
    
    return model

def create_model_tokenizer_cr(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        torch_dtype = torch.bfloat16) 
    
    if "llama" in args.model:

        if "Llama-3" in args.model:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                model_max_length=args.max_seq_length,
                padding="max_length",
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                model_max_length=args.max_seq_length,
                padding="max_length",
            )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            model_max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"


    return model, tokenizer


def create_peft_model_it(model, args):

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    return model, peft_config



def create_peft_model_cr(model, args):

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    return model, peft_config

