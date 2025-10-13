# hf.py
import os
import copy
import asyncio
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Union
from util import logger

import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    RevisionNotFoundError
)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

@lru_cache(maxsize=5)
def initialize_hf_client(
    model_name: str, 
    device_map: str = "auto", 
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> tuple:
    logger.info(f"Loading Hugging Face model: {model_name}")

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
            load_in_8bit=load_in_8bit if not load_in_4bit else False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer, generation_config
    
    except (RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError) as e:
        logger.error(f"Model loading error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        raise

def format_chat_messages(
    messages: List[Dict[str, str]], 
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None
) -> str:
    formatted_messages = []
    
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        if "apply_chat_template" in dir(tokenizer):
            return tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    except Exception:
        logger.warning("Chat template application failed, using fallback formatting")
    
    return "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in formatted_messages]
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (RuntimeError, OSError, torch.cuda.OutOfMemoryError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def hf_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    **generation_params
) -> str:
    model, tokenizer, generation_config = initialize_hf_client(model_name)
    
    input_text = format_chat_messages(
        history_messages + [{"role": "user", "content": prompt}],
        tokenizer,
        system_prompt
    )
    
    params = generation_config.to_dict()
    params.update({
        "max_new_tokens": generation_params.get("max_tokens", 512),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })
    
    generation_params.pop("max_tokens", None)
    generation_params.pop("hashing_kv", None)
    generation_params.pop("keyword_extraction", None)
    generation_params.pop("max_tokens", None)
    
    params.update(generation_params)
    
    def _sync_generate():
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=8192  
        ).to(model.device)
        
        stop_token_ids = [tokenizer.eos_token_id]
        if tokenizer.pad_token_id:
            stop_token_ids.append(tokenizer.pad_token_id)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                **params
            )
        
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
    
    return await asyncio.to_thread(_sync_generate)

async def hf_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    model_name: Optional[str] = None,
    keyword_extraction: bool = False,
    **kwargs
) -> Union[str, Dict]:

    if not model_name:
        if "hashing_kv" in kwargs and "global_config" in kwargs["hashing_kv"]:
            model_name = kwargs["hashing_kv"]["global_config"]["llm_model_name"]
        else:
            raise ValueError("Model name not provided and could not be obtained from configuration")
    
    result = await hf_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    return result
