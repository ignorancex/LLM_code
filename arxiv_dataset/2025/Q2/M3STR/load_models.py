import torch
from vllm import LLM
from transformers import (
    AutoModel,
    AutoTokenizer,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration
)
from utils import model_path_map



def load_llava(model_name):
    prompt_template = "USER: <image>\n{}\nASSISTANT:"
    # model_type = args.model_type
    model_path = model_path_map["llava"][model_name]
    llm = LLM(model=model_path, gpu_memory_utilization=0.9, disable_mm_preprocessor_cache=True)
    stop_token_ids = None
    return llm, prompt_template, stop_token_ids



def load_chameleon(model_name):
    prompt_template = "{}<image>"
    model_path = model_path_map["chameleon"][model_name]
    llm = LLM(model=model_path, gpu_memory_utilization=0.8)
    stop_token_ids = None
    return llm, prompt_template, stop_token_ids


def load_instruct_blip(model_name):
    prompt_template = "<Image>Question: {} Answer:"
    model_path = model_path_map["instructblip"][model_name]
    llm = InstructBlipForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    processor = InstructBlipProcessor.from_pretrained(model_path)
    return llm, prompt_template, processor


def load_internvl(model_name):
    model_path = model_path_map["internvl"][model_name]
    print(model_path)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=5,
        gpu_memory_utilization=0.9
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    """
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                tokenize=False,
                add_generation_prompt=True)
    """
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, tokenizer, stop_token_ids



def run_minicpm(model_name):
    model_path = model_path_map["minicpm"][model_name]
    print(model_path)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        disable_mm_preprocessor_cache=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if "2.6" in model_name:
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    elif "2.5" in model_name:
        stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
    else:
        stop_token_ids = [tokenizer.eos_id]
    return llm, tokenizer, stop_token_ids



def load_phi3v(model_name):

    prompt = "<|user|>\n<|image_1|>\n{}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    model_path = model_path_map["phi"][model_name]
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=5,
        gpu_memory_utilization=0.48
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

