from typing import List, Dict
from vllm import SamplingParams
from evaluate import extract_answer
from agentic_reason.config import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_CODE_QUERY,
    END_CODE_QUERY,
    CHAT_TEMPLATE
)

from prompts import (
    get_webpage_to_reasonchain_instruction,
)

def generate_webpage_to_reasonchain_batch(
    original_questions: List[str],
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    batch_output_records: List[Dict],
    llm,
    tokenizer,
    max_tokens: int = 32768,
    coherent: bool = False,
) -> List[str]:
    """
    Generate reasoning chains from webpage content in batch mode.
    
    Args:
        original_questions: List of original questions
        prev_reasonings: List of previous reasoning steps
        search_queries: List of search queries used
        documents: List of retrieved documents
        dataset_name: Name of the dataset being processed
        batch_output_records: List to collect batch outputs
        llm: Language model instance
        tokenizer: Tokenizer instance
        max_tokens: Maximum tokens for generation
        coherent: Whether to maintain coherence in generation
    
    Returns:
        List[str]: List of extracted information from model outputs
    """
    # Reference to original implementation
    # Reference: scripts/run_search_o1.py lines 286-325
    
    user_prompts = [
        get_webpage_to_reasonchain_instruction(r, sq, doc)
        for r, sq, doc in zip(prev_reasonings, search_queries, documents)
    ]

    prompts = [{"role": "user", "content": up} for up in user_prompts]
    prompts = [tokenizer.apply_chat_template([p],chat_template=CHAT_TEMPLATE,tokenize=False, add_generation_prompt=True) for p in prompts]

    output = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
        )
    )

    raw_outputs = [out.outputs[0].text for out in output]
    extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

    for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
        batch_output_records.append({
            'prompt': p,
            'raw_output': r,
            'extracted_info': e
        })

    return extracted_infos

def run_generation(
    sequences: List[Dict],
    llm,
    tokenizer,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float = 1.1,
    max_tokens: int = 8192,
    stop_tokens: List[str] = None
) -> List:
    """
    Run generation on a batch of sequences.
    
    Args:
        sequences: List of sequence dictionaries containing prompts
        llm: Language model instance
        tokenizer: Tokenizer instance
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        max_tokens: Maximum tokens to generate
        stop_tokens: List of stop tokens
    
    Returns:
        List of generation outputs
    """
    # Reference: scripts/run_rag_agent.py lines 342-357
    if repetition_penalty is None:
        repetition_penalty = 1.1
    if max_tokens is None:
        max_tokens = 8192
    
    prompts = [s['prompt'] for s in sequences]
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop=[END_SEARCH_QUERY, END_CODE_QUERY, tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    output_list = llm.generate(prompts, sampling_params=sampling_params)

    # Fix outputs that end with BEGIN_SEARCH_QUERY by adding END_SEARCH_QUERY
    for out in output_list:
        text = out.outputs[0].text
        if BEGIN_SEARCH_QUERY in text and END_SEARCH_QUERY not in text:
            out.outputs[0].text = text + END_SEARCH_QUERY
        if BEGIN_CODE_QUERY in text and END_CODE_QUERY not in text:
            out.outputs[0].text = text + END_CODE_QUERY
    
    return output_list
