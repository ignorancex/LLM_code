from colorama import Fore, Style
import logging
from typing import List, Dict, Literal, Tuple
import time
from vllm import LLM, RequestOutput
from transformers import AutoTokenizer, AutoConfig
import torch
from tqdm import tqdm
import math
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vllm_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name: str,
    seed: int,
    tensor_parallel_size: int,
    v100: bool,
    max_model_len: int,
): 
    # Load model and tokenizer
    logger.info(f"Loading model: {Fore.GREEN}{model_name_or_path}{Style.RESET_ALL}")
    start_time = time.time()
    config = AutoConfig.from_pretrained(model_name_or_path)


    v100_parameters = {
        'dtype': torch.float16,
        'enable_chunked_prefill': False,
        'gpu_memory_utilization': 0.85,
        "max_model_len": 4900,
    }

    a6000_parameters = {
        'quantization': 'fp8',
        # 'dtype': torch.bfloat16,
        'gpu_memory_utilization': 0.85,
        "max_model_len": 4900
    }
    model = LLM(
        model = model_name_or_path,
        seed = seed,
        tensor_parallel_size = tensor_parallel_size,
        swap_space = 0,
        **(v100_parameters if v100 else a6000_parameters),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def get_output_text(
    outputs,
    post_processor = lambda x: x,
):
    output_texts = []
    for output in outputs:
        assert isinstance(output.outputs, list)
        output_texts.append(
            [post_processor(out.text) for out in output.outputs]
        )
    
    if isinstance(output_texts[0], List) and len(output_texts[0]) == 1:
        output_texts = [out[0] for out in output_texts]

    logger.info(f"{Fore.RED}Before post-processing{Style.RESET_ALL}: {outputs[0].outputs[0].text}")
    logger.info(f"{Fore.GREEN}After post-processing{Style.RESET_ALL}: {output_texts[0]}")

    return output_texts

def median_index(prob_vector):
    # prob_vector is shape (n,)
    cdf = np.cumsum(prob_vector)
    # Find smallest index i with cdf[i] >= 0.5
    index = np.searchsorted(cdf, 0.5)
    return index

def medians_of_stack(prob_matrix):
    # prob_matrix is shape (m, n), where m is number of distributions
    # and n is dimension of the simplex
    # Compute CDF row-by-row and apply median_index
    medians = [median_index(row) for row in prob_matrix]
    return np.array(medians)

def calcaulte_weighted_score_from_probs(
    score_probs: List,
    score_grids: List,
    effective_temperature: float = 0.0,
    use_median: bool = False,
) -> List[float]:
    score_probs = torch.tensor(score_probs)
    score_grids = torch.tensor(score_grids)
    assert score_probs.shape[-1] == score_grids.shape[-1], f"Last dimension of all_probs ({score_probs.shape[-1]}) and score_grids ({score_grids.shape[-1]}) do not match."
    if effective_temperature > 0.0:
        alpha = 1.0 / effective_temperature - 1
        tempered_probs = torch.pow(score_probs, alpha)
        tempered_probs = tempered_probs / (tempered_probs.sum(dim=-1, keepdim=True) + 1e-20)
        # Check if contains nan
        if torch.isnan(tempered_probs).sum() > 0:
            # Print the rows that contain nan
            row_idx = torch.isnan(tempered_probs).any(dim=-1)
            # Print the first row that contains nan 
            logger.info(f"Row with nan: {score_probs[row_idx][0]}")
            logger.info(f"Row with nan: {tempered_probs[row_idx][0]}")
    else:
        tempered_probs = score_probs
    
    if len(score_grids.shape) == 1:
        score_grids = score_grids.unsqueeze(0)
    
    # Get the wieghted scores
    weighted_scores = (score_grids * tempered_probs).sum(dim=-1)
    # Get the median from the tempered_probs
    median_scores = medians_of_stack(tempered_probs) + 1
    if use_median:
        return median_scores.tolist()
    else:
        return weighted_scores.tolist()
    
    

def raft_score_processor(
    score_token_probs: torch.Tensor,
    token_idx_to_score: Dict[int, int],
    effective_temperature: float = 0.0,
) -> List[float]:
    logger.info(f"score_token_probs have shape: {score_token_probs.shape}")
    raft_scores = []
    score_grids = [v for _, v in token_idx_to_score.items()]
    score_grids = torch.tensor(score_grids).unsqueeze(0)
    
    if effective_temperature > 0.0:
        alpha = 1.0 / effective_temperature - 1
        score_token_probs = score_token_probs ** alpha
        score_token_probs = score_token_probs / score_token_probs.sum(dim=-1, keepdim=True)

    logger.info(f"Score token probs shape: {score_token_probs.shape}")
    logger.info(f"Score grids shape: {score_grids}")
    weighted_scores = (score_grids * score_token_probs).sum(dim=-1)
    logger.info(f"Weighted scores shape: {weighted_scores.shape}")
    raft_scores = weighted_scores.tolist()

    return raft_scores

def get_score_probs_from_transformer_models(
    model,
    tokenizer,
    inputs: List[str],
    token_idx_to_score: Dict[int, int],
    batch_size: int = 1,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    # Get the score grids and score token ids
    score_token_ids = [k for k in token_idx_to_score.keys()]

    # Add pad token if not already in the tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_batches = math.ceil(len(inputs) / batch_size)

    score_probs = []

    for i in tqdm(range(total_batches), desc="Getting logits", total=total_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_inputs = tokenizer(
            inputs[start:end], 
            return_tensors='pt', 
            padding=True,
            padding_side = 'right',
            truncation=True,
        ).to(device)
        seq_len = batch_inputs['attention_mask'].sum(-1)
        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits.detach().cpu()
        for idx, (logit, sql) in enumerate(zip(logits, seq_len)):
            # logger.info(f"Sequence length is {sql}")
            # current_token_seq = batch_inputs['input_ids'][idx]
            # logger.info(f"The input token ids are: {current_token_seq}")
            # logger.info(f"The input tokens are: {tokenizer.convert_ids_to_tokens(current_token_seq)}")
            # logger.info(f"The token we are extracting logits for is: {current_token_seq[sql - 1]}")
            # logger.info(f"This token corresponds to {tokenizer.convert_ids_to_tokens([current_token_seq[sql - 1]])}")
            all_probs = torch.nn.functional.softmax(logit[sql - 1], dim=-1)
            score_token_probs = all_probs[score_token_ids]
            score_probs.append(score_token_probs)
    score_probs = torch.stack(score_probs)

    return score_probs

def get_perplexity_from_transformer_models(
    model,
    tokenizer,
    inputs: List[str],
    token_idx_to_score: Dict[int, int],
    batch_size: int = 1,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    # Get the score grids and score token ids
    score_token_ids = [k for k in token_idx_to_score.keys()]

    # Add pad token if not already in the tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_batches = math.ceil(len(inputs) / batch_size)

    perplexities = []
    score_probs = []

    for i in tqdm(range(total_batches), desc="Getting logits", total=total_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_inputs = tokenizer(
            inputs[start:end], 
            return_tensors='pt', 
            padding=True,
            padding_side = 'right',
            truncation=True,
        ).to(device)

        instructions = [input_str.split('[/INST]')[0] for input_str in inputs[start:end]]
        tokenized_instructions = tokenizer(instructions, return_tensors='pt', padding=True, padding_side='right', truncation=True)
        instruction_lens = tokenized_instructions['attention_mask'].sum(-1) - 1 # The -1 is to account for the eos token, which is not part of the instruction when there is response
        # logger.info(f"Instruction lengths are: {instruction_lens}")

        seq_len = batch_inputs['attention_mask'].sum(-1)
        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits.detach().cpu()
        for idx, (logit, inst_len, sql) in enumerate(zip(logits, instruction_lens, seq_len)):
            all_prob_at_score_pos = torch.nn.functional.softmax(logit[sql - 1], dim=-1)
            score_token_probs = all_prob_at_score_pos[score_token_ids]
            score_probs.append(score_token_probs)

            # Position: 0 1 2 3 4 5 6 7
            # Inputs:   I I I I C C C C
            # Outputs:  I I I C C C C S
            # Instruction length: 4. CoT Len: 4. SQL: 8
            # Perplexirt of CoT = logit[3:7] = logit[inst_len - 1: sql - 1]

            # Calculate perplexity
            seq_logits = logit[inst_len - 1: sql - 1]
            # logger.info(f"Getting the sequence logits from {inst_len - 1} to {sql - 1}")
            # logger.info(f"The token at {inst_len - 1} is {tokenizer.convert_ids_to_tokens([batch_inputs['input_ids'][idx][inst_len - 1]])}")
            # logger.info(f"The token at {sql - 1} is {tokenizer.convert_ids_to_tokens([batch_inputs['input_ids'][idx][sql - 1]])}")

            # Calculate perplexity
            reshaped_logits = seq_logits.view(-1, seq_logits.size(-1))
            targets = batch_inputs['input_ids'][idx][inst_len: sql].view(-1).to(reshaped_logits.device)
            perplexity = torch.nn.functional.cross_entropy(reshaped_logits, targets, reduction='mean').exp()
            perplexities.append(perplexity.item())
            # logger.info(f"Perplexity is: {perplexity.item()}")
    
    score_probs = torch.stack(score_probs)
    perplexities = np.array(perplexities)
    return perplexities, score_probs
