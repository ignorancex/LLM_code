# Example of running generation with PARGS on the tldr dataset, with GPT2-SFT LM and deberta-v3 reward model

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import time
import pandas as pd
import os
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"
tqdm.pandas()

# =========================================================================================
# =================================== Load the models ===================================
# GPT2 model
gpt2_tokenizer = AutoTokenizer.from_pretrained("vistagi/gpt2-large-tldr-sum")
gpt2_model = AutoModelForCausalLM.from_pretrained("vistagi/gpt2-large-tldr-sum").to(device)

# decoding reward models
decode_reward_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large")
decode_reward_model_partial = AutoModelForSequenceClassification.from_pretrained("./partial_reward_modeling_tldr").to(device)


# =========================================================================================
def get_reward(Q, A, reward_tokenizer, reward_model, RM_cache=None):
    tokenizer_output = reward_tokenizer(Q, A, return_tensors='pt').to(device)
    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    input_ids_last = input_ids[:, -1:]
    attention_mask_last = attention_mask[:, -1:]

    # Only need to pass the info of the last added token
    last_tokenizer_output = {
        "input_ids": input_ids_last,
        "attention_mask": attention_mask_last,
    }
    
    if RM_cache == None: # run with the entire input
        rm_out = reward_model(**tokenizer_output, return_dict=True, use_cache=True)
    else: # run with the added token, using previous cache
        rm_out = reward_model(**last_tokenizer_output, return_dict=True, use_cache=True, past_key_values=RM_cache)
    
    external_reward = rm_out.logits[0].cpu().detach().item() # reward value as in PPO and DPO
    
    # Since the last token is a candidate, we do not cache the last token
    cache = [
    (
        past_key_value[0][:, :, :-1, :],  # Exclude last key
        past_key_value[1][:, :, :-1, :]   # Exclude last value
    )
    for past_key_value in rm_out.past_key_values
    ]
    
    # clear unused cache
    del rm_out
    del RM_cache
    torch.cuda.empty_cache()
    
    return external_reward, cache

# def get_reward(Q, A, reward_tokenizer, reward_model):
#     inputs = reward_tokenizer(Q, A, return_tensors='pt').to(device)
#     external_reward = reward_model(**inputs).logits[0].cpu().detach().item()
#     return external_reward


# ======================================================================================================
# ============================================== PARGS DECODING =========================================
# mode1: 1 - greedy, 2 - sampling

def PARGS_decoding(llm_model=None, llm_tokenizer=None, 
                reward_model=None, reward_tokenizer=None, topk=10,
                prompt=None, max_generation_length=64, mode=2, w=2.0):
    
    tokenizer_output = llm_tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokenizer_output.input_ids
    
    # use sequence to keep the entire generation
    sequence = torch.tensor([[]],dtype=torch.int64).to(device)
    
    past_key_values = None
    for t in range(0, max_generation_length):
        RM_cache = None
        if t == 0:
            output = llm_model.generate(**tokenizer_output, max_new_tokens=1, 
                                    pad_token_id=llm_tokenizer.eos_token_id, 
                                    output_scores=True, return_dict_in_generate=True, 
                                    renormalize_logits=True, use_cache=True)
        else:
            output = llm_model.generate(inputs=torch.cat((input_ids, sequence), dim=-1), max_new_tokens=1, 
                                pad_token_id=llm_tokenizer.eos_token_id, 
                                output_scores=True, return_dict_in_generate=True, 
                                renormalize_logits=True, use_cache=True, past_key_values=past_key_values)
        
        past_key_values = output.past_key_values
        topk_tokens = torch.topk(output["scores"][0][0], topk)

        # create tensor of the Reward-guided score
        RG_score = []
        for i in range(0, topk):
            token_index = topk_tokens.indices[i].reshape(1,1)
            token_prob = topk_tokens.values[i].item()
            temp_sequence = torch.cat((sequence, token_index), dim=-1)
            sequence_reward, RM_cache = get_reward(prompt, llm_tokenizer.decode(temp_sequence[0]), 
                                        reward_tokenizer, reward_model, RM_cache=RM_cache)
            RG_score.append(token_prob + w * sequence_reward)

        score_tensor = torch.tensor(RG_score)
        
        if mode == 1:
            sampled_id = torch.topk(score_tensor, 1).indices[0]
        elif mode == 2:
            sampled_id = torch.distributions.categorical.Categorical(logits=score_tensor).sample().item()
            
        sampled_token = topk_tokens.indices[sampled_id].reshape(1,1)
        sequence = torch.cat((sequence, sampled_token), dim=-1)
        
        if sequence[0][-1].item() == llm_tokenizer.eos_token_id:
            print(f"EOS BREAK: {t}")
            break
        
        # clear unused cache
        del output
        del RM_cache
        torch.cuda.empty_cache()
    
    generation = llm_tokenizer.decode(sequence[0], skip_special_tokens=True)
    return {"sequence": generation}


# ==========================================================================================================
# ========================================= testing function ===============================================
def test(prompt=None, topk=10, max_generation_length=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_inputs = gpt2_tokenizer(prompt, return_tensors='pt').to(device)

    # RGNS
    PARGS_output = PARGS_decoding(llm_model=gpt2_model, llm_tokenizer=gpt2_tokenizer, 
                reward_model=decode_reward_model_partial, reward_tokenizer=decode_reward_tokenizer,
                topk=topk, prompt=prompt, max_generation_length=max_generation_length, mode1=2, w=2.0)
    PARGS_score = get_reward(prompt, PARGS_output['sequence'], evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"PARGS generation: {PARGS_output['sequence']}")
    print(f"PARGS_score: {PARGS_score}")
    print("\n\n")
    
    return 0


# ============================================================================
# =================================== Main ===================================
def test_main(sample_size=50, seed=42, topk=10, max_generation_length=64):

    # load datasets
    tldr_dataset = load_dataset("CarperAI/openai_summarize_tldr")
    test_data = tldr_dataset["test"]
    test_data = test_data.shuffle(seed=seed)
    
    all_results = pd.DataFrame()

    # run tests
    for i in range(0, sample_size):

        prompt = "Context: " + test_data[i]['prompt']
        label = test_data[i]['label']
        
        print(f"Prompt:\n{prompt}\n")
        print(f"Label:\n{label}")
        
        test(prompt, topk, max_generation_length)

if __name__ == "__main__":
    test_main(sample_size=100, seed=42, topk=10, max_generation_length=64)

