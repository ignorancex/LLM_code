# This script generates a .csv file containing the reward scores of all baseline generations,
# and json files containing the generation sentences.

# Use show_result.py to make plot and view the average reward.
# use GPT4-eval to see win-loss rate by GPT4 evaluation


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

ARGS = []
PARGSG = []
PARGS = []
PPO = []
DPO = []
BESTN = []
NS = []

# =========================================================================================
# =================================== Load the models ===================================
# GPT2 model
gpt2_tokenizer = AutoTokenizer.from_pretrained("vistagi/gpt2-large-tldr-sum")
gpt2_model = AutoModelForCausalLM.from_pretrained("vistagi/gpt2-large-tldr-sum").to(device)

# decoding reward models
decode_reward_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large")
decode_reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large").to(device)
decode_reward_model_partial = AutoModelForSequenceClassification.from_pretrained("./partial_reward_modeling_tldr").to(device)

# evaluation reward model
evaluate_reward_tokenizer = decode_reward_tokenizer
evaluate_reward_model = decode_reward_model

# DPO model
DPO_model = AutoModelForCausalLM.from_pretrained("./GPT2_large_tldr_DPO").to(device)

# PPO model
PPO_model = AutoModelForCausalLM.from_pretrained("vistagi/gpt2-large-tldr-sum-rlhf").to(device)


# =========================================================================================

def get_reward(Q, A, reward_tokenizer, reward_model):
    inputs = reward_tokenizer(Q, A, return_tensors='pt').to(device)
    external_reward = reward_model(**inputs).logits[0].cpu().detach().item()
    return external_reward

# ======================================================================================================
# ==================================== ARGS DECODING ===================================================
def ARGS_decoding(llm_model=None, llm_tokenizer=None, 
                reward_model=None, reward_tokenizer=None, topk=10,
                prompt=None, max_generation_length=64, w=1):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_ids = llm_tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    sequence = torch.tensor([[]],dtype=torch.int64).to(device)
    
    for t in range(0, max_generation_length):
        if t == 0:
            output = llm_model.generate(inputs=input_ids, max_new_tokens=1, 
                                    pad_token_id=llm_tokenizer.eos_token_id, 
                                    output_scores=True, return_dict_in_generate=True, 
                                    renormalize_logits=True)
        else:
            output = llm_model.generate(inputs=torch.cat((input_ids, sequence), dim=-1), max_new_tokens=1, 
                                pad_token_id=llm_tokenizer.eos_token_id, 
                                output_scores=True, return_dict_in_generate=True, 
                                renormalize_logits=True)
        
        topk_tokens = torch.topk(output["scores"][0][0], topk)
        RG_score = []
        
        # create the vector of the Reward-guided score
        for i in range(0, topk):
            token_index = topk_tokens.indices[i].reshape(1,1)
            token_prob = topk_tokens.values[i].item()
            temp_sequence = torch.cat((sequence, token_index), dim=-1)
            sequence_reward = get_reward(prompt, llm_tokenizer.decode(temp_sequence[0]), 
                                        reward_tokenizer, reward_model)
            RG_score.append(token_prob + w * sequence_reward)
            
        score_tensor = torch.tensor(RG_score)
        sampled_id = torch.topk(score_tensor, 1).indices[0]
            
        sampled_token = topk_tokens.indices[sampled_id].reshape(1,1)
        sequence = torch.cat((sequence, sampled_token), dim=-1)
        if sequence[0][-1].item() == llm_tokenizer.eos_token_id:
            print(f"EOS BREAK: {t}")
            break

    generation = llm_tokenizer.decode(sequence[0], skip_special_tokens=True)
    return {"sequence": generation}


# ======================================================================================================
# ============================================== RGNS DECODING =========================================
# mode1: 1 - greedy, 2 - sampling

def PARGS_decoding(llm_model=None, llm_tokenizer=None, 
                reward_model=None, reward_tokenizer=None, topk=10,
                prompt=None, max_generation_length=64, mode1=2, w=1):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_ids = llm_tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    sequence = torch.tensor([[]],dtype=torch.int64).to(device)
    
    for t in range(0, max_generation_length):
        if t == 0:
            output = llm_model.generate(inputs=input_ids, max_new_tokens=1, 
                                    pad_token_id=llm_tokenizer.eos_token_id, 
                                    output_scores=True, return_dict_in_generate=True, 
                                    renormalize_logits=True)
        else:
            output = llm_model.generate(inputs=torch.cat((input_ids, sequence), dim=-1), max_new_tokens=1, 
                                pad_token_id=llm_tokenizer.eos_token_id, 
                                output_scores=True, return_dict_in_generate=True, 
                                renormalize_logits=True)
        
        topk_tokens = torch.topk(output["scores"][0][0], topk)
        RG_score = []
        
        # create the vector of the Reward-guided score
        for i in range(0, topk):
            token_index = topk_tokens.indices[i].reshape(1,1)
            token_prob = topk_tokens.values[i].item()
            temp_sequence = torch.cat((sequence, token_index), dim=-1)
            sequence_reward = get_reward(prompt, llm_tokenizer.decode(temp_sequence[0]), 
                                        reward_tokenizer, reward_model)

            RG_score.append(token_prob + w * sequence_reward)

        score_tensor = torch.tensor(RG_score)
        
        if mode1 == 1:
            sampled_id = torch.topk(score_tensor, 1).indices[0]
        elif mode1 == 2:
            sampled_id = torch.distributions.categorical.Categorical(logits=score_tensor).sample().item()
            
        sampled_token = topk_tokens.indices[sampled_id].reshape(1,1)
        sequence = torch.cat((sequence, sampled_token), dim=-1)
        
        if sequence[0][-1].item() == llm_tokenizer.eos_token_id:
            print(f"EOS BREAK: {t}")
            break
    
    generation = llm_tokenizer.decode(sequence[0], skip_special_tokens=True)
    return {"sequence": generation}


# ==========================================================================================================
# ========================================= testing function ===============================================
def test(prompt=None, topk=10, max_generation_length=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ARGS_score = 0
    RGNS_score = 0
    RGGR_score = 0
    PPO_score = 0
    DPO_score = 0
    BestN_score = 0
    NS_score = 0
    
    model_inputs = gpt2_tokenizer(prompt, return_tensors='pt').to(device)
    
    # ARGS
    ARGS_output = ARGS_decoding(llm_model=gpt2_model, llm_tokenizer=gpt2_tokenizer, 
                reward_model=decode_reward_model, reward_tokenizer=decode_reward_tokenizer,
                topk=topk, prompt=prompt, max_generation_length=max_generation_length, w=1)
    ARGS_score = get_reward(prompt, ARGS_output['sequence'], evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"ARGS generation: {ARGS_output['sequence']}")
    print(f"ARGS_score: {ARGS_score}")
    print("\n\n")
    ARGS.append({"prompt": prompt, "result": ARGS_output['sequence']})
    
    # DPO
    DPO_output = DPO_model.generate(**model_inputs, max_new_tokens=max_generation_length, 
                                    do_sample=True, top_p=0, top_k=topk)
    DPO_generation = gpt2_tokenizer.decode(DPO_output[0][len(model_inputs[0]):], skip_special_tokens=True)
    print(f"DPO generation: {DPO_generation}")
    DPO_score = get_reward(prompt, DPO_generation, evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"DPO_score: {DPO_score}")
    print("\n\n")
    DPO.append({"prompt": prompt, "result": DPO_generation})
    
    # PPO
    PPO_output = PPO_model.generate(**model_inputs, max_new_tokens=max_generation_length, 
                                    do_sample=True, top_p=0, top_k=topk)
    PPO_generation = gpt2_tokenizer.decode(PPO_output[0][len(model_inputs[0]):], skip_special_tokens=True)
    print(f"PPO generation: {PPO_generation}")
    PPO_score = get_reward(prompt, PPO_generation, evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"PPO_score: {PPO_score}")
    print("\n\n")
    PPO.append({"prompt": prompt, "result": PPO_generation})
    
    # PARGSG
    PARGSG_output = PARGS_decoding(llm_model=gpt2_model, llm_tokenizer=gpt2_tokenizer, 
                reward_model=decode_reward_model_partial, reward_tokenizer=decode_reward_tokenizer,
                topk=topk, prompt=prompt, max_generation_length=max_generation_length, mode1=1, w=1.0)
    PARGSG_score = get_reward(prompt, PARGSG_output['sequence'], evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"PARGSG generation: {PARGSG_output['sequence']}")
    print(f"PARGSG_score: {PARGSG_score}")
    print("\n\n")
    PARGSG.append({"prompt": prompt, "result": PARGSG_output['sequence']})
    
    
    # PARGS
    PARGS_output = PARGS_decoding(llm_model=gpt2_model, llm_tokenizer=gpt2_tokenizer, 
                reward_model=decode_reward_model_partial, reward_tokenizer=decode_reward_tokenizer,
                topk=topk, prompt=prompt, max_generation_length=max_generation_length, mode1=2, w=2.0)
    PARGS_score = get_reward(prompt, PARGS_output['sequence'], evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"PARGS generation: {PARGS_output['sequence']}")
    print(f"PARGS_score: {PARGS_score}")
    print("\n\n")
    PARGS.append({"prompt": prompt, "result": PARGS_output['sequence']})
    
    # BestN
    BestN_generations = []
    BestN_scores = []
    for i in range(0, topk):
        NS_output = gpt2_model.generate(**model_inputs, max_new_tokens=max_generation_length, 
                                        do_sample=True, top_p=0, top_k=topk)
        NS_generation = gpt2_tokenizer.decode(NS_output[0][len(model_inputs[0]):], skip_special_tokens=True)
        BestN_generations.append(NS_generation)
    
    for i in range(0, topk):
        NS_score = get_reward(prompt, BestN_generations[i], evaluate_reward_tokenizer, evaluate_reward_model)
        BestN_scores.append(NS_score)
    index_of_max = BestN_scores.index(max(BestN_scores))
    BestN_generation = BestN_generations[index_of_max]
    BestN_score = BestN_scores[index_of_max]
    print(f"BestN generation: {BestN_generation}")
    print(f"BestN_score: {BestN_score}")
    print("\n\n")
    BESTN.append({"prompt": prompt, "result": BestN_generation})
    
    # NS
    NS_output = gpt2_model.generate(**model_inputs, max_new_tokens=max_generation_length, 
                                    do_sample=True, top_p=0, top_k=topk)
    NS_generation = gpt2_tokenizer.decode(NS_output[0][len(model_inputs[0]):], skip_special_tokens=True)
    print(f"NS generation: {NS_generation}")
    NS_score = get_reward(prompt, NS_generation, evaluate_reward_tokenizer, evaluate_reward_model)
    print(f"NS_score: {NS_score}")
    print("\n\n")
    NS.append({"prompt": prompt, "result": NS_generation})
    
    ret = {"ARGS": round(ARGS_score, 4),
            "PPO": round(PPO_score, 4), 
            "DPO": round(DPO_score, 4),
            "PARGSG": round(RGGR15_score, 4),
            "PARGS": round(RGNS25_score, 4),
            "BestN": round(BestN_score, 4),
            "NS": round(NS_score, 4),
            "label": 0}
    
    return ret


# ============================================================================
# =================================== Main ===================================
def test_main(sample_size=50, seed=42, topk=10, max_generation_length=64):
    
    print("===MAIN===")
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
        
        result = test(prompt, topk, max_generation_length)
        label_score = get_reward(prompt, label, evaluate_reward_tokenizer, evaluate_reward_model)
        result["label"] = round(label_score, 4)
        
        all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)
    
    # output the results into csv file
    filename = f"./baseline_results.csv"
    
    with open(filename, 'wb') as f:
        f.truncate()
    all_results.to_csv(filename, index=False)
    
    with open("./generation_args.json", "w") as outfile:
        outfile.truncate()
        json.dump(ARGS, outfile, ensure_ascii=False)
    with open("./generation_bestn.json", "w") as outfile:
        outfile.truncate()
        json.dump(BESTN, outfile, ensure_ascii=False)
    with open("./generation_ns.json", "w") as outfile:
        outfile.truncate()
        json.dump(NS, outfile, ensure_ascii=False)
    with open("./generation_pargs_g.json", "w") as outfile:
        outfile.truncate()
        json.dump(PARGSG, outfile, ensure_ascii=False)
    with open("./generation_pargs.json", "w") as outfile:
        outfile.truncate()
        json.dump(PARGS, outfile, ensure_ascii=False)
    with open("./generation_DPO.json", "w") as outfile:
        json.dump(DPO, outfile, ensure_ascii=False)
    with open("./generation_PPO.json", "w") as outfile:
        json.dump(PPO, outfile, ensure_ascii=False)

if __name__ == "__main__":
    test_main(sample_size=100, seed=42, topk=10, max_generation_length=64)

