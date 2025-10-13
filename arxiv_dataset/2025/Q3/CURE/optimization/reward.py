import os
import ast
import json
import random
import argparse
import numpy as np


import optimization_config



# You can customize this reward.py to obtain your reward function. 
# The output of this module is a list, where each element is a dictionary with the keys 'prompt', 'response', and 'reward'.
# The reward here will be directly used as advantage, so you need to normalize them.








# read the configurations and load them as global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--max_generation_len", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--max_len_threshold", type=int, default=optimization_config.max_len_threshold)
    parser.add_argument("--min_len_threshold", type=int, default=optimization_config.min_len_threshold)
    parser.add_argument("--separate_training", type=str2bool, default=optimization_config.separate_training)
    parser.add_argument("--enable_efficient", type=str2bool, default=optimization_config.enable_efficient)
    parser.add_argument("--post_stage", type=str2bool, default=optimization_config.post_stage)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))



# read the inference data

outputs_name =  pretrained_model.replace("/", ".") + "-" + dataset

os.makedirs(os.path.dirname("./temp_data/outputs-rl-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-rl-" + outputs_name + ".json", 'r') as f:
    data = json.load(f)




# obatin the rollout samples and the corresponding reward/advantages

def normalize_reward(reward_arr):
    if np.all(reward_arr == 1) and enable_efficient:
        return reward_arr
    mean = np.mean(reward_arr)
    std = np.std(reward_arr)
    if std.item() == 0:
        return None
    return (reward_arr - mean) / std

def normalize_balance_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    sum_pos = x[pos_mask].sum()
    sum_neg_abs = abs(x[neg_mask].sum())
    if sum_pos * sum_neg_abs == 0:
        return None
    scale_factor = sum_neg_abs / sum_pos
    x[pos_mask] *= scale_factor
    return x / x.std()

def length_regularize(reward_arr, response_length_list):
    reward_arr = np.sign(reward_arr)
    pos_list = np.where(reward_arr == 1)[0].tolist()
    neg_list = np.where(reward_arr == -1)[0].tolist()
    pos_response_length = np.array([response_length_list[j] for j in pos_list])
    threshold = np.median(pos_response_length).item()
    if np.sum((pos_response_length - threshold)**2) == 0: # no variance
        return normalize_balance_std(np.sign(reward_arr))
    threshold = max(min(threshold, max_len_threshold), min_len_threshold)
    length_reg_reward = np.zeros(len(reward_arr), float)
    length_reg_reward[pos_list] = - pos_response_length + threshold
    length_reg_reward[neg_list] = np.min(length_reg_reward).copy()
    length_reg_reward = normalize_balance_std(length_reg_reward)
    return length_reg_reward

code_data = []
case_data = []
index_list = []

for i in range(len(data)):
    if data[i]["all_case_bool_table"] is None:
        continue
    
    t = data[i]["num_ground_truth_test"]
    all_test_table_i = np.array(data[i]["all_case_bool_table"])[:, :t].copy()
    all_case_table_i = np.array(data[i]["all_case_bool_table"])[:, t:].copy()

    # reward for code
    code_reward = np.mean(all_test_table_i, 1)
    #code_reward = all_test_table_i.all(axis=1).astype(float)
    code_reward = normalize_reward(code_reward)
    if code_reward is not None:
        if enable_efficient:
            code_reward = length_regularize(code_reward, data[i]["code_response_length"])
        if code_reward is not None:
            code_reward = code_reward.tolist()
            for j in range(len(code_reward)):
                code_data_i = {}
                code_data_i["prompt"] = data[i]["code_generation_prompt"]
                if data[i]["code_response_length"][j] < max_generation_len:
                    code_data_i["response"] = data[i]["full_code_generation"][j] + "<|im_end|>"
                else:
                    code_data_i["response"] = data[i]["full_code_generation"][j]
                code_data_i["reward"] = code_reward[j]
                code_data.append(code_data_i)
    
    # reward for case
    correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
    if len(correct_code_list) > 0:
        # get reward sign
        correct_code_table = all_case_table_i[correct_code_list, :].copy()
        index_list = np.where(np.all(correct_code_table, axis=0))[0].tolist()
        reward_sign = -np.ones(correct_code_table.shape[1], dtype=float)
        reward_sign[index_list] = 1
        case_reward = reward_sign.copy()
        # get reward scale
        wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
        if len(wrong_code_list) > 0:
            reward_scale = np.ones(correct_code_table.shape[1], dtype=float)
            correct_case_list = np.where(correct_code_table.all(axis=0))[0].tolist()
            wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
            if len(correct_case_list):
                wrong_code_correct_case_table = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
                if post_stage == False:
                    mean_p01 = np.mean(~wrong_code_correct_case_table, 0)
                else:
                    mean_p01 = (~np.any(wrong_code_correct_case_table, axis=0)).astype(float)
                reward_scale[correct_case_list] = reward_scale[correct_case_list] * mean_p01
            if len(wrong_case_list):
                wrong_code_wrong_case_table = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
                if post_stage == False:
                    mean_p00 = np.mean(wrong_code_wrong_case_table, 0)
                else:
                    mean_p00 = (np.any(wrong_code_wrong_case_table, axis=0)).astype(float)
                reward_scale[wrong_case_list] = reward_scale[wrong_case_list] * mean_p00
            case_reward = case_reward * reward_scale
        
        case_reward = normalize_reward(case_reward)
        if case_reward is not None:
            if enable_efficient:
                case_reward = length_regularize(case_reward, data[i]["case_response_length"])
            if case_reward is not None:
                case_reward = case_reward.tolist()
                for j in range(len(case_reward)):
                    case_data_i = {}
                    case_data_i["prompt"] = data[i]["case_generation_prompt"]
                    if data[i]["case_response_length"][j] < max_generation_len:
                        case_data_i["response"] = data[i]["full_case_generation"][j] + "<|im_end|>"
                    else:
                        case_data_i["response"] = data[i]["full_case_generation"][j]
                    case_data_i["reward"] = case_reward[j]
                    case_data.append(case_data_i)




final_data = code_data + case_data
random.shuffle(final_data)

if separate_training == False:
    with open("./temp_data/rl_data.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
else:
    with open("./temp_data/rl_code_data.json", "w", encoding="utf-8") as f:
        json.dump(code_data, f, indent=2, ensure_ascii=False)
    with open("./temp_data/rl_case_data.json", "w", encoding="utf-8") as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)


