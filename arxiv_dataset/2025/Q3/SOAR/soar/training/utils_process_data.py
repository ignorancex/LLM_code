import numpy as np
import json
import tqdm as tqdm
import copy
from soar.llm_utils import merge_results
from soar.prompt import  get_repair_prompt, prompt_repair_cot_v1
from soar.prompt import get_solver_prompt,prompt_cot_fewshot_v1,prompt_cot_v1, prompt_wo_fewshot_v1_
import pickle

def select_example(data, n_sample,force_n_sample=False,n_sursample_max=1,seed=0):
    np.random.seed(seed)
    n_data=len(data)
    idx_data_sample = np.random.choice(len(data),n_data,replace=False)
    datapoints = [data[i] for i in idx_data_sample]
    if n_data>n_sample:
        datapoints = datapoints[:n_sample]
        return datapoints
    else :
        return datapoints    

def get_dataset_HER(path,train_val_data,n_sample=None,use_fewshot_example=False,use_cot=False,show_output_test=True, shuffle=True):
    """need to clean this code (merge step 1 and 2) 
        rm use_fewshot_example
    """
    print("process data step 1")
    
    if use_cot:
        prompt_solver = prompt_cot_v1
    else:
        prompt_solver=prompt_wo_fewshot_v1_
    if ".json" in path:
        with open(path, 'r') as f:   
            data = json.load(f)
    else:
        with open(path, 'rb') as f:   
            data = pickle.load(f)
    
    merge_pre_post=merge_results(data)
    list_k = list(merge_pre_post.keys())

    max_respone=n_sample
    if n_sample == None:
        max_sample = 0
        for _,v in merge_pre_post.items():
            max_sample = max(max_sample,len(v))
        n_sample = max_sample
    max_respone = n_sample

    all_data={list_k[i]:{"response":[],"hyperparameters":[]} for i in range(len(list_k))}
    # for dat in tqdm(data):
        # dict_response = data
    for key_id,task_responses in merge_pre_post.items():
        # all_data[key_id]
        task2solve = train_val_data[key_id]

        # shuffle list_id_response
        list_id_response = list(range(len(task_responses)))

        list_response = []
        for id_response in list_id_response:
            list_response.append(task_responses[id_response])
            if len(list_response)>=max_respone:
                break

        n_prog_2sample = min(max_respone,len(list_response))
        list_response = list(np.random.choice(list_response,n_prog_2sample,replace=False))

        list_all_response_to_keep = list_response 
        list_correct = [True for _ in range(len(list_response))]
        
        for id_response in range(len(list_all_response_to_keep)):
            if use_cot:
                clean_response = list_all_response_to_keep[id_response]["text"]
                #rm cot for repair exemple
                if "type" in list_all_response_to_keep[id_response]:
                    if 'refined' == list_all_response_to_keep[id_response]["type"]:
                        clean_response = "<reasoning>\nReasoning not found, but here is the correct implementation of `transform` function.\n</reasoning>\n"+"```python\n"+list_all_response_to_keep[id_response]["code"] + "\n```"
            else:
                clean_response = "```python\n"+list_all_response_to_keep[id_response]["code"] + "\n```"
            
            task2solve = copy.deepcopy(train_val_data[key_id])
            for id_io_pair in range(len(task2solve["train"])): # "test"c
                task2solve["train"][id_io_pair]["output"]= list_all_response_to_keep[id_response]["predicted_train_output"][id_io_pair]
            for id_io_pair in range(len(task2solve["test"])): # "test"
                task2solve["test"][id_io_pair]["output"]= list_all_response_to_keep[id_response]["predicted_test_output"][id_io_pair]
            if shuffle:
                task2solve["train"] = np.random.permutation(task2solve["train"]).tolist()
                task2solve["test"] = np.random.permutation(task2solve["test"]).tolist()

            try:
                instruction = get_solver_prompt(task2solve=task2solve,fewshot_examples=[], prompt_solver=prompt_solver,
                                grid_display_mode="numpy",
                                alt_colors = True, prompt_colors = True,randomize_pair_order=False,show_output_test=show_output_test)
                message={"conversations": [{"role": "system", "content": "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by reasoning and generating Python code."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": clean_response},
                    ]
                }
                all_data[key_id]["response"].append(message)
                all_data[key_id]["hyperparameters"].append({"correct":list_correct[id_response],"grid_display_mode":"numpy","alt_colors":True})
            except:
                print("error with datapoint")
    print("process data step 2")

    data = all_data
    dataset=[]
    list_len=[]
    for id,(key,val) in enumerate(data.items()):
        data2sample = val["response"]
        if len(data2sample)==0:
            continue
        list_correct=[val["hyperparameters"][i]["correct"] for i in range(len(val["hyperparameters"]))]
        data_accepted=[data2sample[i] for i in range(len(data2sample)) if list_correct[i]]
        list_len.append(len(data_accepted))
        if len(data_accepted)==0:
            continue 
        # normal part
        
        datapoints = select_example(data_accepted, n_sample)
        dataset.extend(copy.deepcopy(datapoints))

    print("len(dataset)",len(dataset))
    # remove all the data that are not in the dataset
    for id,(key,val) in enumerate(data.items()):
        for key2remove in list(val.keys()):
                try:
                    for key2remove2 in list(val[key2remove].keys()):
                            del val[key2remove][key2remove2]
                except:
                    pass
                del data[key][key2remove]
    list_key=list(data.keys())
    
    for k in list_key:
        del data[k]
    del data
    return dataset

def get_her_repair_sft(path,train_val_data,n_sample=None, shuffle=False):
    """process data for HER repair sft"""
    dataset=[]
    prompt_solve = prompt_repair_cot_v1
    if ".json" in path:
        with open(path, 'r') as f:   
            data = json.load(f)
    else:
        with open(path, 'rb') as f:   
            data = pickle.load(f)

    data_merged=merge_results(data)
    list_k = list(data_merged.keys())

    if n_sample == None:
        max_sample = 0
        for _,v in data_merged.items():
            max_sample = max(max_sample,len(v))
        n_sample = max_sample
    max_respone = n_sample

    all_data={list_k[i]:{"response":[],"hyperparameters":[]} for i in range(len(list_k))}
    for key_id,task_responses in data_merged.items():
        task2solve = train_val_data[key_id]

        # shuffle list_id_response
        list_id_response = list(range(len(task_responses)))

        list_response = []
        for id_response in list_id_response:
            list_response.append(task_responses[id_response])

        n_prog_2sample = min(max_respone,len(list_response))
        list_response = list(np.random.choice(list_response,min(n_prog_2sample,len(list_response)),replace=False))

        list_all_response_to_keep = list_response 
        list_correct = [True for _ in range(len(list_response))]
        
        for id_response in range(len(list_all_response_to_keep)):
            # prompt formatting
            data_repair = data_merged[key_id][id_response]
            clean_response = "```python\n"+list_all_response_to_keep[id_response]["code"] + "\n```"
            data_pre_repair = data_repair["parent_response"]
            task_k = copy.deepcopy(task2solve)
            for id_train in range(len(task_k["train"])):
                task_k["train"][id_train]["output"] = copy.deepcopy(data_repair["predicted_train_output"][id_train])
            for id_test in range(len(task_k["test"])):
                task_k["test"][id_test]["output"] = copy.deepcopy(data_repair["predicted_test_output"][id_test])
            if shuffle:
                train_indices = np.random.permutation(len(task_k["train"]))
                test_indices = np.random.permutation(len(task_k["test"]))

                # Apply the same shuffling to task_k
                task_k["train"] = [task_k["train"][i] for i in train_indices]
                task_k["test"] = [task_k["test"][i] for i in test_indices]
                data_pre_repair["predicted_train_output"] = [data_pre_repair["predicted_train_output"][i] for i in train_indices]
                data_pre_repair["predicted_test_output"] = [data_pre_repair["predicted_test_output"][i] for i in test_indices]
                data_pre_repair["correct_train_input"] = [data_pre_repair["correct_train_input"][i] for i in train_indices]
                data_pre_repair["correct_test_input"] = [data_pre_repair["correct_test_input"][i] for i in test_indices]
                
            instruction = get_repair_prompt(task2solve = task_k,
                                    list_previous_response = [data_pre_repair],
                                    prompt_solver = prompt_solve,
                                    grid_display_mode="numpy",
                                    alt_colors = True,
                                    prompt_colors = True,
                                    max_example = -1,show_output_test=False)

            message={"conversations": [{"role": "system", "content": "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by reasoning and generating Python code."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": clean_response},
                ]
            }
            all_data[key_id]["response"].append(message)
            all_data[key_id]["hyperparameters"].append({"correct":list_correct[id_response],"grid_display_mode":"numpy","alt_colors":True})
            dataset.append(message)
    return dataset

def apply_chat_template_orpo(example, tokenizer):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
        example["chosen"] = example["chosen"][-1]["content"] + tokenizer.eos_token
        example["rejected"] = example["rejected"][-1]["content"] + tokenizer.eos_token
    return example

def formatting_prompts_func(examples,tokenizer):
    convos = examples["conversations"]
    print("convos",convos[0])
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


def get_len_text(example,tokenizer):
    example["len_prompt"]=len(tokenizer.encode(example["prompt"]))
    example["len_response"]=len(max(tokenizer.encode(example["chosen"]),tokenizer.encode(example["rejected"])))
    return example

def get_len_text_sft(example,tokenizer,):
    example["len_prompt"]=len(tokenizer.encode(example["text"]))
    return example
