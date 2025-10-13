import re
import ast
from openai import OpenAI
import copy
import pickle
import json
import random
from time import sleep
import numpy as np
import os

def get_API_client_cfg(model,temperature=0.0):
    """
    model= ["gpt-4o-mini","deepseek-chat"]
    """
    cfg_generation={"temperature":temperature,"model": model}
    from key import deepseek_key,openai_key

    if "deepseek" in model:
        client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

    if "gpt" in model.lower():
        client = OpenAI(api_key=openai_key)

    return client,cfg_generation

def merge_results(results):
    if isinstance(results, dict):
        return results
    all_results = {}
    for i in results:
        list_k = list(i['dict_response'].keys())
        for task_id in list_k:
            if not task_id in all_results:
                all_results[task_id] = []
            for j in i['dict_response'][task_id]:
                all_results[task_id].append(j)
                
    return all_results

def to_formated_list(results):
    if isinstance(results, list):
        return results
    return [{"dict_response": results}]

def merged_dic_results(res1,res2):
    """
    merge two dict of results
    """
    for k,v in res2.items():
        if k in res1:
            res1[k]+=v
        else:
            res1[k]=v
    return res1

def get_dic_only_correct(dic_results):
    """
    Extracts only the correct responses from the given dictionary of results.
    
    Args:
        dic_results (dict): Dictionary containing results with keys as task IDs and values as lists of responses.
        
    Returns:
        dict: Dictionary with task IDs as keys and lists of correct responses as values.
    """
    dic_correct = {}
    for k, v in dic_results.items():
        for resp in v:
            if all(resp["correct_test_input"] + resp["correct_train_input"]):
                if k not in dic_correct:
                    dic_correct[k] = []
                dic_correct[k].append(resp)
    return dic_correct

def return_chat_format(message: str, system_prompt=None):
    """
    take a message and return the format for the chat
    """
    if system_prompt is None:
        system_prompt = "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by reasoning and generating Python code."
    return [{"role": "system", "content": system_prompt},
    {"role": "user", "content": message}]


def prompt2token_chat(tokenizer,prompts: list[str],tokenize=True, out_chat_format = False):
    """
    take a prompt and return the tokenized version in chat format
    """
    messages_list=[]
    for prompt in prompts:
        messages_list.append(return_chat_format(prompt))
    if out_chat_format:
        return messages_list
    return [tokenizer.apply_chat_template(messages, tokenize=tokenize,add_generation_prompt=True) for messages in messages_list]

def prompt2_chat(prompts: list[str]):
    """
    take a prompt and return the tokenized version in chat format
    """
    messages_list=[]
    for prompt in prompts:
        messages_list.append(return_chat_format(prompt))
    return messages_list


# =================  generation postprocessing  =================

def keep_functions_and_imports(code):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Filter for functions and imports
        filtered_nodes = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))
        ]
        
        # Create a new module with the filtered nodes
        new_module = ast.Module(body=filtered_nodes, type_ignores=[])
        
        # Convert the new AST back to source code
        code=ast.unparse(new_module)
        if code != None:
            code=code.strip()
        return code
    except Exception as e:
        # print("Error:",e)
        return code
    

def extract_transform_and_imports(code):
    """check if the code contains a transform function and return it with the imports"""
    tree = ast.parse(code)
    
    imports = []
    transform_func = None
    
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef) and node.name == 'transform':
            transform_func = ast.unparse(node)
    
    return '\n'.join(imports+[""] + [transform_func])

def process_response(text):
    if text == None:
        return ""
    clean_code = extract_transform(text)
    text = text.split("```python\n")[0]
    if text != None:
        text = text.strip()
    clean_response=""
    if not "<reasoning>" in text:
        clean_response+="<reasoning>\n"
    clean_response+=text
    if not "</reasoning>" in text:
        clean_response+="\n</reasoning>\n"

    clean_response += "\n```python\n" + clean_code + "\n```"
    return clean_response

def extract_transform(markdown_text):
    """
    Extract the transform function from the LLM generated text.
    """
    pattern = r'```python\n(.*?)```'
    
    # Find all Python code blocks
    code_blocks = re.findall(pattern, markdown_text, re.DOTALL)
    transform_function = None
    # Search for the transform function in the code blocks
    flag_def = False
    for block in code_blocks:
        if 'def transform(' in block:
            # Extract the transform function including imports
            transform_function = block
            flag_def = True
            # try:
            #     transform_function = extract_transform_and_imports(block)
            # except:
            #     pass
        if not flag_def:
            transform_function = block

            
    # If no transform function is found, return None
    if transform_function != None:
        transform_function = postprocess_transform(transform_function)
    else:
        transform_function = ""
    return transform_function


def postprocess_transform(transform):
    """
    Postprocess the transform function.
    """
    # Add import numpy as np if not present and np is used
    transform = keep_functions_and_imports(transform)
    if 'np.' in transform and 'import numpy as np' not in transform:
        transform = 'import numpy as np\n' + transform

    transform = check_and_remove_test_transform(transform)
    return transform

def check_and_remove_test_transform(code_list):
    """Check if the code contains a test_transform function and remove it if it exists."""
    if not isinstance(code_list, list):
        code_list = [code_list]
        flag_not_list= True
    else:
        flag_not_list = False
    result = []
    
    for code in code_list:
        # First check if test_transform exists
        has_test_transform = False
        try:
            has_test_transform = "def test_transform" in code or "def check_transform" in code
                    
        except SyntaxError:
            # If code can't be parsed, add it as is
            result.append(code)
            continue
            
        # If test_transform exists, remove it
        if has_test_transform:
            try:
                tree = ast.parse(code)
                # Filter out the test_transform function
                new_body = [node for node in tree.body if not (
                    isinstance(node, ast.FunctionDef) and (node.name == 'test_transform' or node.name == 'check_transform'))]
                tree.body = new_body
                # Convert modified AST back to source code
                result.append('\n'.join(ast.unparse(tree).splitlines()))
            except SyntaxError:
                # If there's an error during removal, add original code
                result.append(code)
        else:
            # If no test_transform function, add code as is
            result.append(code)
    if flag_not_list:
        return result[0]
    else:
        return result
# =================  generation postprocessing (check how many puzzle are good ... )  =================

# process api
# check if out is a list of list of dict
# out = [choice.message.content for choice in completion.choices]

def processe_vllm_generate(completions):
    completions_all_task=[]
    for i in range(len(completions)):
        completions_task=[]
        for j in range(len(completions[i].outputs)):
            completions_task.append(completions[i].outputs[j].text)
        completions_all_task.append(completions_task)
    return completions_all_task

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def save_pkl_secure(path_pkl, results):
    """ Save pickle files with file existence check """
    n_max_try = 3
    n_try = 0
    
    # Pickle file handling
    n_try = 0
    if not ".pkl" in path_pkl:
        path_pickle = path_pkl.replace('.json', '.pkl')
    else:
        path_pickle = path_pkl        
    while n_try < n_max_try:
        n_try += 1
        sleeptime = random.uniform(1, 10)
        print("Attempt:", n_try, "- Sleeping for:", sleeptime, "seconds")
        sleep(sleeptime)

        try:
            if not os.path.exists(path_pickle):
                with open(path_pickle, 'wb') as outfile:
                    pickle.dump([], outfile)

            with open(path_pickle, 'rb+') as outfile:
                pickle_content = pickle.load(outfile)
                pickle_content.append(results)
                outfile.seek(0)
                pickle.dump(pickle_content, outfile)
                outfile.truncate()
            break
        except Exception as e:
            print("Error in Pickle Process:", e)

def save_pkl_secure_force(path_pkl, results):
    """
    Attempts to securely save the given results to a pickle file.
    """

    n_max_try = 3
    n_try = 0
    if not ".pkl" in path_pkl:
        path_pickle = path_pkl.replace('.json', '.pkl')
    else:
        path_pickle = path_pkl
    
    while n_try < n_max_try:
        n_try += 1
        sleeptime = random.uniform(1, 10)
        print("Attempt:", n_try, "- Sleeping for:", sleeptime, "seconds")
        sleep(sleeptime)

        try:

            with open(path_pickle, 'wb') as outfile:
                pickle_content = []
                pickle_content.append(results)
                pickle.dump(pickle_content, outfile)
            break
        except Exception as e:
            print("Error in Pickle Process:", e)

            
def format_all_generation(out,list_task_id,use_vllm_generate=True,keep_full_tex=False,use_cot=False):
    """
    format solution:
    out: list[list[str]], where str -> is raw output of the model, if use_vllm_generate=True, out = llm.generate(...)
    use_vllm_generate: bool, change `out` type format
    extract_response_from_api: bool, True to handle openai api response (RequestOutput object) 
    """
    if use_vllm_generate:
        out = processe_vllm_generate(out)
    outputs_copy = out#copy.deepcopy(out)
    # isinstance(out,List)
    if not isinstance(outputs_copy[0],list): # if just one solution
        outputs_copy = [[outputs_copy[i]] for i in range(len(outputs_copy))]
    count_none=0
    dict_response = {}
    for i in range(len(outputs_copy)):
        res_task_i=[]
        for j in range(len(outputs_copy[i])):
            if outputs_copy[i][j] == None:
                count_none+=1
                output_dict = {"text":""}
                output_dict["code"] = ""
            else:
                if use_cot:
                    output_dict = {"text":process_response(outputs_copy[i][j])}
                else:
                    output_dict = {"text":outputs_copy[i][j]}
                if keep_full_tex:
                    output_dict["full_text"] = outputs_copy[i][j]   
                output_dict["code"] = extract_transform(output_dict["text"])
            output_dict["code"] = check_and_remove_test_transform(output_dict["code"])
            res_task_i.append(output_dict)
        dict_response[list_task_id[i]] = res_task_i
    print("Number of None:",count_none)
    return dict_response
    
def format_repair(out,data,list_coord,use_vllm_generate=True):
    """
    format solution:
    out: list[list[str]], where str -> is raw output of the model, if use_vllm_generate=True, out = llm.generate(...)
    use_vllm_generate: bool, change `out` type format
    extract_response_from_api: bool, True to handle openai api response (RequestOutput object) 
    """
    if use_vllm_generate:
        out = processe_vllm_generate(out)
    outputs_copy = copy.deepcopy(out)
    # isinstance(out,List)
    if not isinstance(outputs_copy[0],list): # if just one solution
        outputs_copy = [[outputs_copy[i]] for i in range(len(outputs_copy))]

    for i in range(len(outputs_copy)):
        (key_task,idx_response) = list_coord[i]
        res_task_i=[]
        for j in range(len(outputs_copy[i])):
            
            output_dict = {"text":process_response(outputs_copy[i][j])}
            output_dict["code"] = extract_transform(output_dict["text"])
            res_task_i.append(output_dict)
        data[key_task][idx_response]["repair"] = res_task_i
    
    return data
    


def get_dic_task_solved(dic_res,task_id=None,mode="train"):
    """dic_res {task_id: [res1,res2]} 
    return {task_id: number of correct response}
    """
    if task_id is None:
        task_id = dic_res.keys()
    dic_task_solved = {k:0 for k in task_id}
    for k in task_id:
        count_correct_res=0
        if k in dic_res:
            for res in dic_res[k]:
                if mode=="train":
                    cond = all(res["correct_train_input"]) and all(res["correct_test_input"])
                else:
                    cond = all(res["correct_train_input"])
                if cond:
                    count_correct_res+=1
        dic_task_solved[k] = count_correct_res
    return dic_task_solved



def get_number_of_solved_tasks(dict_response, n_best_codes=2):
    """
    Calculates the number of solved tasks based on different criteria.

    test_task_solved_real: Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs
    and checking if they solve test examples.

    test_task_solved_oracle: Number of tasks solved, only checking test input-output pairs.

    train_task_solved_oracle: Number of tasks solved, only checking train input-output pairs.

    test_task_solved_with_all_train_correct: Number of tasks solved with all train input-output pairs correct and all test output correct.
    """
    test_task_solved_real = 0
    test_task_solved_oracle = 0
    train_task_solved_oracle = 0
    test_task_solved_with_all_train_correct = 0
    n_responses = len(list(dict_response.values())[0])
    for task_responses in dict_response.values():
        sorted_responses = sorted(task_responses, key=lambda x: sum(x["correct_train_input"]), reverse=True)
        best_responses = sorted_responses[:n_best_codes]

        if any(all(response["correct_test_input"]) for response in best_responses):
            test_task_solved_real += 1

        if any(all(response["correct_test_input"]) for response in task_responses):
            test_task_solved_oracle += 1

        if any(all(response["correct_train_input"]) for response in task_responses):
            train_task_solved_oracle += 1

        if any(all(response["correct_train_input"]) and all(response["correct_test_input"]) for response in task_responses):
            test_task_solved_with_all_train_correct += 1

    return {
        "n_tasks": len(dict_response),
        "n_responses":n_responses,
        "test_task_solved_real": test_task_solved_real, # Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs and checking if they solve test examples.
        "test_task_solved_oracle": test_task_solved_oracle, # Number of tasks solved, only checking test input-output pairs.
        "train_task_solved_oracle": train_task_solved_oracle, # Number of tasks solved, only checking train input-output pairs.
        "test_task_solved_with_all_train_correct": test_task_solved_with_all_train_correct # Number of tasks solved with all train input-output pairs correct and all test output correct.
    }

def get_number_of_N_solved_tasks(dict_response, n_best_codes=2,N=-1):
    """
    Calculates the number of solved tasks based on different criteria.

    test_task_solved_real: Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs
    and checking if they solve test examples.

    test_task_solved_oracle: Number of tasks solved, only checking test input-output pairs.

    train_task_solved_oracle: Number of tasks solved, only checking train input-output pairs.

    test_task_solved_with_all_train_correct: Number of tasks solved with all train input-output pairs correct and all test output correct.
    """
    test_task_solved_real = 0
    test_task_solved_oracle = 0
    train_task_solved_oracle = 0
    test_task_solved_with_all_train_correct = 0
    n_responses = len(list(dict_response.values())[0])
    for task_responses in dict_response.values():
        if N != -1:
            task_responses_N = task_responses[:N]
        else:
            task_responses_N = task_responses
        sorted_responses = sorted(task_responses_N, key=lambda x: sum(x["correct_train_input"]), reverse=True)

        best_responses = sorted_responses[:n_best_codes]

        if any(all(response["correct_test_input"]) for response in best_responses):
            test_task_solved_real += 1
        if any(all(response["correct_test_input"]) for response in task_responses_N):
            test_task_solved_oracle += 1

        if any(all(response["correct_train_input"]) for response in task_responses_N):
            train_task_solved_oracle += 1

        if any(all(response["correct_train_input"]) and all(response["correct_test_input"]) for response in task_responses_N):
            test_task_solved_with_all_train_correct += 1


    return {
        "n_tasks": len(dict_response),
        "n_responses":n_responses,
        "test_task_solved_real": test_task_solved_real, # Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs and checking if they solve test examples.
        "test_task_solved_oracle": test_task_solved_oracle, # Number of tasks solved, only checking test input-output pairs.
        "train_task_solved_oracle": train_task_solved_oracle, # Number of tasks solved, only checking train input-output pairs.
        "test_task_solved_with_all_train_correct": test_task_solved_with_all_train_correct # Number of tasks solved with all train input-output pairs correct and all test output correct.
    }


def get_info_solved_tasks(dict_response, n_best_codes=2):
    """
    Calculates the number of solved tasks based on different criteria.

    test_task_solved_real: Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs
    and checking if they solve test examples.

    test_task_solved_oracle: Number of tasks solved, only checking test input-output pairs.

    train_task_solved_oracle: Number of tasks solved, only checking train input-output pairs.

    test_task_solved_with_all_train_correct: Number of tasks solved with all train input-output pairs correct and all test output correct.
    """
    test_task_solved_real = 0
    test_task_solved_oracle = 0
    train_task_solved_oracle = 0
    test_task_solved_with_all_train_correct = 0
    n_responses = len(list(dict_response.values())[0])
    dic_res={}
    for key_id,task_responses in dict_response.items():
        sorted_responses = sorted(task_responses, key=lambda x: sum(x["correct_train_input"]), reverse=True)
        best_responses = sorted_responses[:n_best_codes]

        test_task_solved_real = sum(all(response["correct_test_input"]) for response in best_responses)

        test_task_solved_oracle = sum(all(response["correct_test_input"]) for response in task_responses)

        train_task_solved_oracle = sum(all(response["correct_train_input"]) for response in task_responses)

        test_task_solved_with_all_train_correct = sum(all(response["correct_train_input"]) and all(response["correct_test_input"]) for response in task_responses)
        
        dic_res[key_id] = {
        "n_tasks": len(dict_response),
        "n_responses":n_responses,
        "test_task_solved_real": test_task_solved_real, # Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs and checking if they solve test examples.
        "test_task_solved_oracle": test_task_solved_oracle, # Number of tasks solved, only checking test input-output pairs.
        "train_task_solved_oracle": train_task_solved_oracle, # Number of tasks solved, only checking train input-output pairs.
        "test_task_solved_with_all_train_correct": test_task_solved_with_all_train_correct # Number of tasks solved with all train input-output pairs correct and all test output correct.
        }

    return dic_res


import numpy as np
from typing import Dict, List, Any

def check_resp_ok(resp: Dict[str, List[Any]],task=None) -> bool:
    try:
        # correct_train = np.sum(resp["correct_train_input"])>=1
        # Combine both lists to avoid duplicate code
        all_grids = resp["predicted_test_output"] + resp["predicted_train_output"]

        # Use list comprehension instead of loops
        if not all(check_one_out(grid) for grid in all_grids):
            return False
        
        if task is not None:
            # check if test output is not just background

            has_only_background_output= np.sum([not(np.any(task["train"][i]["output"])) for i in range(len(task["train"]))])>=1
            list_pred_test= [not(np.any(resp["predicted_test_output"][i])) for i in range(len(resp["predicted_test_output"]))]
            if np.all(list_pred_test) and not has_only_background_output:
                # print("test output is just background")
                return False

            list_train_input=[task["train"][i]["input"] for i in range(len(task["train"]))]
            list_train_output=[task["train"][i]["output"] for i in range(len(task["train"]))]
            
            # remove task if all predicted train output are similar
            mean_dist_predicted_output_example = np.mean(compute_inter_grid_metrics(resp["predicted_train_output"], grid_distance))
            mean_dist_train_output_example = np.mean(compute_inter_grid_metrics(list_train_output, grid_distance))
            if mean_dist_predicted_output_example < min(0.2,mean_dist_train_output_example):
                return False
            
            # check if output is not the same as input
            list_check_id = [(resp["predicted_train_output"][i]==list_train_input[i]) for i in range(len(resp["predicted_train_output"]))]
            list_check_id_given_io = [(list_train_output[i]==list_train_input[i]) for i in range(len(list_train_output))]

            # if 30% grids are the same as input rm
            if np.mean(list_check_id)>max(0.3,np.mean(list_check_id_given_io)):
                return False
        return True
        
    except Exception:
        return False

def check_one_out(out: List[List[int]], debug: bool = False) -> bool:
    """Check if puzzle is valid"""
    try:
        # Quick validation checks
        if not out or not out[0]:  # Check for empty grid
            return False
            
        x_size, y_size = len(out), len(out[0])
        if x_size > 40 or y_size > 40:  # Size validation
            return False
            
        # Convert to numpy array once and use it for type checking
        grid = np.array(out)
        if grid.dtype not in (np.int32, np.int64):  # Check if all elements are integers
                # check if all values are in [0,1,2,3,4,5,6,7,8,9]
            grid = grid.astype(int)
        if not np.all((0 <= grid) & (grid <= 9)):
            return False
            
        # No need to store shape variables if not used
        # _ = grid.shape
        x_shape,y_shape = grid.shape

        # Hash check (if still needed)
        hash(tuple(map(tuple, out)))
        
        return True
        
    except Exception:  # Catch any conversion errors
        return False

import numpy as np

def grid_distance(grid1, grid2, background=True):
    """
    Compute distance between two grids using Hamming distance on padded grids
    
    Args:
        grid1: First grid (numpy array)
        grid2: Second grid (numpy array)
    Returns:
        float: Normalized distance between 0 and 1
    """
    if isinstance(grid1, list):
        grid1 = np.array(grid1)
    if isinstance(grid2, list):
        grid2 = np.array(grid2)
    def pad_grids(grid1, grid2):
        max_rows = max(grid1.shape[0], grid2.shape[0])
        max_cols = max(grid1.shape[1], grid2.shape[1])
        padded_grid1 = np.pad(grid1, ((0, max_rows - grid1.shape[0]), 
                                     (0, max_cols - grid1.shape[1])), 'constant',constant_values=-1)
        padded_grid2 = np.pad(grid2, ((0, max_rows - grid2.shape[0]), 
                                     (0, max_cols - grid2.shape[1])), 'constant',constant_values=-1)
        return padded_grid1, padded_grid2
    
    # Pad both grids to same dimensions
    padded_grid1, padded_grid2 = pad_grids(grid1, grid2)
    if background:
        # Create mask for non-background positions
        mask = (padded_grid1 != 0) | (padded_grid2 != 0)
        
        # Calculate Hamming distance ignoring background pixels
        hamming_dist = np.sum((padded_grid1[mask] != padded_grid2[mask]))
        # Count total non-background positions for normalization
        total_non_background = np.sum(mask)
        # Avoid division by zero
        normalized_dist = hamming_dist / total_non_background if total_non_background > 0 else 0.0

    else:
        # Calculate Hamming distance (number of positions where grids differ)
        hamming_dist = np.sum(padded_grid1 != padded_grid2)
        
        # Normalize by total size of padded grids
        normalized_dist = hamming_dist / padded_grid1.size
    return normalized_dist

def compute_inter_grid_metrics(list_grid,metric):
    """
    Compute distances between all pairs of grids in the list
    
    Args:
        list_grid: List of numpy arrays (grids)
    Returns:
        list of tuples: (grid1_index, grid2_index, distance)
    """


    num_grids = len(list_grid)
    inter_grid_distances = []

    for i in range(num_grids):
        for j in range(i + 1, num_grids):
            distance = metric(list_grid[i], list_grid[j])
            inter_grid_distances.append(distance)

    return inter_grid_distances

def give_n_majority_vote_v2(task_responses, n_output=2, c=1000,return_responses=False):
    """ compared to v1, given all test output combined whereas v1 was majority voting for each test output independently"""

    len_test = len(task_responses[0]["predicted_test_output"])
    
    # Initialize the response structure
    list_response = [{
        "predicted_test_output": [[] for _ in range(len_test)],
        "correct_test_input": [False for _ in range(len_test)],
        "majority_vote": 0
    } for _ in range(n_output)]
    
    # Collect all valid responses across all test cases
    all_valid_responses = []
    
    for response_idx, response in enumerate(task_responses):
        valid_response = {
            "outputs": [], 
            "corrections": [], 
            "train_accuracy": np.mean(response["correct_train_input"]),
            "original_idx": response_idx  # Store the original index
        }
        
        # Process all test outputs for this response
        valid = True
        for id_test in range(len_test):
            output = response["predicted_test_output"][id_test]
            try:
                output = np.array(output).tolist()
                # Store the output and its correctness
                valid_response["outputs"].append(serialize_list(output))
                valid_response["corrections"].append(response["correct_test_input"][id_test])
            except:
                # If any test case fails, consider the entire response invalid
                valid = False
                break
                
        if valid:
            all_valid_responses.append(valid_response)
    
    # Count occurrences of complete response patterns
    response_counts = {}
    response_weights = {}
    response_originals = {}
    pattern_to_original_idxs = {}  # Now storing lists of indices
    
    for valid_response in all_valid_responses:
        # Create a tuple of all outputs to use as a key
        response_key = tuple(valid_response["outputs"])
        
        if response_key in response_counts:
            response_counts[response_key] += 1
            response_weights[response_key] += 1 + c * valid_response["train_accuracy"]
            # Add this response's index to the list for this pattern
            pattern_to_original_idxs[response_key].append(valid_response["original_idx"])
        else:
            response_counts[response_key] = 1
            response_weights[response_key] = 1 + c * valid_response["train_accuracy"]
            # Store the original outputs and corrections
            response_originals[response_key] = {
                "outputs": [deserialize_list(out) for out in valid_response["outputs"]],
                "corrections": valid_response["corrections"]
            }
            # Initialize the list of indices for this pattern
            pattern_to_original_idxs[response_key] = [valid_response["original_idx"]]
    
    # Get top n_output patterns based on weights
    top_patterns = sorted(response_weights.items(), key=lambda x: x[1], reverse=True)[:n_output]
    
    # Fill the response structure with the top patterns
    for i, (pattern, weight) in enumerate(top_patterns):
        if i < n_output:
            original_data = response_originals[pattern]
            
            for id_test in range(len_test):
                list_response[i]["predicted_test_output"][id_test] = original_data["outputs"][id_test]
                list_response[i]["correct_test_input"][id_test] = original_data["corrections"][id_test]
            list_response[i]["majority_vote"] = response_counts[pattern]
            list_response[i]["weight"] = weight
    
    # Get all original responses for each of the top patterns
    top_original_responses = []
    for pattern, weight in top_patterns:
        if pattern in pattern_to_original_idxs:
            # Get all responses that match this pattern
            pattern_responses = [task_responses[idx] for idx in pattern_to_original_idxs[pattern]]
            top_original_responses.append({"weight":weight,"response":pattern_responses})
    if return_responses:
        return list_response, top_original_responses
    else:
        return list_response
    

# Helper functions
def serialize_list(lst):
    """Convert a nested list to a tuple for hashing"""
    if isinstance(lst, list):
        return tuple(serialize_list(item) for item in lst)
    return lst

def deserialize_list(tpl):
    """Convert a nested tuple back to a list"""
    if isinstance(tpl, tuple):
        return [deserialize_list(item) for item in tpl]
    return tpl

def weighted_sample(list_original_resp, N):
    """
    Sample N items from multiple categories based on category weights and item quality.
    
    Args:
        list_original_resp: List of dictionaries, each with 'weight' and 'response' keys.
                           Format: [{"weight": w1, "response": [{"correct_train_input": [True,False,...], ...}, ...]}, ...]
                           Each response is a dictionary with at least a 'quality' key.
        N: Number of samples to return
        
    Returns:
        List of sampled responses
    """
    def compute_quality(response):
        return np.mean(np.array(response["correct_train_input"])*1000)
    # Extract weights and normalize them
    weights = [item["weight"] for item in list_original_resp]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate how many samples to take from each unique output 
    category_counts = np.random.multinomial(N, normalized_weights)
    
    sampled_responses = []
    
    # Sample from each category according to its allocated count
    for i, count in enumerate(category_counts):
        category_responses = list_original_resp[i]["response"]
        
        # Skip empty categories or when count is 0
        if count == 0 or not category_responses:
            continue
            
        # Extract quality scores for this category's responses
        quality_scores = [compute_quality(resp)+ 1e-10 for resp in category_responses]
        
        # Handle case where all quality scores are 0
        if sum(quality_scores) == 0:
            # If all qualities are 0, use uniform sampling
            quality_weights = None
        else:
            # Normalize quality scores to use as weights
            quality_weights = [q / sum(quality_scores) for q in quality_scores]
        
        # If we need more samples than available, sample with replacement
        if count > len(category_responses):
            indices = random.choices(range(len(category_responses)), weights=quality_weights, k=count)
            samples = [category_responses[idx] for idx in indices]
        # Otherwise sample without replacement, but respecting quality weights
        else:
            # For weighted sampling without replacement
            # First check if we have enough non-zero weights
            if quality_weights:
                non_zero_weights = sum(1 for w in quality_weights if w > 0)
                if non_zero_weights < count:
                    # Not enough non-zero weights, fall back to sampling with replacement
                    indices = random.choices(range(len(category_responses)), weights=quality_weights, k=count)
                else:
                    # We have enough non-zero weights, proceed with sampling without replacement
                    indices = np.random.choice(
                        range(len(category_responses)), 
                        size=count, 
                        replace=False, 
                        p=quality_weights
                    )
            else:
                # No weights provided, use uniform sampling without replacement
                indices = np.random.choice(range(len(category_responses)), size=count, replace=False)
            
            samples = [category_responses[idx] for idx in indices]
            
        sampled_responses.extend(samples)
    
    return sampled_responses


def get_number_of_solved_tasks_bis(dict_response, n_best_codes=2,c=1000,mv=True,N=-1):
    """
    Calculates the number of solved tasks based on different criteria.

    test_task_solved_real: Number of tasks solved by taking at most n_best_codes with the most correct train input-outputs
    and checking if they solve test examples.

    test_task_solved_oracle: Number of tasks solved, only checking test input-output pairs.

    train_task_solved_oracle: Number of tasks solved, only checking train input-output pairs.

    test_task_solved_with_all_train_correct: Number of tasks solved with all train input-output pairs correct and all test output correct.
    """
    test_task_solved_real_kaggle_mv = 0
    test_task_solved_oracle = 0
    train_task_solved_oracle = 0
    test_task_solved_oracle_kaggle=0
    test_task_solved_with_all_train_correct = 0
    n_responses = len(list(dict_response.values())[0])
    flage_mv=True
    for k,task_responses in dict_response.items():
        n_responses = max(n_responses,len(task_responses))
        if N != -1: 
            task_responses = task_responses[:N]

        try:
            if mv:
                best_responses_mv=give_n_majority_vote_v2(task_responses, n_output=n_best_codes,c=c)
                test_task_solved_real_kaggle_mv += define_correct_kaggle(best_responses_mv)[0]
        except Exception as e:
            if flage_mv:
                print("error mv", e)
            flage_mv=False
   
        if any(all(response["correct_test_input"]) for response in task_responses):
            test_task_solved_oracle += 1

        test_task_solved_oracle_kaggle += define_correct_kaggle(task_responses)[0]

        if any(all(response["correct_train_input"]) for response in task_responses):
            train_task_solved_oracle += 1

        if any(all(response["correct_train_input"]) and all(response["correct_test_input"]) for response in task_responses):
            test_task_solved_with_all_train_correct += 1

    dic_all_res={
        "n_tasks": len(dict_response),
        "n_responses":n_responses,
        "test_task_solved_real_kaggle_mv": test_task_solved_real_kaggle_mv,
        "test_task_solved_oracle": test_task_solved_oracle, # Number of tasks solved, only checking test input-output pairs.
        "train_task_solved_oracle": train_task_solved_oracle, # Number of tasks solved, only checking train input-output pairs.
        "test_task_solved_oracle_kaggle": test_task_solved_oracle_kaggle,
        "test_task_solved_with_all_train_correct": test_task_solved_with_all_train_correct # Number of tasks solved with all train input-output pairs correct and all test output correct.
    }
    return dic_all_res


def get_info(dic_res,list_key_print=None):
    for k,v in dic_res.items():
        if list_key_print is None or k in list_key_print:
            print(k," :",v)

def define_correct_kaggle(task_responses):
    if len(task_responses)==0:
        return 0,0  
        
    len_test = len(task_responses[0]["correct_test_input"])
    correctness=[]
    for id_test in range(len_test):
        correctness.append(any(response["correct_test_input"][id_test] for response in task_responses))
    return np.mean(correctness),len_test


def give_n_majority_vote(task_responses, n_output=2,c=1000):
    len_test = len(task_responses[0]["predicted_test_output"])
    list_reponse=[{
            "predicted_test_output": [[] for _ in range(len_test)],
            "correct_test_input": [False for _ in range(len_test)],
            "majority_vote": [0 for _ in range(len_test)]
        } for _ in range(n_output)]
    for id_test in range(len_test):
        valid_outputs = []
        val_out=[]
        valid_corrections = [] 
        valid_train =[]
        
        # Collect valid outputs and their correctness
        for response in task_responses:
            output = response["predicted_test_output"][id_test]
            try:
                output = np.array(output).tolist()
            except:
                continue

            try:
                if True:#check_one_out(output):
                    # Convert the nested list to tuple for Counter
                    serialized_output = serialize_list(output)
                    valid_outputs.append(serialized_output)
                    val_out.append(output)
                    valid_corrections.append(response["correct_test_input"][id_test])
                    valid_train.append(np.mean(response["correct_train_input"]))
            except:
                pass
        if valid_outputs:
            dic_count_out={}
            dic_all_out={}
            for id_out in range(len(valid_outputs)):
                v_out = valid_outputs[id_out]
                if v_out in dic_all_out:
                    dic_all_out[v_out] += 1 + c * valid_train[id_out]
                    dic_count_out[v_out] += 1
                else:
                    dic_all_out[v_out] = 1 + c * valid_train[id_out]
                    dic_count_out[v_out] = 1
            # Get most common outputs from dic_all_out
            most_common = sorted(dic_all_out.items(), key=lambda x: x[1], reverse=True)[:n_output]
            most_common=[item[0] for item in most_common]
            list_majority = [dic_count_out[item] for item in most_common]
                        
            # Get corresponding corrections for each top output
            for idout,top_output_tuple in enumerate(output for output in most_common):
                # Find indices where this output appears
                for i, output in enumerate(valid_outputs):
                    if output == top_output_tuple:
                        # top_corrections.append(valid_corrections[i])
                        list_reponse[idout]["predicted_test_output"][id_test]=val_out[i]
                        list_reponse[idout]["correct_test_input"][id_test]=valid_corrections[i]
                        list_reponse[idout]["majority_vote"][id_test]=list_majority[idout]
                        break
                
    return list_reponse



def rm_untested_task(dic_res):
    """remove task that are not tested (should be useless rm ?)"""
    dic_res_rm = {}
    for k,v in dic_res.items():
        dic_res_rm[k]=[]
        for resp in v:
            if "correct_train_input" in resp:
                dic_res_rm[k].append(resp)  
    return dic_res_rm