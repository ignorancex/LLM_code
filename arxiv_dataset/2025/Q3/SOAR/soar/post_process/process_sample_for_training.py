import numpy as np
import pickle
from soar.llm_utils import merge_results, to_formated_list
from soar.llm_utils import get_number_of_solved_tasks_bis, get_info, give_n_majority_vote_v2, weighted_sample
from soar.post_process.filtering import rm_duplicate_resps
from tqdm import tqdm
import argparse
# should move this in main
parser = argparse.ArgumentParser()
parser.add_argument("--path_model", type=str, default="/home/flowers/work/hf/Qwen2.5-Coder-7B-Instruct-AWQ")
parser.add_argument("--path_embed_model", type=str, default="/home/flowers/work/hf/CodeRankEmbed")
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--N_sample_task", type=int, default=50)
parser.add_argument("--path_folder_solution", type=str, default="/home/flowers/work/arc/aces_arc/save_results/full_pipeline_scratch/yolo/Qwen2.5-Coder-32B-Instruct/gen-1/refinement/")
parser.add_argument("--sampling_her",type=str,default="greedy_div",help="sampling strategy for the responses, choice among greedy_div, greedy, uniform, correct, greedy_div_mv")
parser.add_argument("--use_prev_gen",action=argparse.BooleanOptionalAction ,default=False,help="merge solution of this generation with previous generation")
parser.add_argument("--use_repair_data",action=argparse.BooleanOptionalAction ,default=True,help="use_repair_data")
parser.add_argument("--dedup",action=argparse.BooleanOptionalAction ,default=False,help="deduplication of the responses")

args = parser.parse_args()

if args.dedup:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.path_embed_model, trust_remote_code=True)


if not args.sampling_her in ["greedy_div","greedy","uniform","correct","greedy_div_mv"]:
    raise ValueError("sampling_her must be in ['greedy_div','greedy','uniform','correct','greedy_div_mv']")
path_model=args.path_model


split=args.split
N_sample_task = args.N_sample_task
path_save= args.path_folder_solution + f"data4train-{args.split}-n{args.N_sample_task}-{args.sampling_her}.pkl"

def get_deduplicate_output_one_task(task):
    """dedup based on output grid"""
    dic_unique_output = {}
    for result in task:

        list_out = (
            result["predicted_train_output"] +
            result["predicted_test_output"]
            )
        list_out_str = str(list_out)
        if list_out_str in dic_unique_output:
            dic_unique_output[list_out_str] = result
    return list(dic_unique_output.values())

def get_sample_task_uniform(data, N_sample_task):
    list_k = list(data.keys())
    dic_sample = {}
    for k in list_k:
        # sample N_sample_task responses uniformly
        N_sample = min(N_sample_task,len(data[k]))
        dic_sample[k] = np.random.choice(data[k], N_sample, replace=False).tolist()
    return dic_sample

def get_sample_task_greedy(data, N_sample_task):
    list_k = list(data.keys())
    dic_sample = {}
    for k in list_k:
        # sample N_sample_task responses with the highest score (correct_train_input)
        dic_sample[k]  = sorted(data[k], key=lambda x: np.mean(x["correct_train_input"]), reverse=True)[:N_sample_task]
    return dic_sample


def get_sample_task_correct(data, N_sample_task):
    list_k = list(data.keys())
    dic_sample = {}
    for k in list_k:
        # sample N_sample_task responses with the highest score (correct_train_input)
        v_filtered = [v for v in data[k] if all(v["correct_train_input"] + v["correct_test_input"])]
        if len(v_filtered)!=0:
            N_sample = min(N_sample_task,len(v_filtered))
            dic_sample[k] = np.random.choice(v_filtered, N_sample, replace=False).tolist()
    return dic_sample

def get_sample_task_greedy_div(data, N_sample_task,split="train",majority_vote=False):
    """sample N_sample_task responses for each task, half of them are sampled greedily, the other half are sampled uniformly among task """

    N_sample_greedy = N_sample_task // 2 #// 2
    N_sample_uniform = N_sample_task - N_sample_greedy
    list_k = list(data.keys())
    min_number_sample = 100000
    dic_sample = {}
    for k in tqdm(list_k):
        # shuffle data
        data[k] = np.random.permutation(data[k]).tolist()

        if split=="train":
            if args.dedup:
                data[k] = rm_duplicate_resps(model, data[k], threshold = 0.9)
            dic_resp_sorted  = sorted(data[k], key=lambda x: (np.mean(x["correct_train_input"]+ (np.array(x["correct_test_input"])*1.1).tolist()),-len(x["code"])), reverse=True)
        else:
            dic_resp_sorted  = sorted(data[k], key=lambda x:(np.mean(x["correct_train_input"]),-len(x["code"])), reverse=True)
        

        if majority_vote:
            
            for id_resp in range(len(dic_resp_sorted)):
                dic_resp_sorted[id_resp]["id_resp_rm"] = id_resp
            _,list_original_resp = give_n_majority_vote_v2(dic_resp_sorted, n_output=min(N_sample_greedy,len(data[k])),return_responses=True)
            sampled_responses = weighted_sample(list_original_resp, N_sample_greedy)
            dic_sample[k] = sampled_responses
            id_to_rm = [dic["id_resp_rm"] for dic in sampled_responses]
            dic_other_resp = [dic for dic in dic_resp_sorted if dic["id_resp_rm"] not in id_to_rm]
        else:
            dic_sample[k] = dic_resp_sorted[:N_sample_greedy]
            # take example that were not sampled

            dic_other_resp = dic_resp_sorted[N_sample_greedy:]
        # sample N_sample_uniform responses uniformly
        list_diverse=[]
        
        
        # dedup given output
        dic_other_resp = get_deduplicate_output_one_task(dic_other_resp)
        for resp in dic_other_resp:
            if not any(resp["correct_train_input"]):
                list_diverse.append(resp)
          
        if len(list_diverse)<=N_sample_uniform:
            # in case we have not enough example wo correct train input 
            for resp in dic_other_resp:
                if np.sum(resp["correct_train_input"])==1:
                    list_diverse.append(resp)
        if len(list_diverse)<=N_sample_uniform:
            # in case we have not enough example with 1 correct train input
            for resp in dic_other_resp:
                if np.mean(resp["correct_train_input"])<1/2:
                    list_diverse.append(resp)

        dic_sample[k] += np.random.choice(list_diverse, min(N_sample_uniform,len(list_diverse)), replace=False).tolist()
        min_number_sample = min(min_number_sample,len(dic_sample[k]))

    print("min_number_sample: ",min_number_sample)
    return dic_sample


def rm_long_code(solution_merged, path_model):
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(path_model)
    n_del = 0
    
    # Batch process all codes at once
    all_codes = []
    indices = []
    for k, v in solution_merged.items():
        for i, item in enumerate(v):
            all_codes.append(item["code"])
            indices.append((k, i))
    
    # Tokenize all codes in one batch
    # Process in mini-batches
    batch_size = 32768*4  # Adjust based on your memory constraints
    lengths = []
    
    for i in tqdm(range(0, len(all_codes), batch_size), desc="Tokenizing in batches"):
        batch_codes = all_codes[i:i+batch_size]
        batch_tokenized = tokenizer(batch_codes, padding=False)["input_ids"]
        batch_lengths = [len(x) for x in batch_tokenized]
        lengths.extend(batch_lengths)

    # Process deletions of long codes
    to_delete = {}
    for idx, (k, i) in enumerate(indices):
        if lengths[idx] > 4096 and not any(solution_merged[k][i]["correct_train_input"]):
            if k not in to_delete:
                to_delete[k] = []
            to_delete[k].append(i)
            n_del += 1
    
    # Delete items (in reverse order for each key)
    for k in to_delete:
        for i in sorted(to_delete[k], reverse=True):
            del solution_merged[k][i]
    
    return solution_merged


def count_resp(dic):
    count = 0
    for k in dic.keys():
        count+=len(dic[k])
    return count


def get_previous_paths(path):
    """
    .../aces_arc/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-1/solution/
    -> [.../aces_arc/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-0/solution/, .../aces_arc/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-1/solution/]
    """

    # Find the pattern 'gen-X' in the path
    
    import re
    match = re.search(r'gen-(\d+)/', path)
    if not match:
        print("no match for gen-")
        return [path]
    if not args.use_prev_gen:
        print("use_prev_gen is False, return only current path")
        return [path]

    current_gen = int(match.group(1))
    base_path = path[:match.start()]
    suffix = path[match.end():]
    
    # Generate all paths from gen-0 to current gen
    paths = []
    for i in range(current_gen + 1):

        new_path = f"{base_path}gen-{i}/{suffix}"
        paths.append(new_path)
    return paths[::-1]

# load files

solution = []
paths_all_gen = get_previous_paths(args.path_folder_solution)

print("==="*10)
print(f"found {len(paths_all_gen)} generations:")
print("==="*10)

for path in paths_all_gen:
    if args.use_repair_data:
        path_solution = f"{path}{split}_sol_repair.pkl"
    else:
        path_solution = f"{path}{split}_sol.pkl"
    try:
        with open(path_solution, 'rb') as f:
            print(f"loading {path_solution}")
            data = to_formated_list(pickle.load(f))
            solution += data
    except:
        print(f"error: can't load {path_solution}")

solution_merged = merge_results(solution)
print("result before  rm long code:")
get_info(get_number_of_solved_tasks_bis(solution_merged,mv=True))
solution_merged = rm_long_code(solution_merged,path_model)
print("result after rm long code:")

get_info(get_number_of_solved_tasks_bis(solution_merged,mv=True))

if args.sampling_her == "greedy_div":
    data_sampled_her = get_sample_task_greedy_div(solution_merged, N_sample_task,split=split)
elif args.sampling_her == "greedy":
    data_sampled_her = get_sample_task_greedy(solution_merged, N_sample_task)
elif args.sampling_her == "uniform":
    data_sampled_her = get_sample_task_uniform(solution_merged, N_sample_task)
elif args.sampling_her == "correct":
    data_sampled_her = get_sample_task_correct(solution_merged, N_sample_task)
elif args.sampling_her == "greedy_div_mv":
    data_sampled_her = get_sample_task_greedy_div(solution_merged, N_sample_task,split=split,majority_vote=True)
print("result after sampling:")

get_info(get_number_of_solved_tasks_bis(data_sampled_her,mv=True))
print("save data sampled at: ", path_save)
with open(path_save, "wb") as f:
    pickle.dump(to_formated_list(data_sampled_her), f)

