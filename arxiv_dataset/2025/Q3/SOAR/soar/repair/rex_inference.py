import numpy as np
import random
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from soar.api import LLM_serv
from soar.preprocess import get_dataset
from soar.repair.rex import REX
from soar.llm_utils import merge_results,to_formated_list
import pickle
from collections import defaultdict

def parse_arguments():
    """Parse command-line arguments for ARC experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Argument parsing for ARC experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Path configurations
    path_group = parser.add_argument_group("Path configurations")
    path_group.add_argument("--base_path", type=str, default="/home/flowers/work/SOAR/",
                          help="Path to git project root (evaluate_model)")
    path_group.add_argument("--path_archive", type=str, help="Path to load archive")
    path_group.add_argument("--extra_save_name", type=str, default="",
                          help="Extra save name")
    path_group.add_argument("--path_save_res", type=str,
                          help="Results saving path")
    # Model parameters
    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument("--path_model", type=str, default="qwen",
                           help="HF model identifier")
    model_group.add_argument("--n_gpu", type=int, default=1,
                           help="Number of GPUs for parallel processing")
    model_group.add_argument("--model_len", type=int, default=30000,
                           help="Context length limit for vLLM models")
    model_group.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True,
                           help="Use FP8 precision (requires compatible GPU)")
    model_group.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                           help="set to True if you want thinking (for qwen3)")

    # Inference parameters
    inference_group = parser.add_argument_group("Inference parameters")
    inference_group.add_argument("--temperature", type=float, default=1.0,
                               help="Sampling temperature")
    inference_group.add_argument("--top_p", type=float, default=1.0,
                               help="Top-p nucleus sampling")
    inference_group.add_argument("--min_p", type=float, default=0.05,
                                help="Min_p sampling")
    inference_group.add_argument("--top_k", type=float, default=-1,
                                help="Top-k sampling")    
    inference_group.add_argument("--max_tokens", type=int, default=-1,
                                help="max gen tokens")
    
    # dataset parameters
    dataset_group = parser.add_argument_group("Experimental flags")
    dataset_group.add_argument("--split", type=str, default="train",
                            help="'train' or 'val'")
    dataset_group.add_argument("--arc2", action=argparse.BooleanOptionalAction,
                            help="arc 2")

    # REX parameters
    rex_group = parser.add_argument_group("Experimental flags")
    rex_group.add_argument("--sampling_method", type=str,
                        help="uniform, uniform_correctness, REX, REX_ablation_discount, coverage_test_units ",default="REX")
    rex_group.add_argument("--correctness", type=str, 
                        help="correctness = zero low medium max (use with sampling_method = uniform_correctness) #uncompilable ",default="medium")
    rex_group.add_argument("--crossover", action=argparse.BooleanOptionalAction, 
                        help="crossover")


    rex_group.add_argument("--total_budget", type=int, default=600, 
                        help="total budget / API calls")
    rex_group.add_argument("--n_completion", type=int, default=4, 
                        help="number of completions per prompt")
    rex_group.add_argument("--use_prev_gen",action=argparse.BooleanOptionalAction ,default=False,
                        help="merge solution of this generation with previous generation")

     # Advanced parameters
    advanced_group = parser.add_argument_group("Advanced parameters")
    advanced_group.add_argument("--smart_inference", action=argparse.BooleanOptionalAction, default=True,
                            help="Optimize inference for solved tasks")
    advanced_group.add_argument("--seed", type=int, default=0,
                            help="Random seed for reproducibility")
    advanced_group.add_argument("--gpu_mem", type=float, default=0.9,
                            help="GPU memory utilization ratio (reduce that if OOM errors occur)")
    return parser.parse_args()

def get_category_correctness(response,split="train"): 
    cat = np.mean(response["correct_train_input"]) 
    if cat==0: # Zero correct response 
        return "zero"
    elif cat<=0.34: # Low correctness ]0,0.34] 
        return "low" 
    elif cat <0.98: # Medium correctness ]0.34,0.98[ 
        return "medium" 
    else: # High correctness ]0.98,1] 
        if split=="train": 
            if np.mean(response["correct_test_input"])!=1: 
                return "max_zero"
            else:
                return "max"
        else:
            return "max" 
        
def get_previous_paths(path):
    """
    get previous paths (from previous gen) from a path
    .../aces_arc/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-1/solution/....pkl
    -> [.../aces_arc/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-0/solution/....pkl, .../aces_arc/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-1/solution/....pkl]
    """

    # Find the pattern 'gen-X' in the path
    
    import re
    match = re.search(r'gen-(\d+)/', path)
    if not match:
        return [path]
    if not args.use_prev_gen:
        return [path]
    current_gen = int(match.group(1))
    base_path = path[:match.start()]
    suffix = path[match.end():]
    
    # Generate all paths from gen-0 to current gen
    paths = []
    for i in range(current_gen + 1):

        new_path = f"{base_path}gen-{i}/{suffix}"
        paths.append(new_path)
    return paths

def rm_last_slash(path):
    if path.endswith("/"):
        return path[:-1]
    return path

args = parse_arguments()
#get data
train_data, val_data, test_data = get_dataset(data_path=args.base_path,arc_2=args.arc2)
data2test = train_data
data2test.update(val_data)

# sample program to refine
np.random.seed(args.seed)

if args.extra_save_name!="":
    extra_extra_name="_"+args.extra_save_name
else:
    extra_extra_name=""
extra_name = f"_sampl-{args.sampling_method}_{args.seed}"+extra_extra_name #correct-{args.correctness}
#remove what is after the last /
path_save = args.path_archive.split("/solution/")[0]+"/refinement/"+f"{args.split}_{extra_name}.pkl"


if args.path_save_res is not None:  
    path_save = rm_last_slash(args.path_save_res) + f"/{args.split}_{extra_name}.pkl"


n_completions = args.n_completion
# if path_save exists load it

print("path_save: ", path_save)
if os.path.exists(path_save):
    print(f"File {path_save} exists")
    print("load from checkpoint")
    with open(path_save, 'rb') as f:
        archive = pickle.load(f)
        archive = to_formated_list(archive)
        archive = merge_results(archive)
    print("Data loaded: ", path_save)
    keep_dic={}
    list_len=[]
    for k,v in archive.items():
        list_len.append( len([i for i in v if i["type"]=="refined"]))
        
    max_repair= max(list_len) 
    print("already repaired: ", max_repair)
    
    N_repair= int(args.total_budget - max_repair)
    print("N_remaining", N_repair)
    dic_sample_code=archive
else:            
    # check if folder exists
    path_save_folder = os.path.dirname(path_save)
    if not os.path.exists(path_save_folder):
        os.makedirs(path_save_folder,exist_ok=True)
    # load inital data and sample
    print("current path: ", args.path_archive)
    list_path = get_previous_paths(args.path_archive)
    print("list data loaded: ")
    all_archive = []
    for p in list_path:
        print("- ",p)
        try:
            with open(p, 'rb') as f:
                archive = pickle.load(f)
                archive = to_formated_list(archive)
                all_archive += archive
        except:
            print(f"error loading: {p}")
    archive = merge_results(all_archive)
                

    # sample data for repair
    # limit number of sample per category for diversity and it as archive as quite large it help with RAM consumption
    n_sample_per_category=300

    dic_sample_code={}
    list_task = list(archive.keys())


    # sample 400 from each category
    
    if args.split == "train":
        list_category_keep = ["zero","low","medium","max_zero"]
    else:
        list_category_keep = ["zero","low","medium","max"]
    for task_id in list_task:
        # Group responses by category
        categories = defaultdict(list)
        task_responses = archive[task_id]
        
        for response in task_responses:
            category = get_category_correctness(response,split=args.split)
            if category in list_category_keep:
                categories[category].append(response)
        
        # Sample n_sample_per_category from each category (or all if less than n_sample_per_category)
        samples = []
        for category in list_category_keep:
            category_responses = categories[category]
            sample_size = min(n_sample_per_category, len(category_responses))
            samples.extend(random.sample(category_responses, sample_size))
        
        # Store merged samples for this task
        if len(samples) > 0:
            dic_sample_code[task_id] = samples
    print("number of initial key to repair: ", len(dic_sample_code))
    archive = dic_sample_code

    N_repair = args.total_budget // n_completions

rex = REX(archive=dic_sample_code,
                data2test=data2test, 
                path_save=path_save,
                sampling_method=args.sampling_method,
                correctness=args.correctness,
                C=20,
                n_completions=n_completions,
                N_budget = N_repair,
                temperature=args.temperature,
                split=args.split,
                sglang = True,
                )


MAX_TIMEOUT = int(60 * 60 * 3)
llm = LLM_serv(args.path_model, model_len=args.model_len, seed=args.seed, n_gpu=args.n_gpu,
                temperature=args.temperature, top_p = args.top_p, top_k=args.top_k, max_timeout=MAX_TIMEOUT, min_p=args.min_p,gpu_mem=args.gpu_mem,fp8=args.fp8,max_tokens=args.max_tokens,enable_thinking=args.enable_thinking)
try:
    rex.run(llm, tokenizer = None,save=True)

except Exception as e:
    print("Error during REX run:")
    llm.terminate()
    raise e
llm.terminate()