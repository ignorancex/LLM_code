import os
import argparse
import time
import pickle
import gc
import numpy as np
from tqdm import trange

from soar.preprocess import get_dataset
from soar.llm_utils import (
    save_pkl_secure, 
    get_number_of_solved_tasks, 
    format_all_generation, 
    merge_results, 
    get_dic_task_solved,
    get_dic_only_correct,
    rm_untested_task
)
from soar.prompt import (
    get_solver_prompt, 
    get_list_fewshot_example, 
    prompt_wo_fewshot_v1_, 
    prompt_fewshot_v1,
    format_fewshot_examples
)
from soar.api import LLM_serv
from soar.sandbox.execute_code_less_safe import check_solutions
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_TIMEOUT = int(60 * 60 * 3)  # 5 hours

# Enable CUDA launch blocking for synchronous error reporting
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable CUDA device-side assertions
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# Alternative name that some versions may use
# os.environ['PYTORCH_USE_CUDA_DSA'] = '1'


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
    path_group.add_argument("--path_save_res", type=str,
                          help="Results saving path")
    path_group.add_argument("--path_archive_to_solve", type=str, default="",
                          help="Path to archive with unsolved tasks (inference on tasks that are not solved at least 2 times)")
    path_group.add_argument("--path_fewshot", type=str, default="",
                          help="Path to fewshot examples archive")
    path_group.add_argument("--extra_save_name", type=str, default="",
                          help="Extra save name")

    # Model parameters
    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument("-m", "--path_model", type=str, default="qwen",
                           help="HF model identifier")
    model_group.add_argument("--n_gpu", type=int, default=1,
                           help="Number of GPUs for parallel processing")
    model_group.add_argument("--model_len", type=int, default=30000,
                           help="Context length limit for vLLM models")
    model_group.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True,
                           help="Use FP8 precision (requires compatible GPU)")
    model_group.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                           help="set to True if you want thinking (for qwen3)")
    #TODO: add thinking for qwen3
    # check that for good use for enable_thinking: https://qwen.readthedocs.io/en/latest/deployment/sglang.html#thinking-non-thinking-modes


    # Inference parameters
    inference_group = parser.add_argument_group("Inference parameters")
    inference_group.add_argument("-k", type=int, default=3000,
                               help="k value for pass@k metric")
    inference_group.add_argument("--bs_inference", type=int, default=50,
                               help="Max completions per task per iteration")
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

    ## advanced inference parameters
    inference_group.add_argument("--max_running_requests", type=int, default=0,
                               help="Maximum concurrent inference requests")
    inference_group.add_argument("--schedule_conservativeness", type=float, default=-1,
                               help="Scheduler aggressiveness (-1=auto)")
    inference_group.add_argument("--repetition_penalty", type=float, help="", default=0)


    # dataset parameters
    dataset_group = parser.add_argument_group("Experimental flags")
    dataset_group.add_argument("--split", type=str, default="train",
                            help="'train', 'val' or 'train_eval'")

    dataset_group.add_argument("--use_fewshot_example", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable fewshot example usage")
    dataset_group.add_argument("--arc2", action=argparse.BooleanOptionalAction,
                            help="Enable ARCv2 protocol compatibility")

    # Advanced parameters
    advanced_group = parser.add_argument_group("Advanced parameters")
    advanced_group.add_argument("--smart_inference", action=argparse.BooleanOptionalAction, default=True,
                            help="Optimize inference for solved tasks")
    advanced_group.add_argument("--seed", type=int, default=0,
                              help="Random seed for reproducibility")
    advanced_group.add_argument("--gpu_mem", type=float, default=0.9,
                              help="GPU memory utilization ratio (reduce that if OOM errors occur)")

    return parser.parse_args()


def setup_model(args):
    """setup the LLM model for inference."""
    llm = LLM_serv(args.path_model, model_len=args.model_len, seed=args.seed, n_gpu=args.n_gpu, temperature=args.temperature,
                    max_timeout=MAX_TIMEOUT, min_p=args.min_p,gpu_mem=args.gpu_mem,fp8=args.fp8,repetition_penalty=args.repetition_penalty,
                    top_p = args.top_p, top_k=args.top_k,max_running_requests=args.max_running_requests,schedule_conservativeness=args.schedule_conservativeness,
                    enable_thinking=args.enable_thinking,max_tokens=args.max_tokens)
    return llm



def get_path_pickle(args,dataset_name):
    """Generate the path for saving results in pickle format based on the dataset name and additional parameters."""
    path_pkl_save = os.path.join(args.path_save_res, f"{dataset_name}_s{args.seed}_{args.extra_save_name}.pkl")
    return path_pkl_save

def save_results(args, dict_response, data2test, dataset_name, dic_task_already_solved=None):
    """Check solution and save the results of the evaluation to a pkl file."""
    path_pkl_save = get_path_pickle(args,dataset_name)
    gc.collect()
    
    # check solutions
    dict_results = get_number_of_solved_tasks(dict_response, n_best_codes=2)
    
    # save results to pkl file
    results2save = {"dict_response": dict_response, 'result': dict_results}
    save_pkl_secure(path_pkl_save, results2save)
    if dic_task_already_solved!=None:
        is_val = "val" in dataset_name
        dic_task_already_solved_current_gen = get_dic_task_solved(dict_response,val = is_val)
        for k,v in dic_task_already_solved_current_gen.items():
            dic_task_already_solved[k]+=v
             
    print(f"Results for {dataset_name}:")
    print("Model ID:", args.path_model)
    if dic_task_already_solved!=None:
        print("Number of solved tasks:")
        print("task solved: ",len([k for k,v in dic_task_already_solved.items() if v>0])," total task:",len(dic_task_already_solved))
    return dic_task_already_solved


def get_dic_task_solved(dic_res,task_id=None,val=False):
    """
    Get a dictionary with the number of solved tasks.
    """
    if task_id is None:
        task_id = dic_res.keys()
    dic_task_solved = {k:0 for k in task_id}
    for k in task_id:
        count_correct_res=0
        if k in dic_res:
            for res in dic_res[k]:
                if val:
                    if all(res["correct_train_input"]): #and all(res["correct_test_input"]):
                        count_correct_res+=1
                else:
                    if all(res["correct_train_input"]) and all(res["correct_test_input"]):
                        count_correct_res+=1
        dic_task_solved[k] = count_correct_res
    return dic_task_solved


def load_fewshot(args):
    with open(args.path_fewshot, "rb") as f:
        data_fewshot = merge_results(pickle.load(f))
    list_k_fewshot,list_fewshot_examples = format_fewshot_examples(data_fewshot,args.base_path)
    return list_k_fewshot,list_fewshot_examples


def run_evaluation(args, model, data2test, dataset_name):
    # check if path to save results exists, if not create it
    path_pkl_save = get_path_pickle(args,dataset_name)
    name_dir = os.path.dirname(path_pkl_save)
    os.makedirs(name_dir, exist_ok=True)
    print("result will be saved here: ",path_pkl_save)
    dic_task_already_solved = {k:0 for k in data2test.keys()}
    
    # load prompt fewshot examples
    if args.use_fewshot_example:
        # rewrite case where fewshot examples are sample on the fly (gen-0)
        if args.path_fewshot != "":
            list_k_fewshot,list_all_fewshot_examples = load_fewshot(args)
        else:
            list_k_fewshot,list_all_fewshot_examples = [],[]
    else:
        list_k_fewshot,list_all_fewshot_examples = [],[]
    # check if we already have partial results for this generation (exit if we already have enough solutions)
    try:
        with open(path_pkl_save, 'rb+') as outfile:
            content = pickle.load(outfile)
            content = merge_results(content)
            max_len = max([len(v) for k,v in content.items()])
            del content
    except:
        max_len = 0

    solution2gen = args.k - max_len
    if solution2gen <= 0:
        print(f"Already generated {max_len} solutions for {dataset_name}")
        model.terminate()
        exit()

    k = min(args.bs_inference,solution2gen)
    n_iter = solution2gen // k

    # if solution2gen is not divisible by k, we will have one last iteration with the remaining tasks
    flag_last_iter = False 
    k_remain = solution2gen % k
    if k_remain != 0:
        flag_last_iter = True
        n_iter += 1

    for idx_fewshot in trange(n_iter):
        print(f"Generating solution Iteration: {idx_fewshot + 1}/{n_iter}")
        if idx_fewshot == n_iter - 1 and flag_last_iter:
            k = k_remain

        if args.smart_inference:
            list_task_id_to_solve = [k_id for k_id,v in dic_task_already_solved.items() if v<=100]  
        else:
            list_task_id_to_solve = list(data2test.keys())
            
        # get prompts for the tasks to solve
        prompts_formated = []
        for key in list_task_id_to_solve:
            task2solve = data2test[key]

            # get fewshot examples 
            flag_key = any(k != key for k in list_k_fewshot)
            if args.use_fewshot_example and len(list_all_fewshot_examples) != 0 and flag_key:
                idx_fewshot_selected = idx_fewshot % len(list_all_fewshot_examples)
                if list_k_fewshot[idx_fewshot_selected] != key:
                    list_fewshot_examples = [list_all_fewshot_examples[idx_fewshot_selected]]
                else:
                    # check if at least one key is not in list_k_fewshot to avoid infinite loop
                    while list_k_fewshot[idx_fewshot_selected] == key:
                        idx_fewshot_selected = (idx_fewshot_selected + 1) % len(list_all_fewshot_examples)
                    list_fewshot_examples = [list_all_fewshot_examples[idx_fewshot_selected]]
            else:
                list_fewshot_examples = []
            if len(list_fewshot_examples) == 0:
                prompt_solver = prompt_wo_fewshot_v1_
            else:
                prompt_solver = prompt_fewshot_v1
            prompt_formated = get_solver_prompt(
                task2solve,
                list_fewshot_examples,
                prompt_solver=prompt_solver,
                grid_display_mode="numpy",
                alt_colors=True
            )
            prompts_formated.append(prompt_formated)        

        # generate LLM completions 
        time_start = time.time()
        print("number of completion: ",k)
        results = model.generate(prompts_formated,n=k)

        # format results
        dict_response = format_all_generation(results, list_task_id_to_solve, use_vllm_generate=False)
        # remove tasks that are not tested
        dict_response = check_solutions(dict_response, data2test, keep_only_correct_results=False)
        dict_response = rm_untested_task(dict_response)

        if dataset_name == "train" and args.path_fewshot != "":
            list_k_fs,list_fs_examples = format_fewshot_examples(dict_response,args.base_path)
            for k_fs, fs_example in zip(list_k_fs, list_fs_examples):
                if k_fs not in list_k_fewshot:
                    list_k_fewshot.append(k_fs)
                    list_all_fewshot_examples.append(fs_example)
        # save results
        dic_task_already_solved = save_results(args, dict_response, data2test, dataset_name,dic_task_already_solved)

    time_end = time.time()
    print("===="*10)
    print(f"Total time for {dataset_name}: {time_end - time_start:.2f} seconds")
    print("===="*10)

def load_archive_to_solve(args, train_data):
    """Load the archive with unsolved tasks and filter the train data accordingly.
    (This is used to solve tasks that are not solved at least 2 times in the archive)"""
    with open(args.path_archive_to_solve, "rb") as f:
        archvive_given = merge_results(pickle.load(f))
        dic_k_solved = get_dic_task_solved(archvive_given, train_data.keys())
        list_k_unsolved = [k for k,v in dic_k_solved.items() if v<2]
        train_data = {k:train_data[k] for k in list_k_unsolved}
    return train_data

def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    # set seed
    if args.use_fewshot_example:
        print("use fewshot example for inference")


    print("Time experiment start:", time.ctime())
    train_data_arc1, val_data_arc1, _ = get_dataset(data_path=args.base_path)
    train_data, val_data, _ = get_dataset(data_path=args.base_path,arc_2=args.arc2)
    if args.arc2:
        #rm train_data that are in train_data_arc1
        train_data = {k:train_data[k] for k in train_data.keys() if (k not in train_data_arc1.keys()) and (k not in val_data_arc1.keys())}
        print("inference on n data")
    

 
    model = setup_model(args)
    try:
        if args.split == "train_eval":
            run_evaluation(args, model, train_data, "train")
            run_evaluation(args, model, val_data, "val")

        elif args.split == "val":
            run_evaluation(args, model, val_data, "val")
        elif args.split == "train":
            # used to solve tasks that are not solved at least 2 times in the archive
            if args.path_archive_to_solve != "":
                train_data = load_archive_to_solve(args,train_data)
            run_evaluation(args, model, train_data, "train")
        else:
            raise ValueError(f"Unknown split: {args.split}. Use 'train', 'val' or 'train_eval'.")
    except Exception as e:
        # end sglang server
        model.terminate()
        raise e
    
    # end sglang server
    model.terminate()

if __name__ == "__main__":
    main()