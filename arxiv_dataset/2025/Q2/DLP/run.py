import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2Tokenizer
from lib.prune import prune_wanda,prune_magnitude,prune_sparsegpt
from lib.prune import prune_wanda_outlier_structure,prune_sparsegpt_outlier,prune_wanda_outlier,prune_mag_outlier
from lib.prune import get_dlp_ratios
from lib.prune import prune_sparsegpt_dlp, prune_mag_dlp, prune_wanda_dlp_structure,prune_wanda_dlp
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity
import sys
print('# of gpus: ', torch.cuda.device_count())
from pdb import set_trace as st


import json
from accelerate.logging import get_logger


from transformers import (
    MODEL_MAPPING,
)
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def get_llm(model):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        # cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def main():


    ########################## pruning ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Enoch/llama-7b-hf", help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured")
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="result", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--alpha', type=float, default=0.15, help='alpha')
    parser.add_argument('--threshold', type=float, default=1, help='threshold of outlier')
    parser.add_argument('--strategy', type=str, default="median", help='compute importance method')
    parser.add_argument('--force_compute_ratios', action="store_true", help="whether to force compute the ratio")
    parser.add_argument('--eval_zero_shot', action="store_true", help="whether to zero-shot eval")
    parser.add_argument('--dataset', type=str,default="wikitext2",help="The name of the dataset to use.")
    parser.add_argument("--ratio_path", default="ratios", type=str)
    ########################## gptq ################################
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument('--wbits', type=int, default=16,help='Whether to quantize as well.')
    parser.add_argument('--sym', action='store_true',help='Whether to perform symmetric quantization.')
    parser.add_argument('--percdamp', type=float, default=.01,help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--groupsize', type=int, default=-1,help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--act-order', action='store_true',help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--static-groups', action='store_true',help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')
    
    ########################## gptq ################################
    
    #### saving parameters #####
    
    parser.add_argument(
    "--save_log", action="store_true", help="save log")
    

    #### data parameters #####
    
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )
    
    
    parser.add_argument(
        '--Hyper_m', 
        type=float,
        default=5, )
    
    args = parser.parse_args()
    
    print("args.alpha",args.alpha)
    print ("args.nsamples",args.nsamples)
    print ("args.dataset",args.dataset)
    print ("args.prune_method",args.prune_method)
    print(args)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print("args.sparsity_type",args.sparsity_type)
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
        # args.sparsity_ratio = prune_n / prune_m
        # print("sparsity_ratio:{}".format(args.sparsity_ratio))

    model_name = args.model.split("/")[-1]
   

    print(f"loading llm model {args.model}")
    
    # if "structure" in args.prune_method:
    #     ratios = get_structure_ratio(args)
    
    
    # Offline load moodel
    # args.model = args.cache_dir + "/models--" + args.model.replace("/", "--") + "/model"

    model = get_llm(args.model)
    
    
    print ("model is =================================================================================")
    print (model.__class__.__name__)
    print (model)
    
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)


    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    print ("target sparsity", args.sparsity_ratio)   
    
    model.eval()


    if "dlp" in args.prune_method:
        ratio_file = args.ratio_path + "/" + model_name + "_sparsity_" + str(args.sparsity_ratio)+ "_alpha_" + str(args.alpha)  + ".json"
        if not os.path.exists(ratio_file) or args.force_compute_ratios:
            imp_ratio = get_dlp_ratios(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            with open(ratio_file, 'w') as json_file:
                json.dump(imp_ratio, json_file, indent=4, ensure_ascii=False)
        else:
            with open(ratio_file,  'r', encoding='utf-8') as json_file:
                imp_ratio = json.load(json_file)
            
                
    print("pruning starts")
    ############################ baseline   ############################
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    ############################ baseline   ############################
    
    ############################ owl   ############################
    elif args.prune_method == "wanda_owl":
        prune_wanda_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    elif args.prune_method == "magnitude_owl":
        prune_mag_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    elif args.prune_method == "sparsegpt_owl":
        prune_sparsegpt_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    elif args.prune_method == "wanda_owl_structure":
        prune_wanda_outlier_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
  ############################ owl   ############################
  
    ############################ dlp   ############################
    elif args.prune_method == "wanda_dlp":
        prune_wanda_dlp(args, model, tokenizer, device,imp_ratio, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "wanda_dlp_structure":
        prune_wanda_dlp_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt_dlp":
        prune_sparsegpt_dlp(args, model, tokenizer, device, imp_ratio, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude_dlp":
        prune_mag_dlp(args, model, tokenizer, device, imp_ratio, prune_n=prune_n, prune_m=prune_m)
    ############################ dlp   ############################
    elif args.prune_method == "dense":
        pass


    print(f" prune method is {args.prune_method}")  
    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(model, tokenizer, device, dataset=args.dataset)
    print(f"ppl on {args.dataset} {ppl_test}")

    sys.stdout.flush()

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")

    if args.save_log:
        dirname = "results/{}".format(args.model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        filename = f"log_{args.prune_method}_.txt"
        save_filepath = os.path.join(dirname, filename)
        with open(save_filepath, "a") as f:
            print("method\tactual_sparsity\tsparsity_pattern\talpha\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.alpha}\t{ppl_test:.4f}", file=f, flush=True)
                


    
    if args.eval_zero_shot:
        accelerate=True
        task_list = ["boolq", "rte", "hellaswag", "arc_challenge",  "openbookqa", 'winogrande', 'arc_easy']
        num_shot = 0
        
        
        if args.save_model:
            eval_model = args.save_model
        else:
            eval_model = args.model
        results = eval_zero_shot(eval_model, task_list, num_shot, accelerate)
        model_name = eval_model.split("/")[-1]
        dirname = "eval_zero_shot"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open('{}/results_zero_shot_{}.json'.format(dirname, model_name), 'a') as file:
            json.dump(results, file, indent=2)
    
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
