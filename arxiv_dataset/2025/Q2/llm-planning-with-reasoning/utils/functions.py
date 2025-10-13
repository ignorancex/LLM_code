import random
import torch
import numpy as np
from accelerate.utils import set_seed
import json
from tqdm import tqdm
from copy import deepcopy
import evaluate
import nltk
from nltk.data import find

# finds the last matching substring from the list
def find_last_occurrence(main_str, substrings):
    last_index = -1
    last_substring = None

    for sub in substrings:
        index = main_str.rfind(sub)  # Find from end
        if index != -1 and (last_index == -1 or index > last_index):
            last_index = index
            last_substring = sub

    return last_substring, last_index

# check if the nltk resource is available: punkt_tab
# download NLTK's punkt_tab tokenizer if not already downloaded
def check_nltk_resource(resource = 'punkt_tab',verbose = True):
    try:
        find(f'tokenizers/{resource}')
        if verbose:
            print(f"Requirement already satisfied: '{resource}' exists in the NLTK data directory.")
    except LookupError:
        if verbose:
            print(f"Downloading '{resource}'...")
        nltk.download(f'{resource}')

# swap between ram and vram
def swap_memory(model,device=None,verbose=False):
    # if device is not provided, swap between cpu and cuda
    if not device:
        device = "cuda" if model.device == "cpu" else "cpu"
    
    # make the swap
    model.to(device)
    
    # print the device
    if verbose:
        print(f"Model on {model.device}")
    
    # Releases unused GPU memory (if any)
    if device == "cpu":
        torch.cuda.empty_cache()  

# set result filename
def set_result_filename(evaluator_name, generator_name, db_name, method_name, evaluator_type = ''):
    return generator_name.split("/")[-1] + "_" + evaluator_name.split("/")[-1]+ "_" + evaluator_type + "_" + db_name + "_" + method_name

# Set seed for all libraries
def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# intrinsic evaluation
def eval_intrinsic(evaluator, test_fname, evaluation_config, log_fname="", useSchema=False, params_log = ""):
    # Load test data and evaluation configuration
    test_data = json.load(open(test_fname))
    evaluation_config = json.load(open(evaluation_config))

    results = [] # Predicted labels
    labels = [] # Ground truth labels
    log = [] # Log data

    # Pairwise selection accuracy
    pairs_count = 0 # Pair count
    pws_acc = 0 # Pairwise selection accuracy

    # Example-level metrics
    hit = 0 # Hit @ 1
    mrr = 0 # Mean Reciprocal Rank

    for ex in tqdm(test_data):
        # Spider data format:
        # ex has has the following keys:
        # ['db_id', 'schema', 'question', 'sql', 'exec_res', 'top_n', 'top_n_exec_res', 'top_n_label']
        
        sql_completions = ex["top_n"] # Candidate completions


        # Evaluate the completions
        if evaluator.__class__.__name__ == "LLMReasoningEvaluator":
            # for reason
            scores, evaluation_log = evaluator.score(ex, sql_completions, evaluation_config, useSchema=useSchema)
        else:
            # for base and FT
            scores = evaluator.score(ex["db_id"], ex["question"], sql_completions, evaluation_config)
            evaluation_log = ""

        scores = [-s for s in scores] # Negative scores for ranking

        # Pairwise selection accuracy
        # For each pair of completions, if the scores are different, check if the higher scoring completion is correct
        for a in range(len(sql_completions)):
            for b in range(a + 1, len(sql_completions)):
                if ex["top_n_label"][a] != ex["top_n_label"][b]:
                    pairs_count += 1
                    if (
                        (ex["top_n_label"][a] == 1 and scores[a] > scores[b]) or
                        (ex["top_n_label"][b] == 1 and scores[b] > scores[a])
                    ):
                        pws_acc += 1


        # Convert scores to binary classification labels
        cls_res = [(1 if s > 0.5 else 0) for s in scores]
        results += cls_res
        labels += ex["top_n_label"]

        # Log data
        ex_log = deepcopy(ex)
        ex_log["pred_scores"] = scores
        ex_log["pred_labels"] = cls_res
        ex_log["evaluation_log"] = evaluation_log
        log.append(ex_log)

        # Hit @ 1 and MRR
        scores_labels = [(s, g) for s, g in zip(scores, ex["top_n_label"])]
        scores_labels.sort(key=lambda x: x[0], reverse=True)
        reranked_labels = [tu[1] for tu in scores_labels]

        if reranked_labels[0] == 1:
            hit += 1
        for idx, l in enumerate(reranked_labels):
            if l == 1:
                mrr += (1 / (idx + 1))
                break

    # Compute metrics

    # acc_metric = evaluate.load("accuracy")
    # acc = acc_metric.compute(predictions=results, references=labels)["accuracy"]

    f1_metric = evaluate.load("f1")
    pos_f1 = f1_metric.compute(predictions=results, references=labels, pos_label=1)["f1"]
    neg_f1 = f1_metric.compute(predictions=results, references=labels, pos_label=0)["f1"]
    macro_f1 = (pos_f1 + neg_f1) / 2

    # log results
    print_log = "Pair Count: {}\nPWS Acc: {:<20.4f}\nSQL Count: {}\nPos F1: {:<20.4f}\nNeg F1: {:<20.4f}\nMacro F1: {:<20.4f}\nHit @ 1: {:<20.4f}\nMRR: {:<20.4f}\n".format(
            pairs_count, pws_acc / pairs_count, len(results), pos_f1, neg_f1, macro_f1, hit / len(test_data), mrr / len(test_data)
        )
    result_log = {
       "Pair Count":  pairs_count,
       "PWS Acc": pws_acc / pairs_count,
       "SQL Count": len(results),
       "Pos F1": pos_f1,
       "Neg F1:": neg_f1,
       "Macro F1": macro_f1,
       "Hit@1": hit / len(test_data),
       "MRR" : mrr / len(test_data)
    }
    log.append(result_log)
    log.append(params_log)

    # print results
    print(print_log)
    print("PWS Acc\tPos F1\tNeg F1\tMacro F1\tHit@1\tMRR\n")
    print(
        "{:<20.4f},\t{:<20.4f},\t{:<20.4f},\t{:<20.4f},\t{:<20.4f},\t{:<20.4f}\n".format(
            pws_acc / pairs_count, pos_f1, neg_f1, macro_f1, hit / len(test_data), mrr / len(test_data)
        )
    )

    # Log data
    if log_fname != "":
        out = open("log/" + log_fname, "w+", encoding="utf-8")
        json.dump(log, out, indent=2)
        out.close()


# run llm planning end-to-end and save the results
def run_end2end(generator, evaluator,generation_config, evaluation_config, planner, retriever_gen, retriever_eval, test_data,dataset_name,result_fname,log_fname,device_swap=False,prompt_method=0, useSchema=False, params_log = ""):
    results = [] # store the results
    log = [] # store the logs

    # generate responses using planner
    for ex in tqdm(test_data):
        res_sql = planner(ex, generator, evaluator, retriever_gen, retriever_eval,generation_config, evaluation_config, log, device_swap,prompt_method,useSchema=useSchema)
        
        # add the result to the results list
        if dataset_name == "spider":
            results.append(res_sql + "\t" + ex["db_id"]) # spider
        elif dataset_name == "bird":
            results.append(res_sql + "\t----- bird -----\t" + ex["db_id"])
        else:
            raise Exception("Invalid dataset name.")

    # save the results for evaluation
    if dataset_name == "spider":
        out = open(result_fname, "w+", encoding="utf-8")
        out.write("\n".join(results))
        out.close()
    elif dataset_name == "bird":
        out = open(result_fname, "w+", encoding="utf-8")
        json.dump(results, out, indent=2)
        out.close()
    else:
        raise Exception("Invalid dataset name.")

    log.append(params_log) # add params log

    # save the logs
    if log_fname != "":
        out = open(log_fname, "w+", encoding="utf-8")
        json.dump(log, out, indent=2)
        out.close()


def fetch_generated_results(generator_outputs, ex):
    """
    match and find ex in the generator outputs 
    """
    matchkeys = ['db_id', 'schema', 'question', 'sql', 'difficulty']
    for g in generator_outputs:
        matched = True
        for k in matchkeys:
            if g[k] != ex[k]:
                matched = False
        if matched == True:
            return g['top_n']
    return None