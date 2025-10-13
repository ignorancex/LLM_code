from utils.normalize_sql import normalize_sql
from func_timeout import func_timeout
from copy import deepcopy
from transformers import GenerationConfig

import numpy as np
import json
import torch
from utils.functions import swap_memory

from utils.inference_utils import segment_step
import heapq

from utils.prompts import prepare_eval_prompt, prepare_eval_res_prompt, prepare_prompt, prepare_prompt_correction, prepare_prompt_step, prepare_promptVR
from utils.prompts import extract_completionsV3, extract_completionsV2, extract_completionsV4
#--------------------------------
# Helper functions
#--------------------------------

def generate_completions(generator,prompt,config,device_swap, max_new_tokens = 300):
    # move model to GPU
    if device_swap: 
        swap_memory(generator.model, device="cuda", verbose=False)
    # generate completions
    responses = generator.generate(prompt, config, max_new_tokens=max_new_tokens)
    # move model back to CPU
    if device_swap:
        swap_memory(generator.model, device="cpu", verbose=False)   
    return responses

def evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval=None, device_swap=False):
    if device_swap: # move model to GPU
        swap_memory(evaluator.model, device="cuda", verbose=False)

    if retriever_eval is None:
        scores = evaluator.score(example["db_id"], example["question"], sql_completions, evaluation_config)
    else:
        scores = evaluator.score_fewshot(example["db_id"], example["question"], sql_completions, retriever_eval, evaluation_config)

    if device_swap: # move model to GPU
        swap_memory(evaluator.model, device="cpu", verbose=False)
    
    return scores

def log_example(log, example, sql_completions=None, scores=None, candidates_scores=None, responses = None,evaluation_log = None):
    example_log = deepcopy(example)
    if sql_completions is not None:
        example_log["top_n"] = sql_completions
    if scores is not None:
        example_log["scores"] = scores
    if candidates_scores is not None:
        example_log["candidates"] = candidates_scores
    if responses is not None:
        example_log["responses"] = responses
    if evaluation_log is not None:
        example_log["evaluations"] = evaluation_log
    log.append(example_log)

#--------------------------------
# LLM Planners main functions
#--------------------------------

# rerank completions and return best completion
def rerank(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False, prompt_method=0,useSchema=False):
    # load configs
    config = json.load(open(generation_config))
    evaluation_config = json.load(open(evaluation_config))
    # for reasoning evaluator
    evaluator_config = {
        "_from_model_config": True,
        "do_sample": False,
        "num_return_sequences": 1
    }

    # prepare prompt
    headstr = "SELECT"
    prompt, _ = prepare_prompt(example, retriever_gen, prompt_method, headstr=headstr)
    #headstr = None
    #prompt, _ = prepare_promptV2(example, retriever_gen, prompt_method, headstr=headstr)
    
    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap,max_new_tokens = 300)

    # extract completions. V3 works better for all prompt methods.
    #sql_completions = extract_completions(responses,prompt_method)
    #sql_completions = extract_completionsV2(responses, prompt, headstr=headstr)
    sql_completions = extract_completionsV3(responses, prompt, headstr=headstr)

    # evaluate completions
    if evaluator.__class__.__name__ == "LLMReasoningEvaluator":
        # evaluate completion via reasoning
        scores, evaluation_log = evaluator.score(example, sql_completions, evaluation_config,\
                                                 retriever_eval=retriever_eval, device_swap=device_swap,\
                                                      evaluator_config=evaluator_config, useSchema=useSchema)
    else:
        scores = evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval, device_swap)
        evaluation_log = ""
    
    scores =np.array(scores)

    # Find the minimum score
    min_score = np.min(scores)

    # Get indices of the minimum score
    min_indices = np.where(scores == min_score)[0]

    # Select the smallest candidate among them
    final_candidate = min(sql_completions[i] for i in min_indices)

    # log
    log_example(log, example, sql_completions=sql_completions, scores=scores.tolist(), evaluation_log = evaluation_log)

    # return best completion
    return final_candidate.replace("\n", " ")

def rerank_part1(example, generator, retriever_gen, generation_config, log, device_swap=False, prompt_method=0):
    # load configs
    config = json.load(open(generation_config))

    # prepare prompt
    headstr = "SELECT"
    prompt, _ = prepare_prompt(example, retriever_gen, prompt_method, headstr=headstr)
    #headstr = None
    #prompt, _ = prepare_promptV2(example, retriever_gen, prompt_method, headstr=headstr)
    
    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap,max_new_tokens = 300)

    # extract completions. V3 works better for all prompt methods.
    #sql_completions = extract_completions(responses,prompt_method)
    #sql_completions = extract_completionsV2(responses, prompt, headstr=headstr)
    sql_completions = extract_completionsV3(responses, prompt, headstr=headstr)


    # log
    log_example(log, example, sql_completions=sql_completions)

    # return generation
    return log

def rerank_part2(example, evaluator, retriever_eval, sql_completions, evaluation_config, log, device_swap=False, useSchema=False):
    # load configs
    evaluation_config = json.load(open(evaluation_config))
    # for reasoning evaluator
    evaluator_config = {
        "_from_model_config": True,
        "do_sample": False,
        "num_return_sequences": 1
    }


    # evaluate completions
    if evaluator.__class__.__name__ == "LLMReasoningEvaluator":
        # evaluate completion via reasoning
        scores, evaluation_log = evaluator.score(example, sql_completions, evaluation_config,\
                                                 retriever_eval=retriever_eval, device_swap=device_swap,\
                                                      evaluator_config=evaluator_config, useSchema=useSchema)
    else:
        scores = evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval, device_swap)
        evaluation_log = ""
    
    scores =np.array(scores)

    # Find the minimum score
    min_score = np.min(scores)

    # Get indices of the minimum score
    min_indices = np.where(scores == min_score)[0]

    # Select the smallest candidate among them
    final_candidate = min(sql_completions[i] for i in min_indices)

    # log
    log_example(log, example, sql_completions=sql_completions, scores=scores.tolist(), evaluation_log = evaluation_log)

    # return best completion
    return final_candidate.replace("\n", " ")

# greedy completion. Only one completion is generated and returned
def greedy(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False,prompt_method=0,useSchema=False):
    # load configs
    config = json.load(open(generation_config))

    # prepare prompt
    #headstr = "SELECT"
    #prompt, _ = prepare_prompt(example, retriever_gen, prompt_method, headstr=headstr)
    #headstr = None
    #prompt, _ = prepare_promptV2(example, retriever_gen, prompt_method, headstr=headstr)
    
    # for reasoning model
    headstr = None
    prompt, _ = prepare_promptVR(example, retriever_gen, prompt_method, headstr=headstr)

    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap,max_new_tokens = 1024)

    # extract completions. V3 works better for all prompt methods.
    #sql_completions = extract_completions(responses,prompt_method)
    #sql_completions = extract_completionsV2(responses, prompt, headstr=headstr)
    # sql_completions = extract_completionsV3(responses, prompt, headstr=headstr)
    
    # for reasoning model
    sql_completions = extract_completionsV4(responses)

    # debug print
    #print(f"Case:==========================\n\nResponse:\n{responses[0]}")
    #print(f"\n\nSQL:{sql_completions[0]}") if len(sql_completions)>0 else print("No SQL")


    # log
    log_example(log, example, sql_completions=sql_completions, responses=responses)

    # return completion
    return sql_completions[0].replace("\n", " ")

# iterative correction. Multiple completions are generated and scored. The best completion is used as input for the next iteration.
def iter_correction(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False, prompt_method=0):
    # load configs
    config = json.load(open(generation_config))
    evaluation_config = json.load(open(evaluation_config))
    evaluator_config = {
        "_from_model_config": True,
        "do_sample": False,
        "num_return_sequences": 1
    }

    # Step 1: Prompt the generator and sample initial plans.
    headstr = "SELECT"
    prompt, _ = prepare_prompt(example, retriever_gen, prompt_method, headstr=headstr)
    #headstr = None
    #prompt, _ = prepare_promptV2(example, retriever_gen, prompt_method, headstr=headstr)
    
    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap,max_new_tokens = 300)

    # extract completions. V3 works better for all prompt methods.
    #sql_completions = extract_completions(responses,prompt_method)
    #sql_completions = extract_completionsV2(responses, prompt, headstr=headstr)
    sql_completions = extract_completionsV3(responses, prompt, headstr=headstr)

    # Planning iteration setup.
    current_score = 18 #0
    patience = 0
    candidates_scores = {}
    answer_sql = ""
    all_evaluation_log = ''

    for t in range(5):
        # Step 2: Score the current batch of plans.
        
        # scores = evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval, device_swap)
        scores, evaluation_log = evaluator.score( example, sql_completions, evaluation_config, retriever_eval, device_swap, evaluator_config)
        
        all_evaluation_log += f"\n# i == {t}\n\n" + evaluation_log

        # Step 3: Find the plan with highest score. Scores are negated for min heap implementation in tree search.
        best_score = min(scores)

        # Step 4: Check termination conditions and replace the old plan with the currently best one, if any.
        if best_score < 0.0: #-0.75:
            answer_sql = sql_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_sql] = best_score
            break
        elif best_score >= current_score:
            patience += 1
            if patience >= 2:
                break
        else:
            answer_sql = sql_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_sql] = best_score
            patience = 0


        # Step 5: Prompt the generator for 0-shot correction. Sample a new batch of plans.
        prompt = prepare_prompt_correction(example, answer_sql,prompt_method, headstr=headstr)

        # generate completions
        responses = generate_completions(generator,prompt,config,device_swap,max_new_tokens = 300)

        # extract completions
        #sql_completions = extract_completions(responses,prompt_method)
        #sql_completions = extract_completionsV2(responses, prompt, headstr=headstr)
        sql_completions = extract_completionsV3(responses, prompt, headstr=headstr)

    answer = answer_sql.replace("\n", " ")

    # log
    log_example(log, example, candidates_scores=candidates_scores, evaluation_log = all_evaluation_log)

    return answer

# Monte-Carlo Tree Search
def tree_search_mc(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False, prompt_method=0):
    headstr = "SELECT"
    # load configs
    config = json.load(open(generation_config))
    evaluation_config = json.load(open(evaluation_config))

    # prepare prompt
    prompt, model_inp = prepare_prompt(example, retriever_gen, prompt_method, headstr=headstr)
    
    partial_sql = ""
    current_score = 18
    heap = []
    candidates = []
    sql_score_cache = {}

    for t in range(50):
        # Step 1: Prompt the generator and sample initial steps.
        # generate completions
        responses = generate_completions(generator,prompt,config,device_swap,max_new_tokens = 300)

        # extract completions
        #sql_completions = extract_completions(responses,prompt_method)
        sql_completions = extract_completionsV2(responses, prompt, headstr=headstr)

        # Segment generator completions to find the first new step.
        steps = set([
            (
                segment_step(sql[len(partial_sql):].lstrip()).rstrip()
                if len(sql) > len(partial_sql)
                else sql
            )
            for sql in sql_completions
        ])

        # Step 2: For each new step, score it with the best the Monte-Carlo rollout.
        step_score = {}
        for s in steps:
            # Sample Monte-Carlo rollouts.
            mc_prompt = prepare_prompt_step(model_inp, partial_sql, s, prompt_method)

            # generate completions
            mc_rollouts = generate_completions(generator,mc_prompt,config,device_swap,max_new_tokens = 300)

            # extract completions
            #mc_sql_completions = extract_completions(mc_rollouts,prompt_method)
            mc_sql_completions = extract_completionsV2(mc_rollouts, mc_prompt, headstr="")

            # Evaluate Monte-Carlo rollouts.
            if retriever_eval is None:
                try:
                    scores = func_timeout(300.0, evaluator.score, args=(example["db_id"], example["question"], mc_sql_completions, evaluation_config))
                except:
                    scores = [0 for c in mc_sql_completions]
            else:
                try:
                    scores = func_timeout(300.0, evaluator.score_fewshot, args=(example["db_id"], example["question"], mc_sql_completions, retriever_eval, evaluation_config))
                except:
                    scores = [0 for c in mc_sql_completions]

            # Find the plan with highest score. Scores are negated for min heap implementation.
            step_score[s] = min(scores)

        # Step 3: Update the heap memory with new steps and scores.
        for k, v in step_score.items():
            if v <= 0: # prune if all rollouts have execution errors
                heapq.heappush(heap, (v, partial_sql + " " + k))

        # If the heap is empty before finding the first complete program (e.g. all of them are non-executable), return the first initial plan.
        if len(heap) == 0:
            if not partial_sql.endswith(";"):
                partial_sql = sql_completions[0]
            break

        # Step 4: Pop out the current best (partial) plan in heap. If complete, return it.
        current_score, partial_sql = heapq.heappop(heap)
        if partial_sql.endswith(";"):
            break
        else:
            partial_sql = normalize_sql(partial_sql).rstrip(";")

            prompt = prepare_prompt_step(model_inp, partial_sql, s="", prompt_method=prompt_method)
    
        # For debugging
        # print(current_score, partial_sql)
        # print(heap)

    answer = partial_sql.replace("\n", " ")
    heapq.heappush(heap, (current_score, partial_sql))

    # log data
    example_log = deepcopy(example)
    example_log["heap"] = dict([(t[1], t[0]) for t in heap])
    log.append(example_log)

    return answer