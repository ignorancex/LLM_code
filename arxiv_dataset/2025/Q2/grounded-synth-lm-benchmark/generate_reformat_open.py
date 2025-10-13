import numpy as np
import os
import random
import json
import time
import random

import answer_parser
from model_interface import SglModelAsync, SglModelSync
import utils

import prompts


def build_og_msg(context:str, question:str, answer:str):
    
    prompt = prompts.QUESTION_REFORMAT_OPEN_PROMPT.format(context=context, question=question, answer=answer)
    return prompt


def reformat_questions(dataset: list[dict], remote:str, model:str, async_flag:bool=False) -> list[dict]:

    
    # Create a list to store model responses
    model_responses = list()
    # Copy over question (into orig_question), id, and context
    for item in dataset:
        response_item = {}
        if 'question' in item:
            response_item['orig_question'] = item['question']
        else:
            raise ValueError("No question found in {item}")
        if 'answer' in item:
            response_item['orig_answer'] = item['answer']
        else:
            raise ValueError("No answer found in {item}")
        if 'context' in item:
            response_item['context'] = item['context']
        else:
            raise ValueError("No context found in {item}")
        response_item['failed'] = False
        model_responses.append(response_item)


    # build the prompts
    
    prompts = [build_og_msg(d['context'], d['orig_question'], d['orig_answer']) for d in model_responses]

    if async_flag:
        model = SglModelAsync(remote=remote, model=model)
    else:
        model = SglModelSync(remote=remote, model=model)
    results, total_time = model.generate(prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")
    print(f"total toks/sec: {total_tokens / total_time}")
    print(f"total output toks/sec: {total_output_tokens / total_time}")

    failed_responses = list()
    for i in range(len(results)):
        res = results[i]
        orig_data = dataset[i]
        if res['error'] is not None:
            model_responses[i]['response'] = None
            model_responses[i]['failed'] = True
            model_responses[i]['error'] = res['error']
            failed_responses.append(model_responses[i])
        else:
            model_responses[i]['response'] = res['content']
            parsed = answer_parser.parse_generated_open(res['content'])
            if parsed is None:
                model_responses[i]['failed'] = True
                failed_responses.append(model_responses[i])
            else:
                model_responses[i]['question'] = parsed['question']
                model_responses[i]['answer'] = parsed['correct_answer']
                model_responses[i]['explanation'] = parsed['explanation']
                model_responses[i]['orig_question'] = orig_data['question']
                model_responses[i]['orig_answer'] = orig_data['answer']

    # remove failed responses
    model_responses = [d for d in model_responses if not d['failed']]
    # Remove the 'failed' key from each response
    for response in model_responses:
        if 'failed' in response:
            del response['failed']
    
    return model_responses, failed_responses
    



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts a jsonl dataset into MMLU format to be used as an MCQ evaluation.')
    parser.add_argument('--dataset', type=str, default='squadv2.jsonl', help='dataset to generate, options: squadv2, ucinlp_drop')
    parser.add_argument('--src_dataset_dir', type=str, default='./data-subset', help='source dataset directory')
    parser.add_argument('--out_dataset_dir', type=str, required=True, help='output dataset directory')
    parser.add_argument('--remote', type=str, default="openai")
    parser.add_argument('--model', type=str, default="gpt-4.1-nano")
    parser.add_argument('--disable_async', action='store_false', dest='async_flag', help='Set to disable async processing (async is enabled by default)')
    parser.add_argument('--sample_count', type=int, default=100, help='number of total questions to generate, set to <=0 for all')
    args = parser.parse_args()
    print("Generating reformatted Open Ended Questions")
    print(args)

    start_time = time.time()
    args.dataset = os.path.join(args.src_dataset_dir, args.dataset)

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    # unpack the dataset into list of context, question, answer, instead of context with a list of questions
    dataset = utils.flatten_by_context(dataset)

    if args.sample_count > 0 and args.sample_count < len(dataset):
        dataset = random.sample(dataset, args.sample_count)

    # verify that each element in the dataset has the following keys: question, answer, context
    for item in dataset:
        if 'question' not in item or 'answer' not in item or 'context' not in item:
            raise ValueError('each element in the dataset must have the following keys: question, answer, context')

    print("Dataset has %d contexts" % len(dataset))

    model_responses, failed_responses = reformat_questions(dataset, args.remote, args.model, args.async_flag)

    elapsed_time = time.time() - start_time

    # dump the new MCQ dataset to json in the MMLU format
    base_path = os.path.splitext(args.dataset)[0]  # Remove extension
    fn_basename = os.path.basename(base_path) + "_reformat"
    out_fldr = os.path.join(args.out_dataset_dir, fn_basename)
    os.makedirs(out_fldr, exist_ok=True)
    output_fn = os.path.join(out_fldr, f'{fn_basename}.jsonl')

    print(f"Saving {len(model_responses)} questions to {output_fn}")
    with open(output_fn, 'w') as f:
        json.dump(model_responses, f, indent=2)



