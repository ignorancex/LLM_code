import numpy as np
import json
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answer_parser
from model_interface import SglModelAsync
import prompts


# def build_clarity_prompt(context:str, question:str):
#     prompt = f"""
# <question>{question}</question>

# Your task is to rate from 1 to 5 the clarity and understandability of the question.
# A rating of 1 is unclear, and 5 is the clearest.
# A rating of 5 should be used for questions that are understandable and contain all the necessary information, even if the question is complex.

# Think through it step by step. Determine the question clarity. Then respond with "ANSWER: <answer>". Respond only with the number in [1, 2, 3, 4, 5].
# \n\n"""
#     return prompt

# def build_clarity_scores(dataset: list[dict], remote:str, model:str) -> list[dict]:

#     # build the prompts
#     prompts = [build_clarity_prompt(d.get('context', ''), d['question']) for d in dataset]

#     model = SglModelAsync(remote=remote, model=model)
#     results, total_time = model.generate(prompts)
#     print(f"in total took: {total_time} seconds")
#     print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

#     total_input_tokens = sum([res['input_tokens'] for res in results])
#     total_output_tokens = sum([res['output_tokens'] for res in results])
#     total_tokens = sum([res['total_tokens'] for res in results])
#     print(f"total input tokens: {total_input_tokens}")
#     print(f"total output tokens: {total_output_tokens}")
#     print(f"total tokens: {total_tokens}")

#     for i in range(len(results)):
#         res = results[i]
#         if res['error'] is not None:
#             raise Exception(f"Error: {res['error']}")
#         else:
#             parsed = answer_parser.parse_number(res['content'], valid_options=[1,2,3,4,5])
#             if parsed is None:
#                 # raise Exception(f"Failed to parse response: {res['content']}")
#                 dataset[i]['clarity_score'] = 1  # default to min score
#             else:
#                 dataset[i]['clarity_score'] = parsed

#     return dataset


# def build_difficulty_prompt(context:str, question:str):
#     prompt = f"""
# <context>{context}</context>
# <question>{question}</question>

# Your task is to rate from 1 to 5 the difficulty of the question.
# A rating of 1 is the easiest, and 5 is the hardest.
# A rating of 5 is reserved for questions that require a very deep understanding of the content by a professional domain expert.

# Think through it step by step. Determine the question difficulty. Then respond with "ANSWER: <answer>". Respond only with the number in [1, 2, 3, 4, 5].
# \n\n"""
#     return prompt

# def build_difficulty_scores(dataset: list[dict], remote:str, model:str) -> list[dict]:

#     # build the prompts
#     prompts = [build_difficulty_prompt(d.get('context', ''), d['question']) for d in dataset]

#     model = SglModelAsync(remote=remote, model=model)
#     results, total_time = model.generate(prompts)
#     print(f"in total took: {total_time} seconds")
#     print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

#     total_input_tokens = sum([res['input_tokens'] for res in results])
#     total_output_tokens = sum([res['output_tokens'] for res in results])
#     total_tokens = sum([res['total_tokens'] for res in results])
#     print(f"total input tokens: {total_input_tokens}")
#     print(f"total output tokens: {total_output_tokens}")
#     print(f"total tokens: {total_tokens}")

#     for i in range(len(results)):
#         res = results[i]
#         if res['error'] is not None:
#             raise Exception(f"Error: {res['error']}")
#         else:
#             parsed = answer_parser.parse_number(res['content'], valid_options=[1,2,3,4,5])
#             if parsed is None:
#                 raise Exception(f"Failed to parse response: {res['content']}")
#             else:
#                 dataset[i]['difficulty_score'] = parsed

#     return dataset


# def build_answerability_prompt(context:str, question:str):
#     # <context>{context}</context>
#     # Your task is to rate from 1 to 5 if the question can be fully understood without the context, or whether required disambiguation information is missing from the question. 
#     prompt = f"""
# Your task is to rate from 1 to 5 if the question can be fully understood without additional background information, or whether required disambiguation information is missing from the question. 
# "As of the 2015 NFL season, how many Super Bowl titles had the Denver Broncos won?" is a 5.
# "What event in 1861 contributed to the temporary strength of republicanism in Britain during Queen Victoria's reign, as it led to her seclusion and reduced public appearances?" is a 5.
# "In which year was the country not a member of FIFA, as indicated in the table?" is a 1.
# "As of the census of 2000, how many families were residing in the city?" is a 1.

# <question>{question}</question>

# Think through it step by step. Determine if the question is understadable. Then respond with "ANSWER: <answer>". Respond only with the number in [1, 2, 3, 4, 5].
# """
#     return prompt


# def build_answerabiltiy_scores(dataset: list[dict], remote:str, model:str) -> list[dict]:

#     # build the prompts
#     prompts = [build_answerability_prompt(d.get('context', ''), d['question']) for d in dataset]

#     model = SglModelAsync(remote=remote, model=model)
#     results, total_time = model.generate(prompts)
#     print(f"in total took: {total_time} seconds")
#     print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

#     total_input_tokens = sum([res['input_tokens'] for res in results])
#     total_output_tokens = sum([res['output_tokens'] for res in results])
#     total_tokens = sum([res['total_tokens'] for res in results])
#     print(f"total input tokens: {total_input_tokens}")
#     print(f"total output tokens: {total_output_tokens}")
#     print(f"total tokens: {total_tokens}")

#     for i in range(len(results)):
#         res = results[i]
#         if res['error'] is not None:
#             raise Exception(f"Error: {res['error']}")
#         else:
#             parsed = answer_parser.parse_number(res['content'], valid_options=[1,2,3,4,5])
#             if parsed is None:
#                 raise Exception(f"Failed to parse response: {res['content']}")
#             else:
#                 dataset[i]['answerability_score'] = parsed

#     # # Extract the lowest 10 answerability scores and print the questions
#     # if len(dataset) > 0 and all('answerability_score' in item for item in dataset):
#     #     # Sort the dataset by answerability score (ascending)
#     #     sorted_by_score = sorted(dataset, key=lambda x: x['answerability_score'])
        
#     #     # Get the lowest 10 (or fewer if dataset is smaller)
#     #     lowest_n = min(10, len(sorted_by_score))
#     #     lowest_scores = sorted_by_score[:lowest_n]
        
#     #     print("\n=== Lowest Answerability Scores ===")
#     #     for i, item in enumerate(lowest_scores):
#     #         print(f"{i+1}. Score: {item['answerability_score']} - Question: {item['question']}")
#     #     print("===================================\n")

#     return dataset



# def build_relevance_prompt(context:str, question:str):
#     prompt = f"""
# <context>{context}</context>
# <question>{question}</question>

# Your task is to rate from 1 to 5 if the question is relevant to the context. Respond only with the number in [1, 2, 3, 4, 5].
# """
#     return prompt


# def build_coverage_prompt(context:str, question:str, answer:str):
#     prompt = f"""
# <context>{context}</context>
# <question>{question}</question>
# <answer>{answer}</answer>

# Your task is to rate from 1 to 5 if the answer can be extracted from the context and the question. Respond only with the number in [1, 2, 3, 4, 5].
# """
#     return prompt

# def build_coverage_scores(dataset: list[dict], remote:str, model:str) -> list[dict]:

#     # build the prompts
#     prompts = [build_coverage_prompt(d.get('context', ''), d['question'], d['choices'][d['answer']]) for d in dataset]

#     model = SglModelAsync(remote=remote, model=model)
#     results, total_time = model.generate(prompts)
#     print(f"in total took: {total_time} seconds")
#     print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

#     total_input_tokens = sum([res['input_tokens'] for res in results])
#     total_output_tokens = sum([res['output_tokens'] for res in results])
#     total_tokens = sum([res['total_tokens'] for res in results])
#     print(f"total input tokens: {total_input_tokens}")
#     print(f"total output tokens: {total_output_tokens}")
#     print(f"total tokens: {total_tokens}")

#     for i in range(len(results)):
#         res = results[i]
#         if res['error'] is not None:
#             raise Exception(f"Error: {res['error']}")
#         else:
#             parsed = answer_parser.parse_number(res['content'], valid_options=[1,2,3,4,5])
#             if parsed is None:
#                 raise Exception(f"Failed to parse response: {res['content']}")
#             else:
#                 dataset[i]['coverage_score'] = parsed

#     return dataset


# def build_relevance_prompt(context:str, question:str):
#     prompt = f"""
# <context>{context}</context>
# <question>{question}</question>

# Your task is to rate from 1 to 5 if the question is relevant to the context. Respond only with the number in [1, 2, 3, 4, 5].
# """
#     return prompt




# def build_relevance_scores(dataset: list[dict], remote:str, model:str) -> list[dict]:

#     # build the prompts
#     prompts = [build_relevance_prompt(d.get('context', ''), d['question']) for d in dataset]

#     model = SglModelAsync(remote=remote, model=model)
#     results, total_time = model.generate(prompts)
#     print(f"in total took: {total_time} seconds")
#     print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

#     total_input_tokens = sum([res['input_tokens'] for res in results])
#     total_output_tokens = sum([res['output_tokens'] for res in results])
#     total_tokens = sum([res['total_tokens'] for res in results])
#     print(f"total input tokens: {total_input_tokens}")
#     print(f"total output tokens: {total_output_tokens}")
#     print(f"total tokens: {total_tokens}")

#     for i in range(len(results)):
#         res = results[i]
#         if res['error'] is not None:
#             raise Exception(f"Error: {res['error']}")
#         else:
#             parsed = answer_parser.parse_number(res['content'], valid_options=[1,2,3,4,5])
#             if parsed is None:
#                 raise Exception(f"Failed to parse response: {res['content']}")
#             else:
#                 dataset[i]['relevance_score'] = parsed

#     return dataset
    

# def compute_relevance(dataset_fp, remote, model, force=False):
#     with open(dataset_fp, 'r') as f:
#         dataset = json.load(f)

#     if 'relevance_score' in dataset[0] and not force:
#         return

#     print(f"Computing relevance for {dataset_fp}")    
#     print("Dataset has %d contexts" % len(dataset))


#     dataset = build_relevance_scores(dataset, remote, model)

#     scores = [d['relevance_score'] for d in dataset]
#     print(f"Average relevance score: {np.mean(scores)}")

#     print(f"Saving {len(dataset)} questions to {dataset_fp}")
#     with open(dataset_fp, 'w') as f:
#         json.dump(dataset, f, indent=2)


# def compute_coverage(dataset_fp, remote, model, force=False):
#     with open(dataset_fp, 'r') as f:
#         dataset = json.load(f)

#     if 'coverage_score' in dataset[0] and not force:
#         return
    
#     print(f"Computing coverage for {dataset_fp}")
#     print("Dataset has %d contexts" % len(dataset))


#     dataset = build_coverage_scores(dataset, remote, model)

#     scores = [d['coverage_score'] for d in dataset]
#     print(f"Average coverage score: {np.mean(scores)}")

#     print(f"Saving {len(dataset)} questions to {dataset_fp}")
#     with open(dataset_fp, 'w') as f:
#         json.dump(dataset, f, indent=2)

    
# def compute_answerability(dataset_fp, remote, model, force=False):
#     with open(dataset_fp, 'r') as f:
#         dataset = json.load(f)

#     if 'answerability_score' in dataset[0] and not force:
#         return
#     print(f"Computing answerability for {dataset_fp}")
#     print("Dataset has %d contexts" % len(dataset))


#     dataset = build_answerabiltiy_scores(dataset, remote, model)

#     scores = [d['answerability_score'] for d in dataset]
#     print(f"Average answerability score: {np.mean(scores)}")

#     print(f"Saving {len(dataset)} questions to {dataset_fp}")
#     with open(dataset_fp, 'w') as f:
#         json.dump(dataset, f, indent=2)


# def compute_difficulty(dataset_fp, remote, model, force=False):
#     with open(dataset_fp, 'r') as f:
#         dataset = json.load(f)

#     if 'difficulty_score' in dataset[0] and not force:
#         return
#     print(f"Computing difficulty for {dataset_fp}")
#     print("Dataset has %d contexts" % len(dataset))


#     dataset = build_difficulty_scores(dataset, remote, model)

#     scores = [d['difficulty_score'] for d in dataset]
#     print(f"Average difficulty score: {np.mean(scores)}")

#     print(f"Saving {len(dataset)} questions to {dataset_fp}")
#     with open(dataset_fp, 'w') as f:
#         json.dump(dataset, f, indent=2)

# def compute_clarity(dataset_fp, remote, model, force=False):
#     with open(dataset_fp, 'r') as f:
#         dataset = json.load(f)

#     if 'clarity_score' in dataset[0] and not force:
#         return
    
#     print(f"Computing clarity for {dataset_fp}")
#     print("Dataset has %d contexts" % len(dataset))


#     dataset = build_clarity_scores(dataset, remote, model)

#     scores = [d['clarity_score'] for d in dataset]
#     print(f"Average clarity score: {np.mean(scores)}")

#     print(f"Saving {len(dataset)} questions to {dataset_fp}")
#     with open(dataset_fp, 'w') as f:
#         json.dump(dataset, f, indent=2)


def compute_meta_scores(dataset_fp, remote, model, force=False):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

    any_missing = False
    for d in dataset:
        if 'question_clarity_score' not in d or d['question_clarity_score'] is None:
            any_missing = True
        if 'question_difficulty_score' not in d or d['question_difficulty_score'] is None:
            any_missing = True
        if 'question_groundedness_score' not in d or d['question_groundedness_score'] is None:
            any_missing = True
    if not any_missing and not force:
        return
    
    print(f"Computing meta properties for {dataset_fp}")
    print("Dataset has %d contexts" % len(dataset))

    # build the prompts
    model_prompts = [prompts.META_PROPERTIES_PROMPT.format(context=d.get('context', ''), question=d['question'], answer=d['answer']) for d in dataset]

    model = SglModelAsync(remote=remote, model=model)
    results, total_time = model.generate(model_prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")

    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        else:
            parsed = answer_parser.parse_meta_properties_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
                dataset[i]['question_clarity_score'] = None  
                dataset[i]['question_difficulty_score'] = None
                dataset[i]['question_relevance_score'] = None
                dataset[i]['question_groundedness_score'] = None
            else:
                dataset[i]['question_clarity_score'] = parsed['clarity_score']
                dataset[i]['question_difficulty_score'] = parsed['difficulty_score']
                dataset[i]['question_groundedness_score'] = parsed['groundedness_score']


    scores = [d['question_clarity_score'] for d in dataset]
    print(f"Average question clarity score: {np.mean(scores)}")

    scores = [d['question_difficulty_score'] for d in dataset]
    print(f"Average question difficulty score: {np.mean(scores)}")

    scores = [d['question_groundedness_score'] for d in dataset]
    print(f"Average question groundedness score: {np.mean(scores)}")

    print(f"Saving {len(dataset)} questions to {dataset_fp}")
    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)



def evaluate_dataset_relevance_features(ifp, remote='opchat', model='Llama-4-Maverick-17B-128E-Instruct-FP8'):

    fns = []
    for fn in os.listdir(ifp):
        if os.path.isdir(os.path.join(ifp, fn)) and ('novel' in fn or 'reformat' in fn):
            # Look for json files within these directories
            dataset_dir = os.path.join(ifp, fn)
            for file in os.listdir(dataset_dir):
                if file.endswith('.jsonl'):
                    # fns.append(os.path.join(dataset_dir, file))
                    fns.append(file.replace('.jsonl', ''))
    fns.sort()
    if len(fns) == 0:
        print(f"evaluate_dataset_relevance_features: No datasets found in {ifp}")
        return
    print(f"evaluate_dataset_relevance_features: Found {len(fns)} datasets")
    print(f"evaluate_dataset_relevance_features: Datasets: {fns}")



    force_flag = False
    print(f"evaluate_dataset_relevance_features: Computing squishy statistics for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}/{fn}.jsonl"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")
        compute_meta_scores(dataset_fp, remote, model, force=force_flag)
    



    avg_difficulty_score = {}
    avg_clarity_score = {}
    avg_groundedness_score = {}
    for fn in fns:
        dataset_fp = f"{ifp}/{fn}/{fn}.jsonl"
        with open(dataset_fp, 'r') as f:
            data = json.load(f)

        invalid_idx = list()
        for i in range(len(data)):
            d = data[i]
            if d.get('question_groundedness_score') is None or not (1 <= d['question_groundedness_score'] <= 10):
                invalid_idx.append(i)
            if d.get('question_difficulty_score') is None or not (1 <= d['question_difficulty_score'] <= 10):
                invalid_idx.append(i)
            if d.get('question_clarity_score') is None or not (1 <= d['question_clarity_score'] <= 10):
                invalid_idx.append(i)
        
        # Print invalid data entries
        if invalid_idx:
            print(f"Found {len(invalid_idx)} invalid entries in dataset {fn}:")
            for idx in invalid_idx:
                print(f"  Entry {idx}:")
                print(f"    Question: {data[idx]['question']}")
                print(f"    Question Groundedness: {data[idx].get('question_groundedness_score', 'missing')}")
                print(f"    Question Difficulty: {data[idx].get('question_difficulty_score', 'missing')}")
                print(f"    Question Clarity: {data[idx].get('question_clarity_score', 'missing')}")
                print()

            raise Exception(f"Found {len(invalid_idx)} invalid entries in dataset {fn}")
        
        
        avg_difficulty_score[fn] = np.mean([d['question_difficulty_score'] for d in data])
        avg_clarity_score[fn] = np.mean([d['question_clarity_score'] for d in data])
        avg_groundedness_score[fn] = np.mean([d['question_groundedness_score'] for d in data])

        print(f"Dataset: {os.path.basename(fn)}")
        print(f"  Average question groundedness score: {avg_groundedness_score[fn]:.4f}")
        print(f"  Average question difficulty score: {avg_difficulty_score[fn]:.4f}")
        print(f"  Average question clarity score: {avg_clarity_score[fn]:.4f}")
        print()



if __name__ == '__main__':
    ifp = './data-subset-1k-p3-v1'
    evaluate_dataset_relevance_features(ifp)