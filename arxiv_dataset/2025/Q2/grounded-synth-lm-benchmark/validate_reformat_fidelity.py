import os
import json
import numpy as np
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, util

import prompts
import answer_parser
from model_interface import SglModelAsync




def compute_scores(dataset_fp, open_ended, remote, model, force=False):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

    any_missing = False
    for d in dataset:
        if 'reformat_question_similarity_score' not in d or d['reformat_question_similarity_score'] is None:
            any_missing = True
        if 'reformat_answer_similarity_score' not in d or d['reformat_answer_similarity_score'] is None:
            any_missing = True
    if not any_missing and not force:
        return
    
    for d in dataset:
        if 'orig_question' not in d or 'orig_answer' not in d:
            return
    
    print(f"Computing meta properties for {dataset_fp}")
    print("Dataset has %d contexts" % len(dataset))

    # build the prompts
    if open_ended:
        model_prompts = [prompts.REFORMAT_VALIDATION_PROMPT.format(context=d.get('context', ''), question1=d['orig_question'], answer1=d['orig_answer'], question2=d['question'], answer2=d['answer']) for d in dataset]
    else:
        model_prompts = [prompts.REFORMAT_VALIDATION_PROMPT.format(context=d.get('context', ''), question1=d['orig_question'], answer1=d['orig_answer'], question2=d['question'], answer2=d['choices'][d['answer']]) for d in dataset]

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
            parsed = answer_parser.parse_reformat_validity_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
                dataset[i]['reformat_question_similarity_score'] = None  
                dataset[i]['reformat_answer_similarity_score'] = None
            else:
                dataset[i]['reformat_question_similarity_score'] = parsed['question_similarity_score']
                dataset[i]['reformat_answer_similarity_score'] = parsed['answer_similarity_score']


    scores = [d['reformat_question_similarity_score'] for d in dataset]
    print(f"Average reformat question similarity score: {np.mean(scores)}")

    scores = [d['reformat_answer_similarity_score'] for d in dataset]
    print(f"Average reformat answer similarity score: {np.mean(scores)}")

    print(f"Saving {len(dataset)} questions to {dataset_fp}")
    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)


def compute_cosine_similarity(dataset_fp, open_ended, model=None, force=False):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

    any_missing = False
    for d in dataset:
        if 'reformat_question_cosine_similarity_to_orig' not in d or d['reformat_question_cosine_similarity_to_orig'] is None:
            any_missing = True
        if 'reformat_answer_cosine_similarity_to_orig' not in d or d['reformat_answer_cosine_similarity_to_orig'] is None:
            any_missing = True
    if not any_missing and not force:
        return
    
    for d in dataset:
        if 'orig_question' not in d or 'orig_answer' not in d:
            return
    
    print(f"Computing cosine similarity for reformatted questions for {dataset_fp}")

    if model is None:
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")

    questions = [d['question'] for d in dataset]
    orig_questions = [d['orig_question'] for d in dataset]
    if open_ended:
        answers = [d['answer'] for d in dataset]
    else:
        answers = [d['choices'][d['answer']] for d in dataset]
    orig_answers = [d['orig_answer'] for d in dataset]

    # Batch processing through GPU for better performance
    batch_size = 1024  # Adjust based on GPU memory
    embeddings_questions = []
    embeddings_orig_questions = []
    embeddings_answers = []
    embeddings_orig_answers = []
    
    # print('  Encoding contexts into embedding space using local GPU')
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings_questions.append(batch_embeddings.detach().cpu().numpy())
    
    # Concatenate all batches   
    embeddings_questions = np.concatenate(embeddings_questions, axis=0)

    for i in range(0, len(orig_questions), batch_size):
        batch = orig_questions[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings_orig_questions.append(batch_embeddings.detach().cpu().numpy())

    embeddings_orig_questions = np.concatenate(embeddings_orig_questions, axis=0)

    for i in range(0, len(answers), batch_size):
        batch = answers[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings_answers.append(batch_embeddings.detach().cpu().numpy())

    embeddings_answers = np.concatenate(embeddings_answers, axis=0)

    for i in range(0, len(orig_answers), batch_size):
        batch = orig_answers[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings_orig_answers.append(batch_embeddings.detach().cpu().numpy())

    embeddings_orig_answers = np.concatenate(embeddings_orig_answers, axis=0)

    # print('  Computing cosine similarity matrix')
    similarities_questions = util.pytorch_cos_sim(embeddings_questions, embeddings_orig_questions)
    similarities_questions = similarities_questions.cpu().numpy()

    similarities_answers = util.pytorch_cos_sim(embeddings_answers, embeddings_orig_answers)
    similarities_answers = similarities_answers.cpu().numpy()

    for i in range(len(dataset)):
        item = dataset[i]
        item['reformat_question_cosine_similarity_to_orig'] = float(similarities_questions[i,i])
        item['reformat_answer_cosine_similarity_to_orig'] = float(similarities_answers[i,i])

    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    






def validate_reformat_fidelity(ifp, open_ended=False, remote='opchat', model='Llama-4-Maverick-17B-128E-Instruct-FP8'):

    fns = []
    for fn in os.listdir(ifp):
        if os.path.isdir(os.path.join(ifp, fn)) and 'reformat' in fn:
            # Look for json files within these directories
            dataset_dir = os.path.join(ifp, fn)
            for file in os.listdir(dataset_dir):
                if file.endswith('.jsonl'):
                    # fns.append(os.path.join(dataset_dir, file))
                    fns.append(file.replace('.jsonl', ''))
    fns.sort()
    if len(fns) == 0:
        print(f"validate_reformat_fidelity: No datasets found in {ifp}")
        return
    print(f"validate_reformat_fidelity: Found {len(fns)} datasets")
    print(f"validate_reformat_fidelity: Datasets: {fns}")


    emd_model = SentenceTransformer('all-MiniLM-L6-v2') 
    # emd_model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    force_flag = True
    print(f"validate_reformat_fidelity: Validating reformat fidelity for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}/{fn}.jsonl"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")
        compute_cosine_similarity(dataset_fp, open_ended, model=emd_model, force=force_flag)
        compute_scores(dataset_fp, open_ended, remote, model, force=force_flag)
    



    avg_question_similarity_score = {}
    avg_answer_similarity_score = {}
    for fn in fns:
        if 'sec_qa_reformat' in fn:
            continue
        dataset_fp = f"{ifp}/{fn}/{fn}.jsonl"
        with open(dataset_fp, 'r') as f:
            data = json.load(f)

        invalid_idx = list()
        for i in range(len(data)):
            d = data[i]
            if d.get('reformat_question_similarity_score') is None or not (1 <= d['reformat_question_similarity_score'] <= 10):
                invalid_idx.append(i)
            if d.get('reformat_answer_similarity_score') is None or not (1 <= d['reformat_answer_similarity_score'] <= 10):
                invalid_idx.append(i)

        invalid_idx = list(set(invalid_idx))
        
        # Print invalid data entries
        if invalid_idx:
            print(f"Found {len(invalid_idx)} invalid entries in dataset {fn}:")
            for idx in invalid_idx: 
                print(f"  Entry {idx}:")
                print(f"    Question: {data[idx]['question']}")
                print(f"    Reformat Question Similarity: {data[idx].get('reformat_question_similarity_score', 'missing')}")
                print(f"    Reformat Answer Similarity: {data[idx].get('reformat_answer_similarity_score', 'missing')}")
                print()

            raise Exception(f"Found {len(invalid_idx)} invalid entries in dataset {fn}")
        
        
        avg_question_similarity_score[fn] = np.mean([d['reformat_question_similarity_score'] for d in data])
        avg_answer_similarity_score[fn] = np.mean([d['reformat_answer_similarity_score'] for d in data])

        print(f"Dataset: {os.path.basename(fn)}")
        print(f"  Average reformat question similarity score: {avg_question_similarity_score[fn]:.4f}")
        print(f"  Average reformat answer similarity score: {avg_answer_similarity_score[fn]:.4f}")
        print()



if __name__ == '__main__':
    ifp = './data-subset-1k-v0-L3-70B'
    validate_reformat_fidelity(ifp)







