import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import similarity_filter



def eval_question_alignent(dataset_fp):
    if 'reformat' in dataset_fp:
        print("  reformat dataset detected, skipping")
        return

    

    metric_names = ['diversity_score_novel_questions', 'diversity_score_reformat_questions', 'diversity_score_orig_questions']

    novel_dataset_fp = dataset_fp
    novel_folders = []
    # List all items in the directory
    for item in os.listdir(dataset_fp):
        item_path = os.path.join(dataset_fp, item)
        # Check if it's a directory and has 'reformat' in the name
        if os.path.isdir(item_path) and 'novel' in item:
            novel_folders.append(item)

    # build the paired reformat names
    reformat_dataset_base_fp = os.path.dirname(dataset_fp)
    ds_base = os.path.basename(dataset_fp)
    toks = ds_base.split('-')
    toks.insert(-1, 'reformat')
    toks[-1] = 'L4M'  # only use the L4M model for reformat
    ds_base = '-'.join(toks)
    reformat_dataset_fp = os.path.join(reformat_dataset_base_fp, ds_base)
   

    reformat_folders = []
    for folder in novel_folders:
        reformat_folders.append(folder.replace('novel', 'reformat'))
    

    scores_dict = dict()
    if os.path.exists(os.path.join(dataset_fp, 'question_diversity_scores.json')):  
        with open(os.path.join(dataset_fp, 'question_diversity_scores.json'), 'r') as f:
            scores_dict = json.load(f)

    print("  loading embedding model")
    similarity_obj = similarity_filter.CosineSimilarity()
    
    for i in range(len(novel_folders)):
        
        novel_folder = novel_folders[i]
        base_fn = os.path.basename(novel_folder).replace('_novel', '')
        reformat_folder = reformat_folders[i]

        already_computed = 0
        if base_fn in scores_dict:
            for metric in metric_names:
                if metric in scores_dict[base_fn] and scores_dict[base_fn][metric] is not None and not np.isnan(scores_dict[base_fn][metric]):
                    already_computed += 1

        if already_computed == len(metric_names):
            print(f"Skipping {base_fn} because it is already computed")
            continue

        print(f"Processing {base_fn}")
        print("Computing question alignment for %s and %s" % (novel_folder, reformat_folder))
        cur_novel_dataset_fp = os.path.join(novel_dataset_fp, novel_folder, novel_folder + '.jsonl')
        cur_reformat_dataset_fp = os.path.join(reformat_dataset_fp, reformat_folder, reformat_folder + '.jsonl')
        ds_novel, ds_reformat, orig_score = compute_question_alignment(cur_novel_dataset_fp, cur_reformat_dataset_fp, similarity_obj)
        
        scores_dict[base_fn] = {
            'diversity_score_novel_questions': ds_novel,
            'diversity_score_reformat_questions': ds_reformat,
            'diversity_score_orig_questions': orig_score
        }

    
    with open(os.path.join(dataset_fp, 'question_diversity_scores.json'), 'w') as f:
        json.dump(scores_dict, f, indent=2)
    


def compute_question_alignment(novel_dataset_fp, reformat_dataset_fp, similarity_obj):
    print("Computing question alignment for %s and %s" % (novel_dataset_fp, reformat_dataset_fp))
    if not os.path.exists(novel_dataset_fp):
        print("  novel datasets does not exist")
        print(f"  novel_dataset_fp: {novel_dataset_fp}")
        return None, None, None
    
    
    with open(novel_dataset_fp, 'r') as f:
        novel_dataset = json.load(f)
    


    novel_questions = [d['question'] for d in novel_dataset]
    print("  novel_questions: ", len(novel_questions))
    novel_sim_mat = similarity_obj.get_similarity_matrix(novel_questions, novel_questions, mask_lower_triangular=True)
    mask = novel_sim_mat > 0.0
    novel_diversity_score = np.sum(novel_sim_mat[mask]) / np.sum(mask)

    if not os.path.exists(reformat_dataset_fp):
        print("  reformat datasets does not exist")
        print(f"  reformat_dataset_fp: {reformat_dataset_fp}")
        return novel_diversity_score, None, None

    with open(reformat_dataset_fp, 'r') as f:
        reformat_dataset = json.load(f)

    reformat_questions = [d['question'] for d in reformat_dataset]
    print("  reformat_questions: ", len(reformat_questions))
    reformat_sim_mat = similarity_obj.get_similarity_matrix(reformat_questions, reformat_questions, mask_lower_triangular=True)
    mask = reformat_sim_mat > 0.0
    reformat_diversity_score = np.sum(reformat_sim_mat[mask]) / np.sum(mask)

    if 'orig_question' in reformat_dataset[0]:
        orig_questions = [d['orig_question'] for d in reformat_dataset]
        print("  orig_questions: ", len(orig_questions))
        orig_sim_mat = similarity_obj.get_similarity_matrix(orig_questions, orig_questions, mask_lower_triangular=True)
        mask = orig_sim_mat > 0.0
        orig_diversity_score = np.sum(orig_sim_mat[mask]) / np.sum(mask)
    else:
        orig_diversity_score = None

    return novel_diversity_score, reformat_diversity_score, orig_diversity_score



