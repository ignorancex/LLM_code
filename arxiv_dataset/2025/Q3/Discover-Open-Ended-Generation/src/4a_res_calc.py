import os
import random
import csv
import tqdm
import argparse
import itertools
import wandb
import logging
from time import strftime
import sys
from llm_utils import *
from prettytable import PrettyTable
import pandas as pd
from datasets import load_dataset
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact, chi2_contingency
from natsort import natsorted
from sklearn.cluster import HDBSCAN


CACHE_DIR = '/scratch/jpan23'
os.environ['SENTENCE_TRANSFORMERS_HOME '] = CACHE_DIR

# from transformers import AutoTokenizer, Llama4ForConditionalGeneration, BitsAndBytesConfig

DATA_DIR = '../data'
PROMPT_DIR = 'prompt_instructions'
RES = '../SUMM_COMBINE_ALL'
T = strftime('%Y%m%d-%H%M')


def sig_test_chi2(phrases, in_tables):
    # phrases = [...]  # your phrase list
    p_values = []
    comp_tab = []

    # # Example: fill these with your real counts for each phrase
    # for phrase in phrases:
    #     # Calculate your a, b, c, d values for this phrase
    #     a = ...  # In-category docs with phrase
    #     b = ...  # In-category docs without phrase
    #     c = ...  # Out-category docs with phrase
    #     d = ...  # Out-category docs without phrase
    # for a, b , c, d in tables.values():

    for phrase in phrases:
        table = in_tables[phrase]
        print(f"Phrase: {phrase}, table={table}")
        # odds_ratio, p = fisher_exact(table)
        chi2, p, dof, expected = chi2_contingency(table)
        first_column = [(r[0], x[0]) for r, x in zip(table, expected)]
        p_values.append(p)
        comp_tab.append(first_column)

    # Now apply FDR correction (Benjamini-Hochberg)
    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # for i, phrase in enumerate(tables.keys()): # table check this 
    #     if rejected[i]:
    #         print(f"Phrase: {phrase}, FDR-corrected p={pvals_corrected[i]:.4g}, is significant.")

    return rejected, p_values
    # return p_values

# 1. collect phrases by location
def collect_phrases_by_location(processed_df, location_list):

    character_to_documents = defaultdict(list)

    filtered_df = processed_df[processed_df['location'].isin(location_list)]
    character_cols = [col for col in filtered_df.columns if col.startswith('character_')]
    phrases_cols = [col for col in filtered_df.columns if col.startswith('phrases_')]

    for _, row in filtered_df.iterrows():
        for char_col, phrases_col in zip(character_cols, phrases_cols):
            character = row[char_col]
            phrases = row[phrases_col]

            if pd.notna(character):
                if isinstance(phrases, str):
                    try:
                        phrases = ast.literal_eval(phrases)
                    except Exception:
                        print(f"Error parsing phrases for character {character}: {phrases}")
                        phrases = []
                
                if isinstance(phrases, list):
                    character_to_documents[character].append(phrases) # list of list
                    # character_to_documents[character].extend(phrases) # one document per character

    return character_to_documents

# 2. merge similar phrases [inside helper()]

# 3. Apply merged phrase mapping
def apply_phrase_mapping(documents, phrase_to_rep):
    new_docs = []
    for phrases in documents:
        new_doc = [phrase_to_rep[p] for p in phrases]
        new_docs.append(new_doc)
    return new_docs

def apply_phrase_mapping_filter(documents, phrase_to_rep, remove_phrases):
    new_docs = []
    # t1, t2 = 0, 0
    for phrases in documents:
        new_doc = []
        for p in phrases:
            new_p = phrase_to_rep[p]
            # if new_p in kept_phrases:
            #     new_doc.append(new_p)
            if new_p not in remove_phrases:
                new_doc.append(new_p)
        new_docs.append(new_doc)
    return new_docs


def calculate_phrase_tfidf(documents):
    N = 0
    cat_docs = []
    df = {}
    cat_df = []
    docs_per_cat = []
    chi2_contin_table = defaultdict(list)
    for doc in documents:
        N += len(doc)
        docs_per_cat.append(len(doc))
        temp = []
        df_temp = {}
        for phrase_list in doc:
            temp.extend(phrase_list)
            unique_terms = set(phrase_list)
            print("L206: ", len(unique_terms), len(phrase_list))  
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1
                df_temp[term] = df_temp.get(term, 0) + 1
                chi2_contin_table[term]
        cat_df.append(df_temp)
        cat_docs.append(temp)
    
    doc_scores = []
    fisher_table = []
    total_cat = len(cat_docs)
    chi2_t = []

    for i, doc in enumerate(cat_docs):
        tf = Counter(doc)
        total_terms = len(doc)

        tfidf = {}
        score = {}

        contin_tab = {}
        cat_len = len(cat_docs[i])

        for term in tf:
            tf_val = tf[term] / total_terms
            idf_val = math.log(N / df[term]) if df[term] > 0 else 0
            R_val = cat_df[i][term] / df[term] if df[term] > 0 else 0
            # score[term] = tf_val * idf_val * R_val
            # score[term] = R_val
            tfidf[term] = tf_val * idf_val

            # fors stats test
            if total_cat == 2:
                a = cat_df[i][term] # in cat, docs with phrases
                b = docs_per_cat[i] - a # in cat, docs without phrase
                c = df[term] - a # not in cat, docs with phrase
                d = N - (docs_per_cat[i] + df[term] - a) # not in cat, docs without phrase
            else:
                cat_df_t_all = []
                for j in range(total_cat):
                    cat_df_t_all.append(cat_df[j][term] if term in cat_df[j] else 0)

                min_cat_df_t = min(cat_df_t_all)
                a = cat_df[i][term] # in cat, docs with phrases
                b = docs_per_cat[i] - a # in cat, docs without phrase
                # c = df[term] - a # not in cat, docs with phrase
                # d = N - (docs_per_cat[i] + df[term] - a) # not in cat, docs without phrase

                ### multiple category
                c = min_cat_df_t if min_cat_df_t < a else a
                d = N - (docs_per_cat[i] + c) # not in cat, docs without phrase

                print("L216: ", min_cat_df_t, df[term] - a, df[term], a)

            # score[term] = R_val * (a - c) / N

            score[term] = (a - c) / cat_len ###

            contin_tab[term] = [a, b, c, d]

            chi2_contin_table[term].append([a, b])  # in cat, docs with/without phrase

        for term_not_in_cat in chi2_contin_table.keys():
            if term_not_in_cat not in tf:
                chi2_contin_table[term_not_in_cat].append([0, docs_per_cat[i]])

        # doc_scores.append(tfidf)
        doc_scores.append(score)
        fisher_table.append(contin_tab)

    return doc_scores, fisher_table, chi2_contin_table



def helper():
    # data = args.dataset

    if "gpt" in args.model_name:
        perspectiveModel = ChatGPT(args.model_name, temperature=args.temperature, verbose=args.verbose)
    elif "Qwen3" in args.model_name:
        perspectiveModel = QWEN3(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Qwen" in args.model_name:
        perspectiveModel = QWEN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Mistral" in args.model_name:
        perspectiveModel = MISTRAL(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama-4" in args.model_name:
        perspectiveModel = LLM4(args.model_name, temperature=args.temperature, quanti=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name and 'Vision' not in args.model_name:
        perspectiveModel = LLM(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name and 'Vision' in args.model_name:
        perspectiveModel = LLM3_2_V(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "command" in args.model_name:
        perspectiveModel = COHERE(args.model_name, temperature=args.temperature, verbose=args.verbose)
    else:
        perspectiveModel = SentenceTransformer('all-mpnet-base-v2', cache_folder=CACHE_DIR) # all-MiniLM-L6-v2
        args.model_name = 'sentence-transformer'
        
    ### 2. find similar phrases via a LLM or sentence transformer
    def merge_similar_phrases(phrases, threshold=0.75):
        if len(phrases) == 0:
            return {}

        if args.model_name == 'sentence-transformer':
            embeddings = perspectiveModel.encode(phrases, convert_to_tensor=True)
            similarity_matrix = perspectiveModel.similarity(embeddings, embeddings)
        else:
            embeddings = perspectiveModel.encode_phrases(phrases)
            similarity_matrix = util.cos_sim(embeddings, embeddings)

        representative_map = {}
        similar_phrase_map = defaultdict(list)
        visited = set()

        for i in range(len(phrases)):
            if i in visited:
                continue
            representative_map[phrases[i]] = phrases[i]  # itself
            similar_phrase_map[phrases[i]].append(phrases[i])
            visited.add(i)
            for j in range(i + 1, len(phrases)):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    representative_map[phrases[j]] = phrases[i]
                    similar_phrase_map[phrases[i]].append(phrases[j])
                    visited.add(j)

        return representative_map, similar_phrase_map  # phrase -> representative phrase

    def hdbscan_cluster_merge(
        phrases,
        min_cluster_size=2,
        handle_noise=True
    ):
        # Step 1: Embed the phrases
        if args.model_name == 'sentence-transformer':
            embeddings = perspectiveModel.encode(phrases, convert_to_tensor=True)
        else:
            embeddings = perspectiveModel.encode_phrases(phrases)

        # Step 2: Run HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        embeddings = embeddings.detach().cpu().numpy() 
        labels = clusterer.fit_predict(embeddings)

        # Step 3: Group phrases/embeddings by cluster id
        cluster_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Ignore noise for now
                cluster_to_indices[label].append(idx)
        
        representative_map = {}
        similar_phrase_map = {}

        # Step 4: For each cluster, pick a representative (closest to centroid)
        for cluster_id, indices in cluster_to_indices.items():
            cluster_phrases = [phrases[i] for i in indices]
            cluster_embeds = np.array([embeddings[i] for i in indices])
            centroid = np.mean(cluster_embeds, axis=0)
            distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
            rep_idx_in_cluster = np.argmin(distances)
            rep_idx = indices[rep_idx_in_cluster]
            representative = phrases[rep_idx]
            for i in indices:
                representative_map[phrases[i]] = representative
            similar_phrase_map[representative] = cluster_phrases

        # Step 5: Optionally handle noise points (label == -1)
        if handle_noise:
            for idx, label in enumerate(labels):
                if label == -1:
                    representative_map[phrases[idx]] = phrases[idx]
                    similar_phrase_map[phrases[idx]] = [phrases[idx]]

        return representative_map, similar_phrase_map, labels
    
    ###

    table = PrettyTable() ### table for use?
    
    place = json.load(open("../data/places.json"))

    place_list = list(place.keys())

    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING WITH      ")
    print("------------------------")

    logging.info("\n------------------------")
    logging.info("    EVALUATING WITH      ")
    logging.info("------------------------")
    print(f"MODEL: {args.model_name}")
    # print(f"CAT: {cat}")
    logging.info(f"MODEL: {args.model_name}")
    # logging.info(f"CAT: {cat}")
    print("------------------------\n")

    # logging.info(f"NUM PROBS: {args.num_probs}")
    logging.info("------------------------\n")

    # csv_name = os.path.join(DATA_DIR, 'test_data.csv')  # test_data
    # can add categories later
    if args.info_ == 'one':

        filename = '' # one no des

        if args.condition == 'gender':    
            filename = ''
        
        if args.condition == 'race':
            filename = ''
        
        if args.condition == 'religions':
            filename = ''


    if args.info_ == 'two1':
        if args.condition == 'gender':
            # gender
            if args.add_in == 'ob':
                filename = '' # ob 20

            if args.add_in == 'base':
                filename = '' # cb 20
            
            if args.add_in == 'simple2':
                filename = ''
            
            if args.add_in == 'simple3':
                filename = ''

            if args.add_in == 'llama3_2_11b':
                filename = ''

            if args.add_in == 'llama2_7b':
                filename = ''

            if args.add_in == 'qwen3_8b':
                filename = ''
            
        elif args.condition == 'race':
            # race
            if args.add_in == 'base':
                # cb base
                filename = ''
            
            if args.add_in == 'simple2':
                filename = ''
            
            if args.add_in == 'simple3':
                filename = ''

            if args.add_in == 'llama3_2_11b':
                filename = ''

            if args.add_in == 'qwen3_8b':
                filename = ''

            if args.add_in == 'ob':
                # ob
                filename = ''
        elif args.condition == 'religions':
            # religions
            if args.add_in == 'base':
                # cb base
                filename = ''
            
            if args.add_in == 'simple2':
                filename = ''
            
            if args.add_in == 'simple3':
                filename = ''

            if args.add_in == 'llama3_2_11b':
                filename = ''

            if args.add_in == 'qwen3_8b':
                filename = ''

            if args.add_in == 'ob':
                # ob
                filename = ''

    csv_name = os.path.join(RES, filename)

    match = re.search(r'_(.*?)_all_summ', filename)
    model_name_ori = match.group(1)

    processed_df = pd.read_csv(csv_name)

    processed_rows = []
    
    for idx, p in enumerate(tqdm.tqdm(place_list)):
        location_list = place[p]['places']

        character_to_documents = collect_phrases_by_location(processed_df, location_list)

        row_data = {"location cat": p}

        all_phrases = []
        character_to_merged_docs = {}

        # 1. Merge similar phrases globally across all characters
        all_phrases = [phrase for docs in character_to_documents.values() for doc in docs for phrase in doc]

        if len(all_phrases) == 0:
            continue

        # 2. Merge similar phrases globally
        if args.metric == 'sim':
            phrase_to_rep, similar_phrase_map = merge_similar_phrases(all_phrases, threshold=0.83)
        if args.metric == 'hdbscan':
            phrase_to_rep, similar_phrase_map, _ = hdbscan_cluster_merge(all_phrases)

        # 3. Apply phrase mapping per character
        character_to_flat_doc = {}
        for character, docs in character_to_documents.items():
            mapped = apply_phrase_mapping(docs, phrase_to_rep)
            flat_phrases = [p for doc in mapped for p in doc]
            character_to_flat_doc[character] = flat_phrases
            character_to_merged_docs[character] = mapped

        # 4. Prepare documents (1 per character) for TF-IDF
        characters = list(character_to_merged_docs.keys())
        documents = [character_to_merged_docs[c] for c in characters]

        # 5. Compute TF-IDF on all merged documents
        per_char_scores, per_char_tables, chi2_table = calculate_phrase_tfidf(documents)


        # for character, scores, tables in zip(characters, per_char_scores, per_char_tables):
        for character, scores in zip(characters, per_char_scores):
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            all_vals = list(scores.values())
            unique_vals = sorted(set(all_vals))
            # smallest_vals = unique_vals[:2]

            phrases_sorted = [p for p, s in sorted_items if s >= 0]
            scores_sorted = [s for _, s in sorted_items if s >= 0]

            # rejected, pvals_corrected = sig_test(phrases_sorted, tables)
            rejected, pvals_corrected = sig_test_chi2(phrases_sorted, chi2_table)
            # pvals_corrected = sig_test_chi2(phrases_sorted, chi2_table)


            row_data[character] = phrases_sorted
            row_data[character + "_score"] = scores_sorted
            row_data[character + "_sig"] = rejected
            row_data[character + "_pvals"] = pvals_corrected
            
        # row_data["phrase_mapping"] = similar_phrase_map ### combine similar phrases all dictionary
        processed_rows.append(row_data)

        # return character_final_phrases
    
    p_df = pd.DataFrame(processed_rows)

    concept_dir = os.path.join('../SUMM_COMBINE_REMOVE_DUP_&CALC')
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    # T = strftime('%Y%m%d-%H%M')
    temp = str(args.temperature)

    # save_concepts_dir = os.path.join(concept_dir, args.T + '_' + args.info_ + '_' + model_name_ori + '_' + args.metric + '_' + args.condition + '_' + args.add_in + '_all_summ.csv')
    save_concepts_dir = os.path.join(concept_dir, args.info_ + '_' + model_name_ori + '_' + args.metric + '_' + args.condition + '_' + args.add_in + '_all_summ.csv')

    # save_concepts_dir = os.path.join(concept_dir, args.info_ + '_' + model_name_ori + '_all_summ_' + args.cat + '.csv')
    p_df.to_csv(save_concepts_dir, index = False, header=True)

    print("\n------------------------")
    print("         COMPLETE        ")
    print("------------------------")



def main():
    ### add cat
    # python evaluate.py --model_name=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset=bbq --num_probs=10
    parser = argparse.ArgumentParser()
    # parser.add_argument('--condition', type=str, default='disambig') # disambig, ambig
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--temperature', type=float, default=0.3)
    # parser.add_argument('--num_probs', '-n', type=int, default=200)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit? and quantized for llama 4?
    # parser.add_argument('--dataset', type=str, default='crows') # choose from crows, stereo_intra, stereo_inter, bbq_<category>
    parser.add_argument('--info_', type=str, default='one') # one, two1, two2
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--metric', type=str, default='sim') # sim, hdbscan
    parser.add_argument('--condition', type=str, default ="gender") # 
    parser.add_argument('--add_in', type=str, default='') # all
    
    global args
    args = parser.parse_args()

    log_dir = os.path.join('../logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


    # for _ in range(10):
    args.T = strftime('%Y%m%d-%H%M')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    # log_file = '../logs/' + args.T + '_' + args.dataset + '_output_results.log'
    log_file = '../logs/' + args.T + '_output_results.log'
    if log_file is None:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

    
    # logging.basicConfig(filename='../logs/' + t + '_output.log', level=logging.INFO)
    logging.info(args)
    print(args)

    helper()

    logging.shutdown()

if __name__ == '__main__':
    main()