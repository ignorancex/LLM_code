import math
import time
from typing import Tuple, Dict, List
import os
import sys
import math
import numpy as np
import pandas as pd
import fingerprint_2d as fingerprint
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import shutil
import subprocess
from rdkit import DataStructs
from itertools import combinations
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm 
import ipdb

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist



def calculate_average_blosum62(seq1, seq2,tanimoto_matrix):
    """
    For two aligned sequences' two-dimensional fingerprint arrays, calculate the Tanimoto similarity between each pair of amino acid fingerprints, and then take the average value as the overall similarity between the sequences.
    
    parameters:
    - seq1: aligned first sequence, shape in [aligned length].
    - seq2: aligned second sequence, shape in [aligned length].
    - tanimoto_matrix: similarity among each 2 amino acids and gap
    
    return:
    - average_similarity: the average tanimoto simularity of all the position of a sequence pair
    - average_dissimilarity: the average tanimoto simularity of the position in difference of a sequence pair
    """
    total_similarity = 0
    dissimilarity = 0
    dis_pairs = 0
    
    num_pairs = len(seq1)  # the number of pairs
    
    for i in range(num_pairs):
    
        similarity = tanimoto_matrix.loc[seq1[i],seq2[i]]
        total_similarity += similarity
        
        if seq1[i] != seq2[i]:
            dissimilarity += similarity
            dis_pairs += 1
    
    average_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
    average_dissimilarity = dissimilarity / dis_pairs if dis_pairs > 0 else 0
    
    return average_similarity, average_dissimilarity

    
    
def Blosum62(seq1, seq2):
    
    normalized_blosum62 = pd.read_csv("./data/blosum62_normalized.csv",index_col=0)
    # ipdb.set_trace()
    average_similarity, average_dissimilarity = calculate_average_blosum62(seq1, seq2,normalized_blosum62)
    
    return average_similarity, average_dissimilarity
    
    
def levenstein_align(seq1, seq2):
    
    differences = sum(1 for base1, base2 in zip(seq1, seq2) if base1 != base2)
    
    mutations = []
    for pos, (base1, base2) in enumerate(zip(seq1, seq2), start=1):  
        if base1 != base2:
            mutations.append(f"({pos}, {base1}, {base2})")
    
    mutation_info = '|'.join(mutations)
    return differences, mutation_info
    
def smith_waterman(seq1, seq2, scoring_matrix=matlist.blosum62, gap_open=-11, gap_extend=-1, lambda_value=0.251, K=0.031):

    
    # Smith-Waterman for alignment
    alignments = pairwise2.align.localds(seq1, seq2, scoring_matrix, gap_open, gap_extend)
    
    if len(alignments): 
      # select the highest score result
      top_alignment = alignments[0]
    
      aligned_seq1, aligned_seq2, score, start, end = top_alignment
      
      # calculate the number of residues match the same position
      identical_residues = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
      aligned_columns = max(len(aligned_seq1), len(aligned_seq2))  # the number of aligned columns(include gaps)
      
      # calculate bit score using the new lambda and K values
      bit_score = (lambda_value * score - math.log(K)) / math.log(2)
      
      # calculate the index needed
      identity_ratio = identical_residues / aligned_columns
      
      # calculate bit score per column
      bit_score_per_column = bit_score / aligned_columns if aligned_columns else 0
      
      # print result
      print(f"Aligned Seq 1: {aligned_seq1}")
      print(f"Aligned Seq 2: {aligned_seq2}")
      print(f"Score: {score}")
      print(f"Identity Ratio: {identity_ratio:.4f}")
    
    else:
      identity_ratio = 0
      score = 0
      aligned_seq1, aligned_seq2 = None, None
      bit_score_per_column = 0
    # return results
    return identity_ratio, aligned_seq1, aligned_seq2, score, bit_score_per_column
    
    
  

def calculate_average_similarity(seq1, seq2,tanimoto_matrix):
    """
    For two aligned sequences' two-dimensional fingerprint arrays, calculate the Tanimoto similarity between each pair of amino acid fingerprints, and then take the average value as the overall similarity between the sequences.
    
    parameters:
    - seq1: aligned first sequence, shape in [aligned length].
    - seq2: aligned second sequence, shape in [aligned length].
    - tanimoto_matrix: similarity among each 2 amino acids and gap
    
    return:
    - average_similarity: the average tanimoto simularity of all the position of a sequence pair
    - average_dissimilarity: the average tanimoto simularity of the position in difference of a sequence pair
    """
    total_similarity = 0
    dissimilarity = 0
    dis_pairs = 0
    
    num_pairs = len(seq1)  # the number of pairs
    
    for i in range(num_pairs):
    
        similarity = tanimoto_matrix.loc[seq1[i],seq2[i]]
        total_similarity += similarity
        
        if seq1[i] != seq2[i]:
            dissimilarity += similarity
            dis_pairs += 1
    
    average_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
    average_dissimilarity = dissimilarity / dis_pairs if dis_pairs > 0 else 0
    
    return average_similarity, average_dissimilarity

    
    
def TanimotoSimilarity(seq1, seq2):
    
    tanimoto_matrix = pd.read_csv('./data/tanimoto.csv',index_col=0)
    
    # ipdb.set_trace()
    average_similarity, average_dissimilarity = calculate_average_similarity(seq1, seq2,tanimoto_matrix)
    
    return average_similarity, average_dissimilarity



def process_sequence_pair(seq_pair, condition, diff=5,similarity=0.9):
    idx1, seq1 = seq_pair[0]
    idx2, seq2 = seq_pair[1]
    
    lev = levenshtein_distance(seq1['Sequence'], seq2['Sequence'])
    
    seq_id, align1, align2, score, bit_score_per_column = smith_waterman(seq1['Sequence'], seq2['Sequence'])
    # '''
    if seq_id:
        sim,dissim = TanimotoSimilarity(align1, align2)
        avg_sim, avg_dis = Blosum62(align1, align2)
        leven_dist, mutation_info = levenstein_align(align1, align2)
    else:
        sim = 0
        dissim = 0
        avg_sim, avg_dis,leven_dist, mutation_info = 0, 0, None, None
    # '''
    # calculate the subtraction of activities
    dmic = seq1['Activity'] - seq2['Activity']
    activity_diff = np.abs(dmic)
    
    
    if condition == 'all':
        print('='*60)
        print('save the information of all pairs')
        print('This would take a long time')
        print('='*60)
        
        
        # add activity cliff pairs
        sequence1,mic1 = seq1['Sequence'],seq1['Activity'] 
        sequence2,mic2 = seq2['Sequence'],seq2['Activity']
        
        # return needed result
        return {
            'seq1': sequence1,
            'seq2': sequence2,
            'mic1': mic1,
            'mic2': mic2,
            'dmic': dmic, 
            'alignment seq1': align1,
            'alignment seq2': align2,
            'lev':lev,
            'seq id':seq_id,
            'score': score,
            'bit score': bit_score_per_column,
            'tanimoto simularity':sim,
            'tanimoto dissimularity':dissim,
            'blosum62 average':avg_sim,
            'blosum62 disaverage':avg_dis,
            'levenstein aligned':leven_dist,
            'mutation':mutation_info 
        }
        
        
    else:
      # judging whether save info
      save_flag = True
      
      if activity_diff <= np.log10(diff):
        save_flag = False
        
      else:
        if condition == 'levenstein aligned' and leven_dist != 1: 
          save_flag = False
        if condition == 'blosum62 average' and avg_sim < similarity:# 0.9 
          save_flag = False 
        if condition == 'tanimoto average' and sim < similarity:# 0.9 
          save_flag = False 
        if condition == 'levenstein' and lev != 1: 
          save_flag = False 
        if condition == 'sequence identity' and seq_id < similarity: 
          save_flag = False 
        
      if save_flag:
        # add activity cliff pairs
        sequence1,mic1 = seq1['Sequence'],seq1['Activity'] 
        sequence2,mic2 = seq2['Sequence'],seq2['Activity']
        
        # return needed result
        return {
            'seq1': sequence1,
            'seq2': sequence2,
            'mic1': mic1,
            'mic2': mic2,
            'dmic': dmic, 
            'alignment seq1': align1,
            'alignment seq2': align2,
            'lev':lev,
            'seq id':seq_id,
            'score': score,
            'bit score': bit_score_per_column,
            'tanimoto simularity':sim,
            'tanimoto dissimularity':dissim,
            'blosum62 average':avg_sim,
            'blosum62 disaverage':avg_dis,
            'levenstein aligned':leven_dist,
            'mutation':mutation_info 
        }
      else:
        return None
    
def main():
    args = parse_args()
    # read data
    name = args.data # './data/grampa_s_aureus_7_25.csv'
    data = pd.read_csv(args.data)
    
    condition = args.condition
    allowed_conditions = ["levenstein aligned","levenstein","blosum62 average","tanimoto average","sequence identity",'all']
    if condition not in allowed_conditions:
      print(f"Error: The condition '{condition}' is not allowed.")
      print(f"Allowed conditions are: {allowed_conditions}")
      sys.exit(1) 
      
    diff = args.diff
    similarity = args.threshold
    # create sequence pairs
    sequence_pairs = list(combinations(data.iterrows(), 2))
    
    # setting the size of thread pool
    pool_size = 128  # or adjust by demand
    
    new_rows = []
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        # batch submit sequence pairs
        future_to_pair = {executor.submit(process_sequence_pair, pair, condition, diff,similarity): pair for pair in sequence_pairs}
        
        for future in as_completed(future_to_pair):
            try:
                result = future.result()
                if result is not None:
                  new_rows.append(result)
            except Exception as e:
                print(f"Exception in processing pair: {future_to_pair[future]}, error: {e}")
    # '''
    # save result
    ac_sequences = pd.DataFrame(new_rows).drop_duplicates(subset=None, keep='first', inplace=False)
    ac_sequences.to_csv(name.replace('.csv',f'_acpairs_{condition}_{diff}-fold_sim-{similarity}.csv'), index=False)
    print(f'Finally saved {len(new_rows)} pairs')

def parse_args():
    parser = argparse.ArgumentParser(description="Generating AMP-Cliffs.")
    parser.add_argument('--condition','-c', type=str,default='blosum62 average', help='The condition to filter AMP-Cliff, supporting "levenstein aligned","levenstein","blosum62 average","tanimoto average","sequence identity","all"')
    parser.add_argument('--data','-d', type=str, default='./data/grampa_s_aureus_7_25.csv', help='Path to the data CSV file')
    parser.add_argument('--diff','-f', type=int, default=5, help='dilution difference,should be int like: 2,3,4,5')
    parser.add_argument('--threshold','-t', type=float, default=0.9, help='similarity threshold of blosum62 average and tanimoto average,should be float like: 0.85,0.9,0.95')
    return parser.parse_args()
if __name__=='__main__':
    main()
