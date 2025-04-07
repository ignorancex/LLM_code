#encoding=utf-8

import os
import sys
import time
import json
import random
sys.path.append('../CorrFA_for_Summarizaion/')
from Evaluation import Str2Srl
from Evaluation import Srl2Tree
from Evaluation import CWeighting
from Evaluation import Correlation
from check_fact_coverage import read_from_tree, doc_summary_cover, cut_by_length

SRL_ARCHIVE_PATH='../CorrFA_for_Summarizaion/Evaluation/srl-model-2018.05.25.tar.gz'
INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
CW_GOLD_PATH = './tmp_'+sys.argv[1]+'.txt'
TMP_CHECKPOINT_FILE = './tmp_'+sys.argv[1]+'.checkpoint'
MAX_DOCS_IN_CLUSTER = 5
MAX_LEN_SUMM = 200
MAX_LEN_DOC = 500

srl_obj = Str2Srl.Str2Srl(SRL_ARCHIVE_PATH)
tree_obj = Srl2Tree.Srl2Tree()
corr_obj = Correlation.Correlation()

def process(docs, summs):
    '''
    @@ Get SRL
    '''
    srl_res = srl_obj.annotation_process(docs, summs)

    '''
    @@ Get Tree
    '''
    tree_res = tree_obj.annotation_process(srl_res)
    doc_trees, gold_trees = read_from_tree(tree_res)

    '''
    @@ Get Content Weights
    '''
    coverage, _ = doc_summary_cover(doc_trees, gold_trees)
    coverage_scores = [coverage[tree] for tree in coverage]
    return sum(coverage_scores) / len(coverage_scores)

if __name__ == '__main__':
    #SAMPLE_NUM = 1000
    #src_path = '../../Factroid_Summarization/XSum/bbc-summary-no_structure/xsum_dev_src.jsonl'
    #tgt_path = '../../Factroid_Summarization/XSum/bbc-summary-no_structure/xsum_dev_tgt.jsonl'
    SAMPLE_NUM = -1
    src_path = 'tmp_src.txt'
    tgt_path = 'tmp_tgt.txt'
    srcs = [cut_by_length(line.strip(), MAX_LEN_DOC).lower() for line in open(src_path)]
    tgts = [cut_by_length(line.strip(), MAX_LEN_SUMM).lower() for line in open(tgt_path)]
    if SAMPLE_NUM > 0:
        pairs = list(zip(srcs, tgts))
        sampled_pairs = random.sample(pairs, SAMPLE_NUM)
        sampled_srcs = [item[0] for item in sampled_pairs]
        sampled_tgts = [item[1] for item in sampled_pairs]
    else:
        sampled_srcs = srcs
        sampled_tgts = tgts
    cov = process(sampled_srcs, sampled_tgts)
    print (cov)

