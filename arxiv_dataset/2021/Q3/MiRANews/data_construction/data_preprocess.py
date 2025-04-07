#coding=utf8

from processor.util import preprocess_high_freq
from processor.process_unsupervised_select import UnsupervisedSelect
from processor.process_hier_unsupervised_select import HierUnsupervisedSelect
from processor.process_hier_one2one import HierOneToOne
from processor.process_hier_multi2one import HierMultiToOne
from processor.process_one2one import OneToOne
from processor.process_multi2one_lead import MultiToOneLead
from processor.process_from_gold_selection import GoldSelect
from processor.process_hier_from_gold_selection import HierGoldSelect
from processor.text_to_json import PreproTrainJson
from processor.rouge_select import RougeSelectGT

import argparse
import torch
import sys

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name", default='multi')
    parser.add_argument("-input_file", default='/scratch/xxu/multi-multi/multi_multi_clean.jsonl')
    parser.add_argument("-root_dir", default='/scratch/xxu/multi-multi/raw_data/')
    parser.add_argument("-black_list_dir", default='/scratch/xxu/multi-multi/raw_data/')
    parser.add_argument("-output_dir", default='/scratch/xxu/multi-multi/raw_data/')
    parser.add_argument("-tokenizer_model_path", default='')
    parser.add_argument("-potential_model_path", default='')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-mode", default='lead')
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-max_len_doc", default=500, type=int)
    parser.add_argument("-min_doc_sent_num", default=3, type=int)
    parser.add_argument("-max_len_paragraph", default=200, type=int) #for hier-transformer
    parser.add_argument("-max_num_paragraph", default=24, type=int) #for hier-transformer
    parser.add_argument('-max_len_sup', default=500, type=int)
    parser.add_argument('-max_len_summ', default=200, type=int)
    parser.add_argument("-min_summ_sent_num", default=1, type=int)
    parser.add_argument('-max_docs_in_cluster', default=5, type=int)
    parser.add_argument('-max_paragraph_in_cluster', default=50, type=int)
    parser.add_argument('-min_sentence_length', default=3, type=int)
    parser.add_argument('-max_sentence_length', default=500, type=int)
    parser.add_argument("-cls_tok", default='<s>')
    parser.add_argument("-sep_tok", default='</s>')
    parser.add_argument('-paraphrase_threshold_max', default=0.9, type=float)
    parser.add_argument('-coherent_threshold_max', default=0.6, type=float)
    parser.add_argument('-paraphrase_threshold_mean', default=0.9, type=float)
    parser.add_argument('-coherent_threshold_mean', default=0.6, type=float)
    parser.add_argument('-paraphrase_threshold_min', default=0.9, type=float)
    parser.add_argument('-coherent_threshold_min', default=0.6, type=float)

    parser.add_argument("-save_path", default='/scratch/xxu/multi-multi/json/')
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-test_shard_size", default=500, type=int)

    parser.add_argument("-top_k_per_sentence", default=5, type=int)
    parser.add_argument("-thred_num", default=30, type=int)

    parser.add_argument("-gt_input", default='')
    parser.add_argument("-gt_output_src", default='')
    parser.add_argument("-gt_output_tgt", default='')

    args = parser.parse_args()

    high_freq_src, high_freq_tgt = preprocess_high_freq(args.black_list_dir+'/train.json')
    if args.mode == 'hier_one_to_one':
        processor_obj = HierOneToOne(args, high_freq_src, high_freq_tgt)
    if args.mode == 'hier_multi_to_one':
        processor_obj = HierMultiToOne(args, high_freq_src, high_freq_tgt)
    if args.mode == 'hier_unsupervised_select':
        processor_obj = HierUnsupervisedSelect(args, high_freq_src, high_freq_tgt)
    if args.mode == 'hier_gold_select':
        processor_obj = HierGoldSelect(args, high_freq_src, high_freq_tgt)
    if args.mode == 'one_to_one':
        processor_obj = OneToOne(args, high_freq_src, high_freq_tgt)
    if args.mode == 'multi_to_one_lead':
        processor_obj = MultiToOneLead(args, high_freq_src, high_freq_tgt)
    if args.mode == 'unsupervised_select':
        processor_obj = UnsupervisedSelect(args, high_freq_src, high_freq_tgt)
    if args.mode == 'gold_select':
        processor_obj = GoldSelect(args, high_freq_src, high_freq_tgt)
    if args.mode == 'rouge_gt':
        processor_obj = RougeSelectGT(args, high_freq_src, high_freq_tgt)

    # Process
    if args.mode == 'rouge_gt':
        processor_obj.process()
    else:
        processor_obj.run()
    
    # PostProcess
    if args.mode not in ['hier_one_to_one', 
                         'hier_multi_to_one',
                         'hier_unsupervised_select',
                         'hier_gold_select',
                         'rouge_gt']:
        json_obj = PreproTrainJson(args)
        json_obj.preprocess()


