#coding=utf8

from processor.util import trunc_string, split_paragraph
import torch.nn as nn
import json
import torch
from transformers import AutoTokenizer

class HierGoldSelect(object):
    def __init__(self, args, high_freq_src, high_freq_tgt):
        self.args = args

        root_dir = self.args.root_dir
        output_dir = self.args.output_dir
        dataset_name = self.args.dataset_name
        self.train_src_input = root_dir+'/'+dataset_name+'_train_src.jsonl'
        self.train_tgt_input = root_dir+'/'+dataset_name+'_train_tgt.jsonl'
        self.train_src_output = output_dir+'/'+dataset_name+'_train_src.jsonl'
        self.train_tgt_output = output_dir+'/'+dataset_name+'_train_tgt.jsonl'

        self.dev_src_input = root_dir+'/'+dataset_name+'_dev_src.jsonl'
        self.dev_tgt_input = root_dir+'/'+dataset_name+'_dev_tgt.jsonl'
        self.dev_src_output = output_dir+'/'+dataset_name+'_dev_src.jsonl'
        self.dev_tgt_output = output_dir+'/'+dataset_name+'_dev_tgt.jsonl'

        self.test_src_input = root_dir+'/'+dataset_name+'_test_src.jsonl'
        self.test_tgt_input = root_dir+'/'+dataset_name+'_test_tgt.jsonl'
        self.test_src_output = output_dir+'/'+dataset_name+'_test_src.jsonl'
        self.test_tgt_output = output_dir+'/'+dataset_name+'_test_tgt.jsonl'

        self.high_freq_src = high_freq_src
        self.high_freq_tgt = high_freq_tgt

        if self.args.tokenizer_model_path == '':
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.tokenizer_model_path,
                    do_lower_case=True,
                    use_fast=True,
                    revision="main",
                    use_auth_token=False,
                    local_files_only=False)

    def _get_selected_sentences(self, sentences, labels, doc_ids):
        selected_sentences = []
        for i, sentence in enumerate(sentences):
            if labels[i] == 1 and doc_ids[i] != 0:
                selected_sentences.append(sentence)
        return selected_sentences

    def _get_main_document(self, sentences, doc_ids):
        main_doc = []
        for i, sentence in enumerate(sentences):
            if doc_ids[i] == 0:
                main_doc.append(sentence)
        return main_doc

    def process(self, src_in_path, tgt_in_path, src_out_path, tgt_out_path):
        fpout_src = open(src_out_path, 'w')
        for line in open(src_in_path):
            json_obj = json.loads(line.strip())
            sentences = json_obj['docs']
            doc_ids = json_obj['doc_ids']
            labels = json_obj['labels']
            selected_sentences = self._get_selected_sentences(sentences, labels, doc_ids)
            main_doc = self._get_main_document(sentences, doc_ids)
            main_doc = split_paragraph('\t'.join(main_doc),
                                    self.args.max_len_paragraph,
                                    self.args.max_num_paragraph,
                                    self.args.min_sentence_length,
                                    self.high_freq_src,
                                    tokenizer=self.tokenizer)
            selected_sents = split_paragraph('\t'.join(selected_sentences),
                                    self.args.max_len_paragraph,
                                    self.args.max_num_paragraph,
                                    self.args.min_sentence_length,
                                    self.high_freq_src,
                                    tokenizer=self.tokenizer)
            if len(selected_sents) > 0:
                selected_sents[-1][-1] += ' <SUPP_START>'
                src = selected_sents + main_doc
            else:
                main_doc[0][0] = '<SUPP_START> ' + main_doc[0][0]
            fpout_src.write(json.dumps(src)+'\n')

        fpout_tgt = open(tgt_out_path, 'w')
        for line in open(tgt_in_path):
            summary, _ = trunc_string(line.strip(), 
                                    self.args.max_len_summ,
                                    self.args.min_sentence_length,
                                    self.high_freq_tgt)
            fpout_tgt.write(' '.join(summary.split('\t'))+'\n')

        fpout_src.close()
        fpout_tgt.close()

    def run(self):
        self.process(self.train_src_input,
                     self.train_tgt_input,
                     self.train_src_output,
                     self.train_tgt_output)
        self.process(self.dev_src_input,
                     self.dev_tgt_input,
                     self.dev_src_output,
                     self.dev_tgt_output)
        self.process(self.test_src_input,
                     self.test_tgt_input,
                     self.test_src_output,
                     self.test_tgt_output)


