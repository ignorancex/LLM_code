#coding=utf8

from processor.util import trunc_string
import torch.nn as nn
import json
import torch
from transformers import AutoTokenizer

class OneToOne(object):
    def __init__(self, args, high_freq_src, high_freq_tgt):
        self.args = args

        root_dir = self.args.root_dir
        output_dir = self.args.output_dir
        dataset_name = self.args.dataset_name
        self.train_path = root_dir + '/train.json'
        self.train_src_path = output_dir+'/'+dataset_name+'_train_src.jsonl'
        self.train_tgt_path = output_dir+'/'+dataset_name+'_train_tgt.jsonl'

        self.dev_path = root_dir + '/dev.json'
        self.dev_src_path = output_dir+'/'+dataset_name+'_dev_src.jsonl'
        self.dev_tgt_path = output_dir+'/'+dataset_name+'_dev_tgt.jsonl'

        self.test_path = root_dir + '/test.json'
        self.test_src_path = output_dir+'/'+dataset_name+'_test_src.jsonl'
        self.test_tgt_path = output_dir+'/'+dataset_name+'_test_tgt.jsonl'

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


    def _read_one_line(self, line):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        main_docs = []; summs = []
        sup_doc_num = len(pair_obj)-1
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            main_doc, _ = trunc_string(document, 
                                    self.args.max_len_doc, 
                                    self.args.min_sentence_length,
                                    self.high_freq_src,
                                    tokenizer=self.tokenizer)
            summary, _ = trunc_string(summary, 
                                    self.args.max_len_summ,
                                    self.args.min_sentence_length,
                                    self.high_freq_tgt)
            if len(main_doc.split('\t')) < self.args.min_doc_sent_num:
                continue
            if len(summary.split('\t')) < self.args.min_summ_sent_num:
                continue
            main_docs.append(main_doc)
            summs.append(summary)
            if len(main_docs) == self.args.max_docs_in_cluster:
                break
        return main_docs, summs

    def process(self, path, src_path, tgt_path):
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        with open(path) as f:
            line = f.read().strip()
            json_obj = json.loads(line)
            for line in json_obj:
                main_docs, summs = self._read_one_line(line)
                if len(main_docs) < 2:
                    continue
                for i in range(len(main_docs)):
                    src = main_docs[i]
                    tgt = summs[i]
                    fpout_src.write(src+'\n')
                    fpout_tgt.write(tgt+'\n')
            fpout_src.close()
            fpout_tgt.close()

    def run(self):
        self.process(self.train_path,
                        self.train_src_path,
                        self.train_tgt_path)
        self.process(self.dev_path,
                        self.dev_src_path,
                        self.dev_tgt_path)
        self.process(self.test_path,
                        self.test_src_path,
                        self.test_tgt_path)


