#coding=utf8

from transformers import AutoTokenizer
from processor.util import trunc_string
import json

class MultiToOneLead(object):
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

        tmp_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.potential_model_path,
                    do_lower_case=True,
                    use_fast=True,
                    revision="main",
                    use_auth_token=False,
                    local_files_only=False)

        self.cls_tok = tmp_tokenizer.cls_token
        self.sep_tok = tmp_tokenizer.sep_token


    def process(self, path, src_path, tgt_path):
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        with open(path) as f:
            line = f.read().strip()
            json_obj = json.loads(line)

            for line in json_obj:
                flist = line.strip().split('\t')
                cluster_id = flist[0]
                summ_url = flist[1]
                pairs = flist[2]
                pair_obj = json.loads(pairs)

                docs = []; summs = []; sups = []
                sup_doc_num = len(pair_obj)-1
                if self.args.max_len_sup == -1:
                    max_sup_len = self.args.max_len_doc
                else:
                    max_sup_len = self.args.max_len_sup / sup_doc_num
                for pid in pair_obj:
                    pair = pair_obj[pid]
                    document = '\t'.join(pair['[DOCUMENT]'])
                    source = pair['[SORUCE]'].replace(':80', '')
                    if source.find('www.newser.com') > -1:
                        summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
                    else:
                        summary = '\t'.join(pair['[SUMMARY]'])
                    document, _ = trunc_string(document, 
                                            self.args.max_len_doc, 
                                            self.args.min_sentence_length,
                                            self.high_freq_src, 
                                            tokenizer=self.tokenizer)
                    sup_document, _ = trunc_string(document, 
                                            max_sup_len, 
                                            self.args.min_sentence_length,
                                            self.high_freq_src,
                                            tokenizer=self.tokenizer)
                    summary, _ = trunc_string(summary, 
                                            self.args.max_len_summ,
                                            self.args.min_sentence_length,
                                            self.high_freq_tgt)

                    if len(document.split('\t')) < self.args.min_doc_sent_num:
                        continue
                    if len(summary.split('\t')) < self.args.min_summ_sent_num:
                        continue
                    document = ' '.join(document.split('\t'))
                    summary = ' '.join(summary.split('\t'))
                    sup_document = ' '.join(sup_document.split('\t'))
                    docs.append(document)
                    summs.append(summary)
                    sups.append(sup_document)

                    if len(docs) == self.args.max_docs_in_cluster:
                        break

                if len(docs) < 2:
                    continue

                for i in range(len(docs)):
                    src, tgt = self._prepare_sup_docs(docs, summs, sups, i)
                    fpout_src.write(src+'\n')
                    fpout_tgt.write(tgt+'\n')

            fpout_src.close()
            fpout_tgt.close()

    def _prepare_sup_docs(self, docs, summs, sup_docs, idx):
        new_docs = [docs[idx]]
        for i in range(len(docs)):
            if i == idx:
                continue
            new_docs.append(sup_docs[i])
        return (' '+self.sep_tok+' ').join(new_docs), summs[idx]

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


