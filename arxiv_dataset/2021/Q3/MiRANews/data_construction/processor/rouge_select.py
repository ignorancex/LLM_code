#coding=utf8

from processor.util import trunc_string
import torch.nn as nn
import json
import torch
import rouge
import numpy
from transformers import AutoTokenizer

class RougeSelectGT(object):
    def __init__(self, args, high_freq_src, high_freq_tgt):
        self.args = args

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

        self.evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)


    def _read_one_line(self, line):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        main_docs = []; summs = []
        sup_doc_num = len(pair_obj)-1
        max_len_per_doc = self.args.max_len_doc/len(pair_obj)
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            main_doc, _ = trunc_string(document, 
                                    max_len_per_doc, 
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

    def rouge_selector(self, docs, summary, curr_main_id):
        doc_cluster = []
        doc_cluster.append(docs[curr_main_id])
        for i in range(len(docs)):
            if i == curr_main_id:
                continue
            doc_cluster.append(docs[i])
        docs = doc_cluster

        summary_sentences = summary.split('\t')
        doc_sentences = []; doc_id = []
        scores_for_each_summ_sentence = []
        for i, doc in enumerate(docs):
            sentences = doc.split('\t')
            doc_sentences.extend(sentences)
            if i == 0:
                doc_id.extend([0]*len(sentences))
            else:
                doc_id.extend([1]*len(sentences))
        selection_label = [0] * len(doc_sentences)

        for summ in summary_sentences:
            one_summ_scores = []
            for i, doc in enumerate(docs):
                sentences = doc.split('\t')
                rouge_score = [0.0] * len(sentences)
                references = [summ] * len(sentences)
                all_hypothesis = sentences
                all_references = references
                scores = self.evaluator.get_scores(all_hypothesis, all_references)
                for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                    for hypothesis_id, results_per_ref in enumerate(results):
                        nb_references = len(results_per_ref['f'])
                        for reference_id in range(nb_references):
                            rouge_score[hypothesis_id] += results_per_ref['f'][reference_id]
                rouge_score = [s/3 for s in rouge_score]
                one_summ_scores.extend(rouge_score)
            scores_for_each_summ_sentence.append(one_summ_scores)

        for i, summ_scores in enumerate(scores_for_each_summ_sentence):
            idxs = numpy.argsort(summ_scores).tolist()[-self.args.top_k_per_sentence:]
            for idx in idxs:
                selection_label[idx] |= 1
        return doc_sentences, doc_id, selection_label

    def process(self):
        src_path=self.args.gt_output_src
        tgt_path=self.args.gt_output_tgt
        path=self.args.gt_input

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
                    doc_sentences, doc_id, selection_label = self.rouge_selector(main_docs, summs[i], i)
                    src = {'docs':doc_sentences, 'doc_ids':doc_id, 'labels':selection_label}
                    tgt = summs[i]
                    fpout_src.write(json.dumps(src)+'\n')
                    fpout_tgt.write(tgt+'\n')
            fpout_src.close()
            fpout_tgt.close()

