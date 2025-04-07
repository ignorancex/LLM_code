#coding=utf8

from transformers import (
        BartTokenizer, 
        BertTokenizer, 
        BertModel,
        AutoTokenizer
)
from processor.util import trunc_string
import torch.nn as nn
import json
import torch

class UnsupervisedSelect():
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

        # Tokenizer used in length controlling
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

        # Tokenizer used in the summarization model training 
        tmp_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.potential_model_path,
                    do_lower_case=True,
                    use_fast=True,
                    revision="main",
                    use_auth_token=False,
                    local_files_only=False)

        self.cls_tok = tmp_tokenizer.cls_token
        self.sep_tok = tmp_tokenizer.sep_token

        # For semantic similarity calculation
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.args.device)
        self.bert_model.eval()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _read_one_line(self, line):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        main_docs = []; summs = []; left_docs = []
        sup_doc_num = len(pair_obj)-1
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            main_doc, left_doc = trunc_string(document, 
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
            left_docs.append(left_doc)
            summs.append(summary)
            if len(main_docs) == self.args.max_docs_in_cluster:
                break
        return main_docs, left_docs, summs

    def _batch_emb(self, sents):
        idx = 0; emb_list = []
        while idx < len(sents):
            with torch.no_grad():
                tmp_sents = sents[idx:min(len(sents), idx+self.args.batch_size)]
                inputs = self.bert_tokenizer(tmp_sents, 
                                                padding=True, 
                                                truncation=True, 
                                                return_tensors="pt").to(self.args.device)
                outputs = self.bert_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                last_hidden_states = last_hidden_states[:,0,:].squeeze(dim=1)
                emb_list.append(last_hidden_states)
                idx += self.args.batch_size
        return torch.cat(emb_list)

    def _get_embeddings(self, main_docs, left_docs):
        # Sentence embedding
        main_doc_embs = []
        for doc in main_docs:
            sents = doc.split('\t')
            embs = self._batch_emb(sents)
            main_doc_embs.append(embs)
        left_doc_embs = []
        for doc in left_docs:
            sents = doc.split('\t')
            embs = self._batch_emb(sents)
            left_doc_embs.append(embs)
        return main_doc_embs, left_doc_embs

    def _cosin_sim(self, y1, y2):
        dim_1 = y1.size(0)
        dim_2 = y2.size(0)
        y1 = torch.repeat_interleave(y1, dim_2, dim=0)
        y2 = torch.cat(dim_1*[y2])
        cos_scores = self.cos(y1, y2)
        return torch.reshape(cos_scores, (-1, dim_2))

    def _rank_one_example(self, main_docs, left_docs, main_doc_embs, left_doc_embs, idx):
        sentences = []; embeddings = []
        curr_main_doc = main_docs[idx]
        curr_main_emb = main_doc_embs[idx]
        sentences.extend(left_docs[idx].split('\t'))
        embeddings.append(left_doc_embs[idx])
        for i in range(len(main_docs)):
            if i == idx:
                continue
            sentences.extend(main_docs[i].split('\t'))
            sentences.extend(left_docs[i].split('\t'))
            embeddings.append(main_doc_embs[i])
            embeddings.append(left_doc_embs[i])
        cand_sentence_emb = torch.cat(embeddings)
        cosin_sim_scores = self._cosin_sim(curr_main_emb, cand_sentence_emb)
        '''
        coherent_scores = cosin_sim_scores * (cosin_sim_scores > self.args.coherent_threshold)
        filterd_scores = coherent_scores + (-1000 * (coherent_scores >= self.args.paraphrase_threshold))
        sum_scores = torch.mean(filterd_scores, dim=0)
        sorted_scores, indices = torch.sort(sum_scores, descending=True)
        sorted_scores = sorted_scores.tolist()
        indices = indices.tolist()
        sorted_sentences = []
        for i in range(len(indices)):
            sent_id = indices[i]
            score = sorted_scores[i]
            if score > self.args.coherent_threshold:
                sorted_sentences.append(sentences[sent_id])
        '''

        max_scores = torch.max(cosin_sim_scores, dim=0)[0]
        mean_scores = torch.mean(cosin_sim_scores, dim=0)
        min_scores = torch.min(cosin_sim_scores, dim=0)[0]

        _, indices = torch.sort(mean_scores, descending=True)
        indices = indices.tolist()
        max_scores = max_scores.tolist()
        mean_scores = mean_scores.tolist()
        min_scores = min_scores.tolist()

        sorted_sentences = []
        for i in range(len(indices)):
            sent_id = indices[i]

            max_score = max_scores[sent_id]
            if max_score < self.args.coherent_threshold_max:
                continue
            if max_score > self.args.paraphrase_threshold_max:
                continue

            mean_score = mean_scores[sent_id]
            if mean_score < self.args.coherent_threshold_mean:
                continue
            if mean_score > self.args.paraphrase_threshold_mean:
                continue

            min_score = min_scores[sent_id]
            if min_score < self.args.coherent_threshold_min:
                continue
            if min_score > self.args.paraphrase_threshold_min:
                continue

            sorted_sentences.append(sentences[sent_id])

        sups = '\t'.join(sorted_sentences)
        return (curr_main_doc + ' ' + self.sep_tok + ' ' + sups).replace('\t', ' ')

    def process(self, path, src_path, tgt_path):
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        with open(path) as f:
            line = f.read().strip()
            json_obj = json.loads(line)
            for line in json_obj:
                try:
                    main_docs, left_docs, summs = self._read_one_line(line)
                    if len(main_docs) < 2:
                        continue
                    main_doc_embs, left_doc_embs = self._get_embeddings(main_docs, left_docs)
                    for i in range(len(main_docs)):
                        src = self._rank_one_example(main_docs, left_docs, main_doc_embs, left_doc_embs, i)
                        tgt = summs[i].replace('\t', ' ')
                        fpout_src.write(src+'\n')
                        fpout_tgt.write(tgt+'\n')
                except:
                    pass
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


