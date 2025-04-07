#coding=utf8

from transformers import (
        BartTokenizer, 
        BertTokenizer, 
        BertModel,
        AutoTokenizer
)
import torch.nn as nn
import json
import torch

class UnsupervisedThreshold():
    def __init__(self, device, data_path):

        self.data_path = data_path
        self.device = device
        self.batch_size = 64

        # For semantic similarity calculation
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(device)
        self.bert_model.eval()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _batch_emb(self, sents):
        idx = 0; emb_list = []
        while idx < len(sents):
            with torch.no_grad():
                tmp_sents = sents[idx:min(len(sents), idx+self.batch_size)]
                inputs = self.bert_tokenizer(tmp_sents, 
                                                padding=True, 
                                                truncation=True, 
                                                return_tensors="pt").to(self.device)
                outputs = self.bert_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                last_hidden_states = last_hidden_states[:,0,:].squeeze(dim=1)
                emb_list.append(last_hidden_states)
                idx += self.batch_size
        return torch.cat(emb_list)

    def _get_embeddings(self, main_docs, left_docs):
        # Sentence embedding
        main_doc_embs = self._batch_emb(main_docs)
        left_doc_embs = self._batch_emb(left_docs)
        return main_doc_embs, left_doc_embs

    def _cosin_sim(self, y1, y2):
        dim_1 = y1.size(0)
        dim_2 = y2.size(0)
        y1 = torch.repeat_interleave(y1, dim_2, dim=0)
        y2 = torch.cat(dim_1*[y2])
        cos_scores = self.cos(y1, y2)
        return torch.reshape(cos_scores, (-1, dim_2))

    def _calculate_one_example(self, main_embs, assist_embs):
        cosin_sim_scores = self._cosin_sim(main_embs, assist_embs)
        mean_scores = torch.mean(cosin_sim_scores, dim=0)
        max_scores = torch.max(cosin_sim_scores, dim=0)[0]
        min_scores = torch.min(cosin_sim_scores, dim=0)[0]
        return mean_scores, max_scores, min_scores

    def _get_main_document(self, sentences, doc_ids):
        main_doc = []
        for i, sentence in enumerate(sentences):
            if doc_ids[i] == 0:
                main_doc.append(sentence)
        return main_doc

    def _get_selected_sentences(self, sentences, labels, doc_ids):
        selected_sentences = []
        for i, sentence in enumerate(sentences):
            if labels[i] == 1 and doc_ids[i] != 0:
                selected_sentences.append(sentence)
        return selected_sentences

    def _get_unselected_sentences(self, sentences, labels, doc_ids):
        selected_sentences = []
        for i, sentence in enumerate(sentences):
            if labels[i] == 0 and doc_ids[i] != 0:
                selected_sentences.append(sentence)
        return selected_sentences

    def process(self):
        selected_mean = []; unselected_mean = []
        selected_max = []; unselected_max = []
        selected_min = []; unselected_min = []
        
        for line in open(self.data_path):
            try:
                line = line.strip()
                json_obj = json.loads(line)
                docs = json_obj["docs"]
                doc_ids = json_obj["doc_ids"]
                selections = json_obj["labels"]

                main_doc = self._get_main_document(docs, doc_ids)
                selected_assist = self._get_selected_sentences(docs, selections, doc_ids)
                unselected_assist = self._get_unselected_sentences(docs, selections, doc_ids)
                if len(selected_assist) == 0 or len(unselected_assist) == 0:
                    continue

                main_doc_embs = self._batch_emb(main_doc)
                assist_embs = self._batch_emb(selected_assist)
                unrelated_embs = self._batch_emb(unselected_assist)

                mean_scores, max_scores, min_scores = self._calculate_one_example(main_doc_embs, assist_embs)
                selected_mean.extend(mean_scores.tolist())
                selected_max.extend(max_scores.tolist())
                selected_min.extend(min_scores.tolist())

                mean_scores, max_scores, min_scores = self._calculate_one_example(main_doc_embs, unrelated_embs)
                unselected_mean.extend(mean_scores.tolist())
                unselected_max.extend(max_scores.tolist())
                unselected_min.extend(min_scores.tolist())
            except:
                pass

        fpout = open('../tmp/tmp.mean.select.json', 'w')
        fpout.write(json.dumps(selected_mean))
        fpout.close()

        fpout = open('../tmp/tmp.max.select.json', 'w')
        fpout.write(json.dumps(selected_max))
        fpout.close()

        fpout = open('../tmp/tmp.min.select.json', 'w')
        fpout.write(json.dumps(selected_min))
        fpout.close()

        fpout = open('../tmp/tmp.mean.unselect.json', 'w')
        fpout.write(json.dumps(unselected_mean))
        fpout.close()

        fpout = open('../tmp/tmp.max.unselect.json', 'w')
        fpout.write(json.dumps(unselected_max))
        fpout.close()

        fpout = open('../tmp/tmp.min.unselect.json', 'w')
        fpout.write(json.dumps(unselected_min))
        fpout.close()


if __name__ == '__main__':
    input_path = '/scratch/xxu/multi-multi/supervised_content_labels/multi_train_src.jsonl'
    obj = UnsupervisedThreshold(3, input_path)
    obj.process()

