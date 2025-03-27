import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer

from h02_bert_embeddings.bert_per_word import BertPerWordModel
from utils import constants
# from utils import utils
# from .ud import UdProcessor


class BertProcessor:
    # pylint: disable=arguments-differ
    name = 'bert'

    def __init__(self, bert_option, tgt_words=None):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_option)
        self.model = self.load_bert_model(bert_option)
        self.tgt_words = tgt_words

        self.pad_id = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')

    @staticmethod
    def load_bert_model(bert_option):
        model = BertPerWordModel(bert_option)
        model.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model.to(device=constants.device)

    def process_batch(self, batch, embeddings_dict):
        batch, all_bert_tokens, all_bert2target_map = self.tokenize(batch, self.bert_tokenizer)
        embeddings_new = self.embed_batch(batch, all_bert_tokens, all_bert2target_map)

        self.update_embeddings(batch, embeddings_new, embeddings_dict)

    @staticmethod
    def tokenize(batch, tokenizer):
        all_batch = []
        all_bert_tokens = []
        all_bert2target_map = []

        # Initialise all the trees and embeddings
        for sentence in batch:

            # Tokenize the sentence
            bert_mapping = []
            bert_tokens = []
            for token in sentence:
                bert_decomposition = tokenizer.tokenize(token)
                if len(bert_decomposition) == 0:
                    bert_decomposition = ['[UNK]']

                bert_tokens += bert_decomposition
                bert_mapping.append(len(bert_decomposition))

            if len(bert_tokens) > 510: # 512 + CLS and SEP
                continue

            all_batch.append(sentence)
            all_bert2target_map.append(bert_mapping)
            all_bert_tokens.append(bert_tokens)

        return all_batch, all_bert_tokens, all_bert2target_map

    def embed_batch(self, batch, batch_bert, batch_map):
        input_ids, attention_mask, mappings, _ = \
            self.get_batch_tensors(batch_bert, batch_map, self.pad_id, self.bert_tokenizer)

        with torch.no_grad():
            embeddings = self.run_batch(batch, input_ids, attention_mask, mappings)
        return embeddings.cpu().numpy()

    def run_batch(self, _, input_ids, attention_mask, mappings):
        return self.model(input_ids, attention_mask, mappings)

    def update_embeddings(self, batch, embeddings_new, embeddings_dict):
        # Embeddings per word
        for i, sentence in enumerate(batch):
            for j, token in enumerate(sentence):
                if self.tgt_words is not None and token not in self.tgt_words:
                    continue

                embeddings_dict[token] = embeddings_dict.get(token, []) + [embeddings_new[i, j].tolist()]

    @classmethod
    def get_batch_tensors(cls, batch, batch_map, pad_id, tokenizer):
        lengths_bert = [(len(sentence) + 2) for sentence in batch]  # +2 for CLS/SEP
        longest_sent_bert = max(lengths_bert)
        lengths_orig = [(len(sentence)) for sentence in batch_map]
        longest_sent_orig = max(lengths_orig)

        # Pad it & build up attention mask
        input_ids = np.ones((len(batch), longest_sent_bert)) * pad_id
        attention_mask = np.zeros((len(batch), longest_sent_bert))
        mappings = np.ones((len(batch), longest_sent_orig)) * -1

        for i, sentence in enumerate(batch):
            sentence_len = lengths_bert[i]

            input_ids[i, :sentence_len] = cls.get_sentence_ids(sentence, tokenizer)
            # Mask is 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            attention_mask[i, :sentence_len] = 1
            mappings[i, :len(batch_map[i])] = batch_map[i]

        # Move data to torch and cuda
        input_ids = torch.LongTensor(input_ids).to(device=constants.device)
        attention_mask = torch.LongTensor(attention_mask).to(device=constants.device)
        mappings = torch.LongTensor(mappings).to(device=constants.device)

        return input_ids, attention_mask, mappings, lengths_orig

    @staticmethod
    def get_sentence_ids(sentence, tokenizer):
        return tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + sentence + ["[SEP]"])
