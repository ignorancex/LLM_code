import math
# from os import path
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.functional import F
from transformers import BertTokenizer

from h02_bert_embeddings.bert import BertProcessor as BertEmbeddingsProcessor
from h02_bert_embeddings.bert_runner import BertEmbeddingsGetter
from h03_bert_surprisal.bert_mlm_per_word import BertPerWordForMaskedLM
from h03_bert_surprisal.data_parallel import TransparentDataParallel
from utils import constants
from utils import utils


class BertProcessor(BertEmbeddingsProcessor):
    def __init__(self, bert_fname, bert_option, tgt_words=None):
        self.bert_fname = bert_fname
        super().__init__(bert_option, tgt_words=tgt_words)

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def load_bert_model(self, _):
        model = BertPerWordForMaskedLM.load(self.bert_fname, 'cpu')
        model.eval()

        model = model.to(device=constants.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = TransparentDataParallel(model)
        return model

    def run_batch(self, batch, input_ids, attention_mask, mappings):
        logits = self.model(input_ids, attention_mask, mappings)

        batch_size = logits.size(0)
        batch_len = logits.size(1)
        vocab_size = logits.size(2)

        target_ids = \
            self.model.get_target_tensor(batch)

        logits_flat = logits.view(-1, vocab_size)
        tgt_ids_flat = target_ids.view(-1)
        logprobs = self.loss(logits_flat, tgt_ids_flat) / math.log(2)

        return logprobs.reshape(batch_size, batch_len)


class BertSurprisalGetter(BertEmbeddingsGetter):
    def __init__(self, bert_fname, bert_option, batch_size, dump_size, tgt_words):
        self.bert_fname = bert_fname
        super().__init__(bert_option, batch_size, dump_size, tgt_words)

    def load_bert(self, bert_option, tgt_words):
        logging.info("Loading pre-trained BERT network")
        bert = BertProcessor(self.bert_fname, bert_option, tgt_words=tgt_words)
        return bert

    def get_surprisals(self, src_file, tgt_path):
        with torch.no_grad():
            self.process_file(src_file, tgt_path)
