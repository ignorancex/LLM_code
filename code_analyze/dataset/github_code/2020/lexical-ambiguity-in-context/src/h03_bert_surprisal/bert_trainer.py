import math
from os import path
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
from transformers import BertTokenizer, BertModel

from h03_bert_surprisal.bert_mlm_per_word import BertPerWordForMaskedLM
from h03_bert_surprisal.data_parallel import TransparentDataParallel
from h02_bert_embeddings.bert_runner import BertEmbeddingsGetter
from h02_bert_embeddings.bert import BertProcessor
from utils import constants
from utils import utils


class BertMLMTrainer(BertEmbeddingsGetter):
    def __init__(self, bert_option, batch_size, min_freq_vocab):
        self.bert_option = bert_option
        self.batch_size = batch_size
        self.max_vocab = 100000
        self.min_freq_vocab = min_freq_vocab

        self.n_skipped = 0
        self.src_fname = None

    def train(self, src_file, tgt_file):
        self.vocab = self.get_vocab(src_file)
        self.model, self.bert_tokenizer = self.init_bert(self.vocab)
        self.pad_id_bert = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
        self.pad_id_self = self.vocab['<PAD>']

        self.train_file(src_file)
        self.model.save(tgt_file)

    def get_vocab(self, src_file):
        vocab = {}

        with open(src_file, 'r') as f:
            for line in tqdm(f, desc='Getting vocab', mininterval=.2):
                for token in line.split(' '):
                    vocab[token] = vocab.get(token, 0) + 1

        vocab_map = self.get_vocab_dict(vocab)
        return vocab_map

    def get_vocab_dict(self, vocab):
        vocab = {x: count for x, count in vocab.items() if count >= self.min_freq_vocab}
        if len(vocab) > self.max_vocab:
            print('Caped vocab size from %d to %d' % (len(vocab), self.max_vocab))
        else:
            print('Vocab size is %d' % (len(vocab)))

        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        vocab_dict = {x: i + 2 for i, (x, _) in enumerate(vocab[:self.max_vocab])}
        vocab_dict['<PAD>'] = 0
        vocab_dict['<UNK>'] = 1
        return vocab_dict

    def init_bert(self, vocab):
        logging.info("Loading pre-trained BERT network")
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_option)
        mask_idx = bert_tokenizer.convert_tokens_to_ids('[MASK]')

        model = BertPerWordForMaskedLM(self.bert_option, vocab, mask_idx)
        model.train()
        if torch.cuda.device_count() > 1:
            model = TransparentDataParallel(model)
        model = model.to(device=constants.device)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.vocab['<PAD>'])
        self.optim = optim.Adam(model.parameters())
        self.running_loss_temp = []

        return model, bert_tokenizer

    def train_file(self, src_file):
        n_sentences = utils.get_n_lines(src_file)

        with tqdm(total=n_sentences, desc='Processing sentences. 0 skipped',
                  mininterval=.2) as pbar:
            self.run(src_file, pbar)

    def run(self, src_file, pbar):
        tqdm.write('\tRunning on cuda')
        f_iterator = self.iterate_wiki(src_file, pbar)

        for batch_id, batch in enumerate(f_iterator):
            self.train_batch(batch_id, batch)

    def train_batch(self, batch_id, batch):
        batch, batch_bert, batch_map = BertProcessor.tokenize(batch, self.bert_tokenizer)
        input_ids, attention_mask, mappings, _ = \
            BertProcessor.get_batch_tensors(batch_bert, batch_map, self.pad_id_bert, self.bert_tokenizer)

        target_ids = \
            self.model.get_target_tensor(batch)

        train_loss = self.run_batch(
            input_ids, attention_mask, mappings, target_ids)

        self.running_loss_temp += [train_loss]
        if (batch_id % 100) == 0:
            tqdm.write('train_loss is %f' % (sum(self.running_loss_temp) / len(self.running_loss_temp)))
            self.running_loss_temp = []

    def run_batch(self, input_ids, attention_mask, mappings, target_ids):
        self.optim.zero_grad()
        logits = self.model(input_ids, attention_mask, mappings)
        loss = self.loss(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        if math.isnan(loss.detach().cpu().item()):
            return float('inf')

        loss.backward()
        self.optim.step()

        return loss.detach().cpu().item()
