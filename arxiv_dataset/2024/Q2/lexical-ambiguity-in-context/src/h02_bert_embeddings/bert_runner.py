import math
from os import path
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from transformers import BertTokenizer

from h02_bert_embeddings.bert import BertProcessor
from utils import constants
from utils import utils


class BertEmbeddingsGetter(object):
    def __init__(self, bert_option, batch_size, dump_size, tgt_words):
        self.bert_option = bert_option
        self.batch_size = batch_size
        self.dump_size = dump_size
        self.tgt_words = tgt_words

        self.bert = self.load_bert(bert_option, tgt_words)
        self.n_skipped = 0
        self.src_fname = None

    @classmethod
    def load_bert(cls, bert_option, tgt_words):
        logging.info("Loading pre-trained BERT network")
        bert = BertProcessor(bert_option, tgt_words=tgt_words)
        return bert

    def get_embeddings(self, src_file, tgt_path):
        with torch.no_grad():
            self.process_file(src_file, tgt_path)

    def process_file(self, src_file, tgt_path):
        n_lines = utils.get_n_lines(src_file)
        self.n_skipped = 0

        with tqdm(total=n_lines, desc='Processing sentences. 0 skipped',
                  mininterval=.2) as pbar:
            self.run(src_file, tgt_path, pbar)

    def run(self, src_file, tgt_path, pbar):
        tqdm.write('\tRunning on %s' % constants.device)

        self.dump_id = 0
        processed_size = 0
        embeddings = {}
        n_skip = len(utils.get_filenames(tgt_path))

        for batch in self.iterate_wiki(src_file, pbar):
            processed_size += len(batch)

            if self.dump_id < n_skip:
                if processed_size >= self.dump_size:
                    self.dump_id += 1
                    processed_size = 0
                continue

            self.bert.process_batch(batch, embeddings)

            if processed_size >= self.dump_size:
                self.write_results(embeddings, tgt_path)
                embeddings = {}
                processed_size = 0

        if processed_size > 0:
            self.write_results(embeddings, tgt_path)

    def iterate_wiki(self, src_file, pbar):
        batch = []
        for sentence in self.get_next_sentence(src_file, pbar):
            batch += [sentence]
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def get_next_sentence(self, src_file, pbar):
        with open(src_file, 'r') as f:
            for line in f:
                tokens = line.strip().split(' ')
                pbar.update(1)
                if len(tokens) > 100 or len(tokens) <= 2:
                    self.n_skipped += 1
                    pbar.set_description('Processing sentences. %d skipped' % self.n_skipped)
                    continue

                yield tokens

    def write_results(self, results, tgt_path):
        results = {key: np.matrix(values) for key, values in results.items()}
        self._write_results(results, tgt_path)

    def _write_results(self, results, tgt_path):
        fname = 'results--%05d.pickle' % \
            (self.dump_id)
        tqdm.write("\tSaving partial results: %s" % fname)
        filename = path.join(tgt_path, fname)
        utils.write_pickle(filename, results)

        self.dump_id += 1
