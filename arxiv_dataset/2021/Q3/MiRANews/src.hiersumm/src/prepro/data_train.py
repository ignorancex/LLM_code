import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import copy
import torch
import subprocess
from collections import Counter
from os.path import join as pjoin
from multiprocess import Pool
from others.logging import logger
import sentencepiece

class TrainData():
    def __init__(self, args):
        self.args = args
        self.sp = sentencepiece.SentencePieceProcessor(model_file=args.vocab_path)
        # Tgt
        self.beg_token = '<S>'
        self.end_token = '</S>'
        self.sep_token = '<Q>'

        # Tgt_text
        self.end_str = '</t>'
        self.beg_str = '<t>'

    def preprocess(self, src, tgt, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        tgt_txt = ' {} {} '.format(self.end_str, self.beg_str).join(tgt)
        tgt[0] = self.beg_token + ' ' + tgt[0]
        tgt[-1] = tgt[-1] + ' ' + self.end_token
        for i in range(len(tgt)-1):
            tgt[i] = tgt[i] + ' ' + self.sep_token
        tgt_subtoken_idxs = self.sp.encode(' '.join(tgt), out_type=int)
        src_subtoken_idxs = [self.sp.encode(' '.join(s), out_type=int) for s in src]
        checked_src_idxs = []
        for i, item in enumerate(src_subtoken_idxs):
            if len(item) > self.args.max_src_ntokens_per_sent:
                checked_src_idxs.append(item[:self.args.max_src_ntokens_per_sent])
            else:
                checked_src_idxs.append(item)
        return checked_src_idxs, tgt_subtoken_idxs, tgt_txt


class PreproTrainData():
    def __init__(self, args):
        self.args = args

    def _preprocess(self, params):
        corpus_type, json_file, args, save_file = params
        is_test = corpus_type == 'test'
        if (os.path.exists(save_file)):
            logger.info('Ignore %s' % save_file)
            return

        processor = TrainData(args)
        logger.info('Processing %s' % json_file)
        jobs = json.load(open(json_file))
        datasets = []
        for d in jobs:
            source, tgt = d['src'], d['tgt']
            b_data = processor.preprocess(source, tgt, is_test=is_test)
            if (b_data is None):
                continue
            src_subtoken_idxs, tgt_subtoken_idxs, tgt_txt = b_data
            if len(src_subtoken_idxs) > 0:
                b_data_dict = {"src": src_subtoken_idxs,
                               "tgt": tgt_subtoken_idxs,
                               "tgt_str": tgt_txt}
                datasets.append(b_data_dict)

        logger.info('Processed instances %d' % len(datasets))
        logger.info('Saving to %s' % save_file)
        torch.save(datasets, save_file)
        datasets = []
        gc.collect()

    def preprocess(self):
        if (self.args.dataset != ''):
            datasets = [self.args.dataset]
        else:
            datasets = ['dev', 'train', 'test']
        for corpus_type in datasets:
            a_lst = []
            for json_f in glob.glob(pjoin(self.args.raw_path, '*' + corpus_type + '.*.json')):
                real_name = json_f.split('/')[-1]
                a_lst.append((corpus_type, json_f, self.args, pjoin(self.args.save_path, real_name.replace('json', 'bart.pt'))))
            print(a_lst)
            pool = Pool(self.args.n_cpus)
            for d in pool.imap(self._preprocess, a_lst):
                pass
            pool.close()
            pool.join()


class PreproTrainJson():
    def __init__(self, args):
        self.args = args

    def _load_src(self, corpus_type):
        srcs = []
        for line in open(root_src):
            flist = json.loads(line.strip())
            srcs.append([sen.split() for sen in flist])
        return srcs

    def preprocess(self):
        if (self.args.dataset != ''):
            datasets = [self.args.dataset]
        else:
            datasets = ['train', 'test', 'dev']

        for corpus_type in datasets:
            #relation_dict
            srcs = []
            root_src = self.args.raw_path + corpus_type + "_src.jsonl"
            for line in open(root_src):
                srcs.append(json.loads(line.strip()))

            tgts = []
            root_tgt = self.args.raw_path + corpus_type + "_tgt.jsonl"
            for line in open(root_tgt):
                tgts.append(line.strip().split('\t'))

            json_objs = []
            for i, src in enumerate(srcs):
                json_objs.append({'src': src, 'tgt': tgts[i], 'example_id':i})

            dataset = []
            p_ct = 0
            for d in json_objs:
                if (d is None):
                    continue
                dataset.append(d)
                if (len(dataset) > self.args.shard_size):
                    pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path, corpus_type, p_ct)
                    with open(pt_file, 'w') as save:
                        save.write(json.dumps(dataset))
                        p_ct += 1
                        dataset = []
            if (len(dataset) > 0):
                pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

