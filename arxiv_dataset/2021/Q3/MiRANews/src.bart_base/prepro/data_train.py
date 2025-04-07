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
from transformers import BartTokenizer

class TrainData():
    def __init__(self, args):
        self.args = args
        if args.large:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', do_lower_case=True)
        else:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)

        # Src
        self.sep_token = '</s>'
        self.cls_token = '<s>'
        # Tgt
        self.beg_token = '<s>'
        self.end_token = '</s>'
        # Share
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

    def preprocess(self, src, tgt, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        if len(idxs) == 0:
            return None

        # Src
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} '.format(self.sep_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        # Tgt
        sep_str = ' ' + self.end_token + ' '
        tgt_subtokens_str = self.beg_token + ' ' + sep_str.join([' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' ' + self.end_token
        tgt_subtokens = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtokens) < self.args.min_tgt_ntokens):
            return None
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtokens)

        tgt_txt = '\t'.join([' '.join(tt) for tt in tgt])
        #src_txt = [original_src_txt[i] for i in idxs]
        src_txt = original_src_txt

        return src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt


class PreproTrainData():
    def __init__(self, args):
        self.args = args

    def _preprocess(self, params):
        corpus_type, json_file, args, save_file = params
        is_test = corpus_type == 'test'
        if (os.path.exists(save_file)):
            logger.info('Ignore %s' % save_file)
            return

        bert = TrainData(args)
        logger.info('Processing %s' % json_file)
        jobs = json.load(open(json_file))
        datasets = []
        for d in jobs:
            source, tgt = d['src'], d['tgt']
            if (args.lower):
                source = [' '.join(s).lower().split() for s in source]
                tgt = [' '.join(s).lower().split() for s in tgt]
            b_data = bert.preprocess(source, tgt, is_test=is_test)
            if (b_data is None):
                continue
            src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt = b_data

            b_data_dict = {"src": src_subtoken_idxs,
                           "tgt": tgt_subtoken_idxs,
                           "example_id": d['example_id'],
                           "src_txt": src_txt,
                           "tgt_txt": tgt_txt}
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
        root_src = self.args.raw_path + corpus_type + "_src.jsonl"
        for line in open(root_src):
            flist = line.strip().split("\t")
            srcs.append([sen.split() for sen in flist])
        return srcs

    def preprocess(self):
        if (self.args.dataset != ''):
            datasets = [self.args.dataset]
        else:
            datasets = ['train', 'test', 'dev']

        for corpus_type in datasets:
            #relation_dict
            srcs = self._load_src(corpus_type)

            tgts = []
            root_tgt = self.args.raw_path + corpus_type + "_tgt.jsonl"
            for line in open(root_tgt):
                tgts.append([item.split() for item in line.strip().split('\t')])

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

