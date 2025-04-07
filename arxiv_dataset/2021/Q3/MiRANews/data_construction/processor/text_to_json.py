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
from transformers import BartTokenizer

class PreproTrainJson():
    def __init__(self, args):
        self.args = args

    def _load_src(self, corpus_type):
        srcs = []
        root_src = self.args.output_dir + self.args.dataset_name + "_" + corpus_type + "_src.jsonl"
        for line in open(root_src):
            flist = line.strip().split("\t")
            srcs.append(' '.join(flist))
        return srcs

    def preprocess(self):
        datasets = ['train', 'test', 'dev']

        for corpus_type in datasets:
            if corpus_type == 'test':
                shard_size = self.args.test_shard_size
            else:
                shard_size = self.args.shard_size
            srcs = self._load_src(corpus_type)

            tgts = []
            root_tgt = self.args.output_dir + self.args.dataset_name + "_" + corpus_type + "_tgt.jsonl"
            for line in open(root_tgt):
                tgts.append(' '.join([item for item in line.strip().split('\t')]))

            json_objs = []
            for i, src in enumerate(srcs):
                json_objs.append(json.dumps({'text': src.lower(), 'summary': tgts[i].lower()}))

            dataset = []
            p_ct = 0
            for d in json_objs:
                if (d is None):
                    continue
                dataset.append(d)
                if (len(dataset) > shard_size):
                    pt_file = "{:s}/{:s}.{:s}.{:d}.json".format(self.args.save_path,self.args.dataset_name,corpus_type,p_ct)
                    with open(pt_file, 'w') as save:
                        save.write('\n'.join(dataset))
                        p_ct += 1
                        dataset = []
            if (len(dataset) > 0):
                pt_file = "{:s}/{:s}.{:s}.{:d}.json".format(self.args.save_path, self.args.dataset_name, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write('\n'.join(dataset))
                    p_ct += 1
                    dataset = []
