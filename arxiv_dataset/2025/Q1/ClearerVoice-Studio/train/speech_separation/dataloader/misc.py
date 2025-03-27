
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import os 
import sys
import librosa

def read_and_config_file(input_path, args, decode=False):
    processed_list = []

    if decode:
        if os.path.isdir(input_path):
            processed_list = librosa.util.find_files(input_path, ext="wav")
            if len(processed_list) == 0:
                processed_list = librosa.util.find_files(input_path, ext="flac")
        else:
            if input_path.lower().endswith(".wav") or input_path.lower().endswith(".flac"):
                processed_list.append(input_path)
            else:
                with open(input_path) as fid:
                    for line in fid:
                        path_s = line.strip().split()
                        processed_list.append(path_s[0])
        return processed_list

    with open(input_path) as fid:
        for line in fid:
            multi_paths = line.strip().split()
            if args.load_type == 'one_input_one_output':            
                sample = {'inputs': multi_paths[0], 'labels':multi_paths[1]}
            elif args.load_type == 'one_input_multi_outputs':
                multi_labels = []
                for i in range(1, len(multi_paths)):
                    multi_labels.append(multi_paths[i])
                sample = {'inputs': multi_paths[0], 'labels':multi_labels}
            elif args.load_type == 'multi_inputs_multi_outputs':
                multi_inputs = []
                multi_labels = []
                for i in range(0, len(multi_paths)//2):
                    multi_inputs.append(multi_paths[i])
                    multi_labels.append(i+len(multi_paths)//2)
                sample = {'inputs': multi_inputs, 'labels':multi_labels}
            processed_list.append(sample)
    return processed_list
