
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

def read_and_config_file(input_path, decode=0):
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
            tmp_paths = line.strip().split()
            if len(tmp_paths) == 3:
                sample = {'inputs': tmp_paths[0], 'labels':tmp_paths[1], 'duration':float(tmp_paths[2])}
            elif len(tmp_paths) == 2:
                sample = {'inputs': tmp_paths[0], 'labels':tmp_paths[1]}
            processed_list.append(sample)
    return processed_list
