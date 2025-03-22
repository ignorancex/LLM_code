
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
import torchaudio
from utils.misc import power_compress, power_uncompress, stft, istft, compute_fbank
MAX_WAV_VALUE = 32768.0

def decode_one_audio(model, device, inputs, args):
    if args.network == 'MossFormer2_SS_16K':
        return decode_one_audio_mossformer2_ss_16k(model, device, inputs, args)
    else:
       print("in decode, {args.network} is found!")
       return 

def decode_one_audio_mossformer2_ss_16k(model, device, inputs, args):
    out = []
    #inputs, utt_id, nsamples = data_reader[idx]
    decode_do_segement = False
    window = args.sampling_rate * args.decode_window  #decoding window length
    stride = int(window*0.75) #decoding stride if segmentation is used
    #print('inputs:{}'.format(inputs.shape))
    b,t = inputs.shape
    if t > window * args.one_time_decode_length: #120:
        print('The sequence is longer than {} seconds, using segmentation decoding...'.format(args.one_time_decode_length))
        decode_do_segement=True ##set segment decoding to true for very long sequence

    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    #print('inputs after padding:{}'.format(inputs.shape))
    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b,t = inputs.shape
    if decode_do_segement: # int(1.5*window) and decode_do_segement:
        outputs = np.zeros((args.num_spks,t))
        give_up_length=(window - stride)//2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:,current_idx:current_idx+window]
            tmp_out_list = model(tmp_input,)
            for spk in range(args.num_spks):
                tmp_out_list[spk] = tmp_out_list[spk][0,:].cpu().numpy()
                if current_idx == 0:
                    outputs[spk, current_idx:current_idx+window-give_up_length] = tmp_out_list[spk][:-give_up_length]
                else:
                    outputs[spk, current_idx+give_up_length:current_idx+window-give_up_length] = tmp_out_list[spk][give_up_length:-give_up_length]
            current_idx += stride
        for spk in range(args.num_spks):
            out.append(outputs[spk,:])
    else:
        out_list=model(inputs)
        for spk in range(args.num_spks):
            out.append(out_list[spk][0,:].cpu().numpy())

    max_abs = 0
    for spk in range(args.num_spks):
        if max_abs < max(abs(out[spk])):
            max_abs = max(abs(out[spk]))
    for spk in range(args.num_spks):
        out[spk] = out[spk]/max_abs
    return out

