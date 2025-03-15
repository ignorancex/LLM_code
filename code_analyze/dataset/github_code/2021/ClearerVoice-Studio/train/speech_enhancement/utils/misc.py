
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import os 
import sys
import librosa
import torchaudio
MAX_WAV_VALUE = 32768.0
EPS = 1e-6

def read_and_config_file(input_path, decode=0):
    processed_list = []

    if decode:
        if os.path.isdir(input_path):
            processed_list = librosa.util.find_files(input_path, ext="wav")
            if len(processed_list) == 0:
                processed_list = librosa.util.find_files(input_path, ext="flac")
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
            elif len(tmp) == 2:
                sample = {'inputs': tmp_paths[0], 'labels':tmp_paths[1]}
            processed_list.append(sample)
    return processed_list

def load_checkpoint(checkpoint_path, use_cuda):
    #if use_cuda:
    #    checkpoint = torch.load(checkpoint_path)
    #else:
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]


def reload_for_eval(model, checkpoint_dir, use_cuda):
    print('reloading from: {}'.format(checkpoint_dir))
    best_name = os.path.join(checkpoint_dir, 'last_best_checkpoint')
    ckpt_name = os.path.join(checkpoint_dir, 'last_checkpoint')
    if os.path.isfile(best_name):
        name = best_name 
    elif os.path.isfile(ckpt_name):
        name = ckpt_name
    else:
        print('Warning: There is no exited checkpoint or best_model!!!!!!!!!!!!')
        return
    with open(name, 'r') as f:
        model_name = f.readline().strip()
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    print('checkpoint_path: {}'.format(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path, use_cuda)

    '''
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    '''
    if 'model' in checkpoint:
        pretrained_model = checkpoint['model']
    else:
        pretrained_model = checkpoint
    state = model.state_dict()
    for key in state.keys():
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            state[key] = pretrained_model[key]
        elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
             state[key] = pretrained_model[key.replace('module.', '')]
        elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
             state[key] = pretrained_model['module.'+key]
        elif self.print: print(f'{key} not loaded')
    model.load_state_dict(state)

    print('=> Reload well-trained model {} for decoding.'.format(
            model_name))

def reload_model(model, optimizer, checkpoint_dir, use_cuda=True, strict=True):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint['model'], strict=strict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print('=> Reload previous model and optimizer.')
    else:
        print('[!] checkpoint directory is empty. Train a new model ...')
        epoch = 0
        step = 0
    return epoch, step

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, mode='checkpoint'):
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(epoch, step))
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'step': step}, checkpoint_path)

    with open(os.path.join(checkpoint_dir, mode), 'w') as f:
        f.write('model.ckpt-{}-{}.pt'.format(epoch, step))
    print("=> Save checkpoint:", checkpoint_path)

def setup_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

def stft(x, args, center=False):
    win_type = args.win_type
    win_len = args.win_len
    win_inc = args.win_inc
    fft_len = args.fft_len
    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=False)
        window = window.to(x.device)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=False)
        window = window.to(x.device)
    else:
        print(f"in stft, {win_type} is not supported!")
        return
    return torch.stft(x, fft_len, win_inc, win_len, center=center, window=window, return_complex=False)

def istft(x, args, slen=None, center=False, normalized=False, onsided=None, return_complex=False):
    win_type = args.win_type
    win_len = args.win_len
    win_inc = args.win_inc
    fft_len = args.fft_len
    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=False)
        window = window.to(x.device)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=False)
        window = window.to(x.device)
    else:
        print(f"in istft, {win_type} is not supported!")
        return
    '''
    if torch.__version__<='2.1.0':
        return torch.istft(x, n_fft=fft_len, hop_length=win_inc, win_length=win_len, window=window, center=False, normalized=False, onesided=None, length=slen, return_complex=False)
    else:
        x_complex = torch.view_as_complex(x) # x[...,0] + 1j*x[...,1]
        return torch.istft(x_complex, n_fft=fft_len, hop_length=win_inc, win_length=win_len, window=window, center=False, normalized=False, onesided=None, length=slen, return_complex=False)
    '''
    try:
        output = torch.istft(x, n_fft=fft_len, hop_length=win_inc, win_length=win_len, window=window, center=False, normalized=False, onesided=None, length=slen, return_complex=False)
    except:
        x_complex = torch.view_as_complex(x)
        output = torch.istft(x_complex, n_fft=fft_len, hop_length=win_inc, win_length=win_len, window=window, center=False, normalized=False, onesided=None, length=slen, return_complex=False)
    return output
def compute_fbank(audio_in, args):
    frame_length = args.win_len / args.sampling_rate * 1000
    frame_shift = args.win_inc / args.sampling_rate * 1000

    return torchaudio.compliance.kaldi.fbank(audio_in, dither=1.0, frame_length=frame_length, frame_shift=frame_shift, num_mel_bins=args.num_mels,
                                             sample_frequency=args.sampling_rate, window_type=args.win_type)

