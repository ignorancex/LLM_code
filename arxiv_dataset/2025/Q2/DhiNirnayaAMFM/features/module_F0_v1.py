#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:19:05 2023

@author: self
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pysptk
import scipy

def f0estimate_rapt(path, fs):
    data_audio, fs = librosa.load(path, sr=fs) 
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))

    size_frame=0.03 # sec
    size_step=0.01 # sec
    minf0=60 #hz
    maxf0=350 #hz
    voice_bias=-0.2
    size_frameS=size_frame*float(fs)
    size_stepS=size_step*float(fs)
    overlap=size_stepS/size_frameS
    #nF=int((len(data_audio)/size_frameS/overlap))-1
    data_audiof=np.asarray(data_audio, dtype=np.float32)

    # Calculate frame rate
    frame_rate = fs / (overlap * size_frameS)
    #print("Frame rate for F0:", frame_rate, "Hz")
    F0=pysptk.sptk.rapt(data_audiof*(2**15), fs, int(size_stepS), min=minf0, max=maxf0, voice_bias=voice_bias, otype='f0')
    
    return F0, frame_rate, size_stepS
    
