import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from scipy import signal

def L2Distance(x, y, dim=0, if_mean=True):
    if if_mean:
        distance = torch.mean(torch.sqrt(torch.sum((x - y) ** 2, dim=dim)))
    else:
        distance = torch.sqrt(torch.sum((x - y) ** 2, dim=dim))
    return distance

def reset_grad(opt_g,opt_c1,opt_c2):
    opt_g.zero_grad()
    opt_c1.zero_grad()
    opt_c2.zero_grad()

def discrepancy(out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def create_mini_batches(data, labels, batch_size):
    data_size=data.shape[0]
    itr_len=int(data_size/batch_size)
    data=data[0:itr_len*batch_size]
    labels=labels[0:itr_len*batch_size]
    #Create batches
    data_batch= np.empty([itr_len, batch_size, data.shape[1], data.shape[2]])
    label_batch=np.empty([itr_len, batch_size])
    j=0

    for i in range(itr_len):
      data_batch[i]=data[j:j+batch_size]
      label_batch[i]=labels[j:j+batch_size]
      j=j+batch_size

    label_batch=label_batch.astype(int)
    return data_batch, label_batch


def create_mini_batches_T(data, data_orig, labels, batch_size):
    data_size=data.shape[0]
    itr_len=int(data_size/batch_size)
    data=data[0:itr_len*batch_size]
    data_orig=data_orig[0:itr_len*batch_size]
    labels=labels[0:itr_len*batch_size]
    #Create batches
    data_batch= np.empty([itr_len, batch_size, data.shape[1], data.shape[2]])
    data_orig_batch= np.empty([itr_len, batch_size, data_orig.shape[1], data_orig.shape[2]])
    label_batch=np.empty([itr_len, batch_size])
    j=0

    for i in range(itr_len):
      data_batch[i]=data[j:j+batch_size]
      data_orig_batch[i]=data_orig[j:j+batch_size]
      label_batch[i]=labels[j:j+batch_size]
      j=j+batch_size

    label_batch=label_batch.astype(int)
    return data_batch, data_orig_batch, label_batch

def Calculate_PSD_DE(data,sampling_rate,fs,fStart,fEnd):
      #Compute PSD and DE
      STFTN=sampling_rate
      window=int(data.shape[1]/fs)
      WindowPoints=fs*window
      fStartNum=np.zeros([len(fStart)],dtype=int)
      fEndNum=np.zeros([len(fEnd)],dtype=int)
      for i in range(0,len(fStart)):
          fStartNum[i]=int(fStart[i]/fs*STFTN)
          fEndNum[i]=int(fEnd[i]/fs*STFTN)

      n=data.shape[0]
      m=data.shape[1]
      PSD = torch.zeros([n,len(fStart)])
      DE = torch.zeros([n,len(fStart)])
      Hlength=window*fs
      Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])
      Hwindow = torch.from_numpy(Hwindow)
      Hwindow=Hwindow.to('cuda')
      WindowPoints=fs*window

      for j in range(0,n):
          temp=data[j]
          Hdata=temp*Hwindow
          FFTdata=torch.fft.fft(Hdata,STFTN)
          magFFTdata=abs(FFTdata[0:int(STFTN/2)])
          for p in range(0,len(fStart)):
              E = 0
              for p0 in range(fStartNum[p]-1,fEndNum[p]):
                  E=E+magFFTdata[p0]*magFFTdata[p0]
              E = E/(fEndNum[p]-fStartNum[p]+1)
              PSD[j][p] = E
              DE[j][p] = math.log(100*E,2)

      PSD=PSD.flatten()
      DE=DE.flatten()
      return PSD, DE

def add_gaussian_noise(signal, mean=0, std=1):
    signal=signal.to('cuda')
    noise = torch.normal(mean=mean, std=std, size=(signal.shape[0],signal.shape[1],signal.shape[2]))
    noise=noise.to('cuda')
    noisy_signal = signal + noise
    return noisy_signal
