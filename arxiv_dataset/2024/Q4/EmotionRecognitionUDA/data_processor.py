import numpy as np
import pickle as pickle
import os
import time
import math
import torch
from sklearn.preprocessing import normalize
from scipy.fftpack import fft,ifft
from scipy import signal
from scipy.io import loadmat

class Data_Processor():
    def filter_signal(self, orig_signal, sampling_freq, low_freq, high_freq):
      fs=sampling_freq
      f1 = low_freq
      f2 = high_freq
      Wn = [f1*2/fs, f2*2/fs]
      N = 3
      a, b = signal.butter(N=N, Wn=Wn, btype='bandpass')   # Bandpass filtering
      filtered_signal = signal.filtfilt(a, b, orig_signal)
      return filtered_signal

    def Calculate_PSD_DE(self, data,sampling_rate,fs,fStart,fEnd):
        ####Calculate PSD and DE #####
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

    def Load_DEAP(self, SubjectList, window_size, step_size):
      #5 frequency bands: delta, theta, alpha, beta, gamma
      fStart = [1,4,8,14,31]
      fEnd = [4,8,14,31,50]
      sampling_rate=128
      fs=128
      DataSeg_PSD_all_trials=torch.zeros((1,160)) #160=32 channels*5 frequency bands
      DataSeg_DE_all_trials=torch.zeros((1,160))
      labels_all_trials=torch.zeros((1,1))
      count=0
      data_path_DEAP='/data/DEAP/'
      data_P_path_DEAP='/data/DEAP/preprocessed/'

      for sub in SubjectList:
          DataSeg_PSD_Each_Sub=torch.zeros((1,160))
          DataSeg_DE_Each_Sub=torch.zeros((1,160))
          labels_Each_Sub=torch.zeros((1,1))
          data_orig_Each_Sub=torch.zeros((1,32,window_size))
          count1=0
          with open(data_path_DEAP+'s' + sub + '.dat', 'rb') as file:
            print(file.name)
            subject = pickle.load(file, encoding='latin1')
            data=subject["data"]
            labels=subject["labels"]
            data=data[:,0:32,:]

            for i in range (0,40):    # loop over 40 trails
                    data_curr = data[i]
                    labels_curr = labels[i]
                    labels_curr=labels_curr[0]
                    labels_curr=labels_curr.reshape(1,1)
                    data_curr=data_curr[:,384:] #Remove first 3s
                    data_filtered_curr = data_curr
                    start = 0
                    #Convert to tensor
                    data_filtered_curr = torch.from_numpy(data_filtered_curr.copy())
                    data_filtered_curr=data_filtered_curr.to('cuda')
                    data_filtered_curr=data_filtered_curr.float()
                    labels_curr = torch.from_numpy(labels_curr)
                    labels_curr=labels_curr.to('cuda')
                    labels_curr=labels_curr.float()

                    while start + window_size < data_filtered_curr.shape[1]:
                        DataSeg_filtered_curr=data_filtered_curr[:,start : start + window_size]
                        DataSeg_orig_curr=data_curr[:,start : start + window_size]
                        DataSeg_orig_curr=DataSeg_orig_curr.reshape(1,DataSeg_orig_curr.shape[0],DataSeg_orig_curr.shape[1])
                        DataSeg_orig_curr = torch.from_numpy(DataSeg_orig_curr)
                        DataSeg_orig_curr=DataSeg_orig_curr.to('cuda')
                        DataSeg_orig_curr=DataSeg_orig_curr.float()
                        PSD_tmp,DE_tmp=self.Calculate_PSD_DE(DataSeg_filtered_curr,sampling_rate,fs,fStart,fEnd)
                        PSD_tmp=PSD_tmp.reshape(1,PSD_tmp.shape[0])
                        DE_tmp=DE_tmp.reshape(1,DE_tmp.shape[0])

                        if count==0:
                            DataSeg_PSD_all_trials=PSD_tmp
                            DataSeg_DE_all_trials=DE_tmp
                            labels_all_trials=labels_curr
                        else:
                            DataSeg_PSD_all_trials=torch.concatenate((DataSeg_PSD_all_trials,PSD_tmp), axis=0)
                            DataSeg_DE_all_trials=torch.concatenate((DataSeg_DE_all_trials,DE_tmp), axis=0)
                            labels_all_trials=torch.concatenate((labels_all_trials,labels_curr), axis=0)

                        if count1==0:
                            DataSeg_PSD_Each_Sub=PSD_tmp
                            DataSeg_DE_Each_Sub=DE_tmp
                            labels_Each_Sub=labels_curr
                            data_orig_Each_Sub=DataSeg_orig_curr
                        else:
                            DataSeg_PSD_Each_Sub=torch.concatenate((DataSeg_PSD_Each_Sub,PSD_tmp), axis=0)
                            DataSeg_DE_Each_Sub=torch.concatenate((DataSeg_DE_Each_Sub,DE_tmp), axis=0)
                            labels_Each_Sub=torch.concatenate((labels_Each_Sub,labels_curr), axis=0)
                            data_orig_Each_Sub=torch.concatenate((data_orig_Each_Sub,DataSeg_orig_curr), axis=0)
                        start = start + step_size
                        count=count+1
                        count1=count1+1

          DataSeg_PSD_Each_Sub_np=DataSeg_PSD_Each_Sub.detach().cpu().data.numpy()
          DataSeg_DE_Each_Sub_np=DataSeg_DE_Each_Sub.detach().cpu().data.numpy()
          labels_Each_Sub_np=labels_Each_Sub.detach().cpu().data.numpy()
          data_orig_Each_Sub_np=data_orig_Each_Sub.detach().cpu().data.numpy()
          np.save(data_P_path_DEAP+'PSD_sub_'+ sub + '.npy', DataSeg_PSD_Each_Sub_np)
          np.save(data_P_path_DEAP+'DE_sub_'+ sub + '.npy', DataSeg_DE_Each_Sub_np)
          np.save(data_P_path_DEAP+'labels_sub_'+ sub + '.npy', labels_Each_Sub_np)
          np.save(data_P_path_DEAP+'Data_Orig_sub_'+ sub + '.npy', data_orig_Each_Sub_np)

      return DataSeg_PSD_all_trials, DataSeg_DE_all_trials, labels_all_trials


    def Load_SEED(self, SubjectList, sessionList, window_size, step_size):
        #5 frequency bands: delta, theta, alpha, beta, gamma
        fStart = [1,4,8,14,31]
        fEnd = [4,8,14,31,50]
        sampling_rate=200
        fs=200
        DataSeg_PSD_all_trials=torch.zeros((1,160)) #160=32 channels*5 frequency bands
        DataSeg_DE_all_trials=torch.zeros((1,160))
        labels_all_trials=torch.zeros((1,1))
        count=0
        data_path_SEED='/data/SEED/'
        data_P_path_SEED='/data/SEED/preprocessed/'

        #Load labels
        with open(data_path_SEED+'label.mat', 'rb') as file:
                label_mat = loadmat(file.name)
                labels=label_mat["label"]
                labels=labels[0]
                labels=labels+1

        for sub in SubjectList:
            DataSeg_PSD_Each_Sub=torch.zeros((1,160))
            DataSeg_DE_Each_Sub=torch.zeros((1,160))
            labels_Each_Sub=torch.zeros((1,1))
            data_orig_Each_Sub=torch.zeros((1,32,window_size))
            count1=0
            for i in range(len(sessionList)): #loop over all sessions
                ses=sessionList[i]
                with open(data_path_SEED+ sub +'_'+ ses +'.mat', 'rb') as file:
                        data_mat = loadmat(file.name)
                        print(file.name)
                        count_idx=0
                        label_idx=0
                        for key in data_mat:
                          if (count_idx>2): #First 3 are headers
                              data_curr_tmp = data_mat[key]
                              keep_indices = [0,2,3,4,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,52,54,58,59,60] #keep 32 channels common with DEAP
                              data_curr_list = [data_curr_tmp[ii] for ii in keep_indices]
                              data_curr = np.array(data_curr_list)
                              data_filtered_curr = self.filter_signal(data_curr,fs,0.3,50) #Filter with passband 0.3-50Hz
                              data_filtered_curr = torch.from_numpy(data_filtered_curr.copy())
                              data_filtered_curr=data_filtered_curr.to('cuda')
                              data_filtered_curr=data_filtered_curr.float()

                              labels_curr=labels[label_idx]
                              labels_curr=labels_curr.reshape(1,1)
                              label_idx=label_idx+1
                              labels_curr = torch.from_numpy(labels_curr)
                              labels_curr=labels_curr.to('cuda')
                              labels_curr=labels_curr.float()
                              start = 0
                              count2=0

                              while start + window_size < data_filtered_curr.shape[1]:
                                  DataSeg_filtered_curr=data_filtered_curr[:,start : start + window_size]
                                  DataSeg_orig_curr=data_curr[:,start : start + window_size]
                                  DataSeg_orig_curr=DataSeg_orig_curr.reshape(1,DataSeg_orig_curr.shape[0],DataSeg_orig_curr.shape[1])
                                  DataSeg_orig_curr = torch.from_numpy(DataSeg_orig_curr)
                                  DataSeg_orig_curr=DataSeg_orig_curr.to('cuda')
                                  DataSeg_orig_curr=DataSeg_orig_curr.float()
                                  PSD_tmp,DE_tmp=self.Calculate_PSD_DE(DataSeg_filtered_curr,sampling_rate,fs,fStart,fEnd)
                                  PSD_tmp=PSD_tmp.reshape(1,PSD_tmp.shape[0])
                                  DE_tmp=DE_tmp.reshape(1,DE_tmp.shape[0])

                                  if count==0:
                                      DataSeg_PSD_all_trials=PSD_tmp
                                      DataSeg_DE_all_trials=DE_tmp
                                      labels_all_trials=labels_curr
                                  else:
                                      DataSeg_PSD_all_trials=torch.concatenate((DataSeg_PSD_all_trials,PSD_tmp), axis=0)
                                      DataSeg_DE_all_trials=torch.concatenate((DataSeg_DE_all_trials,DE_tmp), axis=0)
                                      labels_all_trials=torch.concatenate((labels_all_trials,labels_curr), axis=0)

                                  if count1==0:
                                      DataSeg_PSD_Each_Sub=PSD_tmp
                                      DataSeg_DE_Each_Sub=DE_tmp
                                      labels_Each_Sub=labels_curr
                                      data_orig_Each_Sub=DataSeg_orig_curr
                                  else:
                                      DataSeg_PSD_Each_Sub=torch.concatenate((DataSeg_PSD_Each_Sub,PSD_tmp), axis=0)
                                      DataSeg_DE_Each_Sub=torch.concatenate((DataSeg_DE_Each_Sub,DE_tmp), axis=0)
                                      labels_Each_Sub=torch.concatenate((labels_Each_Sub,labels_curr), axis=0)
                                      data_orig_Each_Sub=torch.concatenate((data_orig_Each_Sub,DataSeg_orig_curr), axis=0)

                                  start = start + step_size
                                  count=count+1
                                  count2=count2+1
                                  count1=count1+1

                          count_idx=count_idx+1

            DataSeg_PSD_Each_Sub_np=DataSeg_PSD_Each_Sub.detach().cpu().data.numpy()
            DataSeg_DE_Each_Sub_np=DataSeg_DE_Each_Sub.detach().cpu().data.numpy()
            labels_Each_Sub_np=labels_Each_Sub.detach().cpu().data.numpy()
            data_orig_Each_Sub_np=data_orig_Each_Sub.detach().cpu().data.numpy()
            np.save(data_P_path_SEED+'PSD_sub_'+ sub + '.npy', DataSeg_PSD_Each_Sub_np)
            np.save(data_P_path_SEED+'DE_sub_'+ sub + '.npy', DataSeg_DE_Each_Sub_np)
            np.save(data_P_path_SEED+'labels_sub_'+ sub + '.npy', labels_Each_Sub_np)
            np.save(data_P_path_SEED+'Data_Orig_sub_'+ sub + '.npy', data_orig_Each_Sub_np)

        return DataSeg_PSD_all_trials, DataSeg_DE_all_trials, labels_all_trials
