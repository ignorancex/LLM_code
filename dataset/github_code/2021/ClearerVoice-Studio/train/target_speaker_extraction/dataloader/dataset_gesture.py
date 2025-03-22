import numpy as np
import math, os, csv

import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import soundfile as sf

from .utils import DistributedSampler

def get_dataloader_gesture(args, partition):
    datasets = dataset_gesture(args, partition)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler,
            collate_fn=custom_collate_fn)
    
    return sampler, generator

def custom_collate_fn(batch):
    a_mix, a_tgt, ref_tgt = batch[0]
    a_mix = torch.tensor(a_mix)
    a_tgt = torch.tensor(a_tgt) 
    ref_tgt = torch.tensor(ref_tgt) 
    return a_mix, a_tgt, ref_tgt

class dataset_gesture(data.Dataset):
    def __init__(self, args, partition):
        self.minibatch =[]
        self.args = args
        self.partition = partition
        self.max_length = args.max_length
        self.audio_sr=args.audio_sr
        self.ref_sr=args.ref_sr
        self.speaker_no=args.speaker_no
        self.batch_size=args.batch_size

        self.mix_lst_path = args.mix_lst_path
        self.audio_direc = args.audio_direc
        self.visual_direc = args.reference_direc
        
        mix_lst=open(self.mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
        mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end


    def _load_gesture(self, visual_path, min_length_visual):
        visual = np.load(visual_path)
        visual = visual.reshape(visual.shape[0], 30)

        visual = visual[:min_length_visual,...]
        if visual.shape[0] < min_length_visual:
            visual = np.pad(visual, ((0,int(min_length_visual - visual.shape[0])),(0,0)), mode = 'constant')
        return visual


    def _audioread(self, path, min_length_audio, sampling_rate):
        data, fs = sf.read(path, dtype='float32')    
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        if len(data.shape) >1:
            data = data[:, 0]    

        data = data[:min_length_audio]
        if data.shape[0] < min_length_audio:
            data = np.pad(data, (0, int(min_length_audio - data.shape[0])),mode = 'constant')
        return  data


    def __getitem__(self, index):
        mix_audios=[]
        tgt_audios=[]
        tgt_visuals=[]
        
        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst[-1].split(',')[-1])      # truncate to the shortest utterance in the batch
        min_length_visual = math.floor(min_length_second*self.ref_sr)
        min_length_audio = math.floor(min_length_second*self.audio_sr)

        for line_cache in batch_lst:
            line=line_cache.split(',')

            c=0
            # read target visual
            tgt_visual_path=self.visual_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.npy'
            v_tgt = self._load_gesture(tgt_visual_path, min_length_visual)

            # read tgt audio
            tgt_audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
            a_tgt = self._audioread(tgt_audio_path, min_length_audio, self.audio_sr)
            target_power = np.linalg.norm(a_tgt, 2)**2 / a_tgt.size
            snr_0 = 10**(float(line[c*4+4])/20)
            
            # read int audio
            c=1
            int_audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
            a_int = self._audioread(int_audio_path, min_length_audio, self.audio_sr)
            intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
            a_int *= np.sqrt(target_power/intef_power)
            snr_1 = 10**(float(line[c*4+4])/20)

            if self.args.speaker_no == 2:
                max_snr = max(snr_0, snr_1)
                a_tgt /= max_snr
                a_int /= max_snr

                a_tgt = a_tgt * snr_0
                a_int = a_int * snr_1

                a_mix = a_tgt + a_int

            elif self.args.speaker_no == 3:
                c=2
                int_audio_path_2=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                a_int2 = self._audioread(int_audio_path_2, min_length_audio, self.audio_sr)
                intef_power_2 = np.linalg.norm(a_int2, 2)**2 / a_int2.size
                a_int2 *= np.sqrt(target_power/intef_power_2)
                snr_2 = 10**(float(line[c*4+4])/20)
                
                max_snr = max(snr_0, snr_1, snr_2)
                a_tgt /= max_snr
                a_int /= max_snr
                a_int2 /= max_snr

                a_tgt = a_tgt * snr_0
                a_int = a_int * snr_1
                a_int2 = a_int2 * snr_2

                a_mix = a_tgt + a_int + a_int2
            else:
                raise NameError('Wrong speaker_no selection')

            # random start
            a_max_length = int(self.max_length*self.audio_sr)
            v_max_length = int(self.max_length*self.ref_sr)
            if min_length_visual > v_max_length:
                v_start=np.random.randint(0, (min_length_visual - v_max_length))
                a_start= int(v_start/self.ref_sr*self.audio_sr)
                v_tgt = v_tgt[v_start:v_start+v_max_length]
                a_mix = a_mix[a_start:a_start+a_max_length]
                a_tgt = a_tgt[a_start:a_start+a_max_length]
                a_int = a_int[a_start:a_start+a_max_length]
            
            
            # audio normalization
            max_val = np.max(np.abs(a_mix))
            if max_val > 1:
                a_mix /= max_val
                a_tgt /= max_val
                a_int /= max_val

            mix_audios.append(a_mix)
            tgt_audios.append(a_tgt)
            tgt_visuals.append(v_tgt)

        return np.asarray(mix_audios, dtype=np.float32), np.asarray(tgt_audios, dtype=np.float32), np.asarray(tgt_visuals, dtype=np.float32)


    def __len__(self):
        return len(self.minibatch)










