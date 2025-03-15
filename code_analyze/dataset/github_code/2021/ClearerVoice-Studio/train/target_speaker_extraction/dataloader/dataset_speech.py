import numpy as np
import math, os, csv

import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import soundfile as sf

from .utils import DistributedSampler

def get_dataloader_speech(args, partition):
    datasets = dataset_speech(args, partition)

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
    a_mix, a_tgt, (aux, aux_len, speakers) = batch[0]
    a_mix = torch.tensor(a_mix)
    a_tgt = torch.tensor(a_tgt) 
    aux = torch.tensor(aux)
    aux_len = torch.tensor(aux_len)
    speakers = torch.tensor(speakers)
    return a_mix, a_tgt, (aux, aux_len, speakers)

class dataset_speech(data.Dataset):
    def __init__(self, args, partition):
        self.minibatch =[]
        self.args = args
        self.partition = partition
        
        self.max_length = args.max_length
        self.audio_sr=args.audio_sr
        self.ref_sr=args.ref_sr
        self.batch_size=args.batch_size

        self.audio_direc = args.audio_direc
        self.reference_direc = args.reference_direc

        mix_scp=f'{args.mix_lst_path}{partition}/mix_with_length.scp'
        ref_scp=f'{args.mix_lst_path}{partition}/ref.scp'
        aux_scp=f'{args.mix_lst_path}{partition}/aux.scp'

        mix_lst=open(mix_scp).read().splitlines()
        ref_lst=open(ref_scp).read().splitlines()
        aux_lst=open(aux_scp).read().splitlines()
        
        mix_lst = [x.split(' ') for x in mix_lst]
        ref_lst = [x.split(' ') for x in ref_lst]
        aux_lst = [x.split(' ') for x in aux_lst]

        data_list = []
        for i in range(len(mix_lst)):
            data_list.append((mix_lst[i][0], mix_lst[i][1], ref_lst[i][1], aux_lst[i][1], mix_lst[i][2]))
        data_list = sorted(data_list, key=lambda data: float(data[-1]), reverse=True)#[:200]

        start = 0
        while True:
            end = min(len(data_list), start + self.batch_size)
            self.minibatch.append(data_list[start:end])
            if end == len(data_list):
                break
            start = end

        spk_pth = f'{args.mix_lst_path}/wsj0_2mix_extr_tr.spk'
        spk_lst=open(spk_pth).read().splitlines()
        ID_idx = 0
        speaker_dict={}
        for line in spk_lst:
            speaker_dict[line]=ID_idx
            ID_idx += 1
        self.speaker_dict=speaker_dict

    def _audioread(self, path, sampling_rate, min_length_audio=None):
        data, fs = sf.read(path, dtype='float32')    
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        if len(data.shape) >1:
            data = data[:, 0]    

        if min_length_audio != None:
            data = data[:min_length_audio]
            if data.shape[0] < min_length_audio:
                data = np.pad(data, (0, int(min_length_audio - data.shape[0])),mode = 'constant')
        return  data


    def __getitem__(self, index):
        mix_audios=[]
        tgt_audios=[]
        tgt_refs=[]
        speakers=[]
        
        batch_lst = self.minibatch[index]
        min_length_audio = math.floor(float(batch_lst[-1][-1])*self.audio_sr)  # truncate to the shortest utterance in the batch
        max_ref_length = 0

        for line in batch_lst:

            a_mix_audio_path=self.audio_direc+line[1]
            a_mix = self._audioread(a_mix_audio_path, self.audio_sr, min_length_audio)

            a_tgt_audio_path=self.audio_direc+line[2]
            a_tgt = self._audioread(a_tgt_audio_path, self.audio_sr, min_length_audio)

            a_ref_audio_path=self.reference_direc+line[3]
            a_ref = self._audioread(a_ref_audio_path, self.ref_sr)
            max_ref_length = max(max_ref_length, a_ref.shape[0])

            if self.partition == 'test':
                speakers.append(-1)
            else:
                spk = a_ref_audio_path.split('/')[-2]
                speakers.append(self.speaker_dict[spk])

            # random start
            a_max_length = int(self.max_length*self.audio_sr)
            if min_length_audio > a_max_length:
                a_start=np.random.randint(0, (min_length_audio - a_max_length))
                a_mix = a_mix[a_start:a_start+a_max_length]
                a_tgt = a_tgt[a_start:a_start+a_max_length]
            
            mix_audios.append(a_mix)
            tgt_audios.append(a_tgt)
            tgt_refs.append(a_ref)

        aux_length = []
        aux = []
        for ref in tgt_refs:
            length = ref.shape[0]
            aux_length.append(length)
            ref= np.pad(ref, (0, int(max_ref_length - length)), mode = 'edge')
            aux.append(ref)


        return np.asarray(mix_audios, dtype=np.float32), np.asarray(tgt_audios, dtype=np.float32), (np.asarray(aux, dtype=np.float32), np.asarray(aux_length), speakers)


    def __len__(self):
        return len(self.minibatch)

