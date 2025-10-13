import librosa
import torch as th
import numpy as np
import soundfile as sf

import sys, os
sys.path.append(os.path.dirname(__file__))
# from speex_linear.lp_or_tde import LP_or_TDE


def audio(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
    return wave_data

def get_firstchannel_read(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def parse_scp(scp, path_list):
    with open(scp) as fid: 
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({"inputs": tmp[0], "duration": tmp[1]})
            else:
                path_list.append({"inputs": tmp[0]})

class DataReaderTSE(object):
    def __init__(self, filename, sample_rate): 
        self.file_list = []
        parse_scp(filename, self.file_list)
        self.sample_rate = sample_rate


    def extract_feature(self, path):
        mic_path = path["inputs"]
        utt_id = mic_path.split("/")[-1]
        mic_name = mic_path.split("/")[-1].split(".")[0]

        ref_path = mic_path.replace("noisy/", "enrol/")
        ref_name = ref_path.split("/")[-1].split(".")[0]

        mic = get_firstchannel_read(mic_path, self.sample_rate).astype(np.float32)
        ref = get_firstchannel_read(ref_path, self.sample_rate).astype(np.float32)
        
        if ref.shape[0] > mic.shape[0]:
            min_len = mic.shape[0]
            ref = ref[:min_len]

        inputs_mic = np.reshape(mic, [1, mic.shape[0]]).astype(np.float32)
        inputs_ref = np.reshape(ref, [1, ref.shape[0]]).astype(np.float32)
        
        inputs_mic = th.from_numpy(inputs_mic)
        inputs_ref = th.from_numpy(inputs_ref)
        
        # print(f'e: {inputs_e.shape}')
        # print(f'mic: {inputs_mic.shape}')
        # print(f'ref: {inputs_ref.shape}')

        egs = {
            "mic": inputs_mic,
            "ref": inputs_ref, 
            "utt_id": utt_id,
            "mic_name": mic_name,
            "ref_name": ref_name
            # "max_norm": max_norm
        }
        return egs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])
