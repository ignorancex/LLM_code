import os
import librosa
import torch
import numpy as np
import soundfile as sf
import pkbar
import argparse
import torchaudio

sample_rate = 16000


def split_cough(file, filename):
    
    out, sr = torchaudio.load(str(file), frame_offset=0, num_frames=-1)
    out = out / max(abs(out.min()), out.max())
    framed_wav = librosa.util.frame(out, frame_length=360, hop_length=180)
    ste = torch.sum(torch.pow(torch.tensor(framed_wav),2),dim=1)[0]

    # cough start idx
    cough_idx = torch.tensor([])
    overthreshold = torch.where(ste>=14.5)[0]
    
    cough_idx = torch.cat((cough_idx, overthreshold[0].unsqueeze(0)))
    for i in range(len(overthreshold)):
        if torch.where(overthreshold > cough_idx[i] + 300)[0].shape[0] == 0:
            break
        cough = overthreshold[min(torch.where(overthreshold > cough_idx[i] + 200)[0])]
        cough_idx = torch.cat((cough_idx, cough.unsqueeze(0)))
    cough_idx = np.array(cough_idx.int())

    end_add_idx = torch.tensor([])
    for i in range(len(cough_idx)):
        if i == len(cough_idx)-1:
            end = max(torch.where(ste[cough_idx[i]:] >= 0.1)[0])
        else:
            end = max(torch.where(ste[cough_idx[i]:cough_idx[i+1]-50] >= 0.1)[0])
        end_add_idx = torch.cat((end_add_idx, end.unsqueeze(0)))
    end_add_idx = np.array(end_add_idx.int())

    for i in range(len(cough_idx)):
        if i == 0:
            cough_wav = out[:,180*cough_idx[i]:180*(cough_idx[i]+end_add_idx[i])]
        elif i == len(cough_idx -1):
            cough_wav = out[:,180*cough_idx[i]:]
        else:
            cough_wav = torch.cat((cough_wav, out[:,180*cough_idx[i]:180*(cough_idx[i]+end_add_idx[i])]), dim=1)
    
    return cough_wav, sr
    

def process_downsample(input_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        pbar = pkbar.Pbar('Downsampling audios of {}'.format(dir), len(files))
        pcount = 0
        for file in files:
            pbar.update(pcount)
            if file.split('.')[-1] != 'wav':
                continue
            target_file = os.path.join(root, file)
            
            cough_wav, sr = split_cough(target_file, file)
            
            sf.write(os.path.join(save_dir, file), cough_wav[0], sr, format='WAV', subtype = 'PCM_16')
            pcount += 1

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)

    a = parser.parse_args()
    
    input_dir = a.input_dir
    save_dir = a.save_dir
    process_downsample(input_dir, save_dir)
