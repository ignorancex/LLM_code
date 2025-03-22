# Generate 🎵 music video
# Taken from: https://colab.research.google.com/drive/1BXNHZBai-pXtP-ncliouXo_kUiG1Pq7M?usp=sharing#scrollTo=zFWf0Wi1_4P_
# by: Artemii Novoselov @EarthML1

import requests
import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import dnnlib
import librosa
from scipy.io import wavfile

import time
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from PIL import ImageOps

network_pkl = '/home/diego/Documents/CVC/stylegan3-fun/pretrained/white_veil_5000.pkl'
device = torch.device('cuda:0')

with dnnlib.util.open_url(network_pkl) as fp:
  G = pickle.load(fp)['G_ema'].to(device)

seed =  42#@param {type:"number"}

#@markdown How variable should the video be? (lower values - less variable)
#if you are reading that - you are smart enough to map frequencies to psi as well
truncation_psi = 0.5 #@param {type:"number"}

#@markdown How *strongly* should the image change?
effect_strength =  1#@param {type:"number"}

zs = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)

#@markdown Link to MP3 audio file (you can also extact music from a Youtube link)
audio_link = 'https://www.youtube.com/watch?v=QiEbg9I8yFU' #@param {type:"string"}
# if 'youtu.be' not in audio_link:
#     !wget {audio_link} -O audio.mp3
# else:
#     !youtube-dl --extract-audio --audio-format mp3 https://youtu.be/0OkiUUU3Odw -o music_temp.mp3
#     !ffmpeg -i music_temp.mp3 -af silenceremove=1:0:-50dB audio.mp3

#@markdown Cut audio to N seconds
cut_start =  20#@param {type:"number"}
cut_end =  20#@param {type:"number"}

cut_len = cut_end-cut_start

#@markdown How many frames to use for interpolation?
interp_frames =  20#@param {type:"number"}

#@markdown Which frequencies to use?
freqs = 'low' #@param ['low', 'high', 'all']

arr, fr = librosa.load('audio.mp3')
arr = arr[int(fr*cut_start):int(fr*cut_end)]

wavfile.write('audio.wav', fr, arr)

# stft = torch.stft(torch.tensor(arr),
#            G.mapping.z_dim*2-1,
#            hop_length=G.mapping.z_dim//4,
#            center=False,
#            pad_mode='reflect',
#            normalized=True,
#            onesided=True,
#            return_complex=True)

stft=librosa.feature.melspectrogram(y=arr,
                               sr=fr,
                               n_fft=2048,
                               hop_length=G.mapping.z_dim*4,
                               n_mels=G.mapping.z_dim)

stft = torch.log(torch.tensor(stft).abs())

if freqs == 'low':
    stft[stft.size(0)//2:, :] *= 10

if freqs == 'high':
    stft[:stft.size(0)//2, :] *= 10


#FRAMES
import time
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm

zq = []
with torch.no_grad():
    timestring = time.strftime('%Y%m%d%H%M%S')
    # rand_z = torch.randn(stft.size(-1), G.mapping.z_dim).to(device)
    # q = (G.mapping(rand_z, None, truncation_psi=truncation_psi))

    for i in range(stft.size(-1)):
        frame = stft[:,i].T.to(device)
        z = torch.mean(G.mapping(frame.unsqueeze(0), None, truncation_psi=truncation_psi), dim=0)
        zq.append(z.unsqueeze(0)*effect_strength)

    count = 0
    for k in tqdm(range(len(zq)-1)):
        i_val = torch.linspace(0,1,interp_frames).to(device)
        for interpolation in tqdm(i_val, leave=False):
            interp = torch.lerp(zq[k], zq[k+1], interpolation)
            images = G.synthesis(interp)
            images = ((images + 1)/2).clamp(0,1)
            pil_image = TF.to_pil_image(images[0].cpu())
            os.makedirs(f'samples/{timestring}', exist_ok=True)
            pil_image.save(f'samples/{timestring}/{count:04}.png')
            count+=1


#VIDEO
from base64 import b64encode
from tqdm.notebook import tqdm
from PIL import Image

fps = count/cut_len

frames = []
# tqdm.write('Generating video...')
for i in sorted(os.listdir(f'samples/{timestring}')): #
    frames.append(Image.open(f"samples/{timestring}/{i}"))

from subprocess import Popen, PIPE
p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', 'video.mp4'], stdin=PIPE)
for im in tqdm(frames):
    im.save(p.stdin, 'PNG')
p.stdin.close()
p.wait()

# !ffmpeg -y -i video.mp4 -i audio.wav -map 0 -map 1:a -c:v copy -shortest video_audio.mp4

# mp4 = open('video.mp4','rb').read()
mp4 = open('video_audio.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


#@markdown P.S.: *If it crushed - look for `video-audio.mp4` in `stylegan3` folder*