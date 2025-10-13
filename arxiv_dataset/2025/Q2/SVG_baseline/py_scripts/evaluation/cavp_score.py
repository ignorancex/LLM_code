"""
Based on the official implementation of DiffFoley (Apache-2.0 license)
https://github.com/luosiallen/Diff-Foley
"""

import os
import torch
import numpy as np
import shutil
import sys
from moviepy.editor import VideoFileClip
import torch.nn.functional as F
import torch.nn as nn
from wav2spec import get_spectrogram
from moviepy.editor import VideoFileClip
import random
from demo_util import Extract_CAVP_Features
import argparse

parser = argparse.ArgumentParser(description='CAVP')
parser.add_argument(
    '--path',
    type=str,
    help=
    'path to the folder where generated sounding videos exist, in the formate of mp4'
)
parser.add_argument('--checkpoints',
                    type=str,
                    default="./diff_foley_ckpt/cavp_epoch66.ckpt",
                    help='path to the pretrained checkpoints')
args = parser.parse_args()

# Set Device:
device = torch.device("cuda")
# Default Setting:

fps = 4  #  CAVP default FPS=4, Don't change it.
batch_size = 40  # Don't change it.
cavp_config_path = "./config/Stage1_CAVP.yaml"  #  CAVP Config
cavp_ckpt_path = args.checkpoints  # "./diff_foley_ckpt/cavp_epoch66.ckpt"  #  CAVP Ckpt

# Initalize CAVP Model:
extract_cavp = Extract_CAVP_Features(fps=fps,
                                     batch_size=batch_size,
                                     device=device,
                                     config_path=cavp_config_path,
                                     ckpt_path=cavp_ckpt_path)

path = args.path
f_list = os.listdir(path)

score_mean = 0
tmp_path = "./generate_samples/temp_folder"
inx = 0
with torch.no_grad():
    for i in f_list:
        name = i.split('.mp4')[0]
        name_ = name
        # while not name in video_list:
        #     continue

        video_path = f"{path}/{name}.mp4"

        # name_=random.choice(f_list).split('.mp4')[0]
        # while (name_==name):
        #     name_=random.choice(f_list).split('.mp4')[0]
        audio_path = f"{path}/{name_}.mp4"

        video_ = VideoFileClip(audio_path)
        audio_clip = video_.audio
        duration_a = video_.duration
        temp_audio_path = f"./{name}.wav"
        audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')

        video = VideoFileClip(video_path)
        duration = video.duration
        duration = min(duration, duration_a)
        start_second = 0  # Video start second
        if duration < 1:
            continue
        inx += 1
        truncate_second = int(
            duration)  #10         # Video end = start_second + truncate_second

        audio = get_spectrogram(temp_audio_path, 16000 * truncate_second)[1]
        os.remove(temp_audio_path)

        # Extract Video CAVP Features & New Video Path:
        cavp_feats, new_video_path = extract_cavp(video_path,
                                                  start_second,
                                                  truncate_second,
                                                  tmp_path=tmp_path)

        spec = torch.from_numpy(audio).unsqueeze(0).unsqueeze(
            1).cuda().float()  # B x 1 x Mel x T
        spec = spec.permute(0, 1, 3, 2)  # B x 1 x T x Mel
        spec_feat = extract_cavp.stage1_model.spec_encoder(spec)  # B x T x C
        spec_feat = extract_cavp.stage1_model.spec_project_head(
            spec_feat).squeeze()
        spec_feat = F.normalize(spec_feat, dim=-1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        score_ = cos(torch.from_numpy(cavp_feats).cuda(), spec_feat)
        score_mean += score_.mean()
        print(inx, video_path, audio_path)

        #if inx==1000: break
    score_mean /= inx
    print(
        f'the mean cosine similarity between video and audio is {score_mean}')
    shutil.rmtree(tmp_path)
