# this file reads all the subjects' raw DMOS zscores and averages them to create a video level score

import os
import numpy as np
from collections import defaultdict
import glob
import pandas as pd

filenames = glob.glob(os.path.join('./zscore','*.csv'))

video_dict = defaultdict(list)

for subject_csv in filenames:
    subject_df = pd.read_csv(subject_csv)
    video_names = subject_df['video']
    for index,v in enumerate(video_names):
        score = subject_df['zscore'].iloc[index]
        mos_score = subject_df['score'].iloc[index]
        video_dict[v].append(mos_score)



print(video_dict)
print(len(video_dict))
scores = []
names = []
std_devs = []
content = []
offset = []

for key,val in video_dict.items():
    print(key)
    print(len(val))
    #if(len(val)<20):
    #    continue
    scores.append(np.mean(val))
    std_devs.append(np.std(val))
    names.append(key)
    split_name = key.split('_')
    content.append(split_name[0])
    offset.append(split_name[2] if 'offset' in key else 0)
score_df = pd.DataFrame(list(zip(names,scores,std_devs,content,offset)),columns=['video','mos','mos_std','content','offset'])
score_df.to_csv('./avsync_mos_score.csv')
