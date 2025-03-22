import os
import numpy as np
from collections import defaultdict
import glob
import pandas as pd

filenames = glob.glob(os.path.join('./scores','*.csv'))

video_dict = defaultdict(list)
subject_dict=defaultdict(list)

for subject_csv in filenames:

    num = int(os.path.splitext(os.path.basename(subject_csv))[0].split('_')[2])
    print(num)
    
    subject_df = pd.read_csv(subject_csv,names=['video','score'],header=0)
    video_names = subject_df['video']
    for index,v in enumerate(video_names):
        score = subject_df['score'].iloc[index]
        video_dict[v].append(score)
        subject_dict[v].append(str(num))



print(video_dict)
print(len(video_dict))
scores = []
names = []
std_devs = []
content = []
offset = []
subject_ids = []

for key,val in video_dict.items():
    print(key)
    print(len(val))
    #if(len(val)<20):
    #    continue
    subject_ids.append(subject_dict[key])
    scores.append(val)
    names.append(key)
    split_name = key.split('_')
    content.append(split_name[0])
    offset.append(split_name[2] if 'offset' in key else 0)


score_df = pd.DataFrame(list(zip(names,scores,subject_ids,content,offset)),columns=['video','mos_list','subject_ids','content','offset'])
score_df.to_csv('./avsync_for_sureal.csv')