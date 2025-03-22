import pandas as pd 
import numpy as np
import glob
from scipy.stats import ttest_ind
import os 

from collections import defaultdict

filenames = glob.glob(os.path.join('./zscore','*.csv'))

video_dict = defaultdict(list)
subject_dict=defaultdict(list)

for subject_csv in filenames:

    num = int(os.path.splitext(os.path.basename(subject_csv))[0].split('_')[2])
    print(num)
    
    subject_df = pd.read_csv(subject_csv)
    offset_200ms = subject_df[subject_df['video'].str.contains('offset_-0.2_')]
    video_names = offset_200ms['video']
    for index,v in enumerate(video_names):
        score = offset_200ms['zscore'].iloc[index]
        video_dict[v].append(score)

for key,val in video_dict.items():
    if('newsreader' in key):
        for key2,val2 in video_dict.items():
            if(key!=key2):
                print(key,key2)
                print(np.mean(val),np.mean(val2))
                print(ttest_ind(val,val2))


