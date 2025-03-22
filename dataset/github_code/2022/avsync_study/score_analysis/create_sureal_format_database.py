import pandas as pd
from ast import literal_eval
import json
import os
import numpy as np
from collections import defaultdict
import glob


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


db_df = pd.DataFrame(list(zip(names,scores,subject_ids,content,offset)),columns=['video','mos_list','subject_ids','content','offset'])
db_df.to_csv('./avsync_for_sureal.csv')


json_db = {}
json_db['ref_videos']=[]
json_db['dis_videos']=[]

ref_count = 0
dis_count = 0

unique_contents = list(db_df['content'].unique())


include_ref = True
for index, row in db_df.iterrows():
    scores = literal_eval(row['mos_list'])
    subject_ids = literal_eval(row['subject_ids'])
    print(subject_ids)
    print(len(subject_ids))
    zipbObj = zip(subject_ids, scores)
    dict_scores= dict(zipbObj)
    if('offset' not in row['video'] ):
            
                
        json_db['ref_videos'].append({'content_id': unique_contents.index(row['content']), 'content_name': row['content'],\
                'path': row['video']})
        ref_count+=1
        if(include_ref):
            json_db['dis_videos'].append({'content_id': unique_contents.index(row['content']), 'asset_id': dis_count,\
                    'os':dict_scores,'path': row['video']})
            dis_count+=1
    else:
    

        json_db['dis_videos'].append({'content_id': unique_contents.index(row['content']), 'asset_id': dis_count,\
                'os':dict_scores,'path': row['video']})
        dis_count+=1

print(len(json_db['dis_videos']))
with open('avsync_surealformat.json', 'w') as outfile:
    json.dump(json_db, outfile)
