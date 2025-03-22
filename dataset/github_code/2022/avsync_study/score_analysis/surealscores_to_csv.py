import pandas as pd 
import json

sureal_json_file = './sureal/output.json'
with open(sureal_json_file) as f:
    sureal_data = json.load(f)
observers = sureal_data['observers']
videos = sureal_data['dis_videos']
orig_csv = pd.read_csv('./avsync_for_sureal.csv')

vids = [v['dis_video_name'] for v in sureal_data['dis_videos']]
offsets = [orig_csv[orig_csv['video']==v]['offset'] for v in sureal_data['dis_videos']]
mos = [v['models']["P910"]["quality_score"] for v in sureal_data['dis_videos']]
mos_std = [v['models']['P910']['quality_score_std'] for v in sureal_data['dis_videos']]
dmos_list= []


for dis_index,v in enumerate(vids):
    print(v)
    content = orig_csv[orig_csv['video']==v]['content'].values[0]
    
    

df_dict = {'video':vids,'mos':mos,'mos_std':mos_std,\
        'content':orig_csv['content'].values,'offset':orig_csv['offset'].values}

df = pd.DataFrame.from_dict(df_dict)
df.to_csv('sureal.csv')