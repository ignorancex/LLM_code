import pandas as pd 
import numpy as np
import os 
import glob 

generate = False 
if(generate == True):
    files = glob.glob('scores/*.csv')
    for f in files:
        df = pd.read_csv(f,names=['video','score'])
        df['zscore'] = (df['score'] - df['score'].mean()) / df['score'].std()
        
        print(df['video'])
        offset = [v.split('_')[2] if 'offset' in v else 0 for v in df['video'] ]
        df['offset'] = offset
        df['content'] = [v.split('_')[0] for v in df['video']]
        df.to_csv('zscore/'+os.path.basename(f),index=False) 

