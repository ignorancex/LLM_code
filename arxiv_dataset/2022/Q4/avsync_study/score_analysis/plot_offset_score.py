import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 
import glob


def sort_by_x(x,y):
    new_x, new_y = zip(*sorted(zip(x.astype(np.float32), np.asarray(y))))
    return np.asarray(new_x),new_y
    
score_df = pd.read_csv('sureal.csv')
unique_content = score_df['content'].unique()
print(unique_content)

for content in unique_content:
    scores = score_df[score_df['content']==content].mos.values
    offsets = score_df[score_df['content']==content].offset.values
    sorted_offsets,sorted_scores = sort_by_x(offsets,scores)
    sorted_offsets = np.asarray(sorted_offsets).astype(np.float32)
    sorted_scores = np.asarray(sorted_scores).astype(np.float32)
    plt.plot(sorted_offsets,-sorted_scores,'g+',color='green',linestyle='dashed')
    plt.xlabel('Offset (s)')
    plt.ylabel('Acceptability Score')
    plt.title('Acceptability Score vs Offset for '+content)
    plt.savefig(os.path.join('sureal_plots',content+'.png'))
    plt.close()