#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    df = pd.read_csv('./data/wordFreqAbstract.csv')
    cutPoint = np.percentile(df.freq,95)

    fig,ax = plt.subplots()
    sns.barplot(x='word',y='freq',data = df[df.freq > cutPoint],palette = ['blue'])
    ax.set_xticklabels([x.get_text() for x in ax.get_xticklabels()],rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')
    ax.tick_params(size=2.,direction='in')
    
    fig.set_tight_layout(True)

    plt.savefig('./_G/F4/top5PercentWords.pdf')
    plt.close()
    


    
