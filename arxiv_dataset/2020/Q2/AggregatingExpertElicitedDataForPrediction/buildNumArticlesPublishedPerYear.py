#mcandrew

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from matplotlib.font_manager import FontProperties

def mm2inch(x):
    return x/25.4

def ccdf(x):
    N,x = len(x),sorted(x)
    return x,1.-np.arange(1,N+1)/N

def boldAnnot(ax,x,y,txt,ha='left',va='top',size=12):
    font0 = FontProperties()
    boldFont = font0.copy()
    boldFont.set_weight('bold')
    boldFont.set_size(size)
    ax.text(x,y,txt,transform=ax.transAxes,ha=ha,va=va,fontsize=size,fontproperties=boldFont)

if __name__ == "__main__":

    merged = pd.read_csv('./data/publicationsAndRefTags.csv')

    fig,axs = plt.subplots(1,2)

    ax = axs[0]
    
    year,count = zip(*sorted([ (x,y) for (x,y) in  Counter(merged.PY).items()]))
    prop = [x/sum(count) for x in count]

    ax.plot(year,np.cumsum(prop),'k-')
    ax.plot(year,np.cumsum(prop),'ko')

    ax.set_xlabel('Year published')
    ax.set_ylabel('Cumulative proportion')

    ax.tick_params(size=2.,direction='in')
    boldAnnot(ax=ax,x=0.01,y=0.99,txt='A.',ha='left',va='top',size=12)

    ax = axs[1]
    ax.bar(year,count, color='b')
    ax.set(xlabel = 'Year published', ylabel = 'Number of articles')

    ax.tick_params(size=2.,direction='in')
    boldAnnot(ax=ax,x=0.01,y=0.99,txt='B.',ha='left',va='top',size=12)
    
    fig.set_tight_layout(True)

    fig.set_size_inches(mm2inch(183),mm2inch(183)/1.6)
    plt.savefig('./_G/F2/NumArticlesPublishedPerYear.pdf')
    plt.close()
