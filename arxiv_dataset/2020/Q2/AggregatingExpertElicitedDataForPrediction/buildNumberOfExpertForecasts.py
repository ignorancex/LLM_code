#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

def ccdf(x):
    N,x = len(x),sorted(x)
    return x,1.-np.arange(1,N+1)/N
def mm2inch(x):
    return 1.0*x/25.4

def boldAnnot(ax,x,y,txt,ha='left',va='top',size=12):
    font0 = FontProperties()
    boldFont = font0.copy()
    boldFont.set_weight('bold')
    boldFont.set_size(size)
    ax.text(x,y,txt,transform=ax.transAxes,ha=ha,va=va,fontsize=size,fontproperties=boldFont)

if __name__ == "__main__":

    allAns = pd.read_csv('./data/answers2CheckList/allAnswers__20191218.csv')
    print(allAns.columns)
    
    numForecasts = []
    for x in allAns['How many total forecasts were made?']:
        if x == '6 models, each produced 3 forecasts (2005,2010,2015 power quantity)':
            x = 6
        if x == '5 (scenarios)':
            x = 5
        
        try:
            numForecasts.append(int(x))
        except:
            pass
    numForecasts = np.array(numForecasts)
    
    fig,axs = plt.subplots(1,2)

    ax = axs[0]
    x,px = ccdf(numForecasts)
    ax.semilogx(x,px)
    ax.set_xlabel('Total number of forecasts',fontsize=10)
    ax.set_ylabel(r'Prob. of collecting more than $f$ forecasts $P(F>f)$',fontsize=10)

    ax.axvline(10**0, color = 'k', ls='--', alpha=0.20)
    ax.axvline(10**1, color = 'k', ls='--', alpha=0.20)
    ax.axvline(10**2, color = 'k', ls='--', alpha=0.20)
    ax.axvline(10**3, color = 'k', ls='--', alpha=0.20)
    ax.axvline(10**4, color = 'k', ls='--', alpha=0.20)
    ax.axvline(10**5, color = 'k', ls='--', alpha=0.20)

    ax.tick_params(direction='in',size=2)
    boldAnnot(ax=ax,x=0.99,y=0.99,txt='A.',ha='right',va='top',size=12)  
    
    discretized = {'cat':[],'count':[]}
    
    discretized['cat']    = [r'$<10$']
    discretized['count'] = [sum(numForecasts<10)]

    discretized['cat'].append( r'$<100$' )
    discretized['count'].append(sum(numForecasts<100))

    discretized['cat'].append(r'$<10^3$')
    discretized['count'].append(sum(numForecasts<1000))

    discretized['cat'].append(r'$<10^4$')
    discretized['count'].append(sum(numForecasts<10**4))

    discretized['cat'].append(r'$<10^5$')
    discretized['count'].append(sum(numForecasts<=10**5))
    
    discretized = pd.DataFrame(discretized)
    discretized['prop'] = discretized['count']/len(numForecasts)

    ax = axs[1]
    sns.barplot(x='cat',y='count',data=discretized,ax=ax,palette=['blue'])
    ax.set_xlabel('Total number of forecasts',fontsize=10)
    ax.set_ylabel('Number of articles',fontsize=10)

    ax.tick_params(direction='in',size=2)
    boldAnnot(ax=ax,x=0.99,y=0.99,txt='B.',ha='right',va='top',size=12)  

    fig.set_size_inches(1.2*mm2inch(183),mm2inch(183)/1.6)
    fig.set_tight_layout(True)
    plt.savefig('./_G/F6/numberOfForecasts.pdf')
    plt.close()
    
