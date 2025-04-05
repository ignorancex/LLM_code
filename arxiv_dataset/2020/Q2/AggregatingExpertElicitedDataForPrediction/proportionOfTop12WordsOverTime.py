#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from stemming.porter2 import stem

import statsmodels.api as sm
import statsmodels.formula.api as smf

def mm2inch(x):
    return x/25.4

def ccdf(x):
    N,x = len(x),sorted(x)
    return x,1.-np.arange(1,N+1)/N

def countWords(X):
    tokenizer = RegexpTokenizer(r'\w+')
    allAbstractText = tokenizer.tokenize(' '.join(X.AB))
    allAbstractText = [stem(x.lower()) for x in allAbstractText]
    
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    allAbstractTextNoStops = [x for x in allAbstractText if x not in default_stopwords]
    
    word2freq = dict(Counter(allAbstractTextNoStops))
    df = {'word':[],'freq':[]}
    for (w,f) in word2freq.items():
        df['word'].append(w)
        df['freq'].append(f)
    df = pd.DataFrame(df)
    df = df.sort_values('freq',ascending=False)
    return df

def countWordsPerArticle(X):
    tokenizer = RegexpTokenizer(r'\w+')

    word2freq = {}
    for (index,article) in X.iterrows():
        abstractText = tokenizer.tokenize(article.AB)
        abstractText = [stem(x.lower()) for x in abstractText]
    
        default_stopwords = set(nltk.corpus.stopwords.words('english'))
        abstractTextNoStops = [x for x in abstractText if x not in default_stopwords]

        for word in dict(Counter(abstractTextNoStops)):
            try:
                word2freq[word]+=1
            except KeyError:
                word2freq[word] = 1
        
    df = {'word':[],'freq':[]}
    for (w,f) in word2freq.items():
        df['word'].append(w)
        df['freq'].append(f)
    df = pd.DataFrame(df)
    df = df.sort_values('freq',ascending=False)
    return df

def convert2dict(X):
    return X.set_index('word').to_dict()['freq'] 

def findTopWordsyOverallPrev(merged):
    year2proportions = {}
    for year in np.arange(1992,2018+1):
        subset = merged[merged.PY <= year]
        year2proportions[year] = convert2dict( countWords(subset) )
    
    df = pd.DataFrame(year2proportions)
    df = df.replace(np.nan,0)
    df = df.apply( lambda x: x/sum(x),0)

    df['wrds'] = df.index
    df = df.melt(id_vars = ['wrds'])
    df['variable'] = df.variable.astype(float)

    df = df.rename(columns = {'variable':'year','value':'prop'})
    
    top2018Words = df[df.year==2018].sort_values('prop')[-12:]['wrds']
    top2018Words = list(top2018Words[::-1])
    return top2018Words

def countNumOfWords(X):
    return len(' '.join(X.AB).split(' '))    

def transformArticlesPerYear2Counts(year2ArticleCounts):
    Y2A = pd.DataFrame(year2ArticleCounts)
    Y2A = Y2A.replace(np.nan,0)

    Y2A['wrds'] = Y2A.index

    Y2A = Y2A.melt(id_vars = ['wrds'])

    Y2A = Y2A.rename(columns = {'value':'articleCount','variable':'year'})
    Y2A['year'] = Y2A.year.astype(int)

    Y2A = Y2A.merge( year2NumArticles, on = ['year'] )
    Y2A['AS'] = Y2A.articleCount
    Y2A['AF'] = Y2A.numArticles - Y2A.AS
    return Y2A

def transformWordsPerYear2Counts(year2WordCounts):
    Y2W = pd.DataFrame(year2WordCounts)
    Y2W = Y2W.replace(np.nan,0)

    Y2W['wrds'] = Y2W.index

    Y2W = Y2W.melt(id_vars = ['wrds'])

    Y2W = Y2W.rename(columns = {'value':'wordCount','variable':'year'})
    Y2W['year'] = Y2W.year.astype(int)

    Y2W = Y2W.merge( year2NumWords, on = ['year'] )
    Y2W['WS'] = Y2W.wordCount
    Y2W['WF'] = Y2W.numWords - Y2W.WS
    return Y2W



if __name__ == "__main__":

    allAns = pd.read_csv('data/answers2CheckList/allAnswers__20191218.csv')
    #allAns = pd.read_csv('../../code/allAnswers__2019WithoutNoAnswers.csv')
    allAns = allAns.replace('Y/N',-99)
    allAns =allAns[~allAns.manuscript.isna()]

    dois = []
    for di in allAns.manuscript:
        di = di.replace("b'",'').replace("'",'').replace(':','/').replace('.pdf','') 
        dois.append(di)
    allAns['DI'] = dois
    allAns['DI'] = [x.lower() for x in allAns.DI]

    text = pd.read_csv('./data/WoSTitlesAndAbstracts/refTags_addedMissingDOIs.csv')
    text['DI'] = [str(x).lower() for x in text.DI]
    
    merged = text.merge(allAns,on=['DI'])

    year2ArticleCounts,year2WordCounts = {},{}
    year2NumArticles = {'year':[],'numArticles':[]}
    year2NumWords    = {'year':[],'numWords':[]}
    for year in np.arange(1992,2018+1):
        subset = merged[merged.PY == year]

        year2NumArticles['year'].append(year)
        year2NumArticles['numArticles'].append(subset.shape[0])

        year2NumWords['year'].append(year)
        year2NumWords['numWords'].append(countNumOfWords(subset))
        
        year2WordCounts[year]    = convert2dict( countWords(subset) )
        year2ArticleCounts[year] = convert2dict( countWordsPerArticle(subset) )

    year2NumArticles = pd.DataFrame(year2NumArticles)
    year2NumWords    = pd.DataFrame(year2NumWords)
    
    Y2A = transformArticlesPerYear2Counts(year2ArticleCounts)
    Y2W = transformWordsPerYear2Counts(year2WordCounts)

    Y2B = Y2A.merge(Y2W,left_on=['year','wrds'],right_on=['year','wrds'])

    top2018Words = findTopWordsyOverallPrev(merged)
    
    D = Y2B
    D = D.loc[[ True if x in top2018Words else False  for x in D.wrds] ,: ]
    D['P'] = D.WS/(D.WS+D.WF)
    
    n=0
    colors = sns.color_palette(n_colors=12)
    r,c= 0,0
    fig,axs = plt.subplots(3,4)
    for w in top2018Words:
        ax = axs[r,c]
       
        subset = D.loc[D.wrds==w,:]
        BP = sns.barplot(x='year',y='P',data=subset,ax=ax,color=colors[n])
        n+=1
        
        ax.tick_params(direction='in',size=1.)

        if r == 0: 
           ax.set_ylim(0,0.06) 
        elif r == 1:
            ax.set_ylim(0,0.04)
        elif r == 2:
            ax.set_ylim(0,0.03)

        ax.set_xlabel('')

        xticks = np.array(subset.year)
        xticks = [ '{:d}'.format(x) if n % 5 ==0 else ''  for (n,x) in enumerate(xticks)]
        ax.set_xticklabels(xticks,fontsize=6)

        ax.set_yticks(ax.get_yticks()[1:])
        ax.set_yticklabels(['{:d}%'.format(int(100*y)) for y in ax.get_yticks()],fontsize=7)
        ax.set_ylabel('')
        
        ax.text(x=0.99,y=0.99,s=w,transform=ax.transAxes,ha='right',va='top',fontsize=10)
 
        c+=1
        if c == 4:
            r+=1;c=0

    fig.text(0,0.50,'Prob. word appears in abstract',rotation=90,va='center',ha='left')
            
    fig.set_size_inches(mm2inch(183.),mm2inch(183)/1.6)
    fig.set_tight_layout(True)
    plt.savefig('./_G/F3/proportionOfTop12WordsOverTime.pdf')
    plt.close()
