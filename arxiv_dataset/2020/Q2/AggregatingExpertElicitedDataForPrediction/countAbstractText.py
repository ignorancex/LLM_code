#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from stemming.porter2 import stem 

def ccdf(x):
    N,x = len(x),sorted(x)
    return x,1.-np.arange(1,N+1)/N

if __name__ == "__main__":

    allAns = pd.read_csv('./data/answers2CheckList/allAnswers__20191218.csv')
    allAns = allAns.replace('Y/N',-99)
    allAns = allAns[ ~(allAns['Identify the primary predictive target?']=='predictive target')]
    allAns =allAns[~allAns.manuscript.isna()]

    dois = []
    for di in allAns.manuscript:
        di = di.replace("b'",'').replace("'",'').replace(':','/').replace('.pdf','') 
        dois.append(di)
    allAns['DI'] = dois

    text = pd.read_csv('./data/WoSTitlesAndAbstracts/refTags.csv')
    
    merged = text.merge(allAns,on=['DI'])

    tokenizer = RegexpTokenizer(r'\w+')
    allAbstractText = tokenizer.tokenize(' '.join(merged.AB))
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
    df.to_csv('./data/wordFreqAbstract.csv')
   
