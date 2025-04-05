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
    merged.to_csv('./data/publicationsAndRefTags.csv')
