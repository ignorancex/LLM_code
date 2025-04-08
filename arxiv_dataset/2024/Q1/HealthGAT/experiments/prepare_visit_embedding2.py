import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

import itertools
import collections
import time
import random

from src.utils import PickleUtils
from stellargraph import IndexedArray, StellarGraph

##############################################################
# Create doctor-service table
##############################################################
def doc():
    doc_emb_data = pd.read_parquet('saved_data/visit_init_emb.parquet')
    print(doc_emb_data.shape)
    print(doc_emb_data.head())
    # load trained svc embedding
    X = PickleUtils.loader('saved_data/visit_emb_prime.pkl')
    print(X.shape)
    
    X=pd.DataFrame(X)
    X= X.add_suffix('_emb')
    
    doc_emb_data=pd.concat([doc_emb_data,X],axis=1)
    print(doc_emb_data.shape)
    print(doc_emb_data.head())
    
    
    
    doc_emb_data = doc_emb_data.groupby(['patientunitstayid']).mean().reset_index()
    #doc_emb_data['embedding']=doc_emb_data.iloc[:,3:].values.tolist()
    #doc_emb_data=doc_emb_data[['patientunitstayid','embedding','readmission','seq']]
    print(doc_emb_data.shape)
    print(doc_emb_data['patientunitstayid'].nunique())
    print(doc_emb_data.head())
    
    doc_emb_data.to_parquet('saved_data/visit_temporal_emb.parquet', index=False)
