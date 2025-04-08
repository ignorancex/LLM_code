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
#     doc_table = pd.read_csv('saved_data/carePlanCareProvider.csv')
#     doc_table = doc_table.dropna(subset=['specialty'])
#     print(doc_table[doc_table['patientunitstayid']==141168])
#     enc_spec = doc_table.groupby('patientunitstayid')['specialty'] \
#         .apply(lambda x: x.value_counts().index[0]) \
#         .reset_index(name='specialty')
#     enc_spec.to_csv('saved_data/enc_spec.csv', index=False)
#     print(enc_spec[enc_spec['patientunitstayid']==141168])
    
    pat_table = pd.read_parquet('saved_data/pat_table.parquet')
    print(pat_table['patientunitstayid'].unique())
    med_jny = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')
    #print(med_jny.head())
#     med_jny = med_jny.merge(enc_spec, on='patientunitstayid', how='inner')
    print(med_jny[med_jny['patientunitstayid']==141203])
    maxOffset=med_jny['offset'].max()
    print(maxOffset)
    print(med_jny['offset'].mean())
    print(med_jny[med_jny['patientunitstayid']==141203]['offset'].max())
    for i in range(0,maxOffset,1440):
        med_jny.loc[(med_jny['offset']>i) & (med_jny['offset']<=i+1440), 'seq']=i
#     spec_dict = pd.DataFrame({'specialty':med_jny.specialty.unique(), 'spec_id':list(range(med_jny.specialty.nunique()))})
#     spec_dict.to_csv('saved_data/spec_dict.csv', index=False)

    # load trained svc embedding
    X = np.loadtxt('./node2vec/emb/ppd_eICU.emb', skiprows=1)
    X_coor = np.array([x[1:] for x in X])
    X_id = np.array([int(x[0]) for x in X])
    ii = np.argsort(X_id)
    X_coor = X_coor[ii]
    PickleUtils.saver('saved_data/svc_emb.pkl', X_coor)

    # initialize doctor embedding
    def get_init_emb(x, emb):
        return np.mean(emb[x['svc_id']], axis=0)

    spec_init_emb = med_jny.groupby(['patientunitstayid','seq']).apply(get_init_emb, emb=X_coor).reset_index(name='embedding')
    print(spec_init_emb[spec_init_emb['patientunitstayid']==141203])
    spec_svc = med_jny.groupby(['patientunitstayid','seq'])['svc_id'].apply(lambda x: list(x.unique())) \
        .reset_index(name='svc_id')
    print(spec_svc[spec_svc['patientunitstayid']==141203])
    spec_init_emb = spec_init_emb.merge(spec_svc, on=['patientunitstayid','seq'], how='inner')
    print(spec_init_emb[spec_init_emb['patientunitstayid']==141203])
    spec_init_emb = spec_init_emb.merge(pat_table[['patientunitstayid','readmission']], on='patientunitstayid', how='inner')
    print(spec_init_emb[spec_init_emb['patientunitstayid']==141203])
    def func(x):                                                           
        x['label'] = x['svc_id'].shift(-1)
        return x

    spec_init_emb_new = spec_init_emb.groupby('patientunitstayid').apply(func)
    print(spec_init_emb_new[spec_init_emb_new['patientunitstayid']==141203])
    #spec_init_emb_new['label'].fillna(0, inplace=True)
#     pat = pd.read_csv('saved_data/patient.csv')
#     spec_init_emb_new=spec_init_emb_new.merge(pat[['patientunitstayid','hospitaldischargestatus']],on=['patientunitstayid'], how='inner')
#     print(spec_init_emb_new[spec_init_emb_new['patientunitstayid']==141203])
#     spec_init_emb_new['label'].fillna(spec_init_emb_new['hospitaldischargestatus'], inplace=True)
#     print(spec_init_emb_new[spec_init_emb_new['patientunitstayid']==141203])
#     spec_init_emb = spec_init_emb.merge(enc_spec, on='patientunitstayid', how='inner') \
#         .merge(spec_dict, on='specialty', how='inner') \
#         .drop(columns='specialty')
#     print(spec_init_emb[spec_init_emb['patientunitstayid']==141168])
    spec_init_emb_new.to_parquet('saved_data/visit_init_emb.parquet', index=False)
