'''
@author: Matthew C. Robinson
@email: matthew67robinson@gmail.com

Largely copied/adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/gc/loadData.py
INFO FROM SOURCE SCRIPT:
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0
(see http://www.bioinf.jku.at/research/lsc/LICENSE and
https://github.com/ml-jku/lsc/blob/master/LICENSE)
'''

import numpy as np
import pandas as pd
import itertools
import scipy
import scipy.io
import scipy.sparse
import pickle

with open('./dataPythonReduced/folds0.pckl', 'rb') as f:
    folds = pickle.load(f)

with open('./dataPythonReduced/labelsHard.pckl', 'rb') as f:
    targetMat=pickle.load(f) # shape is num compounds by num assays
    sampleAnnInd=pickle.load(f)
    targetAnnInd=pickle.load(f)

targetMat=targetMat
targetMat=targetMat.copy().tocsr()
targetMat.sort_indices()
targetAnnInd=targetAnnInd
targetAnnInd=targetAnnInd-targetAnnInd.min()

folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
targetMatTransposed=targetMat[sampleAnnInd[list(itertools.chain(*folds))]].T.tocsr()
targetMatTransposed.sort_indices()
trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])

with open('./dataPythonReduced/chembl20Deepchem.pckl', "rb") as f:
    fps=pickle.load(f)

fps_list=[]
for i in range(len(fps)):
    if i%1000==0:
        print(i)
    try:    
        fp = AllChem.GetMorganFingerprintAsBitVect(fps[i],3,nBits=1024)
    except:
        print('failed on:', i)
        fp = None
    fps_list.append(fp)

fps_arr = np.array(fps_list)

with open('./dataPythonReduced/rdkit_ecfp6.pckl', "wb") as f:
    pickle.dump(fps_arr,f)

with open('./dataPythonReduced/rdkit_ecfp6.pckl', 'rb') as f:
    mychemblConvertedMols=pickle.load(f)
    chemblMolsArr=np.arange(len(mychemblConvertedMols))
    
fpInputData=chemblMolsArr
fpSampleIndex=sampleAnnInd

import gc
gc.collect()

allSamples=np.array([], dtype=np.int64)
if not (fpInputData is None):
    allSamples=np.union1d(allSamples, fpSampleIndex.index.values)
if not (fpInputData is None):
    allSamples=np.intersect1d(allSamples, fpSampleIndex.index.values)
allSamples=allSamples.tolist()

if not (fpSampleIndex is None):
    folds=[np.intersect1d(fold, fpSampleIndex.index.values).tolist() for fold in folds]

with open('../data_for_replication/sparse_target_matrix.pckl', 'wb') as f:
    pickle.dump(targetMat,f)

with open('../data_for_replication/folds.pckl', 'wb') as f:
    pickle.dump(folds,f)

with open('../data_for_replication/fp_data.pckl', 'wb') as f:
    pickle.dump(mychemblConvertedMols,f)
