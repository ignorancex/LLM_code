import numpy as np
import torch
from numpy import linalg as LA
import torch.nn as nn
import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
# import visdom
import random as rd
import argparse
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy.random as random
from scipy.io import loadmat
from collections import Counter
from PIL import Image
from scipy.sparse import csr_matrix, csc_matrix
from scipy.io import mmread
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

COUNTS_MATRIX = 'counts_matrix_in_vitro.npz'
CLONE_ANNOTATION = 'clone_annotation_in_vitro.npz'
CELL_METADATA = 'cell_metadata_in_vitro.txt'

def obtainInitialData():
    '''
    Helper function for setting up the data loaders. 
    Returns:
        day2 (list): List of indices in cell metadata corresponding to day 2 cells
        day4_6 (pandas.Int64Index): List of indices in cell metadata corresponding to day 4/6 cells
        day4_6_neutrophil (pandas.DataFrame): Contains the portion of cell metadata corresponding to day 4/6 neutrophils 
        day4_6_monocyte (pandas.DataFrame): Contains the portion of cell metadata corresponding to day 4/6 monocytes
        countsInVitroCscMatrix (np.ndarray): Matrix reporting the number of transcripts (UMIs) for each gene in each cell (rows are cells and columns are genes)
        clone_data (np.ndarray): Binary matrix indicating the clonal membership of each cell. 
    '''
    countsInVitroCscMatrix = np.load(COUNTS_MATRIX, mmap_mode='r+')
    clone_data = np.load(CLONE_ANNOTATION)
    clone_data = csc_matrix(
        (clone_data['data'], clone_data['indices'], clone_data['indptr']), shape=(130887, 5864)).toarray()
    countsInVitroCscMatrix = csc_matrix(
        (countsInVitroCscMatrix['data'], countsInVitroCscMatrix['indices'], countsInVitroCscMatrix['indptr']), shape=(130887, 25289)).toarray()
    metadata = pd.read_csv(CELL_METADATA, sep='\\t', header=0)
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6_neutrophil = day4_6.loc[(day4_6['Annotation'] == 'Neutrophil')]
    day4_6_monocyte = day4_6.loc[(day4_6['Annotation'] == 'Monocyte')]
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                          | (day4_6['Annotation'] == 'Monocyte')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    return day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data


def PCA_and_normalize(data_1, data_2, dat1_test, dat2_test, setting):
    '''
    Performs PCA (to 10 dimensions) and L2 normalization on the gene expression data
    Parameters:
        data_1 (np.ndarray): Day 2 geneexpression train data 
        data_2 (np.ndarray): Day 4-6 geneexpression train data
        dat1_test (np.ndarray): Day 2 geneexpression test data 
        dat2_test (np.ndarray): Day 4-6 geneexpression test data 
        setting (str): Either 'supervised', 'unsupervised', or 'semisupervised'. Used for saving the pca model.  
    Returns:
        data_1 (np.ndarray): PCA-transformed and normalized day 2 geneexpression train data
        data_2 (np.ndarray): PCA-transformed and normalized day 4/6 geneexpression train data
        dat1_test (np.ndarray): PCA-transformed and normalized day 2 geneexpression test data 
        dat2_test (np.ndarray): PCA-transformed and normalized day 4/6 geneexpression test data  
    '''
    pca = IncrementalPCA(n_components=100, batch_size=100)
    allElems = np.concatenate((preprocessing.normalize(data_1, norm='l2', axis=0),  preprocessing.normalize(
        data_2, norm='l2', axis=0), preprocessing.normalize(dat1_test, norm='l2', axis=0), preprocessing.normalize(dat2_test, norm='l2', axis=0)), axis=0)
    allElems = pca.fit_transform(allElems)
    pca_name = "pca" + setting + ".p"
    pickle.dump(pca, open(pca_name, "wb"))
    data_1 = allElems[:len(data_1)]
    data_2 = allElems[len(data_1):len(data_1) + len(data_2)]
    dat1_test = allElems[len(data_1) + len(data_2)
                             : len(data_1) + len(data_2) + dat1_test.shape[0]]
    dat2_test = allElems[len(data_1) + len(data_2) + len(dat1_test):len(
        data_1) + len(data_2) + dat1_test.shape[0] + dat2_test.shape[0]]
    return data_1, data_2, dat1_test, dat2_test


def setup_data_loaders_unsupervised():
    '''
    Function for creating the data loaders in the unsupervised setting. Pairs each day 2 cell with a random day 4/6 neutrophil / monocyte. 
    Saves the following files: 
        unsupervised_dat1_train (np.ndarray): Rows correspond to the day 2 train cells, and there are 10 columns corresponding to the gene expression vectors after PCA 
        unsupervised_dat2_train (np.ndarray): Rows correspond to the day 4/6 train cells, and there are 10 columns corresponding to the gene expression vectors after PCA 
        unsupervised_cluster_train (np.ndarray): Binary matrix indicating the cluster membership of each day 2 train cell  
        dat1_test_unsupervised (np.ndarray): Day 2 test data (with 10 columns corresponding to the geneexpression vectors after PCA)
        dat2_test_unsupervised (np.ndarray): Day 4/6 test data (with 10 columns corresponding to the geneexpression vectors after PCA)
        day4_6_unsupervised (np.ndarray): List of indices in cell metadata corresponding to the day 4/6 cells used in the dataset
    '''
    day2, day4_6, _, _, countsInVitroCscMatrix, clone_data = obtainInitialData()
    data_1_train, data_2_train, cluster_train = [], [], []
    train = np.load('train', allow_pickle=True)
    randomIndices = rd.sample(range(0, len(day4_6)), len(train))
    day4_6indices = []
    for i in range(len(train)):
        data_1_train.append(countsInVitroCscMatrix[day2[train[i]]])
        data_2_train.append(countsInVitroCscMatrix[day4_6[randomIndices[i]]])
        cluster_train.append(1*clone_data[day2[train[i]]])
        day4_6indices.append(day4_6[randomIndices[i]])
    dat1_test, dat2_test = np.load('dat1_test_actual', allow_pickle=True), np.load(
        'dat2_test_actual', allow_pickle=True)
    data_1_train, data_2_train, dat1_test, dat2_test = PCA_and_normalize(
        data_1_train, data_2_train, dat1_test, dat2_test, "unsupervised")
    np.array(data_1_train).dump('unsupervised_dat1_train')
    np.array(data_2_train).dump('unsupervised_dat2_train')
    np.array(cluster_train).dump('unsupervised_cluster_train')
    np.array(dat1_test).dump('dat1_test_unsupervised')
    np.array(dat2_test).dump('dat2_test_unsupervised')
    np.array(day4_6indices).dump('day4_6_unsupervised')


def setup_data_loaders_semisupervised(numPoints):
    '''
    Function for creating the data loaders in the semisupervised setting. 
    Parameters: 
        numPoints (int): Denotes the number of supervised points. Can range from 1 to 1526 as there are 1527 day 2 cells. 
    Saves the following files:
        semi_dat1_train[numPoints] (np.ndarray): Rows correspond to the day 2 train cells, and there are 10 columns corresponding to the gene expression vectors after PCA 
        semi_dat2_train[numPoints] (np.ndarray): Rows correspond to the day 4/6 train cells, and there are 10 columns corresponding to the gene expression vectors after PCA 
        dat1_test_semi[numPoints] (np.ndarray): Day 2 test data (with 10 columns corresponding to the geneexpression vectors after PCA)
        dat2_test_semi[numPoints] (np.ndarray): Day 4/6 test data (with 10 columns corresponding to the geneexpression vectors after PCA)
        day4_6_semi[numPoints] (np.ndarray): List of indices in cell metadata corresponding to the day 4/6 cells used in the dataset
    '''
    day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data = obtainInitialData()
    neutrophil_indices = day4_6_neutrophil.index.tolist()
    monocyte_indices = day4_6_monocyte.index.tolist()
    data_1_train, data_2_train, day46_to_day2 = [], [], {}
    train = np.load('train', allow_pickle=True)
    day4_6indices = []
    for i in range(numPoints):
        clone_index = np.where(clone_data[day2[train[i]]] == 1)[0]
        indices = np.where(clone_data[:, clone_index] == 1)[0].tolist()
        indices_n = list(set(indices) & set(neutrophil_indices))
        indices_m = list(set(indices) & set(monocyte_indices))
        n_count = len(indices_n)
        m_count = len(indices_m)        
        cell_index = day4_6_neutrophil.loc[indices_n] if n_count > m_count else day4_6_monocyte.loc[indices_m]
        for row in cell_index.iterrows():
            if row[0] not in day46_to_day2 and not np.isnan(row[1]['Time point']):
                day46_to_day2[row[0]] = day2[train[i]]
                data_1_train.append(countsInVitroCscMatrix[day2[train[i]]])
                data_2_train.append(countsInVitroCscMatrix[row[0]])
                day4_6indices.append(row[0])
                break
    day2X = []
    for i in range(len(train)):
        day2X.append(day2[train[i]])
    remainingday2 = list(set(day2X) - set(day46_to_day2.values()))
    remainingday4_6 = list(set(day4_6) - set(day46_to_day2.keys()))
    randomIndicesday4 = rd.sample(
        range(0, len(remainingday4_6)), len(remainingday2))
    for i in range(len(randomIndicesday4)):
        data_1_train.append(countsInVitroCscMatrix[remainingday2[i]])
        day4_6indices.append(remainingday4_6[randomIndicesday4[i]])
        data_2_train.append(
            countsInVitroCscMatrix[remainingday4_6[randomIndicesday4[i]]])
    dat1_test, dat2_test = np.load('dat1_test_actual', allow_pickle=True), np.load(
        'dat2_test_actual', allow_pickle=True)
    data_1_train, data_2_train, dat1_test, dat2_test = PCA_and_normalize(
        data_1_train, data_2_train, dat1_test, dat2_test, "semi" + str(numPoints))
    np.array(data_1_train).dump('semi_dat1_train'+str(numPoints))
    np.array(data_2_train).dump('semi_dat2_train'+str(numPoints))
    np.array(dat1_test).dump('dat1_test_semi'+str(numPoints))
    np.array(dat2_test).dump('dat2_test_semi'+str(numPoints))
    np.array(day4_6indices).dump('day4_6_semi' + str(numPoints))


def setup_data_loaders_supervised():
    '''
    Function for creating the data loaders in the supervised setting. 
    Saves the following files:
        supervised_dat1_train (np.ndarray): Rows correspond to the day 2 train cells, and there are 10 columns corresponding to the gene expression vectors after PCA
        supervised_dat2_train (np.ndarray): Rows correspond to the day 4/6 train cells, and there are 10 columns corresponding to the gene expression vectors after PCA
        dat1_test_supervised (np.ndarray): Day 2 test data (with 10 columns corresponding to the geneexpression vectors after PCA)
        dat2_test_supervised (np.ndarray): Day 4/6 test data (with 10 columns corresponding to the geneexpression vectors after PCA)
        day4_6_supervised (np.ndarray): List of indices in cell metadata corresponding to the day 4/6 cells used in the dataset
    '''
    day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data = obtainInitialData()
    neutrophil_indices = day4_6_neutrophil.index.tolist()
    monocyte_indices = day4_6_monocyte.index.tolist()
    data_1_train, data_2_train, day46_to_day2 = [], [], {}
    train = np.load('train', allow_pickle=True)
    day4_6indices = []
    for i in range(len(train)):
        ind = day2[train[i]]
        clone_index = np.where(clone_data[ind] == 1)[0]
        indices = np.where(clone_data[:, clone_index] == 1)[0].tolist()
        indices_n = list(set(indices) & set(neutrophil_indices))
        indices_m = list(set(indices) & set(monocyte_indices))
        n_count = len(indices_n)
        m_count = len(indices_m)        
        cell_index = day4_6_neutrophil.loc[indices_n] if n_count > m_count else day4_6_monocyte.loc[indices_m]
        for row in cell_index.iterrows():
            if row[0] not in day46_to_day2 and not np.isnan(row[1]['Time point']):
                day46_to_day2[row[0]] = ind
                data_1_train.append(countsInVitroCscMatrix[ind])
                data_2_train.append(countsInVitroCscMatrix[row[0]])
                day4_6indices.append(row[0])
                break
    dat1_test, dat2_test = np.load('dat1_test_actual', allow_pickle=True), np.load(
        'dat2_test_actual', allow_pickle=True)
    data_1_train, data_2_train, dat1_test, dat2_test = PCA_and_normalize(
        data_1_train, data_2_train, dat1_test, dat2_test, "supervised")
    np.array(data_1_train).dump('supervised_dat1_train')
    np.array(data_2_train).dump('supervised_dat2_train')
    np.array(dat1_test).dump('dat1_test_supervised')
    np.array(dat2_test).dump('dat2_test_supervised')
    np.array(day4_6indices).dump('day4_6_supervised')


def createNewTest():
    '''
    (Call this function BEFORE setting up the data loaders) Function creates a random 80%/20% train /test split of the day 2 undifferentiated cells. 
    Saves the following files: 
        dat1_test_actual (np.ndarray): Day 2 test data, with rows corresponding to day 2 cells, and columns corresponding to the full gene expression vector
        dat2_test_actual (np.ndarray): Day 4/6 test data, with rows corresponding to day 4/6 cells, and columns corresponding to the full gene expression vector
        clusters_test (np.ndarray): Binary matrix indicating the cluster membership of each day 2 test cell
        day2Ind_test (np.ndarray): List of indices in cell metadata corresponding to the day 2 cells used in the test dataset
        day4_6Ind_test (np.ndarray): List of indices in cell metadata corresponding to the day 4/6 cells used in the test dataset
        train (np.ndarray): List of indices in cell metadata corresponding to the day 2 cells used in the train dataset 
    '''
    day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data  = obtainInitialData()
    neutrophil_indices = day4_6_neutrophil.index.tolist()
    monocyte_indices = day4_6_monocyte.index.tolist()
    testIndices = rd.sample(range(len(day2)), 305)
    trainIndices = list(set(range(len(day2))) - set(testIndices))
    day2Ind, day46_ind, dat1_test, dat2_test, day46_to_day2 = [], [], [], [], {}
    clusters = []
    for i in range(len(testIndices)):
        clone_index = np.where(clone_data[day2[testIndices[i]]] == 1)[0]
        indices = np.where(clone_data[:, clone_index] == 1)[0].tolist()
        indices_n = list(set(indices) & set(neutrophil_indices))
        indices_m = list(set(indices) & set(monocyte_indices))
        n_count = len(indices_n)
        m_count = len(indices_m)        
        cell_index = day4_6_neutrophil.loc[indices_n] if n_count > m_count else day4_6_monocyte.loc[indices_m]
        for row in cell_index.iterrows():
            if row[0] not in day46_to_day2 and not np.isnan(row[1]['Time point']):
                day46_to_day2[row[0]] = day2[testIndices[i]]
                dat1_test.append(countsInVitroCscMatrix[day2[testIndices[i]]])
                dat2_test.append(countsInVitroCscMatrix[row[0]])
                day2Ind.append(day2[testIndices[i]])
                day46_ind.append(row[0])
                clusters.append(1*clone_data[day2[testIndices[i]]])
                break
    np.array(dat1_test).dump('dat1_test_actual')
    np.array(dat2_test).dump('dat2_test_actual')
    np.array(clusters).dump('clusters_test')
    np.array(day2Ind).dump('day2Ind_test')
    np.array(day46_ind).dump('day4_6Ind_test')
    np.array(trainIndices).dump('train')

createNewTest()
setup_data_loaders_unsupervised()
setup_data_loaders_semisupervised(300)
setup_data_loaders_supervised()

