import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import pandas as pd
from scipy import stats, linalg
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fsolve, least_squares, minimize, curve_fit
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
import cvxpy as cp
from sympy import symbols, Eq, solve
from os.path import join
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from joblib import Parallel, delayed
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import h5py
import tensorflow as tf
from pointnet_implementation import *
from kscore.estimators import *
from kscore.kernels import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EOT_tools import *

#Code to perform point cloud classification experiment

random.seed = 1

#Data processing

#Identifying labels for point cloud types in pointcloud-c used in the main paper.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_obj_dict={0:'plane',2:'bed', 17:'guitar', 22:'television',37:'vases'}
label_list=[0,2,17,22,37]
ordered_labels={0:0,2:1,17:2,22:3,37:4}
invert_ordered={0:0,1:2,2:17,3:22,4:37}

pointnet = PointNet()
pointnet.to(device);
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.0002)

#Class for dictionary entries. ''data'' is array of 3D arrays, labels is an integer, ''coefficients'' is a probability vector with length= # reference pointclouds.
class entry:
    def __init__(self,labels,data,coefficients):
        self.labels=labels
        self.data=data
        self.coefficients=coefficients

def get_labels(obj):
    return obj.labels
def split(input_list, size1, size2):
    if size1 + size2 != len(input_list):
        raise ValueError("Sizes do not match the length of the input list")
    shuffled_list = input_list.copy()
    permutation = list(range(len(shuffled_list)))
    random.shuffle(permutation)
    shuffled_list = [shuffled_list[i] for i in permutation]
    list1 = shuffled_list[:size1]
    list2 = shuffled_list[size1:]
    return list1, list2, permutation

def apply_permutation(input_list, permutation):
    if len(input_list) != len(permutation):
        raise ValueError("Input list and permutation must be of the same length") 
    return [input_list[i] for i in permutation]


def map_labels(labels,dictionary):
    new_labels=[]
    for i in labels:
        new_labels.append(dictionary[int(i[0])])
    return np.array(new_labels)

#NN utilities: Functions for training pointnet.

def train(model,batchlist, val_loader=None,  epochs=4):
    for epoch in range(epochs): 
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(batchlist, 0):
            shuffled_batch=np.random.permutation(batchlist[i])
            point_list=[]
            label_list=[]
            for pointcloud in shuffled_batch:
                point_list.append(pointcloud.data)
                label_list.append(pointcloud.labels)
            point_list=np.array(point_list)
            point_list=np.transpose(point_list,(0,2,1))
            label_list=np.array(label_list)
            label_list=map_labels(label_list,ordered_labels)
            inputs, labels = torch.from_numpy(point_list).to(device).float(), torch.from_numpy(label_list).to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs)
            loss = pointnetloss(outputs, labels.squeeze(), m3x3, m64x64)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:   
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(batchlist), running_loss / 10))
                running_loss = 0.0
        pointnet.eval()
        correct = total = 0
        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

def compute_accuracy(pred_labels, true_labels):
    assert len(pred_labels) == len(true_labels), "Length of predicted labels and true labels must be the same."

    correct = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)
    total = len(pred_labels)
    accuracy = (correct / total) * 100
    return accuracy

def score_results(list_of_entr,atoms):
    maxes=[]
    coeff_list=[]
    entry_label_list=[]
    dictionary_labels=[]
    for entr in list_of_entr:
        coeff_list.append(entr.coefficients)
        entry_label_list.append(entr.labels)
    for i in atoms:
        dictionary_labels.append(i.labels)
    dictionary_labels=np.array(dictionary_labels).flatten()
    dictionary_labels=list(set(dictionary_labels))
    dictionary_labels.sort()
    
    for coeff in coeff_list:
        maximum=np.argmax(coeff)
        maxes.append(dictionary_labels[maximum])
    maxes=np.array(maxes)
    entry_label_list=np.array(entry_label_list).flatten()
    score_vec=(maxes != entry_label_list).astype(int)
    return score_vec

def reset_weights(model):
    def weights_init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.normal_(m.weight, mean=0, std=0.1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    model.apply(weights_init)

#Data Preprocessing

#The pointcloud corruptions are (in order): dropout_local_1, dropout_local_2, dropout_global_4, dropout_jitter_4, and add_global_4
#The processed ''clean'' pointclodus are stored in ''batch_list''. The corrupted pointclouds are stored in ''corrupted_batch_list''.

def load_data(file_path, label_list):
    with h5py.File(file_path, 'r') as f:
        data_file = f['data']
        label_file = f['label']
        dataset = data_file[:]
        all_labels = label_file[:]
    reduced_data_indices = [find_indices(all_labels, j) for j in label_list]
    reduced_labeled_data = [[labeled_data(dataset[j], all_labels[j]) for j in indices] for indices in reduced_data_indices]
    return np.transpose(np.array(reduced_labeled_data))

def prepare_data():
    batch_list = load_data('pointcloud-c/clean.h5', label_list)
    corrupted_batchlist = np.array([
        load_data('pointcloud-c/dropout_local_1.h5', label_list),
        load_data('pointcloud-c/dropout_local_2.h5', label_list),
        load_data('pointcloud-c/dropout_global_4.h5', label_list),
        load_data('pointcloud-c/jitter_4.h5', label_list),
        load_data('pointcloud-c/add_local_4.h5', label_list)
    ]).transpose((0, 2, 1))
    return batch_list, corrupted_batchlist
######
#Classification using Sinkhorn, entropy-regularized and unregularized functionals
#For ''benchmark_trial'', data=''batchlist'', corrupted_batches=''corrupted_batchlist'', m= # reference pointclouds
#functional can either be "Entropy", "Sinkhorn" or "Unregularized".

def benchmark_trial(train_data,test_data,permutation,corrupted_batches,m,regularization,inner_trials,functional):
    time_vec=[]
    #Split data into train and test sets (80 train, 20 test for each class of pointclouds). m reference point clouds are selected from the training data
    train_data_points=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        train_data_points.append(type_bin)
    #Label arrays for training and test sets are constructed.
    train_data_labels=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        train_data_labels.append(type_bin)
    test_data_points=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        test_data_points.append(type_bin)
    test_data_labels=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        test_data_labels.append(type_bin)
    #Corrupted point cloud test arrays are constructed. 
    corrupted_data=corrupted_batches
    corrupted_test_data=[]
    for corr_batch in corrupted_data:
        corr_test_set=[]
        for cloud_type in corr_batch:
            test_batch=[]
            shuffled=apply_permutation(cloud_type,permutation)
            i_test=shuffled[80:]
            corr_test_set.append(i_test)
        corrupted_test_data.append(corr_test_set)
    corrupted_test_data.append(test_data)
    corrupted_test_data_copy=corrupted_test_data.copy()
    corrupted_test_data_copy=np.array(corrupted_test_data_copy).transpose(0,2,1)
    corrupted_test_data=np.reshape(corrupted_test_data,(6,100))
    
    #Barycentric coding classification

    big_wass_learning_vec=[]
    for trial in np.arange(inner_trials):
        start_time = time.time()
        test_entry_list=[]
        #Convert "labeled_data" objects to "entry" objects, default coefficient set to 0. 
        for corr_batch in corrupted_test_data:
            corr_test_entry_list=[]
            for pointcloud in corr_batch:
                points=pointcloud.data
                label=pointcloud.labels
                coefs=0
                new_entry=entry(label,points,coefs)
                corr_test_entry_list.append(new_entry)
            test_entry_list.append(corr_test_entry_list)

        references=[]
        for i in train_data:
            sampled_ref=random.sample(list(i),k=m)
            references.append(sampled_ref)
        reference_list=list(np.array(references).flatten())
        sorted_references = sorted(reference_list, key=get_labels)
        small_wass_learning_vec=[]
        counter=0

        for i in corrupted_test_data:
            counter=counter+1
            #Sinkhorn functional
            if functional=='Sinkhorn':
                learned_entries=sink_dictionary_learn(i,sorted_references,regularization,m)

            #Unregularized functional/Tangential Wasserstein projection
            elif functional=='Unregularized':
                learned_entries=tangent_dictionary_learn(i,sorted_references,m)  

            #Entropy-regularized functional
            elif functional=='Entropy':
                learned_entries=dictionary_learn(i,sorted_references,regularization,m)

            wass_learning_results=1-np.mean(score_results(list(learned_entries),sorted_references))
            small_wass_learning_vec.append(wass_learning_results)
        big_wass_learning_vec.append(small_wass_learning_vec)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_vec.append(elapsed_time)
        print(trial)

    #NN classification using PointNet

    reset_weights(pointnet)
    train(pointnet,np.transpose(train_data))
    pointnet.eval()
    all_preds = []
    all_labels = []
    nn_result_vec=[]
    with torch.no_grad():
        for corr_batch in corrupted_test_data_copy:
            for i, data in enumerate(corr_batch):
                #print(i,data)
                print('Batch [%4d / %4d]' % (i+1, len(corr_batch)))
                shuffled_batch=np.random.permutation(data)
                point_list=[]
                label_list=[]
                for pointcloud in shuffled_batch:
                    point_list.append(pointcloud.data)
                    label_list.append(pointcloud.labels)
                label_list=map_labels(label_list,ordered_labels)
                inputs, labels = torch.from_numpy(np.array(point_list)).to(device).float(), torch.from_numpy(np.array(label_list)).to(device)
                outputs, __, __ = pointnet(inputs.transpose(1,2))
                _, preds = torch.max(outputs.data, 1)
                all_preds += list(preds.numpy())
                all_labels += list(labels.numpy())
            nn_results=compute_accuracy(all_preds,all_labels)
            nn_result_vec.append(nn_results)
    return big_wass_learning_vec,nn_result_vec,time_vec,train_data_points,test_data_points,train_data_labels,test_data_labels

def test_train_split(data):
    data=np.transpose(data)
    #Split data into train and test sets (80 train, 20 test for each class of pointclouds). m reference point clouds are selected from the training data
    train_data=[]
    test_data=[]
    for i in data:
        i_train,i_test,permutation=split(i,80,20)
        train_data.append(i_train)
        test_data.append(i_test)
    return train_data,test_data,permutation

def benchmark_experiment(train_data,test_data,permutation,inner_trials,outer_trials,functional,reg):
    wass_learning_vec=[]
    nn_learning_vec=[]
    os.mkdir('{}_folder_{}'.format(functional,reg))
    for i in np.arange(outer_trials):
        A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_data,test_data,permutation,corrupted_batchlist,3,reg,inner_trials,functional)
        np.save('{}_folder_{}/wass_trial_{}'.format(functional,reg,i),A)
        np.save('{}_folder_{}/nn_trial_{}'.format(functional,reg,i),B)
        np.save('{}_folder_{}/time_vec_{}'.format(functional,reg,i),time_vec)
        wass_learning_vec.append(A)
        nn_learning_vec.append(B)
        print(i)
    return wass_learning_vec,nn_learning_vec

###########
#Classification experiment using the doubly regularized functional
#Score estimation relies on supplemental code to ``Nonparametric Score Estimation'' by Yuhao Zhou, Jiaxin Shi, Jun Zhu. https://github.com/miskcoo/kscore
##### Doubly reg classification tools
from kscore.kernels import *
from kscore.estimators import *

def add_isotropic_gaussian_noise(points, mean, std_dev):
    noise = np.random.normal(loc=mean, scale=std_dev, size=points.shape)
    noisy_points = points + noise
    return noisy_points

#Estimates the score function for a given pointcloud. 
#Different estimators and kernels can be chosen, see kscore/estimators and kscore/kernels for details. 
def score_estimate(data,kernel_width):
    nu_estimator = NuMethod(lam=0.00001, kernel=CurlFreeIMQ())
    estimator=nu_estimator
    estimator.fit(data,kernel_hyperparams=kernel_width)
    return estimator

#grad_field perturbs the data with isotropic gaussian noise of a given standard deviation, and then estimates the score function of the perturbed data.
def grad_field(data,kernel_width,noise_stdv):
    noised_data=add_isotropic_gaussian_noise(data,0.0,noise_stdv)
    noised_data=tf.cast(noised_data,dtype=tf.float32)
    estimator=score_estimate(noised_data,kernel_width)
    score_field=estimator.compute_gradients(data)
    return score_field

#doubly-reg_analysis_noise computes the barycentric coding coefficient vector for a given array of points (base_measure_points) and reference measure points. 
#Our choice of inner_regularization=0.01 and outer_regularization=0.009. See https://arxiv.org/pdf/2303.11844 for details.
def doubly_reg_analysis_noise(reference_measure_points,base_measure_points,inner_regularization,outer_regularization):
    uniform_masses=np.dot(np.ones(len(base_measure_points)),1/len(base_measure_points))
    base_measure=measure(base_measure_points,uniform_masses)
    grad=grad_field(base_measure_points,5.0,0.05)
    grad=np.dot(grad,outer_regularization)
    reference_measures=[]
    for i in reference_measure_points:
        reference_mass=np.dot(np.ones(len(i)),1/len(i))
        reference_meas=measure(i,reference_mass)
        reference_measures.append(reference_meas)
    entmap_list=[]
    def process_reference_map(reference_measure):
        g = get_potential(base_measure, reference_measure, inner_regularization)
        extended = highdim_extended_map(g[1], base_measure, reference_measure, inner_regularization, 3)
        return extended
    
    entmap_list = Parallel(n_jobs=-1)(delayed(process_reference_map)(i) for i in reference_measures)
    l2norms=[]
    for p in np.arange(len(entmap_list)):
        for q in np.arange(len(entmap_list)):
            T_p=base_measure_points-entmap_list[p]+grad
            T_q=base_measure_points-entmap_list[q]+grad
            dotvec=[]
            for k in np.arange(len(T_p)):
                innerprod=np.dot(T_p[k],T_q[k])
                dotvec.append(innerprod)
            l2diff=np.dot(dotvec,uniform_masses)
            l2norms.append(l2diff)
    l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
    x=cp.Variable(len(entmap_list))
    objective=cp.Minimize(cp.quad_form(x,l2matrix))
    constraints=[x>=0,cp.sum(x)==1]
    problem=cp.Problem(objective,constraints)
    problem.solve()
    optimal_x=x.value
    return optimal_x

#doubly_reg_build_coefficients_noise updates a list of ''entry'' objects by changing their coefficients to barycentric coding coefficients for a given set of labeled reference point clouds (''dictionary'').
def doubly_reg_build_coefficients_noise(list_of_entries,dictionary,inner_regularization,outer_regularization):
    new_list=[]
    dictionary_points_list=[]
    counter=0
    for atom in dictionary:
        dictionary_points_list.append(atom.data)
    for entr in list_of_entries:
        counter=counter+1
        print(counter)
        coeff=doubly_reg_analysis_noise(dictionary_points_list,entr.data,inner_regularization,outer_regularization)
        new_entry=entry(entr.labels,entr.data,coeff)
        new_list.append(new_entry)
    return new_list

def doubly_reg_noise_dictionary_learn(list_of_entries,dictionary,inner_regularization,outer_regularization,m):
    coefficients=doubly_reg_build_coefficients_noise(list_of_entries,dictionary,inner_regularization,outer_regularization)
    processed_coefficients=combine_coefficients_list(coefficients,m)
    return processed_coefficients

#Classification using doubly-regularized functional
#For data=''batchlist'', corrupted_batches=''corrupted_batchlist'', m= # reference pointclouds
#In our experiments, outer_regularization=0.01, inner_regularization=0.009

def doubly_reg_benchmark_trial(train_data,test_data,permutation,corrupted_batches,m,inner_regularization,outer_regularization,inner_trials):
    time_vec=[]
    train_data_points=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        train_data_points.append(type_bin)
    train_data_labels=[]
    for i in train_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        train_data_labels.append(type_bin)
    test_data_points=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.data)
        test_data_points.append(type_bin)
    test_data_labels=[]
    for i in test_data:
        type_bin=[]
        for j in i:
            type_bin.append(j.labels)
        test_data_labels.append(type_bin)
    corrupted_data=corrupted_batches
    corrupted_test_data=[]
    for corr_batch in corrupted_data:
        corr_test_set=[]
        for cloud_type in corr_batch:
            test_batch=[]
            shuffled=apply_permutation(cloud_type,permutation)
            i_test=shuffled[80:]
            corr_test_set.append(i_test)
        corrupted_test_data.append(corr_test_set)
    corrupted_test_data.append(test_data)
    corrupted_test_data_copy=corrupted_test_data.copy()
    corrupted_test_data_copy=np.array(corrupted_test_data_copy).transpose(0,2,1)
    corrupted_test_data=np.reshape(corrupted_test_data,(6,100))
  #Wass learning
    coefficient_vec=[]
    big_wass_learning_vec=[]
    for trial in np.arange(inner_trials):
        start_time = time.time()
        test_entry_list=[]
        for corr_batch in corrupted_test_data:
            corr_test_entry_list=[]
            for pointcloud in corr_batch:
                points=pointcloud.data
                label=pointcloud.labels
                coefs=0
                new_entry=entry(label,points,coefs)
                corr_test_entry_list.append(new_entry)
            test_entry_list.append(corr_test_entry_list)
        references=[]
        for i in train_data:
            sampled_ref=random.sample(list(i),k=m)
            references.append(sampled_ref)
        reference_list=list(np.array(references).flatten())
        sorted_references = sorted(reference_list, key=get_labels)
        small_wass_learning_vec=[]
        for i in corrupted_test_data:
            noise_learned_entries=doubly_reg_noise_dictionary_learn(i,sorted_references,inner_regularization,outer_regularization,m)
            coefficient_vec.append(noise_learned_entries)
            noise_learning_results=1-np.mean(score_results(list(noise_learned_entries),sorted_references))
            print(noise_learning_results)
            small_wass_learning_vec.append(noise_learning_results)
        big_wass_learning_vec.append(small_wass_learning_vec)
        print(trial)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_vec.append(elapsed_time)
    #NN
    reset_weights(pointnet)
    train(pointnet,np.transpose(train_data))
    pointnet.eval()
    all_preds = []
    all_labels = []
    nn_result_vec=[]
    with torch.no_grad():
        for corr_batch in corrupted_test_data_copy:
            for i, data in enumerate(corr_batch):
                #print(i,data)
                print('Batch [%4d / %4d]' % (i+1, len(corr_batch)))
                shuffled_batch=np.random.permutation(data)
                point_list=[]
                label_list=[]
                for pointcloud in shuffled_batch:
                    point_list.append(pointcloud.data)
                    label_list.append(pointcloud.labels)
                label_list=map_labels(label_list,ordered_labels)
                inputs, labels = torch.from_numpy(np.array(point_list)).to(device).float(), torch.from_numpy(np.array(label_list)).to(device)
                outputs, __, __ = pointnet(inputs.transpose(1,2))
                _, preds = torch.max(outputs.data, 1)
                all_preds += list(preds.numpy())
                all_labels += list(labels.numpy())
            nn_results=compute_accuracy(all_preds,all_labels)
            nn_result_vec.append(nn_results)
    return big_wass_learning_vec,nn_result_vec,time_vec,train_data_points,test_data_points,train_data_labels,test_data_labels,coefficient_vec



#Script

os.mkdir('{}_folder_{}'.format('Sinkhorn',0.009))
os.mkdir('{}_folder_{}'.format('Entropy',0.009))
os.mkdir('{}_folder_{}'.format('Sinkhorn',0.089))
os.mkdir('{}_folder_{}'.format('Entropy',0.089))
os.mkdir('{}_folder_{}'.format('Sinkhorn',0.75))
os.mkdir('{}_folder_{}'.format('Entropy',0.75))
os.mkdir('{}_folder_{}'.format('Unregularized',0))
os.mkdir('dreg_folder')
os.mkdir('nn_folder')

outer_trials=15
loweps_sinkhorn_vec=[]
loweps_entropy_vec=[]
mideps_sinkhorn_vec=[]
mideps_entropy_vec=[]
higheps_sinkhorn_vec=[]
higheps_entropy_vec=[]
unreg_vec=[]
dreg_vec=[]
nn_vec=[]

batch_list,corrupted_batchlist=prepare_data()

for i in np.arange(outer_trials):
    train_split,test_split,perm=test_train_split(batch_list)
    
    #Unreg
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0,1,'Unregularized')
    np.save('{}_folder_{}/wass_trial_{}'.format('Unregularized',0,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Unregularized',0,i),time_vec)
    np.save('nn_folder/nn_trial_{}'.format(i),B)
    unreg_vec.append(A)
    nn_vec.append(B)

    #Doubly
    A,B,time_vec,train_points,test_points,train_labels,test_labels,coefficient_vec=doubly_reg_benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.009,0.01,1)
    np.save('dreg_folder/wass_trial_{}'.format(i),A)
    np.save('dreg_folder/time_vec_{}'.format(i),time_vec)
    dreg_vec.append(A)

    #Sinkhorn 0.009
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.009,1,'Sinkhorn')
    np.save('{}_folder_{}/wass_trial_{}'.format('Sinkhorn',0.009,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Sinkhorn',0.009,i),time_vec)
    loweps_sinkhorn_vec.append(A)

    #Entropy 0.009
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.009,1,'Entropy')
    np.save('{}_folder_{}/wass_trial_{}'.format('Entropy',0.009,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Entropy',0.009,i),time_vec)
    loweps_entropy_vec.append(A)
    #Sinkhorn 0.089
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.089,1,'Sinkhorn')
    np.save('{}_folder_{}/wass_trial_{}'.format('Sinkhorn',0.089,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Sinkhorn',0.089,i),time_vec)
    mideps_sinkhorn_vec.append(A)

    #Entropy 0.089
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.089,1,'Entropy')
    np.save('{}_folder_{}/wass_trial_{}'.format('Entropy',0.089,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Entropy',0.089,i),time_vec)
    mideps_entropy_vec.append(A)

    #Sinkhorn 0.75
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.75,1,'Sinkhorn')
    np.save('{}_folder_{}/wass_trial_{}'.format('Sinkhorn',0.75,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Sinkhorn',0.75,i),time_vec)
    higheps_sinkhorn_vec.append(A)

    #Entropy 0.75
    A,B,time_vec,train_points,test_points,train_labels,test_labels=benchmark_trial(train_split,test_split,perm,corrupted_batchlist,3,0.75,1,'Entropy')
    np.save('{}_folder_{}/wass_trial_{}'.format('Entropy',0.75,i),A)
    np.save('{}_folder_{}/time_vec_{}'.format('Entropy',0.75,i),time_vec)
    higheps_entropy_vec.append(A)




#Constructs mean and confidence intervals given an error tolerance
def means_and_confidence_intervals(vector_array,confidence_level):
    inner_means=[]
    for i in vector_array:
        inner_means.append(np.mean(i,axis=0))
    inner_means=np.array(inner_means)
    errors=[]
    outer_means=[]
    all_data=[inner_means[:,0],inner_means[:,1],inner_means[:,2],inner_means[:,3],inner_means[:,4],inner_means[:,5]]
    for i in all_data:
        t_statistic, p_value = stats.ttest_1samp(i, np.mean(i))
        sample_mean = np.mean(i)
        outer_means.append(sample_mean)
        sample_sem = stats.sem(i)
        degrees_of_freedom = len(i) - 1
        confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom, sample_mean, sample_sem)
        errors.append((confidence_interval[1]-confidence_interval[0])/2)
    return outer_means,errors

alpha=0.95

loweps_sinkhorn_means,loweps_sinkhorn_errors=means_and_confidence_intervals(loweps_sinkhorn_vec,alpha)
loweps_entropy_means,loweps_entropy_errors=means_and_confidence_intervals(loweps_entropy_vec,alpha)
mideps_sinkhorn_means,mideps_sinkhorn_errors=means_and_confidence_intervals(mideps_sinkhorn_vec,alpha)
mideps_entropy_means,mideps_entropy_errors=means_and_confidence_intervals(mideps_entropy_vec,alpha)
higheps_sinkhorn_means,higheps_sinkhorn_errors=means_and_confidence_intervals(higheps_sinkhorn_vec,alpha)
higheps_entropy_means,higheps_entropy_errors=means_and_confidence_intervals(higheps_entropy_vec,alpha)
unreg_means,unreg_errors=means_and_confidence_intervals(unreg_vec,alpha)
dreg_means,dreg_errors=means_and_confidence_intervals(dreg_vec,alpha)
nn_means,nn_errors=means_and_confidence_intervals(nn_vec,alpha)


