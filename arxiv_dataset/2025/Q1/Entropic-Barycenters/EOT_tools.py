import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import h5py
import pandas as pd
from scipy import stats,linalg
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import ot
import random
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
from scipy.optimize import minimize
import cvxpy as cp 
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
from os.path  import join
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import plotly.express as px
import h5py
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import torch.nn as nn
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch.nn.init as init  # Import the init module

#Class for labeled pointcloud data. ''data'' is array of 3D arrays, ''labels'' is an integer.
class labeled_data():
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels

#Class for probability measure. ''Points'' is array of d-dimensional arrays, 'masses'' is array of nonnegative scalars of the same length, adding up to one. 
class measure:
    def __init__(self,points,masses):
        self.points=points
        self.masses=masses

#Produces an empirical measure by sampling from the input_measure
def empirical_measure(input_measure,samples):
    points=input_measure.points
    masses=input_measure.masses
    uniform_mass=np.dot(np.ones(samples),1/samples)
    empirical_support=[]
    labels = np.random.choice(len(masses), size=samples, p=masses)
    for i in labels:
        empirical_support.append(points[i])
    output=measure(empirical_support,uniform_mass)
    return output

def sample_with_replacement(lst, sample_size):
    sample = [random.choice(lst) for _ in range(sample_size)]
    return sample

def sample_array(array,samples):
    index_list=np.arange(len(array))
    sampled_idx=np.random.choice(index_list,size=samples)
    sample_list=[]
    for i in sampled_idx:
        sample_list.append(array[i])
    return sample_list

#Computes the extended entropic map (see Theorem 2.2 in the main paper) between one-dimensional probability measures.
def onedim_extended_map(potential,source,target,epsilon):
    source_points=source.points
    target_points=target.points
    log_potential=np.transpose(epsilon*np.log(potential))
    cost_matrix=1/2*np.power(euclidean_distances(source_points,target_points),2)
    log_potential_matrix=np.tile(log_potential,(len(source_points),1))
    log_coupling=1/epsilon*(log_potential_matrix-cost_matrix)
    gamma=np.exp(log_coupling)
    unnormalized_map=np.dot(gamma,target_points)
    ones=np.ones((len(target_points),1))
    normalization=gamma@ones
    J=np.divide(unnormalized_map,normalization)
    return J

#Randomly samples a rotation matrix 
def random_rotation_matrix(n):
    random_matrix = np.random.rand(n, n)
    q, r = np.linalg.qr(random_matrix)
    d = np.diagonal(r)
    ph = np.sign(d)
    q *= ph
    return q

#Random samples a PSD matrix with entries between [low,high]. Increasing ''alpha'' improves the condition number.
def randcov(low,high,dim,alpha):
    A = np.random.uniform(low,high,(dim, dim))
    B = A @ A.T + alpha*np.eye(dim)
    rotation_matrix, _ = np.linalg.qr(np.random.randn(dim, dim))
    composed_matrix = rotation_matrix @ B @ rotation_matrix.T
    return composed_matrix

#Randomly samples a probability vector of length ''size''.
def random_weights(size):
    vec=np.random.rand(size)
    total=sum(vec)
    normalized=np.dot(vec,1/total)
    return normalized

#Randomly produces a vector of length ''size'' with entries in [low,high].
def random_vec(size,low,high):
    vec=np.random.uniform(low,high,size)
    mat=random_rotation_matrix(size)
    rot_vec=mat@vec
    return rot_vec

#Solvers for the standard deviation of entropy-regularized barycenters of one-dimensional Gaussians with given standard deviations, weights and regularization (see Theorem 2 in https://arxiv.org/pdf/2006.02575 to generalize to arbitrary # of references).

def oneD_barystdv_solver(weights, stdvs, regularization):
    eps_prime = max((regularization / 2) ** 0.5, 0)
    
    def func_to_solve(x):
        summands = [w * (eps_prime**4 + 4 * s**2 * x**2) ** 0.5 for w, s in zip(weights, stdvs)]
        return sum(summands) - (eps_prime**2 + 2 * x**2)
    
    roots = least_squares(func_to_solve, x0=2)
    return roots.x

#Computes the entropic potential between two ''measure'' objects, with a given regularization epsilon.
def get_potential(source,target,epsilon):
    source_points=source.points
    target_points=target.points
    source_masses=source.masses
    target_masses=target.masses
    cost_matrix=1/2*np.power(euclidean_distances(source_points,target_points),2)
    J=ot.sinkhorn(source_masses,target_masses,cost_matrix,epsilon,log='True',stopThr=1e-8,numItermax=4000)
    u=J[1].get('u')
    v=J[1].get('v')
    return u,v

#Computes the (extended) entropic map (see Theorem 2.2 in the main paper) between two ''measure'' objects, given a fixed entropic potential (computed by ''get_potential'').
#If get_potential is used with same ''source'' and ''target'' measure, then we obtain the (non-extended) entropic map between source and target.
def highdim_extended_map(potential,source,target,epsilon,dim):
    source_points=source.points
    target_points=target.points
    log_potential=np.transpose(epsilon*np.log(potential))
    cost_matrix=1/2*np.power(euclidean_distances(source_points,target_points),2)
    log_potential_matrix=np.tile(log_potential,(len(source_points),1))
    log_coupling=1/epsilon*(log_potential_matrix-cost_matrix)
    gamma=np.exp(log_coupling)
    unnormalized_map=np.dot(gamma,target_points)
    ones=np.ones((len(target_points),dim))
    normalization=gamma@ones
    J=np.divide(unnormalized_map,normalization)
    return J

#Computes the entropy-regularized barycenter functional for the uniform measures on a set of "source_points", where the reference measures are uniform measures on the "reference_points".
def barycenter_functional(source_points,reference_points,weights,epsilon):
    cost_vec=[]
    masses=np.dot(1/len(source_points),np.ones(len(source_points)))
    for i in reference_points:
        M=np.power(euclidean_distances(source_points,i),2)
        cost=ot.sinkhorn2(masses,masses,M,epsilon)
        cost_vec.append(cost)
    score=np.dot(np.array(weights),np.array(cost_vec))
    return score

#Solves the analysis problem for the entropy-regularized barycenter functional for uniform measures on the supports of reference measures (''reference_measure_points'') and on the support of the base measure (''base_measure_points'').
#Can be adapted to non-uniform measures by changing ``uniform_masses'' to a different set of masses.
#See Algorithm 1 in main paper for pseudocode. \mu is supported on base_measure_points, \mathcal{V} are supported on the reference_measure_points.

def ent_analysis(reference_measure_points,base_measure_points,regularization):
    uniform_masses=np.dot(np.ones(len(base_measure_points)),1/len(base_measure_points))
    base_measure=measure(base_measure_points,uniform_masses)
    reference_measures=[]
    for i in reference_measure_points:
        reference_mass=np.dot(np.ones(len(i)),1/len(i))
        reference_meas=measure(i,reference_mass)
        reference_measures.append(reference_meas)
    entmap_list=[]
    #compute entropic maps from base measure to reference measures
    def process_reference_map(reference_measure):
        g = get_potential(base_measure, reference_measure, regularization)
        extended = highdim_extended_map(g[1], base_measure, reference_measure, regularization, 3)
        return extended
    entmap_list = Parallel(n_jobs=-1)(delayed(process_reference_map)(i) for i in reference_measures)
    l2norms=[]
    #Construct matrix A^\eps_\mu
    for p in np.arange(len(entmap_list)):
        for q in np.arange(len(entmap_list)):
            T_p=entmap_list[p]-base_measure_points
            T_q=entmap_list[q]-base_measure_points
            dotvec=[]
            for k in np.arange(len(T_p)):
                innerprod=np.dot(T_p[k],T_q[k])
                dotvec.append(innerprod)
            l2diff=np.dot(dotvec,uniform_masses)
            l2norms.append(l2diff)
    l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
    #Solve quadratic program
    x=cp.Variable(len(entmap_list))
    objective=cp.Minimize(cp.quad_form(x,l2matrix))
    constraints=[x>=0,cp.sum(x)==1]
    problem=cp.Problem(objective,constraints)
    problem.solve()
    optimal_x=x.value
    return optimal_x


#Solves the analysis problem for the Sinkhorn barycenter functional for uniform measures on the supports of reference measures (''reference_measure_points'') and on the support of the base measure (''base_measure_points'').
#Can be adapted to non-uniform measures by changing ``uniform_masses'' to a different set of masses.
#See Algorithm 1 in main paper for pseudocode. \mu is supported on base_measure_points, \mathcal{V} are supported on the reference_measure_points.

def sink_analysis(reference_measure_points,base_measure_points,regularization):
    uniform_masses=np.dot(np.ones(len(base_measure_points)),1/len(base_measure_points))
    base_measure=measure(base_measure_points,uniform_masses)
    reference_measures=[]
    for i in reference_measure_points:
        reference_mass=np.dot(np.ones(len(i)),1/len(i))
        reference_meas=measure(i,reference_mass)
        reference_measures.append(reference_meas)
    entmap_list=[]
    #compute entropic maps from base measure to reference measures
    def process_reference_map(reference_measure):
        g = get_potential(base_measure, reference_measure, regularization)
        extended = highdim_extended_map(g[1], base_measure, reference_measure, regularization, 3)
        return extended
    entmap_list = Parallel(n_jobs=-1)(delayed(process_reference_map)(i) for i in reference_measures)
    #compute self entropic map
    g_self=get_potential(base_measure,base_measure,regularization)
    self_map=highdim_extended_map(g_self[1],base_measure,base_measure,regularization,3)
    l2norms=[]
    #Construct matrix S^\eps_\mu
    for p in np.arange(len(entmap_list)):
        for q in np.arange(len(entmap_list)):
            T_p=entmap_list[p]-self_map
            T_q=entmap_list[q]-self_map
            dotvec=[]
            for k in np.arange(len(T_p)):
                innerprod=np.dot(T_p[k],T_q[k])
                dotvec.append(innerprod)
            l2diff=np.dot(dotvec,uniform_masses)
            l2norms.append(l2diff)
    l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
    #Solve quadratic program
    x=cp.Variable(len(entmap_list))
    objective=cp.Minimize(cp.quad_form(x,l2matrix))
    constraints=[x>=0,cp.sum(x)==1]
    problem=cp.Problem(objective,constraints)
    problem.solve()
    optimal_x=x.value
    return optimal_x

#Solves the synthesis problem (via Algorithm 2 in the main paper) for the Entropic barycenter functional for given ''reference_measures'', ''weight_vec''. 
#Iterations are initialized at ''base_measure.''
#Stepsize should be in (0,1), regularization > 0


def entropic_synthesis(regularization,reference_measures,base_measure,weight_vec,stepsize,iterations):
    dim=len(base_measure.points[0])
    initial_masses=base_measure.masses
    source_measure=base_measure
    for i in np.arange(iterations):
        if i%10==0:
            print(i)
        combined_entmap=np.zeros((len(source_measure.points),dim))
        for j in np.arange(len(reference_measures)):
            g = get_potential(source_measure, reference_measures[j], regularization)
            entmap = highdim_extended_map(g[1], source_measure, reference_measures[j], regularization, 3)
            scaled_entmap=np.dot(weight_vec[j],entmap)
            combined_entmap=combined_entmap+scaled_entmap
        update_source_points=np.dot((1-stepsize),source_measure.points)+np.dot(stepsize,combined_entmap)
        source_measure=measure(update_source_points,initial_masses)
    return source_measure

#Solves the synthesis problem (via Algorithm 2 in the main paper) for the Sinkhorn barycenter functional for given ''reference_measures'', ''weight_vec''. 
#Iterations are initialized at ''base_measure.''
#Stepsize should be in (0,1), regularization > 0

def sinkhorn_synthesis(regularization,reference_measures,base_measure,weight_vec,stepsize,iterations):
    dim=len(base_measure.points[0])
    initial_masses=base_measure.masses
    source_measure=base_measure
    for i in np.arange(iterations):
        if i%10==0:
            print(i)
        combined_entmap=np.zeros((len(source_measure.points),dim))
        for j in np.arange(len(reference_measures)):
            g = get_potential(source_measure, reference_measures[j], regularization)
            entmap = highdim_extended_map(g[1], source_measure, reference_measures[j], regularization, dim)
            scaled_entmap=np.dot(weight_vec[j],entmap)
            combined_entmap=combined_entmap+scaled_entmap
        self_potential = get_potential(source_measure, source_measure, regularization)
        self_map = highdim_extended_map(self_potential[1], source_measure, source_measure, regularization, dim)
        update_source_points=source_measure.points-np.dot(stepsize,self_map)+np.dot(stepsize,combined_entmap)
        source_measure=measure(update_source_points,initial_masses)
    return source_measure


####Classification tools


def find_indices(numbers, target):
    return [index for index, value in enumerate(numbers) if value == target]

def split_indices(indices,labels,cutoff):
    #e.g. cutoff=10
    label_list=find_indices(indices,labels)
    return label_list[0:cutoff],label_list[cutoff:]

#atom data is a pointcloud (3D array of points)
#atom index is the index of the pointcloud in the original list
#atom label corresponds to the type of pointcloud (see ''label_obj_dictionary'' in 'classification_example.py')
#atoms are obtained from reference measures
class atom:
    def __init__(self,index,label,data):
        self.index=index
        self.label=label
        self.data=data

#entry data is a pointcloud (3D array of points)
#entry label corresponds to the type of pointcloud (see ''label_obj_dictionary'' in 'classification_example.py')  
#entry coefficients is a probability vector (of length # reference measures), which will be used for classification
class entry:
    def __init__(self,labels,data,coefficients):
        self.labels=labels
        self.data=data
        self.coefficients=coefficients

#''data'' is a list of point clouds (3D arrays) corresponding to a fixed label (e.g. the label associated to planes, see see ''label_obj_dictionary'' in 'classification_example.py')
#Converts pointclouds in ''data'' indexed by ''list_of_indices'' to atom objects.
def generate_atoms(list_of_indices,label,data,m):
    list_of_atoms=[]
    sampled_indices=list_of_indices[0:m]
    for sampled_idx in sampled_indices:
        atm=atom(sampled_idx,label,data[sampled_idx])
        list_of_atoms.append(atm)
    return list_of_atoms

#Preprocessing step for ''build_coefficients''

#''data'' is a list of point clouds (3D arrays) corresponding to a fixed label (e.g. the label associated to planes, see see ''label_obj_dictionary'' in 'classification_example.py')
#Converts pointclouds in ''data'' indexed by ''list_of_indices'' to entry objects.
#Coefficients are default set to zero
def generate_entries(list_of_indices,label,data,max_entries):
    list_of_entries=[]
    counter=0
    for index in list_of_indices:
        if counter<max_entries:
            entr=entry(index,label,data[index],0)
            list_of_entries.append(entr)
            counter=counter+1
        else:
            break
    return list_of_entries

# ''dictionary'' is a list of atoms
# ''list_of_entries'' is a list of entries to be classified
# Computes coefficients to each entry in a list of entries using the entropy-regularized barycenter functional (see Section 5 in main paper for details)
def build_coefficients(list_of_entries,dictionary,regularization):
    new_list=[]
    dictionary_points_list=[]
    for atom in dictionary:
        dictionary_points_list.append(atom.data)
    for entr in list_of_entries:
        coeff=ent_analysis(dictionary_points_list,entr.data,regularization)
        new_entry=entry(entr.labels,entr.data,coeff)
        new_list.append(new_entry)
    return new_list

# ''dictionary'' is a list of atoms
# ''list_of_entries'' is a list of entries to be classified
# Computes coefficients to each entry in a list of entries using the Sinkhorn barycenter functional (see Section 5 in main paper for details)
def sink_build_coefficients(list_of_entries,dictionary,regularization):
    new_list=[]
    dictionary_points_list=[]
    for atom in dictionary:
        dictionary_points_list.append(atom.data)
    for entr in list_of_entries:
        coeff=sink_analysis(dictionary_points_list,entr.data,regularization)
        new_entry=entry(entr.labels,entr.data,coeff)
        new_list.append(new_entry)
    return new_list

#If # of atoms is k x m, where k is the number of unique atom labels, combine_coefficients converts entry coefficients to probability vector of size k
def combine_coefficients(dict_entry,m):
    coeffs=dict_entry.coefficients
    bins=int(len(coeffs)/m)
    new_coeffs=[]
    for i in range(bins):
        start_index=i*m
        end_index=start_index+m
        total=sum(coeffs[start_index:end_index])
        new_coeffs.append(total)
    return new_coeffs

def combine_coefficients_list(list_of_dict_entr,m):
    processed_entries=[]
    for entr in list_of_dict_entr:
        comb_coeff=combine_coefficients(entr,m)
        new_entry=entry(entr.labels,entr.data,comb_coeff)
        processed_entries.append(new_entry)
    return processed_entries

#Performs classification using entropy-regularized barycenter fucntional
# Inputs:
# ''list_of_entries'': list of pointclouds stored as ''entry'' objects
# ''dictionary'': list of pointclouds stored as ''atom'' objects

def dictionary_learn(list_of_entries,dictionary,regularization,m):
    coefficients=build_coefficients(list_of_entries,dictionary,regularization)
    processed_entries=combine_coefficients_list(coefficients,m)
    return processed_entries

#Performs classification using Sinkhorn barycenter fucntional
# Inputs:
# ''list_of_entries'': list of pointclouds stored as ''entry'' objects
# ''dictionary'': list of pointclouds stored as ''atom'' objects

def sink_dictionary_learn(list_of_entries,dictionary,regularization,m):
    coefficients=sink_build_coefficients(list_of_entries,dictionary,regularization)
    processed_entries=combine_coefficients_list(coefficients,m)
    return processed_entries

#computes accuracy for outputs
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
    dictionary_labels=list(set(dictionary_labels))
    dictionary_labels.sort()
    for coeff in coeff_list:
        maximum=np.argmax(coeff)
        maxes.append(dictionary_labels[maximum])
    maxes=np.array(maxes)
    entry_label_list=np.array(entry_label_list)
    score_vec=(maxes != entry_label_list).astype(int)
    return score_vec

    
def plot_results(list_of_entries,label_obj_dict,regularization,sink,atom_attributes,data_attributes,score,m):
    coeff_list=[]
    label_list=[]
    for entr in list_of_entries:
        coeff_list.append(entr.coefficients)
        label_list.append(entr.labels)
    tsne=TSNE(n_components=2,random_state=5)
    X_tsne=tsne.fit_transform(np.array(coeff_list))
    x_values = [point[0] for point in X_tsne]
    y_values = [point[1] for point in X_tsne]
    traces = []
    for label in set(label_list):
        indices = [i for i, lbl in enumerate(label_list) if lbl == label]
        x_vals = [x_values[i] for i in indices]
        y_vals = [y_values[i] for i in indices]
        trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=label_obj_dict[label]  
        )
        traces.append(trace)
    fig = go.Figure(data=traces)
    if sink==True:
        fig.update_layout(title='Dictionary Learning with Sinkhorn Barycenter Functional (reg={}, Batch Size={})<br><sup>Atoms:{}, Data:{}</sup><br><sup>Accuracy={}</sup>'.format(regularization,m,atom_attributes,data_attributes,1-score))
    else:
        fig.update_layout(title='Dictionary Learning with Entropic Barycenter Functional (reg={}, Batch size={})<br><sup>Atoms:{}, Data:{}</sup></sup><br><sup>Accuracy={}</sup>'.format(regularization,m,atom_attributes,data_attributes,1-score))
    fig.show()

#### Unregularized functional/tangential Wasserstein projection

#Code for comparing our classification to unregularized barycentric coding/tangential Wasserstein projection approach in https://arxiv.org/abs/2207.14727. See main paper Section 5 for discussion.
#Implementation is adapted from https://github.com/menghsuanhsieh/tangential-wasserstein-projection/tree/main, the supplement to '`Tangential Wasserstein Projections'' by Gunsilius, Hsieh & Lee (2022)

def baryc_proj(source, target, method):
    n1 = source.shape[0]
    n2 = target.shape[0]   
    p = source.shape[1]
    a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    M = ot.dist(source, target)
    M = M.astype('float64')
    M /= M.max()
    if method == 'emd':
        OTplan = ot.emd(a_ones, b_ones, M, numItermax = 1e7)
        
    elif method == 'entropic':
        OTplan = ot.bregman.sinkhorn_stabilized(a_ones, b_ones, M, reg = 5*1e-3)
    # initialization
    OTmap = np.empty((0, p))
    for i in range(n1):
        # normalization
        OTplan[i,:] = OTplan[i,:] / sum(OTplan[i,:])
        # conditional expectation
        OTmap = np.vstack([OTmap, (target.T @ OTplan[i,:])])
    OTmap = np.array(OTmap).astype('float32')
    return(OTmap)

def tan_wass_proj(target, controls, method = 'emd'):
    target=np.array(target)
    n = target.shape[0]
    d = target.shape[1]
    J = len(controls)
    S = np.mean(target)*n*d*J # Stabilizer: to ground the optimization objective
    uniform_masses=np.dot(1/len(target),np.ones(len(target)))
    # Barycentric Projection
    G_list = []
    proj_list = []
    for i in range(len(controls)):
        temp = baryc_proj(target, controls[i], method)
        G_list.append(temp)
        proj_list.append(temp - target)
    #return mylambda_opt
    l2norms=[]
    for p in np.arange(len(proj_list)):
        for q in np.arange(len(proj_list)):
            T_p=proj_list[p]
            T_q=proj_list[q]
            dotvec=[]
            for k in np.arange(len(T_p)):
                innerprod=np.dot(T_p[k],T_q[k])
                dotvec.append(innerprod)
            l2diff=np.dot(dotvec,uniform_masses)
            l2norms.append(l2diff)
    l2matrix=np.reshape(l2norms,(len(proj_list),len(proj_list)))
    x=cp.Variable(J)
    objective=cp.Minimize(cp.quad_form(x,l2matrix))
    constraints=[x>=0,cp.sum(x)==1]
    problem=cp.Problem(objective,constraints)
    problem.solve()
    optimal_x=x.value
    return optimal_x

def tangent_build_coefficients(list_of_entries,dictionary):
    new_list=[]
    dictionary_points_list=[]
    for atom in dictionary:
        dictionary_points_list.append(atom.data)
    counter=0
    for entr in list_of_entries:
        counter=counter+1
        coeff=tan_wass_proj(entr.data,dictionary_points_list,'emd')
        new_entry=entry(entr.labels,entr.data,coeff)
        new_list.append(new_entry)
    return new_list

def tangent_dictionary_learn(list_of_entries,dictionary,m):
    coefficients=tangent_build_coefficients(list_of_entries,dictionary)
    processed_entries=combine_coefficients_list(coefficients,m)
    return processed_entries



