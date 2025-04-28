import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats,linalg
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import ot
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
from scipy.optimize import minimize
import cvxpy as cp 
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import random
import sys
from os.path  import join

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EOT_tools import *

regularization=1
#sample_range=[10,20,40,80,160,320,640,1280,2560,5120,10240]
sample_range=[10,20]
initial_samps=10
    
def highdim_analysis_experiment_three_refs(regularization,trials,instances,dim,initial_samps):
    losses=[]
    #synthesize
    os.mkdir('experiment_{}'.format(initial_samps))
    weight_vec=random_weights(3)
    mean_1=random_vec(dim,0.2,0.4)
    mean_2=random_vec(dim,0.4,0.6)
    mean_3=random_vec(dim,0.6,0.8)
    A1=randcov(.2,.5,dim,0.7)
    A2=randcov(.2,.5,dim,0.7)
    A3=randcov(.2,.5,dim,0.7)
    Gauss_1=stats.multivariate_normal(mean_1,A1)
    Gauss_2=stats.multivariate_normal(mean_2,A2)
    Gauss_3=stats.multivariate_normal(mean_3,A3)
    Gauss_1_points=Gauss_1.rvs(initial_samps)
    Gauss_2_points=Gauss_2.rvs(initial_samps)
    Gauss_3_points=Gauss_3.rvs(initial_samps)
    uniform_points = np.random.rand(initial_samps, dim)
    uniform_masses=np.dot(np.ones(initial_samps),1/initial_samps)
    uniform_measure=measure(uniform_points,uniform_masses)
    measure_location_vec=[np.array(Gauss_1_points),np.array(Gauss_2_points),np.array(Gauss_3_points)]
    measure_masses=[np.array(uniform_masses),np.array(uniform_masses),np.array(uniform_masses)]
    output=ot.bregman.free_support_sinkhorn_barycenter(measures_locations=measure_location_vec,measures_weights=measure_masses,X_init=np.array(uniform_measure.points),b=np.array(uniform_measure.masses),weights=weight_vec,numItermax=100,reg=regularization,verbose=False)
    for samps in trials:
        trial_loss=[]
        for j in np.arange(instances):
            Gauss_1_points=Gauss_1.rvs(samps)
            Gauss_2_points=Gauss_2.rvs(samps)
            Gauss_3_points=Gauss_3.rvs(samps)
            instance_masses=np.dot(np.ones(samps),1/samps)
            Gauss_1_measure=measure(Gauss_1_points,instance_masses)
            Gauss_2_measure=measure(Gauss_2_points,instance_masses)
            Gauss_3_measure=measure(Gauss_3_points,instance_masses)
            reference=[Gauss_1_measure,Gauss_2_measure,Gauss_3_measure]
            output_1=sample_array(output,samps)
            output_2=sample_array(output,samps)
            output_measure_1=measure(output_1,instance_masses)
            output_measure_2=measure(output_2,instance_masses)
            potential_vec=[]

            for k in reference:
                g=get_potential(output_measure_1,k,regularization)
                potential_vec.append(g[1])
            entmap_list=[]
            for k in np.arange(len(reference)):
                extended=highdim_extended_map(potential_vec[k],output_measure_2,reference[k],regularization,dim)
                entmap_list.append(extended)
            l2norms=[]
            for p in np.arange(len(entmap_list)):
                for q in np.arange(len(entmap_list)):
                    T_p=entmap_list[p]-output_measure_2.points
                    T_q=entmap_list[q]-output_measure_2.points
                    dotvec=[]
                    for k in np.arange(len(T_p)):
                        innerprod=np.dot(T_p[k],T_q[k])
                        dotvec.append(innerprod)
                    l2diff=np.dot(dotvec,instance_masses)
                    l2norms.append(l2diff)
            l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
            x=cp.Variable(len(entmap_list))
            objective=cp.Minimize(cp.quad_form(x,l2matrix))
            constraints=[x>=0,cp.sum(x)==1]
            problem=cp.Problem(objective,constraints)
            problem.solve()
            optimal_x=x.value
            loss=np.linalg.norm(weight_vec-optimal_x)**2
            trial_loss.append(loss)
            if j%10==0:
                print(samps,j)
            params=[initial_samps,reference,potential_vec,[output_1,output_2],optimal_x]
        losses.append(trial_loss)  
    instance_parameters={'means':[mean_1,mean_2,mean_3],'weights':weight_vec,'covariances':[A1,A2,A3],'output':output}
    return losses

losses=highdim_analysis_experiment_three_refs(regularization,sample_range,10,5,initial_samps)


#Parameters used for experiments in paper:
#sample_range=[10,20,40,80,160,320,640,1280,2560,5120,10240]
#initial_samps=100
#losses=highdim_analysis_experiment_three_refs(regularization,sample_range,100,5)

#script to plot results
logtrials=np.log10(sample_range)
avg_loss=[]
for i in losses:
    avg_loss.append(np.mean(i))
log_loss=np.log10(avg_loss)
coefficients = np.polyfit(logtrials, log_loss, 1)
line_of_fit=[]
for i in logtrials:
    line_of_fit.append(coefficients[0]*i+coefficients[1])
plt.figure(figsize=(7, 5))
plt.scatter(logtrials,log_loss, label='(Log_10) L2 loss')
plt.plot(logtrials,line_of_fit,c='red')
textbox_content = "Slope={}, \n Intercept={}".format(round(coefficients[0],4),round(coefficients[1],3))
plt.text(.52,.8,textbox_content, bbox=dict(facecolor='red', alpha=0.5),transform=plt.gca().transAxes,fontsize=20)
plt.xlabel(r'$\log_{10}$ Sample #',fontsize=18)
plt.ylabel(r'$\log_{10} \|\lambda^*-\hat{\lambda}^n\|^2$',fontsize=18)
fig_size = plt.gcf().get_size_inches()
plt.savefig("hdg_experiment.pdf", format= 'pdf', bbox_inches="tight")
plt.show()


