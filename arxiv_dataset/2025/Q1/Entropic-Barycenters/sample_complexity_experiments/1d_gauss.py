import numpy as np
import os
import sys
import numpy.linalg
import pandas as pd
from scipy import stats,linalg
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import ot
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import struct
from array import array
from itertools import product
from scipy.optimize import minimize
import cvxpy as cp 
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import random
from os.path  import join

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EOT_tools import *



regularization=2
#Parameters used for experiments in paper:
sample_range=[10,20,40,80,160,320,640,1280,2560,5120,10240]
instances=100
#sample_range=[10,20]
#instances=10
def random_number_vec(size,low,high):
    vec=np.random.uniform(low,high,size)
    return vec

def oneD_barystdv_solver(weights, stdvs, regularization):
    eps_prime = max((regularization / 2) ** 0.5, 0)
    
    def func_to_solve(x):
        summands = [w * (eps_prime**4 + 4 * s**2 * x**2) ** 0.5 for w, s in zip(weights, stdvs)]
        return sum(summands) - (eps_prime**2 + 2 * x**2)
    
    roots = least_squares(func_to_solve, x0=2)
    return roots.x

def one_dim_exact_analysis_experiment_tworef(regularization,trials,instances):
    sqr_wass_losses=[]
    sqr_l2_loss_vector=[]
    stdv_dist=[]
    condition_no_vec=[]
    for samps in trials:
        trial_loss=[]
        trial_sqr_l2_loss=[]
        trial_stdv_dist=[]
        trial_condition_no=[]
        trial_parameter_vec=[]
        for j in np.arange(instances):
            random_mean_1=random_number_vec(1,0,2)
            random_mean_2=random_number_vec(1,3,5)
            random_stdvs=random_number_vec(2,2,4)
            gauss_1=stats.norm(random_mean_1,random_stdvs[0])
            gauss_2=stats.norm(random_mean_2,random_stdvs[1])
            gauss_1_samples=gauss_1.rvs(samps).reshape(-1, 1)
            gauss_2_samples=gauss_2.rvs(samps).reshape(-1, 1)
            uniform_masses=np.dot(np.ones(samps),1/samps)
            gauss_1_meas=measure(gauss_1_samples,uniform_masses)
            gauss_2_meas=measure(gauss_2_samples,uniform_masses)

            weight_vec=random_weights(2)
            bary_stdv=oneD_barystdv_solver(weight_vec, random_stdvs,regularization)
            bary_mean=weight_vec[0]*random_mean_1+weight_vec[1]*random_mean_2
            barycenter=stats.norm(bary_mean,bary_stdv)

            barycenter_samples_1=barycenter.rvs(samps).reshape(-1, 1)
            barycenter_samples_2=barycenter.rvs(samps).reshape(-1,1)
            barycenter_meas_1=measure(barycenter_samples_1,uniform_masses)
            barycenter_meas_2=measure(barycenter_samples_2,uniform_masses)
            potential_vec=[]
            reference=[gauss_1_meas,gauss_2_meas]

            #get potentials

            for k in reference:
                g=get_potential(barycenter_meas_1,k,regularization)
                potential_vec.append(g[1])
            #build entmaps
            entmap_list=[]
            for k in np.arange(len(reference)):
                extended=onedim_extended_map(potential_vec[k],barycenter_meas_2,reference[k],regularization)
                entmap_list.append(extended)

            #analysis
            l2norms=[]
            for p in np.arange(len(entmap_list)):
                for q in np.arange(len(entmap_list)):
                    T_p=entmap_list[p]-barycenter_meas_2.points
                    T_q=entmap_list[q]-barycenter_meas_2.points
                    dotvec=[]
                    for k in np.arange(len(T_p)):
                        innerprod=np.dot(T_p[k],T_q[k])
                        dotvec.append(innerprod)
                    l2diff=np.dot(dotvec,uniform_masses)
                    l2norms.append(l2diff)
            l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
            condition=np.linalg.cond(l2matrix)
            trial_condition_no.append(condition)
            x=cp.Variable(len(entmap_list))
            objective=cp.Minimize(cp.quad_form(x,l2matrix))
            constraints=[x>=0,cp.sum(x)==1]
            problem=cp.Problem(objective,constraints)
            problem.solve()
            optimal_x=x.value
            sqrl2_loss=np.linalg.norm(weight_vec-optimal_x)**2
            empbary_stdv=oneD_barystdv_solver(optimal_x, random_stdvs,regularization)
            empbary_mean=optimal_x[0]*random_mean_1+optimal_x[1]*random_mean_2
            empbary=stats.norm(empbary_mean,empbary_stdv)
            
            barycenter_samples_4=barycenter.rvs(10000).reshape(-1, 1)
            emp_barycenter_samples=empbary.rvs(10000).reshape(-1, 1)
            ot_output=(empbary_mean-bary_mean)**2+empbary_stdv**2+bary_stdv**2-2*((empbary_stdv**2)*(bary_stdv**2))**0.5
            loss=ot_output**2
            stdv_differential=np.abs(empbary_stdv-bary_stdv)
            trial_loss.append(loss)
            trial_sqr_l2_loss.append(sqrl2_loss)
            trial_stdv_dist.append(stdv_differential)
            if j%10==0:
                print(samps,j)
            parameters=np.array([random_mean_1,random_mean_2,randcov,optimal_x],dtype='object')
            trial_parameter_vec.append(parameters)

        avg_loss=np.mean(trial_loss)
        avg_condition_no=np.mean(trial_condition_no)
        avg_stdv_dist=np.mean(trial_stdv_dist)
        avg_sqr_l2_loss=np.mean(trial_sqr_l2_loss)

        np.save('1D_gaussians_2ref_random_weight_2_wass_loss {}'.format(samps),trial_loss)
        np.save('1D_gaussians_2ref_random_weight_2_l2sqr_loss {}'.format(samps),trial_sqr_l2_loss)
        #np.save('1D_gaussians_2ref_random_weight_2_condition_no {}'.format(samps),trial_condition_no)
        #np.save('1D_gaussians_2ref_random_weight_2_parameters {}'.format(samps),trial_parameter_vec)
        sqr_wass_losses.append(avg_loss)
        sqr_l2_loss_vector.append(avg_sqr_l2_loss)
        condition_no_vec.append(avg_condition_no)
        stdv_dist.append(avg_stdv_dist)

        plt.hist(emp_barycenter_samples, bins=100, density=True, alpha=0.6,label='empiricalbarycenter')
        plt.hist(barycenter_samples_4, bins=100, density=True, alpha=0.6,label='barycenter')
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5, 0, 0))
        plt.subplots_adjust(bottom=0.1) 
    return sqr_wass_losses,condition_no_vec,sqr_l2_loss_vector,stdv_dist


sqr_wass_losses,condition_no,sqr_l2_loss,stdv_dist=one_dim_exact_analysis_experiment_tworef(regularization,sample_range,instances)

#script to plot results
plt.ion()

avg_loss=[]
for i in sqr_l2_loss:
    avg_loss.append(np.mean(i))
log_loss=np.log10(avg_loss)
logtrials=np.log10(sample_range)
coefficients = np.polyfit(logtrials, log_loss, 1)
line_of_fit=[]
for i in logtrials:
    line_of_fit.append(coefficients[0]*i+coefficients[1])
plt.figure(figsize=(7, 5))
plt.scatter(logtrials,log_loss, label='(Log_10) L2 loss')
plt.plot(logtrials,line_of_fit,c='red')
textbox_content = "Slope={}, \n Intercept={}".format(round(coefficients[0],4),round(coefficients[1],3))
plt.text(.52,.8,textbox_content, bbox=dict(facecolor='red', alpha=0.5),transform=plt.gca().transAxes,fontsize=20)
plt.xlabel(r'$\log_{10}$ Sample #',fontsize=20)
plt.ylabel(r'$\log_{10} \|\lambda^*-\hat{\lambda}^n\|^2$',fontsize=20)
#fig_size = plt.gcf().get_size_inches()
plt.savefig("1D_gauss_2ref_experiment.pdf", format= 'pdf', bbox_inches="tight")
plt.show()
plt.ioff()

def one_dim_exact_analysis_experiment_threeref(regularization,trials,instances):
    sqr_wass_losses=[]
    sqr_l2_loss_vector=[]
    stdv_dist=[]
    condition_no_vec=[]
    for samps in trials:
        trial_loss=[]
        trial_sqr_l2_loss=[]
        trial_stdv_dist=[]
        trial_condition_no=[]
        trial_parameter_vec=[]
        for j in np.arange(instances):
            random_mean_1=random_number_vec(1,0,2)
            random_mean_2=random_number_vec(1,3,5)
            random_mean_3=random_number_vec(1,-2,0)
            random_stdvs=random_number_vec(3,2,4)
            gauss_1=stats.norm(random_mean_1,random_stdvs[0])
            gauss_2=stats.norm(random_mean_2,random_stdvs[1])
            gauss_3=stats.norm(random_mean_3,random_stdvs[2])
            gauss_1_samples=gauss_1.rvs(samps).reshape(-1, 1)
            gauss_2_samples=gauss_2.rvs(samps).reshape(-1, 1)
            gauss_3_samples=gauss_3.rvs(samps).reshape(-1,1)
            uniform_masses=np.dot(np.ones(samps),1/samps)
            gauss_1_meas=measure(gauss_1_samples,uniform_masses)
            gauss_2_meas=measure(gauss_2_samples,uniform_masses)
            gauss_3_meas=measure(gauss_3_samples,uniform_masses)

            weight_vec=random_weights(3)
            bary_stdv=oneD_barystdv_solver(weight_vec, random_stdvs,regularization)
            bary_mean=weight_vec[0]*random_mean_1+weight_vec[1]*random_mean_2+weight_vec[2]*random_mean_3
            barycenter=stats.norm(bary_mean,bary_stdv)
            barycenter_samples_1=barycenter.rvs(samps).reshape(-1, 1)
            barycenter_samples_2=barycenter.rvs(samps).reshape(-1,1)
            barycenter_meas_1=measure(barycenter_samples_1,uniform_masses)
            barycenter_meas_2=measure(barycenter_samples_2,uniform_masses)
            potential_vec=[]
            reference=[gauss_1_meas,gauss_2_meas,gauss_3_meas]
            #get potentials

            for k in reference:
                g=get_potential(barycenter_meas_1,k,regularization)
                potential_vec.append(g[1])
            #build entmaps
            entmap_list=[]
            for k in np.arange(len(reference)):
                extended=onedim_extended_map(potential_vec[k],barycenter_meas_2,reference[k],regularization)
                entmap_list.append(extended)

            #analysis
            l2norms=[]
            for p in np.arange(len(entmap_list)):
                for q in np.arange(len(entmap_list)):
                    T_p=entmap_list[p]-barycenter_meas_2.points
                    T_q=entmap_list[q]-barycenter_meas_2.points
                    dotvec=[]
                    for k in np.arange(len(T_p)):
                        innerprod=np.dot(T_p[k],T_q[k])
                        dotvec.append(innerprod)
                    l2diff=np.dot(dotvec,uniform_masses)
                    l2norms.append(l2diff)
            l2matrix=np.reshape(l2norms,(len(entmap_list),len(entmap_list)))
            condition=np.linalg.cond(l2matrix)
            trial_condition_no.append(condition)
            x=cp.Variable(len(entmap_list))
            objective=cp.Minimize(cp.quad_form(x,l2matrix))
            constraints=[x>=0,cp.sum(x)==1]
            problem=cp.Problem(objective,constraints)
            problem.solve()
            optimal_x=x.value
            sqrl2_loss=np.linalg.norm(weight_vec-optimal_x)**2
            empbary_stdv=oneD_barystdv_solver(optimal_x, random_stdvs,regularization)
            empbary_mean=optimal_x[0]*random_mean_1+optimal_x[1]*random_mean_2+optimal_x[2]*random_mean_3
            empbary=stats.norm(empbary_mean,empbary_stdv)
            
            barycenter_samples_4=barycenter.rvs(10000).reshape(-1, 1)
            emp_barycenter_samples=empbary.rvs(10000).reshape(-1, 1)
            ot_output=(empbary_mean-bary_mean)**2+empbary_stdv**2+bary_stdv**2-2*((empbary_stdv**2)*(bary_stdv**2))**0.5
            loss=ot_output**2
            stdv_differential=np.abs(empbary_stdv-bary_stdv)
            trial_loss.append(loss)
            trial_sqr_l2_loss.append(sqrl2_loss)
            trial_stdv_dist.append(stdv_differential)
            if j%10==0:
                print(samps,j)
            parameters=np.array([random_mean_1,random_mean_2,randcov,optimal_x],dtype='object')
            trial_parameter_vec.append(parameters)

        avg_loss=np.mean(trial_loss)
        avg_condition_no=np.mean(trial_condition_no)
        avg_stdv_dist=np.mean(trial_stdv_dist)
        avg_sqr_l2_loss=np.mean(trial_sqr_l2_loss)

        np.save('1D_gaussians_3ref_random_weight_2_wass_loss {}'.format(samps),trial_loss)
        np.save('1D_gaussians_3ref_random_weight_2_l2sqr_loss {}'.format(samps),trial_sqr_l2_loss)
        #np.save('1D_gaussians_3ref_random_weight_2_condition_no {}'.format(samps),trial_condition_no)
        #np.save('1D_gaussians_3ref_random_weight_2_parameters {}'.format(samps),trial_parameter_vec)
        sqr_wass_losses.append(avg_loss)
        sqr_l2_loss_vector.append(avg_sqr_l2_loss)
        condition_no_vec.append(avg_condition_no)
        stdv_dist.append(avg_stdv_dist)

        plt.hist(emp_barycenter_samples, bins=100, density=True, alpha=0.6,label='empiricalbarycenter')
        plt.hist(barycenter_samples_4, bins=100, density=True, alpha=0.6,label='barycenter')
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5, 0, 0))
        plt.subplots_adjust(bottom=0.1) 
    return sqr_wass_losses,condition_no_vec,sqr_l2_loss_vector,stdv_dist


sqr_wass_losses,condition_no,sqr_l2_loss,stdv_dist=one_dim_exact_analysis_experiment_threeref(regularization,sample_range,instances)
plt.ion()
avg_loss=[]
for i in sqr_l2_loss:
    avg_loss.append(np.mean(i))
log_loss=np.log10(avg_loss)
logtrials=np.log10(sample_range)
coefficients = np.polyfit(logtrials, log_loss, 1)
line_of_fit=[]
for i in logtrials:
    line_of_fit.append(coefficients[0]*i+coefficients[1])
plt.figure(figsize=(7, 5))
plt.scatter(logtrials,log_loss, label='(Log_10) L2 loss')
plt.plot(logtrials,line_of_fit,c='red')
textbox_content = "Slope={}, \n Intercept={}".format(round(coefficients[0],4),round(coefficients[1],3))
plt.text(.52,.8,textbox_content, bbox=dict(facecolor='red', alpha=0.5),transform=plt.gca().transAxes,fontsize=20)
plt.xlabel(r'$\log_{10}$ Sample #',fontsize=20)
plt.ylabel(r'$\log_{10} \|\lambda^*-\hat{\lambda}^n\|^2$',fontsize=20)
#fig_size = plt.gcf().get_size_inches()
plt.savefig("1D_gauss_3ref_experiment.pdf", format= 'pdf', bbox_inches="tight")
plt.show()
plt.ioff()