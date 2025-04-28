import sys, importlib, time
sys.path.append('../Modules/')
from scipy import integrate
from scipy import sparse
from scipy import interpolate

import numpy as np

import os
import scipy.io as sio
import scipy.optimize
import itertools
import time

import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from src.Modules.Models.BuildBINNs import BINN
from src.Modules.Models.BuildGGBINNs import GG_BINN
from src.Modules.Utils.ModelWrapper import ModelWrapper

import Utils.PDESolver as PDESolver
from Loaders.DataFormatter import load_cell_migration_data
from Utils.PDESolver import PDE_sim

import torch

def to_torch(x):
    return torch.from_numpy(x).float()
def to_numpy(x):
    return x.detach().cpu().numpy()


def load_model(binn_name, save_name, model_type = "BINN"):

    weight = '_best_val'
    
    if model_type == "BINN":
        binn = BINN(binn_name, coords="spherical")
    elif model_type == "GGBINN":
        binn = GG_BINN(binn_name)

    # wrap model and load weights
    parameters = binn.parameters()
    model = ModelWrapper(
        model=binn,
        optimizer=None,
        loss=None,
        save_name=save_name)
    
    model.save_name += weight
    model.load(model.save_name + '_model', device = 'cuda:0')

    return model, binn

def recover_binn_params(binn):
    # learned diffusion term
    def D(u):
        D = binn.diffusion(to_torch(u)[:, None])
        return to_numpy(D).reshape(-1)

    # learned growth term
    def G(u):
        r = binn.growth(to_torch(u)[:, None])
        return to_numpy(r).reshape(-1)  

    # learned growth term
    def A(x):
        A = binn.advection(to_torch(x)[:, None])
        return to_numpy(A).reshape(-1)

    return D, G, A

def recover_GGbinn_params(binn):
    
    # learned diffusion term
    def PD(u):
        D = binn.Pdiffusion(to_torch(u)[:, None])
        return to_numpy(D).reshape(-1)

    # learned diffusion term
    def MD(u):
        D = binn.Mdiffusion(to_torch(u)[:, None])
        return to_numpy(D).reshape(-1)

    # learned growth term
    def PG(u):
        r = binn.Pgrowth(to_torch(u)[:, None])
        return to_numpy(r).reshape(-1)

    # learned growth term
    def MG(u):
        r = binn.Mgrowth(to_torch(u)[:, None])
        return to_numpy(r).reshape(-1)

    # learned switching term
    def SpTm(u):
        s = binn.switch_pTm(to_torch(u)[:, None])
        return to_numpy(s).reshape(-1)

    # learned switching term
    def SmTp(u):
        s = binn.switch_mTp(to_torch(u)[:, None])
        return to_numpy(s).reshape(-1)
    
    return PD, MD, PG, MG, SpTm, SmTp

def selectBINNModelGG(data_name, binn_model, cell_line, replicate):
    
    #load in data
    inputs, outputs, shape  = load_cell_migration_data(f"{data_name}.npy",cell_line=cell_line,replicate=replicate,plot=False)
    
    #find final time
    X = inputs[:,0]
    T = inputs[:,1]
    x = np.unique(X)
    t = np.unique(T)
    tMin = np.min(t)
    tMax = np.max(t)
    
    # grab initial condition
    u0 = outputs[T==tMin,0].copy()
    
    binn_name = f'spheroid_{binn_model}_{cell_line}_{replicate}'
    save_folder = '../Weights/'
    
    surface_weights = [.1,1.0,10.0]
    pde_weights = [.1,1.0,10]
    
    S,P = np.meshgrid(surface_weights,pde_weights)
    S = S.reshape(-1)
    P = P.reshape(-1)
    
    OOS_MSE_test_min = np.inf
    for surface_weight,pde_weight in zip(S,P):

        try:
            ### Load params
            save_name =  binn_name + "_sf_weight_"+str(surface_weight)+"_pde_weight_"+str(pde_weight)
            model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name,model_type = "GGBINN")

            ### Load in learned PDE terms
            PD, MD, PG, MG, SpTm, SmTp = recover_GGbinn_params(binn)
            
            ### Initial condition from surface fitter
            
            #make tensor of x and t=t_0, then run it through surface fitter
            inputs_0 = to_torch(np.hstack([x[:,None],t[0]*np.ones(x.shape)[:,None]]))
            outputs_0 = to_numpy(binn.surface_fitter(inputs_0))
            
            #Initialize P, M subpopulations
            P0 = outputs_0[:,0][:,None]
            M0 = outputs_0[:,1][:,None]
            U0 = np.vstack([P0,M0])
            
            ### Simulate PDE
            RHS = PDESolver.PDE_RHS_GG
            u_sim = PDESolver.PDE_sim_compartmental(RHS, U0, x, t, PD, MD, PG, MG, SpTm, SmTp)

            ### Extract P and M from u_sim
            xn = len(x)
            P_sim = u_sim[:xn,:]
            M_sim = u_sim[xn:,:]
            #Total population (T = P + M), vectorize
            T_sim = (P_sim + M_sim).reshape(-1)

            ### Compare model output (T_sim_80) to data output (U_80)
            
            T_sim_80 = T_sim[T>.8*tMax].reshape(505,-1)
            U_80     = outputs[T>.8*tMax].reshape(505,-1)
            OOS_MSE = np.linalg.norm(T_sim_80-U_80)

            if OOS_MSE < OOS_MSE_test_min:

                    OOS_MSE_test_min = OOS_MSE

                    sf_hparam = surface_weight
                    pde_hparam = pde_weight

                    P_min_OOS = P_sim
                    M_min_OOS = M_sim                    
                    T_sim_min_OOS = T_sim
                    
                    binn_test_OOS_min = binn

                    data = {}
                    data['surface_weight'] = sf_hparam
                    data['pde_weight']  = pde_hparam
                    data['name'] = binn_name
                    data['simulation_P'] = P_min_OOS
                    data['simulation_M'] = M_min_OOS
                    data['simulation'] = P_min_OOS + M_min_OOS
                    data['OOS_MSE'] = OOS_MSE_test_min
                    
                    np.save("../hparams/"+binn_name+"_hparams.npy",data)
                    
        except:
            
            print("Model: " + binn_model)
            print("No weights found for surface_weight: " + str(surface_weight) + "\n pde_weight: " + str(pde_weight))
    
    
    return sf_hparam, pde_hparam
    
    
def selectBINNModelRD(data_name, binn_model, cell_line, replicate):
    
    #load in data
    inputs, outputs, shape  = load_cell_migration_data(f"{data_name}.npy",cell_line=cell_line,replicate=replicate,plot=False)
    
    #find final time
    X = inputs[:,0]
    T = inputs[:,1]
    x = np.unique(X)
    t = np.unique(T)
    tMin = np.min(t)
    tMax = np.max(t)
    
    binn_name = f'{binn_model}_{cell_line}_{replicate}'
    save_folder = '../Weights/'
    
    surface_weights = [.1,1.0,10.0]
    pde_weights = [.1,1.0,10]
    
    S,P = np.meshgrid(surface_weights,pde_weights)
    S = S.reshape(-1)
    P = P.reshape(-1)
    
    OOS_MSE_test_min = np.inf
    for surface_weight,pde_weight in zip(S,P):

        try:
        
            ### Load params
            save_name =  binn_name + "_sf_weight_"+str(surface_weight)+"_pde_weight_"+str(pde_weight)
            model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name)

            #make tensor of x and t=t_0, then run it through surface fitter for initial condition
            inputs_0 = to_torch(np.hstack([x[:,None],t[0]*np.ones(x.shape)[:,None]]))
            u0 = to_numpy(binn.surface_fitter(inputs_0))[:,0]
                        
            D, G, A = recover_binn_params(binn) 

            ### Simulate PDE
            RHS = PDESolver.PDE_RHS
            u_sim = PDESolver.PDE_sim(RHS, u0, x, t, D, G, A, coords="spherical")

            u_sim = u_sim.reshape(-1)

            ### Compare
            u_sim_80 = u_sim[T>.8*tMax].reshape(len(x),-1)
            U_80     = outputs[T>.8*tMax].reshape(len(x),-1)

            OOS_MSE = np.linalg.norm(u_sim_80-U_80)

            if OOS_MSE < OOS_MSE_test_min:

                    OOS_MSE_test_min = OOS_MSE

                    sf_hparam = surface_weight
                    pde_hparam = pde_weight

                    u_sim_min_OOS = u_sim
                    binn_test_OOS_min = binn

                    data = {}
                    data['surface_weight'] = sf_hparam
                    data['pde_weight']  = pde_hparam
                    data['name'] = binn_name
                    data['simulation'] = u_sim_min_OOS
                    data['OOS_MSE'] = OOS_MSE_test_min
                    np.save("../hparams/"+binn_name+"_hparams.npy",data)
            
        except:
            
            print("Model: " + binn_model)
            print("No weights found for surface_weight: " + str(surface_weight) + "\n pde_weight: " + str(pde_weight))
    
    
    return sf_hparam, pde_hparam