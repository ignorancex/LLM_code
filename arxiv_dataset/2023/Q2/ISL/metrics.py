# metrics and plotting

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import pandas as pd
import networkx as nx
import random
from gen_data import simulate_dag, simulate_lsem
from isl import refit
import warnings
from tqdm import tqdm
from utils import add_int
warnings.filterwarnings('ignore')

def plot_it(B,p=2,s=6,save='',noXA=False,noXM=True, tol=0.1,ones=True):
    """Function to plot any given graph.

    Args:
        B: graph of interest
        s: number of potential mediators
        p: number of potential moderators
        save: save location of produced plot; if '', then the plot will not be saved
        noXA: if True, it will be assumed that interaction has not been added to the graph
        tol: threshold for weights within graph to be considered nonzero
        ones: if true, the weights above the threshod will be set to either 1 or -1 depending on sign
              (i.e. assumes the true graph has weights in {1,0,-1}; set to false if plotting real data)
    """
    if tol != 0 and ones:
        B_pruned = np.where(abs(B) > tol, np.multiply(np.ones_like(B),np.sign(B)), 0)
    elif not ones: B_pruned = np.where(abs(B) > tol, B, 0)
    else: B_pruned = np.copy(B)
    p2 = p if noXA else p*2
    s2 = 0 if noXM else p*s
    if not ones: plt.matshow(B_pruned.T, cmap = 'bwr', norm=colors.CenteredNorm())
    else: plt.matshow(B_pruned.T, cmap = 'bwr', vmin = -1, vmax = 1)
    plt.xticks(ticks=[0,p,p2+1,p2+1+s+s2],
               labels=['X','A','M','Y'])
    plt.yticks(ticks=[0,p,p2+1,p2+1+s+s2],
               labels=['X','A','M','Y'])
    # bounding box around mediators
    currentAxis = plt.gca()
    half = 0.5
    currentAxis.add_patch(Rectangle((p2+1-half, p2+1-half), s, s, edgecolor="yellow", facecolor="none"))
    # bounding box for treatment
    currentAxis.add_patch(Rectangle((0-half, p-half), p2+2+s+s2, 1, edgecolor="orange", facecolor="none"))
    currentAxis.add_patch(Rectangle((p-half, 0-half), 1, p2+2+s+s2, edgecolor="orange", facecolor="none"))
    plt.colorbar()
    if save != '':
        plt.savefig(save)

def calculate_effects(B,p,s,x,tol=0.1,ones=True, noXM=1):
    """Calculate causal effects in ANOCE based on estimated weighted adjacency matrix.
        
        Args:
        B: estimated weighted adjacency matrix B
        p: number of potential moderators
        s: number of potential mediators
        x: the vector of potential moderator values to be used to calculate effects
        tol: threshold for weights within graph to be considered nonzero
        ones: if true, the weights above the threshod will be set to either 1 or -1 depending on sign
              (i.e. assumes the true graph has weights in {1,0,-1}; set to false if plotting real data)
        
        Returns:
        array of effects: [HTE,HDE,HIE,HME,HDM,HIM]
        HTE: heterogeneous total effect
        HDE: heterogeneous direct effect
        HIE: heterogeneous indirect effect
        HME: heterogeneous total effect for mediators
        HDM: heterogeneous direct effect for mediators
        HIM: heterogeneous indirect effect for mediators
        """
    if tol != 0 and ones:
        predB = np.where(abs(B) > tol, np.multiply(np.ones_like(B),np.sign(B)), 0)
    elif not ones: predB = np.where(abs(B) > tol, B, 0)
    else: predB = np.copy(B)
    # Number of nodes in the graph
    d = predB.shape[0]
    y = d-1
    xm = 2*p+s+1
    # Calculate causal effects in ANOCE
    alpha = predB[p, (2*p+1):xm] # A on M
    beta = predB[(2*p+1):xm, y] # M on Y
    B_T_XA = predB[(p+1):(2*p+1),(2*p+1):xm].T
    gamma_XA = predB[(p+1):(2*p+1),y].T
    
    # incorporate gamma_XM into beta (gamma_M)
    if not noXM:
        gamma_XM = predB[(2*p+s+1):-1,-1].T
        beta += gamma_XM.reshape(s,p).dot(x)
    
    DE = predB[p, y] + gamma_XA.dot(x) # natural direct effect DE
    
    trans_BM = predB[(2*p+1):xm, (2*p+1):xm].T
    zeta = np.dot(alpha+B_T_XA.dot(x), np.linalg.inv(np.identity(s) - trans_BM)) # the causal effect of A on M
    IE = np.squeeze(np.dot(zeta, beta)) # natural indirect effect IE
    
    TE = DE + IE # total effect
    
    DM = np.multiply(beta.reshape((s, 1)), zeta.reshape((1, s))).diagonal().copy() # natural direct effect for mediators
    
    # might need to change
    eta = np.zeros((s)) # the individual mediation effect in Chakrabortty et al. (2018)
    for i in range(2*p+1, xm):
        predB_1reduce = np.delete(predB, i, 0)
        predB_1reduce = np.delete(predB_1reduce, i, 1)
        alpha_R = predB_1reduce[p, (2*p+1):xm-1]
        beta_R = predB_1reduce[(2*p+1):xm-1, y-1]
        if not noXM:
            gamma_XM_R = gamma_XM.reshape(s,p)
            gamma_XM_R = np.delete(gamma_XM_R, i-(2*p+1), 0)
            beta_R += gamma_XM_R.dot(x)
        trans_BM_R = predB_1reduce[(2*p+1):xm-1, (2*p+1):xm-1].T
        B_T_XA_R = predB_1reduce[(p+1):(2*p+1),(2*p+1):xm-1].T
        zeta_R = np.dot(alpha_R+B_T_XA_R.dot(x), np.linalg.inv(np.identity(s-1) - trans_BM_R))
        eta[i - 1 - 2*p] = IE - np.squeeze(np.dot(zeta_R, beta_R))
    
    IM = eta - DM # natural indirect effect for mediators

    return [np.round(TE,4),np.round(DE,4),np.round(IE,4), np.round(eta,4), np.round(DM,4), np.round(IM,4)]

def count_accuracy(B_true, B_est,tol=0.1):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / condition negative
        tpr: (true positive) / condition positive
        shd: undirected extra + undirected missing + reverse
    """
    B_est = np.where(abs(B_est) > tol, np.ones_like(B_est)*np.sign(B_est), 0)
    d = B_true.shape[0]
    B_true = np.where(B_true > 0 ,1,0)
    B_est = np.where(B_est > 0 ,1,0)
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, shd

# function to average over simulation results
def get_results(files, seeds, s, p, n_samp, true_graph, tol = 0.4, x = np.array([0.5,0.5]),
               save='', noXM=1):
    """Averages the results over graphs in given files.

    Args:
        files: files containing each estimated graph
        seeds: seeds for each graph
        s: number of potential mediators
        p: number of potential moderators
        n_samp: number of samples used to estimate graph
        true_graph: the true graph
        tol: threshold for weights within graph to be considered nonzero
        x: the vector of potential moderator values to be used to calculate effects
        save: location to save average plot to; empty string means that the plot will not be saved
        

    Returns:
        avgB: average graph of estimated graphs
    """
    fdr = []
    tpr = []
    shd = []
    te = []
    de = []
    ie = []
    tme = []
    dme = []
    ime = []
    B_tru = nx.to_numpy_array(true_graph)
    te_t,de_t,ie_t,tme_t,dme_t,ime_t = calculate_effects(B_tru,p,s,x,tol=0, noXM=noXM)

    avgB = None
    for i, seed in enumerate(seeds):
        data = simulate_lsem(true_graph,n_samp,p,seed=seed,noise=1)[:,:,0]
        with open(files[i], 'rb') as fp:
            predB = refit(pickle.load(fp), data, tol = tol)
        if avgB is None: avgB = np.copy(predB)
        else: avgB += (1/(i+1))*(predB - avgB)
        fdr_i, tpr_i, shd_i = count_accuracy(B_tru, np.copy(predB),tol=0)
        fdr.append(fdr_i)
        tpr.append(tpr_i)
        shd.append(shd_i)
        te_i,de_i,ie_i,tme_i,dme_i,ime_i = calculate_effects(np.copy(predB),p,s,x,tol=0, noXM=noXM)
        te.append(te_i)
        de.append(de_i)
        ie.append(ie_i)
        tme.append(tme_i)
        dme.append(dme_i)
        ime.append(ime_i)
    de = np.array(de)
    ie = np.array(ie)
    dme = np.array(dme)
    ime = np.array(ime)
    fdr = "%1.2f(%1.2f)" %(np.mean(fdr),np.std(fdr))
    tpr = "%1.2f(%1.2f)" %(np.mean(tpr),np.std(tpr))
    shd = "%1.2f(%1.2f)" %(np.mean(shd),np.std(shd))
    de = "%1.2f(%1.2f)" %(np.mean(de-de_t),np.std(de-de_t))
    ie = "%1.2f(%1.2f)" %(np.mean(ie-ie_t),np.std(ie-ie_t))
    
    def me(x):
        x_mean = np.mean(x,0)
        x_std = np.std(x,0)
        return ["%1.2f(%1.2f)" %(x_mean[i-1],x_std[i-1]) for i in range(1,1+len(x_mean))]
    
    dme = me(dme-dme_t)
    ime = me(ime-ime_t)

    plot_it(B_tru,p=p,s=s,tol=0, save='Assets/Simulation/'+save[:2]+'.png' if save else '', noXM=noXM)
    plot_it(avgB,p=p,s=s,tol=0, save='Assets/Simulation/'+save+'.png' if save else '', noXM=noXM)
    print(fdr)
    print(tpr)
    print(shd)
    print(de)
    print(ie)
    print(dme)
    print(ime)
    return avgB

# example of how to create bootstrap intervals in the case of the real data example
# given in the paper where there are 4 potential mediators and 9 potential moderators
def calc_int(x_vec, files, data, B, alpha):
    """Calculates bootrap interval given estimated graphs for bootstrap samples.

    Args:
        x_vec: the vector of potential moderator values to be used to calculate effects
        files: files containing each estimated bootstrap graph
        data: nparray of data used
        B: estimated graph given data
        alpha: significance level
        
    Returns:
        15 x 2 nparray of bootstrap estimates
    """
    
    s = 4
    p = 9

    def percentile(boot=[0], alpha=0, **kwargs):
        return np.quantile(np.copy(boot),[alpha,1-alpha],axis=0)

    def gaussian(est=0, boot=[0], alpha=0, **kwargs):
        se = np.std(np.copy(boot),axis=0)
        z = norm.ppf(1-alpha, loc=0, scale=1)
        return np.array([est-z*se,est+z*se])
        nboot = length(files)
        alpha = alpha / 2
    
    # initialize empty arrays
    boot = np.zeros([nboot,15])

    est = np.zeros([15]) # number of estimates
    TE,DE,IE, M, ME, MI = calculate_effects(B,p,x_vec,tol=0, ones=False)
    est[:3] = [TE,DE,IE]
    est[3:7] = M
    est[7:11] = ME
    est[11:] = MI

    # generate bootstrap estimates
    for j in tqdm(range(nboot)): # tqdm used for running estimate on time remaining
        with open(files[j], 'rb') as fp:
            graph = pickle.load(fp)
        # file contains [graph, data_seed] if different seed then index
        if isinstance(graph, list): graph, data_seed = graph
        else: data_seed = j
        np.random.seed(data_seed)
        data_boot = data[np.random.choice(n_samp,n_samp)] # resample with replacement
        tol_boot = get_opt_tol(data_boot,graph,p)
        est_graph = refit(graph, data_boot, tol=tol_boot, alpha=alpha)

        # calculate estimate for boot
        TE,DE,IE, M, ME, MI = calculate_effects(est_graph,p,x_vec,tol=0, ones=False)
        boot[j,:3] = [TE,DE,IE]
        boot[j,3:7] = M
        boot[j,7:11] = ME
        boot[j,11:] = MI
        
    return percentile(est=est,boot=boot,alpha=alpha) #gaussian can be used instead if desired