"""
Plot Dixit data perturb-seq results 

"""
import numpy as np
import os
#import networkx as nx
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from realdata.dixit.dixit_meta import DIXIT_ESTIMATED_FOLDER, DIXIT_FIGURES_FOLDER
from realdata.dixit.dixit_meta import nnodes, EFFECTIVE_NODES, true_B_dixit_paper, true_B_igsp_paper

B0 = true_B_dixit_paper.copy()
np.fill_diagonal(B0, 0)
B0_skeleton = B0 + B0.T
B0_skeleton[np.where(B0_skeleton)] = 1

B1 = true_B_igsp_paper.copy()
np.fill_diagonal(B1, 0)
B1_skeleton = B1 + B1.T
B1_skeleton[np.where(B1_skeleton)] = 1

ALGS2COLORS = dict(zip(['ours','utigsp_gauss', 'utigsp_star_gauss', 'utigsp_hsic','utigsp_star_hsic'],\
                       mcolors.BASE_COLORS))
ALGS2MARKERS = {'ours':'o','utigsp_gauss': 'P', 'utigsp_star_gauss': '*', 'utigsp_hsic': 'X', 'utigsp_star_hsic': 'x'}
    
xticks_size = 14
yticks_size = 14
xlabel_size = 18
ylabel_size = 18
legend_size = 10
legend_loc = 'upper left'


def read_results(file, B, skeleton, nodes='all', method='utigsp',delete_bi_directions=False):
    tp = []
    fp = []
    tp_skeleton = []
    fp_skeleton = []
    time = []
    for vals in file.keys():
        if method == 'utigsp':
            estimated_dag = file[vals]['estimated_dag']
        else:
            estimated_dag = file[vals]['estimated_cpdag']
            if delete_bi_directions == True:
                estimated_dag[np.where(estimated_dag*estimated_dag.T)] = 0
            
        estimated_skeleton = file[vals]['estimated_skeleton']
        if nodes == 'all':
            tp.append(int(np.sum(estimated_dag*B)))
            fp.append(int(np.sum(estimated_dag)-tp[-1]))
            tp_skeleton.append(int(np.sum(estimated_skeleton*skeleton)/2))
            fp_skeleton.append(int(np.sum(estimated_skeleton)/2 - tp_skeleton[-1]))   
            time.append(file[vals]['time'])

        else:
            tp.append(int(np.sum(estimated_dag[:,EFFECTIVE_NODES]*B[:,EFFECTIVE_NODES])))
            fp.append(int(np.sum(estimated_dag[:,EFFECTIVE_NODES])-tp[-1]))
            tp_skeleton.append(int(np.sum(estimated_skeleton[:,EFFECTIVE_NODES]*skeleton[:,EFFECTIVE_NODES]))-\
                               int(np.sum(estimated_skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES]*skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES])))
            fp_skeleton.append(int(np.sum(estimated_skeleton[:,EFFECTIVE_NODES]) -\
                                   np.sum(estimated_skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES]) - tp_skeleton[-1]))                
            time.append(file[vals]['time'])

    return tp, fp, tp_skeleton, fp_skeleton, time

#%%
# 'dixit_paper' or 'igsp_paper' for ground truth reference
reference = 'dixit_paper'
utigsp_ci_test = 'gauss'    

if reference == 'dixit_paper':
    B = B0
    correct_skeleton = B0_skeleton
elif reference == 'igsp_paper':
    B = B1
    correct_skeleton = B1_skeleton

n_possible_skeleton = int(nnodes*(nnodes-1)/2)
n_true_skeleton = int(np.sum(correct_skeleton)/2)

nI = len(EFFECTIVE_NODES)
n_possible_skeleton_int = int(nI*(nI-1)/2 + nI*(nnodes-nI))
n_true_skeleton_int = int(np.sum(correct_skeleton[:,EFFECTIVE_NODES])-np.sum(correct_skeleton[EFFECTIVE_NODES][:,EFFECTIVE_NODES]))

#%%
# load UTIGSP results
with open(DIXIT_ESTIMATED_FOLDER+'/utigsp_gauss.pkl', 'rb') as f:
    res_utigsp_gauss = pickle.load(f)
    
with open(DIXIT_ESTIMATED_FOLDER+'/utigsp_star_gauss.pkl', 'rb') as f:
    res_utigsp_star_gauss = pickle.load(f)   
    
with open(DIXIT_ESTIMATED_FOLDER+'/utigsp_hsic.pkl', 'rb') as f:
    res_utigsp_hsic = pickle.load(f)
    
with open(DIXIT_ESTIMATED_FOLDER+'/utigsp_star_hsic.pkl', 'rb') as f:
    res_utigsp_star_hsic = pickle.load(f)    

# load our results
with open(DIXIT_ESTIMATED_FOLDER+'/our_results_2.pkl', 'rb') as f:
    res_ours = pickle.load(f)    
    

#%%

utigsp_gauss_tp, utigsp_gauss_fp, utigsp_gauss_tp_skeleton, utigsp_gauss_fp_skeleton, utigsp_gauss_time = \
    read_results(res_utigsp_gauss, B, correct_skeleton,method='utigsp')

utigsp_star_gauss_tp, utigsp_star_gauss_fp, utigsp_star_gauss_tp_skeleton, utigsp_star_gauss_fp_skeleton, utigsp_star_gauss_time = \
    read_results(res_utigsp_star_gauss, B, correct_skeleton,method='utigsp')
 
utigsp_hsic_tp, utigsp_hsic_fp, utigsp_hsic_tp_skeleton, utigsp_hsic_fp_skeleton, utigsp_hsic_time = \
    read_results(res_utigsp_hsic, B, correct_skeleton,method='utigsp')

utigsp_star_hsic_tp, utigsp_star_hsic_fp, utigsp_star_hsic_tp_skeleton, utigsp_star_hsic_fp_skeleton, utigsp_star_hsic_time = \
    read_results(res_utigsp_star_hsic, B, correct_skeleton,method='utigsp')
        
ours_tp, ours_fp, ours_tp_skeleton, ours_fp_skeleton, ours_time_all = read_results(res_ours, B, correct_skeleton,method='ours')
    
ours_time = []    
for ours_time_instant in ours_time_all:
    ours_time.append(sum(list(ours_time_instant.values())))
    
    
#%%  ======= PLOT ROC for directed edges recovery ==========
plt.clf()
plt.scatter(ours_fp,ours_tp,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
if utigsp_ci_test == 'gauss':
    plt.scatter(utigsp_gauss_fp,utigsp_gauss_tp,label='UTIGSP',marker=ALGS2MARKERS['utigsp_gauss'],color=ALGS2COLORS['utigsp_gauss'])
    plt.scatter(utigsp_star_gauss_fp,utigsp_star_gauss_tp,label='UTIGSP*',marker=ALGS2MARKERS['utigsp_star_gauss'],color=ALGS2COLORS['utigsp_star_gauss'])
elif utigsp_ci_test == 'hsic':
    plt.scatter(utigsp_hsic_fp,utigsp_hsic_tp,label='UTIGSP',marker=ALGS2MARKERS['utigsp_hsic'],color=ALGS2COLORS['utigsp_hsic'])
    plt.scatter(utigsp_star_hsic_fp,utigsp_star_hsic_tp,label='UTIGSP*',marker=ALGS2MARKERS['utigsp_star_hsic'],color=ALGS2COLORS['utigsp_star_hsic'])


plt.xlim([0,50])
plt.ylim([0,16])
#plt.title('Directed Edges')
plt.xlabel('False positives',size=xlabel_size)
plt.ylabel('True positives',size=ylabel_size)
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.grid()
plt.legend(fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(os.path.join(DIXIT_FIGURES_FOLDER, 'dixit_directed_'+utigsp_ci_test+'.eps'))

#%%  ======== PLOT ROC for skeleton recovery ===========
plt.clf()
plt.scatter(ours_fp_skeleton,ours_tp_skeleton,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
if utigsp_ci_test == 'gauss':
    plt.scatter(utigsp_gauss_fp_skeleton,utigsp_gauss_tp_skeleton,label='UTIGSP',marker=ALGS2MARKERS['utigsp_gauss'],color=ALGS2COLORS['utigsp_gauss'])
    plt.scatter(utigsp_star_gauss_fp_skeleton,utigsp_star_gauss_tp_skeleton,label='UTIGSP*',marker=ALGS2MARKERS['utigsp_star_gauss'],color=ALGS2COLORS['utigsp_star_gauss'])
elif utigsp_ci_test == 'hsic':
    plt.scatter(utigsp_hsic_fp_skeleton,utigsp_hsic_tp_skeleton,label='UTIGSP',marker=ALGS2MARKERS['utigsp_hsic'],color=ALGS2COLORS['utigsp_hsic'])
    plt.scatter(utigsp_star_hsic_fp_skeleton,utigsp_star_hsic_tp_skeleton,label='UTIGSP*',marker=ALGS2MARKERS['utigsp_star_hsic'],color=ALGS2COLORS['utigsp_star_hsic'])

plt.plot([0, n_possible_skeleton - n_true_skeleton], [0, n_true_skeleton], color='grey')
plt.xlim([0,100])
plt.ylim([0,18])
#plt.title('Skeleton')
plt.xlabel('False positives',size=xlabel_size)
plt.ylabel('True positives',size=ylabel_size)
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.grid()
plt.legend(fontsize=legend_size,loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(DIXIT_FIGURES_FOLDER, 'dixit_skeleton_'+utigsp_ci_test+'.eps'))

#%%
# ========= PLOT ROC for directed recovery: Gauss and HSIC tests together
plt.clf()
plt.scatter(ours_fp,ours_tp,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
plt.scatter(utigsp_gauss_fp,utigsp_gauss_tp,label='UTIGSP-Gauss',marker=ALGS2MARKERS['utigsp_gauss'],color=ALGS2COLORS['utigsp_gauss'])
plt.scatter(utigsp_star_gauss_fp,utigsp_star_gauss_tp,label='UTIGSP*-Gauss',marker=ALGS2MARKERS['utigsp_star_gauss'],color=ALGS2COLORS['utigsp_star_gauss'])
plt.scatter(utigsp_hsic_fp,utigsp_hsic_tp,label='UTIGSP-HSIC',marker=ALGS2MARKERS['utigsp_hsic'],color=ALGS2COLORS['utigsp_hsic'])
plt.scatter(utigsp_star_hsic_fp,utigsp_star_hsic_tp,label='UTIGSP*-HSIC',marker=ALGS2MARKERS['utigsp_star_hsic'],color=ALGS2COLORS['utigsp_star_hsic'])


plt.xlim([0,50])
plt.ylim([0,16])
#plt.title('Directed Edges')
plt.xlabel('False positives',size=xlabel_size)
plt.ylabel('True positives',size=ylabel_size)
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.grid()
plt.legend(fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(os.path.join(DIXIT_FIGURES_FOLDER, 'dixit_directed_all.eps'))

#  ======== PLOT ROC for skeleton recovery ===========
plt.clf()
plt.scatter(ours_fp_skeleton,ours_tp_skeleton,label='ours',marker=ALGS2MARKERS['ours'],color=ALGS2COLORS['ours'])
plt.scatter(utigsp_gauss_fp_skeleton,utigsp_gauss_tp_skeleton,label='UTIGSP-Gauss',marker=ALGS2MARKERS['utigsp_gauss'],color=ALGS2COLORS['utigsp_gauss'])
plt.scatter(utigsp_star_gauss_fp_skeleton,utigsp_star_gauss_tp_skeleton,label='UTIGSP*-Gauss',marker=ALGS2MARKERS['utigsp_star_gauss'],color=ALGS2COLORS['utigsp_star_gauss'])
plt.scatter(utigsp_hsic_fp_skeleton,utigsp_hsic_tp_skeleton,label='UTIGSP-HSIC',marker=ALGS2MARKERS['utigsp_hsic'],color=ALGS2COLORS['utigsp_hsic'])
plt.scatter(utigsp_star_hsic_fp_skeleton,utigsp_star_hsic_tp_skeleton,label='UTIGSP*-HSIC',marker=ALGS2MARKERS['utigsp_star_hsic'],color=ALGS2COLORS['utigsp_star_hsic'])

plt.plot([0, n_possible_skeleton - n_true_skeleton], [0, n_true_skeleton], color='grey')
plt.xlim([0,100])
plt.ylim([0,18])
#plt.title('Skeleton')
plt.xlabel('False positives',size=xlabel_size)
plt.ylabel('True positives',size=ylabel_size)
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.grid()
plt.legend(fontsize=legend_size,loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(DIXIT_FIGURES_FOLDER, 'dixit_skeleton_all.eps'))
