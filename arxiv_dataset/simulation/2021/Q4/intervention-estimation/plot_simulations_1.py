"""
plot the results of simulations in rum_sim_1.py file
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle
from config import SIMULATIONS_ESTIMATED_FOLDER, SIMULATIONS_FIGURES_FOLDER

markers = ['empty','o','v','P','D','X']
    

def load_res(filename,algos='ours'):
    with open(SIMULATIONS_ESTIMATED_FOLDER+'/'+filename+'.pkl','rb') as f:
        res = pickle.load(f)
        
    I_tp = res['I_tp']
    I_fp = res['I_fp']
    I_fn = res['I_fn']
    e_tp = res['e_tp']
    e_fp = res['e_fp']
    e_fn = res['e_fn']
    time_ours = res['time']
    if algos == 'ours':
        return res, I_tp, I_fp, I_fn, e_tp, e_fp, e_fn, time_ours
    elif algos == 'comparison':
        I_tp_ref = res['I_tp_ref']
        I_fp_ref = res['I_fp_ref']
        I_fn_ref = res['I_fn_ref']
        e_tp_ref = res['e_tp_ref']
        e_fp_ref = res['e_fp_ref']
        e_fn_ref = res['e_fn_ref']
        time_ref = res['time_ref']        
        return res, I_tp, I_fp, I_fn, e_tp, e_fp, e_fn, time_ours, \
            I_tp_ref, I_fp_ref, I_fn_ref, e_tp_ref, e_fp_ref, e_fn_ref, time_ref


def scores(I_tp,I_fp,I_fn,e_tp,e_fp,e_fn):
    I_precision = np.sum(I_tp,0) / (np.sum(I_tp,0)+np.sum(I_fp,0))
    I_recall =  np.sum(I_tp,0) / (np.sum(I_tp,0)+np.sum(I_fn,0))
    I_f1 = np.sum(I_tp,0) / (np.sum(I_tp,0)+(np.sum(I_fp,0)+np.sum(I_fn,0))/2)
    
    e_precision = np.sum(e_tp,0) / (np.sum(e_tp,0)+np.sum(e_fp,0))
    e_recall =  np.sum(e_tp,0) / (np.sum(e_tp,0)+np.sum(e_fn,0))
    e_f1 = np.sum(e_tp,0) / (np.sum(e_tp,0)+(np.sum(e_fp,0)+np.sum(e_fn,0))/2)
    
    return I_precision, I_recall, I_f1, e_precision, e_recall ,e_f1


xticks_size = 14
yticks_size = 14
xlabel_size = 18
ylabel_size = 18
legend_size = 12
legend_loc = 'lower right'
linewidth = 2.6
linestyle = '--'
markersize = 9

#%% load results for I recovery, increased variance, ours only
res_inc, I_tp_inc, I_fp_inc, I_fn_inc, e_tp_inc, e_fp_inc, e_fn_inc, time_inc = load_res('increased_variance_1',algos='ours')
I_precision_inc, I_recall_inc, I_f1_inc, e_precision_inc, e_recall_inc, e_f1_inc = \
    scores(I_tp_inc, I_fp_inc, I_fn_inc, e_tp_inc, e_fp_inc, e_fn_inc)

p_list = res_inc['p_list']
density_list = res_inc['density_list']
n_samples_list = np.asarray(res_inc['n_samples_list'])


plt.figure('increase - density 1.5')
for p_idx in range(1,len(p_list)):
    plt.plot(n_samples_list.astype('str'),I_precision_inc[p_idx,0],linestyle=linestyle,linewidth=linewidth,marker=markers[p_idx],markersize=markersize)
    

legend_str = ['p=%d'%p for p in p_list][1:] 
#plt.xticks(ticks=n_samples_list,labels=n_samples_list)
plt.grid()
plt.xlabel('Number of Samples',size=xlabel_size)
#plt.ylabel('Precision of estimating intervention targets',size=12)
plt.ylabel('Precision',size=ylabel_size)
plt.ylim([0.5,1])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(legend_str,fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_inc_density15.eps')

#%
plt.figure('increase - density 2.5')
for p_idx in range(1,len(p_list)):
    plt.plot(n_samples_list.astype('str'),I_precision_inc[p_idx,1],linestyle=linestyle,linewidth=linewidth,marker=markers[p_idx],markersize=markersize)

legend_str = ['p=%d'%p for p in p_list][1:]

#plt.xticks(ticks=n_samples_list,labels=n_samples_list)
plt.grid()
plt.xlabel('Number of Samples',size=xlabel_size)
#plt.ylabel('Precision of estimating intervention targets',size=12)
plt.ylabel('Precision',size=ylabel_size)
plt.ylim([0.3,1])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(legend_str,fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_inc_density25.eps')

#%% load results for I recovery, shifted mean, ours only
res_shift, I_tp_shift, I_fp_shift, I_fn_shift, e_tp_shift, e_fp_shift, e_fn_shift, time_shift =\
    load_res('shifted_mean_1',algos='ours')
I_precision_shift, I_recall_shift, I_f1_shift, e_tp_shift, e_fp_shift, e_f1_shift = \
    scores(I_tp_shift, I_fp_shift, I_fn_shift, e_tp_shift, e_fp_shift, e_fn_shift)

n_samples_list = np.asarray(res_shift['n_samples_list'])
p_list = res_shift['p_list']

plt.figure('shift - density 1.5')
for p_idx in range(1,len(p_list)):
    plt.plot(n_samples_list.astype('str'),I_precision_shift[p_idx,0],linestyle=linestyle,linewidth=linewidth,marker=markers[p_idx],markersize=markersize)

legend_str = ['p=%d'%p for p in p_list][1:]
#plt.xticks(ticks=n_samples_list,labels=n_samples_list)
plt.grid()
plt.xlabel('Number of Samples',size=xlabel_size)
#plt.ylabel('Precision of estimating intervention targets',size=12)
plt.ylabel('Precision',size=ylabel_size)
plt.ylim([0.5,1])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(legend_str,fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_shift_density15.eps')

plt.figure('shift - density 2.5')
for p_idx in range(1,len(p_list)):
    plt.plot(n_samples_list.astype('str'),I_precision_shift[p_idx,1],linestyle=linestyle,linewidth=linewidth,marker=markers[p_idx],markersize=markersize)

legend_str = ['p=%d'%p for p in p_list]

#plt.xticks(ticks=n_samples_list,labels=n_samples_list)
plt.grid()
plt.xlabel('Number of Samples',size=xlabel_size)
#plt.ylabel('Precision of estimating intervention targets',size=12)
plt.ylabel('Precision',size=ylabel_size)
plt.ylim([0.3,1])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(legend_str,fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_shift_density25.eps')

#%% load results, perfect intervention, ours only
res_per, I_tp_per, I_fp_per, I_fn_per, e_tp_per, e_fp_per, e_fn_per, time_per =\
    load_res('perfect_1',algos='ours')
I_precision_per, I_recall_per, I_f1_per, e_tp_per, e_fp_per, e_f1_per = \
    scores(I_tp_per, I_fp_per, I_fn_per, e_tp_per, e_fp_per, e_fn_per)

n_samples_list = np.asarray(res_per['n_samples_list'])
p_list = res_per['p_list']

plt.figure('perfect - density 1.5')
for p_idx in range(1,len(p_list)):
    plt.plot(n_samples_list.astype('str'),I_precision_per[p_idx,0],linestyle=linestyle,linewidth=linewidth,marker=markers[p_idx],markersize=markersize)

legend_str = ['p=%d'%p for p in p_list][1:]
#plt.xticks(ticks=n_samples_list,labels=n_samples_list)
plt.grid()
plt.xlabel('Number of Samples',size=xlabel_size)
#plt.ylabel('Precision of estimating intervention targets',size=12)
plt.ylabel('Precision',size=ylabel_size)
plt.ylim([0.5,1])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(legend_str,fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_per_density15.eps')

plt.figure('perfect - density 2.5')
for p_idx in range(1,len(p_list)):
    plt.plot(n_samples_list.astype('str'),I_precision_per[p_idx,1],linestyle=linestyle,linewidth=linewidth,marker=markers[p_idx],markersize=markersize)

legend_str = ['p=%d'%p for p in p_list][1:]
#plt.xticks(ticks=n_samples_list,labels=n_samples_list)
plt.grid()
plt.xlabel('Number of Samples',size=xlabel_size)
#plt.ylabel('Precision of estimating intervention targets',size=12)
plt.ylabel('Precision',size=ylabel_size)
plt.ylim([0.3,1])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(legend_str,fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_per_density25.eps')

#%% load results, increased variance, for comparison
res_inc, I_tp_inc, I_fp_inc, I_fn_inc, e_tp_inc, e_fp_inc, e_fn_inc, time_inc, \
    I_tp_ref_inc, I_fp_ref_inc, I_fn_ref_inc, e_tp_ref_inc, e_fp_ref_inc, e_fn_ref_inc, time_ref_inc = \
        load_res('increased_variance_comparison_1',algos='comparison')
        
        
I_precision_inc, I_recall_inc, I_f1_inc, e_precision_inc, e_recall_inc, e_f1_inc = \
    scores(I_tp_inc, I_fp_inc, I_fn_inc, e_tp_inc, e_fp_inc, e_fn_inc)
    
I_precision_ref_inc, I_recall_ref_inc, I_f1_ref_inc, e_precision_ref_inc, e_recall_ref_inc, e_f1_ref_inc = \
    scores(I_tp_ref_inc, I_fp_ref_inc, I_fn_ref_inc, e_tp_ref_inc, e_fp_ref_inc, e_fn_ref_inc)

p_list = res_inc['p_list']
density_list = res_inc['density_list']
n_samples_list = np.asarray(res_inc['n_samples_list'])

#%% load results, shifted mean, for comparison

res_shift, I_tp_shift, I_fp_shift, I_fn_shift, e_tp_shift, e_fp_shift, e_fn_shift, time_shift, \
    I_tp_ref_shift, I_fp_ref_shift, I_fn_ref_shift, e_tp_ref_shift, e_fp_ref_shift, e_fn_ref_shift, time_ref_shift = \
        load_res('shifted_mean_comparison_1',algos='comparison')
        
        
I_precision_shift, I_recall_shift, I_f1_shift, e_precision_shift, e_recall_shift, e_f1_shift = \
    scores(I_tp_shift, I_fp_shift, I_fn_shift, e_tp_shift, e_fp_shift, e_fn_shift)
    
I_precision_ref_shift, I_recall_ref_shift, I_f1_ref_shift, e_precision_ref_shift, e_recall_ref_shift, e_f1_ref_shift = \
    scores(I_tp_ref_shift, I_fp_ref_shift, I_fn_ref_shift, e_tp_ref_shift, e_fp_ref_shift, e_fn_ref_shift)

n_samples_list = np.asarray(res_shift['n_samples_list'])
p_list = res_shift['p_list']

#%% load results, shifted mean, for comparison, p = 100 only
res_shift, I_tp_shift, I_fp_shift, I_fn_shift, e_tp_shift, e_fp_shift, e_fn_shift, time_shift, \
    I_tp_ref_shift, I_fp_ref_shift, I_fn_ref_shift, e_tp_ref_shift, e_fp_ref_shift, e_fn_ref_shift, time_ref_shift = \
        load_res('shifted_mean_comparison_2',algos='comparison')
        
        
I_precision_shift, I_recall_shift, I_f1_shift, e_precision_shift, e_recall_shift, e_f1_shift = \
    scores(I_tp_shift, I_fp_shift, I_fn_shift, e_tp_shift, e_fp_shift, e_fn_shift)
    
I_precision_ref_shift, I_recall_ref_shift, I_f1_ref_shift, e_precision_ref_shift, e_recall_ref_shift, e_f1_ref_shift = \
    scores(I_tp_ref_shift, I_fp_ref_shift, I_fn_ref_shift, e_tp_ref_shift, e_fp_ref_shift, e_fn_ref_shift)

n_samples_list = np.asarray(res_shift['n_samples_list'])
p_list = res_shift['p_list']
#%%
# plt.figure('increase_comparison')
# for p_idx in range(len(p_list)):
#     plt.plot(n_samples_list.astype('str'),I_precision_inc[p_idx],'--o')

# for p_idx in range(len(p_list)):
#     plt.plot(n_samples_list.astype('str'),I_precision_ref_inc[p_idx],'--o')

# legend_str = ['p=%d'%p for p in p_list]

# #plt.xticks(ticks=n_samples_list,labels=n_samples_list)
# plt.grid()
# plt.xlabel('Number of Samples')
# plt.ylabel('Precision of estimating intervention targets')
# plt.ylim([0.4,1])
# plt.legend(legend_str)
# #plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/ours_alone_inc.eps')



#%% load results, perfect intervention, comparison
res_per, I_tp_per, I_fp_per, I_fn_per, e_tp_per, e_fp_per, e_fn_per, time_per, \
    I_tp_ref_per, I_fp_ref_per, I_fn_ref_per, e_tp_ref_per, e_fp_ref_per, e_fn_ref_per, time_ref_per = \
        load_res('perfect_comparison_1',algos='comparison')
        
        
I_precision_per, I_recall_per, I_f1_per, e_precision_per, e_recall_per, e_f1_per = \
    scores(I_tp_per, I_fp_per, I_fn_per, e_tp_per, e_fp_per, e_fn_per)
    
I_precision_ref_per, I_recall_ref_per, I_f1_ref_per, e_precision_ref_per, e_recall_ref_per, e_f1_ref_per = \
    scores(I_tp_ref_per, I_fp_ref_per, I_fn_ref_per, e_tp_ref_per, e_fp_ref_per, e_fn_ref_per)
