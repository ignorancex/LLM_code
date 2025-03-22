import os
import argparse
import tables as h5
import pandas as pd
import numpy as np
import gvar as gv
import matplotlib as mpl
import matplotlib.pyplot as plt
# now module for Madras-Sokal autocorr time
from emcee import autocorr

# Figure formatting for paper
fig_width = 6.75 # in inches, 2x as wide as APS column
gr        = 1.618034333 # golden ratio
fig_size  = (fig_width, fig_width / gr)
fig_size2 = (fig_width, fig_width * 1.6)
plt_axes  = [0.14,0.14,0.85,0.85]
axes1     = [0.13,0.13,0.69,0.85]
axes2     = [0.821,0.13,0.12,0.85]
fs_text   = 18 # font size of text
fs_leg    = 14 # legend font size
mrk_size  = '5' # marker size
tick_size = 14 # tick size
lw        = 1 # line width
# saving as rc params
mpl.rcParams['figure.figsize'] = fig_size
mpl.rcParams['lines.linewidth'] = lw
mpl.rcParams['lines.markersize'] = mrk_size
mpl.rcParams['font.size'] = fs_text
mpl.rcParams['legend.fontsize'] = fs_leg
mpl.rcParams['axes.labelsize'] = fs_text
mpl.rcParams['xtick.labelsize'] = tick_size
mpl.rcParams['ytick.labelsize'] = tick_size
mpl.rcParams['backend'] = 'Agg'

# streams and cfg offsets for the 3 ensembles
streams = {
    'a15m135XL_s':['b','c','d','e'],
    'a09m135_s'  :['a','b'],
    'a06m310L_s' :['b','c'],
}
## offset in units of MDTU for visualization
offset = {
    'a15m135XL_s':[0, 2700, 5400, 8100],
    'a09m135_s'  :[0, 3422],
    'a06m310L_s' :[0, 3200],
}

## stitching of a15 pbp
stitching = {
    'a15m135XL_s': [250,5],
    'a09m135_s'  : [0,0],
    'a06m310L_s' : [0,0],
}

## measurement frequency in Len.
freq = {
    'a15m135XL_s':25,
    'a09m135_s'  :4,
    'a06m310L_s' :3,
}

pbp_reps = 5  # number of stochastic point sources for PBP

labels = {'a15m135XL_s':'a15m135XL', 'a09m135_s':'a09m135', 'a06m310L_s':'a06m310L'}
lattices = {'a15m135XL':'a15', 'a09m135':'a09', 'a06m310L':'a06'}
colors_a = {'a15':'#ec5d57', 'a12':'#70bf41', 'a09':'#51a7f9', 'a06':'#00FFFF'}

# we have a total of 6 observables
## The first 4 are measured every MDTU
## The last 2 are measured every TRAJ
obs_all = ['deltaS','pbp_c','pbp_s','pbp_l','acc_rej','plaq']
# forget about deltaS because it's complicated with a15m135
obs_a15 = ['pbp_c','pbp_s','pbp_l','acc_rej','plaq']
# we use MDTU and TRAJ as indices for observables
index_all = ['mdt','traj']
# for a15m135 pbp there is a separate index!!
index_a15 = ['mdt_pbp','traj']

####
def plot_pbp(ensemble,dataset,column):
    colors = ['r','orange','g','b','m']
    if len(streams[ensemble]) == 1:
        cs = [colors_a[lattices[labels[ensemble]]]]
    else:
        cs = colors[::int(len(colors)/len(streams[ensemble]))]
    gs = mpl.gridspec.GridSpec(1, 2,width_ratios=[5, 1])
    gs.update(wspace=0.00)
    #---
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    #---
    binned_data = np.array([])
    for i,s in enumerate(streams[ensemble]):
        # print every saved config
        idx = np.concatenate((np.arange(0,stitching[ensemble][0]),
                              np.arange(stitching[ensemble][0],dataset.loc[s].shape[0],freq[ensemble])))
        data = dataset.loc[s].iloc[idx][column].values
        index = np.concatenate((dataset.loc[s].iloc[idx].index[:stitching[ensemble][0]].values,
                                dataset.loc[s].iloc[idx].index[stitching[ensemble][0]:].values+stitching[ensemble][1]))+offset[ensemble][i]
        avg = np.mean(data)
        #---
        ax0.plot(index,data,label=s,color=cs[i],marker='s',mfc='None',alpha=.5)
        ax0.hlines(y=avg,xmin=index.min(),xmax=index.max(),color=cs[i],linestyles='dashed',linewidths=2)
        #---
        binned_data = np.append(binned_data,data)
    #---
    binned_data = binned_data.reshape((-1,len(streams[ensemble])),order='F')
    bins = np.linspace(start = binned_data.min()-binned_data.std(), stop = binned_data.max()+binned_data.std(), num = int(binned_data.shape[0]/10))
    ax1.hist(binned_data,bins=bins,color=cs[0:len(streams[ensemble])],orientation='horizontal',histtype='bar',align='left', stacked=True, fill=True,alpha=0.7)
    # Remove the inner label numbers of the histograms
    nullfmt = mpl.ticker.NullFormatter()
    ax1.yaxis.set_major_formatter(nullfmt)
    # Remove the inner ticks on the y axis
    nulllct= mpl.ticker.NullLocator()
    ax1.yaxis.set_major_locator(nulllct)
    ax1.set_ylim(bins.min(),bins.max())
    #---
    ax0.set_ylim(bins.min(),bins.max())
    ax0.set_xlabel('HMC trajectory')
    ax0.text(0.05,0.1,labels[ensemble],transform=ax0.transAxes,bbox=dict(boxstyle="round",facecolor='None'),verticalalignment='center')
    ax0.legend(loc=2,ncol=4)
    plt.tight_layout()
    figname = os.path.join('figures',ensemble+'_'+column+'.pdf')
    print('Saving to {}'.format(figname))
    plt.savefig(figname,transparent=True)

######
def parsing_args():
    parser = argparse.ArgumentParser(description='Extract HMC observable for an ensemble')
    parser.add_argument('--ens', type=str, default='a06m310L_s', help='pick an ensemble [%(default)s]')
    parser.add_argument('--autocorr', default=False, action='store_const', const=True, help='Compute autocorr times? [%(default)s]')
    parser.add_argument('--plots', default=False, action='store_const', const=True, help='Plot HMC histories? [%(default)s]')
    ### Setup
    args = parser.parse_args()
    ensemble = args.ens
    acorr = args.autocorr
    plot = args.plots
    return ensemble, acorr, plot

def main():
    ensemble, acorr, plot  = parsing_args()
    filename_obs = 'hmc_obs_'+ensemble+'.h5'
    print('Getting data from {}'.format(filename_obs))
    if (not os.path.isfile(filename_obs)):
        print('File {} not found'.format(filename_obs))
        return
    print('Extract HMC observable for {}'.format(ensemble))
    prex = labels[ensemble]
    print('This ensemble has {} streams: {}'.format(len(streams[ensemble]),streams[ensemble]))

    ### treat a15 separately because of pbp measurements
    if ensemble  == 'a15m135XL_s':
        obs = obs_a15
        index = index_a15
    else:
        obs = obs_all
        index = index_all

    ### Get data
    dataset_mdt = dict()
    dataset_trj = dict()
    dset_mdt = dict()
    dset_trj = dict()
    ## Open file
    f = h5.open_file(filename_obs)
    for i,s in enumerate(streams[ensemble]):
        node_name = "/{}_{}/{}".format(prex,s,index[0])
        index_mdt = f.get_node(node_name).read()
        node_name = "/{}_{}/{}".format(prex,s,index[1])
        index_trj = f.get_node(node_name).read()
        for o in obs:
            node_name = "/{}_{}/{}".format(prex,s,o)
            data = f.get_node(node_name).read()
            if o in ['pbp_c','pbp_s','pbp_l']:
                data = np.reshape(data,(-1,pbp_reps*2))
                dataset_mdt[o] = data.mean(axis=1)
            if o in ['deltaS']:
                dataset_mdt[o] = data
            if o in ['acc_rej']:
                dataset_trj[o] = data[:,1]
            if o in ['plaq']:
                dataset_trj[o] = data.mean(axis=1)/3.
        dset_mdt[s] = pd.DataFrame(data=dataset_mdt,index=index_mdt)
        dset_trj[s] = pd.DataFrame(data=dataset_trj,index=index_trj)
        print("read stream {}".format(i))
    # Close file
    f.close()

    # statistics
    for s in streams[ensemble]:
        print(dset_mdt[s].describe().T)
    for s in streams[ensemble]:
        print(dset_trj[s].describe().T)


    results_pbp = pd.concat(dset_mdt, keys=streams[ensemble])
    #### plotting
    if plot:
        for o in ['pbp_c','pbp_s','pbp_l']:
            plot_pbp(ensemble,results_pbp,o)

    ### Autocorrelations
    if acorr:
        ### Raw analysis of autocorrelations of pbp
        print('PBP autocorrelation times (lower bound) in trajectory length units')
        for s in streams[ensemble]:
            for pbp in ['pbp_l','pbp_s','pbp_c']:
                series = dset_mdt[s][pbp].values
                tau = autocorr.integrated_time(series,c=5,tol=50,quiet=True)
                print('{}/{}: {}'.format(s,pbp,tau))
        ### Raw analysis of autocorrelations of plaquette
        print('Plaquette autocorrelation times (lower bound) in units of saved configurations')
        for s in streams[ensemble]:
            series = dset_trj[s]['plaq'].values
            tau = autocorr.integrated_time(series,c=5,tol=50,quiet=True)
            print('{}/{}: {}'.format(s,'plaq',tau))

    ### Save data into txt files for unew analysis
    newdir = os.path.join('unew_files',labels[ensemble]+'_pbp')
    os.makedirs(newdir,exist_ok=True)
    filename = os.path.join(newdir,'runs_')
    for i,s in enumerate(streams[ensemble]):
        dset_mdt[s].to_csv(filename+str(i)+'.dat',sep=' ',header=False,index=False,columns=['pbp_l','pbp_s','pbp_c'],mode='w')


######

if __name__ == '__main__':
    main()
