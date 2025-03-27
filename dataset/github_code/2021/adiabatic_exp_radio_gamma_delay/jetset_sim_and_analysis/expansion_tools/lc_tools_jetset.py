from astropy.table import Table
import os
import glob
from astropy.units import Unit as u
import itertools
import seaborn as sns
from matplotlib import  pyplot as plt
import numpy as np
from scipy import signal
from jetset.model_parameters import ModelParameterArray

from jetset.model_parameters import ModelParameterArray
from jetset.base_model import Model
from jetset.analytical_model import AnalyticalParameter
from jetset.model_manager import  FitModel
from jetset.minimizer import fit_SED,ModelMinimizer,fit_XY
from jetset.data_loader import Data
from jetset.base_model import Model
import copy
from scipy import interpolate
from itertools import cycle

#def make_lc(temp_ev, nu1,nu2, Tstat=0,Tstop=-1,N=1000,eval_cross_time=True,use_cached=True,rest_frame='obs',name=None,R=None):
#    t,f=temp_ev.make_lc(Tstat,Tstop,N,nu1,nu2,rest_frame=rest_frame,eval_cross_time=eval_cross_time,use_cached=use_cached,R=R)
#    return Table([t*u('s'),f*u('erg s-1 cm-2')],names=('time','flux'),meta={'name': name})

def combine_lc(lc_list,delta_t_array,N,t_start=0,user_t_stop=None,name=None):
    
    lc_list=[copy.deepcopy(lc) for lc in lc_list]
    t_stop=0
    print(len(lc_list))
    for ID,lc in enumerate(lc_list):
        print(ID)
        if ID>0:
            t_stop= t_stop+lc['time'].max()+delta_t_array[ID-1]
            #print(ID,t_stop,lc['time'].max())
            lc['time']+=delta_t_array[ID-1]
        else:
            t_stop+=lc['time'].max()
        print(ID,t_stop,lc['time'].max())


    #print(t_stop)
    if user_t_stop is not None:
        t_stop=user_t_stop
    time_array=np.linspace(0,t_stop,np.int(N))
    flux_array=np.zeros(time_array.shape)
    
    
    for lc in lc_list:  
        flux_array+=np.interp(time_array, lc['time'], lc['flux'],left=0, right=0)
    
 
    return Table([time_array*u('s'),flux_array*u('erg s-1 cm-2')],names=('time','flux'),meta={'name': name})



def forward(x):
    return x/x.max()

def inverse(x):
    return x/x.max()



def sdcf(ts1, ts2, t, dt):

    '''
        Subroutine - sdcf
          DCF algorithm with slot weighting
    '''

    dcf = np.zeros(t.shape[0])
    dcferr = np.zeros(t.shape[0])
    n = np.zeros(t.shape[0])

    dst = np.empty((ts1.shape[0], ts2.shape[0]))
    for i in range(ts1.shape[0]):
        for j in range(ts2.shape[0]):
            dst[i,j] = ts2[j,0] - ts1[i,0]

    for k in range(t.shape[0]):
        tlo = t[k] - dt/2.0
        thi = t[k] + dt/2.0
        ts1idx, ts2idx = np.where((dst < thi) & (dst > tlo))

        mts2 = np.mean(ts2[ts2idx,1])
        mts1 = np.mean(ts1[ts1idx,1])
        n[k] = ts1idx.shape[0]

        dcfdnm = np.sqrt((np.var(ts1[ts1idx,1]) - np.mean(ts1[ts1idx,2])**2) \
                         * (np.var(ts2[ts2idx,1]) - np.mean(ts2[ts2idx,2])**2))

        dcfs = (ts2[ts2idx,1] - mts2) * (ts1[ts1idx,1] - mts1) / dcfdnm

        dcf[k] = np.sum(dcfs) / float(n[k])
        dcferr[k] = np.sqrt(np.sum((dcfs - dcf[k])**2)) / float(n[k] - 1)

    return dcf, dcferr

def find_gaps(t,gap_lim,delat_t):
    dt=np.diff(t)
    t=t[:-1]
    dt_int=interpolate.interp1d(t, dt>gap_lim, kind='previous',fill_value=0,bounds_error=False)
    t_int=np.arange(t.min(),t.max(),delat_t)
    gaps_int=dt_int(t_int)
    return t_int,gaps_int>=1




def plot_2lcs(lc_1,lc_2,average=7,low=None,high=None,dt_cdf=None,t_min=None,t_max=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    tab_list=[lc_1,lc_2]
    
    y_l=[None]*2
    y_l_err=[None]*2
    t_l=[None]*2
    y_label=[None]*2
    labels=[None]*2
    
    fig = plt.figure()
    fig_h=6
    if dt_cdf is not None:
        fig_w=22
        gs_w=13
        d_w=3
    else:
        fig_w=15
        gs_w=7
        d_w=0
    fig.set_size_inches(fig_w,fig_h)


    gs = fig.add_gridspec(2, gs_w)
    ax1 = fig.add_subplot(gs[0,0:gs_w-d_w])
    ax2 = fig.add_subplot(gs[1,0:gs_w-d_w])
    if dt_cdf is not None:
        ax3 = fig.add_subplot(gs[:,gs_w:gs_w+d_w])
        ax4 = fig.add_subplot(gs[:,gs_w+d_w:gs_w+2*d_w])
    plt.tight_layout()

    
    for ID,f in enumerate(tab_list):
        if t_min is not None:
            tab_list[ID]=tab_list[ID][tab_list[ID]['time'].to('d')>=t_min]
        if t_max is not None:
            tab_list[ID]=tab_list[ID][tab_list[ID]['time'].to('d')<=t_max]
        labels[ID]=tab_list[ID].meta['name']

        
        t_l[ID]=tab_list[ID]['time'].to('d').value
        y_l[ID]=tab_list[ID]['flux']
        if 'e_flux' in tab_list[ID].colnames:
            y_l_err[ID]=tab_list[ID]['e_flux']
        else:
            y_l_err[ID]=np.zeros(len(tab_list[ID]))
        
        y_label[ID]=tab_list[ID]['flux'].unit
        
    
                
    ax1.errorbar(x=t_l[0],y=y_l[0],yerr=0,label=labels[0],marker='o',ms=1,lw=0.3,linestyle='dashed',c='gray')
    ax2.errorbar(x=t_l[1],y=y_l[1],yerr=0,label=labels[1],marker='o',ms=1,lw=0.3,linestyle='dashed',c='gray')
    
    
    t_1=np.min([t_l[0].min(),t_l[1].min()])
    t_2=np.max([t_l[0].max(),t_l[1].max()])
    tr=np.arange(t_1,t_2,average)
    t=(tr[:-1]+tr[1:])*0.5
    y1=np.zeros(t.size)
    y2=np.zeros(t.size)
    err_y1=np.zeros(t.size)
    err_y2=np.zeros(t.size)
    for ID in range(tr.size-1):
        t_start=tr[ID]
        t_stop=tr[ID+1]
        m=np.logical_and(t_l[0]>=t_start , t_l[0]<=t_stop)
        y1[ID]=np.mean(y_l[0][m])
        err_y1[ID]=np.sqrt(np.sum(y_l_err[0][m]**2)/m.sum())
        m=np.logical_and(t_l[1]>=t_start , t_l[1]<=t_stop)
        y2[ID]=np.mean(y_l[1][m])
        err_y2[ID]=np.sqrt(np.sum(y_l_err[1][m]**2)/m.sum())
   
    msk=np.logical_and(y1>0,y2>0)
   
    ax1.errorbar(x=t,y=y1,yerr=err_y1,label=labels[0],xerr=average*0.5,marker='o',linestyle='dashed',c='b')
    ax1.legend()
    ax1.set_ylabel(y_label[0])
    
    #ax2.plot(t,y2,label=labels[1],c='g')
    ax2.errorbar(x=t,y=y2,yerr=err_y2,label=labels[1],xerr=average*0.5,marker='o',linestyle='dashed',c='r')
    
    ax2.set_ylabel(y_label[1])
    ax2.set_xlabel('T days')
    ax2.legend()
    
    
    if dt_cdf is not None:
        
        
        ax3.set_xlabel(labels[0])
        ax3.set_ylabel(labels[1])
        ax=sns.regplot(ax=ax3,y=y2, x=y1,scatter_kws={'s':2})
        ax3.errorbar(y1, y2, yerr=err_y2,xerr=err_y1, fmt='none', capsize=0, zorder=0, color='C0',alpha=1.0)
        ax=sns.kdeplot(ax=ax3,y=y2, x=y1)
        fig.axes.append(ax)



        if low is None:
            low = -(t.max()-t.min())*0.5
        
        if high is None:
            high = (    t.max()-t.min())*0.5

        
        n = np.int((high - low) / float(dt_cdf))
        t_dcf = np.linspace(low+(dt_cdf/2.0),  high-(dt_cdf/2.0), n)
        
        #print('x',x,lags)
        #plt.show()
        #t1=np.column_stack((tab_list[0]['time'],tab_list[0]['flux'],tab_list[0]['e_flux']))
        #t2=np.column_stack((tab_list[1]['time'],tab_list[1]['flux'],tab_list[1]['e_flux']))
        t1=np.column_stack((t,y1,err_y1))
        t2=np.column_stack((t,y2,err_y2))
        #print(t1.shape)

        dcf, dcferr = sdcf(t1,t2, t_dcf, dt_cdf)
        msk=~np.isnan(dcf)
        ax4.errorbar(t_dcf[msk], dcf[msk], dcferr[msk], color='k', ls='-', capsize=0)
        print(t_dcf[np.argmax(dcf[msk])])
        ax4.axvline(t_dcf[[msk]][np.argmax(dcf[msk])])
        plt.tight_layout()






def scatter_plot_lcs(name_2,name_1,average=7,gap_lim=None,gap_sampling=1):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    fl=glob.glob('*.ecsv')
    
    name_list=[name_1,name_2]
    tab_list=[None]*2
    y_l=[None]*2
    t_l=[None]*2
    y_label=[None]*2
    labels=[None]*2
    #if name_list is not None:
    fl=[f for f in  fl for n in name_list if n in f]
    print(fl)
    fig = plt.figure()
    fig.set_size_inches(20,6)
    gs = fig.add_gridspec(2, 10)
    ax1 = fig.add_subplot(gs[0,0:7])
    ax2 = fig.add_subplot(gs[1,0:7])
    
    ax3 = fig.add_subplot(gs[:,7:10])
    plt.tight_layout()

    
    for ID,f in enumerate(fl):
        labels[ID]=f.replace('.ecsv','')

        tab_list[ID]=Table.read(f)
        t_l[ID]=tab_list[ID]['time'].to('d')
        y_l[ID]=tab_list[ID]['flux']
        for ID1,name in enumerate(tab_list[ID].meta['comments'][2].split(',')):
            #print('name',name)
            if name=='flux':
                y_label[ID]=tab_list[ID].meta['comments'][1].split(',')[ID1]
                
    ax1.errorbar(x=t_l[0],y=y_l[0],yerr=0,label=labels[0],marker='o',ms=1,lw=0.3,linestyle='dashed',c='gray')
    ax2.errorbar(x=t_l[1],y=y_l[1],yerr=0,label=labels[1],marker='o',ms=1,lw=0.3,linestyle='dashed',c='gray')
    
    if gap_lim is not None:
        f1 = interpolate.interp1d(t_l[0], y_l[0], kind='slinear',fill_value=0,bounds_error=False)
        t1,gps_msk1=find_gaps(t_l[0],gap_lim,gap_sampling)
        f2 = interpolate.interp1d(t_l[1], y_l[1], kind='slinear',fill_value=0,bounds_error=False)
        t2,gps_msk2=find_gaps(t_l[1],gap_lim,gap_sampling)
        y1 = f1(t1)
        y2 = f2(t2)
        y1[gps_msk1]=0
        y2[gps_msk2]=0
        x1=t1
        x2=t2
    else:
        x1=t_l[0]
        x2=t_l[1]
        y1=y_l[0]
        y2=y_l[1]
    
    t_1=np.min([x1.min(),x2.min()])
    t_2=np.max([x1.max(),x2.max()])
    t=np.arange(t_1,t_2,average)

    f1 = interpolate.interp1d(x1, y1, kind='quadratic',fill_value=0,bounds_error=False)
    f2 = interpolate.interp1d(x2, y2, kind='quadratic',fill_value=0,bounds_error=False)
   
        
    y1 = f1(t)
    y2 = f2(t)
    msk=np.logical_and(y1>0,y2>0)
    msk=np.logical_and(msk,y1>y_l[0].min())
    msk=np.logical_and(msk,y2>y_l[1].min())
    #ax1.plot(t,y1,label=labels[0],c='r')
    ax1.errorbar(x=t,y=y1,yerr=0,label=labels[0],marker='+',ms=1,lw=0.3,linestyle='dashed',c='b')
    ax1.legend()
    ax1.set_ylabel(y_label[0])
    
    #ax2.plot(t,y2,label=labels[1],c='g')
    
    ax2.errorbar(x=t,y=y2,yerr=0,label=labels[1],marker='+',ms=1,lw=0.3,linestyle='dashed',c='r')
    
    ax2.set_ylabel(y_label[1])
    ax2.set_xlabel('T [mjd]')
    ax2.legend()
    
    ax3.plot(y1[msk],y2[msk],'o')
    ax3.set_xlabel(labels[0])
    ax3.set_ylabel(labels[1])
    plt.tight_layout()
        
