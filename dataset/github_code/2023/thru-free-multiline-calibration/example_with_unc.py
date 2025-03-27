"""
@author: Ziad (zi.hatab@gmail.com)

Demonstration of multiline calibration with thru-free implementation with uncertainty propagation.
"""

import os
import zipfile
import copy

# pip install numpy matplotlib scikit-rf metas_unclib scipy -U
import skrf as rf
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

# my code
from uncMultiline import uncMultiline

import metas_unclib as munc
munc.use_linprop()

def get_cov_component(metas_val, para):
    # To get the uncertainty due to each parameter while accounting for their correlation 
    cov = []
    for inx in range(len(metas_val)):
        J = munc.get_jacobi2(metas_val[inx], para[inx])
        U = munc.get_covariance(para[inx])
        cov.append(J@U@J.T)
    return np.array(cov).squeeze()

def read_waves_to_S_from_zip(zipfile_full_dir, file_name_contain):
    # read wave parameter files and convert to S-parameters (from a zip file)
    with zipfile.ZipFile(zipfile_full_dir, mode="r") as archive:
        netwks = rf.read_zipped_touchstones(archive)
        A = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_A' in key])
        B = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_B' in key])    
    freq = A[0].frequency
    S = rf.NetworkSet( [rf.Network(s=b.s@np.linalg.inv(a.s), frequency=freq) for a,b in zip(A,B)] )
    return S.mean_s, S.cov(), np.array([s.s for s in S])

class PlotSettings:
    # to make plots look better for publication
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-v0_8-paper')
        # make svg output text and not curves
        plt.rcParams['svg.fonttype'] = 'none'
        # fontsize of the axes title
        plt.rc('axes', titlesize=self.font_size*1.2)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=self.font_size)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        # legend fontsize
        plt.rc('legend', fontsize=self.font_size*1)
        # fontsize of the figure title
        plt.rc('figure', titlesize=self.font_size)
        # controls default text sizes
        plt.rc('text', usetex=self.latex)
        #plt.rc('font', size=self.font_size, family='serif', serif='Times New Roman')
        plt.rc('lines', linewidth=1.5)
    def __exit__(self, exception_type, exception_value, traceback):
        plt.style.use('default')


if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbcm  = lambda x: mag2db(np.exp(x.real*1e-2))  # losses dB/cm
    
    path = os.path.dirname(os.path.realpath(__file__)) + '\\Measurements\\'
    
    # MS line
    file_name = 'line_50_'
    print('Loading files... please wait!!!')
    # these data are already corrected with switch term effects
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path + f'{file_name}_0_0mm.zip', f'{file_name}_0_0mm')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path + f'{file_name}_0_5mm.zip', f'{file_name}_0_5mm')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path + f'{file_name}_1_0mm.zip', f'{file_name}_1_0mm')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path + f'{file_name}_1_5mm.zip', f'{file_name}_1_5mm')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path + f'{file_name}_2_0mm.zip', f'{file_name}_2_0mm')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path + f'{file_name}_3_0mm.zip', f'{file_name}_3_0mm')
    L7, L7_cov, L7S = read_waves_to_S_from_zip(path + f'{file_name}_5_0mm.zip', f'{file_name}_5_0mm')
    L8, L8_cov, L8S = read_waves_to_S_from_zip(path + f'{file_name}_6_5mm.zip', f'{file_name}_6_5mm')
    
    SHORT, SHORT_cov, SHORTS = read_waves_to_S_from_zip(path + 'short1__0_0mm.zip', 'short1__0_0mm')
    NSHORT_A, NSHORT_A_cov, NSHORT_AS = read_waves_to_S_from_zip(path + 'short_A__1_0mm.zip', 'short_A__1_0mm')
    NSHORT_B, NSHORT_B_cov, NSHORT_BS = read_waves_to_S_from_zip(path + 'short_B__1_0mm.zip', 'short_B__1_0mm')
    DUT, DUT_cov, DUTS = read_waves_to_S_from_zip(path + 'line_30__5_0mm.zip', 'line_30__5_0mm')
    NETW = L3
    freq = L1.frequency
    f = freq.f  # frequency axis
    
    lines = [L1, L2, L4, L5, L6, L7, L8]
    line_lengths = [0, 0.5e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 6.5e-3]
    ereff_est = 2.5-0.00001j
    reflect = SHORT
    reflect_est = -1
    reflect_offset = -line_lengths[0]/2
    
    l_std = 0e-6  # for the line
    ulengths  = l_std**2  # the uncMultiline code will automatically repeat it for all lines
    uSlines   = [L1_cov, L2_cov, L4_cov, L5_cov, L6_cov, L7_cov, L8_cov]
    uSreflect = SHORT_cov
    uSnetwork = L3_cov
    uSnetwork_reflect_A = NSHORT_A_cov[:,:2,:2]
    uSnetwork_reflect_B = NSHORT_B_cov[:,-2:,-2:]
    
    # mTRL with linear uncertainty evaluation
    cal = uncMultiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, ulengths=ulengths, uSlines=uSlines, uSreflect=uSreflect)
    cal.run_uncMultiline() # run mTRL with linear uncertainty propagation

    # thru-free using left side error box (port A)
    cal2 = uncMultiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW, 
               network_reflect_A=NSHORT_A.s11, ulengths=ulengths,
               uSlines=uSlines, uSreflect=uSreflect, uSnetwork=uSnetwork,
               uSnetwork_reflect_A=uSnetwork_reflect_A)
    cal2.run_uncMultiline() # run multiline with linear uncertainty propagation
    
    # thru-free using right side error box (port B)
    cal3 = uncMultiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW,
               network_reflect_B=NSHORT_B.s22, ulengths=ulengths,
               uSlines=uSlines, uSreflect=uSreflect, uSnetwork=uSnetwork,
               uSnetwork_reflect_B=uSnetwork_reflect_B)
    cal3.run_uncMultiline() # run multiline with linear uncertainty propagation
    
    # plot data and uncertainty
    dut_cal, dut_cal_metas  = cal.apply_cal(DUT)
    dut_cal2, dut_cal2_metas = cal2.apply_cal(DUT)
    dut_cal3, dut_cal3_metas = cal3.apply_cal(DUT)
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=3)
        ax = axs[0,0]
        val = mag2db(dut_cal_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal2_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal3_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (dB)')
        ax.set_ylim(-40,0)
        ax.set_yticks(np.arange(-40,0.1,10))
        
        ax = axs[0,1]
        val = mag2db(dut_cal_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal2_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal3_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (dB)')
        ax.set_ylim(-4,0)
        ax.set_yticks(np.arange(-4,0.1,1))
        
        ax = axs[1,0]
        val = munc.umath.angle(dut_cal_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal2_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal3_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))
        
        ax = axs[1,1]
        val = munc.umath.angle(dut_cal_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)        
        val = munc.umath.angle(dut_cal2_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal3_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.95), 
                   loc='lower center', ncol=2, borderaxespad=0, columnspacing=1)

    F = ss.savgol_filter(np.eye(len(f)), window_length=9, polyorder=2)  # F is the filtering matrix
    k = 2
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=3)
        ax = axs[0,0]
        with PlotSettings(8):
            # inset axes....
            axin = ax.inset_axes([0.15, 0.6, 0.3, 0.3])
        val = abs(dut_cal_metas[:,0,0])
        val2 = abs(dut_cal2_metas[:,0,0])
        val3 = abs(dut_cal3_metas[:,0,0])
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Slines_metas[:,0]))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='o', markevery=30, markersize=10,
                label='Thru (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='o', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='^', markevery=30, markersize=10,
                label='Network (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='^', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_reflect_A_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='<', markevery=30, markersize=10,
                label='Port-A network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='<', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val3, cal3.Snetwork_reflect_B_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='>', markevery=30, markersize=10,
                label='Port-B network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='>', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='X', markevery=30, markersize=10,
                label='Reflect (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='X', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='v', markevery=30, markersize=10,
                label='Reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='v', markevery=20, markersize=10, color=p[0].get_color())
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (mag)')
        ax.set_yticks(np.arange(0,0.081,0.02))
        ax.set_ylim(-0.002,0.08)        
        with PlotSettings(8):
            axin.set_xlim((120, 150))
            axin.set_xticks([120,130,140,150])
            axin.set_ylim((0, 0.002))
            axin.set_yticks([0,0.001,0.002])
            # axin.set_xticklabels('')
            # axin.set_yticklabels('')
            ax.indicate_inset_zoom(axin, edgecolor="black")
            
        ax = axs[0,1]
        with PlotSettings(8):
            # inset axes....
            axin = ax.inset_axes([0.15, 0.6, 0.3, 0.3])
        val = abs(dut_cal_metas[:,1,0])
        val2 = abs(dut_cal2_metas[:,1,0])
        val3 = abs(dut_cal3_metas[:,1,0])
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Slines_metas[:,0]))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='o', markevery=30, markersize=10,
                label='Thru (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='o', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='^', markevery=30, markersize=10,
                label='Network (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='^', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_reflect_A_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='<', markevery=30, markersize=10,
                label='Port-A network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='<', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val3, cal3.Snetwork_reflect_B_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='>', markevery=30, markersize=10,
                label='Port-B network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='>', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='X', markevery=30, markersize=10,
                label='Reflect (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='X', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='v', markevery=30, markersize=10,
                label='Reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='v', markevery=20, markersize=10, color=p[0].get_color())
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (mag)')
        ax.set_yticks(np.arange(0,0.081,0.02))
        ax.set_ylim(-0.002,0.08)
        with PlotSettings(8):
            axin.set_xlim((120, 150))
            axin.set_xticks([120,130,140,150])
            axin.set_ylim((0, 0.002))
            axin.set_yticks([0,0.001,0.002])
            # axin.set_xticklabels('')
            # axin.set_yticklabels('')
            ax.indicate_inset_zoom(axin, edgecolor="black")
        
        ax = axs[1,0]
        with PlotSettings(8):
            # inset axes....
            axin = ax.inset_axes([0.15, 0.6, 0.3, 0.3])
        val = munc.umath.angle(dut_cal_metas[:,0,0], deg=True)
        val2 = munc.umath.angle(dut_cal2_metas[:,0,0], deg=True)
        val3 = munc.umath.angle(dut_cal3_metas[:,0,0], deg=True)
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Slines_metas[:,0]))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='o', markevery=30, markersize=10,
                label='Thru (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='o', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='^', markevery=30, markersize=10,
                label='Network (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='^', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_reflect_A_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='<', markevery=30, markersize=10,
                label='Port-A network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='<', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val3, cal3.Snetwork_reflect_B_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='>', markevery=30, markersize=10,
                label='Port-B network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='>', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='X', markevery=30, markersize=10,
                label='Reflect (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='X', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='v', markevery=30, markersize=10,
                label='Reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='v', markevery=20, markersize=10, color=p[0].get_color())
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (deg)')
        ax.set_yticks(np.arange(0,16.1,4))
        ax.set_ylim(-0.4,16)
        with PlotSettings(8):
            axin.set_xlim((120, 150))
            axin.set_xticks([120,130,140,150])
            axin.set_ylim((0, 0.5))
            axin.set_yticks([0,0.25,0.5])
            # axin.set_xticklabels('')
            # axin.set_yticklabels('')
            ax.indicate_inset_zoom(axin, edgecolor="black")
            
        ax = axs[1,1]
        with PlotSettings(8):
            # inset axes....
            axin = ax.inset_axes([0.15, 0.6, 0.3, 0.3])
        val = munc.umath.angle(dut_cal_metas[:,1,0], deg=True)
        val2 = munc.umath.angle(dut_cal2_metas[:,1,0], deg=True)
        val3 = munc.umath.angle(dut_cal3_metas[:,1,0], deg=True)
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Slines_metas[:,0]))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='o', markevery=30, markersize=10,
                label='Thru (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='o', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='^', markevery=30, markersize=10,
                label='Network (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='^', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Snetwork_reflect_A_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='<', markevery=30, markersize=10,
                label='Port-A network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='<', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val3, cal3.Snetwork_reflect_B_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='>', markevery=30, markersize=10,
                label='Port-B network-reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='>', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val, cal.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='X', markevery=30, markersize=10,
                label='Reflect (mTRL)')
        axin.plot(f*1e-9, std*k, lw=2, marker='X', markevery=20, markersize=10, color=p[0].get_color())
        
        std = F.T@np.sqrt(get_cov_component(val2, cal2.Sreflect_metas))
        p = ax.plot(f*1e-9, std*k, lw=2, marker='v', markevery=30, markersize=10,
                label='Reflect (thru-free)')
        axin.plot(f*1e-9, std*k, lw=2, marker='v', markevery=20, markersize=10, color=p[0].get_color())
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (deg)')
        ax.set_yticks(np.arange(0,8.1,2))
        ax.set_ylim(-0.2,8)
        with PlotSettings(8):
            axin.set_xlim((120, 150))
            axin.set_xticks([120,130,140,150])
            axin.set_ylim((0, 0.5))
            axin.set_yticks([0,0.25,0.5])
            # axin.set_xticklabels('')
            # axin.set_yticklabels('')
            ax.indicate_inset_zoom(axin, edgecolor="black")
        with PlotSettings(14):
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.95), 
                       loc='lower center', ncol=3, borderaxespad=0, columnspacing=1)
    
    lines = [L8, L1, L2, L4, L5, L6, L7]
    line_lengths = [6.5e-3, 0, 0.5e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]
    ereff_est = 2.5-0.00001j
    reflect = SHORT
    reflect_est = -1
    reflect_offset = -line_lengths[0]/2
    
    l_std = 0e-6  # for the line
    ulengths  = l_std**2  # the umTRL code will automatically repeat it for all lines
    uSlines   = [L8_cov, L1_cov, L2_cov, L4_cov, L5_cov, L6_cov, L7_cov]
    uSreflect = SHORT_cov
    uSnetwork = L3_cov
    uSnetwork_reflect_A = NSHORT_A_cov[:,:2,:2]
    uSnetwork_reflect_B = NSHORT_B_cov[:,-2:,-2:]
    
    # mTRL with linear uncertainty evaluation
    cal = uncMultiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, ulengths=ulengths, uSlines=uSlines, uSreflect=uSreflect)
    cal.run_uncMultiline() # run mTRL with linear uncertainty propagation
    cal2 = copy.copy(cal)
    cal2.shift_plane(-line_lengths[0]/2)
        
    # thru-free using left side error box (port A)
    cal3 = uncMultiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW, 
               network_reflect_A=NSHORT_A.s11, ulengths=ulengths,
               uSlines=uSlines, uSreflect=uSreflect, uSnetwork=uSnetwork,
               uSnetwork_reflect_A=uSnetwork_reflect_A)
    cal3.run_uncMultiline() # run mTRL with linear uncertainty propagation
    
    # thru-free using right side error box (port B)
    cal4 = uncMultiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW,
               network_reflect_B=NSHORT_B.s22, ulengths=ulengths,
               uSlines=uSlines, uSreflect=uSreflect, uSnetwork=uSnetwork,
               uSnetwork_reflect_B=uSnetwork_reflect_B)
    cal4.run_uncMultiline() # run mTRL with linear uncertainty propagation
    
    # plot data and uncertainty
    dut_cal, dut_cal_metas   = cal.apply_cal(DUT)
    dut_cal2, dut_cal2_metas = cal2.apply_cal(DUT)
    dut_cal3, dut_cal3_metas = cal3.apply_cal(DUT)
    dut_cal4, dut_cal4_metas = cal4.apply_cal(DUT)
    
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=3)
        ax = axs[0,0]
        val = mag2db(dut_cal_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal2_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal3_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal4_metas[:,0,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (dB)')
        ax.set_ylim(-40,0)
        ax.set_yticks(np.arange(-40,0.1,10))
        
        ax = axs[0,1]
        val = mag2db(dut_cal_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal2_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal3_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = mag2db(dut_cal4_metas[:,1,0])
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (dB)')
        ax.set_ylim(-4,0)
        ax.set_yticks(np.arange(-4,0.1,1))
        
        ax = axs[1,0]
        val = munc.umath.angle(dut_cal_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal2_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal3_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal4_metas[:,0,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))
        
        ax = axs[1,1]
        val = munc.umath.angle(dut_cal_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)        
        val = munc.umath.angle(dut_cal2_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal3_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        val = munc.umath.angle(dut_cal4_metas[:,1,0], deg=True)
        mu  = munc.get_value(val)
        std = munc.get_stdunc(val)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.95), 
                   loc='lower center', ncol=2, borderaxespad=0, columnspacing=1)
    
    plt.show()
    
    # EOF