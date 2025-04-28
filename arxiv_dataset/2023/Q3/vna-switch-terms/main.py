"""
Author: Ziad (https://github.com/ZiadHatab)

Example on how to extract the switch terms of a VNA using 3-receiver VNA measurements.
"""

import os

# need to be installed via pip
import skrf as rf 
import numpy as np
import matplotlib.pyplot as plt

# my script (MultiCal.py and TUGmTRL must also be in same folder)
from mTRL import mTRL

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

def compute_switch_terms(S):
    """
    S: is a list of skrf networks
    """
    Gamma21 = []  # forward switch term
    Gamma12 = []  # reverse switch term 
    for inx in range(len(S[0].frequency.f)): # iterate through all frequency points
        # create the system matrix
        H = np.array([ [-s.s[inx,0,0]*s.s[inx,0,1]/s.s[inx,1,0], -s.s[inx,1,1], 1, s.s[inx,0,1]/s.s[inx,1,0]] for s in S ])
        _,_,vh = np.linalg.svd(H)    # compute the SVD
        nullspace = vh[-1,:].conj()  # get the nullspace        
        Gamma21.append(nullspace[1]/nullspace[2])  # that is all
        Gamma12.append(nullspace[0]/nullspace[3])
    
    return np.array(Gamma21), np.array(Gamma12)

# main script
if __name__ == '__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(complex).eps))  # eps to ensure positive square-root
    gamma2dbcm  = lambda x: mag2db(np.exp(x.real*1e-2))  # losses dB/cm
    
    # load the measurements
    # files' path are reference to script's path
    path = os.path.dirname(os.path.realpath(__file__)) + '\\'
    s2p_path = path + 'Measurements\\'
    
    # switch terms directly measured from VNA (using the 4th receiver)
    Gamma_21_direct = rf.Network(s2p_path + 'Gamma_21.s1p')
    Gamma_12_direct = rf.Network(s2p_path + 'Gamma_12.s1p')
    
    # Calibration standards
    L1    = rf.Network(s2p_path + 'line_0_0mm.s2p')
    L2    = rf.Network(s2p_path + 'line_2_5mm.s2p')
    L3    = rf.Network(s2p_path + 'line_10_0mm.s2p')
    L4    = rf.Network(s2p_path + 'line_15_0mm.s2p')
    L5    = rf.Network(s2p_path + 'line_50_0mm.s2p')
    SHORT = rf.Network(s2p_path + 'short_0_0mm.s2p')
    
    DUT = rf.Network(s2p_path + 'step_line.s2p')
    f = DUT.frequency.f
    
    # switch terms indirectly measured
    stand1 = rf.Network(s2p_path + 'shunt_series.s2p')
    stand2 = rf.Network(s2p_path + 'series_shunt.s2p')
    myG21, myG12 = compute_switch_terms([stand1, stand2, L5])
    
    Gamma_21_indirect = rf.Network(s=myG21, frequency=L1.frequency)
    Gamma_12_indirect = rf.Network(s=myG12, frequency=L1.frequency)
    
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2.5)
        ax = axs[0,0]
        val = mag2db(Gamma_21_direct.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2, marker='o', markevery=20, markersize=12,
                label='Direct measurement', linestyle='-')
        val = mag2db(Gamma_21_indirect.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2, marker='X', markevery=20, markersize=10,
                label='Indirect measurement', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel(r'$\Gamma_{21}$ (dB)')
        ax.set_yticks(np.arange(-40,0.1,10))
        ax.set_ylim(-40,0)
        ax.legend(loc='upper left', ncol=1, fontsize=12)
        
        ax = axs[0,1]
        val = mag2db(Gamma_12_direct.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2, marker='o', markevery=20, markersize=12,
                label='Direct measurement', linestyle='-')
        val = mag2db(Gamma_12_indirect.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2, marker='X', markevery=20, markersize=10,
                label='Indirect measurement', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel(r'$\Gamma_{12}$ (dB)')
        ax.set_yticks(np.arange(-40,0.1,10))
        ax.set_ylim(-40,0)
        
        ax = axs[1,0]
        val = np.angle(Gamma_21_direct.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2, marker='o', markevery=20, markersize=12,
                label='Direct measurement', linestyle='-')
        val = np.angle(Gamma_21_indirect.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2, marker='X', markevery=20, markersize=10,
                label='Indirect measurement', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel(r'$\Gamma_{21}$ (phase $\times \pi$ rad)')
        ax.set_ylim(-1,1)
        ax.set_yticks(np.arange(-1,1.1,0.4))
        
        ax = axs[1,1]
        val = np.angle(Gamma_12_direct.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2, marker='o', markevery=20, markersize=12,
                label='Direct measurement', linestyle='-')
        val = np.angle(Gamma_12_indirect.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2, marker='X', markevery=20, markersize=10,
                label='Indirect measurement', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel(r'$\Gamma_{12}$ (phase $\times \pi$ rad)')
        ax.set_ylim(-1,1)
        ax.set_yticks(np.arange(-1,1.1,0.4))

    ## the calibration    
    lines = [L1, L2, L3, L4, L5]
    line_lengths = [0e-3, 2.5e-3, 10e-3, 15e-3, 50e-3]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [0]

    cal_no_switch = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=3.5+0j,
               switch_term=None
               )
    cal_no_switch.run_tug()
    dut_cal_no_switch = cal_no_switch.apply_cal(DUT)
    
    cal_direct_switch = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=3.5+0j,
               switch_term=[Gamma_21_direct, Gamma_12_direct]
               )
    cal_direct_switch.run_tug()
    dut_cal_direct_switch = cal_direct_switch.apply_cal(DUT)
    
    cal_indirect_switch = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=3.5+0j,
               switch_term=[Gamma_21_indirect, Gamma_12_indirect]
               )
    cal_indirect_switch.run_tug()
    dut_cal_indirect_switch = cal_indirect_switch.apply_cal(DUT)
    
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2.5)
        ax = axs[0,0]
        val = mag2db(dut_cal_no_switch.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2, markevery=20, markersize=10,
                label='Switch terms not considered', linestyle='-', color='black')
        val = mag2db(dut_cal_direct_switch.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=20, markersize=12,
                label='Directly measured swicth terms', linestyle='-')
        val = mag2db(dut_cal_indirect_switch.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=20, markersize=10,
                label='Indirectly measured swicth terms', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel('S11 (dB)')
        ax.set_yticks(np.arange(-40,0.1,10))
        ax.set_ylim(-40,0)
        
        ax = axs[0,1]
        val = mag2db(dut_cal_no_switch.s[:,1,0])
        ax.plot(f*1e-9, val, lw=2, markevery=20, markersize=10,
                label='Switch terms not considered', linestyle='-', color='black')
        val = mag2db(dut_cal_direct_switch.s[:,1,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=20, markersize=12,
                label='Directly measured swicth terms', linestyle='-')
        val = mag2db(dut_cal_indirect_switch.s[:,1,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=20, markersize=10,
                label='Indirectly measured swicth terms', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel('S21 (dB)')
        ax.set_yticks(np.arange(-4,0.1,1))
        ax.set_ylim(-4,0)
        ax.legend(loc='lower right', ncol=1, fontsize=12)
        
        ax = axs[1,0]
        val = np.angle(dut_cal_no_switch.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2, markevery=20, markersize=10,
                label='Switch terms not considered', linestyle='-', color='black')
        val = np.angle(dut_cal_direct_switch.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=20, markersize=12,
                label='Directly measured swicth terms', linestyle='-')
        val = np.angle(dut_cal_indirect_switch.s[:,0,0])/np.pi
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=20, markersize=10,
                label='Indirectly measured swicth terms', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel(r'S11 (phase $\times \pi$ rad)')
        ax.set_ylim(-1,1)
        ax.set_yticks(np.arange(-1,1.1,0.4))
        
        ax = axs[1,1]
        val = np.angle(dut_cal_no_switch.s[:,1,0])/np.pi
        ax.plot(f*1e-9, val, lw=2, markevery=20, markersize=10,
                label='Switch terms not considered', linestyle='-', color='black')
        val = np.angle(dut_cal_direct_switch.s[:,1,0])/np.pi
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=20, markersize=12,
                label='Directly measured swicth terms', linestyle='-')
        val = np.angle(dut_cal_indirect_switch.s[:,1,0])/np.pi
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=20, markersize=10,
                label='Indirectly measured swicth terms', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,20.1,2))
        ax.set_xlim(0,14)
        ax.set_ylabel(r'S21 (phase $\times \pi$ rad)')
        ax.set_ylim(-1,1)
        ax.set_yticks(np.arange(-1,1.1,0.4))
        
        
    plt.show()
    
# EOF