"""
@author: Ziad (zi.hatab@gmail.com)
example on how to measure the propagation constant from multiple shifted networks.
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import skrf as rf

# my code
from multinetwork import mN

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
    c0 = 299792458
    # useful functions
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbcm  = lambda x: mag2db(np.exp(x.real*1e-2))  # losses dB/cm
    
    kappa_est = -1 + 0j
    ereff_est = 1 - 0.000001j
    
    # ENA
    path = os.path.dirname(os.path.realpath(__file__)) + '\\measurements\\ENA\\'
    file_name = 'line'
    print('Loading files... please wait!!!')
    L1  = rf.Network(path + f'{file_name}_000mm.s2p')
    L2  = rf.Network(path + f'{file_name}_021mm.s2p')
    L3  = rf.Network(path + f'{file_name}_066mm.s2p')
    L4  = rf.Network(path + f'{file_name}_081mm.s2p')
    L5  = rf.Network(path + f'{file_name}_084mm.s2p')
    L6  = rf.Network(path + f'{file_name}_093mm.s2p')
    L7  = rf.Network(path + f'{file_name}_117mm.s2p')
    L8  = rf.Network(path + f'{file_name}_123mm.s2p')
    L9  = rf.Network(path + f'{file_name}_171mm.s2p')
    L10 = rf.Network(path + f'{file_name}_192mm.s2p')
    lengths = np.array([0, 7, 22, 27, 28, 31, 39, 41, 57, 64])*3e-3
    lines = [L1,L2,L3,L4,L5,L6,L7,L8,L9,L10]
    lines = [x['3ghz-18ghz'] for x in lines]
    f_ENA = lines[0].frequency.f  # frequency axis
    cal_ENA = mN(lines=lines, line_lengths=lengths, ereff_est=ereff_est, kappa_est=kappa_est)
    cal_ENA.run_mN()
    
    # ZNA
    path = os.path.dirname(os.path.realpath(__file__)) + '\\measurements\\ZNA\\'
    file_name = 'line'
    print('Loading files... please wait!!!')
    L1  = rf.Network(path + f'{file_name}_000mm.s2p')
    L2  = rf.Network(path + f'{file_name}_021mm.s2p')
    L3  = rf.Network(path + f'{file_name}_066mm.s2p')
    L4  = rf.Network(path + f'{file_name}_081mm.s2p')
    L5  = rf.Network(path + f'{file_name}_084mm.s2p')
    L6  = rf.Network(path + f'{file_name}_093mm.s2p')
    L7  = rf.Network(path + f'{file_name}_117mm.s2p')
    L8  = rf.Network(path + f'{file_name}_123mm.s2p')
    L9  = rf.Network(path + f'{file_name}_171mm.s2p')
    L10 = rf.Network(path + f'{file_name}_192mm.s2p')
    lengths = np.array([0, 7, 22, 27, 28, 31, 39, 41, 57, 64])*3e-3
    lines = [L1,L2,L3,L4,L5,L6,L7,L8,L9,L10]
    lines = [x['3ghz-18ghz'] for x in lines]
    f_ZNA = lines[0].frequency.f  # frequency axis
    cal_ZNA = mN(lines=lines, line_lengths=lengths, ereff_est=ereff_est, kappa_est=kappa_est)
    cal_ZNA.run_mN()
    
    # VectorStar
    path = os.path.dirname(os.path.realpath(__file__)) + '\\measurements\\VectorStar\\'
    file_name = 'line'
    print('Loading files... please wait!!!')
    L1  = rf.Network(path + f'{file_name}_000mm.s2p')
    L2  = rf.Network(path + f'{file_name}_021mm.s2p')
    L3  = rf.Network(path + f'{file_name}_066mm.s2p')
    L4  = rf.Network(path + f'{file_name}_081mm.s2p')
    L5  = rf.Network(path + f'{file_name}_084mm.s2p')
    L6  = rf.Network(path + f'{file_name}_093mm.s2p')
    L7  = rf.Network(path + f'{file_name}_117mm.s2p')
    L8  = rf.Network(path + f'{file_name}_123mm.s2p')
    L9  = rf.Network(path + f'{file_name}_171mm.s2p')
    L10 = rf.Network(path + f'{file_name}_192mm.s2p')
    lengths = np.array([0, 7, 22, 27, 28, 31, 39, 41, 57, 64])*3e-3
    lines = [L1,L2,L3,L4,L5,L6,L7,L8,L9,L10]
    lines = [x['3ghz-18ghz'] for x in lines]
    f_vec = lines[0].frequency.f  # frequency axis
    cal_vec = mN(lines=lines, line_lengths=lengths, ereff_est=ereff_est, kappa_est=kappa_est)
    cal_vec.run_mN()

    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3.8))        
        # fig.set_dpi(600)
        fig.tight_layout(pad=3.3)
        ax = axs[0]
        ax.plot(f_ZNA*1e-9, cal_ZNA.ereff.real, lw=3, label='ZNA', 
                marker='v', markevery=15, markersize=14)
        ax.plot(f_vec*1e-9, cal_vec.ereff.real, lw=3, label='VectorStar', 
                marker='>', markevery=15, markersize=14)
        ax.plot(f_ENA*1e-9, cal_ENA.ereff.real, lw=3, label='ENA', 
                marker='<', markevery=15, markersize=14)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([1.002,1.012])
        ax.set_yticks(np.arange(1.002, 1.0121, 0.002))
        ax.set_xlim(3,18)
        ax.set_xticks(np.arange(3,18.1,3))

        ax = axs[1]
        ax.plot(f_ZNA*1e-9, gamma2dbcm(cal_ZNA.gamma), lw=3, label='ZNA', 
                marker='v', markevery=15, markersize=14)
        ax.plot(f_vec*1e-9, gamma2dbcm(cal_vec.gamma), lw=3, label='VectorStar', 
                marker='>', markevery=15, markersize=14)
        ax.plot(f_ENA*1e-9, gamma2dbcm(cal_ENA.gamma), lw=3, label='ENA', 
                marker='<', markevery=15, markersize=14)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/cm)')
        ax.set_ylim([0,0.01])
        ax.set_yticks(np.arange(0, 0.011, 0.002))
        ax.set_xlim(3,18)
        ax.set_xticks(np.arange(3,18.1,3))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.9), 
                   loc='lower center', ncol=4, borderaxespad=0)
        #fig.savefig(path + 'file_print.pdf', format='pdf', dpi=300, 
        #            bbox_inches='tight', pad_inches = 0)
          
    plt.show()
