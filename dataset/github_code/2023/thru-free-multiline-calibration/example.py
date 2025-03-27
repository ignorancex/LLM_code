"""
@author: Ziad (zi.hatab@gmail.com)

Demonstration of multiline calibration with thru-free implementation.
"""
import os
import zipfile
import copy

# pip install numpy matplotlib scikit-rf -U
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

# my code
from multiline import multiline

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
    
    SHORT, SHORT_cov, SHORTS = read_waves_to_S_from_zip(path + 'short2__0_0mm.zip', 'short2__0_0mm')
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
    
    # multiline calibration
    cal = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est)
    cal.run_multiline()

    # thru-free using left side error box (port A)
    cal2 = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW, network_reflect_A=NSHORT_A.s11)
    cal2.run_multiline()
    
    # thru-free using right side error box (port B)
    cal3 = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW, network_reflect_B=NSHORT_B.s22)
    cal3.run_multiline() 
    
    # plot data and uncertainty
    dut_cal, dut_cal_S   = cal.apply_cal(DUT)
    dut_cal2, dut_cal2_S = cal2.apply_cal(DUT)
    dut_cal3, dut_cal3_S = cal3.apply_cal(DUT)
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=3)
        ax = axs[0,0]
        mu = mag2db(dut_cal_S[:,0,0])
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        mu = mag2db(dut_cal2_S[:,0,0])
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        mu = mag2db(dut_cal3_S[:,0,0])
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (dB)')
        ax.set_ylim(-40,0)
        ax.set_yticks(np.arange(-40,0.1,10))
        
        ax = axs[0,1]
        mu = mag2db(dut_cal_S[:,1,0])
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        mu = mag2db(dut_cal2_S[:,1,0])
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        mu = mag2db(dut_cal3_S[:,1,0])
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (dB)')
        ax.set_ylim(-4,0)
        ax.set_yticks(np.arange(-4,0.1,1))
        
        ax = axs[1,0]
        mu = np.angle(dut_cal_S[:,0,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')
        mu = np.angle(dut_cal2_S[:,0,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        mu = np.angle(dut_cal3_S[:,0,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))
        
        ax = axs[1,1]
        mu = np.angle(dut_cal_S[:,1,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at thru center')        
        mu = np.angle(dut_cal2_S[:,1,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        mu = np.angle(dut_cal3_S[:,1,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='<', markevery=30, markersize=10,
                label='Thru-free (port-B network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.95), 
                   loc='lower center', ncol=3, borderaxespad=0, columnspacing=1)

    # multiline calibration using a line instead of a thru
    lines = [L8, L1, L2, L4, L5, L6, L7]
    line_lengths = [6.5e-3, 0, 0.5e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]
    ereff_est = 2.5-0.00001j
    reflect = SHORT
    reflect_est = -1
    reflect_offset = -line_lengths[0]/2
    
    # multiline calibration
    cal = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est)
    cal.run_multiline() # multiline calibration at line center
    cal2 = copy.copy(cal)
    cal2.shift_plane(-line_lengths[0]/2)  # multiline calibration shifted backwards by line length
        
    # thru-free using left side error box (port A)
    cal3 = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW, network_reflect_A=NSHORT_A.s11)
    cal3.run_multiline() 
    
    # thru-free using right side error box (port B)
    cal4 = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, network=NETW, network_reflect_B=NSHORT_B.s22)
    cal4.run_multiline() 
    
    # plot data and uncertainty
    dut_cal, dut_cal_S   = cal.apply_cal(DUT)
    dut_cal2, dut_cal2_S = cal2.apply_cal(DUT)
    dut_cal3, dut_cal3_S = cal3.apply_cal(DUT)
    dut_cal4, dut_cal4_S = cal4.apply_cal(DUT)
    
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=3)
        ax = axs[0,0]
        mu = mag2db(dut_cal_S[:,0,0])
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        mu = mag2db(dut_cal2_S[:,0,0])
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        mu = mag2db(dut_cal3_S[:,0,0])
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (dB)')
        ax.set_ylim(-40,0)
        ax.set_yticks(np.arange(-40,0.1,10))
        
        ax = axs[0,1]
        mu = mag2db(dut_cal_S[:,1,0])
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        mu = mag2db(dut_cal2_S[:,1,0])
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        mu = mag2db(dut_cal3_S[:,1,0])
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S21 (dB)')
        ax.set_ylim(-4,0)
        ax.set_yticks(np.arange(-4,0.1,1))
        
        ax = axs[1,0]
        mu = np.angle(dut_cal_S[:,0,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')
        mu = np.angle(dut_cal2_S[:,0,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        mu = np.angle(dut_cal3_S[:,0,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.set_ylabel('S11 (deg)')
        ax.set_ylim(-180,180)
        ax.set_yticks(np.arange(-180,181,60))
        
        ax = axs[1,1]
        mu = np.angle(dut_cal_S[:,1,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='o', markevery=30, markersize=10,
                label='mTRL at line center')        
        mu = np.angle(dut_cal2_S[:,1,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='X', markevery=30, markersize=10,
                label='mTRL shifted')
        mu = np.angle(dut_cal3_S[:,1,0], deg=True)
        ax.plot(f*1e-9, mu, lw=2, marker='>', markevery=30, markersize=10,
                label='Thru-free (port-A network-reflect)')
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