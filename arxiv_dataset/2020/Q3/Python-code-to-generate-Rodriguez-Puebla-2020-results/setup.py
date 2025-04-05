from distutils.core import setup
from Cython.Build import cythonize
import os.path #needed for paths
import numpy as np
import sys

setup(
ext_modules = cythonize(os.getcwd() + '/routines/' + "module_gas.pyx")
)

import module_gas
import routines

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager
from matplotlib.ticker import ScalarFormatter
from os import listdir

#from matplotlib.colors import LogNorm
#from scipy.stats import gaussian_kde

import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import ctypes

#(1): conditional distributions parameters:
HI_LTG_params = [0,-0.127167,1.27935,2.59764,8.64563,-0.0180047,0.576922]
H2_LTG_params = [0,-0.0850781,0.830319,0.121454,10.5945,0.840925,0.062628]
HI_ETG_params = [0,-0.0519849,-0.0741485,1.57328,8.35398,-0.819537,0.46762,0.0602505,-0.112526,-0.258828,-0.309552]
H2_ETG_params = [0,0.0585967,-1.49083,0.673683,8.18159,-0.685737,0.375225,0.0174391,0.515328,-1.08394,7.98043]

#(2):
logRHI_all_min=-8.0
logRHI_all_max=3.0
logRH2_all_min=-8.0
logRH2_all_max=3.0
logRHI_ETGs_min=-8.0
logRHI_ETGs_max=3.0
logRH2_ETGs_min=-8.0
logRH2_ETGs_max=3.0
logRgas_LTGs_min=-8.0
logRgas_LTGs_max=3.0
logRHI_LTGs_min=-8.0
logRHI_LTGs_max=3.0
logRH2_LTGs_min=-8.0
logRH2_LTGs_max=3.0

#(3): Rodriguez-Puebla+19 MCMC GSMF best fit parameters
x_stars = [0,-3.0186775,-1.4183881,0.6602491,10.8965073,-2.2664874,-0.2070124,1.2364575,3.5160549,10.8965073]


def ave_logRHj_or_logMHj_vs_Ms(flag_mass, flag_MHj): #logarithmic case <logMHj> and <logRHj>
    if flag_MHj == 1.0:
        name_file = 'logRHI_logMs.dat' if flag_mass == 1.0 else 'logMHI_logMs.dat'
        name_figure = 'logRHI_logMs.pdf' if flag_mass == 1.0 else 'logMHI_logMs.pdf'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure
        labels =  ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle \log {\\rm R_{HI}}\\rangle$'] if flag_mass == 1.0 else ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle \log {\\rm M_{HI}}\\rangle$ $\\rm [M_{\\odot}]$']
    elif flag_MHj == 2.0:
        name_file = 'logRH2_logMs.dat' if flag_mass == 1.0 else 'logMH2_logMs.dat'
        name_figure = 'logRH2_logMs.pdf' if flag_mass == 1.0 else 'logMH2_logMs.pdf'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure
        labels = ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle \log {\\rm R_{H_{2}}}\\rangle$'] if flag_mass == 1.0 else ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle \log {\\rm M_{H_{2}}}\\rangle$ $\\rm [M_{\\odot}]$']

    #===================================================================================#
    NbinsMs = 20
    logMs_min = 7.0
    logMs_max = 12.0

    DeltaMs = (logMs_max - logMs_min) / NbinsMs

    NbinsMHj = 100
    logRHj_min = -6.5
    logRHj_max = 2.5
    logMHj_min = logRHj_min + logMs_min
    logMHj_max = logRHj_max + logMs_max

    Hj_LTG_params = HI_LTG_params if flag_MHj == 1 else H2_LTG_params
    Hj_ETG_params = HI_ETG_params if flag_MHj == 1 else H2_ETG_params

    DeltaMHj = (logMHj_max - logMHj_min) / NbinsMHj

    #===================================================================================#

    if flag_mass == 1.0: #logRHj
        logMs = []
        logRHj_ave_LTGs = []
        SD_logRHj_LTGs = []
        logRHj_ave_ETGs = []
        SD_logRHj_ETGs = []
        logRHj_ave_TOT = []
        SD_logRHj_TOT = []
        results = {}

        f = open(output_path_file, 'w')
        f.write ("#        logMs       logRHj_LTGs_ave    logRHj_LTGs_SD    logRHj_ETGs_ave   logRHj_ETGs_SD   logRHj_all_ave     logRHj_all_SD\n")
        for i in range(NbinsMs + 1):
            logMs_x = logMs_min + i * DeltaMs

            #logRHj
            logRHj_ave_x_LTGs = module_gas.FM_P_logRHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)
            SD_logRHj_x_LTGs = module_gas.SM_P_logRHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)
            SD_logRHj_x_LTGs = np.sqrt(SD_logRHj_x_LTGs)

            logRHj_ave_x_ETGs = module_gas.FM_P_logRHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            SD_logRHj_x_ETGs = module_gas.SM_P_logRHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            SD_logRHj_x_ETGs = np.sqrt(SD_logRHj_x_ETGs)

            logRHj_ave_x_TOT = module_gas.FirstMom_P_logRHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_all_min, logRHI_all_max, flag_MHj)
            SD_logRHj_x_TOT = module_gas.SecondMom_P_logRHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_all_min, logRHI_all_max, flag_MHj)
            SD_logRHj_x_TOT = np.sqrt(SD_logRHj_x_TOT)

            #arrays
            logMs.append(logMs_x)

            logRHj_ave_LTGs.append(logRHj_ave_x_LTGs)
            SD_logRHj_LTGs.append(SD_logRHj_x_LTGs)

            logRHj_ave_ETGs.append(logRHj_ave_x_ETGs)
            SD_logRHj_ETGs.append(SD_logRHj_x_ETGs)

            logRHj_ave_TOT.append(logRHj_ave_x_TOT)
            SD_logRHj_TOT.append(SD_logRHj_x_TOT)

            f.write( "%12g%18g%18g%18g%18g%18g%18g\n" % (logMs_x, logRHj_ave_x_LTGs, SD_logRHj_x_LTGs, logRHj_ave_x_ETGs, SD_logRHj_x_ETGs, logRHj_ave_x_TOT, SD_logRHj_x_TOT) )


        results['x'] = logMs
        results['y_LTGs'] = logRHj_ave_LTGs
        results['y_LTGs_err'] = SD_logRHj_LTGs
        results['y_ETGs'] = logRHj_ave_ETGs
        results['y_ETGs_err'] = SD_logRHj_ETGs
        results['y_TOT'] = logRHj_ave_TOT
        results['y_TOT_err'] = SD_logRHj_TOT

        #print(list(results.keys()))
        f.close()

        #------------------------------------
    elif flag_mass == 2.0: #logMHj
        logMs = []
        logMHj_ave_LTGs = []
        SD_logMHj_LTGs = []
        logMHj_ave_ETGs = []
        SD_logMHj_ETGs = []
        logMHj_ave_TOT = []
        SD_logMHj_TOT = []
        results = {}

        f = open(output_path_file, 'w')
        f.write ("#        logMs       logMHj_LTGs_ave    logMHj_LTGs_SD    logMHj_ETGs_ave   logMHj_ETGs_SD   logMHj_all_ave     logMHj_all_SD\n")
        for i in range(NbinsMs + 1):
            logMs_x = logMs_min + i * DeltaMs

            #MHj
            logMHj_ave_x_LTGs = module_gas.FM_P_logMHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)
            SD_logMHj_x_LTGs = module_gas.SM_P_logMHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)
            SD_logMHj_x_LTGs = np.sqrt(SD_logMHj_x_LTGs)

            logMHj_ave_x_ETGs = module_gas.FM_P_logMHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            SD_logMHj_x_ETGs = module_gas.SM_P_logMHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            SD_logMHj_x_ETGs = np.sqrt(SD_logMHj_x_ETGs)

            logMHj_ave_x_TOT = module_gas.FirstMom_P_logMHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_all_min, logRHI_all_max, flag_MHj)
            SD_logMHj_x_TOT = module_gas.SecondMom_P_logMHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_all_min, logRHI_all_max, flag_MHj)
            SD_logMHj_x_TOT = np.sqrt(SD_logMHj_x_TOT)


            #arrays
            logMs.append(logMs_x)

            logMHj_ave_LTGs.append(logMHj_ave_x_LTGs)
            SD_logMHj_LTGs.append(SD_logMHj_x_LTGs)

            logMHj_ave_ETGs.append(logMHj_ave_x_ETGs)
            SD_logMHj_ETGs.append(SD_logMHj_x_ETGs)

            logMHj_ave_TOT.append(logMHj_ave_x_TOT)
            SD_logMHj_TOT.append(SD_logMHj_x_TOT)
            f.write( "%12g%18g%18g%18g%18g%18g%18g\n" % (logMs_x, logMHj_ave_x_LTGs, SD_logMHj_x_LTGs, logMHj_ave_x_ETGs, SD_logMHj_x_ETGs, logMHj_ave_x_TOT, SD_logMHj_x_TOT) )


        results['x'] = logMs
        results['y_LTGs'] = logMHj_ave_LTGs
        results['y_LTGs_err'] = SD_logMHj_LTGs
        results['y_ETGs'] = logMHj_ave_ETGs
        results['y_ETGs_err'] = SD_logMHj_ETGs
        results['y_TOT'] = logMHj_ave_TOT
        results['y_TOT_err'] = SD_logMHj_TOT

        #print(list(results.keys()))
        f.close()

    #################################################################################
    #                            Figures
    #################################################################################
    #Fonts
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    font_type = {'fontname':'Serif'}

    fig = plt.figure(figsize=None)

    #################################################################################
    ax = plt.subplot2grid((20,20), (4,4), rowspan=10, colspan=10)

    #----------------------------#
    ax.fill_between(results['x'], np.array(results['y_LTGs'])-np.array(results['y_LTGs_err']), np.array(results['y_LTGs'])+np.array(results['y_LTGs_err']), alpha=0.4, edgecolor='blue', facecolor='blue', linewidth=0.1)
    ax.plot(results['x'], results['y_LTGs'], color = 'blue', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"

    ax.fill_between(results['x'], np.array(results['y_ETGs'])-np.array(results['y_ETGs_err']), np.array(results['y_ETGs'])+np.array(results['y_ETGs_err']), alpha=0.4, edgecolor='red', facecolor='red', linewidth=0.1)
    ax.plot(results['x'], results['y_ETGs'], color = 'red', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"

    #ax.fill_between(results['x'], np.array(results['logRHj_ave_TOT'])-np.array(results['SD_logRHj_TOT']), np.array(results['logRHj_ave_TOT'])+np.array(results['SD_logRHj_TOT']), alpha=0.4, edgecolor='black', facecolor='black', linewidth=0.1)
    ax.plot(results['x'], results['y_TOT'], color = 'black', marker=' ', linestyle='-', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"
    ax.plot(results['x'], np.array(results['y_TOT']) + np.array(results['y_TOT_err']), color = 'black', marker=' ', linestyle=':', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"
    ax.plot(results['x'], np.array(results['y_TOT']) - np.array(results['y_TOT_err']), color = 'black', marker=' ', linestyle=':', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=True) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    if flag_mass == 1.0:
        ax.set_xlim(7, 12.0)
        ax.set_ylim(-4.5, 2.0)
    elif flag_mass == 2.0:
        ax.set_xlim(7, 12.0)
        ax.set_ylim(6, 11)

    #axis scale
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #Labels
    ax.set_xlabel(labels[0] , fontsize = '14')
    ax.set_ylabel(labels[1] , fontsize = '14')
    #Ticks
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    ##################################################################################
    #title
    #titlefig = fig.suptitle('Using $\log(\\rm M_{\\ast})-\log (\\rm V_{max})$ and $\log(\\rm M_{\\ast})-\log (\\rm V_{peak})$ correlation for ETGs and LTGs', fontsize='10', fontweight='light', **font_type)
    #titlefig.set_y(0.88) #Where to place the title
    fig.subplots_adjust(top=0.92) #Adjust subplots with respect title

    #configuring subplots
    fig.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=0.0, hspace=0.0)
    #fig.set_tight_layout(True) !To automatically leave the subplots symmetric
    #plt.draw()
    plt.savefig(output_path_figure, Transparent=True)
    plt.show()

#=======================================================================!
# FUNCTION:  ave_RHj_or_MHj_vs_Ms                                       !
#   Computes and plot average MHj and RHj (linear space) as a function
#   of stellar mass. This is done for LT, ET and the whole population of
#   galaxies.
#
#   Inputs:
#   (1) flag_mass = Indicates to computes masses RHj(1) or ratios MHj(2).
#   (2) flag_MHj = Atomic or molecular Hydrogen? (1 or 2 respectively)
#
#  Notes:
#
#
#======================================================================*/

def ave_RHj_or_MHj_vs_Ms(flag_mass, flag_MHj): #linear case <MHj>
    if flag_MHj == 1: #HI
        name_file = 'RHI_Ms.dat' if flag_mass == 1.0 else 'MHI_Ms.dat'
        name_figure = 'RHI_Ms.pdf' if flag_mass == 1.0 else 'MHI_Ms.pdf'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure

        labels =  ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle {\\rm R_{HI}}\\rangle$'] if flag_mass == 1.0 else ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle {\\rm M_{HI}}\\rangle$ $\\rm [M_{\\odot}]$']
    elif flag_MHj == 2: #H2
        name_file = 'RH2_Ms.dat' if flag_mass == 1.0 else 'MH2_Ms.dat'
        name_figure = 'RH2_Ms.pdf' if flag_mass == 1.0 else 'MH2_Ms.pdf'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure
        labels = ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle {\\rm R_{H_{2}}}\\rangle$'] if flag_mass == 1.0 else ['$\\rm \log M_{\\ast}$ $\\rm [M_{\\odot}]$', '$\\langle {\\rm M_{H_{2}}}\\rangle$ $\\rm [M_{\\odot}]$']

    #===================================================================================#
    NbinsMs = 20;
    logMs_min = 7.0;
    logMs_max = 12.0;

    DeltaMs = (logMs_max - logMs_min) / NbinsMs;

    NbinsMHj = 100;
    logRHj_min = -6.5;
    logRHj_max = 2.5;
    logMHj_min = logRHj_min + logMs_min;
    logMHj_max = logRHj_max + logMs_max;

    Hj_LTG_params = HI_LTG_params if flag_MHj == 1 else H2_LTG_params
    Hj_ETG_params = HI_ETG_params if flag_MHj == 1 else H2_ETG_params

    DeltaMHj = (logMHj_max - logMHj_min) / NbinsMHj;

    #===================================================================================#
    if flag_mass == 1.0: #RHj
        logMs = []
        log_RHj_ave_LTGs = []
        SD_log_RHj_ave_LTGs = []

        log_RHj_ave_ETGs = []
        SD_log_RHj_ave_ETGs = []

        log_RHj_ave_TOT = []
        SD_log_RHj_ave_TOT = []

        results = {}

        f = open(output_path_file, 'w')
        f.write ("#        logMs       RHj_LTGs_ave      RHj_LTGs_SD      RHj_ETGs_ave      RHj_ETGs_SD      RHj_all_ave      RHj_all_SD\n")

        for i in range(NbinsMs + 1):
            logMs_x = logMs_min + i * DeltaMs

            RHj_ave_x_LTGs = module_gas.FM_P_RHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)
            RHj_SD_x_LTGs = module_gas.SM_P_RHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)

            RHj_ave_x_ETGs = module_gas.FM_P_RHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            RHj_SD_x_ETGs = module_gas.SM_P_RHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)

            RHj_ave_x_TOT = module_gas.FirstMom_P_RHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_all_min, logRHI_all_max, flag_MHj)
            RHj_SD_x_TOT = module_gas.SecondMom_P_RHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_all_min, logRHI_all_max, flag_MHj)

            logMs.append(logMs_x)
            log_RHj_ave_LTGs.append( RHj_ave_x_LTGs )
            SD_log_RHj_ave_LTGs.append(np.sqrt(RHj_SD_x_LTGs))
            log_RHj_ave_ETGs.append( RHj_ave_x_ETGs )
            SD_log_RHj_ave_ETGs.append(np.sqrt(RHj_SD_x_ETGs))
            log_RHj_ave_TOT.append(RHj_ave_x_TOT)
            SD_log_RHj_ave_TOT.append(np.sqrt(RHj_SD_x_TOT))
            f.write( "%12g%12g%12g%12g%12g%12g%12g\n" % (logMs_x, RHj_ave_x_LTGs, np.sqrt(RHj_SD_x_LTGs), RHj_ave_x_ETGs, np.sqrt(RHj_SD_x_ETGs), RHj_ave_x_TOT, np.sqrt(RHj_SD_x_TOT)) )

        results['x'] = logMs
        results['y_LTGs'] = log_RHj_ave_LTGs
        results['y_LTGs_err'] = SD_log_RHj_ave_LTGs
        results['y_ETGs'] = log_RHj_ave_ETGs
        results['y_ETGs_err'] = SD_log_RHj_ave_ETGs
        results['y_TOT'] = log_RHj_ave_TOT
        results['y_TOT_err'] = SD_log_RHj_ave_TOT

        #print(list(results.keys()))
        f.close()

    elif flag_mass == 2.0: #MHj
        logMs = []
        log_MHj_ave_LTGs = []
        SD_log_MHj_ave_LTGs = []

        log_MHj_ave_ETGs = []
        SD_log_MHj_ave_ETGs = []

        log_MHj_ave_TOT = []
        SD_log_MHj_ave_TOT = []

        SD_logRHj_LTGs = []
        logRHj_ave_ETGs = []
        SD_logRHj_ETGs = []
        logRHj_ave_TOT = []
        SD_logRHj_TOT = []
        results = {}

        f = open(output_path_file, 'w')
        f.write ("#        logMs       MHj_LTGs_ave      MHj_LTGs_SD      MHj_ETGs_ave      MHj_ETGs_SD      MHj_all_ave      MHj_all_SD\n")

        for i in range(NbinsMs + 1):
            logMs_x = logMs_min + i * DeltaMs

            MHj_ave_x_LTGs = module_gas.FM_MHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)
            MHj_SD_x_LTGs = module_gas.SM_MHj_LTGs(logMs_x, Hj_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, flag_MHj)

            MHj_ave_x_ETGs = module_gas.FM_P_MHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            MHj_SD_x_ETGs = module_gas.SM_P_MHj_ETGs(logMs_x, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)

            MHj_ave_x_TOT = module_gas.FirstMom_P_MHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)
            MHj_SD_x_TOT = module_gas.SecondMom_P_MHj_all(logMs_x, Hj_LTG_params, Hj_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, flag_MHj)

            logMs.append(logMs_x)
            log_MHj_ave_LTGs.append( MHj_ave_x_LTGs )
            SD_log_MHj_ave_LTGs.append(np.sqrt(MHj_SD_x_LTGs))
            log_MHj_ave_ETGs.append( MHj_ave_x_ETGs )
            SD_log_MHj_ave_ETGs.append(np.sqrt(MHj_SD_x_ETGs))
            log_MHj_ave_TOT.append(MHj_ave_x_TOT)
            SD_log_MHj_ave_TOT.append(np.sqrt(MHj_SD_x_TOT))
            f.write( "%12g%12g%12g%12g%12g%12g%12g\n" % (logMs_x, MHj_ave_x_LTGs, np.sqrt(MHj_SD_x_LTGs), MHj_ave_x_ETGs, np.sqrt(MHj_SD_x_ETGs), MHj_ave_x_TOT, np.sqrt(MHj_SD_x_TOT)) )

        results['x'] = logMs
        results['y_LTGs'] = log_MHj_ave_LTGs
        results['y_LTGs_err'] = SD_log_MHj_ave_LTGs
        results['y_ETGs'] = log_MHj_ave_ETGs
        results['y_ETGs_err'] = SD_log_MHj_ave_ETGs
        results['y_TOT'] = log_MHj_ave_TOT
        results['y_TOT_err'] = SD_log_MHj_ave_TOT

        #print(list(results.keys()))
        f.close()

    #################################################################################
    #                            Figures
    #################################################################################
    #Fonts
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    font_type = {'fontname':'Serif'}

    fig = plt.figure(figsize=None)

    #################################################################################
    ax = plt.subplot2grid((20,20), (4,4), rowspan=10, colspan=10)

    #----------------------------#
    #plotting results
    ax.fill_between(results['x'], np.array(results['y_LTGs'])-np.array(results['y_LTGs_err']), np.array(results['y_LTGs'])+np.array(results['y_LTGs_err']), alpha=0.4, edgecolor='none', facecolor='blue', linewidth=0.1)
    ax.plot(results['x'], results['y_LTGs'], color = 'blue', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 0.2, zorder = 1) #marker="None"

    ax.fill_between(results['x'], np.array(results['y_ETGs'])-np.array(results['y_ETGs_err']), np.array(results['y_ETGs'])+np.array(results['y_ETGs_err']), alpha=0.4, edgecolor='none', facecolor='red', linewidth=0.1)
    ax.plot(results['x'], results['y_ETGs'], color = 'red', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 0.2, zorder = 1) #marker="None"

    #ax.fill_between(results['x'], np.array(results['logRHj_ave_TOT'])-np.array(results['SD_logRHj_TOT']), np.array(results['logRHj_ave_TOT'])+np.array(results['SD_logRHj_TOT']), alpha=0.4, edgecolor='black', facecolor='black', linewidth=0.1)
    ax.plot(results['x'], results['y_TOT'], color = 'black', marker=' ', linestyle='-', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 0.1) #marker="None"
    ax.plot(results['x'], np.array(results['y_TOT']) + np.array(results['y_TOT_err']), color = 'black', marker=' ', linestyle='--', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 0.2) #marker="None"
    ax.plot(results['x'], np.array(results['y_TOT']) - np.array(results['y_TOT_err']), color = 'black', marker=' ', linestyle='--', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 0.2) #marker="None"

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=True) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    #ax.set_xlim(7, 12.0)
    if flag_mass == 1.0:
        ax.set_ylim(np.power(10, -4.5), np.power(10, 2.0))
    elif flag_mass == 2.0:
        ax.set_ylim(1e6, 1e11)
    #axis scale
    ax.set_xscale("linear")
    ax.set_yscale("log")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #Labels
    ax.set_xlabel(labels[0] , fontsize = '14')
    ax.set_ylabel(labels[1] , fontsize = '14')
    #Ticks
    #To add minor ticks in log scale
    if flag_mass == 1.0:
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=7)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])

    elif flag_mass == 2.0:
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=6)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    #################################################################################
    #title
    #titlefig = fig.suptitle('Using $\log(\\rm M_{\\ast})-\log (\\rm V_{max})$ and $\log(\\rm M_{\\ast})-\log (\\rm V_{peak})$ correlation for ETGs and LTGs', fontsize='10', fontweight='light', **font_type)
    #titlefig.set_y(0.88) #Where to place the title
    fig.subplots_adjust(top=0.92) #Adjust subplots with respect title

    #configuring subplots
    fig.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=0.0, hspace=0.0)
    #fig.set_tight_layout(True) !To automatically leave the subplots symmetric
    #plt.draw()
    plt.savefig(output_path_figure, Transparent=True)
    plt.show()

##################################################################################
def P_MHj_RHj(logMs_ini, logMs_end, flag_MHj, flag_mass):
    if flag_MHj == 1.0: #HI
        name_file = 'P_RHI_weighted' if flag_mass == 1.0 else 'P_MHI_weighted'
        name_figure = 'P_RHI_weighted' if flag_mass == 1.0 else 'P_MHI_weighted'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file + '_logMs_' + str(logMs_ini) + '_' + str(logMs_end) + '.dat'
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure + '_logMs_' + str(logMs_ini) + '_' + str(logMs_end) + '.pdf'

    elif flag_MHj == 2.0: #H2
        name_file = 'P_RH2_weighted' if flag_mass == 1.0 else 'P_MH2_weighted'
        name_figure = 'P_RH2_weighted' if flag_mass == 1.0 else 'P_MH2_weighted'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file + '_logMs_' + str(logMs_ini) + '_' + str(logMs_end) + '.dat'
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure + '_logMs_' + str(logMs_ini) + '_' + str(logMs_end) + '.pdf'

    if flag_MHj == 1:
        labels =  ['$\\rm \log R_{\\rm HI}$', '$\\langle P({\\rm M}_{\\rm HI}|{\\rm M}_{\\ast})\\rangle$'] if flag_mass == 1 else ['$\\rm \log M_{\\rm HI}$ $\\rm [M_{\\odot}]$', '$\\langle P({\\rm M}_{\\rm HI}|{\\rm M}_{\\ast})\\rangle$']
    elif flag_MHj == 2:
        labels =  ['$\\rm \log R_{\\rm H_{2}}$', '$\\langle P({\\rm M}_{\\rm H_{2}}|{\\rm M}_{\\ast})\\rangle$'] if flag_mass == 1 else ['$\\rm \log M_{\\rm H_{2}}$ $\\rm [M_{\\odot}]$', '$\\langle P({\\rm M}_{\\rm H_{2}}|{\\rm M}_{\\ast})\\rangle$']

    #################################################################################
    bin = {}

    logMs_mid = logMs_ini + ( (logMs_end - logMs_ini) * 0.5 )

    Nbins_logMHj = 200
    logMHj_ini = 5.0
    logMHj_end = 11.80
    delta_logMHj = (logMHj_end - logMHj_ini) / (Nbins_logMHj * 1.0)
    N = 110

    Hj_LTG_params = HI_LTG_params if flag_MHj == 1 else H2_LTG_params
    Hj_ETG_params = HI_ETG_params if flag_MHj == 1 else H2_ETG_params

    logMHj = []
    logRHj = []
    PDF_RHj_LTGs = []
    PDF_RHj_ETGs = []
    PDF_RHj_TOT = []
    PDF_RHj_TOT2 = []
    logMHj_x = logMHj_ini

    norm_LTGs = module_gas.Int_norm_Mtype(logMs_ini, logMs_end, x_stars, 1.0)
    norm_ETGs = module_gas.Int_norm_Mtype(logMs_ini, logMs_end, x_stars, 2.0)
    norm_ALL = module_gas.Int_norm(logMs_ini, logMs_end, x_stars)

    f = open(output_path_file, 'w')
    f.write ("#        logMHj      logRHj       PDF_MHj_LT      PDF_MHj_ET      PDF_MHj_ALL \n")
    while(logMHj_x <= logMHj_end):
        logMHj.append(logMHj_x)
        logRHj.append(logMHj_x - logMs_mid)

        PDF_RHj_LT = module_gas.ave_P_MHj_Ms_blue_LTGs(logMHj_x, Hj_LTG_params, x_stars, flag_MHj, logMs_ini, logMs_end, N, norm_LTGs) * (norm_LTGs / norm_ALL)
        PDF_RHj_ET = module_gas.ave_P_MHj_Ms_red_ETGs(logMHj_x, Hj_ETG_params, x_stars, flag_MHj, logMs_ini, logMs_end, N, norm_ETGs) * (norm_ETGs / norm_ALL)
        PDF_RHj_ALL = PDF_RHj_LT + PDF_RHj_ET

        PDF_RHj_LTGs.append(PDF_RHj_LT)
        PDF_RHj_ETGs.append(PDF_RHj_ET)
        PDF_RHj_TOT.append(PDF_RHj_ALL)
        logMHj_x = logMHj_x + delta_logMHj

        f.write( "%15g%15g%15g%15g%15g\n" % (logMHj_x, logMHj_x - logMs_mid, PDF_RHj_LT, PDF_RHj_ET, PDF_RHj_ALL ) )

    f.close()

    bin['logMHj'] = logMHj
    bin['logRHj'] = logRHj
    bin['PDF_RHj_LTGs'] = PDF_RHj_LTGs
    bin['PDF_RHj_ETGs'] = PDF_RHj_ETGs
    bin['PDF_RHj_TOT'] = PDF_RHj_TOT
    #print(list(bin.keys()))

    #################################################################################
    #                            Figures
    #################################################################################
    #Fonts
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    font_type = {'fontname':'Serif'}

    fig = plt.figure(figsize=None)

    #################################################################################
    ax = plt.subplot2grid((20,20), (4,4), rowspan=10, colspan=10)

    #----------------------------#
    if flag_mass == 1:
        ax.plot(bin['logRHj'], bin['PDF_RHj_LTGs'], color = 'blue', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 3) #marker="None"
        ax.plot(bin['logRHj'], bin['PDF_RHj_ETGs'], color = 'red', marker=' ', linestyle='-', mec='none', markerfacecolor='red', markersize='1.0', linewidth = 1.0, zorder = 2) #marker="None"
        ax.plot(bin['logRHj'], bin['PDF_RHj_TOT'], color = 'black', marker=' ', linestyle='-', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"
    elif flag_mass == 2 :
        ax.plot(bin['logMHj'], bin['PDF_RHj_LTGs'], color = 'blue', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 3) #marker="None"
        ax.plot(bin['logMHj'], bin['PDF_RHj_ETGs'], color = 'red', marker=' ', linestyle='-', mec='none', markerfacecolor='red', markersize='1.0', linewidth = 1.0, zorder = 2) #marker="None"
        ax.plot(bin['logMHj'], bin['PDF_RHj_TOT'], color = 'black', marker=' ', linestyle='-', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"

    #----------------------------#
    #LABELS
    string_ini = "%5.2f" % logMs_ini
    string_end = "%5.2f" % logMs_end

    if flag_mass == 1:
        ax.text(-4.5, 0.8, string_ini + '$\\rm \\leq \log M_{\\ast}\\leq$' + string_end, style = 'normal', color='black', fontsize = 14)
    elif flag_mass == 2:
        ax.text(6.5, 0.8, string_ini + '$\\rm \\leq \log M_{\\ast}\\leq$' + string_end, style = 'normal', color='black', fontsize = 14)

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=True) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    if flag_mass == 1:
        ax.set_xlim(-6, 1)
    elif flag_mass == 2:
        ax.set_xlim(5, 12)

    ax.set_ylim(0, 1)
    #axis scale
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #Labels
    ax.set_xlabel(labels[0] , fontsize = '14')
    ax.set_ylabel(labels[1] , fontsize = '14')
    #Ticks
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) #Major ticks
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    #################################################################################
    #title
    #titlefig = fig.suptitle('Using $\log(\\rm M_{\\ast})-\log (\\rm V_{max})$ and $\log(\\rm M_{\\ast})-\log (\\rm V_{peak})$ correlation for ETGs and LTGs', fontsize='10', fontweight='light', **font_type)
    #titlefig.set_y(0.88) #Where to place the title
    fig.subplots_adjust(top=0.92) #Adjust subplots with respect title

    #configuring subplots
    fig.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=0.0, hspace=0.0)
    #fig.set_tight_layout(True) !To automatically leave the subplots symmetric
    #plt.draw()
    plt.savefig(output_path_figure, Transparent=True)
    plt.show()

##################################################################################
def percentiles(flag_res, flag_MHj, per1, per2, per3):

    if flag_MHj == 1.0: #HI
        name_file = 'RHI_percentiles.dat' if flag_res == 1.0 else 'MHI_percentiles.dat'
        name_figure = 'RHI_percentiles.pdf' if flag_res == 1.0 else 'MHI_percentiles.pdf'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure
        labels =  ['$\\rm M_{\\ast}$ $\\rm [M_{\\odot}]$', '${\\rm R_{HI}}$'] if flag_res == 1 else ['$\\rm M_{\\ast}$ $\\rm [M_{\\odot}]$', '${\\rm M_{HI}}$ $\\rm [M_{\\odot}]$']

    elif flag_MHj == 2.0: #H2
        name_file = 'RH2_percentiles.dat' if flag_res == 1.0 else 'MH2_percentiles.dat'
        name_figure = 'RH2_percentiles.pdf' if flag_res == 1.0 else 'MH2_percentiles.pdf'
        output_path_file = os.getcwd() + '/outputs/files/' + name_file
        output_path_figure = os.getcwd() + '/outputs/figures/' + name_figure
        labels =  ['$\\rm M_{\\ast}$ $\\rm [M_{\\odot}]$', '${\\rm R_{H_{2}}}$'] if flag_res == 1 else ['$\\rm M_{\\ast}$ $\\rm [M_{\\odot}]$', '${\\rm M_{H_{2}}}$ $\\rm [M_{\\odot}]$']

    Hj_LTG_params = HI_LTG_params if flag_MHj == 1 else H2_LTG_params
    Hj_ETG_params = HI_ETG_params if flag_MHj == 1 else H2_ETG_params

    #===================================================================================#
    results = {}

    Nbins_logMs = 20
    logMs_ini = 7.0
    logMs_end = 12.0
    delta_logMs = (logMs_end - logMs_ini) / (Nbins_logMs * 1.0)
    logMs_mid = logMs_ini + ( (logMs_end - logMs_ini) * 0.5 )
    N = 110

    per = []

    logMs = []
    p3_LTGs = []
    p2_LTGs = []
    p1_LTGs = []
    p3_ETGs = []
    p2_ETGs = []
    p1_ETGs = []
    p3 = []
    p2 = []
    p1 = []

    logMs_x = logMs_ini

    f = open(output_path_file, 'w')
    if flag_res == 1:
        f.write ("#        logMs       RHj_LTGs_p16      RHj_LTGs_p50      RHj_LTGs_p84       RHj_ETGs_p16      RHj_ETGs_p50      RHj_ETGs_p84       RHj_TOT_p16      RHj_TOT_p50      RHj_TOT_p84       \n")
    elif flag_res == 2:
        f.write ("#        logMs       MHj_LTGs_p16      MHj_LTGs_p50      MHj_LTGs_p84       MHj_ETGs_p16      MHj_ETGs_p50      MHj_ETGs_p84       MHj_TOT_p16      MHj_TOT_p50      MHj_TOT_p84       \n")

    while(logMs_x <= logMs_end):
        logMs.append(logMs_x)

        per_LTGs = module_gas.percentiles_logRHj_LTGs( logMs_x,  Hj_LTG_params,  logRHI_LTGs_min,  logRHI_LTGs_max,  flag_MHj, per1, per2, per3)
        per_ETGs = module_gas.percentiles_logRHj_ETGs( logMs_x,  Hj_ETG_params,  logRHI_ETGs_min,  logRHI_ETGs_max,  flag_MHj, per1, per2, per3)
        per = module_gas.percentiles_logRHj_all( logMs_x,  Hj_LTG_params, Hj_ETG_params,  logRHI_all_min,  logRHI_all_max,  flag_MHj, per1, per2, per3)

        p1_MHj_LTGs = per_LTGs[0] + logMs_x
        p2_MHj_LTGs = per_LTGs[1] + logMs_x
        p3_MHj_LTGs = per_LTGs[2] + logMs_x
        p1_MHj_ETGs = per_ETGs[0] + logMs_x
        p2_MHj_ETGs = per_ETGs[1] + logMs_x
        p3_MHj_ETGs = per_ETGs[2] + logMs_x
        p1_MHj = per[0] + logMs_x
        p2_MHj = per[1] + logMs_x
        p3_MHj = per[2] + logMs_x
        p1_LTGs.append(per_LTGs[0])
        p2_LTGs.append(per_LTGs[1])
        p3_LTGs.append(per_LTGs[2])
        p1_ETGs.append(per_ETGs[0])
        p2_ETGs.append(per_ETGs[1])
        p3_ETGs.append(per_ETGs[2])
        p1.append(per[0])
        p2.append(per[1])
        p3.append(per[2])
        logMs_x = logMs_x + delta_logMs

        if flag_res == 1:
            f.write( "%12g%12g%12g%12g%12g%12g%12g%12g%12g%12g\n" % (logMs_x, np.power(10, per_LTGs[0]), np.power(10, per_LTGs[1]), np.power(10, per_LTGs[2]), np.power(10, per_ETGs[0]), np.power(10, per_ETGs[1]), np.power(10, per_ETGs[2]), np.power(10, per[0]), np.power(10, per[1]), np.power(10, per[2]) ) )
        elif flag_res == 2:
            f.write( "%12g%12g%12g%12g%12g%12g%12g%12g%12g%12g\n" % (logMs_x, np.power(10, p1_MHj_LTGs), np.power(10, p2_MHj_LTGs), np.power(10, p3_MHj_LTGs), np.power(10, p1_MHj_ETGs), np.power(10, p2_MHj_ETGs), np.power(10, p3_MHj_ETGs), np.power(10, p1_MHj), np.power(10, p2_MHj), np.power(10, p3_MHj)) )

    results['logMs'] = logMs
    results['p1_LTGs'] = p1_LTGs if flag_res == 1 else np.array(p1_LTGs) + np.array(logMs)
    results['p2_LTGs'] = p2_LTGs if flag_res == 1 else np.array(p2_LTGs) + np.array(logMs)
    results['p3_LTGs'] = p3_LTGs if flag_res == 1 else np.array(p3_LTGs) + np.array(logMs)
    results['p1_ETGs'] = p1_ETGs if flag_res == 1 else np.array(p1_ETGs) + np.array(logMs)
    results['p2_ETGs'] = p2_ETGs if flag_res == 1 else np.array(p2_ETGs) + np.array(logMs)
    results['p3_ETGs'] = p3_ETGs if flag_res == 1 else np.array(p3_ETGs) + np.array(logMs)
    results['p1'] = p1 if flag_res == 1 else np.array(p1) + np.array(logMs)
    results['p2'] = p2 if flag_res == 1 else np.array(p2) + np.array(logMs)
    results['p3'] = p3 if flag_res == 1 else np.array(p3) + np.array(logMs)
    print(list(results.keys()))
    f.close()

    #################################################################################
    #                            Figures
    #################################################################################
    #Fonts
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    font_type = {'fontname':'Serif'}

    fig = plt.figure(figsize=None)

    #################################################################################
    ax = plt.subplot2grid((20,20), (4,4), rowspan=10, colspan=10)

    #----------------------------#
    #plotting results
    ax.fill_between(np.power(10, results['logMs']), np.power(10, results['p1_LTGs']), np.power(10, results['p3_LTGs']), alpha=0.2, edgecolor='blue', facecolor='blue', linewidth=0.1)
    ax.plot(np.power(10, results['logMs']), np.power(10, results['p2_LTGs']), color = 'blue', marker=' ', linestyle=':', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    ax.fill_between(np.power(10, results['logMs']), np.power(10, results['p1_ETGs']), np.power(10, results['p3_ETGs']), alpha=0.2, edgecolor='red', facecolor='red', linewidth=0.1)
    ax.plot(np.power(10, results['logMs']), np.power(10, results['p2_ETGs']), color = 'red', marker=' ', linestyle=':', mec='none', markerfacecolor='red', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    ax.fill_between(np.power(10, results['logMs']), np.power(10, results['p1']), np.power(10, results['p3']), alpha=0.4, edgecolor='gray', facecolor='gray', linewidth=0.1)
    ax.plot(np.power(10, results['logMs']), np.power(10, results['p2']), color = 'black', marker=' ', linestyle=':', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=True) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    ax.set_xlim(1e7, 1e12)
    if flag_res == 1:
        ax.set_ylim(np.power(10, -4.5), np.power(10, 2.5))
    elif flag_res == 2:
        ax.set_ylim(np.power(10, 6), np.power(10, 11))

    #axis scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #Labels
    ax.set_xlabel(labels[0] , fontsize = '14')
    ax.set_ylabel(labels[1] , fontsize = '14')
    #Ticks
    #To add minor ticks in log scale
    if flag_res == 1:
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=7)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
    elif flag_res == 2:
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=6)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11])

    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    #################################################################################
    #title
    #titlefig = fig.suptitle('Using $\log(\\rm M_{\\ast})-\log (\\rm V_{max})$ and $\log(\\rm M_{\\ast})-\log (\\rm V_{peak})$ correlation for ETGs and LTGs', fontsize='10', fontweight='light', **font_type)
    #titlefig.set_y(0.88) #Where to place the title
    fig.subplots_adjust(top=0.92) #Adjust subplots with respect title

    #configuring subplots
    fig.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=0.0, hspace=0.0)
    #fig.set_tight_layout(True) !To automatically leave the subplots symmetric
    #plt.draw()

    plt.savefig(output_path_figure, Transparent=True)
    plt.show()

##################################################################################
def read_write_mass_functions(path, path_out):
    data = {}
    logM = []
    log_phi_original_neg_systematic = []
    log_phi_original = []
    log_phi_original_post_systematic = []
    log_phi_deconvolved = []


    fin = open(path, 'r')

    header = fin.readline()

    for line in fin:
        line = line.strip() #removes '\n' from the string
        columns = line.split() #makes a list of parameters included in this line
        logM.append( float(columns[0]) )
        log_phi_original_neg_systematic.append( float(columns[1]) )
        log_phi_original.append( float(columns[2]) )
        log_phi_original_post_systematic.append( float(columns[3]) )
        log_phi_deconvolved.append( float(columns[4]) )

    fin.close()

    source = {} # Empty dictionary for each line and populating it with the interested variables
    source['logM'] = logM
    source['log_phi_original_neg_systematic'] = log_phi_original_neg_systematic
    source['log_phi_original'] = log_phi_original
    source['log_phi_original_post_systematic'] = log_phi_original_post_systematic
    source['log_phi_deconvolved'] = log_phi_deconvolved

    #print(list(source.keys())) #This prints what was saved in the dictionary f

    #Saves output
    fout = open(path_out, 'w')
    #fout.write(header)
    for i in range(len(logM)):
        fout.write( "%12g%12g%12g%12g%12g\n" % (logM[i], log_phi_original_neg_systematic[i], log_phi_original[i], log_phi_original_post_systematic[i],  log_phi_deconvolved[i]) )

    fout.close()

    return source

##################################################################################
def Int_MFs(flag_M):

    if flag_M == 1.0:
        name_main = 'gsmf'
        xlabel = '${\\rm M}_{\\ast}$ $[\\rm M_{\\odot}]$'
        ylabel = '$\\phi({\\rm M}_{\\ast})$ $\\rm [dex^{-1}Mpc^{-3}]$'
    elif flag_M == 2.0:
        name_main = 'ghimf'
        xlabel = '${\\rm M}_{\\rm HI}$ $[\\rm M_{\\odot}]$'
        ylabel = '$\\phi({\\rm M}_{\\rm HI})$ $\\rm [dex^{-1}Mpc^{-3}]$'
    elif flag_M == 3.0:
        name_main = 'gh2mf'
        xlabel = '${\\rm M}_{\\rm H_{2}}$ $[\\rm M_{\\odot}]$'
        ylabel = '$\\phi({\\rm M}_{\\rm H_{2}})$ $\\rm [dex^{-1}Mpc^{-3}]$'
    elif flag_M == 4.0:
        name_main = 'ggmf'
        xlabel = '${\\rm M}_{\\rm gas}$ $[\\rm M_{\\odot}]$'
        ylabel = '$\\phi({\\rm M}_{\\rm gas})$ $\\rm [dex^{-1}Mpc^{-3}]$'
    elif flag_M == 5.0:
        name_main = 'gbmf'
        xlabel = '${\\rm M}_{\\rm bar}$ $[\\rm M_{\\odot}]$'
        ylabel = '$\\phi({\\rm M}_{\\rm bar})$ $\\rm [dex^{-1}Mpc^{-3}]$'

    name_all = name_main + '_all.dat'
    name_LTGs = name_main + '_LTGs.dat'
    name_ETGs = name_main + '_ETGs.dat'

    name_figure = 'intrinsic_' + name_main + '.pdf'

    input_folder = os.getcwd() + '/input/' + 'intrinsic_mass_functions/'
    output_folder = os.getcwd() + '/outputs/' + 'files/'
    output_path_figure = os.getcwd() + '/outputs/' + 'figures/'
    files = os.listdir(input_folder)

    path = input_folder + name_all
    path_LTGs = input_folder + name_LTGs
    path_ETGs = input_folder + name_ETGs

    MFs = read_write_mass_functions(path, output_folder + name_all)
    MFs_LTGs = read_write_mass_functions(path_LTGs, output_folder + name_LTGs)
    MFs_ETGs = read_write_mass_functions(path_ETGs, output_folder + name_ETGs)

    #################################################################################
    #                            Figures
    #################################################################################
    #Fonts
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    font_type = {'fontname':'Serif'}

    fig = plt.figure(figsize=None)

    #################################################################################
    ax = plt.subplot2grid((20,20), (2,1), rowspan=5, colspan=5)

    #----------------------------#
    #plotting results
    #ax.plot(np.power(10, MFs['logM']), np.power(10, MFs['log_phi_original']), color = 'black', marker=' ', linestyle='-', mec='none', markerfacecolor='black', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    ax.fill_between(np.power(10, MFs['logM']), np.power(10, np.array(MFs['log_phi_deconvolved']) - (np.array(MFs['log_phi_original_post_systematic']) - np.array(MFs['log_phi_original']))) , np.power(10, np.array(MFs['log_phi_deconvolved']) + (np.array(MFs['log_phi_original']) - np.array(MFs['log_phi_original_neg_systematic'])) ), alpha=0.3, edgecolor='gray', facecolor='gray', linewidth=0.1)
    ax.plot(np.power(10, MFs['logM']), np.power(10, MFs['log_phi_deconvolved']), color = 'gray', marker=' ', linestyle='-', mec='none', markerfacecolor='gray', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=True) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    ax.set_xlim(1e7, 1e12)
    ax.set_ylim(1e-6, 1e0)

    #axis scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    #Labels
    ax.set_xlabel(xlabel , fontsize = '11')
    ax.set_ylabel(ylabel , fontsize = '11')
    #Ticks
    #To add minor ticks in log scale y-axis
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=7)
    ax.yaxis.set_minor_locator(locmin)
    #ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

    #To add minor ticks in log scale x-axis
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=6)
    ax.xaxis.set_minor_locator(locmin)

    ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    #################################################################################
    ax = plt.subplot2grid((20,20), (2,7), rowspan=5, colspan=5)

    #----------------------------#
    #plotting results
    #ax.plot(np.power(10, MFs_LTGs['logM']), np.power(10, MFs_LTGs['log_phi_original']), color = 'blue', marker=' ', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    ax.fill_between(np.power(10, MFs_LTGs['logM']), np.power(10, np.array(MFs_LTGs['log_phi_deconvolved']) - (np.array(MFs_LTGs['log_phi_original_post_systematic']) - np.array(MFs_LTGs['log_phi_original']))) , np.power(10, np.array(MFs_LTGs['log_phi_deconvolved']) + (np.array(MFs_LTGs['log_phi_original']) - np.array(MFs_LTGs['log_phi_original_neg_systematic'])) ), alpha=0.3, edgecolor='#006666', facecolor='#006666', linewidth=0.1)
    ax.plot(np.power(10, MFs_LTGs['logM']), np.power(10, MFs_LTGs['log_phi_deconvolved']), color = '#006666', marker=' ', linestyle='-', mec='none', markerfacecolor='#006666', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=False) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    ax.set_xlim(1e7, 1e12)
    ax.set_ylim(1e-6, 1e0)

    #axis scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    #Labels
    ax.set_xlabel(xlabel, fontsize = '11')
    #ax.set_ylabel('$\\phi({\\rm M}_{x})$ $\\rm [dex^{-1}Mpc^{-3}]$' , fontsize = '11')
    #Ticks
    #To add minor ticks in log scale y-axis
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=7)
    ax.yaxis.set_minor_locator(locmin)
    #ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

    #To add minor ticks in log scale x-axis
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=6)
    ax.xaxis.set_minor_locator(locmin)

    ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    #################################################################################
    ax = plt.subplot2grid((20,20), (2,13), rowspan=5, colspan=5)

    #----------------------------#
    #plotting results
    #ax.plot(np.power(10, MFs_ETGs['logM']), np.power(10, MFs_ETGs['log_phi_original']), color = 'red', marker=' ', linestyle='-', mec='none', markerfacecolor='red', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    ax.fill_between(np.power(10, MFs_ETGs['logM']), np.power(10, np.array(MFs_ETGs['log_phi_deconvolved']) - (np.array(MFs_ETGs['log_phi_original_post_systematic']) - np.array(MFs_ETGs['log_phi_original']))) , np.power(10, np.array(MFs_ETGs['log_phi_deconvolved']) + (np.array(MFs_ETGs['log_phi_original']) - np.array(MFs_ETGs['log_phi_original_neg_systematic'])) ), alpha=0.3, edgecolor='#FF6666', facecolor='#FF6666', linewidth=0.1)
    ax.plot(np.power(10, MFs_ETGs['logM']), np.power(10, MFs_ETGs['log_phi_deconvolved']), color = '#FF6666', marker=' ', linestyle='-', mec='none', markerfacecolor='#006666', markersize='1.0', linewidth = 0.5, zorder = 3) #marker="None"

    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=False) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    ax.set_xlim(1e7, 1e12)
    ax.set_ylim(1e-6, 1e0)

    #axis scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    #Labels
    ax.set_xlabel(xlabel, fontsize = '11')
    #ax.set_ylabel('$\\phi({\\rm M}_{x})$ $\\rm [dex^{-1}Mpc^{-3}]$' , fontsize = '11')
    #Ticks
    #To add minor ticks in log scale y-axis
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=7)
    ax.yaxis.set_minor_locator(locmin)
    #ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

    #To add minor ticks in log scale x-axis
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    locmin = matplotlib.ticker.LogLocator(base=10, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=6)
    ax.xaxis.set_minor_locator(locmin)

    ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')

    #################################################################################
    #title
    #titlefig = fig.suptitle('Using $\log(\\rm M_{\\ast})-\log (\\rm V_{max})$ and $\log(\\rm M_{\\ast})-\log (\\rm V_{peak})$ correlation for ETGs and LTGs', fontsize='10', fontweight='light', **font_type)
    #titlefig.set_y(0.88) #Where to place the title
    fig.subplots_adjust(top=0.92) #Adjust subplots with respect title

    #configuring subplots
    fig.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=0.0, hspace=0.0)
    #fig.set_tight_layout(True) !To automatically leave the subplots symmetric
    #plt.draw()

    plt.savefig(output_path_figure + name_figure, Transparent=True)
    plt.show()

##################################################################################
def MENU():
    os.system('clear')
    print("This code is based on Rodriguez-Puebla+19 results and allows you to compute the following:")
    print("(1): Average MHj (<MHj> or <logMHj>) or RHj (<RHj> or <logRHj>) as a function of stellar mass.")
    print("(2): Percentiles of MHj or RHj as a function of stellar mass.")
    print("(3): Conditional distributions P(MHj|Ms) as a function of MHj or RHj in a stellar mass bin (weighted).")
    print("(4): Intrinsic mass functions (stars, atomic and molecular hydrogen, cold gas or baryonic).")
    print("(5): EXIT\n")

    print("-----------------------------------------------------------------------------------------------------------")
    print("Notes:")
    print("(i) MHj is atomic or molecular Hydrogen, where j=I or j=2")
    print("(ii) RHj = MHj / Ms")
    print("-----------------------------------------------------------------------------------------------------------")
    print("\n")

    what_to_do = input(" What would you like to do? (choose the number in parenthesis corresponding to the option you want) ")

    if int(what_to_do) == 1:
        Averages()
    elif int(what_to_do) == 2:
        percentiles_MHj()
    elif int(what_to_do) == 3:
        conditional_distributions()
    elif int(what_to_do) == 4:
        intrinsic_mass_functions()
    elif int(what_to_do) == 5:
        print("Exit!")

##################################################################################
def Averages():
    os.system('clear')
    print("Average MHj (<MHj> or <logMHj>) or RHj (<RHj> or <logRHj>) as a function of stellar mass:\n")
    flag_MHj = input("What hydrogen component would you like to study? --> MHI = 1 ; MH_2 = 2: ")
    flag_mass = input("Would you like to study RHj or MHj? --> RHj = 1 ; MHj = 2: ")
    flag_space = input("Would you like to compute averages in linear or log space? --> linear = 1 ; log = 2: ")

    if float(flag_space) == 1.0:
        ave_RHj_or_MHj_vs_Ms(flag_mass=float(flag_mass),flag_MHj=float(flag_MHj))
    elif float(flag_space) == 2.0:
        ave_logRHj_or_logMHj_vs_Ms(flag_mass=float(flag_mass),flag_MHj=float(flag_MHj))

    os.system('clear')
    flag = input("Would you like to finish (press 1) or would you like to return to main menu (press 2)?")

    if int(flag) == 1:
        print("Exit!")
    elif int(flag) == 2:
        MENU()

##################################################################################
def percentiles_MHj():
    os.system('clear')
    print("Percentiles of MHj or RHj as a function of stellar mass.:\n")

    flag_MHj = input("What hydrogen component would you like to study? --> MHI = 1 ; MH_2 = 2: ")
    p1 = input("1st percentile (example -> 0.16): ")
    p2 = input("2nd percentile (example -> 0.50): ")
    p3 = input("3rd percentile (example -> 0.84): ")
    flag_res = input("Would you like to study RHj or MHj? --> RHj = 1 ; MHj = 2: ")

    percentiles(flag_res=float(flag_res), flag_MHj=float(flag_MHj), per1=float(p1), per2=float(p2), per3=float(p3))
    os.system('clear')
    flag = input("Would you like to finish (press 1) or would you like to return to main menu (press 2)?")

    if int(flag) == 1:
        print("Exit!")
    elif int(flag) == 2:
        MENU()

##################################################################################
def conditional_distributions():
    os.system('clear')
    print("Conditional distributions P(MHj|Ms) as a function of MHj or RHj in a stellar mass bin (weighted):\n")

    flag_MHj = input("What hydrogen component would you like to study? --> MHI = 1 ; MH_2 = 2: ")

    print("\n")
    print("What stellar mass bin would you like to study?: ")
    logMs_ini = input("Please insert the bin lower value (example -> logMs=7): ")
    logMs_end = input("Please insert the bin higher value (example -> logMs=12): ")

    print("\n")
    print("Would you like to plot P(MHj|Ms) as a function of MHj or RHj?")
    flag_mass = input("RHj = 1 ; MHj = 2: ")
    P_MHj_RHj(logMs_ini=float(logMs_ini), logMs_end=float(logMs_end), flag_MHj=float(flag_MHj), flag_mass=float(flag_mass))

    os.system('clear')
    flag = input("Would you like to finish (press 1) or would you like to return to main menu (press 2)?")

    if int(flag) == 1:
        print("Exit!")
    elif int(flag) == 2:
        MENU()

##################################################################################
def intrinsic_mass_functions():
    os.system('clear')
    print("Intrinsic mass functions:\n")
    print("Here you can plot and visualize stars, HI, H_2, cold gas and baryon intrinsic mass functions")
    print("for all galaxies along early and late populations.\n")

    flag_M = input("Which mass component would you like to study?\nGalaxy Stellar Mass Function = 1\nGalaxy HI Mass Function = 2\nGalaxy H2 Mass Function = 3\nGalaxy Gas Mass Function = 4\nGalaxy Baryon Mass Function = 5\n")
    Int_MFs(flag_M=float(flag_M))

    os.system('clear')
    flag = input("Would you like to finish (press 1) or would you like to return to main menu (press 2)?")

    if int(flag) == 1:
        print("Exit!")
    elif int(flag) == 2:
        MENU()

##################################################################################

MENU()
#Int_MFs(flag_M=2.0)
#ave_logRHj_or_logMHj_vs_Ms(flag_mass=2,flag_MHj=2)
#P_MHj_RHj(logMs_ini=10.5, logMs_end=11.35, flag_MHj=2, flag_mass=1)
#percentiles(flag_res=2, flag_MHj=2, per1=0.16, per2=0.50, per3=0.84)
#ave_RHj_or_MHj_vs_Ms(flag_mass=2, flag_MHj=2)
