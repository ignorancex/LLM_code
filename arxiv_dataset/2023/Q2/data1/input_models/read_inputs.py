#! /usr/bin/env python

'''read SN code-comparison input model files

To produce example plots based on files in https://github.com/sn-rad-trans/data1/input_models/ simply run:

python read_inputs.py

in the directory containing the model files (or set --path2data option)
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pdb import set_trace as stop

### ensure Python2 (2.6 or 2.7) and Python3 compatibility
if sys.version_info.major == 2:
    input = raw_input # input() to mean raw_input() when running Python2

### constants
MSUN = 1.989e+33     # solar mass (g)

###############################################################################

def read_snia_toy_model(file):

    '''read in snia_toyXX.dat file
    '''

    print('INFO - reading file ' + file)

    with open(file, 'r') as f:
        
        ### read in header
        okhdr = 0
        while okhdr == 0:
            line = f.readline()
            if 'nzones' in line:
                nzones = int(line.split()[3])
            elif 'time at which' in line:
                time = float(line.split()[-2])
            elif 'IMEs' in line:
                idx0 = line.index('IMEs =') + 7
                idx1 = line.index(' with relative fractions')
                imes = line[idx0:idx1].lower().split(', ') # 'Ca, S, Si' -> ['ca', 's', 'si']
            elif '#(1)' in line:
                okhdr = 1

        ### output arrays
        out = {}
        out['nzones'] = nzones
        out['time'] = time
        out['imes'] = imes
        out['vel'] = np.zeros(nzones)
        out['dmass'] = np.zeros(nzones)
        out['mass'] = np.zeros(nzones)
        out['ige_0'] = np.zeros(nzones)
        out['ni56_0'] = np.zeros(nzones)
        out['ime'] = np.zeros(nzones)
        out['ti'] = np.zeros(nzones)
        out['c+o'] = np.zeros(nzones)
        out['rad'] = np.zeros(nzones)
        out['dens'] = np.zeros(nzones)
        out['temp'] = np.zeros(nzones)
        out['ni56'] = np.zeros(nzones)
        out['ni'] = np.zeros(nzones)
        out['co'] = np.zeros(nzones)
        out['fe'] = np.zeros(nzones)
        for ime in imes:
            out[ime] = np.zeros(nzones)
        out['o'] = np.zeros(nzones)
        out['c'] = np.zeros(nzones)

        ### read numerical data
        for i in range(nzones):
            line = f.readline()
            split_line = line.split()
            out['vel'][i] = float(split_line[1])
            out['dmass'][i] = float(split_line[2])
            out['mass'][i] = float(split_line[3])
            out['ige_0'][i] = float(split_line[4])
            out['ni56_0'][i] = float(split_line[5])
            out['ime'][i] = float(split_line[6])
            out['ti'][i] = float(split_line[7])
            out['c+o'][i] = float(split_line[8])
            out['rad'][i] = float(split_line[9])
            out['dens'][i] = float(split_line[10])
            out['temp'][i] = float(split_line[11])
            out['ni56'][i] = float(split_line[12])
            out['ni'][i] = float(split_line[13])
            out['co'][i] = float(split_line[14])
            out['fe'][i] = float(split_line[15])
            for iime, ime in enumerate(imes):
                out[ime][i] = float(split_line[16 + iime])
            out['o'][i] = float(split_line[-2])
            out['c'][i] = float(split_line[-1])
                                    
    out['units'] = 'time: days\nvel: km/s\ndmass: msun\nmass: msun\nrad: cm\ndens: g/cm^3\ntemp: K'
    
    return out
    
###############################################################################

def read_ddc_model(file):

    '''read in DDC_*d file

    UPDATE 2022-03-17: mass fractions at t=t0 are no longer present in input files
                       (but we compute 56Ni mass fraction at t=0)
    '''

    THALF_56NI = 6.0749  # 56Ni->56Co decay half-life (days) used in DDC model runs
    decay_const_ni56 = np.log(2) / THALF_56NI
    
    print('INFO - reading file ' + file)

    with open(file, 'r') as f:
        
        ### read in header
        okhdr = 0
        while okhdr == 0:
            line = f.readline()
            if 'TIME' in line:
                time = float(line.split()[-2])
            # elif 's PAST EXPLOSION' in line:
            #     t0 = float(line.split()[4])
            elif 'elemental' in line:
                idx_elem = line.split()[1][1:-1]
                idxelem0 = int(idx_elem[:idx_elem.index(')')]) - 1
                idxelem1 = int(idx_elem[idx_elem.index('(')+1:])
            elif 'isotopic' in line and 'days since explosion' in line:
                idx_iso = line.split()[1][1:-1]
                idxiso0 = int(idx_iso[:idx_iso.index(')')]) - 1
                idxiso1 = int(idx_iso[idx_iso.index('(')+1:])
            # elif 'isotopic' in line and 'seconds since explosion' in line:
            #     idx_iso = line.split()[1][1:-1]
            #     idxiso00 = int(idx_iso[:idx_iso.index(')')]) - 1
            #     idxiso01 = int(idx_iso[idx_iso.index('(')+1:])
            elif '#vel[km/s]' in line:
                elems = line.split()[idxelem0:idxelem1]
                isos = line.split()[idxiso0:idxiso1]
                # iso0s = line.split()[idxiso00:idxiso01]
                line = f.readline()
                okhdr = 1

        ### output arrays
        vel = []
        rad = []
        dvol = []
        dens = []
        dmass = []
        temp = []
        xelem = {}
        xiso = {}
        # xiso0 = {}
        for elem in elems:
            xelem[elem] = []
        for iso in isos:
            xiso[iso] = []
        # for iso in iso0s:
        #     xiso0[iso] = []
        
        ### read numerical data
        while 1:
            line = f.readline()
            # to handle EOF
            if not line:
                break
            else:
                split_line = line.split()
                vel.append(float(split_line[0]))
                rad.append(float(split_line[1]))
                dvol.append(float(split_line[2]))
                dens.append(float(split_line[3]))
                dmass.append(float(split_line[4]))
                temp.append(float(split_line[5]))
                for ielem, elem in enumerate(elems):
                    xelem[elem].append(float(split_line[idxelem0 + ielem]))
                for iiso, iso in enumerate(isos):
                    xiso[iso].append(float(split_line[idxiso0 + iiso]))
                # for iiso, iso in enumerate(iso0s):
                #     xiso0[iso].append(float(split_line[idxiso00 + iiso]))

    # output
    out = {}
    out['time'] = time
    # out['t0'] = t0
    out['elem'] = elems
    out['iso'] = isos
    # out['iso_0'] = iso0s
    out['vel'] = np.array(vel)
    out['rad'] = np.array(rad)
    out['dvol'] = np.array(dvol)
    out['dens'] = np.array(dens)
    out['dmass'] = np.array(dmass)
    out['temp'] = np.array(temp)
    for elem in elems:
        out[elem] = np.array(xelem[elem])
    for iso in isos:
        out[iso] = np.array(xiso[iso])
        if iso == 'ni56':
            # 56Ni at t=0 inferred from 56Ni(t)
            # note that decay_const_ni56 is in /days and time is in days
            out['ni56_0'] = out['ni56'] * np.exp(decay_const_ni56 * time)
    # for iso in iso0s:
    #     out[iso] = np.array(xiso0[iso])
    
    # out['units'] = 'time: days\nt0: seconds\nvel: km/s\nrad: cm\ndvol: cm^3\ndens: g/cm^3\ndmass: g\ntemp: K'
    out['units'] = 'time: days\nvel: km/s\nrad: cm\ndvol: cm^3\ndens: g/cm^3\ndmass: g\ntemp: K'

    return out

###############################################################################

if __name__ == '__main__':

    import argparse
    import matplotlib.ticker as ticker

    # parse command-line options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('--path2data', default='.', type=str, help='path to directory containing the model files')
    args = parser.parse_args()

    toy01 = read_snia_toy_model(args.path2data +'/' + 'snia_toy01_2d.dat')
    toy06 = read_snia_toy_model(args.path2data +'/' + 'snia_toy06_2d.dat')
    ddc10 = read_ddc_model(args.path2data +'/' + 'DDC10_0.976d.dat')
    ddc25 = read_ddc_model(args.path2data +'/' + 'DDC25_1.300d.dat')
    model_dicts = {'toy01':toy01, 'toy06':toy06, 'DDC10':ddc10, 'DDC25':ddc25}

    
    ###########################
    # Table: summary
    ###########################

    with open('models_summary.tex', 'w') as ftab:

        ftab.write('\\begin{table*}\n')
        ftab.write('\caption{Summary of ejecta conditions. We give the composition for representative species (the $^{56}$Ni mass is given prior to any decay).}\n')
        ftab.write('\label{tab:models}\n')
        ftab.write('\\begin{center}\n')
        ftab.write('\\begin{tabular}{lccccccccc}\n')
        ftab.write('\hline\n')
        ftab.write('Model & $M_{\\rm ej}$ & $E_{\\rm kin}$    & M($^{56}$Ni)$_{t=0}$ & M(Fe)   & M(Ca)   & M(S)    & M(Si)   & M(O)     & M(C)    \\\\\n')
        ftab.write('      & (\msun)      & ($10^{51}$\,erg) & (\msun)              & (\msun) & (\msun) & (\msun) & (\msun) & (\msun)  & (\msun) \\\\\n')
        ftab.write('\hline\n')
        models_ordered = ['toy06', 'toy01', 'DDC10', 'DDC25']
        for model in models_ordered:
            dic = model_dicts[model]
            if model[:3] == 'toy':
                dmass_msun = dic['dmass']
                dmass_cgs = dic['dmass'] * MSUN
            elif  model[:3] == 'DDC':
                dmass_msun = dic['dmass'] / MSUN
                dmass_cgs = dic['dmass']
            vel_cgs = dic['vel'] * 1e5
            mej = np.sum(dmass_msun)
            ekin = 0.5 * np.sum(dmass_cgs * vel_cgs**2) # vel in cm/s
            mc  = np.sum(dmass_msun * dic['c'])
            mo  = np.sum(dmass_msun * dic['o'])
            msi = np.sum(dmass_msun * dic['si'])
            ms  = np.sum(dmass_msun * dic['s'])
            mca = np.sum(dmass_msun * dic['ca'])
            mfe = np.sum(dmass_msun * dic['fe'])
            mni560 = np.sum(dmass_msun * dic['ni56_0'])
            ftab.write('{:5s} & {:.2f} & {:.2f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\\n'.format(model, mej, ekin/1e51, mni560, mfe, mca, ms, msi, mo, mc))
        ftab.write('\hline\n')
        ftab.write('\end{tabular}\n')
        ftab.write('\end{center}\n')
        ftab.write('\end{table*}\n')

    print('INFO - created output file models_summary.tex')

        
    ###########################
    # Figure: density profiles
    ###########################

    # compute density at reference time
    tref = 1.0 # days since explosion
    toy01_densscl = (toy01['time'] / tref)**3 
    toy06_densscl = (toy06['time'] / tref)**3 
    ddc10_densscl = (ddc10['time'] / tref)**3 
    ddc25_densscl = (ddc25['time'] / tref)**3 

    # start figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman', size=10) # define absolute font size here...
    plt.rc('xtick', labelsize='small') # ... relative font sizes thereafter
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

    fig = plt.figure(figsize=(3.32, 2.25))
    fig.subplots_adjust(left=.16, bottom=.17, right=.975, top=.99)
    ax = fig.add_subplot(111)

    # axis parameters
    ax.set_xlim(0, 4.2e4)
    ax.set_ylim(-16, -7.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.5))
    ax.set_xlabel(r'Velocity [km s$^{-1}$]')
    ax.set_ylabel(r'$\log_{10}$ Density [g cm$^{-3}$]')

    # plot
    colorcycle = ['#4477AA', '#CC6677', '#DDCC77', '#117733']
    ax.plot(toy06['vel'], np.log10(toy06['dens'] * toy06_densscl), color=colorcycle[0], ls='-' , lw=1.0, label='toy06', zorder=10)
    ax.plot(toy01['vel'], np.log10(toy01['dens'] * toy01_densscl), color=colorcycle[1], ls=':' , lw=2.0, label='toy01', zorder=10)
    ax.plot(ddc10['vel'], np.log10(ddc10['dens'] * ddc10_densscl), color=colorcycle[2], ls='-' , lw=2.0, label='DDC10', zorder=1)
    ax.plot(ddc25['vel'], np.log10(ddc25['dens'] * ddc25_densscl), color=colorcycle[3], ls=':', lw=1.0, label='DDC25', zorder=1)

    # legend and labels
    ax.legend(frameon=False, fontsize='small')
    ax.text(.08, .1, '$t = {:d}$\,d'.format(int(tref)), transform=ax.transAxes)
    
    plt.savefig('density_profile.pdf')
    print('INFO - created output file density_profile.pdf')


    #########################################
    # global params for composition profiles
    #########################################

    # x- and y-lim
    xlim = (0., 4e4)
    ylim = (-5, .99)
    
    # color scheme
    # see https://personal.sron.nl/~pault/#sec:qualitative
    col_ni56 = '#4477AA'
    col_fe   = '#EE6677'
    col_ca   = '#228833'
    col_s    = '#CCBB44'
    col_si   = '#66CCEE'
    col_o    = '#AA3377'
    col_c    = '#BBBBBB'
    col_ime  = 'k' ; ls_ime = ':'
    col_ige  = 'k' ; ls_ige = '--'

    # legend parameters
    legend_dict = dict(
        loc = 'upper left',
        frameon = False,
        borderpad = 0.2,
        ncol = 4,
        columnspacing = 1.5,
        labelspacing = 0.2,
        fontsize = 7.5,
        handlelength = 1.5,
        handletextpad=0.6
    )
    
    
    #####################################
    # Figure: toy01 composition profiles
    #####################################

    # start figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman', size=10) # define absolute font size here...
    plt.rc('xtick', labelsize='small') # ... relative font sizes thereafter
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

    fig = plt.figure(figsize=(3.32, 2.62))
    fig.subplots_adjust(left=.12, bottom=.15, right=.95, top=.87)
    ax = fig.add_subplot(111)

    # axis parameters
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.5))
    ax.set_xlabel(r'Velocity [km s$^{-1}$]')
    ax.set_ylabel(r'$\log_{10}$ Mass Fraction')

    # plot   
    idx = np.where(toy01['ni56_0'] == 0.)
    toy01['ni56_0'][idx] = 1e-99
    idx = np.where(toy01['ca'] == 0.)
    toy01['ca'][idx] = 1e-99
    idx = np.where(toy01['s'] == 0.)
    toy01['s'][idx] = 1e-99
    idx = np.where(toy01['si'] == 0.)
    toy01['si'][idx] = 1e-99
    x_ime = toy01['ca'] + toy01['s'] + toy01['si'] 
    ax.plot(toy01['vel'], np.log10(toy01['ni56_0']), color=col_ni56, ls='-' , lw=1.0, label='$^{56}$Ni$_0$')
    ax.plot(toy01['vel'], np.log10(toy01['ca']), color=col_ca, ls='-' , lw=1.0, label='Ca')
    ax.plot(toy01['vel'], np.log10(toy01['s']), color=col_s, ls='-' , lw=1.0, label='S')
    ax.plot(toy01['vel'], np.log10(toy01['si']), color=col_si, ls='-' , lw=1.0, label='Si')
    lime, = ax.plot(toy01['vel'], np.log10(x_ime), color=col_ime, ls=ls_ime, lw=1.0, label='')

    # upper X-axis (Lagrangian mass)
    mlagrangian = toy01['mass']
    massvals = [.2, .5, .8, .95, .99]
    xticklabels = [str(m) for m in massvals]
    f = interpolate.interp1d(mlagrangian, toy01['vel'])
    v = f(massvals) # returns interpolated velocity locations of mass coordinates
    ax2 = ax.twiny()
    ax2.minorticks_off()
    ax2.set_xlabel('Lagrangian Mass [M$_\odot$]')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(v)
    ax2.set_xticklabels(xticklabels)
    
    # legend and labels
    leg = ax.legend(**legend_dict)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    leg2 = ax.legend([lime], ['IME'], frameon=False, fontsize=7.5,
                     handlelength=1.3, handletextpad=0.4,
                     loc='center right')
    ax.add_artist(leg)

    ax.text(.98, .96, 'toy01', va='top', ha='right', transform=ax.transAxes)
    ax.text(.98, .04, '$t = 2$\,d', fontsize=8, va='bottom', ha='right', transform=ax.transAxes)

    plt.savefig('composition_profile_toy01.pdf')
    print('INFO - created output file composition_profile_toy01.pdf')


    #####################################
    # Figure: toy06 composition profiles
    #####################################

    # start figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman', size=10) # define absolute font size here...
    plt.rc('xtick', labelsize='small') # ... relative font sizes thereafter
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

    fig = plt.figure(figsize=(3.32, 2.62))
    fig.subplots_adjust(left=.12, bottom=.15, right=.95, top=.87)
    ax = fig.add_subplot(111)

    # axis parameters
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.5))
    ax.set_xlabel(r'Velocity [km s$^{-1}$]')
    ax.set_ylabel(r'$\log_{10}$ Mass Fraction')

    # plot   
    idx = np.where(toy06['ni56_0'] == 0.)
    toy06['ni56_0'][idx] = 1e-99
    idx = np.where(toy06['ca'] == 0.)
    toy06['ca'][idx] = 1e-99
    idx = np.where(toy06['s'] == 0.)
    toy06['s'][idx] = 1e-99
    idx = np.where(toy06['si'] == 0.)
    toy06['si'][idx] = 1e-99
    x_ime = toy06['ca'] + toy06['s'] + toy06['si'] 
    ax.plot(toy06['vel'], np.log10(toy06['ni56_0']), color=col_ni56, ls='-' , lw=1.0, label='$^{56}$Ni$_0$')
    ax.plot(toy06['vel'], np.log10(toy06['ca']), color=col_ca, ls='-' , lw=1.0, label='Ca')
    ax.plot(toy06['vel'], np.log10(toy06['s']), color=col_s, ls='-' , lw=1.0, label='S')
    ax.plot(toy06['vel'], np.log10(toy06['si']), color=col_si, ls='-' , lw=1.0, label='Si')
    lime, = ax.plot(toy06['vel'], np.log10(x_ime), color=col_ime, ls=ls_ime, lw=1.0, label='')

    # upper X-axis (Lagrangian mass)
    mlagrangian = toy06['mass']
    massvals = [.2, .5, .8, .95, .99]
    xticklabels = [str(m) for m in massvals]
    f = interpolate.interp1d(mlagrangian, toy06['vel'])
    v = f(massvals) # returns interpolated velocity locations of mass coordinates
    ax2 = ax.twiny()
    ax2.minorticks_off()
    ax2.set_xlabel('Lagrangian Mass [M$_\odot$]')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(v)
    ax2.set_xticklabels(xticklabels)
    
    # legend and labels
    leg = ax.legend(**legend_dict)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)
    
    leg2 = ax.legend([lime], ['IME'], frameon=False, fontsize=7.5,
                     handlelength=1.3, handletextpad=0.4,
                     loc='center right')
    ax.add_artist(leg)

    ax.text(.98, .96, 'toy06', va='top', ha='right', transform=ax.transAxes)
    ax.text(.98, .04, '$t = 2$\,d', fontsize=8, va='bottom', ha='right', transform=ax.transAxes)
 
    plt.savefig('composition_profile_toy06.pdf')
    print('INFO - created output file composition_profile_toy06.pdf')


    #####################################
    # Figure: DDC10 composition profiles
    #####################################

    # start figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman', size=10) # define absolute font size here...
    plt.rc('xtick', labelsize='small') # ... relative font sizes thereafter
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

    fig = plt.figure(figsize=(3.32, 2.62))
    fig.subplots_adjust(left=.12, bottom=.15, right=.95, top=.87)
    ax = fig.add_subplot(111)

    # axis parameters
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.5))
    ax.set_xlabel(r'Velocity [km s$^{-1}$]')
    ax.set_ylabel(r'$\log_{10}$ Mass Fraction')

    # plot
    iges = ['ni','co','fe','mn','cr','v','ti','sc']
    x_ige = np.zeros(ddc10['vel'].size)
    for ige in iges:
        x_ige += ddc10[ige]
    
    imes = ['ca','k','ar','cl','s','si','al','mg','na']
    x_ime = np.zeros(ddc10['vel'].size)
    for ime in imes:
        x_ime += ddc10[ime]
    
    ax.plot(ddc10['vel'], np.log10(ddc10['ni56_0']), color=col_ni56, ls='-' , lw=1.0, label='$^{56}$Ni$_0$')
    ax.plot(ddc10['vel'], np.log10(ddc10['fe']), color=col_fe, ls='-' , lw=1.0, label='Fe')
    ax.plot(ddc10['vel'], np.log10(ddc10['ca']), color=col_ca, ls='-' , lw=1.0, label='Ca')
    ax.plot(ddc10['vel'], np.log10(ddc10['s']), color=col_s, ls='-' , lw=1.0, label='S')
    ax.plot(ddc10['vel'], np.log10(ddc10['si']), color=col_si, ls='-' , lw=1.0, label='Si')
    ax.plot(ddc10['vel'], np.log10(ddc10['o']), color=col_o, ls='-' , lw=1.0, label='O')
    ax.plot(ddc10['vel'], np.log10(ddc10['c']), color=col_c, ls='-' , lw=1.0, label='C')
    lige, = ax.plot(ddc10['vel'], np.log10(x_ige), color=col_ige, ls=ls_ige , lw=1.0, label='')
    lime, = ax.plot(ddc10['vel'], np.log10(x_ime), color=col_ime, ls=ls_ime , lw=1.0, label='')

    # upper X-axis (Lagrangian mass)
    mlagrangian = np.cumsum(ddc10['dmass']) / MSUN
    massvals = [.2, .5, .9, 1.2, 1.38]
    xticklabels = [str(m) for m in massvals]
    f = interpolate.interp1d(mlagrangian, ddc10['vel'])
    v = f(massvals) # returns interpolated velocity locations of mass coordinates
    ax2 = ax.twiny()
    ax2.minorticks_off()
    ax2.set_xlabel('Lagrangian Mass [M$_\odot$]')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(v)
    ax2.set_xticklabels(xticklabels)
    
    # legend and labels
    leg = ax.legend(**legend_dict)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    leg2 = ax.legend([lige, lime], ['IGE', 'IME'], frameon=False, fontsize=7.5,
                     handlelength=1.3, handletextpad=0.4,
                     loc='lower right', bbox_to_anchor=(0,.5,1,1))
    ax.add_artist(leg)

    ax.text(.98, .96, 'DDC10', va='top', ha='right', transform=ax.transAxes)
    ax.text(.98, .04, '$t = 0.976$\,d', fontsize=8, va='bottom', ha='right', transform=ax.transAxes)

    plt.savefig('composition_profile_ddc10.pdf')
    print('INFO - created output file composition_profile_ddc10.pdf')


    #####################################
    # Figure: DDC25 composition profiles
    #####################################

    # start figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman', size=10) # define absolute font size here...
    plt.rc('xtick', labelsize='small') # ... relative font sizes thereafter
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

    fig = plt.figure(figsize=(3.32, 2.62))
    fig.subplots_adjust(left=.12, bottom=.15, right=.95, top=.87)
    ax = fig.add_subplot(111)

    # axis parameters
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.5))
    ax.set_xlabel(r'Velocity [km s$^{-1}$]')
    ax.set_ylabel(r'$\log_{10}$ Mass Fraction')

    # plot   
    iges = ['ni','co','fe','mn','cr','v','ti','sc']
    x_ige = np.zeros(ddc25['vel'].size)
    for ige in iges:
        x_ige += ddc25[ige]
    
    imes = ['ca','k','ar','cl','s','si','al','mg','na']
    x_ime = np.zeros(ddc25['vel'].size)
    for ime in imes:
        x_ime += ddc25[ime]
        
    ax.plot(ddc25['vel'], np.log10(ddc25['ni56_0']), color=col_ni56, ls='-' , lw=1.0, label='$^{56}$Ni$_0$')
    ax.plot(ddc25['vel'], np.log10(ddc25['fe']), color=col_fe, ls='-' , lw=1.0, label='Fe')
    ax.plot(ddc25['vel'], np.log10(ddc25['ca']), color=col_ca, ls='-' , lw=1.0, label='Ca')
    ax.plot(ddc25['vel'], np.log10(ddc25['s']), color=col_s, ls='-' , lw=1.0, label='S')
    ax.plot(ddc25['vel'], np.log10(ddc25['si']), color=col_si, ls='-' , lw=1.0, label='Si')
    ax.plot(ddc25['vel'], np.log10(ddc25['o']), color=col_o, ls='-' , lw=1.0, label='O')
    ax.plot(ddc25['vel'], np.log10(ddc25['c']), color=col_c, ls='-' , lw=1.0, label='C')
    lige, = ax.plot(ddc25['vel'], np.log10(x_ige), color=col_ige, ls=ls_ige , lw=1.0, label='')
    lime, = ax.plot(ddc25['vel'], np.log10(x_ime), color=col_ime, ls=ls_ime , lw=1.0, label='')

    # upper X-axis (Lagrangian mass)
    mlagrangian = np.cumsum(ddc25['dmass']) / MSUN
    massvals = [.2, .5, .9, 1.2, 1.38]
    xticklabels = [str(m) for m in massvals]
    f = interpolate.interp1d(mlagrangian, ddc25['vel'])
    v = f(massvals) # returns interpolated velocity locations of mass coordinates
    ax2 = ax.twiny()
    ax2.minorticks_off()
    ax2.set_xlabel('Lagrangian Mass [M$_\odot$]')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(v)
    ax2.set_xticklabels(xticklabels)
    
    # legend and labels
    leg = ax.legend(**legend_dict)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    leg2 = ax.legend([lige, lime], ['IGE', 'IME'], frameon=False, fontsize=7.5,
                     handlelength=1.3, handletextpad=0.4,
                     loc='lower right', bbox_to_anchor=(0,.5,1,1))
    ax.add_artist(leg)

    ax.text(.98, .96, 'DDC25', va='top', ha='right', transform=ax.transAxes)
    ax.text(.98, .04, '$t = 1.3$\,d', fontsize=8, va='bottom', ha='right', transform=ax.transAxes)

    plt.savefig('composition_profile_ddc25.pdf')
    print('INFO - created output file composition_profile_ddc25.pdf')
