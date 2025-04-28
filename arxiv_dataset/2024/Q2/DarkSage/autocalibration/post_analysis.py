# Adam Stevens, 2022
# Post-analysis of output of PSO with Dark Sage to check fidelity etc

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/adam/Dirty-AstroPy')
from galprops import galread as gr
from galprops import galcalc as gc
from galprops import galplot as gp

import warnings
warnings.filterwarnings("ignore")

fsize = 26
c_edge, c_face = 'k', 'w'
transparent = True
matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'lines.markeredgewidth': 1.0, 'errorbar.capsize': 4.0, 'xtick.top': True, 'ytick.right': True, "text.latex.preamble": [r'\usepackage{amsmath}',], 'axes.edgecolor': c_edge, 'xtick.color': c_edge, 'ytick.color': c_edge, 'figure.facecolor': c_face, 'text.color': c_edge, 'axes.labelcolor': c_edge})

base_dir = '/Users/adam/DarkSage/autocalibration/aux_out/MTNG_mini/OzSTAR_26/'
track_dir = base_dir + 'tracks/'
dump_file = base_dir + 'DS_calibrate_dump_26.txt'

SMF_obs_file = 'GAMA_SMF_highres.csv'
HIMF_obs_file = 'Jones2018_HIMF.dat'
#SFRD_obs_file = 'Driver_SFRD.dat'
SFRD_obs_file = 'CSFH_DSILVA+23_Ver_Final.csv'

# lists that will contain the x and y data for the observations and Dark Sage runs that were used to compute the goodness of fit
SMF_x_obs, SMF_y_obs, SMF_y_mod = [], [], []
HIMF_x_obs, HIMF_y_obs, HIMF_y_mod = [], [], []
CSFH_x_obs, CSFH_y_obs, CSFH_y_mod = [], [], []

# lists of parameter values and goodness of fit
params = []
Nparams = 3
Nparticles = 16


f = open(dump_file, 'r')
lines = f.readlines()
on_a_data_line = False # tracks if the current line in the dump is data of interest
last_line = False
obs = ''
line_start = ''

SMF_obs_data = np.loadtxt(SMF_obs_file, skiprows=3, delimiter=' ')
HIMF_obs_data = np.loadtxt(HIMF_obs_file, skiprows=17, delimiter=' ', usecols=(0,1,2))
#CSFH_obs_data = np.loadtxt(SFRD_obs_file, skiprows=3, delimiter=' ', usecols=(1,2,3,5,6,7))
CSFH_obs_data = np.loadtxt(SFRD_obs_file, skiprows=1, delimiter=' ')

MultipleLocator = matplotlib.ticker.MultipleLocator

for l, line in enumerate(lines):
#    print('doing line', l, on_a_data_line, last_line, obs, line_start)

    # a line that specifies a data file tells us which constraint the next data belong to
    if SMF_obs_file in line:
        obs = 'SMF'
    elif HIMF_obs_file in line:
        obs = 'HIMF'
    elif SFRD_obs_file in line:
        obs = 'CSFH'
    
    # catches an instance where an array of data we want starts
    if line[:5]=='obs x' or line[:5]=='obs y' or line[:5]=='mod y': 
        line_start = line[:5]
        on_a_data_line = True
        line = line[8:]
        data = np.array([], dtype=np.float32) # start appending data into an array
        
    if on_a_data_line:
        if line[-2:]==']\n':
            last_line = True
            line = line[:-2]
        else:
            last_line = False
            line = line[:-1]

        line_data = np.array(line.split())
        data = np.append(data, line_data.astype(np.float32))
            
        if last_line:
            
            if obs == 'SMF' and line_start == 'obs x':
                SMF_x_obs += [data]
            if obs == 'SMF' and line_start == 'obs y':
                SMF_y_obs += [data]
            if obs == 'SMF' and line_start == 'mod y':
                SMF_y_mod += [data]

            if obs == 'HIMF' and line_start == 'obs x':
                HIMF_x_obs += [data]
            if obs == 'HIMF' and line_start == 'obs y':
                HIMF_y_obs += [data]
            if obs == 'HIMF' and line_start == 'mod y':
                HIMF_y_mod += [data]

            if obs == 'CSFH' and line_start == 'obs x':
                CSFH_x_obs += [data]
            if obs == 'CSFH' and line_start == 'obs y':
                CSFH_y_obs += [data]
            if obs == 'CSFH' and line_start == 'mod y':
                CSFH_y_mod += [data]
            
            on_a_data_line = False
    else:
        last_line = False
        
    if 'Particle array' in line:
        if Nparams==4:
#            try:
            param_array = np.array([line[67:81], line[83:97], line[99:113], line[115:129], line[145:]], dtype=np.float64)
#            except ValueError:
#                try:
#                    param_array = np.array([line[67:80], line[82:95], line[97:110], line[126:]], dtype=np.float64)
#                except ValueError:
#                    try:
#                        param_array = np.array([line[67:79], line[81:93], line[95:107], line[123:]], dtype=np.float64)
#                    except ValueError:
#                        try:
#                            param_array = np.array([line[67:78], line[80:91], line[93:104], line[120:]], dtype=np.float64)
#                        except ValueError:
#                            param_array = np.array([line[67:77], line[79:89], line[91:101], line[117:]], dtype=np.float64)
        elif Nparams==3:
            try:
                param_array = np.array([line[67:81], line[83:97], line[99:113], line[129:]], dtype=np.float64)
            except ValueError:
                try:
                    param_array = np.array([line[67:80], line[82:95], line[97:110], line[126:]], dtype=np.float64)
                except ValueError:
                    try:
                        param_array = np.array([line[67:79], line[81:93], line[95:107], line[123:]], dtype=np.float64)
                    except ValueError:
                        try:
                            param_array = np.array([line[67:78], line[80:91], line[93:104], line[120:]], dtype=np.float64)
                        except ValueError:
                            param_array = np.array([line[67:77], line[79:89], line[91:101], line[117:]], dtype=np.float64)
        elif Nparams==2:
            try:
                param_array = np.array([line[67:81], line[83:97], line[113:]], dtype=np.float64)
            except ValueError:
                try:
                    param_array = np.array([line[67:80], line[82:95], line[111:]], dtype=np.float64)
                except ValueError:
                    try:
                        param_array = np.array([line[67:79], line[81:93], line[109:]], dtype=np.float64)
                    except ValueError:
                        try:
                            param_array = np.array([line[67:78], line[80:91], line[107:]], dtype=np.float64)
                        except ValueError:
                            param_array = np.array([line[67:77], line[79:89], line[105:]], dtype=np.float64)


        params += [param_array]


#print(len(SMF_x_obs), len(SMF_y_obs), len(SMF_y_mod))
#print(len(HIMF_x_obs), len(HIMF_y_obs), len(HIMF_y_mod))
#print(len(CSFH_x_obs), len(CSFH_y_obs), len(CSFH_y_mod))
#print(len(params))
params = np.array(params)

for p in range(Nparams+1): print('Min and max of param', p, np.min(params[:,p]), np.max(params[:,p]))

if Nparams<4:
    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
else:
    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False)

ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[0,2]

SMF_yerr = SMF_obs_data[:len(SMF_x_obs[0]),2]
# artificially raising size of errors on HIMF to make them visible and comparable in size to SMF
HIMF_yerr_up = np.log10((HIMF_obs_data[:,1]+HIMF_obs_data[:,2])/HIMF_obs_data[:,1])[-len(HIMF_x_obs[0])-1:-1]
HIMF_yerr_dn = np.log10(HIMF_obs_data[:,1]/(HIMF_obs_data[:,1]-HIMF_obs_data[:,2]))[-len(HIMF_x_obs[0])-1:-1]
CSFH_yerr = np.sum(CSFH_obs_data[:,-3:], axis=1)
CSFH_yerr_up = CSFH_obs_data[:,1]
CSFH_yerr_dn = CSFH_obs_data[:,2]

elw = 2
ec = '#ff8080'
ax1.errorbar(SMF_x_obs[0], SMF_y_obs[0], yerr=SMF_yerr, ecolor=ec, ls='None', elinewidth=elw, capthick=elw, label=r'Driver et al.~(2022)')
ax2.errorbar(HIMF_x_obs[0], HIMF_y_obs[0], yerr=np.row_stack((HIMF_yerr_dn,HIMF_yerr_up)), ecolor=ec, ls='None', elinewidth=elw, capthick=elw, label=r'Jones et al.~(2018)')
ax3.errorbar(CSFH_x_obs[0], CSFH_y_obs[0], yerr=np.row_stack((CSFH_yerr_dn,CSFH_yerr_up)), ecolor=ec, ls='None', elinewidth=elw, capthick=elw, label=r"D'Silva et al.~(2023)")

#Nloop = np.min([len(SMF_x_obs), len(HIMF_x_obs), len(CSFH_x_obs)])
#Nloop = len(SMF_x_obs)
Nloop = len(CSFH_x_obs)
chi2 = np.zeros(Nloop)
for i in range(Nloop):
    HIMF_yerr = HIMF_yerr_dn
    fup = (HIMF_y_mod[i] > HIMF_y_obs[i])
    HIMF_yerr[fup] = HIMF_yerr_up[fup]
    
    CSFH_yerr = CSFH_yerr_dn
    fup = (CSFH_y_mod[i] > CSFH_y_obs[i])
    CSFH_yerr[fup] = CSFH_yerr_up[fup]

    fSMF = np.ones(len(SMF_y_mod[i]), dtype=bool)# (SMF_y_mod[i] > -20)
    fHIMF = np.ones(len(HIMF_y_mod[i]), dtype=bool)#(HIMF_y_mod[i] > -20)
    fCSFH = np.ones(len(CSFH_y_mod[i]), dtype=bool)#(CSFH_y_mod[i] > -20)
    
    chi2_SMF = np.sum(((SMF_y_mod[i] - SMF_y_obs[i]) / SMF_yerr)[fSMF]**2) / (len(SMF_y_mod[i][fSMF]) - Nparams)
    chi2_HIMF = np.sum(((HIMF_y_mod[i] - HIMF_y_obs[i]) / HIMF_yerr)[fHIMF]**2) / (len(HIMF_y_mod[i][fHIMF]) - Nparams)
    chi2_CSFH = np.sum(((CSFH_y_mod[i] - CSFH_y_obs[i]) / CSFH_yerr)[fCSFH]**2) / (len(CSFH_y_mod[i][fCSFH]) - Nparams)
    chi2[i] = np.log10(chi2_SMF * chi2_HIMF * chi2_CSFH)
#    chi2[i] = np.log10(chi2_SMF)
#    chi2[i] = np.log10(chi2_CSFH)
    
#print('argsort chi2', np.argsort(chi2))
#print('argsort eval', np.argsort(params[:,Nparams]))

#print(np.sort(chi2))
cmap = plt.cm.viridis_r
#c = cmap((chi2-np.min(chi2)) / (np.max(chi2) - np.min(chi2)))

#param_sort = np.argsort(params[:,Nparams])
#param_sort = np.argsort(chi2)
#print('best-fitting params', params[param_sort[0],:Nparams])
#print('diff in argsort', np.argsort(params[:,Nparams]) - np.argsort(chi2))
#print('diff in chi2', chi2 - np.log10(params[:,Nparams]))# - np.mean(chi2 - np.log10(params[:,Nparams])))
#print('diff in sorted chi2', np.sort(chi2) - np.log10(np.sort(params[:,Nparams])))


best_fit = np.min(np.log10(params[:,Nparams]))
worst_fit = np.max(np.log10(params[:,Nparams]))
print('best_fit, worst_fit =', best_fit, worst_fit)
c = cmap((np.log10(params[:,Nparams]) - best_fit))# / ((worst_fit-0.5) - best_fit))
param_sort = np.argsort(params[:,Nparams])
print('best-fitting params', params[param_sort[0],:Nparams])

#best_fit = -1.0007519801041953
#worst_fit = 10.895294877017136

#best_fit = np.min(chi2)
#worst_fit = np.max(chi2)
#print('best_fit, worst_fit =', best_fit, worst_fit)
#c = cmap((chi2 - best_fit) / (worst_fit - best_fit))

for i in range(Nloop):
    assert(np.all(SMF_x_obs[i]==SMF_x_obs[0]))
    assert(np.all(HIMF_x_obs[i]==HIMF_x_obs[0]))
    assert(np.all(CSFH_x_obs[i]==CSFH_x_obs[0]))
    ax1.plot(SMF_x_obs[i], SMF_y_mod[i], '-', color=c[i], zorder=-params[i,Nparams], alpha=0.5)
    ax2.plot(HIMF_x_obs[i], HIMF_y_mod[i], '-', color=c[i], zorder=-params[i,Nparams], alpha=0.5)
    ax3.plot(CSFH_x_obs[i], CSFH_y_mod[i], '-', color=c[i], zorder=-params[i,Nparams], alpha=0.5)

ax1.axis([8.2,11.7,-4,-1.5])
ax2.axis([8.9,10.7,-4,-1.5])
ax3.axis([0,14,-3,-0.5])

ax1.set_xlabel(r'$\log_{10}(m_*~[{\rm M}_\odot])$')
ax1.set_ylabel(r'$\log_{10}(\Phi_*~[{\rm cMpc}^{-3}\,{\rm dex}^{-1}])$')

ax2.set_xlabel(r'$\log_{10}(m_{\rm H\,{\LARGE{\textsc i}}}~[{\rm M}_\odot])$')
ax2.set_ylabel(r'$\log_{10}(\Phi_{\rm H\,{\LARGE{\textsc i}}}~[{\rm cMpc}^{-3}\,{\rm dex}^{-1}])$')

ax3.set_xlabel(r'Look-back time [Gyr]')
ax3.set_ylabel(r'$\log_{10}(\bar{\rho}_{\rm SFR}~[{\rm M}_\odot\,{\rm yr}^{-1}\,{\rm cMpc}^{-3}])$')

ax4 = axs[1,0]
ax5 = axs[1,1]
ax6 = axs[1,2]

psr = param_sort[::-1]
ims = ax4.scatter(params[psr,0], params[psr,1], c=c[psr], alpha=0.5, cmap=cmap, vmin=best_fit, vmax=best_fit+1)#, zorder=-params[:,3])
ax4.set_xlabel(r'Black-hole radiative efficiency, $\epsilon$')
ax4.set_ylabel(r'Unstable-disc migration/heating ratio, $f_{\rm move}^{\rm gas}$')

if Nparams>=3:
    ax5.scatter(params[psr,0], params[psr,2]*1e-51, c=c[psr], alpha=0.5)#, zorder=-params[:,3])
    ax6.scatter(params[psr,1], params[psr,2]*1e-51, c=c[psr], alpha=0.5)#, zorder=-params[:,3])
else:
    ax5.scatter(params[psr,0], np.ones(len(params[psr,0])), c=c[psr])
    ax6.scatter(params[psr,1], np.ones(len(params[psr,0])), c=c[psr])
    
ax5.set_xlabel(r'$\epsilon$')
ax5.set_ylabel(r'Energy per supernova, $\mathcal{E}_{\rm SN}~[10^{44}\,{\rm J}]$')

ax6.set_xlabel(r'$f_{\rm move}^{\rm gas}$')
ax6.set_ylabel(r'$\mathcal{E}_{\rm SN}~[10^{44}\,{\rm J}]$')

ax4.axis([0.0378,0.4226,0,1])
ax5.axis([0.0378,0.4226,0.4,2.0])
ax6.axis([0,1,0.4,2.0])

ax1.legend(loc='lower left', frameon=False, bbox_to_anchor=(-0.02,-0.02))
ax2.legend(loc='lower left', frameon=False, bbox_to_anchor=(-0.02,-0.02))
ax3.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.02,-0.02))

ax3.set_xticks(np.arange(0,14,2))
ax6.set_xticks(np.arange(0.2,1.1,0.2))

for ax in axs.ravel():
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
for ax in [ax1, ax2, ax6]:
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
ax4.xaxis.set_minor_locator(MultipleLocator(0.02))
ax5.xaxis.set_minor_locator(MultipleLocator(0.02))

fig.subplots_adjust(hspace=0.17, wspace=0.25, left=0, bottom=0, right=0.98, top=1.0)

cbar_ax = fig.add_axes([0.99, 0.05, 0.015, 0.8])
cbar = fig.colorbar(ims, cax=cbar_ax, label=r'$\log_{10}(\chi_{\rm joint}^2)$')#, ticks=[0,1])
#cbar_ax.set_yticklabels(['Worst', 'Best'], rotation='vertical')
#ticks = cbar_ax.get_yticklabels()
#ticks.set_rotation(90)

if Nparams==4:
    for i in range(2): axs[2,i].scatter(params[psr,3], params[psr,i], c=c[psr])
    axs[2,2].scatter(params[psr,3], params[psr,2]*1e-51, c=c[psr])
    axs[2,0].set_xlabel(r'Type-Ia/Type-II supernova ratio')
    for i in range(1,3): axs[2,i].set_xlabel(r'$n_{\rm SN-Ia} / n_{\rm SN-II}$')
    axs[2,0].set_ylabel(r'$\epsilon$')
    axs[2,1].set_ylabel(r'$f_{\rm move}^{\rm gas}$')
    axs[2,2].set_ylabel(r'$\mathcal{E}_{\rm SN}~[10^{44}\,{\rm J}]$')
    axs[2,0].axis([0.1,0.9,0.0378,0.4226])
    axs[2,1].axis([0.1,0.9,0,1])
    axs[2,2].axis([0.1,0.9,0,3])
    for i in range(3): axs[2,i].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[2,0].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[2,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[2,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    gp.savepng(base_dir+'reconstructed_constraints', xpixplot=1700, ypixplot=1500, addpdf=True, transparent=transparent)

else:
    gp.savepng(base_dir+'reconstructed_constraints', xpixplot=1700, ypixplot=1000, addpdf=True, transparent=transparent)


Nsteps = int(len(params) / Nparticles)
for s in range(Nsteps):
    for ax in axs.ravel(): ax.cla()
        
    ax1.errorbar(SMF_x_obs[0], SMF_y_obs[0], yerr=SMF_yerr, ecolor=ec, ls='None', elinewidth=elw, capthick=elw, label=r'Driver et al.~(2022)')
    ax2.errorbar(HIMF_x_obs[0], HIMF_y_obs[0], yerr=np.row_stack((HIMF_yerr_dn,HIMF_yerr_up)), ecolor=ec, ls='None', elinewidth=elw, capthick=elw, label=r'Jones et al.~(2018)')
    ax3.errorbar(CSFH_x_obs[0], CSFH_y_obs[0], yerr=np.row_stack((CSFH_yerr_dn,CSFH_yerr_up)), ecolor=ec, ls='None', elinewidth=elw, capthick=elw, label=r"D'Silva et al.~(2023)")
    
    for i in range(s*Nparticles, (s+1)*Nparticles):
        ax1.plot(SMF_x_obs[i], SMF_y_mod[i], '-', color=c[i], zorder=-params[i,Nparams])
        ax2.plot(HIMF_x_obs[i], HIMF_y_mod[i], '-', color=c[i], zorder=-params[i,Nparams])
        ax3.plot(CSFH_x_obs[i], CSFH_y_mod[i], '-', color=c[i], zorder=-params[i,Nparams])

    ax1.axis([8.2,11.7,-4,-1.5])
    ax2.axis([8.9,10.7,-4,-1.5])
    ax3.axis([0,14,-3,-0.5])


    ax1.set_xlabel(r'$\log_{10}(m_*~[{\rm M}_\odot])$')
    ax1.set_ylabel(r'$\log_{10}(\Phi_*~[{\rm cMpc}^{-3}\,{\rm dex}^{-1}])$')

    ax2.set_xlabel(r'$\log_{10}(m_{\rm HI}~[{\rm M}_\odot])$')
    ax2.set_ylabel(r'$\log_{10}(\Phi_{\rm HI}~[{\rm cMpc}^{-3}\,{\rm dex}^{-1}])$')

    ax3.set_xlabel(r'Look-back time [Gyr]')
    ax3.set_ylabel(r'$\log_{10}(\bar{\rho}_{\rm SFR}~[{\rm M}_\odot\,{\rm yr}^{-1}\,{\rm cMpc}^{-3}])$')

    param_sort = np.argsort(params[s*Nparticles:(s+1)*Nparticles,Nparams])
    psr = param_sort[::-1]
    
    ax4.scatter(params[s*Nparticles:(s+1)*Nparticles,0][psr], params[s*Nparticles:(s+1)*Nparticles,1][psr], c=c[s*Nparticles:(s+1)*Nparticles][psr])#, zorder=-params[:,3])
    ax4.set_xlabel(r'Black-hole radiative efficiency, $\epsilon$')
    ax4.set_ylabel(r'Unstable-disc migration/heating ratio, $f_{\rm move}^{\rm gas}$')

    if Nparams>=3:
        ax5.scatter(params[s*Nparticles:(s+1)*Nparticles,0][psr], params[s*Nparticles:(s+1)*Nparticles,2][psr]*1e-51, c=c[s*Nparticles:(s+1)*Nparticles][psr])#, zorder=-params[:,3])
        ax6.scatter(params[s*Nparticles:(s+1)*Nparticles,1][psr], params[s*Nparticles:(s+1)*Nparticles,2][psr]*1e-51, c=c[s*Nparticles:(s+1)*Nparticles][psr])#, zorder=-params[:,3])
    else:
        ax5.scatter(params[s*Nparticles:(s+1)*Nparticles,0][psr], np.ones(Nparticles), c=c[s*Nparticles:(s+1)*Nparticles][psr])
        ax6.scatter(params[s*Nparticles:(s+1)*Nparticles,1][psr], np.ones(Nparticles), c=c[s*Nparticles:(s+1)*Nparticles][psr])
        
    ax5.set_xlabel(r'$\epsilon$')
    ax5.set_ylabel(r'Energy per supernova, $\mathcal{E}_{\rm SN}~[10^{44}\,{\rm J}]$')

    ax6.set_xlabel(r'$f_{\rm move}^{\rm gas}$')
    ax6.set_ylabel(r'$\mathcal{E}_{\rm SN}~[10^{44}\,{\rm J}]$')
    
    ax4.axis([0.0378,0.4226,0,1])
    ax5.axis([0.0378,0.4226,0,3])
    ax6.axis([0,1,0,3])
    
    ax1.legend(loc='lower left', frameon=False, bbox_to_anchor=(-0.02,-0.02))
    ax2.legend(loc='lower left', frameon=False, bbox_to_anchor=(-0.02,-0.02))
    ax3.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.02,-0.02))
    
    ax3.set_xticks(np.arange(0,14,2))
    ax6.set_xticks(np.arange(0.2,1.1,0.2))

    if Nparams==4:
        for i in range(2): axs[2,i].scatter(params[s*Nparticles:(s+1)*Nparticles,3][psr], params[s*Nparticles:(s+1)*Nparticles,i][psr], c=c[s*Nparticles:(s+1)*Nparticles][psr])
        axs[2,2].scatter(params[s*Nparticles:(s+1)*Nparticles,3][psr], params[s*Nparticles:(s+1)*Nparticles,2][psr]*1e-51, c=c[s*Nparticles:(s+1)*Nparticles][psr])
        axs[2,0].set_xlabel(r'Type-Ia/Type-II supernova ratio')
        for i in range(1,3): axs[2,i].set_xlabel(r'$n_{\rm SN-Ia} / n_{\rm SN-II}$')
        axs[2,0].set_ylabel(r'$\epsilon$')
        axs[2,1].set_ylabel(r'$f_{\rm move}^{\rm gas}$')
        axs[2,2].set_ylabel(r'$\mathcal{E}_{\rm SN}~[10^{44}\,{\rm J}]$')
        axs[2,0].axis([0.1,0.9,0.0378,0.4226])
        axs[2,1].axis([0.1,0.9,0,1])
        axs[2,2].axis([0.1,0.9,0,3])
        for i in range(3): axs[2,i].xaxis.set_minor_locator(MultipleLocator(0.1))
        axs[2,0].yaxis.set_minor_locator(MultipleLocator(0.02))
        axs[2,0].yaxis.set_minor_locator(MultipleLocator(0.1))
        axs[2,0].yaxis.set_minor_locator(MultipleLocator(0.5))
        gp.savepng(base_dir+'reconstructed_constraints_step'+str(s), xpixplot=1700, ypixplot=1500, addpdf=False, transparent=transparent)
    else:
        gp.savepng(base_dir+'reconstructed_constraints_step'+str(s), xpixplot=1700, ypixplot=1000, transparent=transparent, addpdf=False)
    
    if s==Nsteps-1:
        print('best-fitting params of last step in PSO', params[s*Nparticles+param_sort[0],:Nparams])


