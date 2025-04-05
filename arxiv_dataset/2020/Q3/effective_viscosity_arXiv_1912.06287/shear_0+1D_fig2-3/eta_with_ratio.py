import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import matplotlib.ticker as mtick
import scipy #.optimize

from eos import cs2_qcd_fct
from style import font_choice, linewidth_choice
from bjorken_ns_solver import init_bjorken_ns_solver, approx_effective_viscosity_ns, better_effective_viscosity_ns

hbarc=0.1973

# Speed of sound (EOS)
def cs2_fct(T_in_fm):
    return cs2_qcd_fct(T_in_fm)
    #return 1./3

# Viscosities
def eta_over_s_fct(T_in_fm):

    T_in_GeV=T_in_fm*hbarc;

    T_kink_in_GeV =0.17
    low_T_slope=5
    high_T_slope=0.5
    eta_over_s_at_kink=0.04

    if (T_in_GeV<T_kink_in_GeV):
        eta_over_s=eta_over_s_at_kink + low_T_slope*(T_kink_in_GeV - T_in_GeV);
    else:
        eta_over_s=eta_over_s_at_kink + high_T_slope*(T_in_GeV - T_kink_in_GeV);

    return eta_over_s;


def combined_visc_fct(T_in_fm):
    eta_over_s=eta_over_s_fct(T_in_fm)
    zeta_over_s=0.0 #zeta_over_s_fct(T_in_fm)
    return (4./3.*eta_over_s+zeta_over_s)



###############################################################################
########## Curves are plotted for the following initial temperatures ##########
###############################################################################

T0_list_in_fm=np.array([0.4,0.25])/hbarc

# Other parameters
tau0=0.2
Tf_in_fm=0.15/hbarc

res_dict={}
for T0_in_fm in T0_list_in_fm:
    res_dict[T0_in_fm]={}

    # Get Navier-Stokes solution
    esol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, combined_visc_fct_param=combined_visc_fct)
    res_dict[T0_in_fm]['sol']=esol

    max_possible_tau=1000.
    tauf=scipy.optimize.brentq(lambda tau: esol.integrate(tau) - Tf_in_fm, tau0, max_possible_tau)
    res_dict[T0_in_fm]['tauf']=tauf
    print("tau corresponding to the final temperature=",tauf," fm")

    Veff_approx=approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct, combined_visc_fct)
    Veff_better=better_effective_viscosity_ns(tau0, tauf, esol, cs2_fct, combined_visc_fct)

    res_dict[T0_in_fm]['Veff_approx']=Veff_approx
    res_dict[T0_in_fm]['Veff_better']=Veff_better

    #print("Effective viscosity with T0="+str(T0_in_fm*hbarc)+"MeV and Tfr="+str(Tf_in_fm*hbarc)+"MeV:",Veff_approx)

    eta_over_s_eff_approx=3./4.*Veff_approx
    eta_over_s_eff_better=3./4.*Veff_better

    res_dict[T0_in_fm]['eta_over_s_eff_approx']=eta_over_s_eff_approx
    res_dict[T0_in_fm]['eta_over_s_eff_better']=eta_over_s_eff_better

    print("Effective eta/s with T0="+str(T0_in_fm*hbarc)+"MeV and Tfr="+str(Tf_in_fm*hbarc)+"MeV:",eta_over_s_eff_approx)
    print("More precise effective eta/s with T0="+str(T0_in_fm*hbarc)+"MeV and Tfr="+str(Tf_in_fm*hbarc)+"MeV:",eta_over_s_eff_better)

#print(res_dict)

###################################################################
############################ \eta/s(T) ############################ 
###################################################################

plt.rc('font', **font_choice)

plt.figure()
plt.axes().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
#plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
#plt.yticks([])
#plt.axes().yaxis.set_minor_formatter(NullFormatter())
plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_ticks_position('both')
#plt.axes().yaxis.set_minor_locator(AutoMinorLocator())


Tmax=0.45
Tmin=Tf_in_fm*hbarc
plt.xlim(Tmin,Tmax)
plt.xticks(np.arange(Tmin,Tmax,.05))
ymax=0.25
plt.ylim(0.,ymax)
plt.yticks(np.arange(0.,ymax,.05))

plt.xlabel("T (GeV)")
plt.ylabel(r'$\eta/s$')

T_range=np.arange(0.1,.6,.001)
eta_over_s_list= np.array(list(map(eta_over_s_fct,T_range/hbarc)))

# Plot eta/s itself
plt.plot(T_range, eta_over_s_list,"-",color='red',label=r'$\eta/s(T)$', linewidth=linewidth_choice)

# Plot effective viscosities
color_list=["blue","black"]
for n, (T0_in_fm, ldict) in enumerate(res_dict.items()):

    color=color_list[n]

    eta_over_s_eff_approx=ldict['eta_over_s_eff_approx']
    eta_over_s_eff_better=ldict['eta_over_s_eff_better']

    plt.axhline(y=eta_over_s_eff_approx, xmin=(Tf_in_fm*hbarc-Tmin)/(Tmax-Tmin), xmax=(T0_in_fm*hbarc-Tmin)/(Tmax-Tmin), color=color, linestyle='--', linewidth=3, label=r'$\langle \eta/s \rangle_{eff, T_0='+'{:3d}'.format(int(round(T0_in_fm*hbarc*1000,-1)))+' \\rm{ MeV}}$')
    #plt.arrow(-0.08, eta_over_s_eff_better/ymax, .18, 0, length_includes_head=True, head_width=0.03, head_length=0.03, transform=plt.axes().transAxes, color=color)
    #plt.axes().annotate('', xy=(0., eta_over_s_eff_better/ymax), xycoords='axes fraction', xytext=(-0.1, eta_over_s_eff_better/ymax), arrowprops=dict(arrowstyle="simple", color=color))

plt.legend(loc='upper left', prop={'size': 18})
#plt.legend(loc='lower right', prop={'size': 14})
plt.tight_layout()
plt.savefig("eta_over_s_qcd.pdf")
plt.show()


#################################################
############## Temperature profile ##############
#################################################

tau_plot=10.

#fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})


#plt.figure()
#plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
axes[1].xaxis.set_minor_locator(AutoMinorLocator())
#plt.yticks([])
##plt.axes().yaxis.set_minor_formatter(NullFormatter())
axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
axes[0].yaxis.set_ticks_position('both')
axes[0].yaxis.set_minor_locator(AutoMinorLocator())
axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
axes[1].yaxis.set_ticks_position('both')
axes[1].yaxis.set_minor_locator(AutoMinorLocator())
#
axes[1].set_xlim(tau0,tau_plot)
axes[0].set_ylim(.1,1.0*np.max(T0_list_in_fm)*hbarc)
axes[1].set_ylim(0.95,1.05)
#axes[0].set_xscale('log')
#axes[0].set_yscale('log')
axes[0].set_yticks(np.arange(0.15,0.41,.05))
axes[1].set_yticks(np.arange(0.975,1.0251,0.025))

axes[1].set_xlabel(r'$\tau$ (fm)')
axes[0].set_ylabel(r"$T$ (GeV)")
axes[1].set_ylabel(r"$\frac{T_{\mathrm{viscous,eff.}}}{T_{\mathrm{viscous}}}$")
axes[1].axhline(y=1, color='black', linestyle=':', linewidth=2) 


color_list=["blue","black"]
for n, (T0_in_fm, ldict) in enumerate(res_dict.items()):

    color=color_list[n]
    esol=ldict['sol']

    Veff_approx=ldict['Veff_approx']
    Veff_better=ldict['Veff_better']

    # Temperature profile plot
    esol_eff_approx=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, combined_visc_fct_param=lambda T: Veff_approx)
    esol_eff_better=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, combined_visc_fct_param=lambda T: Veff_better)
    esol_ideal=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct)

    # Only plot labels once
    label_ideal=''
    label_exact=''
    label_eff=''
    if (n == 0):
        label_ideal='Ideal'
        label_exact=r'Viscous w/ $\eta/s(T)$'
        label_eff=r'Viscous w/ $\langle \eta/s \rangle_{eff}$'

    # Plot the exact solution
    tau_range=np.arange(tau0,tau_plot,(tau_plot-tau0)/100)
    T_from_exact= hbarc*np.array(list(map(esol.integrate,tau_range)))
    axes[0].plot(tau_range, T_from_exact,"-",color='red',label=label_exact, linewidth=linewidth_choice)

    # Plot the effective result
    T_from_eff= hbarc*np.array(list(map(esol_eff_approx.integrate,tau_range)))
    #T_from_eff= hbarc*np.array(list(map(esol_eff_better.integrate,tau_range)))
    axes[0].plot(tau_range, T_from_eff,"--",color='blue',label=label_eff, linewidth=linewidth_choice)

    #Plot the ideal solution, for reference
    T_from_ideal= hbarc*np.array(list(map(esol_ideal.integrate,tau_range)))
    axes[0].plot(tau_range, T_from_ideal,":",color='dimgrey',label=label_ideal, linewidth=4)

    axes[1].plot(tau_range, T_from_eff/T_from_exact,"-.",color=['darkgreen','purple'][n],label=r"$T_0={0:.3f}$ GeV".format(T0_in_fm*hbarc), linewidth=4)

    # Validation
    tauf=ldict['tauf']

    #print("At tauf=",tauf," fm")
    #print("Exact: ", esol.integrate(tauf), ", Eff. (approx): ", esol_eff_approx.integrate(tauf), ", Eff. (better): ", esol_eff_better.integrate(tauf))

axes[0].legend(fontsize=14)
axes[1].legend(fontsize=10,ncol=2)

fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

#plt.legend()
fig.savefig("T_profile_eta_over_s_eff_qcd_with_ratio.pdf")
fig.show()



