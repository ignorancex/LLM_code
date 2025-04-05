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

def zeta_over_s_fct(T_in_fm):

    T_in_GeV=T_in_fm*hbarc;

    max_val=0.2
    width=0.02
    T_peak_in_GeV=0.185
    lambda_val=0.0 # Symmetric peak

    diff=T_in_GeV-T_peak_in_GeV;
    sign=np.sign(diff)
    diff_ratio=(diff)/(width*(lambda_val*sign+1));

    return max_val/(1+diff_ratio*diff_ratio);

def combined_visc_fct(T_in_fm):
    eta_over_s=0.0 #eta_over_s_fct(T_in_fm)
    zeta_over_s=zeta_over_s_fct(T_in_fm)
    return (4./3.*eta_over_s+zeta_over_s)



###############################################################################
########## Curves are plotted for the following initial temperatures ##########
###############################################################################

T0_list_in_fm=np.array([0.4,0.25,.2])/hbarc

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
    print("tauf=",tauf)

    Veff_approx=approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct, combined_visc_fct)
    Veff_better=better_effective_viscosity_ns(tau0, tauf, esol, cs2_fct, combined_visc_fct)

    res_dict[T0_in_fm]['Veff_approx']=Veff_approx
    res_dict[T0_in_fm]['Veff_better']=Veff_better

    print("Effective viscosity with T0="+str(T0_in_fm*hbarc)+"MeV and Tfr="+str(Tf_in_fm*hbarc)+"MeV:",Veff_approx)

    zeta_over_s_eff_approx=Veff_approx
    zeta_over_s_eff_better=Veff_better

    res_dict[T0_in_fm]['zeta_over_s_eff_approx']=zeta_over_s_eff_approx
    res_dict[T0_in_fm]['zeta_over_s_eff_better']=zeta_over_s_eff_better

    #print("approx=",zeta_over_s_eff_approx," vs better=",zeta_over_s_eff_better)

#print(res_dict)

###################################################################
############################ \zeta/s(T) ############################ 
###################################################################

plt.rc('font', **font_choice)

plt.figure()
plt.axes().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
#plt.yticks([])
#plt.axes().yaxis.set_minor_formatter(NullFormatter())
plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_ticks_position('both')
plt.axes().yaxis.set_minor_locator(AutoMinorLocator())


Tmax=0.45
Tmin=Tf_in_fm*hbarc
plt.xlim(Tmin,Tmax)
plt.xticks(np.arange(Tmin,Tmax,.05))
ymax=0.25
plt.ylim(0.,ymax)
plt.yticks(np.arange(0.,ymax,.05))

plt.xlabel(r"$T$ (GeV)")
plt.ylabel(r'$\zeta/s$')

T_range=np.arange(0.1,.6,.001)
zeta_over_s_list= np.array(list(map(zeta_over_s_fct,T_range/hbarc)))

# Plot zeta/s itself
plt.plot(T_range, zeta_over_s_list,"-",color='red',label=r'$\zeta/s(T)$', linewidth=linewidth_choice)

# Plot effective viscosities
color_list=["blue","black","#60BD68"]
for n, (T0_in_fm, ldict) in enumerate(res_dict.items()):

    color=color_list[n]

    zeta_over_s_eff_approx=ldict['zeta_over_s_eff_approx']
    zeta_over_s_eff_better=ldict['zeta_over_s_eff_better']

    plt.axhline(y=zeta_over_s_eff_approx, xmin=(Tf_in_fm*hbarc-Tmin)/(Tmax-Tmin), xmax=(T0_in_fm*hbarc-Tmin)/(Tmax-Tmin), color=color, linestyle='--', linewidth=3, label=r'$\langle \zeta/s \rangle_{eff, T_0='+'{:3d}'.format(int(round(T0_in_fm*hbarc*1000,-1)))+' \\rm{ MeV}}$')
    #plt.arrow(-0.08, zeta_over_s_eff_better/ymax, .18, 0, length_includes_head=True, head_width=0.03, head_length=0.03, transform=plt.axes().transAxes, color=color)
    #plt.axes().annotate('', xy=(0., zeta_over_s_eff_better/ymax), xycoords='axes fraction', xytext=(-0.1, zeta_over_s_eff_better/ymax), arrowprops=dict(arrowstyle="simple", color=color))

plt.legend(loc='upper right', prop={'size': 18})
#plt.legend(loc='lower right', prop={'size': 14})
plt.tight_layout()
plt.savefig("zeta_over_s_qcd.pdf")
plt.show()


#################################################
############## Temperature profile ##############
#################################################

tau_plot=10.

plt.figure()
plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
#plt.yticks([])
#plt.axes().yaxis.set_minor_formatter(NullFormatter())
#plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_ticks_position('both')
plt.axes().yaxis.set_minor_locator(AutoMinorLocator())

plt.xlim(tau0,tau_plot)
plt.ylim(.1,1.0*np.max(T0_list_in_fm)*hbarc)
#plt.xscale('log')
#plt.yscale('log')
#plt.yticks(np.arange(0.,0.3,.05))

plt.xlabel(r'$\tau$ (fm)')
plt.ylabel(r"$T$ (GeV)")


color_list=["blue","black","#60BD68"]
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
        label_exact=r'Viscous w/ $\zeta/s(T)$'
        label_eff=r'Viscous w/ $\langle \zeta/s \rangle_{eff}$'

    # Plot the exact solution
    tau_range=np.arange(tau0,tau_plot,(tau_plot-tau0)/100)
    T_from_exact= hbarc*np.array(list(map(esol.integrate,tau_range)))
    plt.plot(tau_range, T_from_exact,"-",color='red',label=label_exact, linewidth=linewidth_choice)

    # Plot the effective result
    T_from_eff= hbarc*np.array(list(map(esol_eff_approx.integrate,tau_range)))
    #T_from_eff= hbarc*np.array(list(map(esol_eff_better.integrate,tau_range)))
    plt.plot(tau_range, T_from_eff,"--",color='blue',label=label_eff, linewidth=linewidth_choice)

    #Plot the ideal solution, for reference
    T_from_ideal= hbarc*np.array(list(map(esol_ideal.integrate,tau_range)))
    plt.plot(tau_range, T_from_ideal,":",color='grey',label=label_ideal, linewidth=2)

    # Validation
    tauf=ldict['tauf']

    print("At tauf=",tauf," fm")
    print("Exact: ", esol.integrate(tauf), ", Eff. (approx): ", esol_eff_approx.integrate(tauf), ", Eff. (better): ", esol_eff_better.integrate(tauf))




plt.legend()
plt.tight_layout()
plt.savefig("T_profile_zeta_over_s_eff_qcd.pdf")
plt.show()



