import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import matplotlib.ticker as mtick
import scipy #.optimize

from eos import cs2_qcd_fct
from style import font_choice, linewidth_choice
from bjorken_ns_solver import init_bjorken_ns_solver, approx_effective_viscosity_ns, better_effective_viscosity_ns
from bjorken_is_solver import init_bjorken_is_solver, Y_factor
from bjorken_approx_sols import PiHat_approx, T_ideal_bjorken_approx

hbarc=0.1973

# Speed of sound (EOS)
def cs2_fct(T_in_fm):
    return cs2_qcd_fct(T_in_fm)
    #return 1./3

# Viscosities

def zeta_over_s_fct(T_in_fm):

    T_in_GeV=T_in_fm*hbarc;

    max_val=0.3
    width=0.03
    T_peak_in_GeV=0.185
    lambda_val=0.0 # Symmetric peak

    diff=T_in_GeV-T_peak_in_GeV;
    sign=np.sign(diff)
    diff_ratio=(diff)/(width*(lambda_val*sign+1));

    return max_val/(1+diff_ratio*diff_ratio);


def tau_Pi_fct(T_in_fm):

    # 14th moments RTA - in MUSIC
    return 1.0
    #return 1/(15.*np.power(1/3-cs2_fct(T_in_fm),2))*zeta_over_s_fct(T_in_fm)/T_in_fm


##############################
############ Test! ###########
##############################
#
#T0_in_fm=0.3/hbarc
#tau0=0.2
#Tf_in_fm=.15/hbarc
#
##print(Y_factor(tau0, T0_in_fm, .2/hbarc, cs2_fct, tau_Pi_fct))
##print(Y_factor(tau0, T0_in_fm, .15/hbarc, cs2_fct, tau_Pi_fct))
##print(Y_factor(tau0, T0_in_fm, .1/hbarc, cs2_fct, tau_Pi_fct))
##exit(1)
#
## Ideal 
#sol_ideal=init_bjorken_ns_solver(T0_in_fm,tau0, cs2_fct)
#
## Navier-Stokes
#sol_ns=init_bjorken_ns_solver(T0_in_fm,tau0, cs2_fct, zeta_over_s_fct)
#
## Israel-Stewart with NS PiHat0
#PiHat0_a=-zeta_over_s_fct(T0_in_fm)/T0_in_fm/tau0
#sol_is_a=init_bjorken_is_solver([T0_in_fm,0.0,PiHat0_a],tau0, cs2_fct, zeta_over_s_fct_param=zeta_over_s_fct, tau_Pi_fct_param=tau_Pi_fct)
#
## Israel-Stewart with no bulk viscosity and compensating  PiHat0
#ns_eff_visc=approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct, zeta_over_s_fct)
#Y_factor=Y_factor(tau0, T0_in_fm, Tf_in_fm, cs2_fct, tau_Pi_fct)
#PiHat0_b=ns_eff_visc/Y_factor
#sol_is_b=init_bjorken_is_solver([T0_in_fm,0.0,PiHat0_b],tau0, cs2_fct, zeta_over_s_fct_param=lambda T: 0.00001, tau_Pi_fct_param=tau_Pi_fct)
#print("PiHat equiv=",PiHat0_b)
#print("ns eff visc=",ns_eff_visc)
#
## Israel-Stewart with NS PiHat0 and constant effective zeta/s
#PiHat0_c=-zeta_over_s_fct(T0_in_fm)/T0_in_fm/tau0
#sol_is_c=init_bjorken_is_solver([T0_in_fm,0.0,PiHat0_c],tau0, cs2_fct, zeta_over_s_fct_param=lambda T: ns_eff_visc, tau_Pi_fct_param=tau_Pi_fct)
#
##music_res=np.loadtxt("../music/evolution_bjorken.dat")
##for tau, edens, T, _, _, Pi_over_sT in music_res:
##    #print(tau, hbarc*T, hbarc*sol_ns.integrate(tau)[0], hbarc*sol_is.integrate(tau), -hbarc*zeta_over_s_fct(T)/T/tau, hbarc*Pi_over_sT)
##    T_is_a, piHat_is_a, PiHat_is_a=sol_is_a.integrate(tau)
##    T_is_b, piHat_is_b, PiHat_is_b=sol_is_b.integrate(tau)
##    T_is_c, piHat_is_c, PiHat_is_c=sol_is_c.integrate(tau)
##    #PiHatApprox=PiHat_approx(tau,tau0,PiHat0, lambda tau: sol_is.integrate(tau)[0], zeta_over_s_fct, tau_Pi_fct)
##    #PiHatApprox=PiHat_approx(tau,tau0,PiHat0, T0_in_fm, T_is, zeta_over_s_fct, tau_Pi_fct)
##    #print(tau, "T:", hbarc*T, " vs ", hbarc*T_is, "; ", Pi_over_sT, " vs ", PiHat_is, "NS:", -zeta_over_s_fct(T)/T/tau, "Approx:", PiHatApprox) 
##    print(tau, "T:", hbarc*T, " vs ", hbarc*T_is_a, " vs ", hbarc*T_is_b, " vs ", hbarc*T_is_c ) 
##    
##    if (T<Tf_in_fm):
##        break
#
#for tau in np.arange(tau0,20,.5):
#    #print(tau, hbarc*T, hbarc*sol_ns.integrate(tau)[0], hbarc*sol_is.integrate(tau), -hbarc*zeta_over_s_fct(T)/T/tau, hbarc*Pi_over_sT)
#    T_id=sol_ideal.integrate(tau)[0]
#    T_is_a, piHat_is_a, PiHat_is_a=sol_is_a.integrate(tau)
#    T_is_b, piHat_is_b, PiHat_is_b=sol_is_b.integrate(tau)
#    T_is_c, piHat_is_c, PiHat_is_c=sol_is_c.integrate(tau)
#    #PiHatApprox=PiHat_approx(tau,tau0,PiHat0, lambda tau: sol_is.integrate(tau)[0], zeta_over_s_fct, tau_Pi_fct)
#    #PiHatApprox=PiHat_approx(tau,tau0,PiHat0, T0_in_fm, T_is, zeta_over_s_fct, tau_Pi_fct)
#    #print(tau, "T:", hbarc*T, " vs ", hbarc*T_is, "; ", Pi_over_sT, " vs ", PiHat_is, "NS:", -zeta_over_s_fct(T)/T/tau, "Approx:", PiHatApprox) 
#    print(tau, "Tid:", hbarc*T_id, " vs ", hbarc*T_is_a, " vs ", hbarc*T_is_b, " vs ", hbarc*T_is_c ) 
#    
#    if (T_is_a<Tf_in_fm):
#        break



###############################################################################
########## 
###############################################################################

# Bjorken evolution parameters
T0_in_fm=0.3/hbarc
tau0=0.2
Tf_in_fm=0.15/hbarc

#
bulk_dict={
    'default':{'Tpeak':.185,'max':.3,'width':0.03, 'color':'red', 'style':'-'},
    #'mimic':{'Tpeak':.0,'max':.0,'width':1000., 'color':'#FAA43A', 'style':'--'},
    'mimic':{'Tpeak':.0,'max':.0,'width':1000., 'color':'purple', 'style':'--'},
}

# Default: Israel-Stewart with NS PiHat0
tmp_PiHat0=-zeta_over_s_fct(T0_in_fm)/T0_in_fm/tau0
bulk_dict['default']['PiHat0']=tmp_PiHat0
bulk_dict['default']['sol']=init_bjorken_is_solver([T0_in_fm,0.0,tmp_PiHat0],tau0, cs2_fct, zeta_over_s_fct_param=zeta_over_s_fct, tau_Pi_fct_param=tau_Pi_fct)

# Israel-Stewart with no bulk viscosity and compensating  PiHat0
ns_eff_visc=approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct, zeta_over_s_fct)
Y_factor=Y_factor(tau0, T0_in_fm, Tf_in_fm, cs2_fct, tau_Pi_fct)
tmp_PiHat0=ns_eff_visc/Y_factor
bulk_dict['mimic']['PiHat0']=tmp_PiHat0
bulk_dict['mimic']['sol']=init_bjorken_is_solver([T0_in_fm,0.0,tmp_PiHat0],tau0, cs2_fct, zeta_over_s_fct_param=lambda T: 1e-5, tau_Pi_fct_param=tau_Pi_fct)


#################################################
############## Temperature profile ##############
#################################################

plt.rc('font', **font_choice)

tau_plot=10.

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})

ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
#plt.yticks([])
#plt.axes().yaxis.set_minor_formatter(NullFormatter())
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax1.yaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax2.set_xlim(tau0,tau_plot)
ax1.set_ylim(.125,1.0*(T0_in_fm)*hbarc)
ax2.set_ylim(0.95,1.05)
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax2.set_xticks(np.arange(.0,tau_plot*1.01,2.))
ax1.set_yticks(np.arange(.15,1.0*(T0_in_fm)*hbarc,.05))
ax2.set_yticks(np.arange(0.95,1.04,.05))

ax2.set_xlabel(r'$\tau$ (fm)')
ax1.set_ylabel(r"$T$ (GeV)")
#ax2.set_ylabel(r"$\frac{T_{\mathrm{viscous, eff}}}{T_{\mathrm{viscous}}}$")
#ax2.set_ylabel(r"$\frac{T\;\rm{with} \; (\zeta/s)=0}{T\; \rm{with}\; \zeta/s(T)}$")
ax2.set_ylabel(r"$\frac{T\;\rm{with} \; \zeta/s=0}{T\; \rm{with}\; \zeta/s(T)}$")

tau_range=np.arange(tau0,tau_plot,(tau_plot-tau0)/100)

esol_eval_dict={}

for key, vals in bulk_dict.items():

    color=vals['color']
    style=vals['style']

    esol=vals['sol']


    #T_array= hbarc*np.array(list(map(esol.integrate,tau_range)))
    T_array=np.array([ hbarc*esol.integrate(tau)[0] for tau in tau_range ])
    ax1.plot(tau_range, T_array,style,color=color,label='', linewidth=linewidth_choice)

    esol_eval_dict[key]=T_array

#ratio=[ (bulk_dict['mimic']['sol'].integrate(tau)[0])/(bulk_dict['default']['sol'].integrate(tau)[0]) for tau in tau_range ]
#
#print("mimic",[ hbarc*((bulk_dict['mimic']['sol']).integrate(tau)[0]) for tau in tau_range ])
#esol=bulk_dict['default']['sol']
#print("default",[ hbarc*((bulk_dict['default']['sol']).integrate(tau)[0]) for tau in tau_range ])
#print("default2",[ hbarc*(esol.integrate(tau)) for tau in tau_range ])

#print("ratio=",ratio)

ax2.plot(tau_range, esol_eval_dict['mimic']/esol_eval_dict['default'],'-.', color='black',label='', linewidth=linewidth_choice)
#ax2.plot(tau_range, [1 for elem in tau_range],':', color='grey',label='', linewidth=2)
ax2.axhline(y=1, color='black', linestyle=':', linewidth=2) 


# Plot the ideal solution, for reference
esol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct)
T_from_ideal= hbarc*np.array(list(map(esol.integrate,tau_range)))
#print("T_from_ideal=",T_from_ideal)
ax1.plot(tau_range, T_from_ideal,"-",color='dimgrey',label='Ideal', linewidth=2)

plt.tight_layout()

############ Plot zeta/s(T) in the same canvas ############

a = plt.axes([0.58, 0.61, .32, .32]) #, facecolor='y')
a.xaxis.set_minor_locator(AutoMinorLocator())
a.yaxis.set_minor_locator(AutoMinorLocator())
a.yaxis.set_ticks_position('both')

a.set_xlabel(r"$T$ (GeV)")
a.set_ylabel(r'$\zeta/s(T)$')

Tmax=0.45
Tmin=Tf_in_fm*hbarc
a.set_xlim(Tmin,Tmax)
a.set_xticks(np.arange(Tmin,Tmax*0.99,.1))
ymax=0.35
a.set_ylim(0.,ymax)
a.set_yticks(np.arange(0.,ymax,.1))

T_range=np.arange(0.1,.6,.001)

#for key, vals in bulk_dict.items():
#
#    color=vals['color']
#    style=vals['style']
#
#    zeta_over_s_list= np.array(list(map(
#            lambda T_in_fm:
#                zeta_over_s_fct(T_in_fm,
#                    max_val=vals['max'],
#                    width_in_GeV=vals['width'],
#                    T_peak_in_GeV=vals['Tpeak']) 
#                                        ,T_range/hbarc)))
#    plt.plot(T_range, zeta_over_s_list,style,color=color,label='', linewidth=linewidth_choice)

# Plot default
color=bulk_dict['default']['color']
style=bulk_dict['default']['style']
zeta_over_s_list= np.array(list(map(
        lambda T_in_fm:
            zeta_over_s_fct(T_in_fm) ,T_range/hbarc)))
a.plot(T_range, zeta_over_s_list,style,color=color,label='', linewidth=linewidth_choice)

# Plot mimic
color=bulk_dict['mimic']['color']
style=bulk_dict['mimic']['style']
a.axhline(y=0.002, color=color, linestyle=style, linewidth=6) 


#plt.axhline(y=0, color='grey', linestyle='-') #, label=r'$\langle \zeta/s \rangle_{eff}=$'+ "{0:.4f}".format(round(effective,4))) 
#a.annotate('', xy=(1., bulk_dict['default']['Veff']/ymax), xycoords='axes fraction', xytext=(0.9, bulk_dict['default']['Veff']/ymax), arrowprops=dict(arrowstyle="simple", color='orange'))
#plt.axhline(y=bulk_dict['default']['Veff'], color='', linestyle='--',) #, label=r'$\langle \zeta/s \rangle_{eff}=$'+ "{0:.4f}".format(round(effective,4))) 


#plt.legend()
plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig("T_profile_equiv_zeta_over_s_Pi0_mimic_with_ratio.pdf")
plt.show()
