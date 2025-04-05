import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
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


def zeta_over_s_fct(T_in_fm, max_val=0.2, width_in_GeV=0.02, T_peak_in_GeV=0.185):

    T_in_GeV=T_in_fm*hbarc;

    lambda_val=0.0 # Symmetric peak

    diff=T_in_GeV-T_peak_in_GeV;
    sign=np.sign(diff)
    diff_ratio=(diff)/(width_in_GeV*(lambda_val*sign+1));

    return max_val/(1+diff_ratio*diff_ratio);




###############################################################################
########## Curves are plotted for the following initial temperatures ##########
###############################################################################

# Bjorken evolution parameters
T0_in_fm=0.3/hbarc
tau0=0.2
Tf_in_fm=0.15/hbarc

#
bulk_dict={
    'default':{'Tpeak':.185,'max':.3,'width':0.03, 'color':'red', 'style':'-'},
    '1':{'Tpeak':.185,'width':0.06, 'color':'blue', 'style':'--'},
    '2':{'Tpeak':.185,'width':0.12, 'color':'black', 'style':':'},
    '3':{'Tpeak':.225,'width':0.03, 'color':'#60BD68', 'style':'-.'},
}

# Compute the effective viscosity for the default value
bulk_dict['default']['Veff']=approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct, 
    lambda T_in_fm: 
        zeta_over_s_fct(T_in_fm,
            max_val=bulk_dict['default']['max'],
            width_in_GeV=bulk_dict['default']['width'],
            T_peak_in_GeV=bulk_dict['default']['Tpeak']) )

# Compute the normalization of each other parametrization so that they have the same effective bulk viscosity
for key, vals in bulk_dict.items():
    
    if (key == 'default')or(key == 'test'):
        continue

    Veff_default=bulk_dict['default']['Veff']

    max_guess=0.2

    Veff=approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct,
            lambda T_in_fm:
                zeta_over_s_fct(T_in_fm,
                    max_val=max_guess,
                    width_in_GeV=vals['width'],
                    T_peak_in_GeV=vals['Tpeak']) )

    bulk_dict[key]['max']=max_guess*Veff_default/Veff

print(bulk_dict)

#################################################
############## Temperature profile ##############
#################################################

plt.rc('font', **font_choice)

tau_plot=10.

plt.figure()
plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
#plt.yticks([])
#plt.axes().yaxis.set_minor_formatter(NullFormatter())
#plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_ticks_position('both')
plt.axes().yaxis.set_minor_locator(AutoMinorLocator())

plt.xlim(tau0,tau_plot)
plt.ylim(.125,1.0*(T0_in_fm)*hbarc)
#plt.xscale('log')
#plt.yscale('log')
#plt.yticks(np.arange(0.,0.3,.05))

plt.xlabel(r'$\tau$ (fm)')
plt.ylabel(r"$T$ (GeV)")

for key, vals in bulk_dict.items():

    color=vals['color']
    style=vals['style']

    esol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, 
            lambda T_in_fm:
                zeta_over_s_fct(T_in_fm,
                    max_val=vals['max'],
                    width_in_GeV=vals['width'],
                    T_peak_in_GeV=vals['Tpeak']))
    tau_range=np.arange(tau0,tau_plot,(tau_plot-tau0)/100)
    T_array= hbarc*np.array(list(map(esol.integrate,tau_range)))
    plt.plot(tau_range, T_array,style,color=color,label='', linewidth=linewidth_choice)


# Plot the ideal solution, for reference
esol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct)
T_from_ideal= hbarc*np.array(list(map(esol.integrate,tau_range)))
plt.plot(tau_range, T_from_ideal,"-",color='dimgrey',label='Ideal', linewidth=2)

plt.tight_layout()

############ Plot zeta/s(T) in the same canvas ############

a = plt.axes([0.53, 0.53, .35, .35]) #, facecolor='y')
a.xaxis.set_minor_locator(AutoMinorLocator())
a.yaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_ticks_position('both')

plt.xlabel(r"$T$ (GeV)")
plt.ylabel(r'$\zeta/s(T)$')

Tmax=0.45
Tmin=Tf_in_fm*hbarc
plt.xlim(Tmin,Tmax)
plt.xticks(np.arange(Tmin,Tmax,.1))
ymax=0.35
plt.ylim(0.,ymax)
plt.yticks(np.arange(0.,ymax,.1))

T_range=np.arange(0.1,.6,.001)

for key, vals in bulk_dict.items():

    color=vals['color']
    style=vals['style']

    zeta_over_s_list= np.array(list(map(
            lambda T_in_fm:
                zeta_over_s_fct(T_in_fm,
                    max_val=vals['max'],
                    width_in_GeV=vals['width'],
                    T_peak_in_GeV=vals['Tpeak']) 
                                        ,T_range/hbarc)))
    plt.plot(T_range, zeta_over_s_list,style,color=color,label='', linewidth=linewidth_choice)

plt.axhline(y=0.001, color='dimgrey', linestyle='-') #, label=r'$\langle \zeta/s \rangle_{eff}=$'+ "{0:.4f}".format(round(effective,4))) 
a.annotate('', xy=(1., bulk_dict['default']['Veff']/ymax), xycoords='axes fraction', xytext=(0.9, bulk_dict['default']['Veff']/ymax), arrowprops=dict(arrowstyle="simple", color='orange'))
#plt.axhline(y=bulk_dict['default']['Veff'], color='', linestyle='--',) #, label=r'$\langle \zeta/s \rangle_{eff}=$'+ "{0:.4f}".format(round(effective,4))) 


#plt.legend()
plt.tight_layout()
plt.savefig("T_profile_equiv_zeta_over_s.pdf")
plt.show()



