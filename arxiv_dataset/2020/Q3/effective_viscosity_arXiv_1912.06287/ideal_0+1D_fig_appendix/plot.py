import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter,AutoMinorLocator
import matplotlib.ticker as mtick

from eos import cs2_qcd_fct
from style import font_choice, linewidth_choice
from bjorken_ns_solver import init_bjorken_ns_solver, approx_effective_viscosity_ns, better_effective_viscosity_ns
from bjorken_approx_sols import T_ideal_bjorken_approx

hbarc=0.1973

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

    tau_list=np.arange(tau0,20.,.3)

    res_dict[T0_in_fm]['tau']=tau_list

    # Get exact solution
    esol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_qcd_fct)
    res_dict[T0_in_fm]['exact']=np.array([esol.integrate(tau) for tau in tau_list])

    # Approx
    res_dict[T0_in_fm]['approx_csm2_3']=np.array([T_ideal_bjorken_approx(T0_in_fm, tau0, tau, 3, cs2_qcd_fct) for tau in tau_list])
    res_dict[T0_in_fm]['approx_csm2_4']=np.array([T_ideal_bjorken_approx(T0_in_fm, tau0, tau, 4, cs2_qcd_fct) for tau in tau_list])
    res_dict[T0_in_fm]['approx_csm2_5']=np.array([T_ideal_bjorken_approx(T0_in_fm, tau0, tau, 5, cs2_qcd_fct) for tau in tau_list])

    # Get exact conformal solution for reference
    esol=init_bjorken_ns_solver(T0_in_fm, tau0, lambda T: 1/3.)
    res_dict[T0_in_fm]['exact_conformal']=np.array([esol.integrate(tau) for tau in tau_list])


#    max_possible_tau=1000.
#    tauf=scipy.optimize.brentq(lambda tau: esol.integrate(tau) - Tf_in_fm, tau0, max_possible_tau)
#    res_dict[T0_in_fm]['tauf']=tauf


#################################################
############## Temperature profile ##############
#################################################

tau_plot=10.


#plt.xlim(tau0,tau_plot)
#plt.ylim(.1,1.0*np.max(T0_list_in_fm)*hbarc)
##plt.xscale('log')
##plt.yscale('log')
##plt.yticks(np.arange(0.,0.3,.05))
#
#plt.xlabel(r'$\tau$ (fm)')
#plt.ylabel("T (GeV)")

#files_ns={'250': 'tau_ideal_and_idealApprox_tau0=0.2_T0=250MeV.dat', '400':'tau_ideal_and_idealApprox_tau0=0.2_T0=400MeV.dat'}
#
#res_dict={}
#for temp, filename in files_ns.items():
#
#    tmp_data=np.loadtxt(filename)
#
#    res_dict[temp]={'tau':tmp_data[:,[0]],'conformal_ideal':tmp_data[:,[1]]*hbarc,'qcd_ideal':tmp_data[:,[2]]*hbarc,'qcd_approx_csm2=3':tmp_data[:,[3]]*hbarc,'qcd_approx_csm2=4':tmp_data[:,[4]]*hbarc,'qcd_approx_csm2=5':tmp_data[:,[5]]*hbarc}
#
#font = {'family' : 'URW Gothic',
#        'weight' : 'bold',
#        'size'   : 14}
#
#plt.rc('font', **font)

plt.rc('font', **font_choice)

for T0_in_fm in T0_list_in_fm:

    if (np.isclose(T0_in_fm,.250/hbarc)):
        temp='250'
    elif (np.isclose(T0_in_fm,.400/hbarc)):
        temp='400'

    #plt.figure()
    #plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
    ##plt.yticks([])
    ##plt.axes().yaxis.set_minor_formatter(NullFormatter())
    ##plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    #plt.gca().yaxis.set_ticks_position('both')
    #plt.axes().yaxis.set_minor_locator(AutoMinorLocator())

    plt.figure()
    plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
    plt.yscale('log')
    plt.yticks([])
    plt.axes().yaxis.set_minor_formatter(NullFormatter())
    plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())

    if (temp == '250'):
        plt.xlim(0,5.)
        plt.ylim(.025,.35)
        plt.yticks(np.arange(.05, .35, .05))
        label_pos=[.96,.96,'top','right']
    else:
        plt.xlim(0,10.)
        plt.ylim(.1,.6)
        plt.yticks(np.arange(.1, .5, .05))
        label_pos=[.04,.05,'bottom','left']

    plt.xlabel(r'$\tau$ (fm)')
    plt.ylabel(r"$T$ (GeV)")


    tau_list=res_dict[T0_in_fm]['tau']=tau_list

    plt.text(label_pos[0], label_pos[1], r'$T_0=$'+temp+" MeV", bbox={'facecolor':'black', 'alpha':0.0, 'pad':2}, verticalalignment=label_pos[2], horizontalalignment=label_pos[3], color='black', fontsize=18, transform=plt.axes().transAxes)

    plt.plot(tau_list, hbarc*res_dict[T0_in_fm]['exact'],"d",color='red',label="Exact solution", markersize=6)
    plt.plot(tau_list, hbarc*res_dict[T0_in_fm]['approx_csm2_3'],"-",color='green',label="Approx. w/ "r'$\bar{c}_s^{-2}=3$',linewidth=4)
    plt.plot(tau_list, hbarc*res_dict[T0_in_fm]['approx_csm2_4'],":",color='blue',label="Approx. w/ "r'$\bar{c}_s^{-2}=4$', linewidth=4)
    plt.plot(tau_list, hbarc*res_dict[T0_in_fm]['approx_csm2_5'],"--",color='black',label="Approx. w/ "r'$\bar{c}_s^{-2}=5$', linewidth=4)
    plt.plot(tau_list, hbarc*res_dict[T0_in_fm]['exact_conformal'],"^",color='orange',label="Exact solution (conformal)", markersize=6)

    plt.legend(prop={'size': 13.},markerscale=1.8)
    plt.tight_layout()
    plt.savefig("ideal_solutions_T0="+temp+".pdf")
    plt.show()

