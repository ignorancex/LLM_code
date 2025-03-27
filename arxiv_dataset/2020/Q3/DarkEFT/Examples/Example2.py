#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Example of use
###, L. Darme 02/07/2019

import matplotlib.pyplot as plt
import numpy as np
# Importing additional user-defined function

import UsefulFunctions as uf
import Amplitudes as am
import Production as br
import Detection as de
import LimitsList as lim
#############################################################################
###############        Several limits example    ##############################
#############################################################################


# This example loads a list of limits, recast them and save them in the Output folder, then plot some of them

if __name__ == "__main__":

    geffem={"gu11":2/3.,"gu22":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gd33":-1/3.,"gl11":-1.,"gl22":-1.}

    ExperimentsList=np.array(["lsnd_decay","charm_decay","seaquest_phase2_decay","faser_decay","ship_decay", \
                              "babar_monogam","belle2_monogam", \
                              "miniboone_scattering","sbnd_scattering","ship_scattering","nova_scattering", \
                              "sn1987_low_cooling","sn1987_high_cooling", \
                              "na62_invisibledecayPi0","bes_invisibledecayJPsi","babar_invisibledecayUpsilon",\
                              "atlas_monojet_down","atlas_monojet_up","lep_monogam"])

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,geffem,"V",True)


    fig=plt.figure(1)
    s = fig.add_subplot(1, 1, 1)
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    s.loglog(Lim['lsnd_decay'][0],Lim['lsnd_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15)
    s.fill_between(Lim['lsnd_decay'][0],ydown,Lim['lsnd_decay'][1],color='xkcd:green',alpha=0.5, zorder=15)
    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
    s.loglog(Lim['charm_decay'][0],Lim['charm_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:darkgreen',alpha=0.5, zorder=15)
    s.loglog(Lim['seaquest_phase2_decay'][0],Lim['seaquest_phase2_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:red',zorder=15)
    s.loglog(Lim['faser_decay'][0],Lim['faser_decay'][1],linestyle='-.',linewidth=1.5,color='xkcd:indigo',zorder=15)

    # -------- Mono photon at LEP
    s.fill_between(Lim['lep_monogam'][0],200,Lim['lep_monogam'][1],color='xkcd:blue',alpha=0.25, zorder=15)
    s.axhline(200,linestyle='--',linewidth=1.,color='xkcd:blue',zorder=15)
    s.loglog(Lim['lep_monogam'][0],Lim['lep_monogam'][1],linestyle='--',linewidth=1.5,color='xkcd:blue',zorder=15)

    # -------- Missing energy searches
    s.loglog(Lim['babar_monogam'][0],Lim['babar_monogam'][1],linestyle='-',linewidth=2,color='xkcd:grey',zorder=15)
    s.fill_between(Lim['babar_monogam'][0],ydown,Lim['babar_monogam'][1],color='xkcd:grey',alpha=0.5, zorder=15)
    s.loglog(Lim['belle2_monogam'][0],Lim['belle2_monogam'][1],linestyle='--',linewidth=1.5,color='xkcd:black',zorder=15)

    # ------ Self consistency
    s.loglog(xbasic,2*xbasic,linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
    s.fill_between(xbasic,ydown,2*xbasic,color='xkcd:grey',alpha=0.5, zorder=15)

    # -------- SN bounds
    gp=np.logical_and(Lim['sn1987_high_cooling'][1]>1.1*Lim['sn1987_low_cooling'][1],Lim['sn1987_high_cooling'][1]>1)
    s.loglog(Lim['sn1987_high_cooling'][0][gp],Lim['sn1987_high_cooling'][1][gp],linestyle='-',linewidth=1,color='xkcd:purple',zorder=15)
    s.loglog(Lim['sn1987_low_cooling'][0][gp],Lim['sn1987_low_cooling'][1][gp],linestyle='--',linewidth=0.5,color='xkcd:purple',zorder=15)
    s.fill_between(Lim['sn1987_low_cooling'][0][gp],Lim['sn1987_low_cooling'][1][gp],Lim['sn1987_high_cooling'][1][gp],color='xkcd:purple',alpha=0.15, zorder=15)

    # ----- Invisible decay of Upsilon

    s.loglog(Lim['babar_invisibledecayUpsilon'][0],Lim['babar_invisibledecayUpsilon'][1],linestyle='-',linewidth=1.1,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['babar_invisibledecayUpsilon'][0],ydown,Lim['babar_invisibledecayUpsilon'][1],color='xkcd:grey',alpha=0.25, zorder=15)

    s.text(0.08,220,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=40)
    s.text(1.1,52,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,110,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.01,2500,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=0)
    s.text(0.015,70,r'LSND',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=20)
    s.text(0.15,610,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)
    s.text(0.28,510,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=35)
    s.text(1.,4.e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=30)
    s.text(1.1,160,r'BaBar ($\Upsilon \to$ inv)' ,fontsize=10, zorder=50)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{\chi_2}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

    #---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/Example2.pdf')
    plt.show()
