#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################## ##############################
#########################  Used to reproduce all the plots from the corresponding paper #########################
######################### ######################### #############################

### Include all the plots of the paper
### L. Darme, T. You, S. Ellis 29/12/2019



############################ Messy library import part #########################
import matplotlib
# matplotlib.use('Agg') # To be used if on Windows
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tic
from matplotlib import rcParams
import argparse

from numpy import pi
from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title, \
     grid, savefig, show


# Importing additional user-defined function

import UsefulFunctions as uf
import Production as br
import Amplitudes as am
import Detection as de
import LimitsList as lim





def tickform(x, pos, exp=8):
    'The two args are the value and tick position'
    return '%1.1f' % (x*1e+8)

rcParams['mathtext.fontset'] = 'stixsans'
mycolors=["#ff8865","#54d0ff","#5b8b00","#00786b"]


#---------- Truncate the color map to avoid color close to white
def LoadDataCS(file,exchange=False):
    s=-1
    data =  np.loadtxt(file)


    xi = data[:,1+s]
    CSData = data[:,2+s]
    if exchange:
        ErrHigh = data[:,3+s] # I did messed up the up and down correction in the bash file
        ErrLow = data[:,4+s]
    else:
        ErrHigh = data[:,4+s] # I did messed up the up and down correction in the bash file
        ErrLow = data[:,3+s]
    Cslow=CSData*(1-ErrLow/100.)
    Cshigh=CSData*(1+ErrHigh/100.)
    return(xi,CSData,Cslow,Cshigh)


####################################
#### Preparating for the plots####
####################################


parser = argparse.ArgumentParser()
parser.add_argument("Plotnbr")
args = parser.parse_args()

#### Am I plotting for Axial vector or for vector case?


Plotnumber=str(args.Plotnbr)

print("Making plots number: " ,Plotnumber)

s=-1


###### Various additional input parameters


############## -------- Masses and Decay rate for mesons
fPi = 0.1307;
MPi = 0.1349766; MEta = 0.547862; MEtap = 0.9577
MRho = 0.77526; MOmega = 0.78265
MDs = 2.00696; MD = 1.8648;MKs = 0.89581;MK = 0.49761

GamPi = 6.58*10**(-25)/(8.52*10**(-17))
GamEta =6.58*10**(-25)/(5.02*10**(-19))
GamEtap =6.58*10**(-25)/(3.32*10**(-21))
GamRho =6.58*10**(-25)/(4.45*10**(-24))
GamOmega =6.58*10**(-25)/(7.75*10**(-23))
BrDtoPiGamma = 0.38

#BrKtoPiGamma=0.38;*)

MJPsi = 3.097;
MCapitalUpsilon = 9.460; # 1S resonance for   Upsilon*)
MPhi = 1.019;
GamJPsi = 6.58*10**(-25)/(7.09*10**(-21))
GamCapitalUpsilon =6.58*10**(-25)/(1.22*10**(-20))
GamPhi = 6.58*10**(-25)/(1.54*10**(-22))

aem=1/137.



geffem={"gu11":2/3.,"gu22":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gd33":-1/3.,"gl11":-1.,"gl22":-1.}
geffZal={"gu11":1/2.,"gd11":-1/2.,"gd22":-1/2.,"gu22":1/2.,"gd33":-1/2,"gl11":-1/2,"gl22":-1/2}

######################### End of Header #####################3


ExperimentsList=np.array(["lsnd_decay","lsnd_decay_2","charm_decay",\
                          "seaquest_phase2_decay","faser_decay","mathusla_decay","ship_decay", \
                          "babar_monogam","belle2_monogam", \
                          "miniboone_scattering","sbnd_scattering","ship_scattering","nova_scattering", \
                          "sn1987_low_cooling","sn1987_high_cooling", \
                          "na62_invisibledecayPi0","bes_invisibledecayJPsi","babar_invisibledecayUpsilon",\
                          "atlas_monojet_down","atlas_monojet_up","lep_monogam"])



##################################################  #########################
#########################  Creating the  plots      #########################
######################### ######################### #########################



#------------------- Plotting

fig=plt.figure(1)
s = fig.add_subplot(1, 1, 1)

s.set_xscale("log", nonposx='clip')
s.set_yscale("log", nonposy='clip')
#----Adjusting the labels and color bar
s.set_xlabel(r'$M_{2}$  [GeV]',fontsize=18)
xdown=0.005;xup=1;
xbasic=np.linspace(xdown,xup,75)




f = tic.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))

majorLocator   = tic.FixedLocator([5e-3,1e-2,0.05,0.1,0.5,1,5])
majorFormatter = tic.FuncFormatter(g)
minorLocator   = tic.MultipleLocator(0.1)
s.xaxis.set_major_locator(majorLocator)
s.xaxis.set_major_formatter(majorFormatter)
s.yaxis.set_major_formatter(majorFormatter)

plt.grid(b=True,alpha=0.3, which='both', linestyle='-')
for tick in s.xaxis.get_major_ticks() + s.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
    tick.label2.set_fontsize(18)

################################################ Production ##################################################################
if Plotnumber == "prod1" :

#     geff = (1/2,-1/2,-1/2) # Z-aligned
    geff = (1/2,1/2,1)
    optype="AV"

   # ---------------- Generate the grid for plotting
    yup = 1e-9;
    ydown = 1e-17;
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    #----Adjusting the labels and color bar
    s.set_ylabel(r'BR [$M \rightarrow (\gamma) \chi \chi$ ]',fontsize=18)


    gu=geff[0];gd=geff[1];ge=geff[2];gs=geff[1]
    fuRho = 0.2215; fdRho = 0.2097;
    fuOmega = 0.1917; fdOmega = 0.201; fPhi = 0.233;  ## Caution, often sqrt(2) differences
    fOmegaeff = (gu* fuOmega + gd*fdOmega)/(np.sqrt(2.));
    fRhoeff = (gu*fuRho - gd*fdRho)/(np.sqrt(2.));
    fPi0Aeff = 2*gu + gd;
    fEtaAeff = np.sqrt(2/3)*(0.55*(2*gu - gd + 2*gs) + 0.35*(2*gu - gd - gs))
    fEtapAeff = np.sqrt(2/3)*(-0.1*(2*gu - gd + 2*gs) + 0.85*(2*gu - gd - gs))
    fPieff = fPi*((gu - gd)/np.sqrt(2))
    fEtaeff =  fPi*(0.5*(gu + gd - 2*gs) + 0.1*(gu + gd + gs))
    fEtapeff = fPi*(-0.2*(gu + gd - 2*gs) + 0.68*(gu + gd + gs))
    fRhoAeff = 0.01*(2*gu - gd)  ## Correction term for our decay, 0.01 contains the effect of the loop induced coupling  e/4/Pi^4*Arho, Arho = 1.2 *);
    fOmegaAeff =   0.01*(2*gu + gd);#


    Del=0;
    xmin=0.001;xmax= 1
    xi = uf.log_sample(xmin,xmax,500)

    Ndarkpi0=am.GamAV_MestoXX(xi/(1+Del),xi,MPi,fPieff,1000.)/br.GamPi
    Ndarketa=am.GamAV_MestoXX(xi/(1+Del),xi,MEta,fEtaeff,1000.)/br.GamEta
    Ndarketap=am.GamAV_MestoXX(xi/(1+Del),xi,MEtap,fEtapeff, 1000.)/br.GamEtap
    NDarkMes=Ndarkpi0+Ndarketa+Ndarketap
    Del=20
    Ndarkpi0_sat=am.GamAV_MestoXX(xi/(1+Del),xi,MPi,fPieff,1000.)/br.GamPi
    Ndarketa_sat=am.GamAV_MestoXX(xi/(1+Del),xi,MEta,fEtaeff,1000.)/br.GamEta
    Ndarketap_sat=am.GamAV_MestoXX(xi/(1+Del),xi,MEtap,fEtapeff, 1000.)/br.GamEtap
    NDarkMes_sat=Ndarkpi0_sat+Ndarketa_sat+Ndarketap_sat


    s.loglog(xi,Ndarkpi0,linestyle='-',linewidth=1.5,color='xkcd:blue',zorder=15,label=r"$\pi^0 \rightarrow \chi \chi$")
    s.loglog(xi,Ndarkpi0_sat,linestyle='--',linewidth=1.5,color='xkcd:blue',zorder=15)
    s.loglog(xi,Ndarketa,linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15,label=r"$\eta \rightarrow \chi \chi$")
    s.loglog(xi,Ndarketa_sat,linestyle='--',linewidth=1.5,color='xkcd:green',zorder=15)
    s.loglog(xi,Ndarketap,linestyle='-',linewidth=1.5,color='xkcd:orange',zorder=15,label=r"$\eta' \rightarrow \chi \chi$")
    s.loglog(xi,Ndarketap_sat,linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)


    # s.text(0.2, 0.95, r"$g_u:g_d=1:1, \delta_\chi=0$",fontsize=14, horizontalalignment='center',verticalalignment='center', transform=s.transAxes)
    s.text(0.2, 0.95, r"$g_u:g_d=\frac{1}{2}:\frac{1}{2}, \delta_\chi=0$",fontsize=14, horizontalalignment='center',verticalalignment='center', transform=s.transAxes)


    plt.tight_layout()
    l = plt.legend(loc=4,fontsize=14,fancybox=True, framealpha=0.9)
    l.set_zorder(20)
#     for lh in l.legendHandles:
#         lh.set_alpha(0.8)

    #---- Saving and showing on screen the Figure
#     plt.savefig('Output/Prod_Zal_AV.pdf')
    plt.savefig('Output/Prod_uni_AV.pdf')
    plt.show()
    plt.clf()
################################################ First Plot ##################################################################
elif Plotnumber == "prod2" :


    geff = (2/3,-1/3,-1) # em-aligned
#     geff = (1/3,1/3,-1/3) # B-L
    optype="V"

   # ---------------- Generate the grid for plotting
    yup = 1e-13;xup=1
    ydown = 1e-22;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))

    s.set_ylabel(r'BR [$M \rightarrow (\gamma) \chi \chi$ ]',fontsize=18)

    gu=geff[0];gd=geff[1];ge=geff[2];gs=geff[1]
    fuRho = 0.2215; fdRho = 0.2097;
    fuOmega = 0.1917; fdOmega = 0.201; fPhi = 0.233;  ## Caution, often sqrt(2) differences
    fOmegaeff = (gu* fuOmega + gd*fdOmega)/(np.sqrt(2.));
    fRhoeff = (gu*fuRho - gd*fdRho)/(np.sqrt(2.));
    fPi0Aeff = 2*gu + gd;
    fEtaAeff = np.sqrt(2/3)*(0.55*(2*gu - gd + 2*gs) + 0.35*(2*gu - gd - gs))
    fEtapAeff = np.sqrt(2/3)*(-0.1*(2*gu - gd + 2*gs) + 0.85*(2*gu - gd - gs))
    fPieff = fPi*((gu - gd)/np.sqrt(2))
    fEtaeff =  fPi*(0.5*(gu + gd - 2*gs) + 0.1*(gu + gd + gs))
    fEtapeff = fPi*(-0.2*(gu + gd - 2*gs) + 0.68*(gu + gd + gs))
    fRhoAeff = 0.01*(2*gu - gd)  ## Correction term for our decay, 0.01 contains the effect of the loop induced coupling  e/4/Pi^4*Arho, Arho = 1.2 *);
    fOmegaAeff =   0.01*(2*gu + gd);#


    Del=0;
    xmin=0.001;xmax= 1
    xi = uf.log_sample(xmin,xmax,500)
    Ndarkpi0=am.GamV_MestoXXgam(xi/(1+Del),xi,MPi,fPi0Aeff,1000.)/br.GamPi
    Ndarketa=am.GamV_MestoXXgam(xi/(1+Del),xi,MEta,fEtaAeff,1000.)/br.GamEta
    Ndarketap=am.GamV_MestoXXgam(xi/(1+Del),xi,MEtap,fEtapAeff, 1000.)/br.GamEtap
    Ndarkrho=am.GamV_MestoXX(xi/(1+Del),xi,MRho,fRhoeff,1000.)/br.GamRho
    Ndarkomega=am.GamV_MestoXX(xi/(1+Del),xi,MOmega,fOmegaeff, 1000.)/br.GamOmega
    NDarkMes=Ndarkpi0+Ndarketa+Ndarketap+Ndarkrho+Ndarkomega
    Del=20
    Ndarkpi0_sat=am.GamV_MestoXXgam(xi/(1+Del),xi,MPi,fPi0Aeff,1000.)/br.GamPi
    Ndarketa_sat=am.GamV_MestoXXgam(xi/(1+Del),xi,MEta,fEtaAeff,1000.)/br.GamEta
    Ndarketap_sat=am.GamV_MestoXXgam(xi/(1+Del),xi,MEtap,fEtapAeff, 1000.)/br.GamEtap
    Ndarkrho_sat=am.GamV_MestoXX(xi/(1+Del),xi,MRho,fRhoeff,1000.)/br.GamRho
    Ndarkomega_sat=am.GamV_MestoXX(xi/(1+Del),xi,MOmega,fOmegaeff, 1000.)/br.GamOmega
    NDarkMes_sat=Ndarkpi0_sat+Ndarketa_sat+Ndarketap_sat+Ndarkrho_sat+Ndarkomega_sat


    # -------- LLP searches

    s.loglog(xi,Ndarkpi0,linestyle='-',linewidth=1.5,color='xkcd:blue',zorder=15,label=r"$\pi^0 \rightarrow \gamma \chi \chi$")
    s.loglog(xi,Ndarkpi0_sat,linestyle='--',linewidth=1.,color='xkcd:blue',zorder=15)
    s.loglog(xi,Ndarketa,linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15,label=r"$\eta \rightarrow \gamma \chi \chi$")
    s.loglog(xi,Ndarketa_sat,linestyle='--',linewidth=1.,color='xkcd:green',zorder=15)
    s.loglog(xi,Ndarketap,linestyle='-',linewidth=1.5,color='xkcd:orange',zorder=15,label=r"$\eta' \rightarrow \gamma \chi \chi$")
    s.loglog(xi,Ndarketap_sat,linestyle='--',linewidth=1.,color='xkcd:orange',zorder=15)
    s.loglog(xi,Ndarkrho,linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15,label=r"$\rho \rightarrow \chi \chi$")
    s.loglog(xi,Ndarkrho_sat,linestyle='--',linewidth=1.,color='xkcd:grey',zorder=15)
    s.loglog(xi,Ndarkomega,linestyle='-',linewidth=1.5,color='xkcd:purple',zorder=15,label=r"$\omega \rightarrow \chi \chi$")
    s.loglog(xi,Ndarkomega_sat,linestyle='--',linewidth=1.,color='xkcd:purple',zorder=15)

    s.text(0.2, 0.95, r"$g_u:g_d=\frac{2}{3}:-\frac{1}{3}, \delta_\chi=0$",fontsize=14, horizontalalignment='center',verticalalignment='center', transform=s.transAxes)
#     s.text(0.2, 0.95, r"$g_u:g_d=\frac{1}{3}:\frac{1}{3}, \delta_\chi=0$",fontsize=14, horizontalalignment='center',verticalalignment='center', transform=s.transAxes)


    plt.tight_layout()
    l = plt.legend(loc=3,fontsize=14,fancybox=True, framealpha=0.9)
    l.set_zorder(20)
    #---- Saving and showing on screen the Figure
    plt.savefig('Output/Prod_em_V.pdf')
#     plt.savefig('Output/Prod_bml_V.pdf')
    plt.show()
    plt.clf()
################################################ Production Plot ##################################################################
# elif Plotnumber == "prod3" : ### Production CS at various beam facilities
#
#    # ---------------- Generate the grid for plotting
#     ydown = 1e-8;yup = 5e-4
#     xdown=0.1;xup=10
#     s.set_xlim((xdown,xup))
#     s.set_ylim((ydown,yup))
#     #----Adjusting the labels and color bar
#     s.set_xlabel(r'$M_{\chi}$  [GeV]',fontsize=18)
#     s.set_ylabel(r' $\sigma ( q q  \rightarrow \chi \chi)$  [pb]',fontsize=18)
#
#     xi_Numi_dd,CS_Numi_dd,CS_Numi_ddlow,CS_Numi_ddhigh =  LoadDataCS('DataForPaper/DirectEff_Final_V_FNALRing_dd.txt')
#     xi_Numi_uu,CS_Numi_uu,CS_Numi_uulow,CS_Numi_uuhigh =  LoadDataCS('DataForPaper/DirectEff_Final_V_FNALRing_uu.txt')
#     xi_SPS_dd,CS_SPS_dd,CS_SPS_ddlow,CS_SPS_ddhigh =  LoadDataCS('DataForPaper/DirectEff_Final_V_SPS_dd.txt')
#     xi_SPS_uu,CS_SPS_uu,CS_SPS_uulow,CS_SPS_uuhigh =  LoadDataCS('DataForPaper/DirectEff_Final_V_SPS_uu.txt')
#     xi_LHC_dd,CS_LHC_dd,CS_LHC_ddlow,CS_LHC_ddhigh =  LoadDataCS('DataForPaper/DirectEff_Final_V_LHC_dd.txt')
#     xi_LHC_uu,CS_LHC_uu,CS_LHC_uulow,CS_LHC_uuhigh =  LoadDataCS('DataForPaper/DirectEff_Final_V_LHC_uu.txt')
#
#     # --------
#
#     s.loglog(xi_SPS_uu,CS_SPS_uu,linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15,label=r"SPS - $u u \rightarrow \chi \chi$")
#     s.fill_between(xi_SPS_uu,CS_SPS_uulow,CS_SPS_uuhigh,color='xkcd:darkgreen',alpha=0.25, zorder=15)
#     s.loglog(xi_SPS_dd,CS_SPS_dd,linestyle='--',linewidth=1.,color='xkcd:green',zorder=15,label=r"$d d \rightarrow \chi \chi$")
#     s.fill_between(xi_SPS_dd,CS_SPS_ddlow,CS_SPS_ddhigh,color='xkcd:green',alpha=0.25, zorder=15)
#     s.loglog(xi_Numi_uu,CS_Numi_uu,linestyle='-',linewidth=1.5,color='xkcd:brick',zorder=15,label=r"NumI - $u u \rightarrow \chi \chi$")
#     s.fill_between(xi_Numi_uu,CS_Numi_uulow,CS_Numi_uuhigh,color='xkcd:brick',alpha=0.25, zorder=15)
#     s.loglog(xi_Numi_dd,CS_Numi_dd,linestyle='--',linewidth=1.,color='xkcd:orange',zorder=15,label=r"$d d \rightarrow \chi \chi$")
#     s.fill_between(xi_Numi_dd,CS_Numi_ddlow,CS_Numi_ddhigh,color='xkcd:orange',alpha=0.25, zorder=15)
#
#     plt.tight_layout()
#     l = plt.legend(loc=3,fontsize=14,fancybox=True, framealpha=0.9)
#     l.set_zorder(20)
#     #---- Saving and showing on screen the Figure
#     plt.savefig('Output/ProdDirect_em_V.pdf')
#   #  plt.savefig('Output/Prod_bml_V.pdf')
#     plt.show()
#     plt.clf()

# elif Plotnumber == "prod4" : ## Attenuation of the production CS at LHC due to the EFT limitations for low lambda
#
#    # ---------------- Generate the grid for plotting
#     ydown = 5e-3;yup = 10
#     xdown=15;xup=1e4
#     s.set_xlim((xdown,xup))
#     s.set_ylim((ydown,yup))
#
#     #----Adjusting the labels and color bar
#     s.set_xlabel(r'$\Lambda$  [GeV]',fontsize=18)
#     s.set_ylabel(r' $\sigma ( p p  \rightarrow \chi \chi) \cdot ( \Lambda ~/ 1 $TeV$)^4$  [pb]',fontsize=18)
#
#     sh=-1
#     xi_Mono,CS_Mono,CS_Monolow,CS_Monohigh =  LoadDataCS('DataForPaper/DirectEff_Final_MonoJet.txt')
#     xi_LHCLam,CS_LHCLam,CS_LHCLamlow,CS_LHCLamhigh =  LoadDataCS('DataForPaper/DirectEff_V_LHC_Lam.txt')
#
#     # -------- LLP searches
#
#     s.loglog(xi_Mono,CS_Mono,linestyle='-',linewidth=1.5,color='xkcd:blue',zorder=15,label=r"LHC - $ p p \rightarrow j \chi \chi$")
#     s.fill_between(xi_Mono,CS_Monolow,CS_Monohigh,color='xkcd:blue',alpha=0.25, zorder=15)
#     s.loglog(xi_LHCLam,CS_LHCLam,linestyle='--',linewidth=1.,color='xkcd:green',zorder=15,label=r"LHC - $p p \rightarrow \chi \chi$")
#     s.fill_between(xi_LHCLam,CS_LHCLamlow,CS_LHCLamhigh,color='xkcd:green',alpha=0.25, zorder=15)
#
#
#     plt.tight_layout()
#     l = plt.legend(loc=4,fontsize=14,fancybox=True, framealpha=0.9)
#     l.set_zorder(20)
#     #---- Saving and showing on screen the Figure
#     plt.savefig('Output/ProdDirect_LHCLam.pdf')
#   #  plt.savefig('Output/Prod_bml_V.pdf')
#     plt.show()
#     plt.clf()



################################################## Full production plots  ##############################
elif Plotnumber== "prod5":

    # Full production for Vector case
    yup = 1e13
    ydown = 1
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    geff={"gu11":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gl11":-1.,"gl22":-1.}


    xProd_Eff_charm,NProd_Eff_charm = br.NProd(0.1,"charm",geff,"V")
#     xProd_Eff_seaquest,NProd_Eff_seaquest = br.NProd(0.1,"seaquest_phase2",(2/3,-1/3,-1),"V")
    xProd_Eff_lsnd,NProd_Eff_lsnd = br.NProd(0.1,"lsnd",geff,"V")

    xProd_DP_charm,NProd_DP_charm = br.NProd_DP("charm")
    xProd_DP_LSND,NProd_DP_LSND = br.NProd_DP("lsnd")


    # -------- Production  in lower energies detector
    s.loglog(xProd_Eff_charm,1e4*NProd_Eff_charm,linestyle='-',linewidth=2.5,color='xkcd:green',zorder=15,label=r"CHARM, $\Lambda=100$ GeV")
    s.loglog(xProd_DP_charm,NProd_DP_charm/1e4,linestyle='--',linewidth=1.5,color='xkcd:darkgreen',zorder=15,label=r"CHARM, $\varepsilon=10^{-5}$")
    s.loglog(xProd_Eff_lsnd,1e4*NProd_Eff_lsnd,linestyle='-',linewidth=2.5,color='xkcd:orange',zorder=15,label=r"LSND, $\Lambda=100$ GeV")
    s.loglog(xProd_DP_LSND,NProd_DP_LSND/1e4,linestyle='--',linewidth=1.5,color='xkcd:rust',zorder=15,label=r"LSND, $\varepsilon=10^{-5}$ ")

    #----Adjusting the labels and color bar

    s.set_ylabel(r'NoE (production)',fontsize=18)
    s.set_xlim((0.006,10))
    s.set_ylim((ydown,yup))

    # ----Axes settings
    f = tic.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    majorLocator   = tic.FixedLocator([1e-3,1e-2,0.1,1,10,100])
    majorFormatter = tic.FuncFormatter(g)
    minorLocator   = tic.MultipleLocator(0.1)
    s.xaxis.set_major_locator(majorLocator)
    s.xaxis.set_major_formatter(majorFormatter)
    s.yaxis.set_major_formatter(majorFormatter)

    s.text(0.52,0.03,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}, \delta_\chi=0.1$', style='italic',fontsize=13,transform = s.transAxes, zorder=20)

    plt.tight_layout()
    l= plt.legend(loc=1,fontsize=13,fancybox=True, framealpha=0.6)
    l.set_zorder(20)
    for lh in l.legendHandles:
        lh.set_alpha(0.8)

    #---- Saving and showing on screen the Figure
    plt.savefig('Output/Production_V_BD.pdf')
    plt.show()


################################################## Full production plots  ##############################
elif Plotnumber=="prod6":

    # Full production for Vector case
    yup = 1e13
    ydown = 1e2

    geff={"gu11":1/2.,"gd11":-1/2.,"gd22":-1/2.,"gl11":-1/2.,"gl22":-1/2.}
    xProd_Eff_charm,NProd_Eff_charm = br.NProd(0.1,"charm",geff,"AV")
    xProd_Eff_lsnd,NProd_Eff_lsnd = br.NProd(0.1,"lsnd",geff,"AV")

    xProd_DP_charm,NProd_DP_charm = br.NProd_DP("charm")
    xProd_DP_LSND,NProd_DP_LSND = br.NProd_DP("lsnd")

    # -------- Production  in lower energies detector
    s.loglog(xProd_Eff_charm,16*NProd_Eff_charm,linestyle='-',linewidth=2.5,color='xkcd:green',zorder=15,label=r"CHARM, $\Lambda=500$ GeV")
    s.loglog(xProd_DP_charm,NProd_DP_charm/1e4,linestyle='--',linewidth=1.5,color='xkcd:darkgreen',zorder=15,label=r"CHARM, $\varepsilon=10^{-5}$")
    s.loglog(xProd_Eff_lsnd,16*NProd_Eff_lsnd,linestyle='-',linewidth=2.5,color='xkcd:orange',zorder=15,label=r"LSND, $\Lambda=500$ GeV")
    s.loglog(xProd_DP_LSND,NProd_DP_LSND/1e4,linestyle='--',linewidth=1.5,color='xkcd:rust',zorder=15,label=r"LSND, $\varepsilon=10^{-5}$ ")


    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{\chi}$  [GeV]',fontsize=18)
    s.set_ylabel(r'NoE (production)',fontsize=18)
    s.set_xlim((0.005,10))
    s.set_ylim((ydown,yup))

    # ----Axes settings
    f = tic.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    majorLocator   = tic.FixedLocator([1e-3,1e-2,0.1,1,10,100])
    majorFormatter = tic.FuncFormatter(g)
    minorLocator   = tic.MultipleLocator(0.1)
    s.xaxis.set_major_locator(majorLocator)
    s.xaxis.set_major_formatter(majorFormatter)
    s.yaxis.set_major_formatter(majorFormatter)

    s.text(0.52,0.03,r'$g_e:g_d:g_u = -\frac{1}{2}:-\frac{1}{2}:\frac{1}{2}, \delta_\chi=0.1$', style='italic',fontsize=13,transform = s.transAxes, zorder=20)


    l= plt.legend(loc=1,fontsize=13,fancybox=True, framealpha=0.6)
    l.set_zorder(20)
    for lh in l.legendHandles:
        lh.set_alpha(0.8)

    plt.tight_layout()
    #---- Saving and showing on screen the Figure
    plt.savefig('Output/Production_AV_BD.pdf')
    plt.show()



################################################## Decay plots  ##############################
elif Plotnumber== "decay1":

    # Full production for Vector case
    yup = 1e0
    ydown = 1e-12

    delta=0.5
    xmin=0.005/(2+delta);xmax= 12.5/(1.99999+delta)
    mx = uf.log_sample(xmin,xmax,1000)
    mx2=mx*(1+delta)
    Lam=5000.;optype="AV"
    geff={"gu11":1/2.,"gd11":-1/2.,"gd22":-1/2.,"gl11":-1/2.,"gl22":-1/2.}
    Probtot=am.GamHDS(mx, delta, Lam,geff,optype)/(6.58*10**(-25)*3*10**8)

    Probee=(delta*np.abs(mx)>2*de.me)*am.GamHDSllAV(mx, delta,de.me,Lam,geff["gl11"])/(6.58*10**(-25)*3*10**8)
    Probmumu=(delta*np.abs(mx)>2*de.mmuon)*am.GamHDSllAV(mx, delta,de.mmuon,Lam,geff["gl22"])/(6.58*10**(-25)*3*10**8)
#     Probeta=(delta*np.abs(mx)>2*de.mmuon)*de.GamHDSllAVThr(mx, delta,de.mmuon,Lam,geff)/(6.58*10**(-25)*3*10**8)
    Probpi=(delta*np.abs(mx)>br.MPi)*am.GamHDSPiAV(mx, delta,Lam,geff)/(6.58*10**(-25)*3*10**8)
    Probeta=(delta*np.abs(mx)>br.MEta)*am.GamHDSEtaAV(mx, delta,Lam,geff)/(6.58*10**(-25)*3*10**8)
    Probetap=(delta*np.abs(mx)>br.MEtap)*am.GamHDSEtapAV(mx, delta,Lam,geff)/(6.58*10**(-25)*3*10**8)

    # -------- Production  in lower energies detector
    s.loglog(mx,Probee,linestyle='--',linewidth=1.5,color='xkcd:darkblue',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 ee$")
    s.loglog(mx,Probmumu,linestyle='--',linewidth=1.5,color='xkcd:darkgreen',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \mu \mu$")
    s.loglog(mx,Probpi,linestyle='-.',linewidth=1.5,color='xkcd:orange',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \pi^0$")
    s.loglog(mx,Probeta,linestyle=':',linewidth=2,color='xkcd:rust',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \eta^0$")
    s.loglog(mx,Probetap,linestyle='-.',linewidth=1.,color='xkcd:purple',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \eta^\prime$")
    s.loglog(mx,Probtot,linestyle='-',linewidth=2.5,color='xkcd:black',zorder=15,label=r"Total")

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{2}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$(1$ m$)~/(c \tau_{\chi_2})$',fontsize=18)
    s.set_xlim((0.05,5))
    s.set_ylim((ydown,yup))


    s.text(0.4,0.94,r'$g_l:g_d:g_u = -\frac{1}{2}:-\frac{1}{2}:\frac{1}{2}$, $\delta_\chi=0.5$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    l= plt.legend(loc=2,fontsize=15,fancybox=True, framealpha=0.6)
    l.set_zorder(20)
    for lh in l.legendHandles:
        lh.set_alpha(0.8)

    #---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/Decay_AV.pdf')
    plt.show()


################################################## Decay plots  ##############################
elif Plotnumber== "decay2":

    # Full production for Vector case
    yup = 1e0
    ydown = 1e-9
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    delta=0.5
    xmin=0.005/(2+delta);xmax= 12.5/(1.99999+delta)
    mx = uf.log_sample(xmin,xmax,1000)
    mx2=mx*(1+delta)
    Lam=5000.;
    geff={"gu11":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gl11":-1.,"gl22":-1.};
    optype="V"
    Probtot=am.GamHDS(mx, delta, Lam,geff,optype)/(6.58*10**(-25)*3*10**8)

    Probee=(delta*np.abs(mx)>2*de.me)*am.GamHDSllVV(mx, delta,de.me,Lam,geff["gl11"])/(6.58*10**(-25)*3*10**8)
    Probmumu=(delta*np.abs(mx)>2*de.mmuon)*am.GamHDSllVV(mx, delta,de.mmuon,Lam,geff["gl22"])/(6.58*10**(-25)*3*10**8)
#     Probeta=(delta*np.abs(mx)>2*de.mmuon)*de.GamHDSllAVThr(mx, delta,de.mmuon,Lam,geff)/(6.58*10**(-25)*3*10**8)
    Probpipi=(delta*np.abs(mx)>2*br.MPi)*am.GamHDSpipiVV(mx, delta,Lam,geff)/(6.58*10**(-25)*3*10**8)
    Probrho=(delta*np.abs(mx)>br.MRho)*am.GamHDSrhoV(mx, delta,Lam,geff)/(6.58*10**(-25)*3*10**8)
    Probomega=(delta*np.abs(mx)>br.MOmega)*am.GamHDSomegaV(mx, delta,Lam,geff)/(6.58*10**(-25)*3*10**8)


    # -------- Production  in lower energies detector
    s.loglog(mx2,Probee,linestyle='--',linewidth=2,color='xkcd:darkblue',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 ee$")
    s.loglog(mx2,Probmumu,linestyle='--',linewidth=1.5,color='xkcd:darkgreen',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \mu \mu$")
    s.loglog(mx2,Probpipi,linestyle=':',linewidth=1.5,color='xkcd:orange',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \pi \pi$")
    s.loglog(mx2,Probrho,linestyle='-.',linewidth=2,color='xkcd:rust',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \rho$")
    s.loglog(mx2,Probomega,linestyle='-.',linewidth=1.,color='xkcd:purple',zorder=15,label=r"$\chi_2 \rightarrow \chi_1 \omega$")
    s.loglog(mx2,Probtot,linestyle='-',linewidth=2.5,color='xkcd:black',zorder=15,label=r"Total")

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{2}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$(1$ m$)~/(c \tau_{\chi_2})$',fontsize=18)
    s.set_xlim((0.2,5))
    s.set_ylim((ydown,yup))
    s.text(0.4,0.94,r'$g_l:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}$, $\delta_\chi=0.5$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    l= plt.legend(loc=2,fontsize=15,fancybox=True, framealpha=0.6)
    l.set_zorder(20)
    for lh in l.legendHandles:
        lh.set_alpha(0.8)

    #---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/Decay_V.pdf')
    plt.show()




################################################### Second categories of plots ################################################
elif Plotnumber == "limit1" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    lim.faser_decay.combthr = 1.1
    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,geffZal,"AV",True)

    xfaserlow=Lim['faser_decay'][0][Lim['faser_decay'][0]<10]
    yfaserlow=Lim['faser_decay'][1][Lim['faser_decay'][0]<10]

    s.loglog(Lim['lsnd_decay'][0],Lim['lsnd_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15)
    s.fill_between(Lim['lsnd_decay'][0],ydown,Lim['lsnd_decay'][1],color='xkcd:green',alpha=0.5, zorder=15)
    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
    s.loglog(Lim['charm_decay'][0],Lim['charm_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:darkgreen',alpha=0.5, zorder=15)
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
#     print(Lim['sn1987_low_cooling'][1])
    gp=np.logical_and(Lim['sn1987_high_cooling'][1]>1.1*Lim['sn1987_low_cooling'][1],Lim['sn1987_high_cooling'][1]>1)
    s.loglog(Lim['sn1987_high_cooling'][0][gp],Lim['sn1987_high_cooling'][1][gp],linestyle='-',linewidth=1,color='xkcd:purple',zorder=15)
    s.loglog(Lim['sn1987_low_cooling'][0][gp],Lim['sn1987_low_cooling'][1][gp],linestyle='--',linewidth=0.5,color='xkcd:purple',zorder=15)
    s.fill_between(Lim['sn1987_low_cooling'][0][gp],Lim['sn1987_low_cooling'][1][gp],Lim['sn1987_high_cooling'][1][gp],color='xkcd:purple',alpha=0.15, zorder=15)

    # ----- Invisible decay of pions
#
#     s.loglog(Lim['e949_pi0decay'][0],Lim['e949_pi0decay'][1],linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
    s.loglog(Lim['na62_invisibledecayPi0'][0],Lim['na62_invisibledecayPi0'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['na62_invisibledecayPi0'][0],ydown,Lim['na62_invisibledecayPi0'][1],color='xkcd:grey',alpha=0.75, zorder=15)

    s.text(1.1,37,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,80,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.01,1.9e3,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=20)
    s.text(0.037,700,r'LSND',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=30)
    s.text(0.15,520,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=40)
    s.text(0.14,310,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=45)
#     s.text(0.15,1130,r'SeaQuest',fontsize=10,color='xkcd:rust', zorder=40,rotation=35)
    s.text(1.7,5.0e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=20)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -\frac{1}{2}:-\frac{1}{2}:\frac{1}{2}, \delta_\chi=10$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    s.text(0.02,160,r'NA62',fontsize=10, zorder=40,rotation=20 )

    #----Adjusting the labels and color bar
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

#---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/FirstGen_AV.pdf')
    plt.show()

elif Plotnumber == "limit2" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')


    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,geffem,"V",True)


# Need to cut the FASER limits in two
    xfaserlow=Lim['faser_decay'][0][Lim['faser_decay'][0]<1.1]
#     xfaserhigh=Lim['faser_decay'][0][Lim['faser_decay'][0]>2.5]
    yfaserlow=Lim['faser_decay'][1][Lim['faser_decay'][0]<1.1]
    yfaserhigh=Lim['faser_decay'][1][Lim['faser_decay'][0]>2.5]


    s.loglog(Lim['lsnd_decay'][0],Lim['lsnd_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15)
    s.fill_between(Lim['lsnd_decay'][0],ydown,Lim['lsnd_decay'][1],color='xkcd:green',alpha=0.5, zorder=15)
    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
    s.loglog(Lim['charm_decay'][0],Lim['charm_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:darkgreen',alpha=0.5, zorder=15)
#     s.loglog(Lim['seaquest_phase2_decay'][0],Lim['seaquest_phase2_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:red',zorder=15)
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

#     s.loglog(Lim['bes_invisibledecayJPsi'][0],Lim['bes_invisibledecayJPsi'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
#     s.fill_between(Lim['bes_invisibledecayJPsi'][0],ydown,Lim['bes_invisibledecayJPsi'][1],color='xkcd:grey',alpha=0.75, zorder=15)

    s.loglog(Lim['babar_invisibledecayUpsilon'][0],Lim['babar_invisibledecayUpsilon'][1],linestyle='-',linewidth=1.1,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['babar_invisibledecayUpsilon'][0],ydown,Lim['babar_invisibledecayUpsilon'][1],color='xkcd:grey',alpha=0.25, zorder=15)


#     s.loglog(Lim['e949_pi0decay'][0],Lim['e949_pi0decay'][1],linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
#     s.loglog(Lim['na62_invisibledecayPi0'][0],Lim['na62_invisibledecayPi0'][1],linestyle='--',linewidth=1.5,color='xkcd:grey',zorder=15)
    s.text(0.08,220,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=40)
    s.text(1.1,52,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,110,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.01,2500,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=0)
    s.text(0.015,70,r'LSND',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=20)
#     s.text(0.15,610,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)
    s.text(0.28,510,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=35)
    s.text(1.,4.e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=30)
    s.text(1.1,160,r'BaBar ($\Upsilon \to$ inv)' ,fontsize=10, zorder=50)

    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}, \delta_\chi=10$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_2$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

#---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/FirstGen_V.pdf')
    plt.show()


################################################## Fourth Plot ##############################
elif Plotnumber=="limit3a": ##### ---------------- Lam vs g

    xu = (4*3.1416)**2;xd = 0.01;
    yu=20e3;yd=10

    xi=uf.log_sample(xd,xu)



    s.loglog(xi,150*np.sqrt(xi),linestyle='-',linewidth=2.5,color='xkcd:green',zorder=15) # LSND
    s.loglog(xi,185*np.sqrt(xi),linestyle='-',linewidth=2.,color='xkcd:darkgreen',zorder=15) # CHARM
    s.fill_between(xi,yd,185*np.sqrt(xi),color='xkcd:darkgreen',alpha=0.5, zorder=15)


    LEPOK=(480*np.sqrt(xi)>200)
    s.loglog(xi[LEPOK],480*np.sqrt(xi[LEPOK]),linestyle='-',linewidth=2.5,color='xkcd:blue',zorder=15) # LEP
    s.fill_between(xi[LEPOK],200,480*np.sqrt(xi[LEPOK]),color='xkcd:blue',alpha=0.5, zorder=15)


    s.loglog(lim.xi_ATLAS,lim.Lim_ATLAS_Up,linestyle='-',linewidth=2.5,color='xkcd:grey',zorder=15) # LHC
    s.loglog(lim.xi_ATLAS,lim.Lim_ATLAS_Down,linestyle='-',linewidth=2.5,color='xkcd:grey',zorder=15)
    s.fill_between(lim.xi_ATLAS,lim.Lim_ATLAS_Down,lim.Lim_ATLAS_Up,color='xkcd:grey',alpha=0.5, zorder=15)

    s.loglog(xi,820*np.sqrt(xi),linestyle='--',linewidth=2.5,color='xkcd:orange',zorder=15) # SHIP optimistic

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$g$ ',fontsize=18)
    s.set_ylabel(r'$\Lambda ~[GeV]$',fontsize=18)
    s.set_xlim((xd,xu))
    s.set_ylim((yd,yu))

    # -------- Some text
    s.text(0.02,135,'SHIP',fontsize=12,color='xkcd:rust', zorder=50,rotation=25)
    s.text(0.02,15,'LSND',fontsize=12,color='xkcd:darkgreen', zorder=50,rotation=25)
    s.text(0.02,32,'CHARM',fontsize=12,color='xkcd:darkgreen', zorder=50,rotation=25)
    s.text(0.5,380,'LEP (DELPHI)',fontsize=12,color='xkcd:navy', zorder=50,rotation=25)
    s.text(25,4200,'LHC (ATLAS)',fontsize=12,color='xkcd:black', zorder=50,rotation=25)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}, \delta_\chi = 10$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    s.text(0.02,0.87,r'$M_2 = 100$ MeV', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    # ----Axes settings
    plt.tight_layout()

    #---- Saving and showing on screen the Figure
    plt.savefig('Output/LimV_gLam.pdf')
    plt.show()

################################################## Fifth Plot ##############################
elif Plotnumber == "limit3b":  ##### ---------------- Lam vs g but at 600 MeV
    s.set_yscale('log')
    s.set_xscale('log')

    xu = 10;xd = 0.02;
    yu=5e3;yd=10;
    xbasic=np.linspace(xd,xu,75)

    optype="V"
    Del_sat=10
    geff=geffem

    xi_charm,EffLim_charm_0p1 = de.MakeEffLimits("charm", lim.charm_decay.mx_ini, lim.charm_decay.lim_ini, geff, optype, 0.1 )
    xi_LSND,EffLim_LSND_op15 = de.MakeEffLimits("lsnd", lim.lsnd_decay.mx_ini, lim.lsnd_decay.lim_ini, geff, optype, 0.1 )
    xi_SHIP,EffLim_SHIP_0p1 = de.GetNaiveDecayLimits( 0.1, "ship",10,geff,optype)
    xi_FASER,EffLim_FASER_0p1 = de.MakeEffLimits("faser", lim.faser_decay.mx_ini, lim.faser_decay.lim_ini, geff, optype, 0.1 )
    xi_MAT,EffLim_MAT_0p1 = de.MakeEffLimits("mathusla", lim.mathusla_decay.mx_ini, lim.mathusla_decay.lim_ini, geff, optype, 0.1 )

    ### Creating the data

    Mx=0.6
#     print(xi_FASER,EffLim_FASER_0p1)

    M1overM2_FASER,EffLimFin_FASER,EffLimFin_Low_FASER = de.MakeTableLimit_OneMass( xi_FASER, EffLim_FASER_0p1,"faser",Mx,0.1,geff,optype)
    M1overM2_LSND,EffLimFin_LSND,EffLimFin_Low_LSND = de.MakeTableLimit_OneMass( xi_LSND, EffLim_LSND_op15,"lsnd",Mx,0.15,geff,optype)

    M1overM2_charm,EffLimFin_charm,EffLimFin_Low_charm = de.MakeTableLimit_OneMass( xi_charm, EffLim_charm_0p1,"charm",Mx,0.1,geff,optype)
    M1overM2_charm_combined, EffLimFin_charm_full=uf.CombineUpDown(M1overM2_charm,EffLimFin_Low_charm,EffLimFin_charm,1)

    xi_SHIP,EffLim_SHIP_op1 = de.GetNaiveDecayLimits( 0.1, "ship",10,geff,optype)
    M1overM2_SHIP,EffLimFin_SHIP,EffLimFin_Low_SHIP = de.MakeTableLimit_OneMass(xi_SHIP, EffLim_SHIP_op1,"ship",Mx,0.1,geff,optype)

    ### Plotting the results

    # Long-lived particles

    s.loglog(M1overM2_FASER,EffLimFin_FASER,linestyle='--',linewidth=1.5,color='xkcd:indigo',zorder=15)
    s.loglog(M1overM2_FASER,EffLimFin_Low_FASER,linestyle='--',linewidth=1.5,color='xkcd:indigo',zorder=15)
    s.fill_between(M1overM2_FASER,EffLimFin_Low_FASER,EffLimFin_FASER,color='xkcd:indigo',alpha=0.1, zorder=15)

    s.loglog(M1overM2_charm,EffLimFin_charm,linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.loglog(M1overM2_charm,EffLimFin_Low_charm,linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.fill_between(M1overM2_charm,EffLimFin_Low_charm,EffLimFin_charm,color='xkcd:darkgreen',alpha=0.25, zorder=15)

    # Missing enenrgy and mono-photon searches

    s.fill_between(xbasic,200,480,color='xkcd:blue',alpha=0.5, zorder=15)
    s.axhline(200,linestyle='--',linewidth=1.,color='xkcd:blue',zorder=15)
    s.axhline(480,linestyle='--',linewidth=1.5,color='xkcd:blue',zorder=15)

    s.axhline(48,linestyle='-',linewidth=1.,color='xkcd:grey',zorder=15)
    s.axhline(102,linestyle='--',linewidth=1.,color='xkcd:grey',zorder=15)
    s.fill_between(xbasic,yd,48,color='xkcd:grey',alpha=0.75, zorder=15)
#    print(Lamlim)
    #----Adjusting the labels and color bar
    s.set_xlim(xd,xu)
    s.set_ylim(yd,yu)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)
    s.set_xlabel(r'$\delta_\chi \equiv (|M_{2}| - |M_{1}|)/|M_1|$',fontsize=18)

    # -------- Some text
    s.text(2.3,50,r'BaBar' ,fontsize=10, zorder=50)
    s.text(2.3,110,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(0.025,230,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    s.text(0.02,0.87,r'$M_2 = 600$ MeV', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    s.text(2.3,800,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40)
    s.text(2.3,1900,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40)
    # ----Axes settings
    f = tic.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))

    majorLocator   = tic.FixedLocator([0.02,0.05,0.1,0.2,0.5,1,2,5,10])
    majorFormatter = tic.FuncFormatter(g)
    s.xaxis.set_major_locator(majorLocator)

    plt.grid(b=True,alpha=0.3, which='both', linestyle='-')

    #---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/LimM1M2_V_600MeV.pdf')
    plt.show()



##############################    V coupling in the proto-phobic case

elif Plotnumber == "limit4" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)

    geffphobic={"gu11":-1/3.,"gd11":2/3.,"gd22":2/3.,"gl11":-1.,"gl22":-1.}


    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,geffphobic,"V",True)
#     Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,(2/3,-1/3,-1),"V",True)
    xfaserlow=Lim['faser_decay'][0][Lim['faser_decay'][0]<1.1]
    xfaserhigh=Lim['faser_decay'][0][Lim['faser_decay'][0]>2.]
    yfaserlow=Lim['faser_decay'][1][Lim['faser_decay'][0]<1.1]
    yfaserhigh=Lim['faser_decay'][1][Lim['faser_decay'][0]>2.]

#     s.loglog(Lim['lsnd_decay'][0],Lim['lsnd_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15)
#     s.fill_between(Lim['lsnd_decay'][0],ydown,Lim['lsnd_decay'][1],color='xkcd:green',alpha=0.5, zorder=15)
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


    s.text(1.1,52,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,110,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.01,2500,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=0)
    s.text(0.15,600,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)
    s.text(0.3,540,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=35)
    s.text(1.,4.2e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=30)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}, \delta_\chi=10$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
    s.text(0.12,320,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=45)

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_2$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)


    plt.tight_layout()

#---- Saving and showing on screen the Figure

    plt.savefig('Output/FirstGen_V_phobic.pdf')
    plt.show()



##############################    AV coupling in the pion phobic case

elif Plotnumber == "limit5" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)

    geffphobicAV={"gu11":1,"gd11":1.,"gd22":1.,"gl11":1.,"gl22":1.}



    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,geffphobicAV,"AV",True)


    xfaserlow=Lim['faser_decay'][0][Lim['faser_decay'][0]<1.1]
    xfaserhigh=Lim['faser_decay'][0][Lim['faser_decay'][0]>2.]
    yfaserlow=Lim['faser_decay'][1][Lim['faser_decay'][0]<1.1]
    yfaserhigh=Lim['faser_decay'][1][Lim['faser_decay'][0]>2.]

#     Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,(2/3,-1/3,-1),"V",True)

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


    s.text(1.1,52,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,110,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.15,570,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=45)
    s.text(0.15,1200,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)
    s.text(0.35,1200,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=20)
    s.text(1.,4.9e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=30)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = 1:1:1 , \delta_\chi=10$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_2$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

    plt.tight_layout()

#---- Saving and showing on screen the Figure

    plt.savefig('Output/FirstGen_AV_phobic.pdf')
    plt.show()

##############################    AV coupling Z-aligned aligned no splitting

elif Plotnumber == "limit6" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)


    Lim,LabelLimit = lim.GetLimits(ExperimentsList,0.01,geffZal,"AV",True)
#     Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,(2/3,-1/3,-1),"V",True)

    s.loglog(Lim['miniboone_scattering'][0],Lim['miniboone_scattering'][1],linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15)
    s.fill_between(Lim['miniboone_scattering'][0],ydown,Lim['miniboone_scattering'][1],color='xkcd:green',alpha=0.5, zorder=15)
    s.loglog(Lim['sbnd_scattering'][0],Lim['sbnd_scattering'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
    s.loglog(Lim['nova_scattering'][0],Lim['nova_scattering'][1],linestyle='--',linewidth=1.5,color='xkcd:rust',zorder=15)

    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:green',zorder=15)
    s.loglog(Lim['charm_decay'][0],Lim['charm_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:darkgreen',alpha=0.5, zorder=15)

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

    # ----- Invisible decay of pions
#
#     s.loglog(Lim['e949_pi0decay'][0],Lim['e949_pi0decay'][1],linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
    s.loglog(Lim['na62_invisibledecayPi0'][0],Lim['na62_invisibledecayPi0'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['na62_invisibledecayPi0'][0],ydown,Lim['na62_invisibledecayPi0'][1],color='xkcd:grey',alpha=0.75, zorder=15)

    s.text(1.1,37,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,80,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)

    s.text(0.01,1700,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=20)
#     s.text(0.015,60,r'LSND',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=20)
    s.text(0.01,180,r'NA62',fontsize=10, zorder=40,rotation=20 )
#     s.text(0.15,650,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)

    s.text(0.02,15,r'MiniBooNE',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=20 )
    s.text(0.01,20,r'SBND',fontsize=10,color='xkcd:orange', zorder=40,rotation=30 )
    s.text(0.01,30,r'NOvA',fontsize=10,color='xkcd:rust', zorder=40,rotation=20 )

    s.text(0.55,35,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=-30)
    s.text(0.55,175,'SHIP',fontsize=10,color='xkcd:green', zorder=50,rotation=30)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -\frac{1}{2}:-\frac{1}{2}:\frac{1}{2}, \delta_\chi=0.01$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{2}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)
    plt.tight_layout()

#---- Saving and showing on screen the Figure

    plt.savefig('Output/FirstGen_AV_tinysplit.pdf')
    plt.show()

##############################    V coupling em aligned no splitting

elif Plotnumber == "limit7" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,0.05,geffem,"V",True)

#     s.loglog(Lim['ship_scattering'][0],Lim['ship_scattering'][1],linestyle='--',linewidth=1.5,color='xkcd:rust',zorder=15)
    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:green',zorder=15)
    s.loglog(Lim['charm_decay'][0],Lim['charm_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:darkgreen',alpha=0.5, zorder=15)

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

    # ----- Invisible decay of pions
#

    s.loglog(Lim['babar_invisibledecayUpsilon'][0],Lim['babar_invisibledecayUpsilon'][1],linestyle='-',linewidth=1.1,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['babar_invisibledecayUpsilon'][0],ydown,Lim['babar_invisibledecayUpsilon'][1],color='xkcd:grey',alpha=0.25, zorder=15)


    s.text(1.1,160,r'BaBar ($\Upsilon \to$ inv)' ,fontsize=10, zorder=50)
    s.text(0.01,52,r'BaBar' ,fontsize=10, zorder=50)
    s.text(0.01,110,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(0.01,220,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.01,2000,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=0)

    s.text(0.5,100,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=-30)
    s.text(1.3,850,'SHIP',fontsize=10,color='xkcd:green', zorder=50,rotation=25)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}, \delta_\chi=0.05$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{2}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

    plt.tight_layout()

#---- Saving and showing on screen the Figure

    plt.savefig('Output/FirstGen_V_tinysplit.pdf')
    plt.show()

elif Plotnumber == "limit8" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    lim.faser_decay.combthr = 1.1
    Lim,LabelLimit = lim.GetLimits(ExperimentsList,0.20,geffZal,"AV",True)


    # --------- Decay searches
    # Need to cut the FASER limits in two
    xmatlow=Lim['mathusla_decay'][0][Lim['mathusla_decay'][0]<10]
    xmathigh=Lim['mathusla_decay'][0][Lim['mathusla_decay'][0]>2.5]
    ymatlow=Lim['mathusla_decay'][1][Lim['mathusla_decay'][0]<10]
    ymathigh=Lim['mathusla_decay'][1][Lim['mathusla_decay'][0]>2.5]

    s.fill_between(Lim['lsnd_decay'][0],ydown,Lim['lsnd_decay'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.loglog(Lim['seaquest_phase2_decay'][0],Lim['seaquest_phase2_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:red',zorder=15)
    s.loglog(Lim['faser_decay'][0],Lim['faser_decay'][1],linestyle='-.',linewidth=1.5,color='xkcd:indigo',zorder=15)

    s.loglog(Lim['mathusla_decay'][0],Lim['mathusla_decay'][1],linestyle='-.',linewidth=1.5,color='xkcd:blue',zorder=15)


    # -------- Mono photon at LEP
    s.fill_between(Lim['lep_monogam'][0],200,Lim['lep_monogam'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.axhline(200,linestyle='--',linewidth=1.,color='xkcd:grey',zorder=15)
    # -------- Missing energy searches

    s.fill_between(Lim['babar_monogam'][0],ydown,Lim['babar_monogam'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.loglog(Lim['belle2_monogam'][0],Lim['belle2_monogam'][1],linestyle='--',linewidth=1.5,color='xkcd:black',zorder=15)

    # ------ Self consistency
    s.loglog(xbasic,2*xbasic,linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
    s.fill_between(xbasic,ydown,2*xbasic,color='xkcd:grey',alpha=0.75, zorder=10)

    # -------- SN bounds
#     print(Lim['sn1987_low_cooling'][1])
    gp=np.logical_and(Lim['sn1987_high_cooling'][1]>1.1*Lim['sn1987_low_cooling'][1],Lim['sn1987_high_cooling'][1]>1)
    s.fill_between(Lim['sn1987_low_cooling'][0][gp],Lim['sn1987_low_cooling'][1][gp],Lim['sn1987_high_cooling'][1][gp],color='xkcd:grey',alpha=0.75, zorder=10)

    # ----- Invisible decay of pions
#
    s.fill_between(Lim['na62_invisibledecayPi0'][0],ydown,Lim['na62_invisibledecayPi0'][1],color='xkcd:grey',alpha=0.75, zorder=10)

    s.text(1.1,80,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(0.14,70,r'MATHUSLA',fontsize=10,color='xkcd:blue', zorder=40,rotation=35)
    s.text(0.14,210,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=45)
    s.text(0.1,480,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)
    s.text(1.,1.4e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=20)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -\frac{1}{2}:-\frac{1}{2}:\frac{1}{2}, \delta_\chi = 0.2$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
#     s.text(0.02,160,r'NA62',fontsize=10, zorder=40,rotation=20 )

    #----Adjusting the labels and color bar
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

#---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/FirstGen_AV_projection.pdf')
    plt.show()

elif Plotnumber == "limit9" :

    # ---------------- Generate the grid for plotting
    yup = 1e4;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    lim.faser_decay.combthr = 1.1
    Lim,LabelLimit = lim.GetLimits(ExperimentsList,0.20,geffem,"V",True)


    # --------- Decay searches
    # Need to cut the FASER limits in two
    xmatlow=Lim['mathusla_decay'][0][Lim['mathusla_decay'][0]<10]
    xmathigh=Lim['mathusla_decay'][0][Lim['mathusla_decay'][0]>2.5]
    ymatlow=Lim['mathusla_decay'][1][Lim['mathusla_decay'][0]<10]
    ymathigh=Lim['mathusla_decay'][1][Lim['mathusla_decay'][0]>2.5]

    s.fill_between(Lim['lsnd_decay'][0],ydown,Lim['lsnd_decay'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
    s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.loglog(Lim['seaquest_phase2_decay'][0],Lim['seaquest_phase2_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:red',zorder=15)
    s.loglog(Lim['faser_decay'][0],Lim['faser_decay'][1],linestyle='-.',linewidth=1.5,color='xkcd:indigo',zorder=15)

    s.loglog(Lim['mathusla_decay'][0],Lim['mathusla_decay'][1],linestyle='-.',linewidth=1.5,color='xkcd:blue',zorder=15)


    # -------- Mono photon at LEP
    s.fill_between(Lim['lep_monogam'][0],200,Lim['lep_monogam'][1],color='xkcd:grey',alpha=0.75, zorder=10)
    s.axhline(200,linestyle='--',linewidth=1.,color='xkcd:grey',zorder=15)
    # -------- Missing energy searches

    s.fill_between(Lim['babar_monogam'][0],ydown,Lim['babar_monogam'][1],color='xkcd:grey',alpha=0.75, zorder=10)
#     s.loglog(Lim['belle2_monogam'][0],Lim['belle2_monogam'][1],linestyle='--',linewidth=1.5,color='xkcd:black',zorder=15)

    # ------ Self consistency
    s.loglog(xbasic,2*xbasic,linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
    s.fill_between(xbasic,ydown,2*xbasic,color='xkcd:grey',alpha=0.75, zorder=10)

    # -------- SN bounds
#     print(Lim['sn1987_low_cooling'][1])
    gp=np.logical_and(Lim['sn1987_high_cooling'][1]>1.1*Lim['sn1987_low_cooling'][1],Lim['sn1987_high_cooling'][1]>1)
    s.fill_between(Lim['sn1987_low_cooling'][0][gp],Lim['sn1987_low_cooling'][1][gp],Lim['sn1987_high_cooling'][1][gp],color='xkcd:grey',alpha=0.75, zorder=10)

    # ----- Invisible decay of pions
#
#     s.loglog(Lim['e949_pi0decay'][0],Lim['e949_pi0decay'][1],linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
#     s.loglog(Lim['na62_invisibledecayPi0'][0],Lim['na62_invisibledecayPi0'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['na62_invisibledecayPi0'][0],ydown,Lim['na62_invisibledecayPi0'][1],color='xkcd:grey',alpha=0.75, zorder=10)

    s.fill_between(Lim['babar_invisibledecayUpsilon'][0],ydown,Lim['babar_invisibledecayUpsilon'][1],color='xkcd:grey',alpha=0.75, zorder=15)


    s.text(0.14,80,r'MATHUSLA',fontsize=10,color='xkcd:blue', zorder=40,rotation=35)
    s.text(0.039,50,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=40)
    s.text(0.04,100,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)
    s.text(1.4,2.e3,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=20)
    s.text(0.02,0.94,r'$g_e:g_d:g_u = -1:-\frac{1}{3}:\frac{2}{3}, \delta_\chi = 0.2$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
#     s.text(0.02,160,r'NA62',fontsize=10, zorder=40,rotation=20 )

    #----Adjusting the labels and color bar
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

#---- Saving and showing on screen the Figure
    plt.tight_layout()
    plt.savefig('Output/FirstGen_V_projection.pdf')
    plt.show()


elif Plotnumber == "heavy1" : ######## Heavy meson limits

    ExperimentsList=np.array(["babar_invisibledecayBmtoKm","belle2_invisibledecayBmtoKm", \
                              "belle_invisibledecayB0toPi0", "belle_invisibledecayB0toK0","babar_invisibledecayBmtoPim",\
                               "e391a_invisibledecayKL0toPi0","e949_invisibledecayKptoPip","na62_invisibledecayKL0toPi0","na62_invisibledecayKptoPip",\
                               "babar_invisibledecayUpsilon"])
### invisible heavy meson decays

    yup = 1e6; ydown = 10
    xup = 10; xdown = 0.0015
    s.set_xlim((xdown, xup))
    s.set_ylim((ydown, yup))

    geff={"gu11":2/3.,"gu22":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gd33":-1/3.,"gl11":-1.,"gl22":-1.,"gd31":1,"gd32":1,"gd21":1}

    gu=2/3;gd=-1/3;ge=-1;
    gbs=1;gsd=1;gbd=1.;
    gdown=((gd,gsd,gbd),(gsd,gd,gbs),(gbd,gbs,gd))
    gup=((gu,0,0),(0,gu,0),(0,0,gu))
    glep=((ge,0,0),(0,ge,0),(0,0,ge))

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10.,geff,"V",True, ReadFromFile=False)


    s.loglog(Lim['babar_invisibledecayBmtoKm'][0], Lim['babar_invisibledecayBmtoKm'][1], linestyle='-', linewidth=1.5, color='xkcd:green', zorder=15)
    s.loglog(Lim['belle2_invisibledecayBmtoKm'][0], Lim['belle2_invisibledecayBmtoKm'][1], linestyle=':', linewidth=1.5, color='xkcd:blue', zorder=15)
    s.loglog(Lim['belle_invisibledecayB0toK0'][0], Lim['belle_invisibledecayB0toK0'][1], linestyle='-', linewidth=1.5, color='xkcd:yellow', zorder=15)
    s.loglog(Lim['belle_invisibledecayB0toPi0'][0], Lim['belle_invisibledecayB0toPi0'][1], linestyle='-', linewidth=1.5, color='xkcd:red', zorder=15)
    s.loglog(Lim['babar_invisibledecayBmtoPim'][0], Lim['babar_invisibledecayBmtoPim'][1], linestyle='-', linewidth=1.5, color='xkcd:purple', zorder=15)
#     s.loglog(Lim['e391a_invisibledecayKL0toPi0'][0], Lim['e391a_invisibledecayKL0toPi0'][1], linestyle='--', linewidth=1.5, color='xkcd:black', zorder=15)
    s.loglog(Lim['e949_invisibledecayKptoPip'][0], Lim['e949_invisibledecayKptoPip'][1], linestyle='-', linewidth=1.5, color='xkcd:grey', zorder=15)
    s.loglog(Lim['na62_invisibledecayKptoPip'][0], Lim['na62_invisibledecayKptoPip'][1], linestyle=':', linewidth=1.5, color='xkcd:grey', zorder=15)
    s.loglog(Lim['na62_invisibledecayKL0toPi0'][0], Lim['na62_invisibledecayKL0toPi0'][1], linestyle='-', linewidth=1.5, color='xkcd:dark', zorder=15)

 ####Upsilon decays

    s.loglog(Lim['babar_invisibledecayUpsilon'][0],Lim['babar_invisibledecayUpsilon'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['babar_invisibledecayUpsilon'][0],ydown,Lim['babar_invisibledecayUpsilon'][1],color='xkcd:grey',alpha=0.75, zorder=15)



    ####B meson decays
    ExperimentsList=np.array(["ship_heavymesondecay"])
    geff["gd21"]=0;
#     gdown=((gd,0.,0),(0.,gd,gbs),(0,gbs,gd))

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10.,geff,"V",PrintToFile=True,ReadFromFile=False,filename='Output/Lim_ship_heavymesondecay_B.dat')
    s.loglog(Lim['ship_heavymesondecay'][0], Lim['ship_heavymesondecay'][1], linestyle='-.', linewidth=1.5, color='xkcd:red', zorder=20)

 ####K meson decays
    geff["gd21"]=1;geff["gd32"]=0;geff["gd31"]=0;geff["gu11"]=0;geff["gd11"]=0;

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10.,geff,"V",True,ReadFromFile=False,filename='Output/Lim_ship_heavymesondecay_K.dat')
    s.loglog(Lim['ship_heavymesondecay'][0], Lim['ship_heavymesondecay'][1], linestyle='-.', linewidth=1.5, color='xkcd:blue', zorder=15)

    s.text(0.002, 8.5*10**3,r'BaBar ($B^- \to K^- \bar{\nu}\nu$),   Belle ($B^0 \to K^0 \bar{\nu}\nu, \pi^0 \bar{\nu}\nu$)', color='xkcd:crimson',fontsize=10, zorder=50)
    s.text(0.002, 1.7*10**4,r'Belle II* ($B^- \to K^- \bar{\nu}\nu$)', color='xkcd:darkblue', fontsize=10, zorder=50)
    s.text(0.002, 4.9*10**3,r'BaBar ($B^- \to \pi^- \bar{\nu}\nu$)', color='xkcd:purple', fontsize=10, zorder=50)
    s.text(0.002, 2.1*10**5,r'NA62 ($K_L^0 \to \pi^0\bar{\nu}\nu$), E949+E787 ($K^+ \to \pi^+a$)', fontsize=10, zorder=50)
    s.text(0.002, 4.7*10**5,r'NA62* ($K^+ \to \pi^+a$)',fontsize=10, zorder=50)
    s.text(1,200,r'BaBar ($\Upsilon \to$ inv)' ,fontsize=10, zorder=50)
#     s.text(0.025,400,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=20)
    s.text(0.002,700,r'SHIP $(K \to \pi\chi \chi)$',fontsize=10,color='xkcd:darkblue', zorder=50,rotation=25)
    s.text(0.9,0.4*10**5,r'SHIP $(B \to K\chi \chi)$',fontsize=10,color='xkcd:rust', zorder=10,rotation=20)

    s.text(0.75,0.94,r'$g_{ij} = 1, \delta_\chi = 10$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)
#

    s.set_xlabel(r'$M_2$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    plt.grid(b=True,alpha=0.3,which='both',linestyle='-')
    for tick in s.xaxis.get_major_ticks() + s.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label2.set_fontsize(18)
#---- Saving and showing on screen the Figure
    plt.savefig('Output/HeavyMeson_V.pdf')
    plt.show()

elif Plotnumber == "heavy2" : #########invisible heavy meson decays BR


    yup = 1e-1; ydown = 1e-5
    xup = 5; xdown = 0.0005
    s.set_xlim((xdown, xup))
    s.set_ylim((ydown, yup))

    ExperimentsList=np.array(["babar_invisibledecayBmtoKm","belle2_invisibledecayBmtoKm", \
                              "belle_invisibledecayB0toPi0", "belle_invisibledecayB0toK0","babar_invisibledecayBmtoPim",\
                               "e391a_invisibledecayKL0toPi0","e949_invisibledecayKptoPip","na62_invisibledecayKptoPip",\
                               "ship_heavymesondecay"])
    gu=2/3;gd=-1/3;ge=-1;
    gbs=1;gsd=1;
    gdown=((gd,0,0),(0,gd,gbs),(0,gbs,gd))
    gup=((gu,0,0),(0,gu,0),(0,0,gu))
    glep=((ge,0,0),(0,ge,0),(0,0,ge))

    s.set_xlabel(r'$M_{\chi}$  [GeV]',fontsize=18)
    s.set_ylabel(r'BR',fontsize=18)

    Del = 0;
    xmin=0.0001; xmax=5
    xi = uf.log_sample(xmin,xmax,100)

    NdarkBmtoKm = am.GamV_MestoXXmes(xi/(1+Del), xi, br.MBminus, br.MKminus, br.fBKeff, Lam=1000.)/br.GamBminus
    NdarkB0toK0 = am.GamV_MestoXXmes(xi/(1+Del), xi, br.MB0, br.MK, br.fBsKeff, Lam=1000.)/br.GamB0
    NdarkB0toPi0 = am.GamV_MestoXXmes(xi/(1+Del), xi, br.MB0, br.MPi, br.fBpieff, Lam=1000.)/br.GamB0
    NdarkBmtoPim = am.GamV_MestoXXmes(xi/(1+Del), xi, br.MBminus, br.Mpim, br.fBpieff, Lam=1000.)/br.GamBminus
    NdarkK0toPi0 = am.GamV_MestoXXmes(xi/(1+Del), xi, br.MK0, br.MPi, br.fKpieff, Lam=1000.)/br.GamKS0
    NdarkKmtoPim = am.GamV_MestoXXmes(xi/(1+Del), xi, br.MKminus, br.Mpim, br.fKpieff, Lam=1000.)/br.GamKplus

    s.loglog(xi,NdarkBmtoKm,linestyle='--',linewidth=1.5,color='xkcd:blue',zorder=15,label=r"$B^- \rightarrow K^- \chi \chi$")
    s.loglog(xi,NdarkBmtoPim,linestyle='--',linewidth=1.5,color='xkcd:black',zorder=15,label=r"$B^- \rightarrow \pi^- \chi \chi$")
    s.loglog(xi,NdarkB0toPi0,linestyle='--',linewidth=1.5,color='xkcd:red',zorder=15,label=r"$B^0 \rightarrow \pi^0 \chi \chi$")
    s.loglog(xi,NdarkB0toK0,linestyle='--',linewidth=1.5,color='xkcd:green',zorder=15,label=r"$B^0 \rightarrow K^0 \chi \chi$")
    s.loglog(xi,NdarkKmtoPim,linestyle='-',linewidth=1.5,color='xkcd:orange',zorder=15,label=r"$K^- \rightarrow \pi^- \chi \chi$")
    s.loglog(xi,NdarkK0toPi0,linestyle='-',linewidth=1.5,color='xkcd:purple',zorder=15,label=r"$K_S \rightarrow \pi^0 \chi \chi$")

    plt.tight_layout()
    l = plt.legend(loc=3, fontsize=14, fancybox=True, framealpha=0.9)
    l.set_zorder(20)
    s.set_xlabel(r'$M_{\chi}$  [GeV]',fontsize=18)
    s.set_ylabel(r'BR [$M^\prime \rightarrow M \chi \chi$ ]',fontsize=18)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout()
    plt.grid(b=True,alpha=0.3,which='both',linestyle='-')
    for tick in s.xaxis.get_major_ticks() + s.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label2.set_fontsize(18)
#---- Saving and showing on screen the Figure
    plt.savefig('Output/BR_HeavyMeson_V.pdf')
    plt.show()


################################################### Practical models  ################################################
elif Plotnumber == "DP" : ######## Dark photon limits


    yup = 0.2;xup=5
    ydown = 1e-5;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)
    s.set_xscale("log", nonposx='clip')
    s.set_yscale("log", nonposy='clip')

    fracAVtoV=-0.016;Mzp=20.
    geffAV={"gu11":fracAVtoV,"gd11":-fracAVtoV,"gd22":-fracAVtoV,"gl11":-fracAVtoV,"gl22":-fracAVtoV}

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,5,geffem,"V",True)
    LimAV,LabelLimitAV = lim.GetLimits(ExperimentsList,5,geffAV,"AV",True)


    # Let us use a bit more the inner functions and make the decay limits for a combined V and AV operator for CHARM
    exp='charm';Delini=0.1;Delf=5;
    xeff,Limeff = de.MakeEffLimits(exp, lim.charm_decay.mx_ini, lim.charm_decay.lim_ini, geffem, "V", Delini )
    xprod, Nprod = br.NProd(Delini,exp,geffem,"V") #
    xifV,EffLimV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,"V",geffem)
    xeff,Limeff = de.MakeEffLimits(exp, lim.charm_decay.mx_ini, lim.charm_decay.lim_ini, geffAV, "AV", Delini )
    xprod, Nprod = br.NProd(Delini,exp,geffAV,"AV") #
    xifAV,EffLimAV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,("AV","V"),(geffAV,geffem))
    xif,EffLim=uf.CombineLimits((xifV,xifAV),(EffLimV,EffLimAV))
    Dexp,Lexp,beamtype=de.GeomForExp(exp) # Need to clean the above functions
    EffLimFin_Low=de.ShortLamLimit(xif, Delf,EffLim,Dexp,Lexp,beamtype,(geffAV,geffem),("AV","V"))
    xi_charm, Lim_charm=uf.CombineUpDown(xif,EffLimFin_Low,EffLim,1.1)


    # Let us use a bit more the inner functions and make the decay limits for a combined V and AV operator for SHIP
#     geffAV=(fracAVtoV,-fracAVtoV,-fracAVtoV);
    exp='ship';Delini=0.1;Delf=5;
    xeff,Limeff = de.GetNaiveDecayLimits( Delini, "ship",10,geffem,"V")
    xprod, Nprod = br.NProd(Delini,exp,geffem,"V") #
    xifV,EffLimV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,"V",geffem)
    xeff,Limeff = de.GetNaiveDecayLimits( Delini, "ship",10,geffAV,"AV")
    xprod, Nprod = br.NProd(Delini,exp,geffAV,"AV") #
    xifAV,EffLimAV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,"AV",geffAV)
    xif,EffLim=uf.CombineLimits((xifV,xifAV),(EffLimV,EffLimAV))
    Dexp,Lexp,beamtype=de.GeomForExp(exp) # Need to clean the above functions
    EffLimFin_Low=de.ShortLamLimit(xif, Delf,EffLim,Dexp,Lexp,beamtype,(geffem,geffAV),("V","AV"))
    xi_ship, Lim_ship=uf.CombineUpDown(xif,EffLimFin_Low,EffLim,1.1)


     # Let us use a bit more the inner functions and make the decay limits for a combined V and AV operator for faser
#     geffAV=(fracAVtoV,-fracAVtoV,-fracAVtoV);
    exp='faser';Delini=0.1;Delf=5;
    xeff,Limeff = de.MakeEffLimits(exp, lim.faser_decay.mx_ini, lim.faser_decay.lim_ini, geffem, "V", Delini )
    xprod, Nprod = br.NProd(Delini,exp,geffem,"V") #
    xifV,EffLimV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,"V",geffem)
    xeff,Limeff = de.MakeEffLimits(exp, lim.faser_decay.mx_ini, lim.faser_decay.lim_ini, geffAV, "AV", Delini )
    xprod, Nprod = br.NProd(Delini,exp,geffAV,"AV") #
    xifAV,EffLimAV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,"AV",geffAV)
    xif,EffLim=uf.CombineLimits((xifV,xifAV),(EffLimV,EffLimAV))
    Dexp,Lexp,beamtype=de.GeomForExp(exp) # Need to clean the above functions
    EffLimFin_Low=de.ShortLamLimit(xif, Delf,EffLim,Dexp,Lexp,beamtype,(geffem,geffAV),("V","AV"))
    xi_faser, Lim_faser=uf.CombineUpDown(xif,EffLimFin_Low,EffLim,1,1.2,100)


    xfaserlow=xi_faser[xi_faser<10.2]
    yfaserlow=Lim_faser[xi_faser<10.2]

    epsup=0.5

    eps_charm=Mzp*Mzp/np.power(Lim_charm,2)/0.3/1.12*(Lim_charm>0.1)
    s.loglog(xi_charm,eps_charm,linestyle='-',linewidth=1.,color='xkcd:darkgreen',zorder=15)
    s.fill_between(xi_charm,eps_charm,epsup,color='xkcd:lime',alpha=1, zorder=-1)

    eps_ship=Mzp*Mzp/np.power(Lim_ship,2)/0.3/1.12*(Lim_ship>0.1)
    s.loglog(xi_ship,eps_ship,linestyle='-.',linewidth=1.5,color='xkcd:orange',zorder=15)
    eps_faserlow=Mzp*Mzp/np.power(yfaserlow,2)/0.3/1.12*(xfaserlow>0.06) +(xfaserlow<0.06)

    s.loglog(xfaserlow,eps_faserlow,linestyle='-.',linewidth=1.5,color='xkcd:indigo',zorder=15)

    s.fill_between(xbasic,0.5,0.024,color='xkcd:blue',alpha=0.25, zorder=15)
    s.axhline(0.024,linestyle='--',linewidth=1.,color='xkcd:blue',zorder=15)

    # -------- SN bounds


    gp=np.logical_and(Lim['sn1987_high_cooling'][1]>1.1*Lim['sn1987_low_cooling'][1],Lim['sn1987_high_cooling'][1]>1)
    eps_sn1987_high=Mzp*Mzp/np.power(Lim['sn1987_high_cooling'][1][gp],2)/0.3/1.12
    eps_sn1987_low=Mzp*Mzp/np.power(Lim['sn1987_low_cooling'][1][gp],2)/0.3/1.12
    s.loglog(Lim['sn1987_high_cooling'][0][gp],eps_sn1987_high,linestyle='-',linewidth=0.5,color='xkcd:purple',zorder=15)
    s.loglog(Lim['sn1987_low_cooling'][0][gp],eps_sn1987_low,linestyle='--',linewidth=0.5,color='xkcd:purple',zorder=15)
    s.fill_between(Lim['sn1987_high_cooling'][0][gp],eps_sn1987_high,eps_sn1987_low,color='xkcd:purple',alpha=0.3, zorder=15)


    s.text(0.01,3e-4,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50)
    s.text(0.01,0.03,r'LEP (EWSB)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    # s.text(0.037,950,r'LSND',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=30)
    s.text(0.43,0.005,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=-65)
    s.text(0.16,0.004,r'FASER',fontsize=10,color='xkcd:indigo', zorder=40,rotation=-60)
    s.text(1.25,0.0005,'SHIP',fontsize=10,color='xkcd:rust', zorder=50,rotation=-65)
    s.text(0.03,0.02 ,r'$\delta_\chi = 5$', style='italic',fontsize=13,transform = s.transAxes, zorder=20)

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{2}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\varepsilon$ ',fontsize=18)

    plt.tight_layout()

#---- Saving and showing on screen the Figure

    plt.savefig('Output/DPOffshellLim.pdf')
    plt.show()


################################################### Practical models  ################################################
elif Plotnumber == "invisible1" : ######## Limits using only invisible


       # ---------------- Generate the grid for plotting
    yup = 1e6;xup=5
    ydown = 10;xdown=0.005
    s.set_xlim((xdown,xup))
    s.set_ylim((ydown,yup))
    xbasic=np.linspace(xdown,xup,75)

    geffOne={"gu11":1,"gd11":1,"gd22":1,"gl11":1.,"gl22":-1.,"gu22":1,"gd33":1,"gd31":1,"gd32":1,"gd21":1,}


#     ExperimentsList=np.array(["babar_invisibledecayBmtoKm","belle2_invisibledecayBmtoKm", \
#                               "belle_invisibledecayB0toPi0", "belle_invisibledecayB0toK0","babar_invisibledecayBmtoPim",\
#                                "e391a_invisibledecayKL0toPi0","e949_invisibledecayKptoPip","na62_invisibledecayKptoPip",\
#                                "ship_heavymesondecay"])

    ExperimentsList=np.array(["babar_monogam","belle2_monogam", \
                          "sn1987_low_cooling","sn1987_high_cooling", \
                          "bes_invisibledecayJPsi","babar_invisibledecayUpsilon",\
                        "babar_invisibledecayBmtoKm","belle2_invisibledecayBmtoKm", \
                        "belle_invisibledecayB0toPi0", "belle_invisibledecayB0toK0","babar_invisibledecayBmtoPim",\
                          "na62_invisibledecayKL0toPi0","e949_invisibledecayKptoPip","na62_invisibledecayKptoPip",\
                          "atlas_monojet_down","atlas_monojet_up","lep_monogam"])
    print(ExperimentsList)

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,0.001,geffOne,"V",False)
#     Lim,LabelLimit = lim.GetLimits(ExperimentsList,10,(2/3,-1/3,-1),"V",True)

#     s.loglog(Lim['miniboone_scattering'][0],Lim['miniboone_scattering'][1],linestyle='-',linewidth=1.5,color='xkcd:green',zorder=15)
#     s.fill_between(Lim['miniboone_scattering'][0],ydown,Lim['miniboone_scattering'][1],color='xkcd:green',alpha=0.5, zorder=15)
#     s.loglog(Lim['sbnd_scattering'][0],Lim['sbnd_scattering'][1],linestyle='--',linewidth=1.5,color='xkcd:orange',zorder=15)
#     s.loglog(Lim['nova_scattering'][0],Lim['nova_scattering'][1],linestyle='--',linewidth=1.5,color='xkcd:rust',zorder=15)

#     s.loglog(Lim['ship_decay'][0],Lim['ship_decay'][1],linestyle='--',linewidth=1.5,color='xkcd:green',zorder=15)
#     s.loglog(Lim['charm_decay'][0],Lim['charm_decay'][1],linestyle='-',linewidth=1.5,color='xkcd:darkgreen',zorder=15)
#     s.fill_between(Lim['charm_decay'][0],ydown,Lim['charm_decay'][1],color='xkcd:darkgreen',alpha=0.5, zorder=15)

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

    # ----- Invisible decay of pions
#
#     s.loglog(Lim['e949_pi0decay'][0],Lim['e949_pi0decay'][1],linestyle='-',linewidth=1.5,color='xkcd:grey',zorder=15)
#     s.loglog(Lim['na62_invisibledecayPi0'][0],Lim['na62_invisibledecayPi0'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
#     s.fill_between(Lim['na62_invisibledecayPi0'][0],ydown,Lim['na62_invisibledecayPi0'][1],color='xkcd:grey',alpha=0.75, zorder=15)

    s.loglog(Lim['babar_invisibledecayUpsilon'][0],Lim['babar_invisibledecayUpsilon'][1],linestyle='-',linewidth=1.5,color='xkcd:dark grey',zorder=15)
    s.fill_between(Lim['babar_invisibledecayUpsilon'][0],ydown,Lim['babar_invisibledecayUpsilon'][1],color='xkcd:grey',alpha=0.75, zorder=15)





    s.loglog(Lim['babar_invisibledecayBmtoKm'][0], Lim['babar_invisibledecayBmtoKm'][1], linestyle='--', linewidth=1.5, color='xkcd:green', zorder=15)
    s.loglog(Lim['belle2_invisibledecayBmtoKm'][0], Lim['belle2_invisibledecayBmtoKm'][1], linestyle=':', linewidth=1.5, color='xkcd:blue', zorder=15)
    s.loglog(Lim['belle_invisibledecayB0toK0'][0], Lim['belle_invisibledecayB0toK0'][1], linestyle='--', linewidth=1.5, color='xkcd:yellow', zorder=15)
    s.loglog(Lim['belle_invisibledecayB0toPi0'][0], Lim['belle_invisibledecayB0toPi0'][1], linestyle='-.', linewidth=1.5, color='xkcd:red', zorder=15)
    s.loglog(Lim['babar_invisibledecayBmtoPim'][0], Lim['babar_invisibledecayBmtoPim'][1], linestyle='-.', linewidth=1.5, color='xkcd:purple', zorder=15)
#     s.loglog(Lim['e391a_invisibledecayKL0toPi0'][0], Lim['e391a_invisibledecayKL0toPi0'][1], linestyle='--', linewidth=1.5, color='xkcd:black', zorder=15)
    s.loglog(Lim['e949_invisibledecayKptoPip'][0], Lim['e949_invisibledecayKptoPip'][1], linestyle='--', linewidth=1.5, color='xkcd:grey', zorder=15)
    s.loglog(Lim['na62_invisibledecayKptoPip'][0], Lim['na62_invisibledecayKptoPip'][1], linestyle=':', linewidth=1.5, color='xkcd:grey', zorder=15)
    s.loglog(Lim['na62_invisibledecayKL0toPi0'][0], Lim['na62_invisibledecayKL0toPi0'][1], linestyle='--', linewidth=1.5, color='xkcd:dark', zorder=15)


    s.text(0.006, 8*10**3,r'BaBar ($B^- \to K^- \bar{\nu}\nu$),   Belle ($B^0 \to K^0 \bar{\nu}\nu, \pi^0 \bar{\nu}\nu$)', color='xkcd:crimson',fontsize=10, zorder=50)
    s.text(0.008, 1.5*10**4,r'Belle II* ($B^- \to K^- \bar{\nu}\nu$)', color='xkcd:darkblue', fontsize=10, zorder=50)
    s.text(0.009, 4.5*10**3,r'BaBar ($B^- \to \pi^- \bar{\nu}\nu$)', color='xkcd:purple', fontsize=10, zorder=50)
#     s.text(0.01, 4.5*10**4,r'E391a ($K_L^0 \to \pi^0\bar{\nu}\nu$)', fontsize=10, zorder=50)
    s.text(0.01, 2.*10**5,r'NA62 ($K_L^0 \to \pi^0\bar{\nu}\nu$), E949+E787 ($K^+ \to \pi^+a$)', fontsize=10, zorder=50)
    s.text(0.01, 4.5*10**5,r'NA62* ($K^+ \to \pi^+a$)',fontsize=10, zorder=50)
#     s.text(0.002,2*10**3,r'SHIP $(K \to \pi\chi \chi)$',fontsize=10,color='xkcd:darkblue', zorder=50,rotation=20)
#     s.text(0.9,5*10**4,r'SHIP $(B \to K\chi \chi)$',fontsize=10,color='xkcd:rust', zorder=10,rotation=20)
#


    s.text(1.1,55,r'BaBar' ,fontsize=10, zorder=50)
    s.text(1.1,120,r'Belle II ($50$ ab${}^{-1}$)' ,fontsize=10, zorder=50)
    s.text(1.1,600,r'LEP (DELPHI)' ,color='xkcd:darkblue',fontsize=10, zorder=50)
    s.text(0.3,300,r'BaBar ($\Upsilon \to$ inv)' ,fontsize=10, zorder=50)

    s.text(0.025,400,r'SN1987 (Cooling)' ,color='xkcd:purple',fontsize=10, zorder=50,rotation=20)
#     s.text(0.015,60,r'LSND',fontsize=10,color='xkcd:darkgreen', zorder=50,rotation=20)
#     s.text(0.01,230,r'NA62',fontsize=10, zorder=40,rotation=20 )
#     s.text(0.15,650,r'SeaQuest',fontsize=10,color='xkcd:red', zorder=40,rotation=35)

#     s.text(0.02,15,r'MiniBooNE',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=20 )
#     s.text(0.01,20,r'SBND',fontsize=10,color='xkcd:orange', zorder=40,rotation=30 )
#     s.text(0.01,30,r'NOvA',fontsize=10,color='xkcd:rust', zorder=40,rotation=20 )
#
#     s.text(0.55,35,r'CHARM',fontsize=10,color='xkcd:darkgreen', zorder=40,rotation=-30)
#     s.text(0.55,130,'SHIP',fontsize=10,color='xkcd:green', zorder=50,rotation=30)
    s.text(0.84,0.94,r'All $ g_i  = 1$', style='italic',fontsize=14,transform = s.transAxes, zorder=20)

    #----Adjusting the labels and color bar
    s.set_xlabel(r'$M_{\chi}$  [GeV]',fontsize=18)
    s.set_ylabel(r'$\Lambda$  / $\sqrt{g}$ [GeV]',fontsize=18)
    plt.tight_layout()

#---- Saving and showing on screen the Figure

    plt.savefig('Output/AllGen_V_invisible.pdf')
    plt.show()





else:
    print("No plot selected")
