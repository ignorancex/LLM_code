#================================RunDL_1D.py===================================#
# Created by Ciaran O'Hare 2020

# Script for running the 1D discovery limits (at fixed mass)
# The atmospheric neutrinos need to be generated first by running both
# python AtmNu_Recoils.py Xe131
# python AtmNu_Recoils.py Ar40
#
# Then the results for Fig. 7 can be run by doing both
# python runDL_1D.py Xe131
# python runDL_1D.py Ar40
#
# It takes about an half an hour on my computer
#==============================================================================#


import sys
sys.path.append('../src')
from numpy import *
from Params import *
from Like import *

#==============================================================================#
# Input
Nuc = eval(sys.argv[1])
print('Nucleus = ',Nuc.Name)
if Nuc.Name=='Xe':
    E_min = 2.0
    E_max = 200.0
    m_chi = 100.0
elif Nuc.Name=='Ar':
    E_min = 10.0
    E_max = 400.0
    m_chi = 100.0
m_vals = logspace(log10(m_chi),log10(m_chi),1)
#==============================================================================#


#==============================================================================#
# Optimisation
ex_min=1.0e1
ex_max=1.0e5
n_ex = 50
ne = 50
np = 20
nt = 10
verbose=False
#==============================================================================#

#==============================================================================#
R_sig_nondir = Rsig_NonDirectional(Nuc,m_vals,E_min,E_max,ne)
R_nu_nondir = Rnu_NonDirectional(Nuc,E_min,E_max,ne)

print('Loading Atm recoils...')
AtmRecoils = LoadAtmRecoils(Nuc)
print('Calculating background...')
R_nu0 = Rnu_Ecosth(Nuc,E_min,E_max,ne,AtmRecoils,CygnusTracking=True)
R_nu1 = Rnu_Ecosth(Nuc,E_min,E_max,ne,AtmRecoils,CygnusTracking=False)
R_nu0_HT = Rnu_Ecosth(Nuc,E_min,E_max,ne,AtmRecoils,CygnusTracking=False,HT=True)

print('Calculating Signal...')
R_w0 = Rsig_Ecosth(67+Jan1,Nuc,m_vals,E_min,E_max,ne,np=np,CygnusTracking=True)
R_w1 = Rsig_Ecosth_TimeAveraged(10,Nuc,m_vals,E_min,E_max,ne,np=np)
R_w0_HT = Rsig_Ecosth_TimeAveraged(10,Nuc,m_vals,E_min,E_max,ne,np=np,HT=True)
#==============================================================================#



#==============================================================================#
f_AtmNu = 0.25
runDL('1D_'+Nuc.Name+'_CR_Stationary_HT',R_w0_HT,R_nu0_HT,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_CygnusTracking_100pc',R_w0,R_nu0,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_Stationary_100pc',R_w1,R_nu1,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_Stationary_75pc',R_w1,R_nu1,m_vals,ex_min,ex_max,n_ex,A_CR=0.75,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_Stationary_50pc',R_w1,R_nu1,m_vals,ex_min,ex_max,n_ex,A_CR=0.5,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_Nondirectional',R_sig_nondir,R_nu_nondir,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)

f_AtmNu = 0.1
runDL('1D_'+Nuc.Name+'_CR_Stationary_HT_lowdPhiAtm',R_w0_HT,R_nu0_HT,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_CygnusTracking_100pc_lowdPhiAtm',R_w0,R_nu0,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_Stationary_100pc_lowdPhiAtm',R_w1,R_nu1,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_Stationary_75pc_lowdPhiAtm',R_w1,R_nu1,m_vals,ex_min,ex_max,n_ex,A_CR=0.75,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_CR_Stationary_50pc_lowdPhiAtm',R_w1,R_nu1,m_vals,ex_min,ex_max,n_ex,A_CR=0.5,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('1D_'+Nuc.Name+'_Nondirectional_lowdPhiAtm',R_sig_nondir,R_nu_nondir,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
