#================================RunDL_2D.py===================================#
# Created by Ciaran O'Hare 2020

# Script for running the 1D discovery limits (at fixed mass)
# The atmospheric neutrinos need to be generated first by running both
# python AtmNu_Recoils.py Xe131
# python AtmNu_Recoils.py Ar40
#
# Then the results for Fig. 8 can be run by doing
# python runDL_1D.py
#
# It takes about an hour on my computer
#==============================================================================#


import sys
sys.path.append('../src')
from numpy import *
from Params import *
from Like import *

#==============================================================================#
# Optimisation
m_min = 3.0
m_max = 1.0e4
nm = 200
ex_min=1.0e3
ex_max=1.0e4
n_ex = 2
ne = 50
np = 20
nt = 10
verbose=False
m_vals = logspace(log10(m_min),log10(m_max),nm)
#==============================================================================#

#==============================================================================#
R_sig_nondir_Xe = Rsig_NonDirectional(Xe131,m_vals,2.0,200.0,ne)
R_nu_nondir_Xe = Rnu_NonDirectional(Xe131,2.0,200.0,ne)
R_sig_nondir_Ar = Rsig_NonDirectional(Ar40,m_vals,10.0,400.0,ne)
R_nu_nondir_Ar = Rnu_NonDirectional(Ar40,10.0,400.0,ne)
R_sig_nondir = column_stack((R_sig_nondir_Xe,R_sig_nondir_Ar))
R_nu_nondir = column_stack((R_nu_nondir_Xe,R_nu_nondir_Ar))

print('Calculating background...')
R_nu_Xe = Rnu_Ecosth(Xe131,2.0,200.0,ne,LoadAtmRecoils(Xe131),CygnusTracking=True)
R_nu_Ar = Rnu_Ecosth(Ar40,10.0,400.0,ne,LoadAtmRecoils(Ar40),CygnusTracking=True)
R_nu = column_stack((R_nu_Xe,R_nu_Ar))
print('Calculating Signal...')
R_sig_Xe = Rsig_Ecosth(67+Jan1,Xe131,m_vals,2.0,200.0,ne,np=np,CygnusTracking=True)
R_sig_Ar = Rsig_Ecosth(67+Jan1,Ar40,m_vals,10.0,400.0,ne,np=np,CygnusTracking=True)
R_sig = column_stack((R_sig_Xe,R_sig_Ar))
#==============================================================================#



#==============================================================================#
f_AtmNu = 0.25
runDL('2D_XeAr_CR',R_sig,R_nu,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('2D_Xe_Nondirectional',R_sig_nondir_Xe,R_nu_nondir_Xe,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)

f_AtmNu = 0.1
runDL('2D_XeAr_CR_lowdPhiAtm',R_sig,R_nu,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
runDL('2D_Xe_Nondirectional_lowdPhiAtm',R_sig_nondir_Xe,R_nu_nondir_Xe,m_vals,ex_min,ex_max,n_ex,f_AtmNu=f_AtmNu,verbose=verbose)
