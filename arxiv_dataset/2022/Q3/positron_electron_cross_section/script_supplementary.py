#        *************************    
#        * Sublementary material *    
#        *************************    
#    
#        * -------------------------------------------------------------------- *    
#        * New determination of the production cross section                    *    
#        *            for secondary positrons and electrons in the Galaxy       *    
#        * -------------------------------------------------------------------- *    
#    
#          Luca Orusa, Mattia Di Mauro, Fiorenza Donato, and Michael Korsmeier    
#         *-------------------------------------------------------------------*    
#                    Published in: Phys. Rev. D 105, 123021    
#                    arXiv:        2203.13143 [astro-ph.HE]    
#
#          IF YOU USE THIS TABLE, PLEASE CITE THE ABOVE PAPER.    
#          Please acknowledge the use of these tables by citing the paper: Phys. Rev. D 105, 123021(2203.13143)
#          If you want to add these tables to a public code,   
#          please add the readme file available at this link(https://github.com/lucaorusa/positron_electron_cross_section)   
#          to the folder where the tables are located.    
#    
#    
#          We provide the script to read the tables available at this link https://github.com/lucaorusa/positron_electron_cross_section
#          that contain energy differential cross section $d\sigma_{ij}/dT$ of electrons and positrons    
#          for cosmic-ray (CR) component i and intersellar medium (ISM)    
#          component j. The energy differential cross  sections are in units of mb/GeV.



import numpy as np
from math import *
import scipy.special as spys
from scipy.interpolate import interpn
import matplotlib.pyplot as pl

#       * -------------------------------------------------------------------------------------------------------------- *
#       Here you can choose between positron and electron cross sections: write 'positron' for e^+ and 'electron' for e^-
#       * -------------------------------------------------------------------------------------------------------------- *

particle='positron'

#       * -------------------------------------------------------------------------------------------------------------- *
#       Here you can choose between best-fit, upper and lower band of the positron and electron cross sections: 
#       write 'best_fit' for the best-fit, 'upper' for the upper band and 'lower' for the lower band
#       * -------------------------------------------------------------------------------------------------------------- *

request='best-fit'

#       * --------------------------------------------------------------- *
#       Here you can choose the projectile nuclei and the target nuclei:. 
#       
#       For projectile nuclei you can choose between:
#       p(Z= 1, A= 1), D(Z= 1, A= 2), He3 (Z= 2, A= 3), He4 (Z= 2, A= 4), C12 (Z= 6, A=12), C13 (Z= 6, A=13), C14 (Z= 6, A=14), N (Z= 7, A=15), O (Z= 7, A=16)
#       
#       For target nuclei you can choose between:
#       p(Z= 1, A= 1), He4 (Z= 2, A= 4)
#       * --------------------------------------------------------------- *

projectile_nuclei='p'
target_nuclei='p'

#       * ------------------------------------------------------------------------------------------------------ *
#       This function provide the differential cross section dsigma/dT_e, where T_e is the particle (e^+ or e^-) 
#       kinetic energy and T_p is the kinetic energy per nucleon of the incident CR.
#       * ------------------------------------------------------------------------------------------------------ *

def particle_cross_section(T_e,T_p,particle,request,projectile_nuclei,target_nuclei):
    
    if(particle=='positron'):
      if(request=='best-fit'):
         table=np.loadtxt('supplementary_table_positrons_best_fit.dat')
      elif(request=='upper'):
         table=np.loadtxt('supplementary_table_positrons_upper_band.dat')
      elif(request=='lower'):
         table=np.loadtxt('supplementary_table_positrons_lower_band.dat')
    elif(particle=='electron'):
      if(request=='best-fit'):
         table=np.loadtxt('supplementary_table_electrons_best_fit.dat')
      elif(request=='upper'):
         table=np.loadtxt('supplementary_table_electrons_upper_band.dat')
      elif(request=='lower'):
         table=np.loadtxt('supplementary_table_electrons_lower_band.dat')


    if(projectile_nuclei=='p' and target_nuclei=='p'):
      index=2
    elif(projectile_nuclei=='p' and target_nuclei=='He4'):
      index=3
    elif(projectile_nuclei=='D' and target_nuclei=='p'):
      index=4
    elif(projectile_nuclei=='D' and target_nuclei=='He4'):
      index=5
    elif(projectile_nuclei=='He3' and target_nuclei=='p'):
      index=6
    elif(projectile_nuclei=='He3' and target_nuclei=='He4'):
      index=7
    elif(projectile_nuclei=='He4' and target_nuclei=='p'):
      index=8
    elif(projectile_nuclei=='He4' and target_nuclei=='He4'):
      index=9
    elif(projectile_nuclei=='C12' and target_nuclei=='p'):
      index=10
    elif(projectile_nuclei=='C12' and target_nuclei=='He4'):
      index=11
    elif(projectile_nuclei=='C13' and target_nuclei=='p'):
      index=12
    elif(projectile_nuclei=='C13' and target_nuclei=='He4'):
      index=13
    elif(projectile_nuclei=='C14' and target_nuclei=='p'):
      index=14
    elif(projectile_nuclei=='C14' and target_nuclei=='He4'):
      index=15
    elif(projectile_nuclei=='N' and target_nuclei=='p'):
      index=16
    elif(projectile_nuclei=='N' and target_nuclei=='He4'):
      index=17
    elif(projectile_nuclei=='O' and target_nuclei=='p'):
      index=18
    elif(projectile_nuclei=='O' and target_nuclei=='He4'):
      index=19

    projectile_kinetic_energy=np.logspace(np.log10(0.1), np.log10(100000),120)
    projectile_kinetic_energy=np.append(projectile_kinetic_energy,np.logspace(np.log10(110158.1938), np.log10(1e6),20))
    particle_kinetic_energy=np.logspace(np.log10(0.01), np.log10(10000),90)
    
    cross_section=np.zeros((len(particle_kinetic_energy),len(projectile_kinetic_energy)))
    count=0
    for j in range(len(projectile_kinetic_energy)):
      cross_section[:,j]=table[count:count+90,index]
      count=count+90


    if((type(T_e)==float or type(T_e)==int) and type(T_p)==tuple):
        cross=np.zeros(len(T_p))
        for i in range(len(T_p)):
          point=np.array([T_e,T_p[i]])
          points=(particle_kinetic_energy,projectile_kinetic_energy)
          result=interpn(points,cross_section,point)
          cross[i]=result
        return cross
    elif((type(T_e)==tuple) and (type(T_p)==float or type(T_p)==int)):

        cross=np.zeros(len(T_e))
        for i in range(len(T_e)):
          point=np.array([T_e[i],T_p])
          points=(particle_kinetic_energy,projectile_kinetic_energy)
          result=interpn(points,cross_section,point)
          cross[i]=result
        return cross
    elif(type(T_e)==tuple and type(T_p)==tuple):
        cross=np.zeros((len(T_e),len(T_p)))
        for j in range(len(T_p)):
          for i in range(len(T_e)):
            point=np.array([T_e[i],T_p[j]])
            points=(particle_kinetic_energy,projectile_kinetic_energy)
            result=interpn(points,cross_section,point)
            cross[i,j]=result
        return cross
    else:
        point=np.array([T_e,T_p])
        points=(particle_kinetic_energy,projectile_kinetic_energy)
        result=interpn(points,cross_section,point)
        return result



T_sample=np.logspace(np.log10(0.01), np.log10(10000),90)
T_sample=tuple(T_sample)
T_proton=100

fig = pl.figure(figsize=(8,6))
pl.plot(T_sample, particle_cross_section(T_sample,T_proton,particle,request,projectile_nuclei,target_nuclei), color="blue", ls='-', lw=1.5,label=r'{0}'.format(round(T_proton,3)))

if(particle=='positron'):
   pl.ylabel(r'$ \frac{d \sigma}{d T_{e^+}}$ [mb/GeV]', fontsize=18)
   pl.xlabel(r'$T_{e^+}$ [GeV]', fontsize=18)
else:
   pl.ylabel(r'$ \frac{d \sigma}{d T_{e^-}}$ [mb/GeV]', fontsize=18)
   pl.xlabel(r'$T_{e^-}$ [GeV]', fontsize=18)
pl.axis([1e-2,min(1e4,T_proton),1e-3,500])
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=1,prop={'size':18},numpoints=1, scatterpoints=1, ncol=2)
fig.tight_layout(pad=0.5)
pl.savefig('Cross_section_{0}_{1}_{2}_{3}_{4}.pdf'.format(round(T_proton,3),particle,request,projectile_nuclei,target_nuclei))


