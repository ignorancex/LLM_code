from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import multiprocessing as mp

sys.path.append("../")
import NMSSM_potential_CT as CTmod

# number of cores for multiprocessing
Ncores = 72

# SM inputs
mh1252=125.**2 # squared SM Higgs mass [GeV^2]
v2=174.**2 # squared SM Higgs vev [GeV^2]
mZ2=91.2**2 # squared Z-boson mass [GeV^2]

###############################################################
# define functions
###############################################################
def lam_align(tb):
   sb2 = tb**2./(1.+tb*tb)
   c2b = (1.-tb*tb)/(1+tb*tb)
   return np.sqrt( (mh1252-mZ2*c2b)/(2.*v2*sb2) )

def Alam_align(lam, kap, tb, mu):
   s2b = 2*tb/(1.+tb*tb)
   return 2.*mu*(1./s2b-kap/lam)

def dl2_mh125(lam, tb):
   sb4 = tb**4./(1.+tb*tb)**2.
   s2b2 = 4.*tb*tb/(1.+tb*tb)**2.
   c2b2 = (1.-tb*tb)**2./(1.+tb*tb)**2.
   return ( mh1252 - mZ2*c2b2 - lam*lam*v2*s2b2 )/( 2.*v2*sb4 )

# set up files for output
header_general = ('#  index  lambda  kappa  tanbeta   mu/GeV  Alam/GeV  Akap/GeV  DeltaLambda2  M1  M2  '
   +'mh125/GeV  mH/GeV  mhS/GeV  '
   +'C_h125^SM  C_h125^NSM  C_h125^S  '
   +'C_H^SM  C_H^NSM  C_H^S  '
   +'C_hS^SM  C_hS^NSM  C_hS^S  ')

header_phase = '# HSM/GeV  HNSM/GeV  HS/GeV  T/GeV'
header_Tc = '# HSM_low/GeV  HNSM_low/GeV  HS_low/GeV  key_low  HSM_high/GeV  HNSM_high/GeV  HS_high/GeV  key_high  T_c/GeV  trantype  '
header_Tn = '# HSM_low/GeV  HNSM_low/GeV  HS_low/GeV  key_low  HSM_high/GeV  HNSM_high/GeV  HS_high/GeV  key_high  T_n/GeV  action/GeV  trantype  '

def write_results(result, cou):
   if len(result) == 3:
      fo = open(fout_path+'/points_failed_general.txt','a')
   else:
      fo = open(fout_path+'/points_good_general.txt','a')
   fo.write(str(int(cou))+'  ')
   # write NMSSM params
   for val in result[0]:
      fo.write('{:3E}  '.format(val))
   # write Higgs masses
   for val in result[1][0]:
      fo.write('{:3E}  '.format(val))
   # write Higgs compositions
   for comp in result[1][1]:
      for val in comp:
         fo.write('{:3E}  '.format(val))
   if len(result) == 3:
      fo.write(str(int(result[2]))+'\n')
      fo.close()
   else:
      fo.write('\n')
      fo.close()
      # write phases
      cou_phase = 0
      for key in result[2]:
         fo = open(fout_path+'/good_'+str(int(cou))+'_phase_'+str(int(key))+'.txt','w')
         fo.write(header_phase)
         fo.write('\n')
         try:
            for i in range(len(result[2][key].T)):
               for val in result[2][key].X[i]:
                  fo.write('{:3E}  '.format(val))
               fo.write('{:3E}\n'.format(result[2][key].T[i]))
         except:
            fo.write('#  cannot read phases...\n')
         fo.close()
         cou_phase += 1
      # write Tc output
      fo = open(fout_path+'/good_'+str(int(cou))+'_Tc_out.txt','w')
      fo.write(header_Tc)
      fo.write('\n')
      try:
         for transition in result[3]:
            for val in transition['low_vev']:
               fo.write('{:3E}  '.format(val))
            fo.write(str(int(transition['low_phase']))+'  ')
            for val in transition['high_vev']:
               fo.write('{:3E}  '.format(val))
            fo.write(str(int(transition['high_phase']))+'  ')
            fo.write('{:3E}  '.format(transition['Tcrit']))
            fo.write(str(int(transition['trantype']))+'\n')
      except:
         fo.write('# cannot read Tc output...\n')
      fo.close()
      # write Tn output
      fo = open(fout_path+'/good_'+str(int(cou))+'_Tn_out.txt','w')
      fo.write(header_Tn)
      fo.write('\n')
      try:
         for transition in result[4]:
            for val in transition['low_vev']:
               fo.write('{:3E}  '.format(val))
            fo.write(str(int(transition['low_phase']))+'  ')
            for val in transition['high_vev']:
               fo.write('{:3E}  '.format(val))
            fo.write(str(int(transition['high_phase']))+'  ')
            fo.write('{:3E}  '.format(transition['Tnuc']))
            fo.write('{:3E}  '.format(transition['action']))
            fo.write(str(int(transition['trantype']))+'\n')
      except:
         fo.write('# cannot read Tn output...\n')
      fo.close()

def CT_calculation(ind, kap, tb, mu, Akap, M1=1e3, M2=1e3):
   """ 
   runs the parameter points in CosmoTransitions
   Checks that
   - an SM-like 125 GeV mass eigenstate is present,
   - the physical minimum is the global minimum, 
   - the high-T minimum restores EW symmetry
   and calculates the thermal history of the points

   returns:
      if one of the checks fails:
         None
      if all checks are passes:
         tuple: containing
            ()[0]: array of NMSSM parameters:
               (  0: lambda,
                  1: kappa,
                  2: tanbeta,
                  3: mu,
                  4: A_lambda,
                  5: A_kappa,
                  6: Delta_lambda_2
                  7: M1
                  8. M2 )
            ()[1]: a tuple containing info on CP-even neutral masses:
                  ()[0]: array with masses of (SM-like, NSM-like, S-like) mass eigenstates
                  ()[1]: array of arrays of mixing angles of these mass eigenstates in the Higgs basis (H^SM, H^NSM, H^S)
            ()[3]: .phases output from CosmoTransitions
            ()[4]: .TcTrans output from CosmoTransitions
            ()[5]: .TnTrans output from CosmoTransitions
   """
   # compute parameters fixed by alignment
   lam = lam_align(tb)
   Alam = Alam_align(lam, kap, tb, mu)
   # compute stop corrections parameter to give right Higgs mass
   dl2 = dl2_mh125(lam, tb)
   # create CosmoTransition object
   NMSSMparams = np.array([lam, kap, tb, mu, Alam, Akap, dl2, M1, M2])
   mymodel = CTmod.model1(NMSSMparams)
   """
   check the CP-even neutral mass matrix  at the physical minimum
   after inluding all corrections to ensure that
   - no mass eigenstate is tachyonic
   - there is a dominantly H^SM mass eigenstate with mass ~125 GeV
   - approximate alignment is satisfied
   """
   if mymodel.checkh125() == False:
      result = [np.array([mymodel.lam, mymodel.kap, mymodel.tb, mymodel.mu, mymodel.Alam, mymodel.Akap, mymodel.dl2, mymodel.M1, mymodel.M2]), mymodel.HiggsMassEigenstates(), 0], ind
      write_results(result[0], result[1])
      return
   """
   check that the physical minimum is the global minimum at zero temperature
   """
   if mymodel.checkZeroTVacuumStructure() == False:
      result = [np.array([mymodel.lam, mymodel.kap, mymodel.tb, mymodel.mu, mymodel.Alam, mymodel.Akap, mymodel.dl2, mymodel.M1, mymodel.M2]), mymodel.HiggsMassEigenstates(), 1], ind
      write_results(result[0], result[1])
      return
   """
   check that the high-T vacuum restores EW symmetry
   """
   if mymodel.checkTmaxVacuumStructure() is False:
      result = [np.array([mymodel.lam, mymodel.kap, mymodel.tb, mymodel.mu, mymodel.Alam, mymodel.Akap, mymodel.dl2, mymodel.M1, mymodel.M2]), mymodel.HiggsMassEigenstates(), 2], ind
      write_results(result[0], result[1])
      return
   """
   try to do the thermal calculation
   """
   Tc_out=[]
   Tn_out=[]
   try:
      mymodel.getPhases()
      mymodel.calcTcTrans()
      Tc_out = mymodel.TcTrans
      mymodel.findAllTransitions()
      Tn_out = mymodel.TnTrans
      if len(Tn_out) == 0:
         raise Exception('no transitions found in Tn calc... I will try harder now!')
   except:
      try:
         mymodel.x_eps*=0.1
         mymodel.getPhases()
         mymodel.calcTcTrans()
         Tc_out = mymodel.TcTrans
         mymodel.findAllTransitions()
         Tn_out = mymodel.TnTrans
      except:
         pass
   result = [np.array([mymodel.lam, mymodel.kap, mymodel.tb, mymodel.mu, mymodel.Alam, mymodel.Akap, mymodel.dl2, mymodel.M1, mymodel.M2]), mymodel.HiggsMassEigenstates(), mymodel.phases, Tc_out, Tn_out], ind
   write_results(result[0], result[1])
   return

def Akap_vspvs(vspvs, kap, tb, mu):
   lam = lam_align(tb)
   return -2.*kap*mu/lam*(1+vspvs)

def kap_kapOlam(kapOlam, tb):
   lam = lam_align(tb)
   return kapOlam*lam


###############################################################
# MakeParameters
###############################################################
fout_path = '../data/Scan_'+time.strftime("%Y%m%d_%H%M")
os.mkdir(fout_path)

Npoints = 100 # number of points to scan over

# choose input parameters
tb = 3. # tan(beta)
kapOlam = -0.1 # kappa/lambda
mu_max = 500. # max(mu/GeV) for scan 
mu_min = 100. # min(mu/GeV) for scane
mu_array = (2*np.random.randint(0,2,size=Npoints)-1)*(np.random.rand(Npoints)*(mu_max - mu_min)+mu_min)
vspvs_min = -3.0 # min(vsp'/vs) for scan
vspvs_max = 7.0 # max(vsp'/vs) for scan
vspvs_array = np.random.rand(Npoints)*(vspvs_max - vspvs_min)+vspvs_min

# compute dependent parameters
kap = kap_kapOlam(kapOlam, tb)
Akap_array = Akap_vspvs(vspvs_array, kap, tb, mu_array)

# prepare output file
fo = open(fout_path+'/params.txt','w')
fo.write('#  index  kappa  tanbeta   mu/GeV  Akap/GeV\n')
cou_1 = 0
for i in range(Npoints):
   cou_1+=1
   fo.write(str(int(cou_1))+'  ')
   fo.write('{:3E}  '.format(kap))
   fo.write('{:3E}  '.format(tb))
   fo.write('{:3E}  '.format(mu_array[i]))
   fo.write('{:3E}  '.format(Akap_array[i]))
   fo.write('\n')
fo.close()

###############################################################
# RunScan
###############################################################
params = np.loadtxt(fout_path+'/params.txt')

# prepare general file for failed points
fo = open(fout_path+'/points_failed_general.txt','w')
fo.write(header_general)
fo.write('failFlag\n')
fo.close()

# prepare general file for good points
fo = open(fout_path+'/points_good_general.txt','w')
fo.write(header_general)
fo.write('\n')
fo.close()

pool = mp.Pool(Ncores)

for irun in range(Npoints):
   pool.apply_async(CT_calculation, args=(params[irun,0], params[irun,1], params[irun,2], params[irun,3], params[irun,4]))

pool.close()
pool.join()
