from __future__ import division
import numpy as np

from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append("../")
import NMSSM_potential_CT as CTmod

###############################################################
inname = '../data/Scan_EXAMPLE'

###############################################################
# parameters for classification
###############################################################
SameXThreshold = 5. # min (Euclidean) distance [GeV] for two minima to be considered as distinct
StrongThreshold = 1. # min v_c/T_c (for Tcrit results) or v_n/T_n (for Tnucl results)
SFOEWPT_lTThreshold = 1. # criterion for sphaleron supression in low-T phase
SFOEWPT_hTThreshold = 0.5 # criterion for sphaleron supression in high-T phase
nuclCond = 140. # S_3/T threshold for nucleation
nuclCondThreshold = 0.1*nuclCond # tolerance for successful calculation of nucleation condition
Tmax = 1e3 # max temperature [GeV], should match cosmoTransition setting
dtFracMax = .25 # allowed tolerance for relative temperature difference when matching phases

CNSMThreshold = 0.1 # max allowed H^NSM mixing angle of SM-like Higgs
CSThreshold = 0.2 # max allowed H^S mixing angle of SM-like Higgs

muLEP=100 # chargino bound from LEP on mu/GeV

v2 = 172.**2 # squared vev of H^SM [Gev^2]
###############################################################
# load parameter lists
###############################################################
failed_params = np.loadtxt(inname+'/points_failed_general.txt')
good_params = np.loadtxt(inname+'/points_good_general.txt')
# remove points violating the mixing limits:
inds_mixing_fail = list(set(np.where(np.abs(good_params[:,14])*good_params[:,3] > CNSMThreshold)[0]) | set(np.where(np.abs(good_params[:,15]) > CSThreshold)[0]))
for i in inds_mixing_fail:
   failed_params = np.concatenate((failed_params, [np.append(np.append(good_params[i,:-1], 0), good_params[i,-1])] ))

good_params = np.delete(good_params, inds_mixing_fail, axis=0)

header = ['index', 'lambda', 'kappa', 'tanbeta', 'mu/GeV', 'Alam/GeV', 'Akap/GeV', 'DeltaLambda2', 'M1', 'M2', 'mh125/GeV', 'mH/GeV', 'mhS/GeV', 'C_h125^SM', 'C_h125^NSM', 'C_h125^S', 'C_H^SM', 'C_H^NSM', 'C_H^S', 'C_hS^SM', 'C_hS^NSM', 'C_hS^S']

header_failed = header+['failFlag']

###############################################################
# get vs'
###############################################################
def add_vsprimeCW(param_list):
   param_list_out = np.zeros((param_list.shape[0], param_list.shape[1]+1))
   param_list_out[:,:-1] = param_list
   for i in range(param_list.shape[0]):
      NMSSMparams = param_list[i,1:10]
      mod = CTmod.model1(NMSSMparams)
      param_list_out[i,-1] = mod.get_vsprimeCW()
   return param_list_out

failed_params = add_vsprimeCW(failed_params)
good_params = add_vsprimeCW(good_params)

header_failed += ['vspCW/GeV']
header_good = header+['vspCW/GeV']

###############################################################
# functions for categorization
###############################################################
def trans_start_from_trivial(transition, ind):
   """
   returns True of the transition starts from the trivial phase,
   False else
   """
   hT_phase = np.loadtxt(inname+'/good_'+str(int(good_params[ind,0]))+'_phase_'+str(int(transition[7]))+'.txt')
   if (np.linalg.norm(hT_phase[-1,0:3]-np.array([0.,0.,0.])) < SameXThreshold
      and hT_phase[-1,3] == Tmax):
      return True
   else:
      return False

def trans_end_physical(transition, ind):
   """
   returns True if the transition ends in the physical phase,
   False else
   """
   lT_phase = np.loadtxt(inname+'/good_'+str(int(good_params[ind,0]))+'_phase_'+str(int(transition[3]))+'.txt')
   Xphys = np.array([np.sqrt(2.*v2), 0., np.sqrt(2.)*good_params[ind,4]/good_params[ind,1]])
   if (np.linalg.norm(lT_phase[0,0:3]-Xphys) < SameXThreshold
      and lT_phase[0,3] == 0.):
      return True
   else:
      return False

def trans_end_HSonly(transition, ind):
   """
   returns True if the transition ends in a phase where H^SM = H^NSM = 0
   False else
   """
   lT_phase = np.loadtxt(inname+'/good_'+str(int(good_params[ind,0]))+'_phase_'+str(int(transition[3]))+'.txt')
   if (np.linalg.norm(lT_phase[0,0:2]) < SameXThreshold
      and np.abs(lT_phase[0,2]) > SameXThreshold):
      return True
   else:
      return False

def trans_end_doublet_only(transition, ind):
   """
   returns True if the transition ends in a phase where H^SM or H^NSM is different from 0
   False else
   """
   lT_phase = np.loadtxt(inname+'/good_'+str(int(good_params[ind,0]))+'_phase_'+str(int(transition[3]))+'.txt')
   if (np.linalg.norm(lT_phase[0,0:2]) > SameXThreshold
      and np.abs(lT_phase[0,2]) < SameXThreshold):
      return True
   else:
      return False

def trans_strongly_first_order(transition):
   """
   returns True if transition is strongly first order,
   False else
   """
   if transition[-1] == 1:
      if np.linalg.norm(transition[0:3]-transition[4:7])/transition[8] > StrongThreshold:
         return True
   else:
      return False

def trans_strongly_EW_first_order(transition):
   """
   returns True if the EW part of the transition is strongly first order,
   False else
   """
   if transition[-1] == 1:
      if np.linalg.norm(transition[0:2])/transition[8] > SFOEWPT_lTThreshold and np.linalg.norm(transition[4:6])/transition[8] < SFOEWPT_hTThreshold:
         return True
   else:
      return False

def transition_real(transition, ind):
   """
   checks if the transition is not just some numerical glitch
   Returns False if transition appear like a glitch.
   True else
   """
   if transition[-1] == 1:
      if np.linalg.norm(transition[0:3]-transition[4:7]) < SameXThreshold:
         return False
      else:
         return True
   if transition[-1] == 2:
      # this is a second order phase transition. Need to get low-T location from phase info
      T = transition[8]
      lT_phase = np.loadtxt(inname+'/good_'+str(int(good_params[ind,0]))+'_phase_'+str(int(transition[3]))+'.txt')
      Tind = np.argmin(np.abs(lT_phase[:,3]-T))
      if np.linalg.norm(lT_phase[Tind,0:3]-transition[4:7]) < SameXThreshold:
         return False
      else:
         return True

def check_nucl_calc_success(transition):
   if transition[-1] == 1 and np.abs(transition[9]/transition[8] - nuclCond) < nuclCondThreshold:
      return True
   elif transition[-1] == 2 and transition[9] == 0:
      return True
   else:
      return False

def check_ltphase_T(transition, ind):
   """ 
   returns True if the low-temperature phase extends up to the transition temperature, False else
   """
   lT_phase = np.loadtxt(inname+'/good_'+str(int(good_params[ind,0]))+'_phase_'+str(int(transition[3]))+'.txt')
   if np.min(np.abs(lT_phase[:,3]-transition[8])) < dtFracMax*transition[8]:
      return True
   else:
      return False

###############################################################
# do the categorization
# the indices of the categories of the points in good_params
# are stored in cat_inds_Tcrit (categorization based on critical
# temperature calculation) and cat_inds_Tnucl (categorization
# based on nucleation temperature calculation)
###############################################################
# categories
cat_labels = [r'no transitions',
   r'1-a', # 1-step SFOEW
   r'1-b', # 1-step FO
   r'1-c', # 1-step 2nd
   r'2(I)-a', # from (0,0,0) > (0,0,v_S'') > SFOEW (physical)
   r'2(I)-b', # from (0,0,0) > (0,0,v_S'') > FO (physical)
   r'2(I)-c', # from (0,0,0) > (0,0,v_S'') > 2nd (physical)
   r'2(II)-aa', # from (0,0,0) > SFOEW (v1,v2,v_S'') > SFOEW (physical)
   r'2(II)-ab', # from (0,0,0) > SFOEW (v1,v2,v_S'') > FO (physical)
   r'2(II)-ac', # from (0,0,0) > SFOEW (v1,v2,v_S'') > 2nd (physical)
   r'2(II)-ba', # from (0,0,0) > FO (v1,v2,v_S'') > SFOEW (physical)
   r'2(II)-bb', # from (0,0,0) > FO (v1,v2,v_S'') > FO (physical)
   r'2(II)-bc', # from (0,0,0) > FO (v1,v2,v_S'') > 2nd (physical)
   r'2(II)-ca', # from (0,0,0) > 2nd (v1,v2,v_S'') > SFOEW (physical)
   r'2(II)-cb', # from (0,0,0) > 2nd (v1,v2,v_S'') > FO (physical)
   r'2(II)-cc', # from (0,0,0) > 2nd (v1,v2,v_S'') > 2nd (physical)
   r'nucleation calculation failed',
   r'something else...',
   r'read error...']

cat_inds_Tcrit = []
cat_inds_Tnucl = []
for entry in cat_labels:
   cat_inds_Tcrit.append([])
   cat_inds_Tnucl.append([])

for i in range(good_params.shape[0]):
   # check the critical-temperature calculation results
   try:
      out_Tcrit = np.loadtxt(inname+'/good_'+str(int(good_params[i,0]))+'_Tc_out.txt')
      if out_Tcrit.shape[0] == 0: # empty transition file
         cat_inds_Tcrit[0].append(i)
      elif len(out_Tcrit.shape) == 1: # only one entry in transition file
         transition = out_Tcrit
         # check that the only transition starts from the trivial phase and ends in the physical phase
         if trans_start_from_trivial(transition, i) and trans_end_physical(transition, i) and check_ltphase_T(transition, i):
            if trans_strongly_EW_first_order(transition): # check if the transition is an SFOEWPT
               cat_inds_Tcrit[1].append(i)
            elif transition[-1] == 1: # or FO
               cat_inds_Tcrit[2].append(i)
            elif transition[-1] == 2: # or 2nd order
               cat_inds_Tcrit[3].append(i)
            else:
               cat_inds_Tcrit[-2].append(i)
         else:
            cat_inds_Tcrit[-2].append(i)
      else: # multiple transitions
         # order the transition by temperature
         j_Tc = np.argsort(out_Tcrit[:,8])[::-1]
         # find the first transition from the high-T minimum
         j_try = 0
         transition = out_Tcrit[j_Tc[j_try],:]
         while trans_start_from_trivial(transition, i) == False and j_try < len(j_Tc)-1 and check_ltphase_T(transition, i) == False:
            j_try += 1
            transition = out_Tcrit[j_Tc[j_try],:]
         # throw point out if there is no transition startingfrom the high-T minimum
         if j_try == len(j_Tc):
            cat_inds_Tcrit[-2].append(i)
         # collect further transitions
         transition_list = [transition]
         active_key = transition[3]
         while j_try < len(j_Tc)-1:
            j_try += 1
            if out_Tcrit[j_Tc[j_try],7] == active_key and check_ltphase_T(out_Tcrit[j_Tc[j_try],:], i):
               transition_list.append(out_Tcrit[j_Tc[j_try],:])
               active_key = out_Tcrit[j_Tc[j_try],3]
         # if there is only 1 step in the chain:
         if len(transition_list) == 1:
            transition = transition_list[0]
            # check that the only transition starts from the trivial phase and ends in the physical phase
            if trans_start_from_trivial(transition, i) and trans_end_physical(transition, i):
               if trans_strongly_EW_first_order(transition): # check if the transition is an SFOEWPT
                  cat_inds_Tcrit[1].append(i)
               elif transition[-1] == 1: # or FO
                  cat_inds_Tcrit[2].append(i)
               elif transition[-1] == 2: # or 2nd order
                  cat_inds_Tcrit[3].append(i)
               else:
                  cat_inds_Tcrit[-2].append(i)
            else:
               cat_inds_Tcrit[-2].append(i)
         else: # there are multiple steps in the chain
            transition_list_cleaned = []
            j = 0
            while j < len(transition_list):
               if transition_real(transition_list[j], i):
                  transition_list_cleaned.append(transition_list[j])
               else:
                  if len(transition_list_cleaned) > 0:
                     transition_list_cleaned[-1][3] = transition_list[j][3]
               j += 1
            # check if this is still a multi-step transition:
            if len(transition_list_cleaned) == 1:
               transition = transition_list_cleaned[0]
               # check that the only transition starts from the trivial phase and ends in the physical phase
               if trans_start_from_trivial(transition, i) and trans_end_physical(transition, i):
                  if trans_strongly_EW_first_order(transition): # check if the transition is an SFOEWPT
                     cat_inds_Tcrit[1].append(i)
                  elif transition[-1] == 1: # or FO
                     cat_inds_Tcrit[2].append(i)
                  elif transition[-1] == 2: # or 2nd order
                     cat_inds_Tcrit[3].append(i)
                  else:
                     cat_inds_Tcrit[-2].append(i)
               else:
                  cat_inds_Tcrit[-2].append(i)
            elif len(transition_list_cleaned) == 2:
               # check that the first step starts from the trivial phase
               if not trans_start_from_trivial(transition_list_cleaned[0], i):
                  cat_inds_Tcrit[-2].append(i)
               # check that the last step ends in the physical phase
               elif not trans_end_physical(transition_list_cleaned[-1], i):
                  cat_inds_Tcrit[-2].append(i)
               else:
                  # is the intermediate phase singlet only?
                  if trans_end_HSonly(transition_list_cleaned[0], i):
                     if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                        cat_inds_Tcrit[4].append(i)
                     elif transition_list_cleaned[1][-1] == 1:
                        cat_inds_Tcrit[5].append(i)
                     elif transition_list_cleaned[1][-1] == 2:
                        cat_inds_Tcrit[6].append(i)
                     else:
                        cat_inds_Tcrit[-2].append(i)
                  else:
                     if trans_strongly_EW_first_order(transition_list_cleaned[0]):
                        if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                           cat_inds_Tcrit[7].append(i)
                        elif transition_list_cleaned[1][-1] == 1:
                           cat_inds_Tcrit[8].append(i)
                        elif transition_list_cleaned[1][-1] == 2:
                           cat_inds_Tcrit[9].append(i)
                        else:
                           cat_inds_Tcrit[-2].append(i)
                     elif transition_list_cleaned[0][-1] == 1:
                        if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                           cat_inds_Tcrit[10].append(i)
                        elif transition_list_cleaned[1][-1] == 1:
                           cat_inds_Tcrit[11].append(i)
                        elif transition_list_cleaned[1][-1] == 2:
                           cat_inds_Tcrit[12].append(i)
                        else:
                           cat_inds_Tcrit[-2].append(i)
                     elif transition_list_cleaned[0][-1] == 2:
                        if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                           cat_inds_Tcrit[13].append(i)
                        elif transition_list_cleaned[1][-1] == 1:
                           cat_inds_Tcrit[14].append(i)
                        elif transition_list_cleaned[1][-1] == 2:
                           cat_inds_Tcrit[15].append(i)
                        else:
                           cat_inds_Tcrit[-2].append(i)
                     else:
                        cat_inds_Tcrit[-2].append(i)
            else:
               cat_inds_Tcrit[-2].append(i)
   except:
      cat_inds_Tcrit[-1].append(i)
   # and the same for the nucleation results
   try:
      out_Tnucl = np.loadtxt(inname+'/good_'+str(int(good_params[i,0]))+'_Tn_out.txt')
      if out_Tnucl.shape[0] == 0: # empty transition file
         cat_inds_Tnucl[0].append(i)
      elif len(out_Tnucl.shape) == 1: # only one entry in transition file
         transition = out_Tnucl
         # check that the only transition starts from the trivial phase and ends in the physical phase
         if trans_start_from_trivial(transition, i) and trans_end_physical(transition, i) and check_nucl_calc_success(transition) and check_ltphase_T(transition, i):
            if trans_strongly_EW_first_order(transition): # check if the transition is an SFOEWPT
               cat_inds_Tnucl[1].append(i)
            elif transition[-1] == 1: # or FO
               cat_inds_Tnucl[2].append(i)
            elif transition[-1] == 2: # or 2nd order
               cat_inds_Tnucl[3].append(i)
            else:
               cat_inds_Tnucl[-2].append(i)
         elif not check_nucl_calc_success(transition):
            cat_inds_Tnucl[-3].append(i)
         else:
            cat_inds_Tnucl[-2].append(i)
      else: # multiple transitions
         # order the transition by temperature
         j_Tn = np.argsort(out_Tnucl[:,8])[::-1]
         # find the first transition from the high-T minimum
         j_try = 0
         transition = out_Tnucl[j_Tn[j_try],:]
         while trans_start_from_trivial(transition, i) == False and j_try < len(j_Tn)-1 and check_ltphase_T(transition, i) == False:
            j_try += 1
            transition = out_Tnucl[j_Tn[j_try],:]
         # throw point out if there is no transition startingfrom the high-T minimum
         if j_try == len(j_Tn):
            cat_inds_Tnucl[-2].append(i)
         # collect further transitions
         transition_list = [transition]
         active_key = transition[3]
         while j_try < len(j_Tn)-1:
            j_try += 1
            if out_Tnucl[j_Tn[j_try],7] == active_key and check_ltphase_T(out_Tnucl[j_Tn[j_try],:], i):
               transition_list.append(out_Tnucl[j_Tn[j_try],:])
               active_key = out_Tnucl[j_Tn[j_try],3]
         # if there is only 1 step in the chain:
         if len(transition_list) == 1:
            transition = transition_list[0]
            # check that the only transition starts from the trivial phase and ends in the physical phase
            if trans_start_from_trivial(transition, i) and trans_end_physical(transition, i) and check_nucl_calc_success(transition):
               if trans_strongly_EW_first_order(transition): # check if the transition is an SFOEWPT
                  cat_inds_Tnucl[1].append(i)
               elif transition[-1] == 1: # or FO
                  cat_inds_Tnucl[2].append(i)
               elif transition[-1] == 2: # or 2nd order
                  cat_inds_Tnucl[3].append(i)
               else:
                  cat_inds_Tnucl[-2].append(i)
            elif not check_nucl_calc_success(transition):
               cat_inds_Tnucl[-3].append(i)
            else:
               cat_inds_Tnucl[-2].append(i)
         else: # there are multiple steps in the chain
            transition_list_cleaned = []
            j = 0
            while j < len(transition_list):
               if transition_real(transition_list[j], i):
                  transition_list_cleaned.append(transition_list[j])
               else:
                  if len(transition_list_cleaned) > 0:
                     transition_list_cleaned[-1][3] = transition_list[j][3]
               j += 1
            # check if this is still a multi-step transition:
            if len(transition_list_cleaned) == 1:
               transition = transition_list_cleaned[0]
               # check that the only transition starts from the trivial phase and ends in the physical phase
               if trans_start_from_trivial(transition, i) and trans_end_physical(transition, i) and check_nucl_calc_success(transition):
                  if trans_strongly_EW_first_order(transition): # check if the transition is an SFOEWPT
                     cat_inds_Tnucl[1].append(i)
                  elif transition[-1] == 1: # or FO
                     cat_inds_Tnucl[2].append(i)
                  elif transition[-1] == 2: # or 2nd order
                     cat_inds_Tnucl[3].append(i)
                  else:
                     cat_inds_Tnucl[-2].append(i)
               elif not check_nucl_calc_success(transition):
                  cat_inds_Tnucl[-3].append(i)
               else:
                  cat_inds_Tnucl[-2].append(i)
            elif len(transition_list_cleaned) == 2:
               # check that the first step starts from the trivial phase
               if not trans_start_from_trivial(transition_list_cleaned[0], i):
                  cat_inds_Tnucl[-2].append(i)
               # check that the last step ends in the physical phase
               elif not trans_end_physical(transition_list_cleaned[-1], i):
                  cat_inds_Tnucl[-2].append(i)
               # check that nucleation calculation was successful for both steps
               elif not (check_nucl_calc_success(transition_list_cleaned[0]) and check_nucl_calc_success(transition_list_cleaned[1])):
                  cat_inds_Tnucl[-3].append(i)
               else:
                  # is the intermediate phase singlet only?
                  if trans_end_HSonly(transition_list_cleaned[0], i):
                     if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                        cat_inds_Tnucl[4].append(i)
                     elif transition_list_cleaned[1][-1] == 1:
                        cat_inds_Tnucl[5].append(i)
                     elif transition_list_cleaned[1][-1] == 2:
                        cat_inds_Tnucl[6].append(i)
                     else:
                        cat_inds_Tnucl[-2].append(i)
                  else:
                     if trans_strongly_EW_first_order(transition_list_cleaned[0]):
                        if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                           cat_inds_Tnucl[7].append(i)
                        elif transition_list_cleaned[1][-1] == 1:
                           cat_inds_Tnucl[8].append(i)
                        elif transition_list_cleaned[1][-1] == 2:
                           cat_inds_Tnucl[9].append(i)
                        else:
                           cat_inds_Tnucl[-2].append(i)
                     elif transition_list_cleaned[0][-1] == 1:
                        if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                           cat_inds_Tnucl[10].append(i)
                        elif transition_list_cleaned[1][-1] == 1:
                           cat_inds_Tnucl[11].append(i)
                        elif transition_list_cleaned[1][-1] == 2:
                           cat_inds_Tnucl[12].append(i)
                        else:
                           cat_inds_Tnucl[-2].append(i)
                     elif transition_list_cleaned[0][-1] == 2:
                        if trans_strongly_EW_first_order(transition_list_cleaned[1]):
                           cat_inds_Tnucl[13].append(i)
                        elif transition_list_cleaned[1][-1] == 1:
                           cat_inds_Tnucl[14].append(i)
                        elif transition_list_cleaned[1][-1] == 2:
                           cat_inds_Tnucl[15].append(i)
                        else:
                           cat_inds_Tnucl[-2].append(i)
                     else:
                        cat_inds_Tnucl[-2].append(i)
            else:
               cat_inds_Tnucl[-2].append(i)
   except:
      cat_inds_Tnucl[-1].append(i)
