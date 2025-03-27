"""
Run a variation of the SFRD 
(this is in a separate file so you dont have to rerun everything if one job fails)
"""
import numpy as np
import os
from subprocess import Popen, PIPE, call
import subprocess
import sys
import paths

import CallCosmicIntegration as CI

if __name__ == "__main__": 
   # Initialize values
   CI.init()
                                             
   ##################################################################
   # muz variations
   Call_Cosmic_Integration(CI.data_dir, CI.COMPASfilename, rCI.ate_file_name, jname = 'muz',
                           ZdepSFRD_param_sets =[[[CI.mu0_best, -0.01, CI.sigma0_best, CI.sigmaz_best, CI.alpha0_best], CI.fid_sfr_parameters],
                                                [[CI.mu0_best, -0.25, CI.sigma0_best, CI.sigmaz_best, CI.alpha0_best], CI.fid_sfr_parameters]],
                           partitions = 'demink,conroy,hernquist,shared', Wtime = "1:00:00", mem = "120000")

