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
   # omega0 variations
   Call_Cosmic_Integration(CI.data_dir, CI.COMPASfilename, CI.rate_file_name, jname = 'omeg0',
                           ZdepSFRD_param_sets =[[[CI.mu0_best, CI.muz_best, 0.8, CI.sigmaz_best, CI.alpha0_best], CI.fid_sfr_parameters],
                                                [[CI.mu0_best, CI.muz_best, 1.4, CI.sigmaz_best, CI.alpha0_best], CI.fid_sfr_parameters]],
                           partitions = 'demink,conroy,hernquist,shared', Wtime = "1:00:00", mem = "120000")
