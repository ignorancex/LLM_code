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

   #################################################################
   # fiducial
   CI.Call_Cosmic_Integration(CI.data_dir, CI.COMPASfilename, CI.rate_file_name, jname = 'fid',
                           ZdepSFRD_param_sets =[[CI.fid_dpdZ_parameters, CI.fid_sfr_parameters]],
                           partitions = 'demink,conroy,hernquist,shared', Wtime = "1:00:00", mem = "120000")
                                               

