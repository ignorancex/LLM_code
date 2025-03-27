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
	#SFR(z) variations
	Call_Cosmic_Integration(CI.root_out_dir, CI.COMPASfilename, CI.rate_file_name, 
	                        ZdepSFRD_param_sets = [[CI.fid_dpdZ_parameters, [0.01, 2.60, 3.20, 6.20]],
	                                              [CI.fid_dpdZ_parameters, [0.01, 2.77, 2.90, 4.70]]],
	                        partitions = 'demink,conroy,hernquist,shared', Wtime = "1:00:00", mem = "120000")
	    


