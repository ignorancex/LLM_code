"""
link your Rate file outputs together in one final rate file
"""
import numpy as np
import os
from subprocess import Popen, PIPE, call
import subprocess
import sys
import paths
import time
from fnmatch import fnmatch
import h5py


import CallCosmicIntegration as CI

if __name__ == "__main__": 
	# Initialize values
	CI.init()

	###############################
	print('Now link all rate files in one final file')
	if os.path.isfile(CI.data_dir+'/RateData/'+CI.rate_file_name):
		dest_file = h5py.File(CI.data_dir+'/RateData/'+CI.rate_file_name,'r+')                                                                         # Open the final file that will contain all the links
	else:
		dest_file = h5py.File(CI.data_dir+'/RateData/'+CI.rate_file_name,'w')                                                                         # Create a new final file that will contain all the links

	for dirpath, dirnames, filenames in os.walk(CI.data_dir+'/RateData/'):                                                                          # walk directory
		absDirpath = os.path.abspath(dirpath)                                                                                       # absolute path
		print('Processing directory', absDirpath)                                                                                   # announce directory being processed
		#
		for filename in filenames:                                                                                                  # for each filename
			if fnmatch(filename, '*_' + CI.rate_file_name):                                                                               # filename matches filter?
				src_filename = absDirpath + '/' + filename
				print('you will be copying file',src_filename)
				src_data     = h5py.File(absDirpath+'/'+filename, 'r')
				#
				for srcGroupName in src_data.keys():                                                                                 # For every group
					#
					for srcdataName in src_data[srcGroupName].keys():                                                                 # And every data set
						#
						if srcGroupName +'/'+srcdataName in  dest_file :         # Check if link already existst
							print('link exists, delete it before recreating')
							del(dest_file[srcGroupName +'/'+srcdataName])                                                              #Yes? than delete it firts
						dest_file[srcGroupName +'/'+srcdataName] = h5py.ExternalLink(src_filename, srcGroupName +'/'+srcdataName)
	src_data.close()

	print('All done :)! ')
	dest_file.close()

	# h5_copy_string = 'python %s/h5copy.py  %s -r 2 -o %s --filter *%s  > %s'%(script_dir, data_dir, data_dir+'/'+CI.rate_file_name, CI.rate_file_name[:-3] ,data_dir+"/slurm_out/combineh5.log" )
	# filter for files with rate_file_name, but remove the extension .h5
	# os.system(h5_copy_string)
	# wait 1 min for h5copy to finish
	# time.sleep(60)
