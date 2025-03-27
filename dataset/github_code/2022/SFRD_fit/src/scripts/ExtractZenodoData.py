# Trying to get Zenodo data to work
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import paths
import init_values as In

data_dir = str(paths.data) +'/'


if __name__ == "__main__": 

    # Initialize values
    In.init()


out_fname = data_dir + In.COMPASfilename #'/COMPAS_Output_wWeights.h5'
tar_name  = data_dir + '/COMPAS_Output_wWeights.tar.gz'


# check if file exists, if not, extract it
if not os.path.isfile(out_fname):
	print('file %s does not exist, extract'%(out_fname))
	# open file
	file = tarfile.open(tar_name)

	# extracting file
	file.extractall(data_dir)
	file.close()
else:
	print('file %s exists, nothing to do'%(out_fname))


print('!!! Je bent nu in deze file!!!!')
data = h5.File(out_fname, 'r')
print(data.keys())

data.close



