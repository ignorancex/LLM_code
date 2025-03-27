"""
# Make smaller hdf5 files out of your data

This notebook is meant to grab the large and bulky COMPAS_Output_wWeights.hdf5 file and extract individual rate files that only contain the rate up to z = 0.5

This file is heavily based on the beautifully written h5copy.py written by Jeff Riley
"""

import sys, os
import numpy as np
import h5py as h5
import contextlib

import time

######################################
## locations and flags
data_dir    = '/n/holystore01/LABS/hernquist_lab/Users/lvanson/CompasOutput/v02.19.04/SFRD_fit_data/fWR1.0coolWind1.0/'

loc         = data_dir  +  '/output/COMPAS_Output_wWeights.h5' # Source file
outFname    = data_dir + 'output/Rate_info.hdf5' # Destination file name
overwrite   = True # overwrite your destination file?

start_time = time.time()



#########################################
# Check if output already exists
#########################################
ok = True
h5FileAccessPropertyList = h5.h5p.create(h5.h5p.FILE_ACCESS)

################################################
print('Reading from: ',loc)

# check whether output file already exists
# if it does exist, check whether it is an existing file or existing directory
existingOutputFile = False
print('outFname:', outFname, '\n')
if os.path.exists(outFname):
    if os.path.isfile(outFname):
        print('its a file')
        existingOutputFile = True
        if not overwrite: 
            ok = False
            print('Error encountered:', outFname, 'is the name of an existing file while overwrite = ',overwrite,'- choose a different output filename')
        else:
            os.remove(outFname)
    elif os.path.isdir(outFname):
        print('Error encountered:', outFname, 'is the name of an existing directory - choose a different output filename')
        existingOutputFile = False
    else:
        print('Error encountered:', outFname, 'is the name of an existing filesystem object - choose a different output filename')
        existingOutputFile = False


#########################################
# Check if you're good and create output
#########################################
if ok:
    print('create the output file')
    # open the output file - create it if necessary
    h5FileAccessPropertyList = h5.h5p.create(h5.h5p.FILE_ACCESS)

    # using low-level functions here so we can provide the propertly list
    if existingOutputFile:                                                                                                  # output file exists?
        try:                                                                                                                # yes
            outHDF5file = h5.h5f.open(outFname.encode('utf-8'), fapl = h5FileAccessPropertyList)                        # open it
        except Exception as e:                                                                                              # error opening file
            print('Error occurred while disabling HDF5 dataset cache:', str(e))                                             # announce error
            ok = False                                                                                                      # fail
    else:                                                                                                                   # output file does not exist
        try:
            outHDF5file = h5.h5f.create(outFname.encode('utf-8'), fapl = h5FileAccessPropertyList)                      # create it
        except Exception as e:                                                                                              # error creating file
            print('Error occurred while disabling HDF5 dataset cache:', str(e))                                             # announce error
            ok = False        

            
    #########################################
    # Start copying
    #########################################
    ## Open source hdf5 file
    with h5.File(loc ,'r') as srcFile:

        # And the destination file
        with contextlib.closing(outHDF5file) as h5OutFid:                                                               # close the file when done...
            # process input files and directories
            with h5.File(h5OutFid) as outFile:
            
                n_copied = 0 #you've not copied anything 
                
                # Loop over all the groups
                for groupkey in srcFile.keys():

                    # we only want to copy over the rate info
                    if 'Rates' in groupkey:
                        srcgGroup    = srcFile[groupkey]
                        outFileGroup = outFile.require_group(groupkey)
                    else:
                        continue

                    print('working on ', groupkey)

                    for key in srcgGroup.keys():
                        print('key', key)
                        
                        try:
                            # These parameteres are the same for every rate group
                            if key in ['Average_SF_mass_needed',  'SEED', 'redshifts']:
                                if n_copied == 0:      # this is the first group, copy everything
                                    # This parameter is annoying and not necessary                        
                                    if key == 'Average_SF_mass_needed': 
                                        continue

                                    # redshifts are bin edges
                                    elif key == 'redshifts':
                                        outFile[key] = srcgGroup[key][:11]

                                    # also copy the rest of them 
                                    else:
                                        outFile[key] = srcgGroup[key][:]

                                elif n_copied >0:
                                    # No need to safe these more than once
                                    continue 

                            # The rest are merger rate info 
                            # Only save the first 10 bins of merger rates
                            elif key in ['merger_rate', 'detection_rateO3', 'detection_rateO1']:
                                outFileGroup[key] = srcgGroup[key][:,:10]

                            # 'merger_rate_z0' and 'DCO_mask' are only 1 wide
                            else:
                                outFileGroup[key] = srcgGroup[key][:]
                        except:
                            print('ERROR in group ',groupkey, '\n key: ', key)
                        
                    n_copied += 1
                    print('keys for this group', outFileGroup.keys() )

                print('final groups in out file', outFile.keys())
                

print("--- %s seconds ---" % (time.time() - start_time))
            
            

                
            
