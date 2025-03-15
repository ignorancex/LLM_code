###imports
import sys
sys.path.append('/apps/python3/3.12.1/lib/python3.12/site-packages')
sys.path.append('g/data/jh2/ax8338/testvenv/bin/')

import numpy as np

import os
import time

os.chdir('/g/data/jh2/ax8338/action/action_function')
import functions as f
import actioncalc as ac
#########################################################################

# ACTUAL USING THE FUNCTIONS

import gc
#### run using python3 action_FINAL.py $i $j

i=int(sys.argv[1])    ###time snapshot


j=0        ### j=0: initial stars, j=1: simulations stars, j=2: dark matter particles

filepath = '/g/data/jh2/ax8338/fit_grid/all_565_fitgrid.h5'.  ##potential fitgrid path

if j==0:
    ID_list = np.loadtxt('/g/data/jh2/ax8338/action/init_sample_565.txt')
    output_file = '/g/data/jh2/ax8338/action/results/init_stars_paper_feb25.h5'



elif j==1:
    ID_list = np.loadtxt('/g/data/jh2/ax8338/action/NEW_unrunaction_newborn_565.txt')
    output_file = '/g/data/jh2/ax8338/action/results/vz_testrun_25jan.h5'


elif j==2:
    ID_list = np.loadtxt('/g/data/jh2/ax8338/action/dm_sample_565.txt')
    output_file = '/g/data/jh2/ax8338/action/results/sample_dm_565.h5'

start=time.time()

dic={}  
try:
    result = ac.action_func(i,j,filepath,ID_list) 
    if result is None:
        print(f"Skipping snapshot of {int(i)} Myr")
            # continue  # Skip to the next snapshot
        
    ID, J, V, C, M,A, Kappa, Nu_low,Nu_middle, R_g, L_z, L_z_gal = result
    print(f'snapshot of {int(i)} Myr done!')
    # Dictionary for the current snapshot
    snapshot_name = f'snapshot_{int(i)}'
    snapshot_data = {
        'ID': ID,
        'J': J,
        'V': V,
        'C': C,
        'M': M,
        'A':A,
        'Kappa': Kappa,
        'Nu_low': Nu_low,
        'Nu_middle':Nu_middle,
        'R_g':R_g,
        'L_z':L_z,
        'L_z_gal':L_z_gal
    }
    # Save the snapshot to the HDF5 file immediately
    
    ac.save_star_data_hdf5(output_file,snapshot_data,snapshot_name)
    end=time.time()
    print(f'Time taken {end-start} s.')
    # Clear the data to free memory
    del ID, J, V, C, M,A, Kappa, Nu_low,Nu_middle, R_g, L_z,L_z_gal, snapshot_data,result
    gc.collect()

except Exception as e:
    print(f"Error processing snapshot of {int(i)} Myr: {e}")
