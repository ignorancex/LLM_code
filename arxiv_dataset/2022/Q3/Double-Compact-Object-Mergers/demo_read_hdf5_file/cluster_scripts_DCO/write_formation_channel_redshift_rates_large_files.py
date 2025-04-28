import numpy as np
import sys
import h5py as h5

#Quick fudge to make import from ../Scripts work
# sys.path.append('../../common_code/')
from PostProcessingScripts import * 
from formation_channels import * 

import subprocess
import time


# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
# [0,    1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14, 15,  16,   17,  18,  19]

if __name__ == '__main__':


    for DCOtype in [ 'all']:
        print()
        print()
        print()
        print()
        print('at DCOtype =', DCOtype)
        # BPSnameslist[1:-1]:
        # for BPSmodelName in  BPSnameslist[5:-1]:
        for BPSmodelName in  ['A']:
            print()
            print()
            print(BPSmodelName)
            
            

            # path for files 
            path_dir = '/n/holystore01/LABS/berger_lab/Lab/fbroekgaarden/DATA/all_dco_legacy_CEbug_fix/'
            path_ = path_dir
            path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
            # full_data_path  = path_ + 'COMPASOutput'+ '.h5'
            full_data_path  = path_ + 'COMPASOutputReduced'+ '.h5'

            fdata = h5.File(full_data_path,'r')
            print('keys: ', fdata.keys())
            # print( 'weights_intrinsicPerRedshift', fdata['weights_intrinsicPerRedshift']['R_333_z_0.249'][...].squeeze)
            fdata.close()

            # subprocess.call(['python3', '../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py'], args_ )
            # command = 'python3 ../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py', +  '--dco_type "BHNS"'
            os.system('python3 ../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration_large_files.py --dco_type all'  +' \
                --mu0 0.035 --muz -0.23 --sigma0 0.39 --sigmaz 0.0 --alpha 0.0 \
                --weight "weight" \
                --zstep 1\
                --m1min 5. \
                --aSF 0.01 --bSF 2.77 --cSF 2.9 --dSF 4.7 \
                --path ' + full_data_path + ' \
                --maxz_append 10 \
                --append_fc_summary True')

            print('done with ', BPSmodelName)

            print()
            print()
            print()
            print()

            print("Waiting for 15 seconds.")
            time.sleep(15)
            print("Wait is over.")

############
    # for DCOtype in [ 'BHNS','BNS', 'BBH',]:
    #     print()
    #     print()
    #     print()
    #     print()
    #     print('at DCOtype =', DCOtype)

    #     for BPSmodelName in  BPSnameslist[:]:
    #         print()
    #         print()
    #         print(BPSmodelName)
            
    #         DCOname = DCOname_dict[DCOtype]

    #         # path for files 
    #         path_dir = '/Volumes/Andromeda2/DATA/AllDCO_bugfix/'
    #         path_ = path_dir
    #         path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
    #         full_data_path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'



    #         # subprocess.call(['python3', '../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py'], args_ )
    #         # command = 'python3 ../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py', +  '--dco_type "BHNS"'
    #         os.system('python3 ../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py --dco_type ' + DCOtype +' \
    #             --mu0 0.035 --muz -0.23 --sigma0 0.39 --sigmaz 0.0 --alpha 0.0 \
    #             --weight "weight" \
    #             --zstep .001\
    #             --m1min 5. \
    #             --aSF 0.01 --bSF 2.77 --cSF 2.9 --dSF 4.7 \
    #             --path ' + full_data_path + ' \
    #             --maxz_append 10 \
    #             --append_fc_summary True')

    #         print('done with ', BPSmodelName)
    #         print()
    #         print()
    #         print()
    #         print()






# do a single one again 

    # for DCOtype in [ 'BBH',]:
    #     print('at DCOtype =', DCOtype)
        
    #     for BPSmodelName in  [BPSnameslist[-1]]:
    #         print(BPSmodelName)
            
    #         DCOname = DCOname_dict[DCOtype]

    #         # path for files 
    #         path_dir = '/Volumes/Andromeda2/DATA/AllDCO_bugfix/'
    #         path_ = path_dir
    #         path_ = path_ + alphabetDirDict[BPSmodelName] +'/'
    #         full_data_path  = path_ + 'COMPASCompactOutput_'+ DCOtype +'_' + BPSmodelName + '.h5'



    #         # subprocess.call(['python3', '../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py'], args_ )
    #         # command = 'python3 ../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py', +  '--dco_type "BHNS"'
    #         os.system('python3 ../../Double-Compact-Object-Mergers/demo_read_hdf5_file/FastCosmicIntegration.py --dco_type "BHNS" \
    #             --mu0 0.035 --muz -0.23 --sigma0 0.39 --sigmaz 0.0 --alpha 0.0 \
    #             --weight "weight" \
    #             --zstep .001\
    #             --m1min 5. \
    #             --aSF 0.01 --bSF 2.77 --cSF 2.9 --dSF 4.7 \
    #             --path ' + full_data_path + ' \
    #             --maxz_append 10 \
    #             --append_fc_summary True')

    #         print('done with ', BPSmodelName)


