"""
This script checks the outcome of all your jobs
"""
import numpy as np
import os
from subprocess import Popen, PIPE, call
import subprocess
import sys
import time
from fnmatch import fnmatch
import h5py
sys.path.append( str(os.getcwd()) + '/src/scripts/')

import init_values as In


if __name__ == "__main__": 

    # Initialize values
    In.init()


    check_job_completionID = np.loadtxt(In.data_dir+ '/RateData/CI_job_IDs.txt', delimiter=',')
    print('check_job_completionID', check_job_completionID)

    ###########################
    # Now wait for your (last) job to be done
    for job_id in check_job_completionID:
        job_id=int(job_id)
        print('job_id', job_id)
        command = "sacct  -j %s --format State "%(job_id)
        print(command)
        done = False
        while not done:
            # First check the status of your job with sacct
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            nline = 0
            while True:
                line = p.stdout.readline()
                nline +=1
                #print(line)
                if not line:
                    break
                if nline == 3:
                    break


            result = str(line)
            print('result = ', result)
            if b"COMPLETE" in line:
                print('YAY your job finished! ID = %s'%(job_id) )
                done = True
            elif b"FAILED" in line:
                print('Job failed :(  ID=%s'%(job_id) )
                done = True
            elif b"CANCELLED" in line:
                print('Job was CANCELLED  ID=%s'%(job_id) )
                done = True
            elif b"TIMEOUT" in line:
                print('Job timed out! :(  ID=%s'%(job_id) )
                done = True
            elif np.logical_or(b"RUNNING" in line, b"PENDING" in line):
                print('darn, still running, check back in 2 min')
                time.sleep(120) # Sleep 2 min and then check back

    print(10* "*" + " You done with all your jobs! " + 10* "*")


    # Copy your files into one filter for files with rate_file_name, but remove the extension .h5
    input_Ratedata_dir = '/n/holystore01/LABS/hernquist_lab/Users/lvanson/home_output/SFRD_fit/src/data/RateData/'
    h5_copy_string = 'python %s/h5copy.py  %s -o %s --filter *%s  > %s'%(In.script_dir, input_Ratedata_dir, In.data_dir+'/RateData/'+In.rate_file_name, In.rate_file_name[:-3] , In.data_dir+"/slurm_out/combineh5.log" )
    
    print(h5_copy_string)

    os.system(h5_copy_string)
    # wait 2 min for h5copy to finish
    # time.sleep(120)

    # rename the job_ID file to trigger re-run when needed
    os.rename(In.data_dir+ '/RateData/CI_job_IDs.txt', In.data_dir+ '/RateData/used_CI_job_IDs.txt')
