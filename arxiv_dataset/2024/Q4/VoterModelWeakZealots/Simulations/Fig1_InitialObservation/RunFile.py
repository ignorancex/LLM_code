from Params import *

import subprocess
import threading

import time

import os,shutil

starttime = time.time()




#########################################
###Argparse##############################
#########################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Number of Regions')
parser.add_argument('-P','--ZealotRatio',type=float,required=True,
        help='Proportion of Zealots')
parser.add_argument('-F','--Fitness',type=float,required=True,
        help='Mutant Fitness')

args = parser.parse_args()

P = float(args.ZealotRatio)
F = float(args.Fitness)


SubSaveDirName = (SaveDirName +
    "/P_%0.3f_F_%0.3f"%(P,F))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for P,F",P,F)
#########################################

shutil.copy("Params.py",SaveDirName)


plist = []

for C in CList:
    p=subprocess.Popen(['nice', '-n', '18', 'python', 'Script.py', '-C', str(C),'-P',str(P),'-F',str(F),'-d',str(SubSaveDirName)])
    plist.append(p)

for p in plist:
    p.wait()


"""
# Maximum number of concurrent threads
max_threads = 40

# Lock to synchronize thread access
lock = threading.Lock()

# Function to execute a command
def execute_command(C):
    command = ['nice', '-n', '18', 'python', 'Script.py', '-C', str(C),'-P',str(P),'-F',str(F),'-d',str(SubSaveDirName)]
    with lock:
        print("Executing:", command)
    subprocess.Popen(command).wait()

# List to store the running threads
threads = []

# Iterate over the CList
for C in CList:
    # Wait until a thread is available
    while len(threads) >= max_threads:
        threads = [thread for thread in threads if thread.is_alive()]

    # Create a new thread and start executing the command
    thread = threading.Thread(target=execute_command, args=(C,))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()
"""

endtime = time.time()


print("Time taken:",endtime-starttime)
