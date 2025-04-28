import subprocess
from Params import *

import time

starttime = time.time()

for P in PList:
    for F in FList:
        #if P < (1-F):
        print("Begin (P,F): ",P,F)
        subprocess.call(['python','RunFile.py','-P',str(P),'-F',str(F)])
        print("Finished (P,F): ",P,F)

endtime = time.time()

print("Time taken:",endtime-starttime)
