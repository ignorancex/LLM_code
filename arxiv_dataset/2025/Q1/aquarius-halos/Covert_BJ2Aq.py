import numpy as np
import h5py
import sys

inputdata = np.genfromtxt(sys.argv[1]) # Here it only accepts .dat files from the "original website".
output = h5py.File(sys.argv[2],'w')
flag = inputdata.T[7] # This flag decides whether the particle is bound.

plist = ['PartType0','PartType1']
num = 1
for p in plist:
    output.create_group(p)
    if p == 'PartType0':
        lim = (flag < 0)
        sgnr = np.zeros(len(inputdata.T[0][lim]))
        output[p].create_dataset('SubgroupNr',data=sgnr)
    elif p == 'PartType1':
        lim = (flag >= 0)
        output[p].create_dataset('SubgroupNr',data=inputdata.T[0][lim])
    else:
        break
    output[p].create_dataset('Mass',data=np.array(inputdata.T[13][lim]))
    coords = np.vstack((inputdata.T[1][lim],inputdata.T[2][lim],inputdata.T[3][lim])).T
    output[p].create_dataset('Coordinates',data=coords)
    output[p].create_dataset('LastTreeIndex',data=inputdata.T[0][lim])
    velos = np.vstack((inputdata.T[4][lim],inputdata.T[5][lim],inputdata.T[6][lim])).T
    output[p].create_dataset('Velocities',data=velos)
    pids = range(num,len(inputdata.T[0][lim])+num)
    output[p].create_dataset('ParticleIDs',data=pids)
    num += len(inputdata.T[0][lim])
output.close()