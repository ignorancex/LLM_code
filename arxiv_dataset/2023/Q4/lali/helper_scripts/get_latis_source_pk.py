import numpy as np
import h5py
from nbodykit.lab import HDFCatalog
from nbodykit.lab import FFTPower

cat = HDFCatalog('./LATIS_source_catalog.hdf5', dataset='source')
print(cat['Coordinates'].shape)
boxsize = [94,54,484]
Nmesh = [19, 11, 95]
mesh = cat.to_mesh(Nmesh=Nmesh, position='Coordinates', 
                BoxSize=boxsize, compensated=True)
print(mesh.compute().shape)
fftpow = FFTPower(mesh, mode='2d', BoxSize=boxsize, kmin=0, kmax=1,dk=0.03, Nmu=6,
                los=[0,0,1], save_3d_power=True)
power = fftpow.power
k, mu, pkmu = power['k'], power['mu'], power['power']
pk3d = fftpow.pk3d



with h5py.File('LATIS_source_power.hdf5','w') as fw:
    fw['k'] = k
    fw['mu'] = mu
    fw['pkmu'] = pkmu
    fw['pk3d'] = pk3d
    fw['x'] = np.squeeze(fftpow.x3d[0])
    fw['y'] = np.squeeze(fftpow.x3d[1])
    fw['z'] = np.squeeze(fftpow.x3d[2])