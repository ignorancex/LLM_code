import numpy as np
import h5py
import importlib
from os import system
from lim_lytomo import lim
from nbodykit.lab import *
import dask.array as da

def make_fake_sim(boxsize):
    
    a = np.linspace(0,1,5)*boxsize*1000
    x, y, z = np.meshgrid(a,a,a)
    parts = np.zeros((a.size**3,3))
    parts[:,0] = x.ravel()
    parts[:,1] = y.ravel()
    parts[:,2] = z.ravel()
    system(' mkdir ./groups_001')
    with h5py.File('./groups_001/fof_n1.hdf5','w') as fw:
        fw['Subhalo/SubhaloPos'] = parts

def get_Mock_lim(compensated=True, sfr_type='SubhaloSFR', mass_cut=10, 
                 co_model='chung+21', angular_res=4, refine_fac=10, sim_type='TNG', boxsize=None, brange=None):
    from astropy.cosmology import Planck15 as cosmo
    from lim_lytomo import stats
    from lim_lytomo import mock_lya
    from lim_lytomo import lim
    importlib.reload(stats)
    importlib.reload(lim)
    importlib.reload(mock_lya)
    
    if boxsize is not None:
        boxsize = [boxsize,boxsize,boxsize]
    
    
    MockLim = lim.MockLim(snap=1, z=2.4442257045541464, axis=3, 
                               basepath='./', boxsize=boxsize, brange=brange,
                               co_model=co_model, sfr_type=sfr_type, halo_type='Subhalo', 
                               compensated=compensated, refine_fac=refine_fac, sim_type='TNG',
                               behroozi_avsfr=None, angular_res=angular_res,
                               mass_cut=mass_cut, silent_mode=True, rsd=False)
    return MockLim

def test_load_halos(MockLim):
    return 0

def test_do_CIC(MockLim):
    """Not working yet, q needs the cat shape"""
    mesh = MockLim.do_CIC(q=np.ones(MockLim.halos['Coordinates'].shape[0]), qlabel='1')
    return mesh

def test_co_temp(MockLim):
    return 0

    
if __name__ == '__main__':
    boxsize= 1 # inc cMpc/h
    make_fake_sim(boxsize)
    mock_full = get_Mock_lim(boxsize=boxsize)
    for b in [2,3]:
        mock_subbox = get_Mock_lim(brange=[0, .25*b, 0, .25*b,0, .25*b])
        c = np.where(np.linspace(0,1,5) <= .25*b)[0].size**3
        assert mock_subbox.halos['Coordinates'].compute().shape[0] == c