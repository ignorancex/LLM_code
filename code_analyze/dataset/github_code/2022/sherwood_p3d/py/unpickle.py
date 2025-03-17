import sherwood_simulation as she_sim
import measured_power
import numpy as np
from os import listdir, path
import fitsio


def from_pickle_to_fits(pickle_fname,fits_fname,power_type):
    # open pickled file to read
    data=measured_power.get_measured_power(fname=pickle_fname)

    # main simulation grid (renamed for cross)
    if power_type == "cross_p3d":
        grid=data.halos
    else:
        grid=data.grid

    # sim metadata
    metadata={'L_HMPC':grid.sim.L_hMpc}
    metadata['N_PART']=grid.sim.n_part

    # grid metadata
    metadata['SNAPSHOT_NUM']=grid.snapshot_num
    metadata['N_XY']=grid.n_xy
    metadata['N_Z']=grid.n_z
    metadata['AXIS']=grid.axis
    if "flux" not in power_type:
        metadata['ADD_RSD']=grid.add_rsd
        metadata['LOGMH_MIN']=grid.logMh_min
        metadata['LOGMH_MAX']=grid.logMh_max
        metadata['SHOT_NOISE']=data.shot_noise
    if power_type != "halo_p3d":
        metadata['MEAN_FLUX']=data.mean_flux
   
    # power metadata
    if power_type != "flux_p1d":
        metadata['N_K_BINS']=data.n_k_bins
        metadata['N_MU_BINS']=data.n_mu_bins
        metadata['K_HMPC_MAX']=data.k_hMpc_max

    # power spectrum measurement
    power={}
    if power_type == "flux_p1d":
        power['P1D_HMPC']=data.p1d_hMpc
        power['KP_HMPC']=data.kp_hMpc
    else:
        power['P3D_HMPC']=data.p3d_hMpc
        power['K_HMPC']=data.k_hMpc
        power['MU']=data.mu
        power['COUNTS']=data.counts

    # open FITS file to write
    print('FITS fname',fits_fname)
    fits = fitsio.FITS(fits_fname,'rw',clobber=True)
    extname=power_type.upper()
    cols=list(power.values())
    names=list(power.keys())
    fits.write(cols, names=names, header=metadata, extname=extname)
    fits.close()


def read_fits(fname,power_type):
    # read FITS file
    hdul = fitsio.FITS(fname)
    hdu = hdul[power_type.upper()]
    for key in ['L_HMPC','N_XY','SNAPSHOT_NUM']:
        print(key,hdu.read_header()[key])
    if power_type == "flux_p1d":
        kp=hdu['KP_HMPC'][:]
        p1d=hdu['P1D_HMPC'][:]
    else:
        p3d_hMpc=hdu['P3D_HMPC'][:]
        k_hMpc=hdu['K_HMPC'][:]
        mu=hdu['MU'][:]
        counts=hdu['COUNTS'][:]
    hdul.close()


# loop over all files in pickled data
repo_dir='/Users/font/Projects/sherwood_p3d/'
for power_type in ['flux_p1d','flux_p3d','halo_p3d','cross_p3d']:
    pickle_dir='{}/pickled_data/{}/'.format(repo_dir,power_type)
    fits_dir='{}/data/{}/'.format(repo_dir,power_type)
    for fname in listdir(pickle_dir):
        pre, suff = path.splitext(fname)
        fits_fname='{}/{}.fits'.format(fits_dir,pre)
        print('\n',power_type,fname)
        from_pickle_to_fits(pickle_dir+fname,fits_fname,power_type)
        read_fits(fits_fname,power_type)

