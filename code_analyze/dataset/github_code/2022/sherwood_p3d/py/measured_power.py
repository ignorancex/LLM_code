import os
import pickle
import fitsio
import sherwood_simulation as she_sim


def get_repo_dir():
    """ Return directory of this repository, using environment variable """

    assert ('SHERWOOD' in os.environ),'Define SHERWOOD environ'
    return os.environ['SHERWOOD']


def get_power_fname(grid,power_type,pickle=False):
    """ Find out (full path) filename for a given measured power spectrum """

    # get full path to repo data
    if pickle:
        data_dir=get_repo_dir()+'/pickled_data/'
        extension='p'
    else:
        data_dir=get_repo_dir()+'/data/'
        extension='fits'

    # get nametag from grid
    nametag=grid.get_nametag()

    if power_type == "flux_p1d":
        return '{}/flux_p1d/p1d_{}.{}'.format(data_dir,nametag,extension)
    elif power_type == "flux_p3d":
        return '{}/flux_p3d/p3d_{}_20_16_20.{}'.format(data_dir,nametag,extension)
    elif power_type == "cross_p3d":
        return '{}/cross_p3d/cross_{}_20_16_20.{}'.format(data_dir,nametag,extension)
    elif power_type == "halo_p3d":
        # only one mu bin when not adding RSDs
        n_mu = (16 if grid.add_rsd else 1)
        return '{}/halo_p3d/halo_{}_20_{}_20.{}'.format(data_dir,nametag,n_mu,extension)
    else:
        raise ValueError("unknown power spectrum type",power_type)


def read_fits_power(fname,power_type):

    # read FITS file
    hdul = fitsio.FITS(fname)
    hdu = hdul[power_type.upper()]
    header = hdu.read_header()
    # setup simulation object 
    sim = she_sim.SherwoodSimulation(L_hMpc=header['L_HMPC'],
                n_part=header['N_PART'])
    # setup grid object
    if "flux" in power_type: 
        grid = she_sim.Grid(simulation=sim,snapshot_num=header['SNAPSHOT_NUM'],
                    n_xy=header['N_XY'],n_z=header['N_Z'],axis=header['AXIS'])
    else:
        logMh_min=header['LOGMH_MIN']
        logMh_max=header['LOGMH_MAX']
        add_rsd=header['ADD_RSD']
        grid=she_sim.HaloGrid(simulation=sim,
                    snapshot_num=header['SNAPSHOT_NUM'],
                    n_xy=header['N_XY'],n_z=header['N_Z'],axis=header['AXIS'],
                    logMh_min=logMh_min,logMh_max=logMh_max,add_rsd=add_rsd)

    # collect information to return
    power={'power_type':power_type,'grid':grid}
    
    # extra metadata depending on power type
    if "flux" not in power_type:
        power['shot_noise']=header['SHOT_NOISE']
    if power_type != "halo_p3d":
        power['mean_flux']=header['MEAN_FLUX']
    if power_type != "flux_p1d":
        power['n_k_bins']=header['N_K_BINS']
        power['n_mu_bins']=header['N_MU_BINS']
        power['k_hMpc_max']=header['K_HMPC_MAX']

    # actual power measurements
    if power_type == "flux_p1d":
        power['kp_hMpc']=hdu['KP_HMPC'][:]
        power['p1d_hMpc']=hdu['P1D_HMPC'][:]
    else:
        power['p3d_hMpc']=hdu['P3D_HMPC'][:]
        power['k_hMpc']=hdu['K_HMPC'][:]
        power['mu']=hdu['MU'][:]
        power['counts']=hdu['COUNTS'][:]

    hdul.close()

    return power


def assert_sim(sim1,sim2):
    assert sim1.L_hMpc==sim2.L_hMpc, 'inconsistent L_hMpc'
    assert sim1.n_part==sim2.n_part, 'inconsistent n_part'


def assert_grid(grid1,grid2,power_type):
    # assert simulation metadata
    assert_sim(grid1.sim,grid2.sim)
    # and extra metadata in grid
    assert grid1.snapshot_num==grid2.snapshot_num, 'inconsistent snapshot_num'
    assert grid1.n_xy==grid2.n_xy, 'inconsistent n_xy'
    assert grid1.n_z==grid2.n_z, 'inconsistent n_z'
    assert grid1.axis==grid2.axis, 'inconsistent axis'
    if "flux" not in power_type:
        assert grid1.add_rsd==grid2.add_rsd, 'inconsistent add_rsd'
        assert grid1.logMh_min==grid2.logMh_min, 'inconsistent logMh_min'
        assert grid1.logMh_max==grid2.logMh_max, 'inconsistent logMh_max'


def get_power_from_grid(grid,power_type):
    """Return measured power spectrum corresponding to input grid"""

    # get filename for corresponding FITS file
    fname = get_power_fname(grid,power_type,pickle=False)

    # get measured power and grid metadata 
    power=read_fits_power(fname,power_type)

    # make sure that grid metadata is consistent
    assert_grid(grid,power['grid'],power_type)

    return power


def get_power_from_pickle(grid=None,power_type=None,fname=None):
    """Return measured power spectrum from pickled file"""

    if fname is None:
        fname = get_power_fname(grid,power_type,pickle=True)
    print('will unpickle from file',fname)
    return pickle.load(open(fname,"rb"))


