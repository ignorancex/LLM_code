import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import cmbnncs.simulator as simulator
import numpy as np
import time
start_time = time.time()


def sim_CMB(cmb_seed, frequ, random='Normal', times=5, spectra_type='lensed_scalar'):
    '''
    spectra_type: 'lensed_scalar', 'unlensed_total'
    '''
    ComCMB = simulator.CMBComponents(nside, spectra_type=spectra_type)
##    ComCMB.ReadParameter('paramsML.ini')
    ComCMB.ParametersSampling(random=random, times=times)
    params = ComCMB.sim_params
#    print ComCMB.paramsample, '\n'
    ComCMB.RealizationSampling(seed = int(cmb_seed))
    ComCMB.WriteMap(frequencies = frequ)
    out_put = ComCMB.out_put
    return out_put, params


#%% generate the CMB full map - training (test) data
nside = 512


part_n = 0 #0,1,...
part_size = 1000
print ('nside:%s'%nside, 'part_n:', part_n, 'part_size:', part_size, 'start_n: %s'%(part_n*part_size))


np.random.seed(1)#note!!!
CMBseed = np.random.choice(1000000, 50000, replace=False)
parameters = []
for i in range(part_size):
    cmb_map, params = sim_CMB(CMBseed[i+part_n*part_size], [90.])
    cmb_I, cmb_Q, cmb_U = cmb_map
    parameters.append(params.reshape(-1))
    
    cmb_I_plane = spherical.sphere2piecePlane(cmb_I, nside=nside)
    cmb_Q_plane = spherical.sphere2piecePlane(cmb_Q, nside=nside)
    cmb_U_plane = spherical.sphere2piecePlane(cmb_U, nside=nside)
    
    # save the full map
    utils.savenpy('samples/full_map_nside%s/CMB/CMB_90GHz_I'%nside, 'CMB_%s'%(i+part_n*part_size), cmb_I_plane, dtype=np.float32)
    utils.savenpy('samples/full_map_nside%s/CMB/CMB_90GHz_Q'%nside, 'CMB_%s'%(i+part_n*part_size), cmb_Q_plane, dtype=np.float32)
    utils.savenpy('samples/full_map_nside%s/CMB/CMB_90GHz_U'%nside, 'CMB_%s'%(i+part_n*part_size), cmb_U_plane, dtype=np.float32)
    
parameters = np.array(parameters)
utils.savenpy('samples/full_map_nside%s/CMB/CMB_90GHz_I'%nside, 'params_partN%s_partSize%s'%(part_n,part_size),
              parameters, dtype=np.float32)


#%%
print ('\n', "Time elapsed: %.3f" %((time.time()-start_time)/60), "mins")

