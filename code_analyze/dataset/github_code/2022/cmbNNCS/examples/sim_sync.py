import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import cmbnncs.simulator as simulator
import numpy as np
import time
start_time = time.time()


def sim_sync(sync_seed, frequ, amplitude_randn, spectralIndex_randn):
    ComSync = simulator.SyncComponents(nside, 1)
##    ComSync.ReadParameter('paramsML.ini')#don't use
    ComSync.ParametersSampling() 
#    print ComSync.paramsample, '\n'
    ComSync.RealizationSampling( seed = int(sync_seed), amplitude_randn=amplitude_randn, spectralIndex_randn=spectralIndex_randn)
    ComSync.WriteMap(frequencies = frequ)
    out_put = ComSync.out_put
    return out_put


#%% generate the Synchrotron full map - training (test) data
nside = 512


# amplitude_randn = '0'; spectralIndex_randn = '0' #training set: 1000 #
amplitude_randn = '0'; spectralIndex_randn = '0.1One' #training set: 1000 ##
# amplitude_randn = '0'; spectralIndex_randn = '0.1Multi' #training set: 1000 #
# amplitude_randn = '0.1One'; spectralIndex_randn = '0' #training set: 1000 #
# amplitude_randn = '0.1One'; spectralIndex_randn = '0.1One' #training set: 1000 #
# amplitude_randn = '0.1One'; spectralIndex_randn = '0.1Multi' #training set: 1000 #
# amplitude_randn = '0.1Multi'; spectralIndex_randn = '0' #training set: 1000 #
# amplitude_randn = '0.1Multi'; spectralIndex_randn = '0.1One' #training set: 1000 #
# amplitude_randn = '0.1Multi'; spectralIndex_randn = '0.1Multi' #training set: 1000 #


part_n = 0 #0,1,...
part_size = 1000
frequencies = [100, 143, 217, 353] #for Planck
# frequencies = [85, 95, 145, 155, 220, 270] #for CMB-S4
print ('sync_freqs: %s'%frequencies, 'part_n: %s'%part_n, 'part_size: %s'%part_size, 'start_n: %s'%(part_n*part_size))


np.random.seed(3)#note!!!
Syncseed = np.random.choice(1000000, 50000, replace=False)
for i in range(part_size):
    for freq in frequencies:
        map_I, map_Q, map_U = sim_sync(Syncseed[i+part_n*part_size], [freq], amplitude_randn, spectralIndex_randn)
        
        map_I_piece = spherical.sphere2piecePlane(map_I, nside=nside)
        map_Q_piece = spherical.sphere2piecePlane(map_Q, nside=nside)
        map_U_piece = spherical.sphere2piecePlane(map_U, nside=nside)
        
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Sync/Sync_A%s_Beta%s_%sGHz_I'%(nside,amplitude_randn,spectralIndex_randn,freq),
                      'Sync_%s'%(i+part_n*part_size), map_I_piece, dtype=np.float32)
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Sync/Sync_A%s_Beta%s_%sGHz_Q'%(nside,amplitude_randn,spectralIndex_randn,freq),
                      'Sync_%s'%(i+part_n*part_size), map_Q_piece, dtype=np.float32)
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Sync/Sync_A%s_Beta%s_%sGHz_U'%(nside,amplitude_randn,spectralIndex_randn,freq),
                      'Sync_%s'%(i+part_n*part_size), map_U_piece, dtype=np.float32)
        

#%%
print ('\n', "Time elapsed: %.3f" %((time.time()-start_time)/60), "mins")

