import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import cmbnncs.simulator as simulator
import numpy as np
import time
start_time = time.time()


def sim_free(free_seed, frequ, amplitude_randn, spectralIndex_randn):
    ComFree = simulator.FFComponents(nside, 1)
##    ComFree.ReadParameter('paramsML.ini')#don't use
    ComFree.ParametersSampling() 
#    print ComSync.paramsample, '\n'
    ComFree.RealizationSampling( seed = int(free_seed), amplitude_randn=amplitude_randn, spectralIndex_randn=spectralIndex_randn)
    ComFree.WriteMap(frequencies = frequ)
    out_put = ComFree.out_put
    return out_put


#%% generate the Free-free full map - training (test) data
nside = 512


# amplitude_randn = '0'; spectralIndex_randn = '0' #training set: 1000 #
amplitude_randn = '0'; spectralIndex_randn = '0.1One' #training set: 1000 ##
# amplitude_randn = '0.1One'; spectralIndex_randn = '0' #training set: 1000 #
# amplitude_randn = '0.1One'; spectralIndex_randn = '0.1One' #training set: 1000 #
# amplitude_randn = '0.1Multi'; spectralIndex_randn = '0' #training set: 1000 #
# amplitude_randn = '0.1Multi'; spectralIndex_randn = '0.1One' #training set: 1000 #


part_n = 0 #0,1,...
part_size = 1000
frequencies = [100, 143, 217, 353] #for Planck
print ('free_freqs: %s'%frequencies, 'part_n: %s'%part_n, 'part_size: %s'%part_size, 'start_n: %s'%(part_n*part_size))


np.random.seed(4)#note!!!
Freefreeseed = np.random.choice(1000000, 50000, replace=False)
for i in range(part_size):
    for freq in frequencies:
        map_I = sim_free(Freefreeseed[i+part_n*part_size], [freq], amplitude_randn, spectralIndex_randn)[0]
        
        map_I_piece = spherical.sphere2piecePlane(map_I, nside=nside)
        
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Free/Free_A%s_Beta%s_%sGHz_I'%(nside,amplitude_randn,spectralIndex_randn,freq),
                      'Free_%s'%(i+part_n*part_size), map_I_piece, dtype=np.float32)


#%%
print ('\n', "Time elapsed: %.3f" %((time.time()-start_time)/60), "mins")

