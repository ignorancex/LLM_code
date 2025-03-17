import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import cmbnncs.simulator as simulator
import numpy as np
import time
start_time = time.time()


def sim_Dust(dust_seed, frequ, amplitude_randn, spectralIndex_randn, temp_randn):
###    ComDust = simulator.DustComponents(nside, 3)
    ComDust = simulator.DustComponents(nside, 1)#use this
##    ComDust.ReadParameter('paramsML.ini')#don't use
    #ParametersSampling() don't use when using model 3 in DustComponents(nside, 2)
    ComDust.ParametersSampling()
    print (ComDust.paramsample, '\n')
    ComDust.RealizationSampling( seed = int(dust_seed), amplitude_randn=amplitude_randn, 
                                spectralIndex_randn=spectralIndex_randn, temp_randn=temp_randn)
    ComDust.WriteMap(frequencies = frequ)
    out_put = ComDust.out_put
    return out_put


#%% generate the Dust full map - training (test) data
nside = 512


# temp_randn = '0'
temp_randn = '0.05Multi'
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
print ('dust_freqs: %s'%frequencies, 'part_n: %s'%part_n, 'part_size: %s'%part_size, 'start_n: %s'%(part_n*part_size))


np.random.seed(2)#note!!!
Dustseed = np.random.choice(1000000, 50000, replace=False)
for i in range(part_size):
    for freq in frequencies:
        map_I, map_Q, map_U = sim_Dust(Dustseed[i+part_n*part_size], [freq], amplitude_randn, 
                                       spectralIndex_randn, temp_randn=temp_randn)
        
        map_I_piece = spherical.sphere2piecePlane(map_I, nside=nside)
        map_Q_piece = spherical.sphere2piecePlane(map_Q, nside=nside)
        map_U_piece = spherical.sphere2piecePlane(map_U, nside=nside)
        
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Dust/Dust_A%s_Beta%s_T%s_%sGHz_I'%(nside,amplitude_randn,spectralIndex_randn,temp_randn,freq),
                      'Dust_%s'%(i+part_n*part_size), map_I_piece, dtype=np.float32)
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Dust/Dust_A%s_Beta%s_T%s_%sGHz_Q'%(nside,amplitude_randn,spectralIndex_randn,temp_randn,freq),
                      'Dust_%s'%(i+part_n*part_size), map_Q_piece, dtype=np.float32)
        utils.savenpy('samples/full_map_nside%s/Foregrounds_oneModel/Dust/Dust_A%s_Beta%s_T%s_%sGHz_U'%(nside,amplitude_randn,spectralIndex_randn,temp_randn,freq),
                      'Dust_%s'%(i+part_n*part_size), map_U_piece, dtype=np.float32)


#%%
print ('\n', "Time elapsed: %.3f" %((time.time()-start_time)/60), "mins")

