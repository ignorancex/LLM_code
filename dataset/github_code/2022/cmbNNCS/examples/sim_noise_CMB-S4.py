import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import cmbnncs.simulator as simulator
import numpy as np
import time
start_time = time.time()


def sim_noise(noise_seed, freq, sens_I, sens_P):
    ComNoise = simulator.NoiseComponents(nside)
    ComNoise.WriteMap(noise_seed, frequencies=freq, sens_I=sens_I, sens_P=sens_P)
    out_put = ComNoise.noise_map
    return out_put


#%% generate the noise full map - training (test) data
nside = 512


part_n = 0
part_size = 1000
print('part_n: %s'%part_n, 'part_size: %s'%part_size, 'start_n: %s'%(part_n*part_size))

frequencies = [85, 95, 145, 155, 220, 270]; sens_IP = [1.6, 1.3, 2.0, 2.0, 5.2, 7.1]; fwhm = [27.0, 24.2, 15.9, 14.8, 10.7, 8.5]


np.random.seed(6)#note!!!
Noiseseed = np.random.choice(1000000, 50000, replace=False)
for i in range(part_size):
    for j in range(len(frequencies)):
        noise_I, noise_Q, noise_U = sim_noise(Noiseseed[i+part_n*part_size], [frequencies[j]], np.array([sens_IP[j]]), np.array([sens_IP[j]]))
        
        map_Q_piece = spherical.sphere2piecePlane(noise_Q, nside=nside)
        map_U_piece = spherical.sphere2piecePlane(noise_U, nside=nside)
        
        utils.savenpy('samples/full_map_nside%s/CMBS4_noise/noise_%sGHz_Q'%(nside, frequencies[j]),
                      'noise_%s'%(i+part_n*part_size), map_Q_piece, dtype=np.float32)
        utils.savenpy('samples/full_map_nside%s/CMBS4_noise/noise_%sGHz_U'%(nside, frequencies[j]),
                      'noise_%s'%(i+part_n*part_size), map_U_piece, dtype=np.float32)
        

#%%
print ('\n', "Time elapsed: %.3f" %((time.time()-start_time)/60), "mins")

