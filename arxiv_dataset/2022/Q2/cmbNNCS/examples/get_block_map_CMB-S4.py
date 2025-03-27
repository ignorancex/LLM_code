import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import numpy as np
import loader


#%%
nside = 512
map_type = 'Q' #'Q', 'U'


#for CMB-S4
comp_sample = 'CMB'; component = 'CMBfwhm'; freq_cmb = 220; fwhm = 10.7
# comp_sample = 'Tot_oneModel'; component = 'TotFwhmNoiseA0Beta0.1OneT0.05Multi'


frequencies = [85, 95, 145, 155, 220, 270]#; fwhm = [27.0, 24.2, 15.9, 14.8, 10.7, 8.5]


tot_freq = ''
for i in range(len(frequencies)):
    tot_freq = tot_freq + str(frequencies[i])


part_n = 0
part_size = 1000
print ('part_n:%s'%part_n, 'part_size:%s'%part_size, 'start_n: %s'%(part_n*part_size))


count = 0
for i in range(part_size):
    count += 1
    if count%5==0:
        print ('%s/%s'%(count, part_size))
    
    if component=='CMBfwhm':
        piece_maps = loader.load_oneFullMap_CMBS4(i+part_n*part_size, comp_sample=comp_sample, component=component,
                                                  freq=freq_cmb, fwhm=fwhm, map_type=map_type, nside=nside)
        save_freq = freq_cmb
    elif component=='TotFwhmNoiseA0Beta0T0.05Multi' or component=='TotFwhmNoiseA0Beta0T0' or component=='TotFwhmNoiseA0Beta0.1OneT0.05Multi' \
        or component=='TotFwhmNoiseA0Beta0.1OneT0.05MultiIndependentNoise' or component=='TotFwhmA0Beta0.1OneT0.05Multi':
        piece_maps = loader.load_oneFullMap_CMBS4(i+part_n*part_size, comp_sample=comp_sample, component=component,
                                                  freq=tot_freq, map_type=map_type, nside=nside)
        save_freq = tot_freq
    
    block_0 = spherical.piecePlanes2blocks(piece_maps, nside=nside)['block_0']
    if component=='CMBfwhm':
        utils.savenpy('samples/full_map_nside%s/%s/%s%sBlock0_%sGHz_%s'%(nside,comp_sample,component,fwhm,save_freq,map_type),'%sBlock0_%s'%(component,i+part_n*part_size),
                      block_0, dtype=np.float32)
    elif component=='TotFwhmNoiseA0Beta0T0.05Multi' or component=='TotFwhmNoiseA0Beta0T0' or component=='TotFwhmNoiseA0Beta0.1OneT0.05Multi' \
        or component=='TotFwhmNoiseA0Beta0.1OneT0.05MultiIndependentNoise' or component=='TotFwhmA0Beta0.1OneT0.05Multi':
        utils.savenpy('samples/full_map_nside%s/%s/%sBlock0_%sGHz_%s'%(nside,comp_sample,component,save_freq,map_type),'%sBlock0_%s'%(component,i+part_n*part_size),
                      block_0, dtype=np.float32)


