import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import numpy as np
import loader
import healpy as hp


nside = 512
map_type = 'I'#'I', 'Q', 'U'


frequencies = [100, 143, 217, 353]


component = 'CMB'; comp_sample = 'CMB'
# component = 'Dust'; amplitude_randn='0'; spectralIndex_randn='0.1One'; temp_randn = '0.05Multi'; comp_sample = 'Foregrounds_oneModel'
# component = 'Sync'; amplitude_randn='0'; spectralIndex_randn='0.1One'; temp_randn = ''; comp_sample = 'Foregrounds_oneModel'
# component = 'Free'; amplitude_randn='0'; spectralIndex_randn='0.1One'; temp_randn = ''; comp_sample = 'Foregrounds_oneModel'
# component = 'AME'; amplitude_randn='0'; spectralIndex_randn='0'; temp_randn = ''; comp_sample = 'Foregrounds_oneModel'


part_n = 0
part_size = 1000
print ('part_n:%s'%part_n, 'part_size:%s'%part_size, component, 'start_n: %s'%(part_n*part_size))

count = 0
beams = loader.get_planck_beams(nside=nside, relative_dir='obs_data')
for i in range(part_size):
    count += 1
    if count%50==0:
        print ('%s/%s'%(count, part_size))
    for freq in frequencies:
        if component=='CMB':
            comp = loader.load_oneFullMap(i+part_n*part_size, comp_sample=comp_sample, component=component, freq=90, map_type=map_type, nside=nside)
        else:
            comp = loader.load_oneFullMap(i+part_n*part_size, comp_sample=comp_sample, component=component, freq=freq, map_type=map_type, nside=nside,
                                          amplitude_randn=amplitude_randn, spectralIndex_randn=spectralIndex_randn, temp_randn=temp_randn)
        comp = spherical.piecePlanes2spheres(comp, nside=nside)
        comp_beam = hp.smoothing(comp, beam_window=beams[str(freq)], iter=1)
        comp_beam = spherical.sphere2piecePlane(comp_beam, nside=nside)
        if component=='CMB':
            utils.savenpy('samples/full_map_nside%s/%s/%sbeam_%sGHz_%s'%(nside,comp_sample,component,freq,map_type),'%sbeam_%s'%(component,i+part_n*part_size),
                          comp_beam, dtype=np.float32)
        elif component=='Dust':
            utils.savenpy('samples/full_map_nside%s/%s/%sbeam/%sbeam_A%s_Beta%s_T%s_%sGHz_%s'%(nside,comp_sample,component,component,amplitude_randn,spectralIndex_randn,temp_randn,freq,map_type),
                          '%sbeam_%s'%(component,i+part_n*part_size), comp_beam, dtype=np.float32)
        else:
            utils.savenpy('samples/full_map_nside%s/%s/%sbeam/%sbeam_A%s_Beta%s_%sGHz_%s'%(nside,comp_sample,component,component,amplitude_randn,spectralIndex_randn,freq,map_type),
                          '%sbeam_%s'%(component,i+part_n*part_size), comp_beam, dtype=np.float32)

