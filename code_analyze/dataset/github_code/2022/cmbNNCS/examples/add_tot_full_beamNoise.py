import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import numpy as np
import loader


'''
Note: the Planck noise can be downloaded from http://pla.esac.esa.int/pla 
'''

def get_Fulltot(num=0, noise_num=0, freq=30, map_type='I', nside=512, comp_samples={}, components={}, 
                amplitude_randns={}, spectralIndex_randns={}, temp_randn=''):
    cmb = loader.load_oneFullMap(num, comp_sample=comp_samples['cmb'], component=components['cmb'], freq=freq, map_type=map_type, nside=nside)
    dust = loader.load_oneFullMap(num, comp_sample=comp_samples['dust'], component=components['dust'], freq=freq, map_type=map_type, nside=nside,
                                  amplitude_randn=amplitude_randns['dust'], spectralIndex_randn=spectralIndex_randns['dust'],
                                  temp_randn=temp_randn)
    sync = loader.load_oneFullMap(num, comp_sample=comp_samples['sync'], component=components['sync'], freq=freq, map_type=map_type, nside=nside,
                                  amplitude_randn=amplitude_randns['sync'], spectralIndex_randn=spectralIndex_randns['sync'])
    ame = loader.load_oneFullMap(num, comp_sample=comp_samples['ame'], component=components['ame'], freq=freq, map_type=map_type, nside=nside,
                                 amplitude_randn=amplitude_randns['ame'], spectralIndex_randn=spectralIndex_randns['ame'])
    noise = loader.load_oneFullMap(noise_num, comp_sample=comp_samples['noise'], component=components['noise'], freq=freq, map_type=map_type, nside=nside)
    if map_type=='I':
        free = loader.load_oneFullMap(num, comp_sample=comp_samples['free'], component=components['free'], freq=freq, map_type=map_type, nside=nside,
                                      amplitude_randn=amplitude_randns['free'], spectralIndex_randn=spectralIndex_randns['free'])
        tot_map = cmb + sync + dust + ame + free + noise#!!!
    else:
        tot_map = cmb + sync + dust + ame + noise#!!!
    return tot_map


#%%
nside = 512 # 256, 512, 1024
map_type = 'I' #'I', 'Q', 'U'


comp_samples = {'cmb':'CMB', 'dust':'Foregrounds_oneModel', 'sync':'Foregrounds_oneModel', 
                'free':'Foregrounds_oneModel', 'ame':'Foregrounds_oneModel',
                'noise':'planck_noise'}


components = {'cmb':'CMBbeam', 'dust':'Dustbeam', 'sync':'Syncbeam', 'free':'Freebeam', 'ame':'AMEbeam', 'noise':'noise'}


amplitude_randns = {'dust':'0', 'sync':'0', 'free':'0', 'ame':'0'}
spectralIndex_randns = {'dust':'0.1One', 'sync':'0.1One', 'free':'0.1One', 'ame':'0'}


temp_randn = '0.05Multi'; tot_comp = 'TotBeamNoiseA0Beta0.1OneT0.05Multi'


tot_sample = 'Tot_oneModel'


frequencies = [100,143,217,353]


tot_freq = ''
for i in range(len(frequencies)):
    tot_freq = tot_freq + str(frequencies[i])


part_n = 0
part_size = 1000
print ('part_n:%s'%part_n, 'part_size:%s'%part_size, 'start_n: %s'%(part_n*part_size))


np.random.seed(10)
noise_nums = np.random.choice(300, 15000)

count = 0
for i in range(part_size):
    count += 1
    if count%5==0:
        print ('%s/%s'%(count, part_size))
    
    # print(noise_nums[i+part_n*part_size])
    
    tot = []
    for freq in frequencies:
        # add cmb & noise & forgrounds
        tot_map = get_Fulltot(num=i+part_n*part_size, noise_num=noise_nums[i+part_n*part_size], freq=freq, map_type=map_type, nside=nside,
                              comp_samples=comp_samples, components=components, amplitude_randns=amplitude_randns, 
                              spectralIndex_randns=spectralIndex_randns, temp_randn=temp_randn)
        
        # all frequencies
        tot.append(tot_map)
    tot = np.array(tot)
    if len(frequencies)==1:
        tot = tot[0]
    
    utils.savenpy('samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside,tot_sample,tot_comp,tot_freq,map_type),'%s_%s'%(tot_comp,i+part_n*part_size),
                  tot, dtype=np.float32)

