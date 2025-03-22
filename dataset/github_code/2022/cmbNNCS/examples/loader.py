import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.data_process as dp
import numpy as np
import healpy as hp
import astropy.io.fits as pf
import torch


#%% beam
def BL(freq, lmax=None, relative_dir='../obs_data'):
    '''
    Note: the beam file LFI_RIMO_R3.31.fits and HFI_RIMO_BEAMS_R3.01.tar.gz can be obtained from http://pla.esac.esa.int/pla
    '''
    if freq<=70:
        beamfile = relative_dir + '/beam/LFI_RIMO_R3.31.fits'#for 30GHz, 44GHz, and 70GHz
        data, header = pf.getdata(beamfile, 'BEAMWF_0%sX0%s'%(freq,freq), header=True)
    else:
        beamfile = relative_dir + '/beam/BeamWf_HFI_R3.01/Bl_T_R3.01_fullsky_%sx%s.fits'%(freq, freq)# for >100GHz
        data, header = pf.getdata(beamfile, 'WINDOW FUNCTION', header=True)
    return data.field(0).flatten()[:lmax+1]

def get_planck_beams(nside=512, relative_dir='../obs_data'):
    lmax = 3*nside + 50
    freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
    beams = {}
    for freq in freqs:
        bl = BL(freq, lmax=lmax, relative_dir=relative_dir)
        beams[str(freq)] = bl
    return beams

def get_gauss_beams(fwhm=7, nside=512, lmax=None):
    """the fwhm of 143GHz is 7.27,
    test fwhm: 7, 6, 5, 4
    """
    if lmax is None:
        lmax = 3*nside + 50
    bl = hp.gauss_beam(fwhm*np.pi/10800., lmax=lmax)
    return bl


#%% load maps
def load_oneFullMap(num, comp_sample='', component='Free', freq=30, map_type='I', 
                    nside=512, amplitude_randn='0', spectralIndex_randn='0', temp_randn=''):
    
    if component=='CMBbeam' or component=='CMB' or component=='noise':
        file_path = 'samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside, comp_sample, component, freq, map_type)
    elif component=='Syncbeam' or component=='Sync' or component=='Freebeam' or component=='Free' or component=='AMEbeam' or component=='AME':
        file_path = 'samples/full_map_nside%s/%s/%s/%s_A%s_Beta%s_%sGHz_%s'%(nside, comp_sample, component, component,
                                                                             amplitude_randn, spectralIndex_randn, freq, map_type)
    elif component=='Dustbeam' or component=='Dust':
        file_path = 'samples/full_map_nside%s/%s/%s/%s_A%s_Beta%s_T%s_%sGHz_%s'%(nside, comp_sample, component, component,
                                                                             amplitude_randn, spectralIndex_randn, temp_randn, freq, map_type)
    elif component=='TotBeamNoiseA0Beta0.1OneT0.05Multi'or component=='TotBeam0.3NoiseA0Beta0.1OneT0.05Multi' or component=='TotBeamA0Beta0.1OneT0.05Multi':
        file_path = 'samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside, comp_sample, component, freq, map_type)
    else:
        raise NameError('The component name %s is not setted!'%component)
    
    file_name = '%s_%s'%(component, num)
    Map = np.load(file_path + '/' + file_name + '.npy')
    return Map

def load_oneFullMap_CMBS4(num, comp_sample='', component='Syncfwhm', freq=30, fwhm=int, map_type='I', 
                          nside=512, amplitude_randn='0', spectralIndex_randn='0', temp_randn=''):
    
    if component=='CMBfwhm':
        file_path = 'samples/full_map_nside%s/%s/%s%s_%sGHz_%s'%(nside, comp_sample, component, fwhm, freq, map_type)
    elif component=='Syncfwhm' or component=='AMEfwhm':
        file_path = 'samples/full_map_nside%s/%s/%s/%s%s_A%s_Beta%s_%sGHz_%s'%(nside, comp_sample, component, component, fwhm,
                                                                               amplitude_randn, spectralIndex_randn, freq, map_type)
    elif component=='Dustfwhm':
        file_path = 'samples/full_map_nside%s/%s/%s/%s%s_A%s_Beta%s_T%s_%sGHz_%s'%(nside, comp_sample, component, component, fwhm,
                                                                                   amplitude_randn, spectralIndex_randn, temp_randn, freq, map_type)
    elif component=='noise':
        file_path = 'samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside, comp_sample, component, freq, map_type)
    elif component=='TotFwhmNoiseA0Beta0T0.05Multi' or component=='TotFwhmNoiseA0Beta0T0' or component=='TotFwhmNoiseA0Beta0.1OneT0.05Multi' \
        or component=='TotFwhmNoiseA0Beta0.1OneT0.05MultiIndependentNoise' or component=='TotFwhmA0Beta0.1OneT0.05Multi':
        file_path = 'samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside, comp_sample, component, freq, map_type)
    else:
        raise NameError('The component name %s is not setted!'%component)
    
    file_name = '%s_%s'%(component, num)
    Map = np.load(file_path + '/' + file_name + '.npy')
    return Map


#%%
def trans_params(component='CMB', map_type='I', block=4):
    '''
    LFComp: low frequency components (Sync, Free, AME)
    '''
    factors = {'CMB_I_0':[5.0,0.0], 'CMB_I_4':[5.0,0.0], 'CMB_Q_0':[2.0,0.0], 'CMB_Q_4':[2.0,0.0], 'CMB_U_0':[2.0,0.0], 'CMB_U_4':[2.0,0.0],
               'Dust_I_0':[30.0,2.0], 'Dust_I_4':[550.0,2.0], 'Dust_Q_0':[5.0,0.0], 'Dust_Q_4':[5.0,0.0], 'Dust_U_0':[5.0,0.0], 'Dust_U_4':[5.0,0.0],
               'Sync_I_0':[10.0,1.0], 'Sync_I_4':[50.0,1.0], 'Sync_Q_0':[5.0,0.0], 'Sync_Q_4':[10.0,0.0], 'Sync_U_0':[1.0,0.0], 'Sync_U_4':[5.0,0.0],
               'Free_I_0':[30.0,1.0], 'Free_I_4':[500.0,1.0],
               'AME_I_4':[50.0,1.0], 'AME_Q_4':[8.0,0.0], 'AME_U_4':[8.0,0.0],
               'LFComp_I_4':[250.0,1.0]}
    return factors['%s_%s_%s'%(component,map_type,block)]

def transform(maps, component='CMB', map_type='I', block=4):
    trans = trans_params(component=component,map_type=map_type,block=block)
    maps = maps / trans[0] + trans[1]
    return maps

def inverse_transform(maps, component='CMB', map_type='I', block=4):
    trans = trans_params(component=component,map_type=map_type,block=block)
    maps = (maps - trans[1]) * trans[0]
    return maps

def load_oneFullMap_2(num, comp_sample='', component='Free', freq=30, map_type='I', nside=512, 
                      out_type='torch', data_source='sim'):
    
    if data_source=='sim':
        file_path = 'samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside, comp_sample, component, freq, map_type)
    elif data_source=='obs':
        file_path = 'obs_data/%s/%s_%sGHz_%s'%(comp_sample, component, freq, map_type)
    file_name = '%s_%s'%(component, num)
    Map = np.load(file_path + '/' + file_name + '.npy')
    
    if out_type=='cuda':
        Map = dp.numpy2cuda(Map)
    elif out_type=='torch':
        Map = dp.numpy2torch(Map)
    elif out_type=='numpy':
        pass
    return Map

def random_arrange_fullMap(map_nums, comp_sample='',component='Tot',map_type='I',
                           channels=[30,90,150,220],out_type='torch',normed=False,learn_component=None, 
                           nside=512,map_shape=None, block_map=False, map_3d=False, data_source='sim'):
    
    if block_map:
        map_shape = [nside, nside]
    else:
        if map_shape is None:
            map_shape = [nside*4, nside*3]
        else:
            map_shape = map_shape
        
    minisample_size = len(map_nums)
    channel_num = len(channels)
    
    frequencies = ''
    for i in range(len(channels)):
        frequencies = '%s%s'%(frequencies, str(channels[i]))
        
    if out_type=='cuda':
        subsample = torch.cuda.FloatTensor(minisample_size, channel_num, map_shape[0], map_shape[1]).zero_()
    elif out_type=='torch':
        subsample = torch.FloatTensor(minisample_size, channel_num, map_shape[0], map_shape[1]).zero_()
    
    for i in range(minisample_size):
        Map = load_oneFullMap_2(map_nums[i], comp_sample=comp_sample, component=component, freq=frequencies, map_type=map_type, nside=nside, out_type=out_type, data_source=data_source)
        subsample[i, :, :, :] = Map #for CMB (with shape of (nside,nside)) or Tot (with shape of (channel_num, nside, nside)) maps
    if normed:
        subsample = transform(subsample, component=learn_component,map_type=map_type,block=4)# block=4

    if map_3d:
        subsample = subsample[:, np.newaxis, :,:,:]
        return subsample
    else:
        return subsample

def load_oneFullMap_CMBS4_2(num, comp_sample='', component='Free', freq=30, fwhm=int, map_type='I', nside=512, 
                            out_type='torch', data_source='sim', block_map=False):
    
    if data_source=='sim':
        if component=='CMBfwhm':
            if block_map:
                file_path = 'samples/full_map_nside%s/%s/%s%sBlock0_%sGHz_%s'%(nside, comp_sample, component, fwhm, freq, map_type)#to be updated!
            else:
                file_path = 'samples/full_map_nside%s/%s/%s%s_%sGHz_%s'%(nside, comp_sample, component, fwhm, freq, map_type)
        else:
            file_path = 'samples/full_map_nside%s/%s/%s_%sGHz_%s'%(nside, comp_sample, component, freq, map_type)
    elif data_source=='obs':
        file_path = 'obs_data/%s/%s_%sGHz_%s'%(comp_sample, component, freq, map_type)
    if block_map:
        file_name = '%sBlock0_%s'%(component, num)#to be updated!
    else:
        file_name = '%s_%s'%(component, num)
    Map = np.load(file_path + '/' + file_name + '.npy')
    
    if out_type=='cuda':
        Map = dp.numpy2cuda(Map)
    elif out_type=='torch':
        Map = dp.numpy2torch(Map)
    elif out_type=='numpy':
        pass
    return Map

def random_arrange_fullMap_CMBS4(map_nums, comp_sample='',component='Tot',map_type='I',
                                 channels=[30,90,150,220],fwhm=int,out_type='torch',normed=False,learn_component=None, 
                                 nside=512,map_shape=None, block_map=False, map_3d=False, data_source='sim'):
    
    if block_map:
        map_shape = [nside, nside]
    else:
        if map_shape is None:
            map_shape = [nside*4, nside*3]
        else:
            map_shape = map_shape
        
    minisample_size = len(map_nums)
    channel_num = len(channels)
    
    frequencies = ''
    for i in range(len(channels)):
        frequencies = '%s%s'%(frequencies, str(channels[i]))
        
    if out_type=='cuda':
        subsample = torch.cuda.FloatTensor(minisample_size, channel_num, map_shape[0], map_shape[1]).zero_()
    elif out_type=='torch':
        subsample = torch.FloatTensor(minisample_size, channel_num, map_shape[0], map_shape[1]).zero_()
    
    for i in range(minisample_size):
        Map = load_oneFullMap_CMBS4_2(map_nums[i], comp_sample=comp_sample, component=component, freq=frequencies, fwhm=fwhm, map_type=map_type, nside=nside, out_type=out_type, data_source=data_source, block_map=block_map)
        subsample[i, :, :, :] = Map #for CMB (with shape of (nside,nside)) or Tot (with shape of (channel_num, nside, nside)) maps
    if normed:
        subsample = transform(subsample, component=learn_component,map_type=map_type,block=4)# block=4

    if map_3d:
        subsample = subsample[:, np.newaxis, :,:,:]
        return subsample
    else:
        return subsample

