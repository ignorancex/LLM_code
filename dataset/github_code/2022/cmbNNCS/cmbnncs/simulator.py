# -*- coding: utf-8 -*-
"""
@author: Si-Yu Li
"""

#the original simmaker_v2_pycamb.py


import sys
sys.path.append('../..')
python_version = sys.version_info.major
import numpy as np
if python_version==3:
    from configparser import ConfigParser
elif python_version == 2:
    import ConfigParser
    import codes.PySM_public_changed.pysm as pysm
    from codes.PySM_public_changed.pysm.nominal import models
import os
import camb


#=============================================================================#
''' added by WGJ '''
# this parameters are get from Table 3 (TT+lowP) of Planck 2015 results -XIII
hubble = [67.31, 0.96]
ombh2 = [0.02222, 0.00023]
omch2 = [0.1197, 0.0022]
re_optical_depth = [0.078, 0.019]
scalar_amp_1 = [2.1955e-9, 0.0791e-9]
scalar_spectral_index_1 = [0.9655, 0.0062]

def sim_power_spectra(sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_As, sim_ns,
                      spectra_type='lensed_scalar', ell_start=2):
    '''
    spectra_type: 'lensed_scalar', 'unlensed_total'
    '''
    #Set up a new set of parameters for CAMB
    #pars = camb.CAMBparams()
    pars = camb.model.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and \
    #helium set using BBN consistency
    pars.set_cosmology(H0=sim_H0, ombh2=sim_ombh2, omch2=sim_omch2, tau=sim_tau)
    pars.InitPower.set_params(As=sim_As, ns=sim_ns)
    pars.set_for_lmax(2150, lens_potential_accuracy=0);
    #calculate results for these parameters
    results = camb.get_results(pars)
    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars)
    # unlensedtotCls = powers['unlensed_total']
    unlensedtotCls = powers[spectra_type]
    #Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries 
    #will be zero by default.
    #The different CL are always in the order TT, EE, BB, TE (with BB=0 for 
    #unlensed scalar results).
    CMB_outputscale = 7.42835025e12
    unlensedtotCls = unlensedtotCls*CMB_outputscale
    if ell_start==2:
        l = np.arange(unlensedtotCls.shape[0])[2:]
        unlensedtotCls = unlensedtotCls[2:, :]
    elif ell_start==0:
        l = np.arange(unlensedtotCls.shape[0])
    unlensedtotCls_ = np.c_[l, unlensedtotCls]
    return unlensedtotCls_

def sim_UniformParam(param, times_l = 5, times_r = 5):
    range_param = [param[0]-param[1]*times_l, param[0]+param[1]*times_r]
    sim_param = np.random.rand() * (range_param[1] - range_param[0]) + range_param[0]
    return sim_param

def sim_NormalParams(param, param_label):
    sim_param = np.random.randn() * param[1] + param[0]
#    print '%s:'%param_label, '%s'%sim
    return sim_param

def get_spectra(random='Normal', times=5, spectra_type='lensed_scalar'):
    if random == 'Normal':
        sim_H0 = sim_NormalParams(hubble, 'hubble')
        sim_ombh2 = sim_NormalParams(ombh2, 'ombh2')
        sim_omch2 = sim_NormalParams(omch2, 'omch2')
#        sim_tau = sim_NormalParams(re_optical_depth, 're_optical_depth')
        sim_tau = 0.078
        sim_As = sim_NormalParams(scalar_amp_1, 'scalar_amp_1')
        sim_ns = sim_NormalParams(scalar_spectral_index_1, 'scalar_spectral_index_1')
    elif random == 'Uniform':
        sim_H0 = sim_UniformParam(hubble, times_l=times, times_r=times)
        sim_ombh2 = sim_UniformParam(ombh2, times_l=times, times_r=times)
        sim_omch2 = sim_UniformParam(omch2, times_l=times, times_r=times)
#        times_tau_l = (0.078-0.003)/0.019
#        sim_tau = sim_UniformParam(re_optical_depth, times_l=times_tau_l, times_r=times)
        sim_tau = 0.078
        sim_As = sim_UniformParam(scalar_amp_1, times_l=times, times_r=times)
        sim_ns = sim_UniformParam(scalar_spectral_index_1, times_l=times, times_r=times)
    elif random == 'fixed':
        sim_H0 = hubble[0]
        sim_ombh2 = ombh2[0]
        sim_omch2 = omch2[0]
        sim_tau = re_optical_depth[0]
        sim_As = scalar_amp_1[0]
        sim_ns = scalar_spectral_index_1[0]
    sim_params = np.c_[sim_H0,sim_ombh2,sim_omch2,sim_tau,sim_As,sim_ns]
    unlensedtotCls = sim_power_spectra(sim_H0,sim_ombh2,sim_omch2,sim_tau,sim_As,sim_ns, spectra_type=spectra_type)
    return unlensedtotCls, sim_params
#=============================================================================#

if python_version==2:
    class myconf(ConfigParser.ConfigParser):
        def __init__(self, defaults = None):
            ConfigParser.ConfigParser.__init__(self, defaults = None)
        def optionxform(self, optionstr):
            return optionstr
elif python_version==3:
    #changed by WGJ
    class myconf(ConfigParser):
        def __init__(self, defaults = None):
            ConfigParser.ConfigParser.__init__(self, defaults = None)
        def optionxform(self, optionstr):
            return optionstr

class Spectra:
    def __init__(self, Cls = None, isCl = True, Name = '', Checkl = True):
        self.Name = Name
        self.Cls = Cls
        self.isCl = isCl
        self.isDl = not self.isCl
        if Checkl:
            self.lcheck()
    def lcheck(self, lmax = None):
        if type(self.Cls) != type(None):
            lmin = int(min(self.Cls[0]))
            if lmin > 0:
                tmp = np.zeros((np.shape(self.Cls)[0] - 1 , lmin)) 
                tmp1 = np.array([range(lmin)])
                tmp = np.concatenate((tmp1, tmp))
                self.Cls = np.concatenate((tmp, self.Cls), axis = 1)
            if lmax == None:
                lmax = max(self.Cls[0])
            lmax = int(lmax)
            self.Cls = self.Cls.transpose()[:lmax + 1].transpose()
    def Cl2Dl(self):
        if self.isDl:
            raise ValueError('Spectra have already been multiplied by factor l(l+1)/2/pi')
        data = self.Cls.transpose()
        for i in range(len(data)):
            data[i][1:] = data[i][1:] * data[i][0] * (data[i][0] + 1.) / 2. / np.pi
        self.Cls = data.transpose()
        self.isDl = True
        self.isCl = False
    def Dl2Cl(self):
        if self.isCl:
            raise ValueError('Spectra have already been divided by factor l(l+1)/2/pi')
        data = self.Cls.transpose()
        for i in range(len(data)):
            if data[i][0] == 0.:
                continue
            data[i][1:] = data[i][1:] / data[i][0] / (data[i][0] + 1.) * 2. * np.pi
        self.Cls = data.transpose()
        self.isDl = False
        self.isCl = True

class Components:
    def __init__(self, nside):
        self.nside = nside
        self.paramsample = {}
        self.parameters = {}
        self.needparamsample = True
        self.needrealizationsample = True
    def ParametersSampling(self, **kwargs):
        if 'seed' in kwargs.keys(): 
            np.random.seed(kwargs['seed'])
        for var in self.parameters.keys():
            if type(self.parameters[var]) == type([]):
                self.paramsample[var] = self.parameters[var][0] + np.random.randn() * self.parameters[var][1]
            else:
                self.paramsample[var] = self.parameters[var]
    def ReadParameter(self, filename = '', section = '', **kwargs):
        cp = myconf()
        cp.readfp(open(filename))
        self.parameters= dict(cp.items(section))
        for item in self.parameters.keys():
            self.parameters[item] = eval(self.parameters[item])
        pass
    def RealizationSampling(self, **kwargs):
        pass
    def WriteMap(self, prefix = '', frequencies = np.array([90.]), instrument_profile = None,
                 output_directory = None):
        if instrument_profile == None:
            instrument_profile = IdealInstrument.copy()
            instrument_profile['frequencies'] = frequencies
            instrument_profile['nside'] = self.nside
            instrument_profile['output_directory'] = output_directory # added
        else:
            instrument_profile['nside'] = self.nside
            if instrument_profile['frequencies'] == None:
                instrument_profile['frequencies'] = frequencies
        instrument_profile['output_prefix'] = prefix
        instrument = pysm.Instrument(instrument_profile)
#        instrument.observe(self.sky) # original
        # self.out_put = instrument.observe(self.sky) ##added
        self.out_put, self.noise_map = instrument.observe(self.sky) ##added
        return

class NoiseComponents(Components):
    def __init__(self, nside):
        Components.__init__(self, nside = nside)
        self.sky = pysm.Sky({})
        self.needparamsample = False
        self.needrealizationsample = True
    def ReadParameter(self):
        pass
    def ParametersSampling(self):
        pass
    def RealizationSampling(self):
        pass
    # def WriteMap(self):
    #     Components.WriteMap(self, instrument_profile = self.instrument)
    
    #added by WGJ
    def WriteMap(self, seed, prefix='', frequencies=np.array([90.]), instrument_profile=None,
                 sens_I=np.array([7.]), sens_P=np.array([7.]), output_directory=None):
        if instrument_profile is None:
            instrument_profile = LiteBIRD_Instrument.copy()
        instrument_profile['noise_seed'] = seed
        instrument_profile['output_prefix'] = prefix
        instrument_profile['frequencies'] = frequencies
        instrument_profile['nside'] = self.nside
        instrument_profile['sens_I'] = sens_I
        instrument_profile['sens_P'] = sens_P
        instrument_profile['output_directory'] = output_directory
        instrument = pysm.Instrument(instrument_profile)
        noise_map = instrument.noiser()
        if len(noise_map)!=1:
            raise ValueError("Please use one frequency!")
        self.noise_map = noise_map[0] #only used for one frequency
        return


class CMBComponents(Components):
    def __init__(self, nside, spectra_type='lensed_scalar'):
        Components.__init__(self, nside = nside)
        self.model = c2(nside)
        self.needparamsample = True
        self.needrealizationsample = True
        self.spectra_type = spectra_type
    def ReadParameter(self, filename = ''):
        Components.ReadParameter(self, filename = filename, section = 'CMB')
    def ParametersSampling(self, random='Normal', times=5):
#        Components.ParametersSampling(self)#original
#        data = ReadClFromCAMB('CMB_ML', params = self.paramsample, runCAMB = True)#original
        data, sim_params = ReadClFromPycamb(random=random, times=times, runCAMB = True, spectra_type=self.spectra_type) # added
        self.sim_params = sim_params
        self.model[0]['cmb_specs'] = data.Cls
    def RealizationSampling(self, seed):
        self.model[0]['cmb_seed'] = seed
        self.sky = pysm.Sky({'cmb':self.model})

class DustComponents(Components):
    def __init__(self, nside, model):
        Components.__init__(self, nside = nside)
        if model == 3:
            self.model = models('d7', nside)
            self.needparamsample = False
            self.needrealizationsample = True
        elif model == 1:
            self.model = models('d1', nside)
            self.needparamsample = False
            self.needrealizationsample = False
        elif model == 2:
            self.model = models('d1', nside)#'d1'
            self.needparamsample = True
            self.needrealizationsample = False
    def ReadParameter(self, filename = ''):
        Components.ReadParameter(self, filename = filename, section = 'Dust')
    def ParametersSampling(self):
        if not self.needparamsample:
            return
        Components.ParametersSampling(self)
    def RealizationSampling(self, seed, amplitude_randn='0.1One', spectralIndex_randn='0.1One', temp_randn='0'):
        if not self.needrealizationsample:
            self.model[0]['seed'] = seed#original
#            self.model[0]['draw_uval_seed'] = seed # added
#            self.model[0]['draw_uval'] = True #added 
            self.model[0]['spectralIndex_randn'] = spectralIndex_randn
            self.model[0]['amplitude_randn'] = amplitude_randn
            self.model[0]['temp_randn'] = temp_randn
            self.sky = pysm.Sky({'dust':self.model})
            return
        self.model[0]['draw_uval_seed'] = seed
        self.sky = pysm.Sky({'dust':self.model})

# added later
class FFComponents(Components):
    def __init__(self, nside, model):
        Components.__init__(self, nside = nside)
        if model == 1:
            self.model = models('f1', nside)
            self.needparamsample = False
            self.needrealizationsample = True
    def ReadParameter(self, filename = ''):
        Components.ReadParameter(self, filename = filename, section = 'Freefree')
    def ParametersSampling(self):
        if not self.needparamsample:
            return
        Components.ParametersSampling(self)
    def RealizationSampling(self, seed, amplitude_randn='0.1One', spectralIndex_randn='0.1One'):
        if not self.needrealizationsample:
            self.sky = pysm.Sky({'freefree':self.model})
            return
        self.model[0]['seed'] = seed
        self.model[0]['spectralIndex_randn'] = spectralIndex_randn
        self.model[0]['amplitude_randn'] = amplitude_randn
        self.sky = pysm.Sky({'freefree':self.model})

# added later
class SyncComponents(Components):
    def __init__(self, nside, model):
        Components.__init__(self, nside = nside)
        if model == 1:
            self.model = models('s1', nside)#'s1'
            self.needparamsample = False
            self.needrealizationsample = True
    def ReadParameter(self, filename = ''):
        Components.ReadParameter(self, filename = filename, section = 'Sync')
    def ParametersSampling(self):
        if not self.needparamsample:
            return
        Components.ParametersSampling(self)
    def RealizationSampling(self, seed, amplitude_randn='0.1One', spectralIndex_randn='0.1One'):
        if not self.needrealizationsample:
            self.sky = pysm.Sky({'synchrotron':self.model})
            return
        self.model[0]['seed'] = seed
        self.model[0]['spectralIndex_randn'] = spectralIndex_randn
        self.model[0]['amplitude_randn'] = amplitude_randn
        self.sky = pysm.Sky({'synchrotron':self.model})

# added by WGJ
class AMEComponents(Components):
    def __init__(self, nside, model):
        Components.__init__(self, nside = nside)
        if model == 1:
            self.model = models('a1', nside)
            self.needparamsample = False
            self.needrealizationsample = True
        elif model == 2:
            self.model = models('a2', nside)
            self.needparamsample = False
            self.needrealizationsample = True
    def ReadParameter(self, filename = ''):
        Components.ReadParameter(self, filename = filename, section = 'Ame')
    def ParametersSampling(self):
        if not self.needparamsample:
            return
        Components.ParametersSampling(self)
    def RealizationSampling(self, seed, amplitude_randn='0.1One', spectralIndex_randn='0.1One'):
        if not self.needrealizationsample:
            self.sky = pysm.Sky({'ame':self.model})
            return
        self.model[0]['seed'] = seed #
        self.model[1]['seed'] = seed #
        self.model[0]['spectralIndex_randn'] = spectralIndex_randn
        self.model[1]['spectralIndex_randn'] = spectralIndex_randn
        self.model[0]['amplitude_randn'] = amplitude_randn
        self.model[1]['amplitude_randn'] = amplitude_randn
        self.sky = pysm.Sky({'ame':self.model})


def ReadClFromCAMB(fileroot, params = {}, runCAMB = False):
    if runCAMB:
        cp = myconf()
        cp.readfp(open("camb/params.ini"))
        cp.set("Nominal", "output_root", fileroot)
        for item in params.keys():
            cp.set("Nominal", item, params[item])
        cp.write(open('camb/params_ML.ini', "w"))
        os.system('camb/camb camb/params_ML.ini')
    data = Spectra(Cls = np.loadtxt(fileroot + '_totCls.dat').transpose(), isCl = False, Name = 'CMB Spectra', Checkl = False)
    data.Cls = np.concatenate([data.Cls, np.zeros([3,len(data.Cls[0])])])
    return data

#===================== added by WGJ ===============#
def ReadClFromPycamb(random=None, times=None, runCAMB = False, spectra_type='lensed_scalar'):
    if runCAMB:
        cls, sim_params = get_spectra(random=random, times=times, spectra_type=spectra_type)
        data = Spectra(Cls = cls.transpose(), isCl = False, \
        Name = 'CMB Spectra', Checkl = False)
    data.Cls = np.concatenate([data.Cls, np.zeros([3,len(data.Cls[0])])])
    return data, sim_params
#==============================================================================

def c2(nside):
    return [{
        'model' : 'synfast',
        'nside' : nside,
        'cmb_seed' : 1111
        }]
    
IdealInstrument = {
       'frequencies' : None,
       'use_smoothing' : False,
       'beams' : None,
       'add_noise' : False,
       'sens_I' : None,
       'sens_P' : None,
       'nside' : 64,
       'noise_seed' : 1111,
       'use_bandpass' : False,
       'output_directory' : './',
       'output_prefix' : '',
       'output_units' : 'uK_CMB'
       }

TestInstrument = {
       'frequencies' : np.array([90.]),
       'use_smoothing' : False,
       'beams' : None,
       'add_noise' : True,
       'sens_I' : np.array([1.]),
       'sens_P' : np.array([1.]),
       'nside' : 64,
       'noise_seed' : 1111,
       'use_bandpass' : False,
       'output_directory' : './',
       'output_prefix' : '',
       'output_units' : 'uK_CMB'
       }

#added by WGJ
LiteBIRD_Instrument = {
       'frequencies' : np.array([90.]), #should change
       'use_smoothing' : False,
       'beams' : None,
       'add_noise' : True,
       'sens_I' : np.array([7]), #shoude change
       'sens_P' : np.array([7]), #shoude change
       'nside' : 64,
       'noise_seed' : 1111,
       'use_bandpass' : False,
       'output_directory' : './',
       'output_prefix' : '',
       'output_units' : 'uK_CMB'
       }


#%%
def sim_CMB(cmb_seed, frequ, random='Normal', times=5, spectra_type='lensed_scalar', 
            nside=512):
    '''
    spectra_type: 'lensed_scalar', 'unlensed_total'
    '''
    ComCMB = CMBComponents(nside, spectra_type=spectra_type)
##    ComCMB.ReadParameter('paramsML.ini')
    ComCMB.ParametersSampling(random=random, times=times)
    params = ComCMB.sim_params
#    print ComCMB.paramsample, '\n'
    ComCMB.RealizationSampling(seed = int(cmb_seed))
    ComCMB.WriteMap(frequencies = frequ)
    out_put = ComCMB.out_put
    return out_put, params

def sim_dust(dust_seed, frequ, amplitude_randn, spectralIndex_randn, temp_randn,
             nside=512):
###    ComDust = sm.DustComponents(nside, 3)
    ComDust = DustComponents(nside, 1)#use this
##    ComDust.ReadParameter('paramsML.ini')#don't use
    #ParametersSampling() don't use when using model 3 in DustComponents(nside, 2)
    ComDust.ParametersSampling()
    print (ComDust.paramsample, '\n')
    ComDust.RealizationSampling( seed = int(dust_seed), amplitude_randn=amplitude_randn, 
                                spectralIndex_randn=spectralIndex_randn, temp_randn=temp_randn)
    ComDust.WriteMap(frequencies = frequ)
    out_put = ComDust.out_put
    return out_put

def sim_sync(sync_seed, frequ, amplitude_randn, spectralIndex_randn,
             nside=512):
    ComSync = SyncComponents(nside, 1)
##    ComSync.ReadParameter('paramsML.ini')#don't use
    ComSync.ParametersSampling() 
#    print ComSync.paramsample, '\n'
    ComSync.RealizationSampling( seed = int(sync_seed), amplitude_randn=amplitude_randn, spectralIndex_randn=spectralIndex_randn)
    ComSync.WriteMap(frequencies = frequ)
    out_put = ComSync.out_put
    return out_put

def sim_free(free_seed, frequ, amplitude_randn, spectralIndex_randn,
             nside=512):
    ComFree = FFComponents(nside, 1)
##    ComFree.ReadParameter('paramsML.ini')#don't use
    ComFree.ParametersSampling() 
#    print ComSync.paramsample, '\n'
    ComFree.RealizationSampling( seed = int(free_seed), amplitude_randn=amplitude_randn, spectralIndex_randn=spectralIndex_randn)
    ComFree.WriteMap(frequencies = frequ)
    out_put = ComFree.out_put
    return out_put

def sim_ame(ame_seed, frequ, amplitude_randn, spectralIndex_randn='',
            nside=512):
    ComAME = AMEComponents(nside, 2)
##    ComAME.ReadParameter('paramsML.ini')#don't use
    ComAME.ParametersSampling() 
#    print ComAME.paramsample, '\n'
    ComAME.RealizationSampling( seed = int(ame_seed), amplitude_randn=amplitude_randn, spectralIndex_randn=spectralIndex_randn)
    ComAME.WriteMap(frequencies = frequ)
    out_put = ComAME.out_put
    return out_put

