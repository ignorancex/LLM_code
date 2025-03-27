import yaml
import os

##########################
# simple function for opening the file
def read_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
##########################

##########################
"""
class that contains input info
"""
class Info(object):
    def __init__(self, input_file):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)
        
        if 'num_parallel' in p:
            self.num_parallel = p['num_parallel']
            assert self.num_parallel >= 1, "num_parallel must be at least 1"
        else:
            self.num_parallel = 1
        
        self.nside = p['nside']
        assert type(self.nside) is int and (self.nside & (self.nside-1) == 0) and self.nside != 0, "nside must be integer power of 2"
        
        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax >= 2, "ellmax must be integer >= 2"
        assert self.ellmax <= 3*self.nside-1, "ellmax must be less than 3*nside-1"
        
        self.ell_sum_max = p['ell_sum_max']
        assert type(self.ell_sum_max) is int and self.ell_sum_max >= 2, "ell_sum_max must be integer >=2"
        assert self.ell_sum_max <= 3*self.nside-1, "ell_sum_max must be less than 3*nside-1"

        self.freqs = p['freqs']
        assert len(self.freqs) == 2, "Currently only two frequency channels are supported"
        self.Nscales = p['Nscales']
        assert type(self.Nscales) is int and self.Nscales > 0, "Nscales"
        self.GN_FWHM_arcmin = p['GN_FWHM_arcmin']
        assert len(self.GN_FWHM_arcmin) == self.Nscales - 1, "GN_FWHM_arcmin"
        assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"
        self.tsz_amp = p['tSZ_amp']
        assert self.tsz_amp >= 0, 'tSZ_amp'
        self.noise = p['noise']
        assert self.noise >= 0, 'noise'

        self.pyilc_path = p['pyilc_path']
        assert type(self.pyilc_path) is str, "TypeError: pyilc_path must be str"
        assert os.path.isdir(self.pyilc_path), "pyilc path does not exist"

        self.cmb_map_file = p['cmb_map_file']
        assert type(self.cmb_map_file) is str, "TypeError: cmb_map_file must be str"
        assert os.path.isfile(self.cmb_map_file), "CMB map file does not exist"

        self.tsz_map_file = p['tsz_map_file']
        assert type(self.tsz_map_file) is str, "TypeError: tsz_map_file must be str"
        assert os.path.isfile(self.tsz_map_file), "tSZ map file does not exist"

        if 'wigner_file' in p:
            self.wigner_file = p['wigner_file']
            assert type(self.wigner_file) is str, "TypeError: wigner_file must be str"
            assert os.path.isfile(self.wigner_file), "wigner_file does not exist"
        else:
            self.wigner_file = None

        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir must be str"
        self.verbose = p['verbose']


        
