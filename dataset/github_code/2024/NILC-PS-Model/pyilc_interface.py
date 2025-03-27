import subprocess
import yaml
import os

def setup_pyilc(inp, env, suppress_printing=False):
    '''
    Sets up yaml files for pyilc and runs the code

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself

    RETURNS
    -------
    None
    '''

    #set up yaml files for pyilc
    pyilc_input_params = {}
    pyilc_input_params['output_dir'] = str(inp.output_dir) + "/pyilc_outputs/"
    pyilc_input_params['output_prefix'] = ""
    pyilc_input_params['save_weights'] = "yes"
    pyilc_input_params['ELLMAX'] = inp.ell_sum_max
    pyilc_input_params['N_scales'] = inp.Nscales
    pyilc_input_params['GN_FWHM_arcmin'] = [inp.GN_FWHM_arcmin[i] for i in range(len(inp.GN_FWHM_arcmin))]
    pyilc_input_params['taper_width'] = 0
    pyilc_input_params['N_freqs'] = len(inp.freqs)
    pyilc_input_params['freqs_delta_ghz'] = inp.freqs
    pyilc_input_params['N_side'] = inp.nside
    pyilc_input_params['wavelet_type'] = "GaussianNeedlets"
    pyilc_input_params['bandpass_type'] = "DeltaBandpasses"
    pyilc_input_params['beam_type'] = "Gaussians"
    pyilc_input_params['beam_FWHM_arcmin'] = [1.4, 1.4] #update with whatever beam FWHM was used in the sim map construction; note that ordering should always be from lowest-res to highest-res maps (here and in the lists of maps, freqs, etc above)    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params['ILC_bias_tol'] = 0.01
    pyilc_input_params['N_deproj'] = 0
    pyilc_input_params['N_SED_params'] = 0
    pyilc_input_params['N_maps_xcorr'] = 0
    pyilc_input_params['save_as'] = 'fits'
    pyilc_input_params['freq_map_files'] = [f'{inp.output_dir}/maps/freq1.fits', f'{inp.output_dir}/maps/freq2.fits']
    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params_preserved_tsz = {'ILC_preserved_comp': 'tSZ'}
    pyilc_input_params_preserved_cmb.update(pyilc_input_params)
    pyilc_input_params_preserved_tsz.update(pyilc_input_params)
    CMB_yaml = f'{inp.output_dir}/pyilc_yaml_files/CMB_preserved.yml'
    tSZ_yaml = f'{inp.output_dir}/pyilc_yaml_files/tSZ_preserved.yml'
    with open(CMB_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_cmb, outfile, default_flow_style=None)
    with open(tSZ_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_tsz, outfile, default_flow_style=None)

    #run pyilc for preserved CMB and preserved tSZ
    stdout = subprocess.DEVNULL if suppress_printing else None
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {CMB_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {tSZ_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ', flush=True)
    
    return

def weight_maps_exist(inp):
    '''
    Checks whether all weight maps for a given simulation and scaling already exist

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    Bool, whether or not weight maps already exist
    '''
    
    for comp in ['CMB', 'tSZ']:
        for freq in range(len(inp.freqs)):
            for scale in range(inp.Nscales):
                if not os.path.exists(f"{inp.output_dir}/pyilc_outputs/weightmap_freq{freq}_scale{scale}_component_{comp}.fits"):
                    return False
    return True