#!/usr/bin/env python

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import ebf
import json
import optparse
from time import perf_counter as clock
import multiprocessing

import GalCraft.modules.ssp_loader as ssp_loader
import GalCraft.modules.binner as binner
import GalCraft.modules.spec_generator as spec_generator
import GalCraft.modules.cube_maker as cube_maker
import GalCraft.modules.cot as cot
import GalCraft.modules.utils as utils
from GalCraft.modules.log import Logger
from GalCraft._version import __version__


def run_GalCraft(CommandOptions):



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -   I N I T I A L I Z E   G A L C R A F T - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    t_init = clock()

    # - - - - - - - - - - INITIALIZATION - - - - - - - - - -

    # Initialize cube name and paths
    cube_name = CommandOptions.configName
    setup_cube_name = 'setup_' + cube_name
    configDir, modelDir, templateDir, outputDir = utils.initialize_Dirs(CommandOptions)
    # Load the Configuration file
    with open(configDir + setup_cube_name + '.json', 'r') as f:
        params = json.load(f)
    # Setup the outputs folder
    filepath = outputDir + cube_name + '/'
    if os.path.exists(filepath)==False: os.mkdir(filepath[:-1])
    # Setup logger
    if params['other_params']['mode'] == 'continue': # For continue mode, change the logger mode from 'w' to 'a'
        params['other_params']['log_mode'] = 'a'
    logger = Logger(logfile=filepath + 'outputs.log', mode=params['other_params']['log_mode']).get_log()
    logger.info('Loaded the setup file (path %s)' % (configDir + setup_cube_name + '.json'))
    logger.info('Outputs DIR: %s' % filepath)
    # Setup cpu cores to use
    if params['other_params']['ncpu'] == None:
        # n_cores = len(os.sched_getaffinity(0))
        n_cores = multiprocessing.cpu_count()
        logger.info('Change the CPU cores from %s to %s due to the CPU availability.' % (params['other_params']['ncpu'], n_cores) )
        params['other_params']['ncpu'] = n_cores
    else:
        logger.info('Run the framework using %s CPU cores.' % params['other_params']['ncpu'])



    # - - - - - - - - - - INSTRUMENT SETUP - - - - - - - - - -

    # Obtain ./instrument/ dir
    this_dir, this_filename = os.path.split(__file__)
    instrumentDir = os.path.join(this_dir, "instrument") + '/'
    # Load instrument information
    # If "instrument"=="MUSE-WFM", it will load "MUSE-WFM.json" and use the "spatial_resolution" and "spatial_bin";
    # If "instrument"=="DIY", you will design your own instrument using "spatial_resolution" and "spatial_bin" in the configuration file;
    # If "instrument"=="DEFAULT", it will generate a huge cube using all the particles and only will be used in the configuration file.
    inst = params['cube_params']['instrument']
    if inst.upper() != 'DIY' and inst.upper()!='DEFAULT':
        with open(instrumentDir + inst + '.json', 'r') as f:
            inst_params = json.load(f)
        for key in inst_params:
            params['cube_params'][key] = inst_params[key]
        logger.info('INST mode, apply the %s instrument spatial properties from the preset file.' % inst)
    elif inst.upper() == 'DEFAULT':
        logger.info('DEFAULT mode, will use all the particles and only the given spatial resolution will be used.')
    else:
        logger.info("DIY mode, apply the given spatial resolution and bin number.")



    # - - - - - - - - - - LOAD SPECTRAL TEMPLATES - - - - - - - - - -

    logger.info('Loading SSP models...')
    t = clock()
    ssp_model = ssp_loader.model(templateDir, instrumentDir, params['ssp_params'], params['other_params'], logger)
    ssp_model.oversample()
    logger.info('SSP model has been successfully loaded, time elapsed: %.2f s' % (clock() - t))



    # - - - - - - - - - - LOAD STELLAR CATALOG - - - - - - - - - -

    logger.info('Loading the E-Galaxia model from %s' % (modelDir + params['other_params']['model_name']))
    t = clock()
    d_t = ebf.read(modelDir + params['other_params']['model_name'],'/')
    # Check the need to locate/rotate the E-Galaxia model.
    if 'vr' in d_t.keys():
        logger.info('Model has already been rotated, no location/rotation transformation is applied.')
        for key in params['oparams']:
            params['oparams'][key] = None
    else:
        logger.info('Specifing the location/rotation of the galaxy')
        cot.observe(d_t, params['oparams'])
    # Check extinction values
    if params['cube_params']['use_extinc'] == True:
        assert 'exbv' in d_t.keys(), "No extinction in the loaded model, process terminated."
    # Check the need to apply the flux calibration due to the total Galaxy Stellar mass.
    if params['other_params']['nparticles'] != None:
        nparticles = params['other_params']['nparticles']
    else:
        if 'shuffle' in params['other_params']['model_name']:
            original_model_name = '_'.join([x for x in params['other_params']['model_name'][:-4].split('_') if 'shuffle' not in x]) + '.ebf'
            d_original = ebf.read(modelDir + original_model_name,'/')
            nparticles = len(d_original['px'])
            del d_original
        else:
            nparticles = len(d_t['vr'])
    # In case nparticles changed due to the "debug" mode.
    if params['other_params']['mode'] == 'debug':
        nparticles = 500000
    logger.info('Number of particles = %s or Galaxy Stellar Mass' % nparticles)
    # Calculate mass_cali_factor for flux calibration
    if params['other_params']['gal_mass'] != None:
        mass_cali_factor = params['other_params']['gal_mass'] / nparticles
        logger.info('Input Galaxy Stellar mass is %s, then the calibration factor is %s.' % (params['other_params']['gal_mass'], mass_cali_factor))
    else:
        logger.info('No flux calibration due to Galaxy Stellar mass is applied.')
        mass_cali_factor = 1
    logger.info('E-Galaxia model has been successfully loaded, time elapsed: %.2f s' % (clock() - t))




    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - -   A N A L Y S I S   M O D U L E S   - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



    # - - - - - - - - - - SPATIAL BINNING - - - - - - - - - -

    if params['other_params']['mode'] != 'continue':
        logger.info('Start binning the model...')
        t = clock()
        d_t_l, statistic_count_l, x_edges_l, y_edges_l = binner.spatial_binner(d_t, params['cube_params'], params['other_params'],
                                                                               ssp_model.age_grid, ssp_model.metal_grid, ssp_model.alpha_grid,
                                                                               filepath, logger, configDir + setup_cube_name,
                                                                               params['oparams']['distance'], nparticles)
        logger.info('Binning process has been finished, time elapsed: %.2f s' % (clock() - t))
    else:
        logger.info('Continue the running from previous run...')
        t = clock()
        d_t_l, statistic_count_l, x_edges_l, y_edges_l = binner.spatial_binner_continue(params['cube_params'], params['other_params'],
                                                                                        filepath, logger, configDir + setup_cube_name)
        logger.info('Binning process has been finished, time elapsed: %.2f s' % (clock() - t))
    d_t = None



    # - - - - - - - - - - START DATACUBE LOOP  - - - - - - - - - -

    for cube_idx in range(len(statistic_count_l)):
        d_t = d_t_l[cube_idx]
        statistic_count = statistic_count_l[cube_idx]
        x_edges = x_edges_l[cube_idx]
        y_edges = y_edges_l[cube_idx]



    # - - - - - - - - - - GENERATE DATACUBE  - - - - - - - - - -

        if params['other_params']['mode'] == 'continue' and os.path.exists(filepath + 'data_cube_' + str(cube_idx) + '.fits'):
            logger.info("Data Cube No.%s exists in %s, will not be run in 'continue' mode" % (cube_idx+1, filepath))
            continue
        else:
            logger.info('Start generating the datacube No.%s...' % (cube_idx+1))
            t = clock()
            Spec_DataCube = spec_generator.Spec_Generator(d_t, x_edges, y_edges, statistic_count, ssp_model,
                                                          params['cube_params'], params['ssp_params'], params['other_params'],
                                                          filepath, logger, cube_idx)
            Spec_DataCube()
            data_cube = Spec_DataCube.data_cube
            data_cube = data_cube * mass_cali_factor
            logger.info('Calibrated the flux based on the input Galaxy Stellar Mass=%s.' % params['other_params']['gal_mass'])
            # Free memory
            Spec_DataCube = None
            d_t = None
            statistic_count = None
            logger.info('Stellar spectra have been generated, time elapsed: %.2f s' % (clock() - t))


    # - - - - - WRITE DATACUBE  - - - - -

        cube_maker.write_cube(data_cube, params, x_edges, y_edges, ssp_model.new_wave, filepath, logger, cube_idx, ssp_model.velscale, __version__)
        logger.info('Data Cube No.%s has been written in %s, total time elapsed: %.2f s' % (cube_idx+1, filepath , clock() - t))




    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  F I N A L I S E   T H E   A N A L Y S I S  - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



    # - - - - - SAVE CONFIG  - - - - -

    logger.info('Write the json file into the outputs folder.')
    with open(filepath + 'setup.json', 'w') as f:
        json.dump(params, f, indent=2)
    logger.info('All the datacubes have been generated successfully, total time elapsed: %.2f s' % (clock() - t_init))
    logger.info("GalCraft completed successfully.")







# ============================================================================ #
#                           M A I N   F U N C T I O N                          #
# ============================================================================ #
def main(args=None):

    # Capture command-line arguments
    parser = optparse.OptionParser(usage="%GalCraft [options] arg")
    parser.add_option("--config",      dest="configName", type="string", \
            help="State the name of the MasterConfig file.")
    parser.add_option("--default-dir", dest="defaultDir", type="string", \
            help="File defining default directories for input, outputs, configuration files, and spectral templates.")
    (CommandOptions, args) = parser.parse_args()

    # Check if required command-line argument is given
    assert CommandOptions.configName != None, "Please specify the path of the Configuration.json file to be used. Exit!"

    # Run the framework
    run_GalCraft(CommandOptions)


if __name__ == '__main__':
    main()
