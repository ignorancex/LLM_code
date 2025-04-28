import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from . import utils
from .spec_generator_pix import SpecGeneratorPix
import numpy as np
import collections
from multiprocessing import Pool
from matplotlib import pyplot as plt



SpecGeneratorPixInit = collections.namedtuple('SpecGeneratorPixInit', ['d_t', 'xb', 'yb', 'zb', 'logage_grid',
                                                                       'metal_grid', 'alpha_grid', 'x_edges', 'y_edges',
                                                                       'use_losvd', 'use_extinc', 'extinc_factor',
                                                                       'single_alpha', 'templatesOversampled_shape',
                                                                       'new_wave', 'wave_oversampled', 'sig_oversampled', 'factor', 'logger',
                                                                       'templatesTransposed', 'interpolator_method'])
SpecGeneratorPixInit.__new__.__defaults__ = (None,) * len(SpecGeneratorPixInit._fields)
SpecGeneratorPixCall = collections.namedtuple('SpecGeneratorPixCall', ['i', 'j'])

init_arguments = SpecGeneratorPixInit()




def init_worker(parameters, *args):

    global init_arguments

    assert not args
    assert isinstance(parameters, SpecGeneratorPixInit)

    init_arguments = parameters



def parallel_specgeneratorpix(pix_parameters):

    global init_arguments

    assert isinstance(pix_parameters, SpecGeneratorPixCall)

    specgeneratorpix = SpecGeneratorPix(d_t=init_arguments.d_t,
                                        xb=init_arguments.xb,
                                        yb=init_arguments.yb,
                                        zb=init_arguments.zb,
                                        logage_grid=init_arguments.logage_grid,
                                        metal_grid=init_arguments.metal_grid,
                                        alpha_grid=init_arguments.alpha_grid,
                                        x_edges=init_arguments.x_edges,
                                        y_edges=init_arguments.y_edges,
                                        use_losvd=init_arguments.use_losvd,
                                        use_extinc=init_arguments.use_extinc,
                                        extinc_factor=init_arguments.extinc_factor,
                                        single_alpha=init_arguments.single_alpha,
                                        templatesOversampled_shape=init_arguments.templatesOversampled_shape,
                                        new_wave=init_arguments.new_wave,
                                        wave_oversampled=init_arguments.wave_oversampled,
                                        sig_oversampled=init_arguments.sig_oversampled,
                                        factor=init_arguments.factor,
                                        logger=init_arguments.logger,
                                        templatesTransposed=init_arguments.templatesTransposed,
                                        interpolator_method=init_arguments.interpolator_method,
                                        )

    result = specgeneratorpix(i=pix_parameters.i,
                              j=pix_parameters.j)

    del specgeneratorpix
    return result





class Spec_Generator():

    def __init__(self, d_t, x_edges, y_edges, statistic_count, ssp_model, cube_params, ssp_params, other_params,
                 filepath, logger, cube_idx):

        self.d_t = d_t
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.statistic_count = statistic_count
        self.ssp_model = ssp_model
        self.cube_params = cube_params
        self.ssp_params = ssp_params
        self.other_params = other_params
        self.filepath = filepath
        self.logger = logger
        self.cube_idx = cube_idx


    def __call__(self):

        self.logger.info('Start the processes of generating spectra using {0} CPU(s).'.format(self.other_params['ncpu']))
        self.logger.info('Initializing the MDF bin and data cube array...')

        self.mass_fraction_pixel_bin = np.zeros(list(self.ssp_model.reg_dim) + [self.x_edges.shape[0] - 1, self.y_edges.shape[0] - 1])
        self.data_cube = np.zeros([self.x_edges.shape[0] - 1, self.y_edges.shape[0] - 1, self.ssp_model.new_wave.shape[0]])

        self.logger.info('Initializing the common variables to be used in each process...')
        initial_arguments = SpecGeneratorPixInit(d_t=self.d_t,
                                                 xb=self.ssp_model.xb,
                                                 yb=self.ssp_model.yb,
                                                 zb=self.ssp_model.zb,
                                                 logage_grid=self.ssp_model.logage_grid,
                                                 metal_grid=self.ssp_model.metal_grid,
                                                 alpha_grid=self.ssp_model.alpha_grid,
                                                 x_edges=self.x_edges,
                                                 y_edges=self.y_edges,
                                                 use_losvd=self.cube_params['use_losvd'],
                                                 use_extinc=self.cube_params['use_extinc'],
                                                 extinc_factor=self.cube_params['extinc_factor'],
                                                 single_alpha=self.ssp_params['single_alpha'],
                                                 templatesOversampled_shape=self.ssp_model.templatesOversampled.shape,
                                                 new_wave=self.ssp_model.new_wave,
                                                 wave_oversampled=self.ssp_model.wave_oversampled,
                                                 sig_oversampled=self.ssp_model.sig_oversampled,
                                                 factor=self.ssp_model.factor,
                                                 logger=self.logger,
                                                 templatesTransposed=self.ssp_model.templatesTransposed,
                                                 interpolator_method=self.ssp_params['spec_interpolator'])


        self.logger.info('Sort the spatial pixel in a descending order in terms of number of particles (to speed up the processing).')
        index_array = np.dstack(np.unravel_index(np.argsort(self.statistic_count.ravel())[::-1], self.statistic_count.shape))

        self.logger.info('Prepare the spatial pixel indexes for the loop.')
        index_i = self.x_edges.shape[0] - 2 - index_array[0, :, 0]
        index_j = index_array[0, :, 1]

        pool = Pool(processes=self.other_params['ncpu'], initializer=init_worker, initargs=[initial_arguments])

        self.logger.info('Start looping each spatial pixel to generate the spectra...')

        generator_input = []

        for i, j in zip(index_i, index_j):

                if self.statistic_count[self.x_edges.shape[0] - 2 - i, j]!=0:

                    generator_input.append(SpecGeneratorPixCall(i=i, j=j))

                if (np.mod(len(generator_input), self.other_params['nprocess_ploop'])==0 and len(generator_input)!=0) or (i==self.x_edges.shape[0]-2 and j==self.y_edges.shape[0]-2):

                    self.logger.info('Allocate processes to generate the spectra for {0} spatial pixels.'.format(len(generator_input)))

                    _results = pool.map_async(parallel_specgeneratorpix, generator_input)
                    results = _results.get()
                    for result in results:
                        self.mass_fraction_pixel_bin[:, :, :, result.i, result.j] = result.mass_fraction_pixel
                        self.data_cube[result.i, result.j, :] = result.galaxy_rebin

                    self.logger.info('Finish one CPU loop, initialize the input list for another loop.')
                    generator_input = []

        if len(generator_input) > 0:
            _results = pool.map_async(parallel_specgeneratorpix, generator_input)
            results = _results.get()
            for result in results:
                self.mass_fraction_pixel_bin[:, :, :, result.i, result.j] = result.mass_fraction_pixel
                self.data_cube[result.i, result.j, :] = result.galaxy_rebin
        self.logger.info('Finish the last loop, initialize the input list for another loop.')


        pool.close()
        pool.join()

        self.logger.info('All the spectra generating process has been finished.')

        self.mass_fraction_pixel_bin = np.transpose(self.mass_fraction_pixel_bin, (0, 1, 2, 4, 3))
        np.save(self.filepath + 'mass_fraction_array_' + str(self.cube_idx) + '.npy', self.mass_fraction_pixel_bin)
        self.logger.info('Save the mass fraction distribution (MFD) of each spatial bin in the folder.')


        self.logger.info('Plot the statictial distribution using MFD grids and save it in the folder.')
        utils.plot_parameter_maps(self.mass_fraction_pixel_bin, self.ssp_model.age_grid_2d, self.ssp_model.metal_grid_2d,
                                  self.ssp_model.alpha_grid, self.ssp_model.reg_dim, self.x_edges, self.y_edges,
                                  xlabel=self.cube_params['x_coord'], ylabel=self.cube_params['y_coord'], cmap=plt.cm.Spectral_r)
        plt.savefig(self.filepath + 'grids_distrib_' + str(self.cube_idx) + '.png', dpi=150)


