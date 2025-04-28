from ..ssp_utils import pegase_util as pegase_lib, \
    miles_util as miles_lib, \
    pegase_interp_util as pegase_interp_lib, \
    conroyinmiles_util as conroyinmiles_lib, \
    xshooter_util as xshooter_lib
from . import utils
import numpy as np
from scipy import ndimage
from multiprocessing import Pool



def process_Oversampling_templates(star, factor, i, j, k):
    starNew = ndimage.interpolation.zoom(star, factor, order=3)  # Oversampling
    return starNew, i, j, k


class model:

    def __init__(self, templateDir, instrumentDir, ssp_params, other_params, logger):

        self.logger = logger

        # setup some ssp_params
        self.ssp_name = ssp_params['model']
        self.imf = ssp_params['imf']
        self.slope = ssp_params['slope']
        self.isochrone = ssp_params['isochrone']
        self.single_alpha = ssp_params['single_alpha']
        self.factor = ssp_params['factor']
        self.FWHM_gal = ssp_params['FWHM_gal']
        self.FWHM_tem = ssp_params['FWHM_tem']
        self.dlam = ssp_params['dlam']
        self.age_range = ssp_params['age_range']
        self.metal_range = ssp_params['metal_range']
        self.wave_range = ssp_params['wave_range']
        self.interpolator_method = ssp_params['spec_interpolator']
        self.ncpu = other_params['ncpu']


        models = []

        if self.ssp_name=='miles':

            if self.single_alpha == False:
                models.append(miles_lib.miles(templateDir + 'miles/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.00/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          age_range=self.age_range, metal_range=self.metal_range))
                models.append(miles_lib.miles(templateDir + 'miles/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.40/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          age_range=self.age_range, metal_range=self.metal_range))
                alpha_grid = np.array([0, 0.4])

            elif self.single_alpha == 'alpha0':
                models.append(miles_lib.miles(templateDir + 'miles/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.00/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

            else:
                models.append(miles_lib.miles(templateDir + 'miles/' + 'MILES_' + self.isochrone + '_' + self.imf + '_baseFe/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        if self.ssp_name=='miles_interp':

            if self.single_alpha == False:
                models.append(miles_lib.miles(templateDir + 'miles_interp/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.00/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          age_range=self.age_range, metal_range=self.metal_range))
                models.append(miles_lib.miles(templateDir + 'miles_interp/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.40/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          age_range=self.age_range, metal_range=self.metal_range))
                alpha_grid = np.array([0, 0.4])

            elif self.single_alpha == 'alpha0':
                models.append(miles_lib.miles(templateDir + 'miles/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.00/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

            else:
                models.append(miles_lib.miles(templateDir + 'miles_interp/' + 'MILES_' + self.isochrone + '_' + self.imf + '_baseFe/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        if self.ssp_name=='miles_loginterp':

            if self.single_alpha == False:
                models.append(miles_lib.miles(templateDir + 'miles_loginterp/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.00/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          age_range=self.age_range, metal_range=self.metal_range))
                models.append(miles_lib.miles(templateDir + 'miles_loginterp/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.40/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          age_range=self.age_range, metal_range=self.metal_range))
                alpha_grid = np.array([0, 0.4])

            elif self.single_alpha == 'alpha0':
                models.append(miles_lib.miles(templateDir + 'miles/' + 'MILES_' + self.isochrone + '_' + self.imf + '_Ep0.00/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

            else:
                models.append(miles_lib.miles(templateDir + 'miles_loginterp/' + 'MILES_' + self.isochrone + '_' + self.imf + '_baseFe/M' + self.imf.lower() + '%.2f' % self.slope + '*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))


        elif self.ssp_name == 'pegasehr_interp':

            assert self.single_alpha == True, "No alpha-enhanced model avaliable for PEGASE-HR yet."
            models.append(pegase_interp_lib.pegase_interp(templateDir + 'pegasehr/' + 'PEGASEHR_' + self.isochrone + '_' + self.imf + '_baseFe/',
                                FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        elif self.ssp_name == 'pegasehr':

            assert self.single_alpha == True, "No alpha-enhanced model avaliable for PEGASE-HR yet."
            models.append(pegase_lib.pegase(templateDir + 'pegasehr/' + 'PEGASEHR_' + self.isochrone + '_' + self.imf + '_baseFe/',
                                            FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))


        elif self.ssp_name == 'conroyinmiles':

            if self.single_alpha == False:
                alpha_grid = np.load(templateDir + 'conroy/ssp_mist_v2.3_milesResUnit/' + 'miles/' + 'alpha_grid.npy')
                for alpha_i in range(len(alpha_grid)):
                    models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesResUnit/' + 'miles/' + 'alpha_%s/' % int(alpha_i),
                                                           FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))
            else:
                # Just load alpha/fe=0.0 dex templates
                models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesResUnit/' + 'miles/' + 'alpha_0/',
                                                       FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        elif self.ssp_name == 'conroyinmiles_moremetalalpha':

            if self.single_alpha == False:
                alpha_grid = np.load(templateDir + 'conroy/ssp_mist_v2.3_milesResUnit_moremetalalpha/' + 'alpha_grid.npy')
                for alpha_i in range(len(alpha_grid)):
                    models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesResUnit_moremetalalpha/' + 'alpha_%s/' % int(alpha_i),
                                                           FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))
            else:
                # Just load alpha/fe=0.0 dex templates
                models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesResUnit_moremetalalpha/' + 'alpha_0/',
                                                       FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        elif self.ssp_name == 'conroyinmilesCaT':

            if self.single_alpha == False:
                alpha_grid = np.load(templateDir + 'conroy/ssp_mist_v2.3_milesconroyCaTResUnit/' + 'alpha_grid.npy')
                for alpha_i in range(len(alpha_grid)):
                    models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesconroyCaTResUnit/' + 'alpha_%s/' % int(alpha_i),
                                                           FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))
            else:
                # Just load alpha/fe=0.0 dex templates
                models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesconroyCaTResUnit/' + 'alpha_0/',
                                                       FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        elif self.ssp_name == 'conroyinmilesCaT_moremetalalpha':

            if self.single_alpha == False:
                alpha_grid = np.load(templateDir + 'conroy/ssp_mist_v2.3_milesconroyCaTResUnit_moremetalalpha/' + 'alpha_grid.npy')
                for alpha_i in range(len(alpha_grid)):
                    models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesconroyCaTResUnit_moremetalalpha/' + 'alpha_%s/' % int(alpha_i),
                                                           FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))
            else:
                # Just load alpha/fe=0.0 dex templates
                models.append(conroyinmiles_lib.conroy(templateDir + 'conroy/ssp_mist_v2.3_milesconroyCaTResUnit_moremetalalpha/' + 'alpha_0/',
                                                       FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        elif self.ssp_name == 'xshooter':
            assert self.single_alpha == True, "No alpha-enhanced model avaliable for X-shooter yet."
            models.append(xshooter_lib.xshooter(templateDir + 'xshooter/' + 'XSHOOTER_' + self.isochrone + '_' + self.imf + '_baseFe/XSL_SSP_*.fits',
                                          FWHM_tem=self.FWHM_tem, age_range=self.age_range, metal_range=self.metal_range))

        templates = np.stack([model.templates for model in models], axis=3)

        # Define some constants from SSP parameters
        reg_dim = templates.shape[1:]
        c = 299792.458 # speed of light in km/s
        wave = models[0].lam_temp
        velscale = c * np.log(wave[1]/wave[0])
        age_grid = models[0].age_grid[:, 0]
        metal_grid = models[0].metal_grid[0, :]
        age_grid_2d = models[0].age_grid
        metal_grid_2d = models[0].metal_grid

        self.FWHM_tem = models[0].FWHM_tem


        # This is for calculating the bins in age, metallicity
        xgrid = np.log10(age_grid) + 9
        ygrid = metal_grid
        x = xgrid
        y = ygrid
        xb = (x[1:] + x[:-1])/2  # internal grid borders
        yb = (y[1:] + y[:-1])/2
        xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])  # 1st/last border
        yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])


        if self.single_alpha == False:
            zgrid = alpha_grid
            z = zgrid
            zb = (z[1:] + z[:-1])/2
            zb = np.hstack([1.5*z[0] - z[1]/2, zb, 1.5*z[-1] - z[-2]/2])
        else:
            # This will be used when generating the interpolator,
            # if single_alpha==False, it will be still fine to get a 3D data
            alpha_grid = np.array([0])
            zb = np.array([0])



        # Setup FWHM_gal, FWHM_tem file and calculate sig
        if type(self.FWHM_gal) == str:
            self.logger.info('Load the line-spread function %s from the folder ./instrument' % self.FWHM_gal)
            lsf_gal = np.genfromtxt(instrumentDir + self.FWHM_gal)
        else:
            lsf_gal = np.zeros([len(wave), 2])
            lsf_gal[:, 0] = wave
            lsf_gal[:, 1] = self.FWHM_gal
        if type(self.FWHM_tem) == str:
            self.logger.info('Load the line-spread function %s from the folder ./instrument' % self.FWHM_tem)
            lsf_tem = np.genfromtxt(instrumentDir + self.FWHM_tem)
        else:
            lsf_tem = np.zeros([len(wave), 2])
            lsf_tem[:, 0] = wave
            lsf_tem[:, 1] = self.FWHM_tem
        sig = utils.cal_degrade_sig(np.interp(wave, lsf_gal[:, 0], lsf_gal[:, 1], left=np.nan, right=np.nan),
                                    np.interp(wave, lsf_tem[:, 0], lsf_tem[:, 1], left=np.nan, right=np.nan),
                                    np.diff(wave)[0])

        if np.all(np.isnan(sig)) == True:
            raise ValueError("The outputs FWHM_gal is lower than the SSP FWHM_tem.")
        elif np.any(np.isnan(sig)) == True:
            logger.info("FWHM_gal is smaller than the SSP FWHM_tem in some wavelength pixels, will remove them and contiue the program...")
            mask_nansig = ~np.isnan(sig)
            wave = wave[mask_nansig]
            templates = templates[mask_nansig, :, :]
            sig = sig[mask_nansig]
        else:
            logger.info("FWHM_gal is larger than the SSP FWHM_tem for all wavelength pixels, continue the program...")


        self.templates = templates
        self.reg_dim = reg_dim
        self.wave = wave
        self.velscale = velscale
        self.age_grid = age_grid
        self.logage_grid = np.log10(age_grid) + 9
        self.metal_grid = metal_grid
        self.alpha_grid = alpha_grid
        self.age_grid_2d = age_grid_2d
        self.metal_grid_2d = metal_grid_2d
        self.xb = xb
        self.yb = yb
        self.zb = zb
        self.lsf_gal = lsf_gal
        self.lsf_tem = lsf_tem
        self.sig = sig

        # Add the new wave
        if self.dlam == None:
            self.new_wave = self.wave
        else:
            self.new_wave = np.arange(wave[0], wave[-1], self.dlam)


        logger.info('==================SSP info==================')
        logger.info('model name:            %s' % self.ssp_name)
        logger.info('IMF:                   %s' % self.imf)
        logger.info('isochrone:             %s' % self.isochrone)
        logger.info('slope:                 %s' % self.slope)
        logger.info('single_alpha:          %s' % self.single_alpha)
        logger.info('factor:                %s' % self.factor)
        logger.info('FWHM_gal:              %s' % self.FWHM_gal)
        logger.info('FWHM_tem:              %s' % self.FWHM_tem)
        logger.info('dlam:                  %s' % self.dlam)
        logger.info('velscale (km/s):       %.2f' % velscale)
        logger.info('age_range (Gyr):       %s' % [float('{:.3f}'.format(i)) for i in age_grid[[0, -1]]])
        logger.info('metal_range (dex):     %s' % [float('{:.3f}'.format(i)) for i in metal_grid[[0, -1]]])
        logger.info('alpha_range (dex):     %s' % [float('{:.3f}'.format(i)) for i in alpha_grid[[0, -1]]])
        logger.info('templates dim:         %s' % list(reg_dim))
        logger.info('n_templates:           %s' % np.prod(reg_dim))
        logger.info('SSP wave range:        %s' % [float('{:.3f}'.format(i)) for i in [self.wave[0], self.wave[-1]]])
        if self.wave_range==None:
            logger.info('Cube wave range:       %s' % [float('{:.3f}'.format(i)) for i in [self.new_wave[0], self.new_wave[-1]]])
        else:
            logger.info('Cube wave range:       %s' % [float('{:.3f}'.format(i)) for i in self.wave_range])
        logger.info('============================================')






    def oversample(self):
        # Oversample the templates and integral the spectra again
        wave_oversampled = ndimage.interpolation.zoom(self.wave, self.factor, order=1)
        templatesOversampled = np.zeros([self.templates.shape[0]*self.factor] + list(self.templates.shape[1:]))


        # Setup FWHM_gal file and calculate sig
        sig_oversampled = utils.cal_degrade_sig(np.interp(wave_oversampled, self.lsf_gal[:, 0], self.lsf_gal[:, 1]),
                                                np.interp(wave_oversampled, self.lsf_tem[:, 0], self.lsf_tem[:, 1]),
                                                np.diff(wave_oversampled)[0])


        pool = Pool(processes=self.ncpu)
        results = []
        for i in range(templatesOversampled.shape[1]):
            for j in range(templatesOversampled.shape[2]):
                for k in range(templatesOversampled.shape[3]):
                    results.append(pool.apply_async(process_Oversampling_templates, (self.templates[:, i, j, k], self.factor, i, j, k, )))
        pool.close()
        pool.join()

        for result in results:
            starNew, i, j, k = result.get()
            templatesOversampled[:, i, j, k] = starNew

        self.templatesOversampled = templatesOversampled
        self.wave_oversampled = wave_oversampled
        self.sig_oversampled = sig_oversampled

        self.templatesTransposed = np.transpose(self.templatesOversampled, (1, 2, 3, 0))
