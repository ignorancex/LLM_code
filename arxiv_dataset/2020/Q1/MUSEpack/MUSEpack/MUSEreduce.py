#!/usr/bin/env python

__version__ = '1.1.3'

__revision__ = '20211102'

import sys
import shutil
import os
import glob
import string
import filecmp
import numpy as np
from astropy.io import ascii, fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime, timedelta
import time
import json


class musereduce:

    '''
    Kwargs:
        configfile : :obj:`str`
            A :obj:`json` configfile for musereduce, where all the parameters
            are set.

        debug : :obj:`bool`, (optional), default: :obj:`None`
            :obj:`True`: :class:`MUSEreduce.musereduce` runs in debug mode and
            ESORex will not be executed. All products must be available.
            It can be used for testing the creation of folder, and creating
            the ``.sof`` files

    '''


    def __init__(self, configfile, debug=False):

        if configfile == None:
            configfile = os.path.dirname(__file__) + "/config.json"
        with open(configfile, "r") as read_file:
            self.config = json.load(read_file)

        self.withrvcorr = self.config['global']['withrvcorr']
        self.OB_list = np.array(self.config['global']['OB_list'])
        self.dithername = self.config['global']['OB']
        self.dithering_multiple_OBs =\
            self.config['global']['dither_multiple_OBs']
        if not self.dithering_multiple_OBs:
            self.OB_list = np.array([self.dithername])
        self.rootpath = self.config['global']['rootpath']
        self.mode = self.config['global']['mode']
        self.auto_sort_data = self.config['global']['auto_sort_data']
        self.using_specific_exposure_time =\
        self.config['global']['using_specific_exposure_time']

        self.n_CPU = self.config['global']['n_CPU']

        self.using_ESO_calibration =\
        self.config['calibration']['using_ESO_calibration']
        self.dark = self.config['calibration']['dark']
        self.renew_statics = self.config['calibration']['renew_statics']

        self.skyreject = self.config['sci_basic']['skyreject']
        if not 'skylines' in self.config['sci_basic']:
            self.skylines = '5577.339,6300.304'
        else:
            self.skylines = self.config['sci_basic']['skylines']
        if not 'execute_std' in self.config['sci_basic']:
            self.reduce_std = True
        else:
            self.reduce_std = self.config['sci_basic']['execute_std']

        self.skyfield = self.config['sky']['sky_field']
        self.skyfraction = self.config['sky']['fraction']
        self.skyignore = self.config['sky']['ignore']
        self.skymethod = self.config['sky']['method']

        self.skysub = self.config['sci_post']['subtract_sky']
        self.autocalib = self.config['sci_post']['autocalib']
        self.raman = self.config['sci_post']['raman']
        if self.mode != 'NFM-AO':
            self.raman = False

        self.weight = self.config['exp_combine']['weight']

        self.user_list =\
            np.array(self.config['dither_collect']['user_list'], dtype=object)

        self.raw_data_dir = None
        self.working_dir = None
        self.combining_OBs_dir = None
        self.calibration_dir = None
        self.ESO_calibration_dir = None
        self.static_calibration_dir = None

        self.debug = debug

    def execute(self):

        '''
        This method executes wrapper and starts the data reduction process
        set in the :obj:`json` config file.

        '''

        startime = time.time()

        print(' ')
        print('##############################################################')
        print('#####                                                    #####')
        print('#####        MUSE data reduction pipeline wrapper        #####')
        print('#####   Must be used with ESORex and ESO MUSE pipeline   #####')
        print('#####      author: Peter Zeidler (zeidler@stsci.edu)     #####')
        print('#####                    Nov 02, 2021                    #####')
        print('#####                   Version: '+str(__version__)+'   \
                #####')
        print('#####                                                    #####')
        print('##############################################################')

        print('... Checking various necessary variables')
        assert sys.version_info > (3, 0), 'YOU ARE NOT USING PYTHON 3.x'
        assert self.config['global']['pipeline_path'],\
            'NO PIPELINE PATH DEFINED'
        assert self.config['global']['rootpath'], 'NO ROOTPATH DEFINED'
        assert self.config['global']['OB'], 'NO OBs given'
        if self.dithering_multiple_OBs and len(self.user_list) > 0:
            print('Currently a user list cannot be provided with multiple OBs')
            sys.exit()

        print('... Perfect, everything checks out')
        print('')

        print('... Settings for data reduction')
        if self.debug:
            print('##################################################')
            print('#####        RUNNING IN DEBUG MODE           #####')
            print('#####      PRODUCTS MUST BE ALL THERE        #####')
            print('#####          NO ESORex EXECUTION           #####')
            print('##################################################')

            print("")
        print('>>> Number of cores: ' + str(self.n_CPU) + ' cores')

        if self.withrvcorr:
            print('>>> bariocentric correction: on')

            if self.skyfield == 'auto':
                print('>>> Skyfield: "automatic" ')
            if self.skyfield == 'object':
                print('>>> Skyfield: "science exposure" ')
        else:
            print('>>> bariocentric correction: off')
            print('>>> sky subtraction: off')

        print('>>> Observation mode: ' + self.mode)

        if self.config['calibration']['execute'] == True:
            if self.using_ESO_calibration:
                print('>>> Using ESO calibration files')
            else:
                print('>>> Using self-processed calibration files')
                if self.dark:
                    print('>>> DARK will be reduced and used')
                if not self.dark:
                    print('>>> DARK will not be reduced and used')
        if self.dark:
            print('>>> DARK will be used: ' + str(self.dark))

        if self.raman:
            print('>>> Removal of Raman lines: ' + str(self.raman))

        if self.dithering_multiple_OBs:
            print('>>> Exposures per pointing are spread over multiple OBs')
            print('==> The pointing name is: ' + self.dithername)
            self.multOB_exp_counter = 0
        else:
            print('>>> All exposures are located in one OB')
            if len(self.user_list) > 0:
                print('>>> Dithering exposures input list:')
                for li in self.user_list:
                    print(li)
        if len(self.OB_list) > 1:
            print('>>> Reducing more than one OB: ', str(len(self.OB_list)))

        print(' ')
        print('... The following modules will be executed')
        if self.config['calibration']['execute']\
        and not self.using_ESO_calibration:
            print('>>> BIAS')
            if self.dark:
                print('>>> DARK')
            print('>>> FLAT')
            print('>>> WAVECAL')
            print('>>> LSF')
            print('>>> TWILIGHT')
        if self.config['sci_basic']['execute']:
            print('>>> SCIBASIC')
            if not self.reduce_std:
                print('    Caution SCIBASIC will not be executed on STD star!!')
        if self.config['std_flux']['execute']:
            print('>>> STANDARD')
        if self.config['sky']['execute']:
            print('>>> CREATE_SKY')
        if self.config['sci_post']['execute']:
            print('>>> SCI_POST')
        if self.config['exp_align']['execute']:
            print('>>> EXP_ALIGN')
        if self.config['exp_combine']['execute']:
            print('>>> EXP_COMBINE')
            print(' ')

        print('#####  All parameters set: Starting the data reduction   #####')

        for OB in self.OB_list:
            print(' ')
            print('... Creating directories')
            print('>>> for OB: ' + OB)
            print(' ')

            if not os.path.exists(self.rootpath + 'reduced/'):
                os.mkdir(self.rootpath + 'reduced/')

            if self.dithering_multiple_OBs:
                self.raw_data_dir = self.rootpath + 'raw/' +\
                self.dithername + '/' + OB + '/'
                self.working_dir = self.rootpath + 'reduced/'\
                + self.dithername + '/' + OB + '/'
                self.combining_OBs_dir = self.rootpath + 'reduced/'\
                + self.dithername + '/'
                if not os.path.exists(self.combining_OBs_dir):
                    os.mkdir(self.combining_OBs_dir)
            else:
                self.raw_data_dir = self.rootpath + 'raw/' + OB + '/'
                self.working_dir = self.rootpath + 'reduced/' + OB + '/'
                self.combining_OBs_dir = None

            self.calibration_dir = self.working_dir + 'calibrations/'
            self.ESO_calibration_dir = self.working_dir + 'ESO_calibrations/'
            self.static_calibration_dir = self.working_dir\
            + 'static_calibration_files/'

            if self.renew_statics and\
            os.path.exists(self.static_calibration_dir):
                shutil.rmtree(self.static_calibration_dir)
            if self.renew_statics and self.auto_sort_data and\
                    os.path.exists(self.ESO_calibration_dir):
                shutil.rmtree(self.ESO_calibration_dir)
            if not os.path.exists(self.working_dir):
                os.mkdir(self.working_dir)
            if not os.path.exists(self.working_dir + 'std/'):
                os.mkdir(self.working_dir + 'std/')
            if not os.path.exists(self.calibration_dir):
                os.mkdir(self.calibration_dir)
            if not os.path.exists(self.ESO_calibration_dir):
                os.mkdir(self.ESO_calibration_dir)
            if not os.path.exists(self.calibration_dir + 'DARK/')\
            and self.dark:
                os.mkdir(self.calibration_dir + 'DARK/')

            if not os.path.exists(self.calibration_dir + 'TWILIGHT/'):
                os.mkdir(self.calibration_dir + 'TWILIGHT/')
            if not os.path.exists(self.calibration_dir + 'SCIENCE/'):
                os.mkdir(self.calibration_dir + 'SCIENCE/')
            if os.path.exists(self.static_calibration_dir):
                shutil.rmtree(self.static_calibration_dir)
            os.mkdir(self.static_calibration_dir)
            for itername in glob.glob(os.path.join(
                    self.config['global']['pipeline_path'], 'calib/muse*/*.*')):
                shutil.copy(itername, self.static_calibration_dir + '.')

            print('... Sorting the data')

            if self.auto_sort_data:
                print('>>> Sorting the raw data')
                _sort_data(self)
            else:
                print('>>> MANUAL INTERACTION NEEDED')

            if not self.using_specific_exposure_time:
                exp_list_SCI =\
                np.concatenate([glob.glob(self.working_dir + '*_SCI.list'),\
                glob.glob(self.working_dir + '*_SKY.list')])
                exp_list_SCI = sorted(exp_list_SCI)

                exp_list_DAR =\
                sorted(glob.glob(self.working_dir + '*_DAR.list'))

                exp_list_TWI =\
                sorted(glob.glob(self.working_dir + '*_TWI.list'))

            if self.using_specific_exposure_time:

                exp_list_SCI =\
                sorted(np.concatenate([glob.glob(self.working_dir + '*'\
                + str('{:04d}'.format(self.using_specific_exposure_time)) +\
                '*_SCI.list'), glob.glob(self.working_dir + '*'\
                + str('{:04d}'.format(self.using_specific_exposure_time))\
                + '*_SKY.list')]))

                exp_list_DAR =\
                sorted(glob.glob(self.working_dir + '*'\
                + str('{:04d}'.format(self.using_specific_exposure_time))\
                + '*_DAR.list'))

                exp_list_TWI =\
                sorted(glob.glob(self.working_dir + '*'\
                + str('{:04d}'.format(self.using_specific_exposure_time))\
                + '*_TWI.list'))

            for exposure in exp_list_SCI:
                exposure_dir = exposure[:-9] + '/'
                if not os.path.exists(exposure_dir):
                    os.mkdir(exposure_dir)

            print(' ')
            print('... reducing OB: ' + OB)
            print(' ')

            ### CALIBRATION PRE-PROCESSING ###

            if self.config['calibration']['execute']:
                create_sof = self.config['calibration']['create_sof']

                if not self.using_ESO_calibration:
                    _bias(self, exp_list_SCI, exp_list_DAR,\
                    exp_list_TWI, create_sof)

                    if self.dark:
                        _dark(self, exp_list_SCI, exp_list_DAR, create_sof)
                    _flat(self, exp_list_SCI, exp_list_TWI, create_sof)
                    _wavecal(self, exp_list_SCI, exp_list_TWI, create_sof)
                    _lsf(self, exp_list_SCI, exp_list_TWI, create_sof)
                    _twilight(self, exp_list_SCI, exp_list_TWI, create_sof)

            ### OBSERVATION PRE-PROCESSING ###
            if self.config['sci_basic']['execute']:
                create_sof = self.config['sci_basic']['create_sof']
                _science_pre(self, exp_list_SCI, create_sof)

            ### OBSERVATION POST-PROCESSING ###
            if self.config['std_flux']['execute']:
                create_sof = self.config['std_flux']['create_sof']
                _std_flux(self, exp_list_SCI, create_sof)

            if self.config['sky']['execute']:
                create_sof = self.config['sky']['create_sof']

                if not self.config['sky']['modified']:
                    _sky(self, exp_list_SCI, create_sof)
                if self.config['sky']['modified']:
                    _modified_sky(self, exp_list_SCI, create_sof)

            ###s SCIENCE POST-PROCESSING ###
            if self.config['sci_post']['execute']:
                create_sof = self.config['sci_post']['create_sof']
                _scipost(self, exp_list_SCI, create_sof, OB)

            if self.config['dither_collect']['execute']:
                _dither_collect(self, exp_list_SCI, OB)

        if self.config['exp_align']['execute']:
            create_sof = self.config['exp_align']['create_sof']
            _exp_align(self, exp_list_SCI, create_sof, OB)

        if self.config['exp_combine']['execute']:
            create_sof = self.config['exp_combine']['create_sof']
            _exp_combine(self, exp_list_SCI, create_sof)

        endtime = time.time()
        print('>>> Total execution time: ',\
        timedelta(seconds=endtime - startime))


def _get_filelist(self, data_dir, filename_wildcard):

    '''
    This module collects the necessary file lists from folders.

    Args:
        data_dir : :obj:`str`
            The directory where the files are located.

        filename_wildcard : :obj:`str`
            The filenames which should be collected
    '''

    os.chdir(data_dir)
    raw_data_list = glob.glob(filename_wildcard)
    os.chdir(self.rootpath)
    return raw_data_list


def _call_esorex(self, exec_dir, esorex_cmd):

    '''
    This module calls the various ESOrex commands and gives it to the
    terminal for execution.

    Args:
        exec_dir : :obj:`str`
            The directory where ESOrex should be executed.

        esorex_cmd : :obj:`str`
            The ESOrex command that needs to be executed
    '''

    os.chdir(exec_dir)
    os.system('export OMP_NUM_THREADS=' + str(self.n_CPU))
    print('esorex ' + esorex_cmd)
    os.system('esorex ' + esorex_cmd)
    os.chdir(self.rootpath)


def _sort_data(self):

    '''
    This module reads the headers of all of the raw data and sorts it
    accordingly. It also assigns the correct calibration files to the exposures

    '''

    file_list = _get_filelist(self, self.raw_data_dir, '*.fits*')
    science_files = np.array([])
    calibration_files = np.array([])
    science_type = np.array([])
    calibration_type = np.array([])
    ESO_calibration_files = np.array([])
    ESO_calibration_type = np.array([])

    cal_categories = ['MASTER_BIAS', 'MASTER_DARK', 'MASTER_FLAT',\
    'TRACE_TABLE', 'WAVECAL_TABLE', 'LSF_PROFILE', 'TWILIGHT_CUBE',\
    'FILTER_LIST', 'EXTINCT_TABLE', 'STD_FLUX_TABLE', 'SKY_LINES',\
    'GEOMETRY_TABLE', 'ASTROMETRY_WCS', 'STD_RESPONSE', 'STD_TELLURIC',\
    'ASTROMETRY_REFERENCE']

    for files in file_list:

        hdu = fits.open(self.raw_data_dir + files)

        dprcatg_exist = hdu[0].header.get('HIERARCH ESO DPR CATG', False)
        procatg_exist = hdu[0].header.get('HIERARCH ESO PRO CATG', False)

        if dprcatg_exist:
            dprcatg = (hdu[0].header['HIERARCH ESO DPR CATG'])
            dprtype = (hdu[0].header['HIERARCH ESO DPR TYPE'])

            if dprcatg == 'SCIENCE':
                science_files = np.append(science_files, files)
                science_type = np.append(science_type, dprtype)

            if dprcatg == 'CALIB':
                calibration_files = np.append(calibration_files, files)
                if dprtype == 'FLAT,LAMP':
                    calibration_type = np.append(calibration_type, 'FLAT')
                elif dprtype == 'FLAT,SKY':
                    calibration_type = np.append(calibration_type, 'SKYFLAT')
                elif dprtype == 'WAVE':
                    calibration_type = np.append(calibration_type, 'ARC')
                elif dprtype == 'WAVE,MASK':
                    calibration_type = np.append(calibration_type, 'MASK')
                elif dprtype == 'FLAT,LAMP,ILLUM':
                    calibration_type = np.append(calibration_type, 'ILLUM')
                else:
                    calibration_type = np.append(calibration_type, dprtype)

        if procatg_exist:
            procatg = (hdu[0].header['HIERARCH ESO PRO CATG'])
            ESO_calibration_files = np.append(ESO_calibration_files, files)

            for cal_category in cal_categories:
                if procatg == cal_category:
                    ESO_calibration_type =\
                    np.append(ESO_calibration_type, cal_category)
                    if not os.path.isfile(self.ESO_calibration_dir\
                    + cal_category):
                        shutil.copy(self.raw_data_dir + files,\
                        self.ESO_calibration_dir + cal_category + '.fits')

    rot_angles = np.zeros(len(science_files))
    points = np.zeros(len(science_files), dtype=object)
    rot_angles_ident = np.zeros(len(science_files))

    for sci_file_idx in range(len(science_files)):

        hdu = fits.open(self.raw_data_dir + science_files[sci_file_idx])[0]
        RA = hdu.header['RA']
        DEC = hdu.header['DEC']
        EXPTIME = hdu.header['EXPTIME']

        points[sci_file_idx] = SkyCoord(ra=RA * u.degree, dec=DEC * u.degree,\
        frame='fk5').to_string('hmsdms', sep='',\
        precision=0).translate(str.maketrans('', '', string.whitespace))\
        + '_' + str(int(EXPTIME)).rjust(4, '0')

        rot_angles[sci_file_idx] = hdu.header['HIERARCH ESO INS DROT POSANG']

    for idx in range(len(rot_angles)):
        ident = 0
        for idx2 in range(len(rot_angles)):
            if rot_angles[idx] == rot_angles[idx2]\
            and points[idx] == points[idx2]:
                rot_angles_ident[idx2] = ident
                ident += 1

    for sci_file_idx in range(len(science_files)):

        hdu = fits.open(self.raw_data_dir + science_files[sci_file_idx])[0]
        RA = hdu.header['RA']
        DEC = hdu.header['DEC']
        EXPTIME = hdu.header['EXPTIME']
        ROT = hdu.header['HIERARCH ESO INS DROT POSANG']
        DATE = hdu.header['MJD-OBS']

        c = SkyCoord(ra=RA * u.degree, dec=DEC * u.degree,\
        frame='fk5').to_string('hmsdms', sep='',\
        precision=0).translate(str.maketrans('', '', string.whitespace))

        filelist_science = c + '_' + str(int(EXPTIME)).rjust(4, '0') + '_'\
        + str(int(ROT)).rjust(3, '0') + '_'\
        + str(int(rot_angles_ident[sci_file_idx])).rjust(2, '0')\
        + '_SCI.list'

        filelist_sky = c + '_' + str(int(EXPTIME)).rjust(4, '0') + '_'\
        + str(int(ROT)).rjust(3, '0') + '_'\
        + str(int(rot_angles_ident[sci_file_idx])).rjust(2, '0') + '_SKY.list'

        filelist_dark = c + '_' + str(int(EXPTIME)).rjust(4, '0') + '_'\
        + str(int(ROT)).rjust(3, '0') + '_'\
        + str(int(rot_angles_ident[sci_file_idx])).rjust(2, '0') + '_DAR.list'

        filelist_twilight = c + '_' + str(int(EXPTIME)).rjust(4, '0') + '_'\
        + str(int(ROT)).rjust(3, '0') + '_'\
        + str(int(rot_angles_ident[sci_file_idx])).rjust(2, '0') + '_TWI.list'

        dark_date = np.array([])
        twilight_date = np.array([])

        for calibfiles in range(len(calibration_files)):
            if calibration_type[calibfiles] == 'DARK':
                hdu_temp = fits.open(self.raw_data_dir\
                + calibration_files[calibfiles])[0]

                temp_date = hdu_temp.header['MJD-OBS']
                dark_date = np.append(dark_date, temp_date)

            if calibration_type[calibfiles] == 'SKYFLAT':
                hdu_temp = fits.open(self.raw_data_dir\
                + calibration_files[calibfiles])[0]

                temp_date = hdu_temp.header['MJD-OBS']
                twilight_date = np.append(twilight_date, temp_date)

        dark_date = np.unique(dark_date)
        twilight_date = np.unique(twilight_date)

        if science_type[sci_file_idx] == 'OBJECT':
            f_science = open(self.working_dir + filelist_science, 'w')

        if science_type[sci_file_idx] == 'SKY':
            f_science = open(self.working_dir + filelist_sky, 'w')

        f_dark = open(self.working_dir + filelist_dark, 'w')
        f_twilight = open(self.working_dir + filelist_twilight, 'w')
        f_science.write(self.raw_data_dir + science_files[sci_file_idx]\
        + '  ' + science_type[sci_file_idx] + '\n')

        for calibfiles in range(len(calibration_files)):
            temp_date = fits.open(self.raw_data_dir\
            + calibration_files[calibfiles])[0].header['MJD-OBS']

            if abs(temp_date - DATE) <= 1.:
                f_science.write(self.raw_data_dir\
                + calibration_files[calibfiles] + '  ' +\
                calibration_type[calibfiles] + '\n')

            if calibration_type[calibfiles] == 'STD'\
            and abs(temp_date - DATE) > 1.:

                f_science.write(self.raw_data_dir\
                + calibration_files[calibfiles]\
                + '  ' + calibration_type[calibfiles] + '\n')

                if calibration_type[calibfiles] == 'ILLUM' and abs(temp_date\
                - DATE) > 1.:
                    f_science.write(self.raw_data_dir\
                    + calibration_files[calibfiles] + '  '\
                    + calibration_type[calibfiles] + '\n')

                print('WARNING: STD and SCI obs more than 24 hours apart')

            if (abs(temp_date - dark_date) <= 1.).all():
                    f_dark.write(self.raw_data_dir\
                    + calibration_files[calibfiles]\
                    + '  ' + calibration_type[calibfiles] + '\n')

            if (abs(temp_date - twilight_date) <= 1.).all():
                    f_twilight.write(self.raw_data_dir\
                    + calibration_files[calibfiles]\
                    + '  ' + calibration_type[calibfiles] + '\n')

        f_science.close()
        f_dark.close()
        f_twilight.close()


def _bias(self, exp_list_SCI, exp_list_DAR, exp_list_TWI, create_sof):

    '''
    This module calls *ESORex'* ``muse_bias``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        exp_list_DAR : :obj:`list`:
            The list of all associated dark files including their category read
            from the fits header

        exp_list_TWI : :obj:`list`:
            The list of all associated twilight files including their category
            read from the fits header 

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Creating the MASTER BIAS')

    esorex_cmd = '--log-file=bias.log --log-level=debug'\
    + ' muse_bias'\
    + ' --nifu=-1'\
    + ' --merge'\
    + ' bias.sof'

    if create_sof:

        if os.path.exists(self.calibration_dir + 'SCIENCE/bias.sof'):
            os.remove(self.calibration_dir + 'SCIENCE/bias.sof')
        if self.dark and os.path.exists(self.calibration_dir\
        + 'DARK/bias.sof'):
            os.remove(self.calibration_dir + 'DARK/bias.sof')
        if os.path.exists(self.calibration_dir + 'TWILIGHT/bias.sof'):
            os.remove(self.calibration_dir + 'TWILIGHT/bias.sof')

        for exposure_ID in range(len(exp_list_SCI)):

            print('>>> processing exposure: ' + str(exposure_ID + 1) + '/'\
                + str(len(exp_list_SCI)))
            print('>>> processing: ' + exp_list_SCI[exposure_ID])
            print(' ')

            raw_data_list = ascii.read(exp_list_SCI[exposure_ID],\
                format='no_header')
            if self.dark:
                raw_data_list_DARK = ascii.read(exp_list_DAR[exposure_ID],\
                format='no_header')
            raw_data_list_TWILIGHT = ascii.read(exp_list_TWI[exposure_ID],\
                format='no_header')

            f_science = open(self.calibration_dir +\
            'SCIENCE/bias_temp.sof', 'w')
            for i in range(len(raw_data_list[1][:])):
                if raw_data_list[i][1] == 'BIAS':
                    f_science.write(raw_data_list[i][0] + '  '\
                    + raw_data_list[i][1] + '\n')
            f_science.close()

            if self.dark:
                f_dark = open(self.calibration_dir
                                + 'DARK/bias_temp.sof', 'w')
                for i in range(len(raw_data_list_DARK[1][:])):
                    if raw_data_list_DARK[i][1] == 'BIAS':
                        f_dark.write(raw_data_list_DARK[i][0] + '  '\
                        + raw_data_list_DARK[i][1] + '\n')
                f_dark.close()

            f_twilight = open(self.calibration_dir\
                + 'TWILIGHT/bias_temp.sof', 'w')
            for i in range(len(raw_data_list_TWILIGHT[1][:])):
                if raw_data_list_TWILIGHT[i][1] == 'BIAS':
                    f_twilight.write(raw_data_list_TWILIGHT[i][0] + '  '\
                    + raw_data_list_TWILIGHT[i][1] + '\n')
            f_twilight.close()

            if os.path.isfile(self.calibration_dir + 'SCIENCE/bias.sof'):
                assert filecmp.cmp(self.calibration_dir + 'SCIENCE/bias.sof',\
                self.calibration_dir + 'SCIENCE/bias_temp.sof'),\
                'CAUTION CALIBRATION FILES ARE DIFFERENT: PLEASE CHECK'

                os.remove(self.calibration_dir + 'SCIENCE/bias_temp.sof')

            else:
                os.rename(self.calibration_dir + 'SCIENCE/bias_temp.sof',\
                self.calibration_dir + 'SCIENCE/bias.sof')

                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'SCIENCE/',\
                    esorex_cmd)

            if os.path.isfile(self.calibration_dir + 'TWILIGHT/bias.sof'):
                assert filecmp.cmp(self.calibration_dir + 'TWILIGHT/bias.sof',\
                self.calibration_dir + 'TWILIGHT/bias_temp.sof'),\
                'CAUTION TWILIGHT FILES ARE DIFFERENT: PLEASE CHECK'

                os.remove(self.calibration_dir + 'TWILIGHT/bias_temp.sof')

            else:
                os.rename(self.calibration_dir + 'TWILIGHT/bias_temp.sof',\
                self.calibration_dir + 'TWILIGHT/bias.sof')

                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'TWILIGHT/',\
                    esorex_cmd)

            if self.dark:
                if os.path.isfile(self.calibration_dir + 'DARK/bias.sof'):
                    assert filecmp.cmp(self.calibration_dir + 'DARK/bias.sof',\
                    self.calibration_dir + 'DARK/bias_temp.sof'),\
                    'CAUTION DARK FILES ARE DIFFERENT: PLEASE CHECK'

                    os.remove(self.calibration_dir + 'DARK/bias_temp.sof')

                else:
                    os.rename(self.calibration_dir + 'DARK/bias_temp.sof',\
                    self.calibration_dir + 'DARK/bias.sof')

                    if not self.debug:
                        _call_esorex(self, self.calibration_dir + 'DARK/',
                        esorex_cmd)

    if not create_sof:
        if not self.debug:
            _call_esorex(self, self.calibration_dir + 'SCIENCE/', esorex_cmd)
            _call_esorex(self, self.calibration_dir + 'TWILIGHT/', esorex_cmd)
        if self.dark:
            if not self.debug:
                _call_esorex(self, self.calibration_dir + 'DARK/', esorex_cmd)


def _dark(self, exp_list_SCI, exp_list_DAR, create_sof):

    '''
    This module calls *ESORex'* ``muse_dark``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        exp_list_DAR : :obj:`list`:
            The list of all associated dark files including their category read
            from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Creating the MASTER DARK')

    esorex_cmd = '--log-file=dark.log --log-level=debug'\
    + ' muse_dark'\
    + ' --nifu=-1'\
    + ' --merge'\
    + ' dark.sof'

    if create_sof:

        if os.path.exists(self.calibration_dir + 'DARK/dark.sof'):
            os.remove(self.calibration_dir + 'DARK/dark.sof')

        for exposure_ID in range(len(exp_list_SCI)):
            print('>>> processing exposure: ' + str(exposure_ID + 1) + '/'\
            + str(len(exp_list_SCI)))
            print('>>> processing: ' + exp_list_SCI[exposure_ID])
            print(' ')

            raw_data_list = ascii.read(exp_list_DAR[exposure_ID],\
            format='no_header')

            f = open(calibration_dir + 'DARK/dark_temp.sof', 'w')

            for i in range(len(raw_data_list[1][:])):
                if raw_data_list[i][1] == 'DARK':
                    f.write(self.raw_data_list[i][0] + '  '\
                    + raw_data_list[i][1] + '\n')
            f.write(self.calibration_dir + 'DARK/MASTER_BIAS.fits \
            MASTER_BIAS\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'DARK/dark.sof'):
                assert filecmp.cmp(self.calibration_dir + 'DARK/dark.sof',\
                self.calibration_dir + 'DARK/dark_temp.sof'),\
                'CAUTION DARK FILES ARE DIFFERENT: PLEASE CHECK'

                os.remove(calibration_dir + 'DARK/dark_temp.sof')

            else:
                os.rename(self.calibration_dir + 'DARK/dark_temp.sof',\
                self.calibration_dir + 'DARK/dark.sof')

                if not self.debug:
                    _call_esorex(self.calibration_dir + 'DARK/', esorex_cmd)

    if not create_sof:
        if not self.debug:
            _call_esorex(self.calibration_dir + 'DARK/', esorex_cmd)


def _flat(self, exp_list_SCI, exp_list_TWI, create_sof):

    '''
    This module calls *ESORex'* ``muse_flat``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        exp_list_TWI : :obj:`list`:
            The list of all associated twilight files including their category
            read from the fits header 

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Creating the MASTER FLAT')

    esorex_cmd = '--log-file=flat.log --log-level=debug'\
    + ' muse_flat'\
    + ' --samples=true'\
    + ' --nifu=-1'\
    + ' --merge'\
    + ' flat.sof'

    if create_sof:

        if os.path.exists(self.calibration_dir + 'SCIENCE/flat.sof'):
            os.remove(self.calibration_dir + 'SCIENCE/flat.sof')
        if os.path.exists(self.calibration_dir + 'TWILIGHT/flat.sof'):
            os.remove(self.calibration_dir + 'TWILIGHT/flat.sof')

        for exposure_ID in range(len(exp_list_SCI)):
            print('>>> processing exposure: ' + str(exposure_ID + 1)\
            + '/' + str(len(exp_list_SCI)))
            print('>>> processing: ' + exp_list_SCI[exposure_ID])
            print(' ')

            raw_data_list = ascii.read(exp_list_SCI[exposure_ID],\
            format='no_header')
            raw_data_list_TWILIGHT = ascii.read(exp_list_TWI[exposure_ID],\
            format='no_header')

            f = open(self.calibration_dir + 'SCIENCE/flat_temp.sof', 'w')
            for i in range(len(raw_data_list[1][:])):
                if raw_data_list[i][1] == 'FLAT':
                    f.write(raw_data_list[i][0]\
                    + '  ' + raw_data_list[i][1] + '\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/MASTER_BIAS.fits MASTER_BIAS\n')
            if self.dark:
                f.write(self.calibration_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'SCIENCE/flat.sof'):
                assert filecmp.cmp(self.calibration_dir + 'SCIENCE/flat.sof',\
                self.calibration_dir + 'SCIENCE/flat_temp.sof'),\
                'CAUTION SCIENCE FILES ARE DIFFERENT: PLEASE CHECK'

                os.remove(self.calibration_dir + 'SCIENCE/flat_temp.sof')

            else:
                os.rename(self.calibration_dir + 'SCIENCE/flat_temp.sof',\
                self.calibration_dir + 'SCIENCE/flat.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'SCIENCE/',\
                    esorex_cmd)

            f = open(self.calibration_dir + 'TWILIGHT/flat_temp.sof', 'w')
            for i in range(len(raw_data_list_TWILIGHT[1][:])):
                if raw_data_list_TWILIGHT[i][1] == 'FLAT':
                    f.write(raw_data_list_TWILIGHT[i][0]\
                    + '  ' + raw_data_list_TWILIGHT[i][1] + '\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/MASTER_BIAS.fits MASTER_BIAS\n')
            if self.dark:
                f.write(self.calibration_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'TWILIGHT/flat.sof'):
                assert filecmp.cmp(self.calibration_dir + 'TWILIGHT/flat.sof',\
                self.calibration_dir + 'TWILIGHT/flat_temp.sof'),\
                'CAUTION TWILIGHT FILES ARE DIFFERENT: PLEASE CHECK'
                os.remove(self.calibration_dir + 'TWILIGHT/flat_temp.sof')

            else:
                os.rename(self.calibration_dir + 'TWILIGHT/flat_temp.sof',\
                self.calibration_dir + 'TWILIGHT/flat.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'TWILIGHT/',\
                    esorex_cmd)

    if not create_sof:
        if not self.debug:
            _call_esorex(self, self.calibration_dir + 'SCIENCE/', esorex_cmd)
            _call_esorex(self, self.calibration_dir + 'TWILIGHT/', esorex_cmd)


def _wavecal(self, exp_list_SCI, exp_list_TWI, create_sof):

    '''
    This module calls *ESORex'* ``muse_wavecal``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        exp_list_TWI : :obj:`list`:
            The list of all associated twilight files including their category
            read from the fits header 

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Creating the WAVELENGTH CALIBRATION')

    esorex_cmd = '--log-file=wavecal.log --log-level=debug'\
    + ' muse_wavecal'\
    + ' --nifu=-1'\
    + ' --residuals'\
    + ' --merge'\
    + ' wavecal.sof'

    if create_sof:

        if os.path.exists(self.calibration_dir + 'SCIENCE/wavecal.sof'):
            os.remove(self.calibration_dir + 'SCIENCE/wavecal.sof')
        if os.path.exists(self.calibration_dir + 'TWILIGHT/wavecal.sof'):
            os.remove(self.calibration_dir + 'TWILIGHT/wavecal.sof')

        for exposure_ID in range(len(exp_list_SCI)):
            print('>>> processing exposure: ' + str(exposure_ID + 1)\
            + '/' + str(len(exp_list_SCI)))
            print('>>> processing: ' + exp_list_SCI[exposure_ID])
            print(' ')

            raw_data_list = ascii.read(exp_list_SCI[exposure_ID],\
            format='no_header')
            raw_data_list_TWILIGHT = ascii.read(exp_list_TWI[exposure_ID],\
            format='no_header')

            f = open(self.calibration_dir + 'SCIENCE/wavecal_temp.sof', 'w')
            for i in range(len(raw_data_list[1][:])):
                if raw_data_list[i][1] == 'ARC':
                    f.write(raw_data_list[i][0]\
                    + '  ' + raw_data_list[i][1] + '\n')
            f.write(self.static_calibration_dir\
            + 'line_catalog.fits LINE_CATALOG\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/MASTER_BIAS.fits MASTER_BIAS\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/TRACE_TABLE.fits TRACE_TABLE\n')
            if self.dark:
                f.write(self.calibration_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'SCIENCE/wavecal.sof'):
                assert filecmp.cmp(self.calibration_dir\
                + 'SCIENCE/wavecal.sof', self.calibration_dir\
                + 'SCIENCE/wavecal_temp.sof'),\
                'CAUTION SCIENCE FILES ARE DIFFERENT: PLEASE CHECK'
                os.remove(self.calibration_dir + 'SCIENCE/wavecal_temp.sof')

            else:
                os.rename(self.calibration_dir + 'SCIENCE/wavecal_temp.sof',\
                self.calibration_dir + 'SCIENCE/wavecal.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'SCIENCE/',\
                    esorex_cmd)

            f = open(self.calibration_dir + 'TWILIGHT/wavecal_temp.sof', 'w')
            for i in range(len(raw_data_list_TWILIGHT[1][:])):
                if raw_data_list_TWILIGHT[i][1] == 'ARC':
                    f.write(raw_data_list_TWILIGHT[i][0]\
                    + '  ' + raw_data_list_TWILIGHT[i][1] + '\n')
            f.write(self.static_calibration_dir\
            + 'line_catalog.fits LINE_CATALOG\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/MASTER_BIAS.fits MASTER_BIAS\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/TRACE_TABLE.fits TRACE_TABLE\n')
            if self.dark:
                f.write(self.calibration_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'TWILIGHT/wavecal.sof'):
                assert filecmp.cmp(self.calibration_dir\
                + 'TWILIGHT/wavecal.sof', self.calibration_dir\
                + 'TWILIGHT/wavecal_temp.sof'),\
                'CAUTION TWILIGHT FILES ARE DIFFERENT: PLEASE CHECK'
                os.remove(self.calibration_dir + 'TWILIGHT/wavecal_temp.sof')


            else:
                os.rename(self.calibration_dir + 'TWILIGHT/wavecal_temp.sof',\
                self.calibration_dir + 'TWILIGHT/wavecal.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'TWILIGHT/',\
                    esorex_cmd)

    if not create_sof:
        if not self.debug:
            _call_esorex(self, self.calibration_dir + 'SCIENCE/', esorex_cmd)
            _call_esorex(self, self.calibration_dir + 'TWILIGHT/', esorex_cmd)


def _lsf(self, exp_list_SCI, exp_list_TWI, create_sof):

    '''
    This module calls *ESORex'* ``muse_lsf``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        exp_list_TWI : :obj:`list`:
            The list of all associated twilight files including their category
            read from the fits header 

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Creating the LINE SPREAD FUNCTION')

    esorex_cmd = '--log-file=lsf.log --log-level=debug'\
    + ' muse_lsf'\
    + ' --nifu=-1'\
    + ' --merge'\
    + ' --save_subtracted'\
    + ' lsf.sof'

    if create_sof:

        if os.path.exists(self.calibration_dir + 'SCIENCE/lsf.sof'):
            os.remove(self.calibration_dir + 'SCIENCE/lsf.sof')
        if os.path.exists(self.calibration_dir + 'TWILIGHT/lsf.sof'):
            os.remove(self.calibration_dir + 'TWILIGHT/lsf.sof')

        for exposure_ID in range(len(exp_list_SCI)):
            print('>>> processing exposure: ' + str(exposure_ID + 1)\
            + '/' + str(len(exp_list_SCI)))
            print('>>> processing: ' + exp_list_SCI[exposure_ID])
            print(' ')

            raw_data_list = ascii.read(exp_list_SCI[exposure_ID],\
            format='no_header')
            raw_data_list_TWILIGHT = ascii.read(exp_list_TWI[exposure_ID],\
            format='no_header')

            f = open(self.calibration_dir + 'SCIENCE/lsf_temp.sof', 'w')
            for i in range(len(raw_data_list[1][:])):
                if raw_data_list[i][1] == 'ARC':
                    f.write(raw_data_list[i][0]\
                    + '  ' + raw_data_list[i][1] + '\n')
            f.write(self.static_calibration_dir\
            + 'line_catalog.fits LINE_CATALOG\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/MASTER_BIAS.fits MASTER_BIAS\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/TRACE_TABLE.fits TRACE_TABLE\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/WAVECAL_TABLE.fits WAVECAL_TABLE\n')
            if self.dark:
                f.write(self.exposure_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.write(self.calibration_dir\
            + 'SCIENCE/MASTER_FLAT.fits MASTER_FLAT\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'SCIENCE/lsf.sof'):
                assert filecmp.cmp(self.calibration_dir + 'SCIENCE/lsf.sof',\
                self.calibration_dir + 'SCIENCE/lsf_temp.sof'),\
                'CAUTION SCIENCE FILES ARE DIFFERENT: PLEASE CHECK'
                os.remove(self.calibration_dir + 'SCIENCE/lsf_temp.sof')

            else:
                os.rename(self.calibration_dir + 'SCIENCE/lsf_temp.sof',\
                self.calibration_dir + 'SCIENCE/lsf.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'SCIENCE/',\
                    esorex_cmd)

            f = open(self.calibration_dir + 'TWILIGHT/lsf_temp.sof', 'w')
            for i in range(len(raw_data_list_TWILIGHT[1][:])):
                if raw_data_list_TWILIGHT[i][1] == 'ARC':
                    f.write(raw_data_list_TWILIGHT[i][0]\
                    + '  ' + raw_data_list_TWILIGHT[i][1] + '\n')
            f.write(self.static_calibration_dir\
            + 'line_catalog.fits LINE_CATALOG\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/MASTER_BIAS.fits MASTER_BIAS\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/TRACE_TABLE.fits TRACE_TABLE\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/WAVECAL_TABLE.fits WAVECAL_TABLE\n')
            if self.dark:
                f.write(self.exposure_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/MASTER_FLAT.fits MASTER_FLAT\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'TWILIGHT/lsf.sof'):
                assert filecmp.cmp(self.calibration_dir + 'TWILIGHT/lsf.sof',\
                self.calibration_dir + 'TWILIGHT/lsf_temp.sof'),\
                'CAUTION TWILIGHT FILES ARE DIFFERENT: PLEASE CHECK'
                os.remove(self.calibration_dir + 'TWILIGHT/lsf_temp.sof')

            else:
                os.rename(self.calibration_dir + 'TWILIGHT/lsf_temp.sof',\
                self.calibration_dir + 'TWILIGHT/lsf.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'TWILIGHT/',\
                    esorex_cmd)

    if not create_sof:
        if not self.debug:
            _call_esorex(self, self.calibration_dir + 'SCIENCE/', esorex_cmd)
            _call_esorex(self, self.calibration_dir + 'TWILIGHT/', esorex_cmd)


def _twilight(self, exp_list_SCI, exp_list_TWI, create_sof):

    '''
    This module calls *ESORex'* ``muse_twilight``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        exp_list_TWI : :obj:`list`:
            The list of all associated twilight files including their category
            read from the fits header 

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Creating the TWILIGHT FLAT')

    esorex_cmd = '--log-file=twilight.log --log-level=debug'\
    + ' muse_twilight'\
    + ' twilight.sof'

    if create_sof:

        if os.path.exists(self.calibration_dir + 'TWILIGHT/twilight.sof'):
            os.remove(self.calibration_dir + 'TWILIGHT/twilight.sof')

        for exposure_ID in range(len(exp_list_SCI)):
            print('>>> processing exposure: ' + str(exposure_ID + 1)\
            + '/' + str(len(exp_list_SCI)))
            print('>>> processing: ' + exp_list_SCI[exposure_ID])
            print(' ')

            raw_data_list = ascii.read(exp_list_SCI[exposure_ID],\
            format='no_header')
            raw_data_list_TWILIGHT = ascii.read(exp_list_TWI[exposure_ID],\
            format='no_header')

            MJDsillum = np.array([])
            MJDsskyflat = np.array([])
            illum_index = np.array([])

            for i in range(len(raw_data_list_TWILIGHT[1][:])):
                if raw_data_list_TWILIGHT[i][1] == 'SKYFLAT':
                    skyflathdu = fits.open(raw_data_list_TWILIGHT[i][0])
                    MJDskyflat = skyflathdu[0].header['MJD-OBS']
                    MJDsskyflat = np.append(MJDsskyflat, MJDskyflat)

                if raw_data_list_TWILIGHT[i][1] == 'ILLUM':
                    illumhdu = fits.open(raw_data_list_TWILIGHT[i][0])
                    MJDillum = illumhdu[0].header['MJD-OBS']
                    MJDsillum = np.append(MJDsillum, MJDillum)
                    illum_index = np.append(illum_index, i)

            choosen_illum = int(illum_index[np.argmin(np.abs(MJDsillum\
            - np.min(MJDskyflat)))])

            f = open(self.calibration_dir + 'TWILIGHT/twilight_temp.sof', 'w')
            for i in range(len(raw_data_list_TWILIGHT[1][:])):
                if raw_data_list_TWILIGHT[i][1] == 'SKYFLAT':
                    f.write(raw_data_list_TWILIGHT[i][0]\
                    + '  ' + raw_data_list_TWILIGHT[i][1] + '\n')
            f.write(raw_data_list_TWILIGHT[choosen_illum][0]\
            + '  ' + raw_data_list_TWILIGHT[choosen_illum][1] + '\n')
            if self.mode == 'WFM-AO' or self.mode == 'WFM-NOAO':
                f.write(self.static_calibration_dir\
                + 'geometry_table_wfm.fits GEOMETRY_TABLE\n')
            if self.mode == 'NFM-AO':
                f.write(self.static_calibration_dir\
                + 'geometry_table_wfm.fits GEOMETRY_TABLE\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/MASTER_BIAS.fits MASTER_BIAS\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/MASTER_FLAT.fits MASTER_FLAT\n')
            if self.dark:
                f.write(self.exposure_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/TRACE_TABLE.fits TRACE_TABLE\n')
            f.write(self.calibration_dir\
            + 'TWILIGHT/WAVECAL_TABLE.fits WAVECAL_TABLE\n')
            if MJDsskyflat.all() < 57823.5:
                f.write(self.static_calibration_dir\
                + 'vignetting_mask.fits VIGNETTING_MASK\n')
            f.close()

            if os.path.isfile(self.calibration_dir + 'TWILIGHT/twilight.sof'):
                assert filecmp.cmp(self.calibration_dir\
                + 'TWILIGHT/twilight.sof',\
                self.calibration_dir + 'TWILIGHT/twilight_temp.sof'),\
                'CAUTION TWILIGHT FILES ARE DIFFERENT: PLEASE CHECK'
                os.remove(self.calibration_dir + 'TWILIGHT/twilight_temp.sof')

            else:
                os.rename(self.calibration_dir + 'TWILIGHT/twilight_temp.sof',\
                self.calibration_dir + 'TWILIGHT/twilight.sof')
                if not self.debug:
                    _call_esorex(self, self.calibration_dir + 'TWILIGHT',\
                    esorex_cmd)

    if not create_sof:
        if not self.debug:
            _call_esorex(self, self.calibration_dir + 'TWILIGHT', esorex_cmd)


def _science_pre(self, exp_list_SCI, create_sof):

    '''
    This module calls *ESORex'* ``muse_scibasic``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... Science PREPROCESSING')

    esorex_cmd = '--log-file=sci_basic_object.log --log-level=debug'\
    + ' muse_scibasic'\
    + ' --nifu=-1'\
    + ' --resample'\
    + ' --saveimage=true'\
    + ' --skyreject=' + self.skyreject\
    + ' --skylines=' + self.skylines \
    + ' --merge'\
    + ' sci_basic_object.sof'

    esorex_cmd_std = '--log-file=sci_basic_std.log --log-level=debug'\
    + ' muse_scibasic'\
    + ' --nifu=-1'\
    + ' --resample'\
    + ' --saveimage=true'\
    + ' --skyreject=15.,15.,1'\
    + ' --merge'\
    + ' sci_basic_std.sof'

    if os.path.exists(self.working_dir + 'std/sci_basic_std.sof'):
        os.remove(self.working_dir + 'std/sci_basic_std.sof')

    for exposure_ID in range(len(exp_list_SCI)):
        print('>>> processing exposure: '\
        + str(exposure_ID + 1) + '/' + str(len(exp_list_SCI)))
        print('>>> processing: ' + exp_list_SCI[exposure_ID])
        print(' ')

        raw_data_list = ascii.read(exp_list_SCI[exposure_ID],\
        format='no_header')
        exposure_dir = exp_list_SCI[exposure_ID][:-9] + '/'

        MJDsillum = np.array([])
        illum_index = np.array([])

        for i in range(len(raw_data_list[1][:])):
            if raw_data_list[i][1] == 'OBJECT' or raw_data_list[i][1] == 'SKY':
                objecthdu = fits.open(raw_data_list[i][0])
                MJDobject = objecthdu[0].header['MJD-OBS']
            if raw_data_list[i][1] == 'STD':
                stdhdu = fits.open(raw_data_list[i][0])
                MJDstd = stdhdu[0].header['MJD-OBS']
            if raw_data_list[i][1] == 'ILLUM':
                illumhdu = fits.open(raw_data_list[i][0])
                MJDillum = illumhdu[0].header['MJD-OBS']
                MJDsillum = np.append(MJDsillum, MJDillum)
                illum_index = np.append(illum_index, i)

        choosen_illum_object = int(illum_index[np.argmin(np.abs(MJDsillum\
        - MJDobject))])
        choosen_illum_std = int(illum_index[np.argmin(np.abs(MJDsillum\
        - MJDstd))])

        f_std = open(self.working_dir + 'std/sci_basic_std_temp.sof', 'w')

        for i in range(len(raw_data_list[1][:])):
            if raw_data_list[i][1] == 'STD':
                f_std.write(raw_data_list[i][0]\
                + '  ' + raw_data_list[i][1] + '\n')

        f_std.write(raw_data_list[choosen_illum_std][0]\
        + '  ' + raw_data_list[choosen_illum_std][1] + '\n')

        if not self.using_ESO_calibration:
            f_std.write(self.calibration_dir\
            + 'SCIENCE/MASTER_BIAS.fits MASTER_BIAS\n')
            f_std.write(self.calibration_dir\
            + 'SCIENCE/MASTER_FLAT.fits MASTER_FLAT\n')
            f_std.write(self.calibration_dir\
            + 'TWILIGHT/TWILIGHT_CUBE.fits TWILIGHT_CUBE\n')
            f_std.write(self.calibration_dir\
            + 'SCIENCE/TRACE_TABLE.fits TRACE_TABLE\n')
            f_std.write(self.calibration_dir\
            + 'SCIENCE/WAVECAL_TABLE.fits WAVECAL_TABLE\n')
            if self.dark:
                f_std.write(self.calibration_dir\
                + 'DARK/MASTER_DARK.fits MASTER_DARK\n')

        if self.using_ESO_calibration:
            f_std.write(self.ESO_calibration_dir\
            + 'MASTER_BIAS.fits MASTER_BIAS\n')
            f_std.write(self.ESO_calibration_dir\
            + 'MASTER_FLAT.fits MASTER_FLAT\n')
            f_std.write(self.ESO_calibration_dir\
            + 'TWILIGHT_CUBE.fits TWILIGHT_CUBE\n')
            f_std.write(self.ESO_calibration_dir\
            + 'TRACE_TABLE.fits TRACE_TABLE\n')
            f_std.write(self.ESO_calibration_dir\
            + 'WAVECAL_TABLE.fits WAVECAL_TABLE\n')
            if self.dark:
                f_std.write(self.ESO_calibration_dir\
                + 'MASTER_DARK.fits MASTER_DARK\n')

        if self.mode == 'WFM-AO' or self.mode == 'WFM-NOAO':
            f_std.write(self.static_calibration_dir\
            + 'geometry_table_wfm.fits GEOMETRY_TABLE\n')
        if self.mode == 'NFM-AO':
            f_std.write(self.static_calibration_dir\
            + 'geometry_table_wfm.fits GEOMETRY_TABLE\n')

        f_std.write(self.static_calibration_dir\
        + 'badpix_table.fits BADPIX_TABLE\n')
        f_std.close()

        if create_sof:

            if os.path.exists(exposure_dir + 'sci_basic_object.sof'):
                os.remove(exposure_dir + 'sci_basic_object.sof')
            f_object = open(exposure_dir + 'sci_basic_object.sof', 'w')

            for i in range(len(raw_data_list[1][:])):
                if raw_data_list[i][1] == 'OBJECT':
                    f_object.write(raw_data_list[i][0]\
                    + '  ' + raw_data_list[i][1] + '\n')
                if raw_data_list[i][1] == 'SKY':
                    f_object.write(raw_data_list[i][0]\
                    + '  ' + raw_data_list[i][1] + '\n')

            f_object.write(raw_data_list[choosen_illum_object][0]\
            + '  ' + raw_data_list[choosen_illum_object][1] + '\n')

            if not self.using_ESO_calibration:
                f_object.write(self.calibration_dir\
                + 'SCIENCE/MASTER_BIAS.fits MASTER_BIAS\n')
                f_object.write(self.calibration_dir\
                + 'SCIENCE/MASTER_FLAT.fits MASTER_FLAT\n')
                f_object.write(self.calibration_dir\
                + 'TWILIGHT/TWILIGHT_CUBE.fits TWILIGHT_CUBE\n')
                f_object.write(self.calibration_dir\
                + 'SCIENCE/TRACE_TABLE.fits TRACE_TABLE\n')
                f_object.write(self.calibration_dir\
                + 'SCIENCE/WAVECAL_TABLE.fits WAVECAL_TABLE\n')
                if self.dark:
                    f_object.write(self.calibration_dir\
                    + 'DARK/MASTER_DARK.fits MASTER_DARK\n')

            if self.using_ESO_calibration:
                f_object.write(self.ESO_calibration_dir\
                + 'MASTER_BIAS.fits MASTER_BIAS\n')
                f_object.write(self.ESO_calibration_dir\
                + 'MASTER_FLAT.fits MASTER_FLAT\n')
                f_object.write(self.ESO_calibration_dir\
                + 'TWILIGHT_CUBE.fits TWILIGHT_CUBE\n')
                f_object.write(self.ESO_calibration_dir\
                + 'TRACE_TABLE.fits TRACE_TABLE\n')
                f_object.write(self.ESO_calibration_dir\
                + 'WAVECAL_TABLE.fits WAVECAL_TABLE\n')
                if self.dark:
                    f_object.write(self.ESO_calibration_dir\
                    + 'MASTER_DARK.fits MASTER_DARK\n')

            if self.mode == 'WFM-AO' or self.mode == 'WFM-NOAO':
                f_object.write(self.static_calibration_dir\
                + 'geometry_table_wfm.fits GEOMETRY_TABLE\n')
            if self.mode == 'NFM-AO':
                f_object.write(self.static_calibration_dir\
                + 'geometry_table_wfm.fits GEOMETRY_TABLE\n')
            f_object.write(self.static_calibration_dir\
            + 'badpix_table.fits BADPIX_TABLE\n')

            f_object.close()

        if not self.debug:
            _call_esorex(self, exposure_dir, esorex_cmd)
        if os.path.isfile(self.working_dir + 'std/sci_basic_std.sof'):
            assert filecmp.cmp(self.working_dir + 'std/sci_basic_std.sof',\
            self.working_dir + 'std/sci_basic_std_temp.sof'),\
            'CAUTION DIFFERENT STD STARS FOR VARIOUS FIELDS: PLEASE CHECK'
            os.remove(self.working_dir + 'std/sci_basic_std_temp.sof')

        else:
            os.rename(self.working_dir + 'std/sci_basic_std_temp.sof',\
            self.working_dir + 'std/sci_basic_std.sof')

    if not self.debug:
        if self.reduce_std:
            _call_esorex(self, self.working_dir + 'std/', esorex_cmd_std)


def _std_flux(self, exp_list_SCI, create_sof):

    '''
    This module calls *ESORex'* ``muse_standard``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... FLUX CALIBRATION')

    esorex_cmd = ' --log-file=std_flux.log --log-level=debug'\
    + ' muse_standard'\
    + ' --filter=white'\
    + ' std_flux.sof'

    PIXTABLE_STD_list = _get_filelist(self, self.working_dir + 'std/',\
    'PIXTABLE_STD*.fits')

    if create_sof:

        if os.path.exists(self.working_dir + 'std/' + 'std_flux.sof'):
            os.remove(self.working_dir + 'std/' + 'std_flux.sof')

        f = open(self.working_dir + 'std/' + 'std_flux.sof', 'w')
        for i in range(len(PIXTABLE_STD_list)):
            f.write(self.working_dir + 'std/' + PIXTABLE_STD_list[i]\
            + ' PIXTABLE_STD\n')
        f.write(self.static_calibration_dir
        + 'extinct_table.fits EXTINCT_TABLE\n')
        f.write(self.static_calibration_dir\
        + 'std_flux_table.fits STD_FLUX_TABLE\n')
        f.write(self.static_calibration_dir\
        + 'filter_list.fits FILTER_LIST\n')
        f.close()

    if not self.debug:
        _call_esorex(self, self.working_dir + 'std/', esorex_cmd)


def _sky(self, exp_list_SCI, create_sof):

    '''
    This module calls *ESORex'* ``muse_sky``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... SKY CREATION')

    sky = np.zeros_like(exp_list_SCI, dtype=bool)
    sci = np.zeros_like(exp_list_SCI, dtype=bool)

    for idx, exposure in enumerate(exp_list_SCI):
        if exposure[-8:-5] == 'SKY': sky[idx] = True
        if exposure[-8:-5] == 'SCI': sci[idx] = True

    if self.skyfield == 'auto' and (sky == True).any():
        exp_list_SCI_sky = np.array(exp_list_SCI)[sky]
    else:
        exp_list_SCI_sky = np.array(exp_list_SCI)

    for exposure_ID in range(len(exp_list_SCI_sky)):
        print('>>> processing exposure: '\
        + str(exposure_ID + 1) + '/' + str(len(exp_list_SCI_sky)))
        print('>>> processing: ' + exp_list_SCI_sky[exposure_ID])
        print(' ')

        raw_data_list = ascii.read(exp_list_SCI_sky[exposure_ID],\
        format='no_header')
        exposure_dir = exp_list_SCI_sky[exposure_ID][:-9] + '/'

        if self.skyfield == 'auto' and (sky == True).any():
            PIXTABLE_SKY_list =\
            _get_filelist(self, exposure_dir, 'PIXTABLE_SKY*.fits')
        else:
            PIXTABLE_SKY_list =\
            _get_filelist(self, exposure_dir, 'PIXTABLE_OBJECT*.fits')

        if create_sof:

            if os.path.exists(exposure_dir + 'sky.sof'):
                os.remove(exposure_dir + 'sky.sof')

            f = open(exposure_dir + 'sky.sof', 'w')
            for i in range(len(PIXTABLE_SKY_list)):
                f.write(exposure_dir + PIXTABLE_SKY_list[i]\
                + ' PIXTABLE_SKY\n')
            if not self.using_ESO_calibration:
                f.write(self.calibration_dir +\
                'SCIENCE/LSF_PROFILE.fits LSF_PROFILE\n')

            if self.using_ESO_calibration:
                f.write(self.ESO_calibration_dir +\
                'LSF_PROFILE.fits LSF_PROFILE\n')

            f.write(self.working_dir\
            + 'std/' + 'STD_RESPONSE_0001.fits STD_RESPONSE\n')
            f.write(self.working_dir\
            + 'std/' + 'STD_TELLURIC_0001.fits STD_TELLURIC\n')

            f.write(self.static_calibration_dir\
            + 'extinct_table.fits EXTINCT_TABLE\n')
            f.write(self.static_calibration_dir\
            + 'sky_lines.fits SKY_LINES\n')

            f.close()

        esorex_cmd = '--log-file=sky.log --log-level=debug'\
        + ' muse_create_sky'\
        + ' --fraction=' + str(self.skyfraction)\
        + ' --ignore=' + str(self.skyignore)\
        + ' sky.sof'

        if self.skyfield == 'auto' and (sky == True).any():
            if not self.debug:
                _call_esorex(self, exposure_dir, esorex_cmd)
        else:
            if not self.debug:
                _call_esorex(self, exposure_dir,\
                '--log-file=sky.log --log-level=debug'\
                + ' muse_create_sky'\
                + ' --fraction=' + str(self.skyfraction)\
                + ' --ignore=' + str(self.skyignore)\
                + ' sky.sof')

    if self.skyfield == 'auto' and (sky == True).any():
        skydate = np.ones_like(exp_list_SCI_sky, dtype=float)

        for idx, exps in enumerate(exp_list_SCI_sky):
            skydate[idx] = fits.open(exps[:-9]\
            + '/PIXTABLE_SKY_0001-01.fits')[0].header['MJD-OBS']
        for idx, exps in enumerate(np.array(exp_list_SCI)[sci]):
            scidate = fits.open(exps[:-9]\
            + '/PIXTABLE_OBJECT_0001-01.fits')[0].header['MJD-OBS']

            ind = np.argmin(abs(skydate - scidate))
            flist = glob.glob(exp_list_SCI_sky[ind][:-9] + '/SKY_*.fits')
            for f in flist:
                shutil.copy(f, exps[:-9] + '/.')


def _modified_sky(self, exp_list_SCI, create_sof):

    '''
    This module calls *ESORex'* ``muse_sky`` with the modified continuum and
    line subtraction as described in `Zeidler et al. 2019`_.

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... SKY CREATION MODIFIED')

    sky = np.zeros_like(exp_list_SCI, dtype=bool)
    sci = np.zeros_like(exp_list_SCI, dtype=bool)

    for idx, exposure in enumerate(exp_list_SCI):
        if exposure[-8:-5] == 'SKY':
            sky[idx] = True
        if exposure[-8:-5] == 'SCI':
            sci[idx] = True

    if self.skyfield == 'auto' and (sky == True).any():
        exp_list_SCI_sky = np.array(exp_list_SCI)[sky]
    else:
        exp_list_SCI_sky = np.array(exp_list_SCI)

    for exposure_ID in range(len(exp_list_SCI_sky)):
        print('>>> processing exposure: ' + str(exposure_ID + 1)\
        + '/' + str(len(exp_list_SCI_sky)))
        print('>>> processing: ' + exp_list_SCI_sky[exposure_ID])
        print(' ')

        raw_data_list = ascii.read(exp_list_SCI_sky[exposure_ID],\
        format='no_header')
        exposure_dir = exp_list_SCI_sky[exposure_ID][:-9] + '/'

        if self.skyfield == 'auto' and (sky == True).any():
            PIXTABLE_SKY_list = _get_filelist(self, exposure_dir,\
            'PIXTABLE_SKY*.fits')
        else:
            PIXTABLE_SKY_list = _get_filelist(self, exposure_dir,\
            'PIXTABLE_OBJECT*.fits')

        if create_sof:

            if os.path.exists(exposure_dir + 'sky.sof'):
                os.remove(exposure_dir + 'sky.sof')

            f = open(exposure_dir + 'sky.sof', 'w')
            for i in range(len(PIXTABLE_SKY_list)):
                f.write(exposure_dir\
                + PIXTABLE_SKY_list[i] + ' PIXTABLE_SKY\n')
            if not self.using_ESO_calibration:
                f.write(self.calibration_dir\
                + 'SCIENCE/LSF_PROFILE.fits LSF_PROFILE\n')

            if self.using_ESO_calibration:
                f.write(self.ESO_calibration_dir\
                + 'LSF_PROFILE.fits LSF_PROFILE\n')

            f.write(self.working_dir\
            + 'std/' + 'STD_RESPONSE_0001.fits STD_RESPONSE\n')
            f.write(self.working_dir\
            + 'std/' + 'STD_TELLURIC_0001.fits STD_TELLURIC\n')

            f.write(self.static_calibration_dir\
            + 'extinct_table.fits EXTINCT_TABLE\n')
            f.write(self.static_calibration_dir\
            + 'sky_lines.fits SKY_LINES\n')

            f.close()

        if self.skyfield == 'auto' and (sky == True).any():
            if not self.debug:
                _call_esorex(self, exposure_dir,\
                '--log-file=sky.log --log-level=debug'\
                + ' muse_create_sky'\
                + ' --fraction=' + str(self.skyfraction)\
                + ' --ignore=' + str(self.skyignore)\
                + ' sky.sof')
        else:
            if not self.debug:
                _call_esorex(self, exposure_dir,\
                '--log-file=sky.log --log-level=debug'\
                + ' muse_create_sky'\
                + ' --fraction=' + str(self.skyfraction)\
                + ' --ignore=' + str(self.skyignore)\
                + ' sky.sof')

        os.chdir(exposure_dir)
        sky_cont_hdu = fits.open('SKY_CONTINUUM.fits', checksum=True)
        sky_cont = sky_cont_hdu[1].data
        for i in range(len(sky_cont)):
            sky_cont[i][1] = 0.
        sky_cont_hdu[1].data = sky_cont
        sky_cont_hdu.writeto('SKY_CONTINUUM_zero.fits',\
        overwrite=True, checksum=True)
        os.chdir(self.rootpath)

        if create_sof:

            if os.path.exists(exposure_dir + 'sky.sof'):
                os.remove(exposure_dir + 'sky.sof')

            f = open(exposure_dir + 'sky.sof', 'w')
            for i in range(len(PIXTABLE_SKY_list)):
                f.write(exposure_dir + PIXTABLE_SKY_list[i]\
                + ' PIXTABLE_SKY\n')
            if not self.using_ESO_calibration:
                f.write(self.calibration_dir\
                + 'SCIENCE/LSF_PROFILE.fits LSF_PROFILE\n')

            if self.using_ESO_calibration:
                f.write(self.ESO_calibration_dir\
                + 'LSF_PROFILE.fits LSF_PROFILE\n')

            f.write(self.working_dir\
            + 'std/' + 'STD_RESPONSE_0001.fits STD_RESPONSE\n')
            f.write(self.working_dir\
            + 'std/' + 'STD_TELLURIC_0001.fits STD_TELLURIC\n')
            f.write(exposure_dir\
            + 'SKY_CONTINUUM_zero.fits SKY_CONTINUUM\n')

            f.write(self.static_calibration_dir\
            + 'extinct_table.fits EXTINCT_TABLE\n')
            f.write(self.static_calibration_dir\
            + 'sky_lines.fits SKY_LINES\n')

            f.close()

        if self.skyfield == 'auto' and (sky == True).any():
            if not self.debug:
                _call_esorex(self, exposure_dir,\
                '--log-file=sky.log --log-level=debug'\
                + ' muse_create_sky'\
                + ' --fraction=' + str(self.skyfraction)\
                + ' --ignore=' + str(self.skyignore)\
                + ' sky.sof')
        else:
            if not self.debug:
                _call_esorex(self, exposure_dir,\
                ' --log-file=sky.log --log-level=debug'\
                + ' muse_create_sky'\
                + ' --fraction=' + str(self.skyfraction)\
                + ' --ignore=' + str(self.skyignore)\
                + ' sky.sof')

        os.chdir(exposure_dir)
        print('SKY_CONTINUUM_zero.fits ==> SKY_CONTINUUM.fits')
        shutil.copy('SKY_CONTINUUM_zero.fits', 'SKY_CONTINUUM.fits')
        hdu = fits.open('SKY_LINES.fits', checksum=True)
        data = hdu[1].data

        lines_to_keep = [i for i, sky_line in enumerate(data) \
        if (sky_line[0][:2] == 'O2' or sky_line[0][:2] == 'OH')]
        data = data[lines_to_keep]

        hdu[1].data = data
        hdu.writeto('SKY_LINES.fits', overwrite=True, checksum=True)
        os.chdir(self.rootpath)

    if self.skyfield == 'auto' and (sky == True).any():
        skydate = np.ones_like(exp_list_SCI_sky, dtype=float)

        for idx, exps in enumerate(exp_list_SCI_sky):
            skydate[idx] = fits.open(exps[:-9]\
            + '/PIXTABLE_SKY_0001-01.fits')[0].header['MJD-OBS']
        for idx, exps in enumerate(np.array(exp_list_SCI)[sci]):
            scidate = fits.open(exps[:-9]\
            + '/PIXTABLE_OBJECT_0001-01.fits')[0].header['MJD-OBS']

            ind = np.argmin(abs(skydate - scidate))
            flist = glob.glob(exp_list_SCI_sky[ind][:-9] + '/SKY_*.fits')
            for f in flist:
                shutil.copy(f, exps[:-9] + '/.')


def _scipost(self, exp_list_SCI, create_sof, OB):

    '''
    This module calls *ESORex'* ``muse_scipost``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

        OB : :obj:`str`:
            The specific ``OB`` to be reduced. 

    '''

    print('... SCIENCE POST-PROCESSING')

    unique_pointings = np.array([])
    unique_tester = ' '

    sci = np.zeros_like(exp_list_SCI, dtype=bool)
    for idx, exposure in enumerate(exp_list_SCI):
        if exposure[-8:-5] == 'SCI':
            sci[idx] = True
    exp_list_SCI = np.array(exp_list_SCI)[sci]

    for expnum in range(len(exp_list_SCI)):
        if unique_tester.find(exp_list_SCI[expnum][:-16]) == -1:
            unique_pointings = np.append(unique_pointings,\
            exp_list_SCI[expnum][:-16])
            unique_tester = unique_tester + exp_list_SCI[expnum][:-16]

    for unique_pointing_num in range(len(unique_pointings)):

        print(' ')
        print('>>> processing pointing: '
        + str(unique_pointing_num + 1) + '/' + str(len(unique_pointings)))
        print(' ')

        unique_pointings_ID = unique_pointings[unique_pointing_num][-18:]
        sec = unique_pointings[unique_pointing_num]
        exp_list = glob.glob(sec + '*SCI.list')

        for exp_num in range(len(exp_list)):

            print(' ')
            print('>>> processing exposure: '\
            + str(exp_num + 1) + '/' + str(len(exp_list)))
            print(' ')

            PIXTABLE_OBJECT_list = _get_filelist(self, exp_list[exp_num][:-9],\
            'PIXTABLE_OBJECT*.fits')

            if create_sof:

                if os.path.exists(exp_list[exp_num][:-9] + '/scipost.sof'):
                    os.remove(exp_list[exp_num][:-9] + '/scipost.sof')

                f = open(exp_list[exp_num][:-9] + '/scipost.sof', 'w')
                for i in range(len(PIXTABLE_OBJECT_list)):
                    f.write(exp_list[exp_num][:-9]\
                    + '/' + PIXTABLE_OBJECT_list[i] + ' PIXTABLE_OBJECT\n')

                if not self.using_ESO_calibration:
                    f.write(self.calibration_dir\
                    + 'SCIENCE/LSF_PROFILE.fits LSF_PROFILE\n')
                if self.using_ESO_calibration:
                    f.write(self.ESO_calibration_dir\
                    + 'LSF_PROFILE.fits LSF_PROFILE\n')

                f.write(self.working_dir\
                + 'std/' + 'STD_RESPONSE_0001.fits STD_RESPONSE\n')
                f.write(self.working_dir\
                + 'std/' + 'STD_TELLURIC_0001.fits STD_TELLURIC\n')
                if self.skysub:
                    f.write(exp_list[exp_num][:-9]\
                    + '/' + 'SKY_LINES.fits SKY_LINES\n')
                    f.write(exp_list[exp_num][:-9]\
                    + '/' + 'SKY_CONTINUUM.fits SKY_CONTINUUM\n')
                if self.mode == 'WFM-AO' or self.mode == 'WFM-NOAO':
                    f.write(self.static_calibration_dir\
                    + 'astrometry_wcs_wfm.fits ASTROMETRY_WCS\n')
                if self.mode == 'NFM-AO':
                    f.write(self.static_calibration_dir\
                    + 'astrometry_wcs_nfm.fits ASTROMETRY_WCS\n')
                if not self.autocalib == 'none':
                    f.write(exp_list[exp_num][:-9]\
                    + '/' + 'SKY_MASK.fits SKY_MASK\n')
                if self.autocalib == 'user':
                    f.write(exp_list[exp_num][:-9]\
                    + '/' + 'AUTOCAL_FACTORS.fits AUTOCAL_FACTORS\n')

                f.write(self.static_calibration_dir\
                + 'extinct_table.fits EXTINCT_TABLE\n')
                f.write(self.static_calibration_dir\
                + 'filter_list.fits FILTER_LIST\n')
                if self.raman:
                    f.write(self.static_calibration_dir\
                    + 'raman_lines.fits RAMAN_LINES\n')

                f.close()
            if self.withrvcorr:

                if self.skysub:
                    print('with sky subtraction...')
                if not self.skysub:
                    print('without sky subtraction...')

                if not self.debug:
                    _call_esorex(self, exp_list[exp_num][:-9],\
                    '--log-file=scipost.log --log-level=debug'\
                    + ' muse_scipost'\
                    + ' --save=cube,skymodel,individual,raman,autocal'\
                    + ' --skymethod=' + self.skymethod\
                    + ' --autocalib=' + self.autocalib
                    + ' --filter=white'\
                    + ' scipost.sof')

                if self.skysub:

                    os.chdir(exp_list[exp_num][:-9])
                    if not self.debug:
                        os.rename('DATACUBE_FINAL.fits',\
                        'DATACUBE_FINAL_wosky.fits')
                        os.rename('IMAGE_FOV_0001.fits',\
                        'IMAGE_FOV_0001_wosky.fits')
                        os.rename('PIXTABLE_REDUCED_0001.fits',\
                        'PIXTABLE_REDUCED_0001_wosky.fits')
                    os.chdir(self.rootpath)

                if not self.skysub:

                    os.chdir(exp_list[exp_num][:-9])
                    if not self.debug:
                        os.rename('DATACUBE_FINAL.fits',\
                        'DATACUBE_FINAL_wsky.fits')
                        os.rename('IMAGE_FOV_0001.fits',\
                        'IMAGE_FOV_0001_wsky.fits')
                        os.rename('PIXTABLE_REDUCED_0001.fits',\
                        'PIXTABLE_REDUCED_0001_wsky.fits')
                    os.chdir(self.rootpath)

            else:

                if self.raman:
                    if not self.debug:
                        _call_esorex(self, exp_list[exp_num][:-9],\
                        '--log-file=scipost.log --log-level=debug'\
                        + ' muse_scipost'\
                        + ' --save=cube,skymodel,individual,raman,autocal'\
                        + ' --skymethod=none'\
                        + ' --filter=white'\
                        + ' --autocalib=' + self.autocalib)\
                        + ' --rvcorr=none'\
                        + ' scipost.sof'

                os.chdir(exp_list[exp_num][:-9])
                if not self.debug:
                    os.rename('DATACUBE_FINAL.fits',\
                    'DATACUBE_FINAL_wskynorvcorr.fits')
                    os.rename('IMAGE_FOV_0001.fits',\
                    'IMAGE_FOV_0001_wskynorvcorr.fits')
                    os.rename('PIXTABLE_REDUCED_0001.fits',\
                    'PIXTABLE_REDUCED_0001_wskynorvcorr.fits')
                os.chdir(self.rootpath)


def _dither_collect(self, exp_list_SCI, OB):

    '''
    This module collects the individual dither exposures for one OB to be
    combined in the steps: :mod:`MUSEreduce._exp_align` and
    :mod:`MUSEreduce._exp_combine`

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

        OB : :obj:`str`:
            The specific ``OB`` to be reduced.

    '''

    print('... COLLECT DITHER POSTITIONS')

    unique_pointings = np.array([])
    unique_tester = ' '

    sci = np.zeros_like(exp_list_SCI, dtype=bool)
    for idx, exposure in enumerate(exp_list_SCI):
        if exposure[-8:-5] == 'SCI':
            sci[idx] = True
    exp_list_SCI = np.array(exp_list_SCI)[sci]

    print(' ')
    print('>>> Copying files:')
    print(' ')

    if len(self.user_list) == 0:
        for expnum in range(len(exp_list_SCI)):
                if unique_tester.find(exp_list_SCI[expnum][:-16]) == -1:
                    unique_pointings = np.append(unique_pointings,\
                    exp_list_SCI[expnum][:-16])
                    unique_tester = unique_tester + exp_list_SCI[expnum][:-16]

    if len(self.user_list) > 0:
        unique_pointings = self.working_dir\
        + np.array([self.user_list[0][:18]], dtype=object)

    for unique_pointing_num in range(len(unique_pointings)):
        unique_pointings_ID = unique_pointings[unique_pointing_num][-18:]
        sec = unique_pointings[unique_pointing_num]

        if len(self.user_list) == 0:
            exp_list = glob.glob(sec + '*SCI.list')
        if len(self.user_list) > 0:
            exp_list = self.working_dir + self.user_list + '_SCI.list'

        if self.dithering_multiple_OBs:
            if self.withrvcorr:
                combining_exposure_dir_withoutsky = self.combining_OBs_dir\
                + unique_pointings_ID + '/withoutsky_withrvcorr'
                combining_exposure_dir_withsky = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withoutrvcorr'

        if not self.dithering_multiple_OBs:
            if self.withrvcorr:
                combining_exposure_dir_withoutsky =\
                sec + '/withoutsky_withrvcorr'
                combining_exposure_dir_withsky = sec + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = sec + '/withsky_withoutrvcorr'

        if not os.path.exists(combining_exposure_dir_withoutsky):
            os.makedirs(combining_exposure_dir_withoutsky)
        if not os.path.exists(combining_exposure_dir_withsky):
            os.makedirs(combining_exposure_dir_withsky)

        if self.withrvcorr:
            files = glob.glob(combining_exposure_dir_withoutsky\
            + '/*FOV_0001*')
            if len(files) > 0:
                for f in files:
                    os.remove(f)
            files = glob.glob(combining_exposure_dir_withsky + '/*FOV_0001*')
            if len(files) > 0:
                for f in files:
                    os.remove(f)
        if not self.withrvcorr:
            files = glob.glob(combining_exposure_dir + '/*FOV_0001*')
            if len(files) > 0:
                for f in files:
                    os.remove(f)

    for unique_pointing_num in range(len(unique_pointings)):

        unique_pointings_ID = unique_pointings[unique_pointing_num][-18:]
        sec = unique_pointings[unique_pointing_num]

        if len(self.user_list) == 0:
            exp_list = glob.glob(sec + '*SCI.list')
        if len(self.user_list) > 0:
            exp_list = self.working_dir + self.user_list + '_SCI.list'

        if self.dithering_multiple_OBs:
            if self.withrvcorr:
                combining_exposure_dir_withoutsky = self.combining_OBs_dir\
                + unique_pointings_ID + '/withoutsky_withrvcorr'
                combining_exposure_dir_withsky = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withoutrvcorr'

        if not self.dithering_multiple_OBs:
            if self.withrvcorr:
                combining_exposure_dir_withoutsky =\
                sec + '/withoutsky_withrvcorr'
                combining_exposure_dir_withsky = sec + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = sec + '/withsky_withoutrvcorr'

        ident_pos = np.zeros(len(exp_list))
        for idx in range(len(exp_list)):
            ident = 0
            for idx2 in range(len(exp_list)):
                if exp_list[idx][-20:-12] == exp_list[idx2][-20:-12]:
                    ident_pos[idx2] = ident
                    ident += 1

        for exp_num in range(len(exp_list)):
            if self.withrvcorr:
                if self.skysub:
                    if self.dithering_multiple_OBs:

                        print(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wosky.fits ==> '\
                        + combining_exposure_dir_withoutsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wosky.fits',\
                        combining_exposure_dir_withoutsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wosky.fits ==> '\
                        + combining_exposure_dir_withoutsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wosky.fits',\
                        combining_exposure_dir_withoutsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wosky.fits ==> '\
                        + combining_exposure_dir_withoutsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wosky.fits',\
                        combining_exposure_dir_withoutsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                    else:
                        print(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wosky.fits ==> '\
                        + combining_exposure_dir_withoutsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wosky.fits',\
                        combining_exposure_dir_withoutsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wosky.fits ==> '\
                        + combining_exposure_dir_withoutsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wosky.fits',\
                        combining_exposure_dir_withoutsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wosky.fits ==> '\
                        + combining_exposure_dir_withoutsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wosky.fits',\
                        combining_exposure_dir_withoutsky +\
                        '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12] + '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                if not self.skysub:
                    if self.dithering_multiple_OBs:
                        print(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wsky.fits ==> '\
                        + combining_exposure_dir_withsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wsky.fits',\
                        combining_exposure_dir_withsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wsky.fits ==> '\
                        + combining_exposure_dir_withsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wsky.fits',\
                        combining_exposure_dir_withsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9] +\
                        ' /PIXTABLE_REDUCED_0001_wsky.fits ==> '\
                        + combining_exposure_dir_withsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wsky.fits',\
                        combining_exposure_dir_withsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                    else:
                        print(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wsky.fits ==> '\
                        + combining_exposure_dir_withsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/DATACUBE_FINAL_wsky.fits',\
                        combining_exposure_dir_withsky\
                        + '/DATACUBE_FINAL_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wsky.fits ==> '\
                        + combining_exposure_dir_withsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/IMAGE_FOV_0001_wsky.fits',\
                        combining_exposure_dir_withsky\
                        + '/IMAGE_FOV_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                        print(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wsky.fits ==> '\
                        + combining_exposure_dir_withsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                        shutil.copy(exp_list[exp_num][:-9]\
                        + '/PIXTABLE_REDUCED_0001_wsky.fits',\
                        combining_exposure_dir_withsky\
                        + '/PIXTABLE_REDUCED_' + OB + '_'\
                        + exp_list[exp_num][-20:-12]+ '_'\
                        + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
            else:

                if dithering_multiple_OBs:
                    print(exp_list[exp_num][:-9]\
                    + '/DATACUBE_FINAL_wskynorvcorr.fits ==> '\
                    + combining_exposure_dir + '/DATACUBE_FINAL_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                    shutil.copy(exp_list[exp_num][:-9]\
                    + '/DATACUBE_FINAL_wskynorvcorr.fits',\
                    combining_exposure_dir + '/DATACUBE_FINAL_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                    print(exp_list[exp_num][:-9]\
                    + '/IMAGE_FOV_0001_wskynorvcorr.fits ==> '\
                    + combining_exposure_dir + '/IMAGE_FOV_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                    shutil.copy(exp_list[exp_num][:-9]\
                    + '/IMAGE_FOV_0001_wskynorvcorr.fits',\
                    combining_exposure_dir + '/IMAGE_FOV_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                    print(exp_list[exp_num][:-9]\
                    + '/PIXTABLE_REDUCED_0001_wskynorvcorr.fits ==> '\
                    + combining_exposure_dir + '/PIXTABLE_REDUCED_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                    shutil.copy(exp_list[exp_num][:-9]\
                    + '/PIXTABLE_REDUCED_0001_wskynorvcorr.fits',\
                    combining_exposure_dir + '/PIXTABLE_REDUCED_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                else:
                    print(exp_list[exp_num][:-9]\
                    + '/DATACUBE_FINAL_wskynorvcorr.fits ==> '\
                    + combining_exposure_dir\
                    + '/DATACUBE_FINAL_' + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                    shutil.copy(exp_list[exp_num][:-9]\
                    + '/DATACUBE_FINAL_wskynorvcorr.fits',\
                    combining_exposure_dir + '/DATACUBE_FINAL_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                    print(exp_list[exp_num][:-9]\
                    + '/IMAGE_FOV_0001_wskynorvcorr.fits ==> '\
                    + combining_exposure_dir\
                    + '/IMAGE_FOV_'+ OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                    shutil.copy(exp_list[exp_num][:-9]\
                    + '/IMAGE_FOV_0001_wskynorvcorr.fits',\
                    combining_exposure_dir + '/IMAGE_FOV_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')

                    print(exp_list[exp_num][:-9]\
                    + '/PIXTABLE_REDUCED_0001_wskynorvcorr.fits ==> '\
                    + combining_exposure_dir\
                    + '/PIXTABLE_REDUCED_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')
                    shutil.copy(exp_list[exp_num][:-9]\
                    + '/PIXTABLE_REDUCED_0001_wskynorvcorr.fits',\
                    combining_exposure_dir + '/PIXTABLE_REDUCED_'\
                    + OB + '_'\
                    + exp_list[exp_num][-20:-12]+ '_'\
                    + str(int(ident_pos[exp_num])).rjust(2, '0') + '.fits')


def _exp_align(self, exp_list_SCI, create_sof, OB):

    '''
    This module calls *ESORex'* ``muse_exp_align``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

        OB : :obj:`str`:
            The specific ``OB`` to be reduced.

    '''

    print('... CUBE ALIGNMENT')

    esorex_cmd = '--log-file=exp_align.log --log-level=debug'\
    + ' muse_exp_align'\
    + ' --srcmin='+str(self.config['exp_align']['srcmin'])\
    + ' --srcmax='+str(self.config['exp_align']['srcmax'])\
    + ' exp_align.sof'

    unique_pointings = np.array([])
    unique_tester = ' '

    sci = np.zeros_like(exp_list_SCI, dtype=bool)
    for idx, exposure in enumerate(exp_list_SCI):
        if exposure[-8:-5] == 'SCI':
            sci[idx] = True
    exp_list_SCI = np.array(exp_list_SCI)[sci]

    if len(self.user_list) == 0:
        for expnum in range(len(exp_list_SCI)):
                if unique_tester.find(exp_list_SCI[expnum][:-16]) == -1:
                    unique_pointings = np.append(unique_pointings,\
                    exp_list_SCI[expnum][:-16])
                    unique_tester = unique_tester + exp_list_SCI[expnum][:-16]

    if len(self.user_list) > 0:
        unique_pointings = unique_pointings = self.working_dir\
        + np.array([self.user_list[0][:18]], dtype=object)

    for unique_pointing_num in range(len(unique_pointings)):

        unique_pointings_ID = unique_pointings[unique_pointing_num][-18:]
        sec = unique_pointings[unique_pointing_num]

        if len(self.user_list) == 0:
            exp_list = glob.glob(sec + '*SCI.list')
        if len(self.user_list) > 0:
            exp_list = self.working_dir + self.user_list + '_SCI.list'

        if self.dithering_multiple_OBs:
            if self.withrvcorr:
                combining_exposure_dir_withoutsky = self.combining_OBs_dir\
                + unique_pointings_ID + '/withoutsky_withrvcorr'
                combining_exposure_dir_withsky = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withoutrvcorr'

        if not self.dithering_multiple_OBs:
            if self.withrvcorr:
                combining_exposure_dir_withoutsky = sec\
                + '/withoutsky_withrvcorr'
                combining_exposure_dir_withsky = sec\
                + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = sec + '/withsky_withoutrvcorr'

    for unique_pointing_num in range(len(unique_pointings)):

        print(' ')
        print('>>> processing pointing: ' + str(unique_pointing_num + 1)\
        + '/' + str(len(unique_pointings)))
        print(' ')

        unique_pointings_ID = unique_pointings[unique_pointing_num][-18:]
        sec = unique_pointings[unique_pointing_num]

        if self.dithering_multiple_OBs:
            if self.withrvcorr:
                print(unique_pointings_ID)
                if self.skysub:
                    combining_exposure_dir_withoutsky = self.combining_OBs_dir\
                    + unique_pointings_ID + '/withoutsky_withrvcorr'
                if not self.skysub:
                    combining_exposure_dir_withsky = self.combining_OBs_dir\
                    + unique_pointings_ID + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withoutrvcorr'

        if not self.dithering_multiple_OBs:
            if self.withrvcorr:
                if self.skysub:
                    combining_exposure_dir_withoutsky = sec\
                    + '/withoutsky_withrvcorr'
                if not self.skysub:
                    combining_exposure_dir_withsky = sec\
                    + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = sec + '/withsky_withoutrvcorr'

        if self.withrvcorr:
            if self.skysub:
                exp_list = _get_filelist(self,\
                combining_exposure_dir_withoutsky, 'IMAGE_FOV_*.fits')
                if create_sof:
                    if os.path.exists(combining_exposure_dir_withoutsky\
                    + '/exp_align.sof'):
                        os.remove(combining_exposure_dir_withoutsky\
                        + '/exp_align.sof')

                    f = open(combining_exposure_dir_withoutsky\
                    + '/exp_align.sof', 'w')
                    for i in range(len(exp_list)):
                        f.write(combining_exposure_dir_withoutsky\
                        + '/' + exp_list[i] + ' IMAGE_FOV\n')
                    f.close()
                if not self.debug:
                    _call_esorex(self, combining_exposure_dir_withoutsky,\
                    esorex_cmd)

            if not self.skysub:
                exp_list = _get_filelist(self,\
                combining_exposure_dir_withsky, 'IMAGE_FOV_*.fits')
                if create_sof:
                    if os.path.exists(combining_exposure_dir_withsky\
                    + '/exp_align.sof'):
                        os.remove(combining_exposure_dir_withsky\
                        + '/exp_align.sof')
                    f = open(combining_exposure_dir_withsky\
                    + '/exp_align.sof', 'w')
                    for i in range(len(exp_list)):
                        f.write(combining_exposure_dir_withsky\
                        + '/' + exp_list[i] + ' IMAGE_FOV\n')
                    f.close()
                if not self.debug:
                    _call_esorex(self, combining_exposure_dir_withsky,\
                    esorex_cmd)

        else:
            exp_list = _get_filelist(self,\
            combining_exposure_dir, 'IMAGE_FOV_*.fits')
            if create_sof:
                if os.path.exists(combining_exposure_dir + '/exp_align.sof'):
                    os.remove(combining_exposure_dir + '/exp_align.sof')
                f = open(combining_exposure_dir + '/exp_align.sof', 'w')
                for i in range(len(exp_list)):
                    f.write(combining_exposure_dir\
                    + '/' + exp_list[i] + ' IMAGE_FOV\n')
                f.close()
            if not self.debug:
                _call_esorex(self, combining_exposure_dir, esorex_cmd)


def _exp_combine(self, exp_list_SCI, create_sof):

    '''
    This module calls *ESORex'* ``muse_exp_combine``

    Args:
        exp_list_SCI : :obj:`list`
            The list of all associated science files including their category
            read from the fits header

        create_sof : :obj:`bool`:
            :obj:`True`: ``bias.sof`` is created and populated

            :obj:`False`: ``bias.sof`` is not created and needs to be provided
            by the user

    '''

    print('... EXPOSURE COMBINATION')

    esorex_cmd = '--log-file=exp_combine.log --log-level=debug'\
    + ' muse_exp_combine'\
    + ' --filter=white'\
    + ' --save=cube'\
    + ' --crsigma=5.'\
    + ' --weight=' + str(self.weight)\
    + ' exp_combine.sof'

    unique_pointings = np.array([])
    unique_tester = ' '

    sci = np.zeros_like(exp_list_SCI, dtype=bool)
    for idx, exposure in enumerate(exp_list_SCI):
        if exposure[-8:-5] == 'SCI':
            sci[idx] = True
    exp_list_SCI = np.array(exp_list_SCI)[sci]

    if len(self.user_list) == 0:
        for expnum in range(len(exp_list_SCI)):
                if unique_tester.find(exp_list_SCI[expnum][:-16]) == -1:
                    unique_pointings = np.append(unique_pointings,\
                    exp_list_SCI[expnum][:-16])
                    unique_tester = unique_tester + exp_list_SCI[expnum][:-16]

    if len(self.user_list) > 0:
        unique_pointings = self.working_dir\
        + np.array([self.user_list[0][:18]], dtype=object)

    for unique_pointing_num in range(len(unique_pointings)):

        print(' ')
        print('>>> processing pointing: ' + str(unique_pointing_num + 1)\
        + '/' + str(len(unique_pointings)))
        print(' ')

        unique_pointings_ID = unique_pointings[unique_pointing_num][-18:]
        sec = unique_pointings[unique_pointing_num]

        if self.dithering_multiple_OBs:
            if self.withrvcorr:
                if self.skysub:
                    combining_exposure_dir_withoutsky = self.combining_OBs_dir\
                    + unique_pointings_ID + '/withoutsky_withrvcorr'
                if not self.skysub:
                    combining_exposure_dir_withsky = self.combining_OBs_dir\
                    + unique_pointings_ID + '/withsky_withrvcorr'
            else:
                combining_exposure_dir = self.combining_OBs_dir\
                + unique_pointings_ID + '/withsky_withoutrvcorr'

        if not self.dithering_multiple_OBs:
            if self.withrvcorr:
                if self.skysub:
                    combining_exposure_dir_withoutsky =\
                    sec + '/withoutsky_withrvcorr'
                if not self.skysub:
                    combining_exposure_dir_withsky =\
                    sec + '/withsky_withrvcorr'

            else:
                combining_exposure_dir = sec + '/withsky_withoutrvcorr'

        if self.withrvcorr:
            if self.skysub:
                pixtable_list = _get_filelist(self,\
                combining_exposure_dir_withoutsky, 'PIXTABLE_REDUCED_*.fits')
                if create_sof:
                    if os.path.exists(combining_exposure_dir_withoutsky\
                    + '/exp_combine.sof'):
                        os.remove(combining_exposure_dir_withoutsky\
                        + '/exp_combine.sof')
                    f = open(combining_exposure_dir_withoutsky\
                    + '/exp_combine.sof', 'w')
                    for i in range(len(pixtable_list)):
                        f.write(combining_exposure_dir_withoutsky\
                        + '/' + pixtable_list[i] + ' PIXTABLE_REDUCED\n')
                    f.write(combining_exposure_dir_withoutsky\
                    + '/' + 'OFFSET_LIST.fits OFFSET_LIST\n')
                    f.write(self.static_calibration_dir\
                    + 'filter_list.fits FILTER_LIST\n')
                    f.close()
                if not self.debug:
                    _call_esorex(self, combining_exposure_dir_withoutsky,\
                    esorex_cmd)

            if not self.skysub:
                pixtable_list = _get_filelist(self,\
                combining_exposure_dir_withsky, 'PIXTABLE_REDUCED_*.fits')
                if create_sof:
                    if os.path.exists(combining_exposure_dir_withsky\
                    + '/exp_combine.sof'):
                        os.remove(combining_exposure_dir_withsky\
                        + '/exp_combine.sof')

                    f = open(combining_exposure_dir_withsky\
                    + '/exp_combine.sof', 'w')
                    for i in range(len(pixtable_list)):
                        f.write(combining_exposure_dir_withsky\
                        + '/' + pixtable_list[i] + ' PIXTABLE_REDUCED\n')
                    f.write(combining_exposure_dir_withsky\
                    + '/' + 'OFFSET_LIST.fits OFFSET_LIST\n')
                    f.write(self.static_calibration_dir\
                    + 'filter_list.fits FILTER_LIST\n')
                    f.close()
                if not self.debug:
                    _call_esorex(self, combining_exposure_dir_withsky,\
                    esorex_cmd)
        else:
            pixtable_list = _get_filelist(self,\
            combining_exposure_dir, 'PIXTABLE_REDUCED_*.fits')
            if create_sof:
                if os.path.exists(combining_exposure_dir + '/exp_combine.sof'):
                    os.remove(combining_exposure_dir + '/exp_combine.sof')
                f = open(combining_exposure_dir + '/exp_combine.sof', 'w')
                for i in range(len(pixtable_list)):
                    f.write(combining_exposure_dir\
                    + '/' + pixtable_list[i] + ' PIXTABLE_REDUCED\n')
                f.write(combining_exposure_dir\
                + '/' + 'OFFSET_LIST.fits OFFSET_LIST\n')
                f.write(self.static_calibration_dir\
                + 'filter_list.fits FILTER_LIST\n')
                f.close()
            if not self.debug:
                _call_esorex(self, combining_exposure_dir, esorex_cmd)
