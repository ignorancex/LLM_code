import os
import pickle
import warnings

import numpy as np
from shutil import copyfile

from astropy import units as u
from astropy.table import Table, Column

from .kinematic_distance import KinematicDistance


class BayesianDistance(object):
    def __init__(self, filename=None):
        """
        initializes the BayesianDistance class

        Parameters
        ----------
        path_to_bdc : file path to the Bayesian distance program
        path_to_input_table : file path of a table containing information of the
            Gaussian decompositions
        verbose : The default is 'True'. Prints status messages to the
            terminal.
        """
        self.path_to_bdc = None
        self.version = '2.4'
        self.path_to_input_table = None
        self.path_to_output_table = None
        self.input_table = None
        self.verbose = True
        self.add_kinematic_distance = True
        self.add_galactocentric_distance = True
        self.check_for_kda_solutions = True
        self.prior_velocity_dispersion = False
        self.colname_lon, self.colname_lat, self.colname_vel,\
            self.colname_e_vel, self.colname_kda, self.colname_name,\
            self.colname_vel_disp = (None for i in range(7))
        self.colnr_lon, self.colnr_lat, self.colnr_vel,\
            self.colnr_e_vel, self.colnr_kda, self.colnr_name,\
            self.colnr_vel_disp = (None for i in range(7))
        self.prob_sa, self.prob_kd, self.prob_gl, self.prob_ps, self.prob_pm =\
            (None for _ in range(5))
        self.table_format = 'ascii'
        self.save_temporary_files = False
        self.max_e_vel = 5.0
        self.default_e_vel = 5.0
        self.kda_info_tables = []
        self.exclude_kda_info_tables = []
        self.kda_weight = 1

        self.random_seed = 177
        self.sample = 1000
        self.beam = None
        self.size_linewidth_index = 0.5
        self.size_linewidth_e_index = 0.1
        self.size_linewidth_sigma_0 = 0.7
        self.size_linewidth_e_sigma_0 = 0.1

        self.use_ncpus = None
        self.plot_probability = False

        self._p = {
            '1.0': {
                'bdc_fortran': 'Bayesian_distance_v1.0.f',
                'summary_suffix': '.prt',
                'fct_extract': self.extract_results_v1p0,
                'R_0': 8.34},
            '2.4': {
                'bdc_fortran': 'Bayesian_distance_2019_fromlist_v2.4.f',
                'summary_suffix': 'summary.prt',
                'fct_extract': self.extract_results_v2p4,
                'R_0': 8.15}
        }

    def say(self, message, end=None):
        """Diagnostic messages."""
        if self.verbose:
            print(message, end=end)

    def check_settings(self):
        self.initialize_bdc()
        self.initialize_table()
        self.set_probability_controls()
        if self.check_for_kda_solutions:
            self.initialize_kda_tables()
        if self.prior_velocity_dispersion:
            self.initialize_prior_velocity_dispersion()

        text = 'Python wrapper for Bayesian distance calculator v{}'.format(
            self.version)
        border = len(text) * '='
        heading = '\n{a}\n{b}\n{a}\n'.format(a=border, b=text)
        self.say(heading)

    def initialize_bdc(self):
        if self.version is None:
            raise Exception("Need to specify 'version'")

        path_script = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))

        self.path_to_bdc = os.path.join(
            path_script, 'BDC', 'v' + self.version)
        path_to_file = os.path.join(
            self.path_to_bdc, self._p[self.version]['bdc_fortran'])

        with open(path_to_file, "r") as fin:
            self.bdc_script = fin.readlines()

    def initialize_table(self):
        if self.path_to_output_table is not None:
            self.path_to_table = self.path_to_output_table

        if self.path_to_table is None:
            errorMessage = str("specify 'path_to_output_table'")
            raise Exception(errorMessage)

        self.dirname_table = os.path.dirname(self.path_to_table)
        if len(self.dirname_table) == 0:
            self.dirname_table = os.getcwd()
        self.table_file = os.path.basename(self.path_to_table)
        self.table_filename, self.table_file_extension =\
            os.path.splitext(self.table_file)
        if not os.path.exists(self.dirname_table):
            os.makedirs(self.dirname_table)

    def initialize_kda_tables(self):
        dirname = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))
        if not self.kda_info_tables:
            files = os.listdir(os.path.join(dirname, 'KDA_info'))
            self.kda_info_tables = [
                name[:-4] for name in files if name.endswith('.ini')]

        if self.exclude_kda_info_tables:
            self.kda_info_tables = [
                table for table in self.kda_info_tables
                if table not in self.exclude_kda_info_tables]

        self._kda_tables = []
        keys = ['GLON', 'GLAT', 'VLSR', 'd_VLSR', 'p_far',
                'cos_pa', 'sin_pa', 'aa', 'bb']
        for tablename in self.kda_info_tables:
            table = Table.read(os.path.join(
                dirname, 'KDA_info', tablename + '.dat'), format='ascii')
            table = table[keys]

            self._kda_tables.append(table)

    def initialize_prior_velocity_dispersion(self):
        try:
            self.beam = self.beam.to(u.rad).value
        except AttributeError:
            err_msg = "'beam' needs to be specified as valid astropy unit"
            raise Exception(err_msg)

        self.kd = KinematicDistance()
        self.kd.initialize()

        np.random.seed = self.random_seed
        self._indices = self.size_linewidth_index + np.random.randn(
            self.sample) * self.size_linewidth_e_index
        self._sigma_0 = self.size_linewidth_sigma_0 + np.random.randn(
            self.sample) * self.size_linewidth_e_sigma_0

    def set_probability_controls(self):
        s = '      '

        default_vals = {
            '1.0': {'SA': 0.5, 'KD': 1.0, 'GL': 1.0, 'PS': 0.25, 'PM': None},
            '2.4': {'SA': 0.85, 'KD': 0.85, 'GL': 0.85, 'PS': 0.15, 'PM': 0.85}
            }

        if self.prob_sa is None:
            self.prob_sa = default_vals[self.version]['SA']
        if self.prob_kd is None:
            self.prob_kd = default_vals[self.version]['KD']
        if self.prob_gl is None:
            self.prob_gl = default_vals[self.version]['GL']
        if self.prob_ps is None:
            self.prob_ps = default_vals[self.version]['PS']
        if self.prob_pm is None:
            self.prob_pm = default_vals[self.version]['PM']

        cwd = os.getcwd()
        os.chdir(self.path_to_bdc)

        with open(os.path.join(
                self.path_to_bdc, 'probability_controls.inp'), 'r') as fin:
            file_content = fin.readlines()
        with open(os.path.join(
                self.path_to_bdc, 'probability_controls.inp'), 'w') as fout:
            for line in file_content:
                if not line.startswith('!'):
                    line = '{s}{a}{s}{b}{s}{c}{s}{d}'.format(
                        s=s, a=self.prob_sa, b=self.prob_kd, c=self.prob_gl,
                        d=self.prob_ps)
                    if self.prob_pm is not None:
                        line += '{s}{a}'.format(s=s, a=self.prob_pm)
                fout.write(line)
        os.chdir(cwd)

        string = str("prob_sa: {a}\nprob_kd: {b}\n"
                     "prob_gl: {c}\nprob_ps: {d}\n".format(
                         a=self.prob_sa, b=self.prob_kd, c=self.prob_gl,
                         d=self.prob_ps))
        if self.version == '2.4':
            string += 'prob_pm: {}\n'.format(self.prob_pm)
        self.say("setting probability controls to the following values:")
        self.say(string)

    def determine_column_indices(self):
        self.colnr_lon = self.input_table.colnames.index(self.colname_lon)
        self.colnr_lat = self.input_table.colnames.index(self.colname_lat)
        self.colnr_vel = self.input_table.colnames.index(self.colname_vel)
        if self.colname_e_vel is not None:
            if not isinstance(self.colname_e_vel, list):
                self.colname_e_vel = [self.colname_e_vel]
            self.colnr_e_vel = [self.input_table.colnames.index(colname)
                                for colname in self.colname_e_vel]
        if self.colname_kda is not None:
            self.colnr_kda = self.input_table.colnames.index(self.colname_kda)
        if self.colname_vel_disp is not None:
            self.colnr_vel_disp = self.input_table.colnames.index(
                self.colname_vel_disp)
        if self.colname_name is not None:
            self.colnr_name = self.input_table.colnames.index(self.colname_name)

    def make_fortran_out(self, source):
        """Create a fortran executable for the source.

        Replaces the default input file in the fortran script of the Bayesian
        distance calculator with the input file of the source, then creates a
        Fortran executable file.
        """
        with open("{}.f".format(self.path_to_source), "w") as fout:
            for line in self.bdc_script:
                fout.write(line.replace('sources_info.inp',
                                        '{}_sources_info.inp'.format(source)))
        os.system('gfortran {}.f -o {}.out'.format(
                self.path_to_source, self.path_to_source))

    def extract_string(self, s, first, last, incl=False):
        """Search for a substring inside a string.

        Parameters
        ----------
        s : string that is searched for the substring
        first : first characters of the substring
        last : last characters of the substring
        incl : defines if the `first` and `last` characters are still part of
            the substring that will be returned. The default is `False`
            (`first` and `last` are not part of the returned substring)

        Returns
        -------
        substring of s

        """
        try:
            if incl is True:
                start = s.index(first)
                end = s.index(last) + len(last)
            else:
                start = s.index(first) + len(first)
                end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""

    def extract_probability_info(self, line, lon, lat, p_far):
        """
        Extract the distance results from the corresponding string in
        the output file of the Bayesian distance calculator tool.
        """
        deleteString = self.extract_string(
                line, 'Probability component', ':', incl=True)
        replaceString = self.extract_string(
                line, 'Probability component', ':')
        line = line.replace(deleteString, replaceString)
        line = line.replace('\n', '')
        comp, dist, err, prob, arm = line.split()
        c_u, c_v, c_w = self.get_cartesian_coords(lon, lat, float(dist))
        # if np.isnan(dist) is True:
        #     dist, err, prob = (0.0 for i in range(3))
        return [comp, dist, err, prob, arm, c_u, c_v, c_w, p_far]

    def extract_results_v1p0(self, input_file_content, result_file_content,
                             kin_dist=None, kda_ref=None):
        """
        Loop through the lines of the output file of the Bayesian distance
        calculator tool and search for the distance results.

        Parameters
        ----------
        result_file_content : List containing read-in lines of the output file
            ({source_name}.prt) of the Bayesian distance calculator tool
        """
        results = []
        flag = False
        for line in result_file_content:
            if flag:
                params = line.split()
                lon, lat, p_far =\
                    float(params[1]), float(params[2]), float(params[4])
                flag = False
            if 'Extra_info' in line:
                flag = True
            searchString = 'Probability component'
            if searchString in line:
                result = self.extract_probability_info(line, lon, lat, p_far)

                if kda_ref is not None:
                    result += [kda_ref]

                if kin_dist is not None:
                    result += kin_dist

                results.append(result)
        return results

    def extract_kinematic_distances(self, result_file_content):
        """"""
        kinDist = [np.NAN, np.NAN]

        flag = 'one'
        for line in result_file_content:
            searchString = 'Kinematic distance(s):'
            if searchString in line:
                if flag == 'one':
                    kinDist[0] = self.extract_kinematic_info(line)
                    flag = 'two'
                elif flag == 'two':
                    kinDist[1] = self.extract_kinematic_info(line)
        return kinDist

    def extract_kinematic_info(self, line):
        """
        Extract the distance results from the corresponding string in
        the output file of the Bayesian distance calculator tool.
        """
        line = line.replace('Kinematic distance(s):', '')
        line = line.replace('\n', '')
        return float(line)

    def extract_results_v2p4(self, input_file_content, result_file_content,
                             kin_dist=None, kda_ref=None):
        for line in input_file_content:
            if line.startswith('!'):
                continue
            params = line.split()
            p_far = params[5]

        for line in result_file_content:
            if line.startswith('!'):
                continue
            params = line.split()

            n_params = len(params)

            lon, lat, vlsr, e_vlsr = params[:4]

            results = []

            for i in range(1, int(n_params / 4)):
                comp = int(n_params / 4) - 1
                dist, e_dist, prob, arm = params[i*4:(i + 1)*4]
                c_u, c_v, c_w = self.get_cartesian_coords(
                    float(lon), float(lat), float(dist))

                result = [comp, dist, e_dist, prob, arm, c_u, c_v, c_w, p_far]

                if kda_ref is not None:
                    result += [kda_ref]

                if kin_dist is not None:
                    result += kin_dist

                results.append(result)
        return results

    def delete_all_temporary_files(self, source):
        for filename in [f for f in os.listdir(self.path_to_bdc) if f.startswith(source)]:
            os.remove(os.path.join(self.path_to_bdc, filename))

    def get_results(self, source, kda_ref=None, name=None):
        """
        Extract the distance results from the output file ({source_name}.prt)
        of the Bayesian distance calculator tool.
        """
        suffix = self._p[self.version]['summary_suffix']
        for filename in [f for f in os.listdir(self.path_to_bdc)
                         if f.startswith(source) and f.endswith(suffix)]:
            with open(os.path.join(self.path_to_bdc, filename), 'r') as fin:
                result_file_content = fin.readlines()

        for filename in [f for f in os.listdir(self.path_to_bdc)
                         if f.startswith(source) and f.endswith("info.inp")]:
            with open(os.path.join(self.path_to_bdc, filename), 'r') as fin:
                input_file_content = fin.readlines()

        if self.add_kinematic_distance:
            if self.version == '1.0':
                kd_content = result_file_content.copy()
            elif self.version == '2.4':
                with open(os.path.join(self.path_to_bdc, source + '.prt'), 'r') as fin:
                    kd_content = fin.readlines()
            kinDist = self.extract_kinematic_distances(kd_content)
        else:
            kinDist = None

        results = self._p[self.version]['fct_extract'](
            input_file_content, result_file_content, kin_dist=kinDist, kda_ref=kda_ref)

        if self.plot_probability:
            if self.save_temporary_files:
                for filename in [f for f in os.listdir(self.path_to_bdc) if f.startswith(source)]:
                    src = os.path.join(self.path_to_bdc, filename)
                    if name is not None:
                        filename = filename.replace(source, name)
                    dst = os.path.join(
                        os.path.dirname(self.path_to_output_table), filename)
                    copyfile(src, dst)

            self.plot_probability_density(
                source, results, input_file_content, name=name)

        self.delete_all_temporary_files(source)

        return results

    def run_bdc_script(self, source, input_string):
        self.path_to_source = os.path.join(self.path_to_bdc, source)
        filepath = '{}_sources_info.inp'.format(self.path_to_source)
        with open(filepath, 'w') as fin:
            fin.write(input_string)
        self.make_fortran_out(source)
        cwd = os.getcwd()
        os.chdir(self.path_to_bdc)
        os.system('./{}.out'.format(source))
        os.chdir(cwd)

    def bdc_calculation_ok(self, source):
        """Check if BDC yielded any distance results."""
        suffix = self._p[self.version]['summary_suffix']
        for filename in [f for f in os.listdir(self.path_to_bdc)
                         if f.startswith(source) and f.endswith(suffix)]:
            with open(os.path.join(self.path_to_bdc, filename), 'r') as fin:
                result_file_content = fin.readlines()
        for line in result_file_content:
            if not line.startswith('!'):
                return True

        return False

    def determine_e_vel(self, row):
        """Determine uncertainty for vlsr value."""
        e_vel = None

        if self.colnr_e_vel is not None:
            e_vel = max([row[colnr] for colnr in self.colnr_e_vel])
            if abs(float(e_vel)) > self.max_e_vel:  # abs(float(vel)):
                e_vel = None

        if self.version == '1.0':
            plusminus = ''
        elif self.version == '2.4':
            # TODO: implement minimum error for velocity
            if e_vel is not None:
                plusminus = '{}\t'.format(e_vel)
            else:
                plusminus = '{}\t'.format(self.default_e_vel)

        return plusminus

    def determine_p_far_and_kda_ref(self, row, lon, lat, vel):
        """Determine KDA prior and corresponding literature reference."""
        p_far = 0.5
        kda_ref = None

        if self.colnr_kda is not None:
            if row[self.colnr_kda] == 'F':
                p_far = 0.5 + 0.5 * self.kda_weight
            elif row[self.colnr_kda] == 'N':
                p_far = 0.5 - 0.5 * self.kda_weight
            elif isinstance(row[self.colnr_kda], float):
                warnings.warn(
                    "KDA solutions need to be given as strings ('N', 'F')")
                p_far = row[self.colnr_kda] * self.kda_weight
        elif self.check_for_kda_solutions:
            p_far, kda_ref = self.check_KDA(lon, lat, vel)

        return round(p_far, 2), kda_ref

    def get_expected_vel_disp(self, distance):
        size = self.beam * distance * 1e3
        sigma_exp = self._sigma_0 * (size)**(self._indices)
        return np.mean(sigma_exp), np.std(sigma_exp)

    def normalized_gauss(self, mean, sigma, x):
        return np.exp(-0.5 * ((x - mean) / sigma)**2)

    def check_limit(self, mean, sigma, vel_disp, prob, limit=0.01):
        limit_prob = None
        if prob < limit:
            if (vel_disp - mean) > 0:
                limit_prob = 'higher'
            else:
                limit_prob = 'lower'
        return limit_prob

    def determine_pfar_from_vel_disp(self, dist_n, dist_f, vel_disp):
        sigma_exp_mean, sigma_exp_std = self.get_expected_vel_disp(dist_n)
        prob_near = self.normalized_gauss(
            sigma_exp_mean, sigma_exp_std, vel_disp)
        limit_n = self.check_limit(
            sigma_exp_mean, sigma_exp_std, vel_disp, prob_near)

        sigma_exp_mean, sigma_exp_std = self.get_expected_vel_disp(dist_f)
        prob_far = self.normalized_gauss(
            sigma_exp_mean, sigma_exp_std, vel_disp)
        limit_f = self.check_limit(
            sigma_exp_mean, sigma_exp_std, vel_disp, prob_far)

        if (limit_f == 'lower') and (limit_n != 'higher'):
            pfar = 0
        else:
            pfar = (-prob_near + prob_far) * 0.5 + 0.5

        return pfar

    def determine_p_far_from_velocity_dispersion(self, row, lon, lat, vel):
        dist_n, dist_f = self.kd.calc_kinematic_distance(lon, lat, vel)
        vel_disp = row[self.colnr_vel_disp]
        p_far = self.determine_pfar_from_vel_disp(dist_n, dist_f, vel_disp)
        return round(p_far, 2)

    def determine(self, row, idx):
        """Determine distance of lbv data point via the BDC."""
        row = list(row)

        source = "SRC{}".format(str(idx).zfill(9))
        lon, lat, vel =\
            row[self.colnr_lon], row[self.colnr_lat], row[self.colnr_vel]

        name = None
        if self.colnr_name is not None:
            name = row[self.colnr_name]

        plusminus = self.determine_e_vel(row)
        p_far, kda_ref = self.determine_p_far_and_kda_ref(row, lon, lat, vel)
        condition = ((p_far == 0.5) and
                     self.prior_velocity_dispersion and
                     (self.colnr_vel_disp is not None))
        if condition:
            p_far = self.determine_p_far_from_velocity_dispersion(
                row, lon, lat, vel)

        input_string = "{a}\t{b}\t{c}\t{d}\t{e}{f}\t-\n".format(
            a=source, b=lon, c=lat, d=vel, e=plusminus, f=p_far)

        self.run_bdc_script(source, input_string)

        #  rerun BDC calculation with p_far = 0.5 if chosen p_far value did not yield distance results
        if (self.version == '2.4') and (p_far != 0.5):
            if not self.bdc_calculation_ok(source):
                self.delete_all_temporary_files(source)

                p_far = 0.5
                input_string = "{a}\t{b}\t{c}\t{d}\t{e}{f}\t-\n".format(
                    a=source, b=lon, c=lat, d=vel, e=plusminus, f=p_far)
                self.run_bdc_script(source, input_string)

        rows = []
        results = self.get_results(source, kda_ref=kda_ref, name=name)
        for result in results:
            rows.append(row + result)

        return rows

    def get_values_from_init_file(self, init_file):
        """Read in values from init file."""
        import ast
        import configparser
        config = configparser.ConfigParser()
        config.read(init_file)

        for key, value in config['DEFAULT'].items():
            try:
                setattr(self, '_' + key, ast.literal_eval(value))
            except ValueError:
                raise Exception('Could not parse parameter {} from config file'.format(key))

    def gaussian_weight(self, x, std=False):
        """Calculate the Gaussian weight.

        Gaussian function: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)
        mean = 0

        Renormalization factor for amplitude, so that Gaussian function is 1 at the fwhm/2; scale = 1 / (np.exp(-np.log(2)) = np.exp(np.log(2)
        fwhm_factor = 2 * np.sqrt(2 * np.log(2)) = 2.354820045
        """
        if std:
            return np.exp((1 - x**2) / 2)  # = np.exp(0.5) * np.exp(-4. * np.log(2) * (x / 2.354820045)**2)
        else:
            return np.exp(np.log(2) * (1 - 4. * x**2))  # np.exp(np.log(2)) * np.exp(-4. * np.log(2) * x**2)

    def point_in_ellipse(self, table, lon, lat):
        """Adapted from: https://stackoverflow.com/questions/7946187/
        See also: https://math.stackexchange.com/questions/426150/"""
        cos_pa = table['cos_pa'].data
        sin_pa = table['sin_pa'].data
        glon = table['GLON'].data
        glat = table['GLAT'].data
        aa = table['aa'].data
        bb = table['bb'].data

        a = (cos_pa * (lon - glon) + sin_pa * (lat - glat))**2
        b = (sin_pa * (lon - glon) - cos_pa * (lat - glat))**2
        epsilon = (a / aa) + (b / bb)

        std = False
        if self._size == 'std':
            std = True
        else:
            epsilon = epsilon / 4

        weight = self.gaussian_weight(np.sqrt(epsilon), std=std)
        weight[weight > 1] = 1

        return weight >= self._threshold_spatial, weight

    def get_weight_velocity(self, table, vel):
        """Calculate the weight for the velocity association.

        Parameters
        ----------
        table : astropy.table.table.Table
            Table containing sources with solved kinematic distance ambiguities.
        vel : float
            vlsr position of the coordinate.

        Returns
        -------
        mask : numpy.ndarray
            Mask that is true for each weight that exceeds _threshold_spectral.
        weight : numpy.ndarray
            Array of the weight values.

        """
        vlsr = table['VLSR'].data.data
        dvlsr = table['d_VLSR'].data.data

        x = np.abs(vlsr - vel) / dvlsr

        # fwhm_factor = 2.354820045
        # if fwhm:
        #     x = np.abs((vlsr - vel) / (dvlsr))
        # else:
        #     x = np.abs((vlsr - vel) / (dvlsr)) / (2 * fwhm_factor)

        std = False
        if self._linewidth == 'std':
            std = True

        weight = self.gaussian_weight(x, std=std)
        weight[weight > 1] = 1

        return weight >= self._threshold_spectral, weight

    def get_kda(self, weights_kda, refs):
        if len(weights_kda) == 0:
            return 0, '--'

        if len(weights_kda) == 1:
            return weights_kda[0], refs[0]

        weights_kda_abs = [abs(x) for x in weights_kda]
        max_weight = max(weights_kda_abs)

        indices = np.argwhere(weights_kda_abs == max_weight).flatten().tolist()

        if len(indices) == 1:
            i = indices[0]
            return weights_kda[i], refs[i]

        if sum(weights_kda[indices]) == 0:
            return 0, '--'
        else:
            list_weights_kda = weights_kda[indices].tolist()
            weight = max(list_weights_kda, key=list_weights_kda.count)
            i = np.argwhere(weights_kda == weight).flatten()[0]
            return weight, refs[i]

    def check_KDA(self, lon, lat, vel):
        weights_kda, refs = np.array([]), []
        dirname = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))

        for table, tablename in zip(self._kda_tables, self.kda_info_tables):
            path_to_ini = os.path.join(dirname, 'KDA_info', tablename + '.ini')
            self.get_values_from_init_file(path_to_ini)
            mask_pp, weight_pp = self.point_in_ellipse(table, lon, lat)
            mask_vlsr, weight_vlsr = self.get_weight_velocity(table, vel)

            mask_total = np.logical_and(mask_pp, mask_vlsr)
            weight_total = weight_pp * weight_vlsr
            weight_total = weight_total[mask_total]
            p_far_values = table['p_far'].data
            p_far_values = p_far_values[mask_total]

            n_values = np.count_nonzero(mask_total)
            if n_values == 0:
                continue
            elif n_values == 1:
                weights_kda = np.append(
                    weights_kda, self._weight_cat * p_far_values * weight_total)
                refs.append(self._reference)
            else:
                weights_kda = np.append(weights_kda, self._weight_cat * (np.average(
                    p_far_values * weight_total, weights=weight_total)))
                refs.append(self._reference)

        weight_kda, ref = self.get_kda(weights_kda, refs)
        p_far = 0.5 + weight_kda

        return round(float(p_far), 2), ref

    def get_cartesian_coords(self, lon, lat, dist):
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        c = SkyCoord(l=lon*u.degree,
                     b=lat*u.degree,
                     distance=dist*u.kpc,
                     frame='galactic')
        c.representation_type = 'cartesian'
        c_u = round(c.u.value, 4)
        c_v = round(c.v.value, 4)
        c_w = round(c.w.value, 4)

        return c_u, c_v, c_w

    def calculate_distances(self):
        self.check_settings()
        self.say('calculating Bayesian distance...')

        if self.input_table is None:
            self.input_table = Table.read(
                self.path_to_input_table, format=self.table_format)
            #  TESTING:
            # self.input_table = self.input_table[62000:62001]
        self.determine_column_indices()

        condition = (self.prior_velocity_dispersion and
                     (self.colnr_vel_disp is None))
        if condition:
            self.colnr_vel_disp = False
            warnings.warn(str("Did not specify 'colnr_vel_disp' or 'colname_vel_disp'. Setting 'prior_velocity_dispersion=False'."))

        from . import BD_multiprocessing
        BD_multiprocessing.init([self, self.input_table])
        results_list = BD_multiprocessing.func(use_ncpus=self.use_ncpus)
        print('SUCCESS\n')

        for i, item in enumerate(results_list):
            if not isinstance(item, list):
                self.say("Error for distance with index {}: {}".format(i, item))
                del results_list[i]
                continue

        results_list = np.array([item for sublist in results_list
                                 for item in sublist])

        if self.save_temporary_files:
            filepath = os.path.join(
                os.path.dirname(self.path_to_table),
                '_bdc_results_list.pickle')
            with open(filepath, 'wb') as p_file:
                pickle.dump(results_list, p_file)

        self.create_astropy_table(results_list)

    def galactocentric_distance(self, glon, dist_los, glat=None):
        """Calculate galactocentric distance.

        Parameters
        ----------
        glon : float [radians]
            Galactic longitude angle of the line of sight. Has to be supplied in [radians].
        dist_los : float [kpc]
            Distance along the line of sight. Has to be supplied in [kpc].
        glat : float [radians]
            Galactic latitude angle of the line of sight. Has to be supplied in [radians].

        Returns
        -------
        Galactocentric distance in [kpc].

        """
        if glat is not None:
            dist_los = dist_los * np.cos(glat)
        R_0 = self._p[self.version]['R_0']
        return np.sqrt(R_0**2 + dist_los**2 - 2*R_0*dist_los*np.cos(glon))

    def create_astropy_table(self, results):
        self.say('creating Astropy table...')

        added_colnames = ['comp', 'dist', 'e_dist', 'prob', 'arm',
                          'c_u', 'c_v', 'c_w', 'p_far']

        dtypeinput_table = []
        for name, dtype in self.input_table.dtype.descr:
            dtypeinput_table.append(dtype)
        added_dtype = ['i4', 'f4', 'f4', 'f4', 'object',
                       'f4', 'f4', 'f4', 'f4']

        if self.check_for_kda_solutions and (self.colname_kda is None):
            added_colnames += ['KDA_ref']
            added_dtype += ['object']
        if self.add_kinematic_distance:
            added_colnames += ['kDist_1', 'kDist_2']
            added_dtype += ['f4', 'f4']

        names = self.input_table.colnames + added_colnames
        dtype = dtypeinput_table + added_dtype

        self.table_results = Table(data=results, names=names, dtype=dtype)

        if self.add_galactocentric_distance:
            rgal = self.galactocentric_distance(
                np.radians(self.table_results[self.colname_lon].data),
                self.table_results['dist'].data,
                glat=np.radians(self.table_results[self.colname_lat].data))
            self.table_results.add_column(Column(data=rgal, name='rgal'))

        for key in ['c_u', 'c_v', 'c_w', 'rgal']:
            if key in self.table_results.colnames:
                self.table_results[key].format = "{0:.3f}"
        for key in ['dist', 'e_dist', 'prob', 'p_far', 'kDist_1', 'kDist_2']:
            if key in self.table_results.colnames:
                self.table_results[key].format = "{0:.2f}"

        self.say(">> saved table '{}' in {}\n".format(
                 self.table_file, self.dirname_table))

        self.table_results.write(self.path_to_table, format=self.table_format,
                                 overwrite=True)

    def choose_distance(self, probabilities, distances, dist_errors):
        """Choose distance from alternative solutions.

        Flags for the chosen distance:
        - 0: only 1 distance solution existed
        - 1: only distance solution for which associated Gaussian fit had amplitude above three standard deviations of flat distance probability density
        - 2: distance solution had the highest probability
        - 3: distance solutions were tied in their probabilites; chosen distance had the lowest distance error
        - 4: distance solutions were tied in their probabilites and distance errors; chosen distance is the near distance

        fwhm_factor = 2 * np.sqrt(2 * np.log(2)) = 2.354820045

        Calculate the integrated area of the Gaussian function:
        area_gauss = amp * fwhm / ((1. / np.sqrt(2*np.pi)) * 2*np.sqrt(2*np.log(2)))

        combining all constants yields a factor of 0.93943727869965132
        """
        if len(probabilities) == 1:
            return [], 0

        #  check if one of the components had a probability of 1; this implies
        #  that the remaining components have a probability of zero. This can #  happen as v2.4 of the BDC by default always returns two components
        if 1 in probabilities:
            remove = np.where(probabilities != 1)[0]
            return remove, 0

        #  to get from integrated intensity (= probabilities) and std (= dist_errors) to amplitude
        amps = probabilities * 0.93943727869965132 / (2.354820045 * dist_errors)
        remove = np.where(amps < 3 * 0.04)[0]
        if len(remove) == 1:
            return remove, 1

        remove = np.where(probabilities == min(probabilities))[0]
        if len(remove) == 1:
            return remove, 2

        remove = np.where(dist_errors == max(dist_errors))[0]
        if len(remove) == 1:
            return remove, 3

        remove = np.argmax(distances)
        return remove, 4

    def get_table_distance_max_probability(self, save=True):
        from tqdm import tqdm
        self.say('creating Astropy table containing only distance results '
                 'with the highest probability...')

        remove_rows, choice_flags = np.array([]), np.array([])

        if self.version == '1.0':
            for idx, component in tqdm(enumerate(self.table_results['comp'])):
                if idx == 0:
                    comps_indices = np.array([idx])
                else:
                    if (component == 1):
                        if comps_indices.size > 1:
                            sort_indices_highest_probability = np.argsort(
                                self.table_results['prob'][comps_indices])[::-1]
                            remove = sort_indices_highest_probability[1:]
                            remove_rows = np.append(remove_rows, comps_indices[remove])
                        comps_indices = np.array([idx])
                    else:
                        comps_indices = np.append(comps_indices, idx)

            #  take care of the last distance results in the list
            sort_indices_highest_probability = np.argsort(
                self.table_results['prob'][comps_indices])[::-1]
            remove = sort_indices_highest_probability[1:]
            remove_rows = np.append(remove_rows, comps_indices[remove])
        elif self.version == '2.4':
            comps_indices = np.array([], dtype='int')

            for idx, component in tqdm(enumerate(self.table_results['comp'])):
                comps_indices = np.append(comps_indices, idx)

                if len(comps_indices) == component:
                    remove, flag = self.choose_distance(
                        self.table_results['prob'][comps_indices],
                        self.table_results['dist'][comps_indices],
                        self.table_results['e_dist'][comps_indices]
                        )
                    remove_rows = np.append(remove_rows, comps_indices[remove])
                    choice_flags = np.append(choice_flags, flag)
                    comps_indices = np.array([], dtype='int')

        remove_rows = remove_rows.astype(int)
        self.table_results.remove_rows(remove_rows)

        if self.version == '2.4':
            self.table_results.add_column(
                Column(data=choice_flags, name='flag', dtype='int'))

        if save:
            self.table_file = '{}{}{}'.format(self.table_filename, '_p_max',
                                              self.table_file_extension)
            self.path_to_table = os.path.join(
                self.dirname_table, self.table_file)

            self.say(">> saved table '{}' in {}".format(
                     self.table_file, self.dirname_table))

            self.table_results.write(self.path_to_table,
                                     format=self.table_format,
                                     overwrite=True)

    def find_index_max_probability(self, indices, arm=False):
        idx = [i for i in indices]
        prob = [self.table['prob'][i] for i in indices]
        max_idx = prob.index(max(prob))

        if arm:
            arms = [self.table['arm'][i] for i in indices]
            return idx[max_idx], arms[max_idx]
        else:
            return idx[max_idx]

    def order_distances(self, results):
        indices = list(range(len(results)))

        distances = np.array([float(result[1]) for result in results])
        dist_errors = np.array([float(result[2]) for result in results])
        probabilities = np.array([float(result[3]) for result in results])

        remove, _ = self.choose_distance(probabilities, distances, dist_errors)

        first_choice = [i for i in indices if i not in remove]
        choices = first_choice + remove.tolist()
        results = [results[i] for i in choices]
        return results

    def plot_probability_density(self, source, results, input_file_content,
                                 name=None):
        import matplotlib.pyplot as plt

        def get_maximum_distance(distance, probability, max_dist=None):
            try:
                prob_threshold = 0.05
                distance_cutoff = distance[probability > prob_threshold][-1]
                return max(max_dist, int(distance_cutoff + 1))
            except IndexError:
                return max_dist

        distKD, probKD = np.loadtxt(
            os.path.join(self.path_to_bdc, '{}_kinematic_distance_pdf.dat'.format(source)),
            usecols=(0, 1), skiprows=2, unpack=True)

        if self.version == '1.0':
            distGL, probGL = np.loadtxt(
                os.path.join(self.path_to_bdc, '{}_latitude_pdf.dat'.format(source)),
                usecols=(0, 1), skiprows=2, unpack=True)

            distSA, probSA = np.loadtxt(
                os.path.join(self.path_to_bdc, '{}_spiral_arm_pdf.dat'.format(source)),
                usecols=(0, 1), skiprows=2, unpack=True)
        elif self.version == '2.4':
            distSA, probSA = np.loadtxt(
                os.path.join(self.path_to_bdc, '{}_arm_latitude_pdf.dat'.format(source)),
                usecols=(0, 1), skiprows=2, unpack=True)

            distGL, probGL = distSA, probSA

        arm_ranges_lower, arm_ranges_upper = np.loadtxt(
            os.path.join(self.path_to_bdc, '{}_arm_ranges.dat'.format(source)),
            usecols=(0, 1), skiprows=2, unpack=True)

        spiral_arms = np.genfromtxt(
            os.path.join(self.path_to_bdc, '{}_arm_ranges.dat'.format(source)),
            skip_header=2, usecols=2, dtype='str')

        skip_spiral_arm_ranges = False
        if spiral_arms.size == 2:
            arm_ranges_lower = [arm_ranges_lower[1]]
            arm_ranges_upper = [arm_ranges_upper[1]]
            spiral_arms = [spiral_arms[1]]
        elif spiral_arms.size > 2:
            arm_ranges_lower = arm_ranges_lower[1:]
            arm_ranges_upper = arm_ranges_upper[1:]
            spiral_arms = spiral_arms[1:]
        else:
            skip_spiral_arm_ranges = True

        distPS, probPS = np.loadtxt(
            os.path.join(self.path_to_bdc, '{}_parallaxes_pdf.dat'.format(source)),
            usecols=(0, 1), skiprows=2, unpack=True)

        distFD, probFD = np.loadtxt(
            os.path.join(self.path_to_bdc, '{}_final_distance_pdf.dat'.format(source)),
            usecols=(0, 1), skiprows=2, unpack=True)

        max_dist = 0
        for dist, prob in zip(
                [distKD, distGL, distSA, distPS], [probKD, probGL, probSA, probPS]):
            max_dist = get_maximum_distance(dist, prob, max_dist=max_dist)

        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('Distance [kpc]', size=20)
        ax.set_ylabel('Probability density [kpc$^{-1}$]', size=20)

        ax.tick_params(axis='both', labelsize=16, pad=8)
        ax.tick_params(axis='both', which='major', direction='out',
                       width=1.25, length=10, pad=8)
        ax.tick_params(axis='both', which='minor', direction='out',
                       width=1.25, length=5)

        ax.plot(distKD, probKD, label='KD', lw=2.5, ls='solid', c='dodgerblue', alpha=0.75)
        ax.plot(distGL, probGL, label='GL', lw=2.5, ls='dotted', c='orange', alpha=1.0)
        ax.plot(distSA, probSA, label='SA', lw=2.5, ls='--', c='indianred', alpha=1.0)
        ax.plot(distPS, probPS, label='PS', lw=2.5, ls='-.', c='forestgreen', alpha=1.0)
        ax.plot(distFD, probFD, label='combined', lw=2, ls='solid', c='black', alpha=1.0)

        if not skip_spiral_arm_ranges:
            for lower, upper, text in zip(
                    arm_ranges_lower, arm_ranges_upper, spiral_arms):
                if lower > max_dist:
                    continue
                horizontalalignment = 'center'
                if text == 'AqR':
                    horizontalalignment = 'left'
                ax.axvspan(lower, upper, alpha=0.15, color='indianred')
                ax.text((upper + lower)/2, ax.get_ylim()[1] * 0.99, text, size=14, color='indianred', horizontalalignment=horizontalalignment, verticalalignment='top')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         box.width, box.height * 0.9])
        # Put a legend below current axis
        leg1 = ax.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.08),
            fancybox=False, shadow=False, ncol=5,
            fontsize=14, numpoints=1, frameon=0)

        markers, texts = [], []

        if self.version == '2.4':
            results = self.order_distances(results)
        for i, result in enumerate(results):
            dist = float(result[1])
            if dist <= 0:
                continue
            e_dist = float(result[2])
            prob = float(result[3])
            index = 'D$_{{\\mathregular{{{}}}}}$'.format(i + 1)
            text = '{a}={b:.1f}$\\pm${c:.1f} kpc ({d:.0%})'.format(
                a=index, b=dist, c=e_dist, d=prob)
            marker = ax.scatter(dist, 0 - i*0.01)
            markers.append(marker)
            texts.append(text)
            ax.errorbar(dist, 0 - i*0.01, xerr=e_dist)

        leg2 = ax.legend(
            markers, texts,
            loc='upper center', bbox_to_anchor=(0.5, 1.135),
            fancybox=False, shadow=False, ncol=len(results),
            fontsize=14, numpoints=1, frameon=0)

        for line in input_file_content:
            if line.startswith('!'):
                continue
            params = line.split()
            glon, glat, vlsr, e_vlsr, p_far = params[1:6]
            break

        text = str('$\\ell$={} deg, $b$={} deg, '
                   'V$_{{\\mathregular{{LSR}}}}$={} $\\pm$ {} km/s, '
                   'P$_{{\\mathregular{{far}}}}$={}'. format(
                       glon, glat, vlsr, e_vlsr, p_far))
        plt.title(text, fontsize=14, pad=50)

        ax.set_xlim([0, max_dist])

        ax.add_artist(leg1)

        if name is not None:
            source = name

        path_to_file = os.path.join(self.dirname_table, source + '.pdf')
        plt.savefig(path_to_file, bbox_inches='tight')
        plt.close()
