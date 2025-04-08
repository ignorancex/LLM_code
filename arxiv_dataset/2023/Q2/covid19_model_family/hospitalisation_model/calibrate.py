"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

import numpy as np
from scipy.optimize import minimize

import plot_scripts
from case_numbers import CaseNumbers
from config import Config
from hospital_numbers import HospitalNumbers
import datetime as dt

from hospital_parameters import HospitalParameters
from hospital_sim import HospitalSim
from rate_factors import RateFactors
from utils import TimeSeries


class Calibrate:
    """
    Class to calibrate parameters to given hospital data
    """

    def __init__(self, config: Config) -> None:
        """
        Class to calibrate parameters to given hospital data
        :param config: config instance
        """
        self.config = config
        self.cases = CaseNumbers(self.config)
        self.hospitals = HospitalNumbers(self.config)
        self.t0transient = self.config.get_start_transient_day()
        self.t0calib = self.config.get_start_calibration_day()
        self.tendcalib = self.config.get_forecast_day()
        self.times = [self.t0calib]
        while self.times[-1] < self.tendcalib:
            self.times.append(self.times[-1] + dt.timedelta(1))
        self.reference = TimeSeries([], [])
        self.input = TimeSeries([], [])
        self.rateFactors = RateFactors(config)

    def calibrate(self) -> None:
        """
        Method calibrates the parameters for all columns/scenarios specified in the config
        The optimal parameters are directly written into the config instance
        :return:
        """
        self.config.parameters = list()  # reset params in config
        for i in range(self.config.get_column_number()):
            self.reference = self.hospitals.get_beds(i)
            self.rf = self.rateFactors.get_factors(i)
            self.input = self.cases.get_cases(i)
            bounds = self.config.get_bounds(i)
            x0 = np.array([0.5 * (x + y) for x, y in bounds])
            # nelder mead method to minimize the error
            x = minimize(lambda p: self.calib_fun(p, i), x0=x0, bounds=bounds, method='nelder-mead').x
            params = HospitalParameters()
            params.set(x)
            # print optimal parameters
            print('calibration result for ' + self.config.get_columns_hospitalized()[i])
            print(params)
            # plot_scripts.plot_calibration(self.config,self.reference,beds,i,self.config.get_columns_hospitalized()[i])
            self.config.parameters.append(x)

    def calib_fun(self, p: np.array, i: int) -> float:
        """
        Runs the simulation and computes the error to the reference
        :param p: 3 element array containing the current parameters
        :param i: index of the column/scenario
        :return: sum of squared errors between simulation result and reference
        """
        params = HospitalParameters()
        params.set(p)
        sim = HospitalSim(self.input, params, self.config.get_admission_delay_distribution(i),
                          self.config.get_stay_delay_distribution(i), self.rf)
        [beds, admissions, releases] = sim.run(self.t0transient, self.tendcalib, self.tendcalib,
                                               self.reference.get_value(self.tendcalib))
        err = 0
        for t in self.times:
            err += (self.reference.get_value(t) - beds.get_value(t)) ** 2
        return err
