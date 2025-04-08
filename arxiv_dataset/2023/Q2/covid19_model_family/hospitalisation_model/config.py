"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

import json
import os
import datetime as dt
import shutil


class Config:
    """
    Class which loads and interprets the json input file
    """

    def __init__(self, filename: str) -> None:
        """
        Class which loads and interprets the json input file
        :param filename: full path to the json file
        """

        # things to guarantee reproducibility:
        self.timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(filename, 'r', encoding='utf8') as f:
            self.content = json.load(f)
        self.scenario = self.content['scenario']
        self.resultfolder = self.content['resultFolder'] + '/' + self.scenario + '_' + self.timestamp
        try:
            os.mkdir(self.resultfolder)
        except:
            None
        # copy config file into result folder
        shutil.copy(filename, self.resultfolder + '/config.json')

        # load fields
        self.filename_case_numbers = self.content['filenameCaseNumbers']
        self.filename_hospitalized = self.content['filenameHospitalized']
        self.filename_forecast = self.content['filenameForecast']
        self.columns_case_numbers = self.content['columnsCaseNumbers']
        self.columns_hospitalized = self.content['columnsHospitalized']
        self.columns_forecast = self.content['columnsForecast']
        self.bounds_rate = self.content['bounds']['rate']
        self.bounds_delay = self.content['bounds']['delay']
        self.bounds_stay = self.content['bounds']['stay']
        # rate factors are only on demand
        try:
            self.filename_rate_factors = self.content['filenameRateFactors']
            self.columns_rate_factors = self.content['columnsRateFactors']
        except:
            self.filename_rate_factors = None
            self.columns_rate_factors = None
        self.forecast_day = dt.datetime.strptime(self.content['forecastDay'], '%Y-%m-%d').date()
        # make sure all identifiers have the same length
        self.m = max(len(self.columns_case_numbers), len(self.columns_hospitalized), len(self.columns_forecast))
        if len(self.columns_case_numbers) < self.m:
            self.columns_case_numbers = [self.columns_case_numbers[0] for x in range(self.m)]
        if len(self.columns_hospitalized) < self.m:
            self.columns_hospitalized = [self.columns_hospitalized[0] for x in range(self.m)]
        if len(self.columns_forecast) < self.m:
            self.columns_forecast = [self.columns_forecast[0] for x in range(self.m)]
        if self.columns_rate_factors != None:
            if len(self.columns_rate_factors) < self.m:
                self.columns_rate_factors = [self.columns_rate_factors[0] for x in range(self.m)]
        if len(self.bounds_rate) < self.m:
            self.bounds_rate = [self.bounds_rate[0] for x in range(self.m)]
        if len(self.bounds_delay) < self.m:
            self.bounds_delay = [self.bounds_delay[0] for x in range(self.m)]
        if len(self.bounds_stay) < self.m:
            self.bounds_stay = [self.bounds_stay[0] for x in range(self.m)]
        # same for distributions
        self.admissionDelayDistributions = self.content['admissionDelayDistributions']
        if len(self.admissionDelayDistributions) < self.m:
            self.admissionDelayDistributions = [self.admissionDelayDistributions[0] for x in range(self.m)]
        self.stayDelayDistributions = self.content['stayDelayDistributions']
        if len(self.stayDelayDistributions) < self.m:
            self.stayDelayDistributions = [self.stayDelayDistributions[0] for x in range(self.m)]
        self.calibrationDays = self.content['calibrationDays']
        self.transientDays = self.content['transientDays']
        self.forecastDays = self.content['forecastDays']
        self.start_calibration_day = self.forecast_day - dt.timedelta(self.calibrationDays)
        self.start_transient_day = self.start_calibration_day - dt.timedelta(self.transientDays)
        self.end_forecast_day = self.forecast_day + dt.timedelta(self.forecastDays)
        # if the json file contains a field "parameters", the parameters are executed in the simulation, but not calibrated
        try:
            self.parameters = self.content['parameters']
        except:
            self.parameters = None

    def has_parameters(self) -> bool:
        """
        :return: returns true, if there are parameter values which can be used for simulation
        """
        return self.parameters != None

    def get_result_folder(self) -> str:
        """
        :return: folder to save results into
        """
        return self.resultfolder

    def get_filename_case_numbers(self) -> str:
        """
        :return: filename to load case numbers from
        """
        return self.filename_case_numbers

    def get_filename_hospitalized(self) -> str:
        """
        :return: filename to load reference hospital numbers from
        """
        return self.filename_hospitalized

    def get_filename_forecast(self) -> str:
        """
        :return: filename to load case number forecast from
        """
        return self.filename_forecast

    def get_filename_rate_factors(self) -> str:
        """
        :return: filename to load rate factors from
        """
        return self.filename_rate_factors

    def get_columns_case_numbers(self) -> list[str]:
        """
        :return: columns referring to the case numbers
        """
        return self.columns_case_numbers

    def get_columns_hospitalized(self) -> list[str]:
        """
        :return: columns referring to the reference hospital numbers
        """
        return self.columns_hospitalized

    def get_columns_forecast(self) -> list[str]:
        """
        :return: columns referring to the case number forecast
        """
        return self.columns_forecast

    def get_columns_rate_factors(self) -> list[str]:
        """
        :return: columns referring to the rate factors
        """
        return self.columns_rate_factors

    def get_scenario(self) -> str:
        """
        :return: identifyer for the scenario
        """
        return self.scenario

    def get_column_number(self) -> int:
        """
        Returns the total number of columns regarded in the experiment.
        Each stands for one simulation scenario (e.g. ICU/normalbeds in different regions)
        :return: number of columns
        """
        return self.m

    def get_bounds(self, index) -> list[list]:
        """
        Returns bounds for the calibration of the specific column/scenario
        :param index: index of the column=scenario
        :return: 3 element vector, each a list with lower and upper bound
        """
        return [self.bounds_rate[index], self.bounds_delay[index], self.bounds_stay[index]]

    def get_parameters(self, index) -> list[float]:
        """
        Returns the currently implemented simulation parameters for rate,admission and stay delay.
        None, by default
        :param index: index of the column=scenario
        :return:
        """
        return self.parameters[index]

    def get_admission_delay_distribution(self, index) -> str:
        """
        :param index: index of the column=scenario
        :return: string representing the shape of the distribution of the time between test and admission
        """
        return self.admissionDelayDistributions[index]

    def get_stay_delay_distribution(self, index) -> str:
        """
        :param index: index of the column=scenario
        :return: string representing the shape of the distribution of the time between admission and release
        """
        return self.stayDelayDistributions[index]

    def get_forecast_day(self) -> dt.date:
        """
        :return: last date for which hospital data is available
        """
        return self.forecast_day

    def get_start_calibration_day(self) -> dt.date:
        """
        :return: start date for the calibration
        """
        return self.start_calibration_day

    def get_start_transient_day(self) -> dt.date:
        """
        :return: start date of the transient phase
        """
        return self.start_transient_day

    def get_end_forecast_day(self) -> dt.date:
        """
        :return: last date of the forecast
        """
        return self.end_forecast_day
