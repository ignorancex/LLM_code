"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

from delays import Delays
from hospital_parameters import HospitalParameters
from utils import TimeSeries
import datetime as dt


class HospitalSim:
    """
    Class to simulate hospitals given parameters for admission rate and admission and stay delay
    """

    def __init__(self, case_numbers: TimeSeries, parameters: HospitalParameters, admissionDelayDistribution: str,
                 stayDelayDistribution: str, rateFactors=None) -> None:
        """
        Class to simulate hospitals given parameters for admission rate and admission and stay delay
        :param case_numbers: time series of case numbers
        :param parameters: object containing the free parameters of the simulation
        :param admissionDelayDistribution: bound parameter specified how the admission delay is distributed
        :param stayDelayDistribution: bound parameter specified how the stay delay is distributed
        :param rateFactors: optional, time series of rate factors to multiply the hospitalisation rate with
        """
        self.case_numbers = case_numbers
        self.rate_factors = rateFactors
        self.admissionDelays = Delays(admissionDelayDistribution)
        self.stayDelays = Delays(stayDelayDistribution)
        self.parameters = parameters

    def run(self, startDate: dt.date, endDate: dt.date, referenceDate: dt.date, referenceBeds: float) -> list[
        TimeSeries]:
        """
        Runs the simulation from startddate to enddate.
        Afterwards, a linear transformation makes sure that the simulation result at date "referenceTime" matches the cases specfied in "referenceBeds"
        :param startDate: startdate of the simulation
        :param endDate: endDate of the simulation
        :param referenceDate: date to match the beds
        :param referenceBeds: occupancy at the reference date
        :return: returns a 3 element list containing time series for occupied beds, admissions and releases
        """
        t = startDate
        # initialise result
        beds = TimeSeries([0], [t])
        admissions = TimeSeries([], [])
        releases = TimeSeries([], [])
        # time loop
        while t < endDate:
            currentBeds = beds.get_value(t)
            t += dt.timedelta(1)
            # schedule new admissions
            if self.rate_factors == None:
                cases = self.case_numbers.get_value(t) * self.parameters.admission_rate
            else:
                cases = self.case_numbers.get_value(t) * self.parameters.admission_rate * self.rate_factors.get_value(t)
            admissionsPlanned = self.admissionDelays.get_delay(self.parameters.mean_admissions,
                                                               self.parameters.day_zero_probability_admissions, t,
                                                               cases)
            for t1, x in admissionsPlanned.items():
                admissions.add_value(t1, x)
            ##evaluate admissions
            adm = admissions.get_value(t)
            currentBeds += adm
            # schedule releases
            releasesPlanned = self.stayDelays.get_delay(self.parameters.mean_releases,
                                                        self.parameters.day_zero_probability_releases, t, adm)
            for t1, x in releasesPlanned.items():
                releases.add_value(t1, x)
            ##evaluate releases
            rel = releases.get_value(t)
            currentBeds -= rel
            # print to result
            beds.add_value(t, currentBeds)

        # scale time series to reference value
        referenceBeds = max([1, referenceBeds])  # make sure you don't multiply by 0 here
        diff = referenceBeds / beds.get_value(referenceDate)  # calculate factor
        for t in beds.get_times():
            beds.multiply_value(t, diff)  # scale up
        return [beds, admissions, releases]
