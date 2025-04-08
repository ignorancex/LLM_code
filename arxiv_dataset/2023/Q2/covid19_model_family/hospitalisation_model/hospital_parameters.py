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


class HospitalParameters:
    """
    Wrapper to manage the free parameters of the simulation
    """

    def __init__(self) -> None:
        """
        Wrapper to manage the free parameters of the simulation.
        Note that the day_zero probabilities are currently unused in the main experiment
        """
        self.mean_admissions = 3
        self.admission_rate = 0.0002
        self.mean_releases = 8
        self.day_zero_probability_admissions = None
        self.day_zero_probability_releases = None

    def __repr__(self) -> str:
        """
        make a nice text representation to print into the console
        """
        return '[admission rate: {:,.08f}, mean admission delay: {:.02f}, mean length of stay: {:.02f}]'.format(
            self.admission_rate, self.mean_admissions, self.mean_releases)

    def set(self, vector: np.array) -> None:
        """
        Set the free parameters to the given values
        :param vector: 3 element vector
        :return:
        """
        self.admission_rate = vector[0]
        self.mean_admissions = vector[1]
        self.mean_releases = vector[2]

    def get(self) -> np.array:
        """
        Returns the three free parameters as vector
        :return: parameters vector
        """
        return np.array([self.admission_rate, self.mean_admissions, self.mean_releases])
