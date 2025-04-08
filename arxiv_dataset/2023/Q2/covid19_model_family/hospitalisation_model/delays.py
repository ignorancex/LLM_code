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
import scipy.stats
import datetime as dt


class Delays:
    """
    Class to create the discrete delay kernel for the simulation
    """

    def __init__(self, distribution: str) -> None:
        """
        Class to create the discrete delay kernel for the simulation.
        The distribution parameter is an identifier which specifies the delay distribution and its shape.
        Currently, only "gamma<x>" is implemented where "<x>" needs to be replaced by the shape of the distribution.
        This class is probably the most important for the whole simulation.
        :param distribution:
        """
        if distribution.startswith('gamma'):
            shape = float(distribution[5:])
            self.base_distribution = lambda x: scipy.stats.gamma.pdf(x, shape)
        else:
            raise ValueError('distribution ' + distribution + ' not implemented')
        self.stencils = dict()

    def _get_stencil(self, meanValue: float, zeroDayProb: float = None) -> np.array:
        """
        creates a discrete kernel for a given mean value and an optional discrete value for the first day
        :param meanValue: mean value of the continuous part of the distribution
        :param zeroDayProb: optional, fraction of cases which are not delayed
        :return: discrete probability array
        """
        stencil = list()
        for x in range(100):
            stencil.append(self.base_distribution(x / meanValue))
        if zeroDayProb != None:
            sm = sum(stencil[1:])
            stencil = [x / sm * (1 - zeroDayProb) for x in stencil]
            stencil[0] = zeroDayProb
        else:
            sm = sum(stencil)
            stencil = [x / sm for x in stencil]
        return np.array(stencil)

    def _calculate_delays(self, time: dt.date, cases: list[float]) -> dict[dt.date, float]:
        """
        Transforms from list of cases to a date:case dictionary
        :param time: time from which the delay is computed
        :param cases: array of delayed cases
        :return: date:case dictionary
        """
        delayed = dict()
        for i, c in enumerate(cases):
            delayed[time + dt.timedelta(i)] = c
        return delayed

    def get_delay(self, mean: float, zeroDayProb: float, time: dt.date, cases: float) -> dict[dt.date, float]:
        """
        Evaluates delays for a given number of cases at a given date
        :param mean: mean value of the continuous part of the distribution
        :param zeroDayProb: optional fraction of cases which are not delayed
        :param time: date to evaluate the delay
        :param cases: number of cases to delay
        :return:
        """
        tup = (mean, zeroDayProb)  # tuple to hash the discrete distribution
        try:
            stencil = self.stencils[tup]  # access discrete distribution
        except:
            stencil = self._get_stencil(mean, zeroDayProb)  # create discrete distribution
            self.stencils[tup] = stencil  # hash it
        casesDelayed = stencil * cases  # delay the cases
        return self._calculate_delays(time, casesDelayed)  # tranform from array to dictionary
