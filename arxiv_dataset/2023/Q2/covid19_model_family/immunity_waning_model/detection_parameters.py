"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""


from config import Config
import datetime as dt


class DetectionParameters:
    def __init__(self, config: Config) -> None:
        """
        Interface between detection rate parameters and simulation.
        :param config: config instance of the simulation
        """
        self.detection_probabilities = dict()
        self.detection_probabilities_raw = config.detectionProbability
        self.mintime = min(self.detection_probabilities_raw.keys())
        self.maxtime = max(self.detection_probabilities_raw.keys())

    def get_detection_probability(self,time:dt.datetime) -> float:
        """
        Returns the internally used list of variants
        :param time: current time
        :return: number between [0,1]
        """
        if not (time in self.detection_probabilities.keys()):
            if (time <= self.mintime):
                self.detection_probabilities[time] = self.detection_probabilities_raw[self.mintime]
            elif (time >= self.maxtime):
                self.detection_probabilities[time] = self.detection_probabilities_raw[self.maxtime]
            else:
                time1 = max([x for x in self.detection_probabilities_raw.keys() if x<= time])
                time2 = min([x for x in self.detection_probabilities_raw.keys() if x> time])
                fac = (time2-time).days/(time2-time1).days
                self.detection_probabilities[time] = fac*self.detection_probabilities_raw[time2]+(1-fac)*self.detection_probabilities_raw[time1]
        return self.detection_probabilities[time]