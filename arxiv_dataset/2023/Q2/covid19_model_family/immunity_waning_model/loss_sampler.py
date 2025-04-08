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


class LossSampler:
    def __init__(self,dist:str,mean:float):
        """
        Samples a waning distribution. Create one for each cause and target
        :param dist: distribution name
        :param mean: mean value for the distribution
        """
        self.mean = mean
        if dist == "exponential":
            self.sampleFun = lambda x: self._samplefun_exponential(x)
        elif dist == "gamma":
            self.sampleFun = lambda x: self._samplefun_gamma(x)
        elif dist == "triangular":
            self.sampleFun = lambda x: self._samplefun_triangular(x)
        elif dist == "weibull":
            self.sampleFun = lambda x: self._samplefun_weibull(x)
        elif dist == "weibull2":
            self.sampleFun = lambda x: self._samplefun_weibull(x, 2)
        elif dist == "uniform":
            self.sampleFun = lambda x: self._samplefun_uniform(x)
        elif dist == 'lognormal':
            self.sampleFun = lambda x: self._samplefun_lognormal(x)
        elif dist == 'logistic':
            self.sampleFun = lambda x: self._samplefun_logistic(x)
        else:
            raise ValueError('Distribution specified in config is unknown')

    def _samplefun_exponential(self, mean) -> int:
        """
        Samples an exponentially distributed waning time
        :param mean: mean value of the exponential distribution
        :return: waning duration in days
        """
        return int(np.random.exponential(scale=mean))

    def _samplefun_gamma(self, mean) -> int:
        """
        Samples a gamma distributed waning time
        :param mean: mean value of the exponential distribution
        :return: waning duration in days
        """
        shp = 4
        return int(np.random.gamma(shape=shp, scale=mean / shp))

    def _samplefun_triangular(self, mean) -> int:
        """
        Samples a triangular distributed waning time. The distribution is fully sammetric between 0, mean and 2*mean
        :param mean: mean = mode of the triangular distribution
        :return: waning duration in days
        """
        shp = 3
        return int(np.random.triangular(0, mean, 2 * mean))

    def _samplefun_weibull(self, scale, shape=1.5) -> int:
        """
        Samples a weibull distributed waning time. The scale parameter is the one parametrized by the config. If shape!=1.0 this is NOT THE MEAN VALUE for this distribution, but something closely related (~life expectancy).
        :param scale: scale parameter of the weibull distribution
        :return: waning duration in days
        """
        return int(np.random.weibull(shape) * scale)

    def _samplefun_uniform(self, mean) -> int:
        """
        Samples a uniformly distributed waning time on [0,2*mean].
        :param mean: mean the uniform distribution
        :return: waning duration in days
        """
        return int(np.random.random() * 2 * mean)

    def _samplefun_lognormal(self, scale) -> int:
        """
        Samples a standard lognormal distributed waning time scaled by the scale parameter. Since E(lognormal(0,1))=sqrt(e), the scale parameter is NOT THE MEAN VALUE for this distribution but ~1/1.6 times the mean value.
        :param scale: factor to multiply the standard lognormal distributed variable with
        :return: waning duration in days
        """
        return int(scale * np.random.lognormal(mean=0, sigma=1))

    def _samplefun_logistic(self, mean, scale=15) -> int:
        """
        Samples a logistic distributed waning time with scale parameter.
        :param mean: mean value of the logoistic distribution
        :return: waning duration in days
        """
        x = int(np.random.logistic(mean, scale))
        return x

    def sample(self) -> int:
        """
        Samples a waning duration in days
        :return: waning duration in days
        """
        return self.sampleFun(self.mean)

    def sample_with_mean(self,mean:float):
        """
        Samples a waning duration in days. Use this to ignore the initialized mean.
        :return: waning duration in days
        """
        return self.sampleFun(mean)