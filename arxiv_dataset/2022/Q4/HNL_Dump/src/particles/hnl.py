
import numpy as np
from ..mixing_type import MixingType
from ..particle_masses import *
from ..utils import DEBUG_AVERAGE_MOMENTUM
from .particle import Particle
from ..fundamental_constants import *
from ..logger import Logger
from ..hnl_decay_channels.e_pair import ElectronPair
from ..hnl_decay_channels.e_mu import ElectronMuon
from ..hnl_decay_channels.e_pi import ElectronPion
from ..hnl_decay_channels.mu_pair import MuonPair
import scipy.integrate as integrate

def I(xu, xd, xl):
    l = lambda a, b, c: a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c
    integrand = lambda x: (x - xl**2 - xd**2)*(1 + xu**2 - x)*np.sqrt(l(x,xl**2,xd**2)*l(1,x,xu**2))/x
    integral = integrate.quad(integrand, (xd + xl)**2, (1-xu)**2)
    return 12*integral[0]

class HNL(Particle):
    def __init__(self, mass, beam=None, parent=None, momenta=[], decay_mode=None):
        super().__init__(mass, beam, parent, momenta)
        self.decay_mode = decay_mode
        self.signal = {}
        self.average_propagation_factor = {}
        self.efficiency = {}
        self.majorana = True
        self.majFactor = 1
        if self.majorana == True:
            self.majFactor = 2;


        # NOTE add more HNL decay channels here
        self.decay_channels = {
            "e+pos+nu": ElectronPair,
            "mu+e+nu": ElectronMuon,
            "mu+mu+nu": MuonPair,
            "e+pi": ElectronPion
        }
        self.active_channels = {}

    def __average_propagation_factor(self, length_detector, decay_rate):
        factors = []
        for p in self.momenta:
            factor = self.majFactor*length_detector*self.m*decay_rate/p.get_total_momentum()
            factors.append(factor)
        return np.average(factors)

    def __avg_non_linear_propagation_factor(self, detector_length, detector_distance, partial_decay_rate, total_decay_rate):
        factors = []
        for p in self.momenta:
            ptot = p.get_total_momentum()
            factor1 = np.exp(-detector_distance*self.majFactor*self.m*total_decay_rate/ptot)
            factor2 = 1 - np.exp(-detector_length*self.majFactor*self.m*total_decay_rate/ptot)
            factor = factor1*factor2*(partial_decay_rate/total_decay_rate)
            factors.append(factor)
        return np.average(factors)

    def __decay_rate_to_electron_up_down(self):
        ckm_factor = 3*0.974**2
        xe = ELECTRON_MASS/self.m
        # only consider decays to up/down quarks which we take to be massless
        return ckm_factor*(GF**2*self.m**5)/(192*np.pi**3)*self.beam.mixing_squared*I(xe, 0, 0)

    def __neutral_decay_rate_to_ffbar(self, fermion):
        x = 0
        c1f = 0
        c2f = 0
        pre_fac = 3*GF**2*self.m**5/(192*np.pi**3)*self.beam.mixing_squared
        if fermion == 'up':
            x = UP_MASS/self.m
            c1f = (1 - (8/3)*SIN_WEINB**2 + (32/9)*SIN_WEINB**4)/4
            c2f = SIN_WEINB**2*((4/3)*SIN_WEINB**2 - 1)/3
        elif fermion == 'down':
            x = DOWN_MASS/self.m
            c1f = (1 - (4/3)*SIN_WEINB**2 + (8/9)*SIN_WEINB**4)/4
            c2f = SIN_WEINB**2*((2/3)*SIN_WEINB**2 - 1)/6
        elif fermion == 'nu':
            return 0.25*pre_fac
        else:
            raise Exception("Invalid fermion type: enter 'up' or 'down' or 'nu")
        
        return pre_fac*(c1f)

    def __decay_rate_to_pion(self):
        x = PION_MASS/self.m
        return ((GF**2)*(self.m**3)*(F_PI**2)/(32*np.pi))*self.beam.mixing_squared*(1-x**2)**2

    def __total_decay_rate_to_non_considered_channels(self):
        # https://arxiv.org/pdf/1805.08567.pdf
        total = self.__neutral_decay_rate_to_ffbar('nu')
        if self.m < 1.:
            total += self.__decay_rate_to_pion()
        else:
            total += self.__neutral_decay_rate_to_ffbar('up') + self.__neutral_decay_rate_to_ffbar('down') 
        if self.beam.mixing_type == MixingType.electron:
            total += self.__decay_rate_to_electron_up_down()
        return total

    def __total_decay_rate(self):
        total_decay_rate = self.__total_decay_rate_to_non_considered_channels()
        for channel_code in self.beam.channels:
            if channel_code in self.decay_channels:
                decay_channel = self.decay_channels[channel_code](beam=self.beam, parent=self)
                total_decay_rate += decay_channel.partial_decay_rate()
        return total_decay_rate

    def __get_prop_factor_for_regime(self, partial_decay):
        if self.beam.linear_regime:
            return self.__average_propagation_factor(self.beam.experiment.DETECTOR_LENGTH, partial_decay)
        else:
            total_decay = self.__total_decay_rate()
            return self.__avg_non_linear_propagation_factor(self.beam.experiment.DETECTOR_LENGTH, self.beam.experiment.DETECTOR_DISRANCE, partial_decay, total_decay)

    def decay(self):
        self.acceptance = self.geometric_cut(0, self.beam.experiment.DETECTOR_OPENING_ANGLE)
        DEBUG_AVERAGE_MOMENTUM(self, "Average HNL momentum after angle cut")
        for channel_code in self.beam.channels:
            if channel_code in self.decay_channels:
                decay_channel = self.decay_channels[channel_code](beam=self.beam, parent=self)
                if not decay_channel.is_kinematically_allowed():
                    continue
                self.active_channels[channel_code] = decay_channel
                partial_decay = decay_channel.partial_decay_rate()
                Logger().log(f"{channel_code} partial decay rate: {partial_decay}")
                self.average_propagation_factor[channel_code] = self.__get_prop_factor_for_regime(partial_decay)
                self.efficiency[channel_code] = decay_channel.decay()
                Logger().log(f"{channel_code} channel efficiency: {self.efficiency[channel_code]}")
        return self
    