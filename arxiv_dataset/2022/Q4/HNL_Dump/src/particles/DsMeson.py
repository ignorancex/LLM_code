from .neutrino import Neutrino
from ..particle_masses import *
from ..utils import get_two_body_momenta
from .particle import Particle
from .electron import Electron
from .tau import Tau
from .hnl import HNL
from ..mixing_type import MixingType
from ..utils import get_two_body_momenta

class DsMeson(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(DS_MASS, beam, parent, momenta)

    def __decay_tau_mixing(self, hnl_mass):
        tau = Tau(parent=self, beam=self.beam)
        other_particle = Neutrino(parent=self, beam=self.beam)
        if TAU_MASS + hnl_mass < DS_MASS:
            # HNL mass is small enough to produce an HNL here
            other_particle = HNL(hnl_mass, beam=self.beam, parent=self)
        
        tau_rest_momenta = get_two_body_momenta(self, tau, other_particle, self.beam.num_samples)

        tau.set_momenta(tau_rest_momenta).boost(self.momenta)

        tau.decay(hnl_mass)

        if isinstance(other_particle, HNL):
            hnl_rest_momenta = get_two_body_momenta(self, other_particle, tau, self.beam.num_samples)
            other_particle.set_momenta(hnl_rest_momenta).boost(self.momenta)
            other_particle.decay()

        self.children.append(other_particle)
        self.children.append(tau)
        

    def __decay_electron_mixing(self, hnl_mass):
        electron = Electron(parent=self, beam=self.beam)
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        
        hnl_rest_momenta = get_two_body_momenta(self, hnl, electron, self.beam.num_samples)
        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)
        hnl.decay()

        self.children.append(hnl)
        self.children.append(electron)

    def decay(self, hnl_mass):
        # D -> N + lepton (electron, muon, tau)
        if self.beam.mixing_type == MixingType.electron and hnl_mass + ELECTRON_MASS < DS_MASS:
            self.__decay_electron_mixing(hnl_mass)
        elif self.beam.mixing_type == MixingType.tau:
            self.__decay_tau_mixing(hnl_mass)
        return self