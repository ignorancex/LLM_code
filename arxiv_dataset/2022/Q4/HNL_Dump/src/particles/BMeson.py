from .particle import Particle
from .hnl import HNL
from .electron import Electron
from .tau import Tau
from ..particle_masses import B_MASS
from ..mixing_type import MixingType
from ..utils import get_two_body_momenta

class BMeson(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(B_MASS, beam, parent, momenta)
    
    def two_body_decay(self, hnl, other_particle):
        hnl_rest_momenta = get_two_body_momenta(self, hnl, other_particle, num_samples=self.beam.num_samples)
        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)
        hnl.decay()

        self.children.append(hnl)
        self.children.append(other_particle)
        
    def decay(self, hnl_mass):
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        if self.beam.mixing_type == MixingType.electron:
            other_particle = Electron(parent=self, beam=self.beam)   
            self.two_body_decay(hnl, other_particle)

        if self.beam.mixing_type == MixingType.tau:
            other_particle = Tau(parent=self, beam=self.beam)   
            self.two_body_decay(hnl, other_particle)


        