from .particle import Particle
from .hnl import HNL
from .electron import Electron
from ..particle_masses import B_MASS
from ..utils import get_two_body_momenta

class B0Meson(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(B_MASS, beam, parent, momenta)
    
    def decay(self, hnl_mass):
        hnl = HNL(hnl_mass, beam=self.beam, parent=self)
        # TODO: implement proper decay channels, this is just an example
        other_particle = Electron(parent=self, beam=self.beam)

        # set the kinematics of the children
        hnl_rest_momenta = get_two_body_momenta(self, hnl, other_particle, num_samples=self.beam.num_samples)

        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)

        hnl.decay()

        self.children.append(hnl)
        self.children.append(other_particle)