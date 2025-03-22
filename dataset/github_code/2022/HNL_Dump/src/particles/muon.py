from .particle import Particle
from ..particle_masses import *

class Muon(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(MUON_MASS, beam, parent, momenta)