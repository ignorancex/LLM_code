
from .particle import Particle
from ..particle_masses import *

class Pion(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(PION_MASS, beam, parent, momenta)