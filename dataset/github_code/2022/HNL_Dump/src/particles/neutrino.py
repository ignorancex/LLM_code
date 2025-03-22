from .particle import Particle
from ..particle_masses import *

class Neutrino(Particle):
    """Massless neutrino"""
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(NEUTRINO_MASS, beam, parent, momenta)