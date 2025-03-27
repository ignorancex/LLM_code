from .particle import Particle
from ..particle_masses import *

class Electron(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(ELECTRON_MASS, beam, parent, momenta)