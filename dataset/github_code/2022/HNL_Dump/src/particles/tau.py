import numpy as np
from .hnl import HNL
from .particle import Particle
from .pion import Pion
from ..particle_masses import *
from ..utils import generate_samples, e_cos_theta_to_momentum4, get_two_body_momenta, PLOT_ENERGY_ANGLE_JACOBIAN
from .tau_decay_modes import TauDecayModes

class Tau(Particle):
    def __init__(self, beam=None, parent=None, momenta=[]):
        super().__init__(TAU_MASS, beam, parent, momenta)
    def decay(self, hnl_mass):
        if hnl_mass > TAU_MASS:
            return self

        hnl = HNL(hnl_mass, beam=self.beam, parent=self, decay_mode=TauDecayModes.hnl_lepton_nu)
        hnl2body = HNL(hnl_mass, beam=self.beam, parent=self, decay_mode=TauDecayModes.hnl_pi)
        pion = Pion(beam=self.beam, parent=self)

        e_max = (self.m**2 + hnl_mass**2)/(2*self.m)
        e = np.linspace(hnl_mass, e_max, 1_000)
        cos_theta = np.linspace(0., 1., 1_000)
        hnl_rest_samples = generate_samples(e, cos_theta, dist_func=lambda e, cos: self.diff_decay_to_hnl_nu_lepton(hnl_mass, e, cos), n_samples=self.beam.num_samples)
        hnl_rest_momenta  = e_cos_theta_to_momentum4(hnl_rest_samples, hnl_mass)
        hnl.set_momenta(hnl_rest_momenta).boost(self.momenta)
        PLOT_ENERGY_ANGLE_JACOBIAN(hnl.momenta, ((0, 200), (0, 0.5)), filename=f"hnls_from_TAU_[{self.beam.mixing_type}]", detector_cut=False,
                          horizontal_line=self.beam.experiment.DETECTOR_OPENING_ANGLE, xLabel = r'$E_N$', yLabel = r'$\theta_N^2 \, \times \, 10^{-3}$')

        hnl.decay()
        
        if hnl_mass + pion.m < self.m:
            hnl_rest_momenta_two_body = get_two_body_momenta(self, hnl, pion, self.beam.num_samples)
            hnl2body.set_momenta(hnl_rest_momenta_two_body).boost(self.momenta)
            hnl2body.decay()

        self.children.append(hnl)
        self.children.append(hnl2body)
        return self

    def diff_decay_to_hnl_nu_lepton(self, hnl_mass, e, cos_theta):
        # valid for all leptons because the lepton mass has been neglected
        x = 2*e/self.m
        beta = np.sqrt(1 - (hnl_mass/e)**2)
        return (3 - 2*x + (3*x - 4)*(1-beta**2)*x/4)*beta*x**2