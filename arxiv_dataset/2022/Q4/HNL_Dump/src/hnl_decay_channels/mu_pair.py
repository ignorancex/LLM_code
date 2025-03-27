import numpy as np
from ..fundamental_constants import *
from ..particles.muon import Muon
from ..particle_masses import *
from ..utils import generate_samples, get_lepton_momenta_lab_frame, allowed_e1_e2_three_body_decays

class MuonPair:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self):
        # Boiarska et al
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))*c1

    def diff_distribution(self, e_plus, e_minus):
        g_p = (SIN_WEINB**2 - 1/2)**2
        g_m = SIN_WEINB**2
        output = e_minus*(self.parent.m - 2*e_minus)*g_m**2 + e_plus*(self.parent.m - 2*e_plus)*g_p**2 \
        + 4*g_m*g_p*(1 - (e_plus + e_minus)/self.parent.m)*MUON_MASS**2
        return output

    def decay(self):
        mu_plus = np.linspace(MUON_MASS, self.parent.m/2, 1000)
        mu_minus = np.linspace(MUON_MASS, self.parent.m/2, 1000)
        lepton_energy_samples = generate_samples(mu_plus, mu_minus, \
            dist_func=lambda mu_p, mu_m: self.diff_distribution(mu_p, mu_m), n_samples=self.beam.num_samples, \
            region=lambda mu_p, mu_m: allowed_e1_e2_three_body_decays(mu_p, mu_m, e_parent=self.parent.m, m1=MUON_MASS, m2=MUON_MASS, m3=NEUTRINO_MASS))
        
        muon1 = Muon()
        muon2 = Muon()
        signal = []
        total_signal_length = min(len(self.parent.momenta), len(lepton_energy_samples))
        # experimental cuts for electron pair
        muon_e_min = 3. #GeV
        mT_max = self.beam.experiment.MT_MAX
        for i in range(total_signal_length):
            momenta = get_lepton_momenta_lab_frame(lepton_energy_samples[i], self.parent.momenta[i], self.parent, muon1, muon2)
            if momenta:
                p1, p2, p_tot = momenta
            else:
                continue

            # apply cuts here
            if p1.get_energy() > muon_e_min and p2.get_energy() > muon_e_min and p_tot.get_transverse_mass() < mT_max:
                # NOTE we only need this signal if we want to plot
                signal.append([p1, p2, p_tot]) 

        efficiency = len(signal)/total_signal_length
        return efficiency

    def is_kinematically_allowed(self):
        return 2*MUON_MASS < self.parent.m 