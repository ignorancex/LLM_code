import numpy as np
from ..fundamental_constants import *
from ..mixing_type import MixingType
from ..utils import PLOT_ENERGY_ANGLE, generate_samples, allowed_e1_e2_three_body_decays, get_lepton_momenta_lab_frame
from ..particle_masses import *
from ..particles.electron import Electron
from ..decay_type import DecayType
# NOTE this is only a parameter to compare with previous work which only considered the CC decay channels
# so this could be removed entirely and treated in full
DECAY_TYPE = DecayType.CCNC
class ElectronPair:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self, decay_type=DecayType.CCNC):
        # https://arxiv.org/pdf/2109.03831.pdf
        c1 = 0.25*(1 - 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        c2 = 0.25*(1 + 4*SIN_WEINB**2 + 8*SIN_WEINB**4)  
        if self.beam.mixing_type == MixingType.electron:
            # Ne -> e+ e- nu_e
            if decay_type == DecayType.CC:
                return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))
            elif decay_type == DecayType.CCNC:
                return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))*c2
            else:
                raise Exception("No value for only neutral current")
        elif self.beam.mixing_type == MixingType.tau:
            # N_tau -> e+ e- nu_tau
            return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3))*c1

    def diff_distribution(self, ep, em, decay_type=DecayType.CCNC):
        # https://arxiv.org/pdf/2109.03831.pdf
        if self.beam.mixing_type == MixingType.tau:
            decay_type = DecayType.NC
        gr = SIN_WEINB**2 - 1/2
        gl = SIN_WEINB**2
        if decay_type == DecayType.CC:
            gr = 0
            gl = 0
        elif decay_type == DecayType.NC:
            gl = gl - 1
        output = gr**2*em*(self.parent.m - 2*em) + (1-gl)**2*ep*(self.parent.m - 2*ep)
        return output

    def decay(self):
        # Decay Ne/tau -> e+ e- nu_e/tau
        e_l_plus = np.linspace(0, self.parent.m/2, 1000)
        e_l_minus = np.linspace(0, self.parent.m/2, 1000)
        lepton_energy_samples = generate_samples(e_l_plus, e_l_minus, \
            dist_func=lambda ep, em: self.diff_distribution(ep, em, decay_type=DECAY_TYPE), n_samples=self.beam.num_samples, \
            region=lambda ep, em: allowed_e1_e2_three_body_decays(ep, em, e_parent=self.parent.m, m1=ELECTRON_MASS, m2=ELECTRON_MASS, m3=NEUTRINO_MASS))
        
        elec1 = Electron()
        elec2 = Electron()
        total_momenta = []
        cut_signal = []
        total_signal_length = min(len(self.parent.momenta), len(lepton_energy_samples))
        # experimental cuts for electron pair
        e_min = 0.8 #GeV
        mT_max = self.beam.experiment.MT_MAX
        for i in range(total_signal_length):
            momenta = get_lepton_momenta_lab_frame(lepton_energy_samples[i], self.parent.momenta[i], self.parent, elec1, elec2)
            if momenta:
                p1, p2, p_tot = momenta
                total_momenta.append(p_tot) 
            else:
                continue

            # apply cuts here
            if p1.get_energy() > e_min and p2.get_energy() > e_min and p_tot.get_transverse_mass() < mT_max:
                cut_signal.append(p_tot) 

        PLOT_ENERGY_ANGLE(total_momenta,((0, 200), (0, 0.08)), filename=f"e+e-from_hnls_[{self.beam.mixing_type}]") 
        efficiency = len(cut_signal)/total_signal_length
        return efficiency

    def is_kinematically_allowed(self):
        return 2*ELECTRON_MASS < self.parent.m 