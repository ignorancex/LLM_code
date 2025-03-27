import numpy as np
from ..fundamental_constants import *
from ..mixing_type import MixingType
from ..particle_masses import *
from ..utils import generate_samples, allowed_e1_e2_three_body_decays, get_lepton_momenta_lab_frame
from ..particles.electron import Electron
from ..particles.muon import Muon

class ElectronMuon:
    def __init__(self, beam, parent) -> None:
        self.beam = beam
        self.parent = parent

    def partial_decay_rate(self):
        if self.beam.mixing_type == MixingType.electron:
            # https://arxiv.org/abs/hep-ph/9703333
            xm = MUON_MASS/self.parent.m
            func_form = 1 - 8*xm**2 + 8*xm**6 + xm**8 - 12*xm**4*np.log(xm**2)
            return self.beam.mixing_squared*(GF**2*self.parent.m**5/(192*np.pi**3)*func_form)

    def diff_distrubution(self, e_elec, e_muon):
        # https://arxiv.org/abs/hep-ph/9703333
        xm = MUON_MASS/self.parent.m
        x_mu = 2*e_muon/self.parent.m
        return x_mu*(1 - x_mu + xm**2)

    def decay(self):
        e_elec = np.linspace(ELECTRON_MASS, (self.parent.m**2 + ELECTRON_MASS**2)/(2*self.parent.m), 1000, endpoint=False)
        e_muon = np.linspace(MUON_MASS, (self.parent.m**2 + MUON_MASS**2)/(2*self.parent.m), 1000, endpoint=False)
        # NOTE region found from https://halldweb.jlab.org/DocDB/0033/003345/002/dalitz.pdf
        lepton_energy_samples = generate_samples(e_elec, e_muon, \
            dist_func=self.diff_distrubution, n_samples=self.beam.num_samples, \
            region=lambda e_elec, e_muon: allowed_e1_e2_three_body_decays(e_elec, e_muon, e_parent=self.parent.m, m1=ELECTRON_MASS, m2=MUON_MASS, m3=NEUTRINO_MASS))
        
        elec = Electron()
        muon = Muon()
        signal = []
        total_signal_length = min(len(self.parent.momenta), len(lepton_energy_samples))
        # experimental cuts for electron pair
        elec_e_min = 0.8 #GeV
        muon_e_min = self.beam.experiment.MUON_E_MIN
        mT_max = self.beam.experiment.MT_MAX
        for i in range(total_signal_length):
            momenta = get_lepton_momenta_lab_frame(lepton_energy_samples[i], self.parent.momenta[i], self.parent, elec, muon)
            if momenta:
                p_elec, p_muon, p_tot = momenta
            else:
                continue

            # apply cuts here
            if p_elec.get_energy() > elec_e_min and p_muon.get_energy() > muon_e_min and p_tot.get_transverse_mass() < mT_max:
                # NOTE we only need this signal if we want to plot
                signal.append([p_elec, p_muon, p_tot]) 

        efficiency = len(signal)/total_signal_length
        return efficiency

    def is_kinematically_allowed(self):
        return ELECTRON_MASS + MUON_MASS < self.parent.m 