import numpy as np
from .particles.tau_decay_modes import TauDecayModes
from .mixing_type import MixingType
from .particles.hnl import HNL
from .particles.tau import Tau
from .particles.DMeson import DMeson
from .particles.DsMeson import DsMeson 
from .particles.BMeson import BMeson
import src.branching_ratios as BR
import src.cross_sections as CS
import src.particle_masses as PM
from .logger import Logger

class SignalProcessor:
    def __init__(self, beam) -> None:
        self.beam = beam
    
    def get_upper_bound(self):
        hnls = self.beam.find_instances_of_type(HNL)
        if len(hnls) == 0:
            raise Exception("No HNLs!")

        total_decays = self.get_total_decays(hnls)

        upper_bound_squared = np.sqrt(self.beam.experiment.OBSERVED_EVENTS/total_decays)
        return upper_bound_squared

    def get_total_decays(self, hnls):
        total_decays = 0
        for hnl in hnls:
            total_decays_from_channel = self.get_total_decays_for_hnl(hnl)
            total_decays += total_decays_from_channel
        # NOTE: factor 2 to account for anti-HNLs
        return 2*total_decays

    def get_total_decays_for_hnl(self, hnl):
        logger = Logger()
        total_decays = 0
        for channel in hnl.active_channels:
            efficiency = hnl.efficiency[channel]
            prop_factor = hnl.average_propagation_factor[channel]
            acceptance = hnl.acceptance
            total_flux = 0
            logger.log(f"Propagation factor: {prop_factor}")
            logger.log(f"Acceptance: {acceptance}")

            if self.beam.mixing_type == MixingType.electron:
                if isinstance(hnl.parent, DMeson):
                    total_flux = self.__get_normalised_hnl_flux_from_DpDm_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (D): {:e}".format(total_flux))
                elif isinstance(hnl.parent, DsMeson):
                    total_flux = self.__get_normalised_electron_hnl_flux_from_Ds_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (Ds): {:e}".format(total_flux))
                if isinstance(hnl.parent, BMeson):
                    total_flux = self.__get_normalised_hnl_flux_from_Bpm_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (B): {:e}".format(total_flux))
            elif self.beam.mixing_type == MixingType.tau:
                if isinstance(hnl.parent, Tau):
                    if hnl.decay_mode == TauDecayModes.hnl_pi:
                        total_flux = self.__get_normalised_hnl_flux_from_tau_two_body(hnl_mass=hnl.m)
                        logger.log("Flux norm (tau 2-body): {:e}".format(total_flux))
                    elif hnl.decay_mode == TauDecayModes.hnl_lepton_nu:
                        total_flux = self.__get_normalised_hnl_flux_from_tau_three_body(hnl_mass=hnl.m)
                        logger.log("Flux norm (tau 3-body): {:e}".format(total_flux))
                elif isinstance(hnl.parent, DsMeson):
                    total_flux = self.__get_normalised_tau_hnl_flux_from_Ds_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (Ds): {:e}".format(total_flux))
                elif isinstance(hnl.parent, BMeson):
                    total_flux = self.__get_normalised_hnl_flux_from_Bpm_mesons(hnl_mass=hnl.m)
                    logger.log("Flux norm (B): {:e}".format(total_flux))
            if acceptance == 0:
                raise Exception("Zero acceptance")
            total_decays += total_flux*prop_factor*efficiency*acceptance
        return total_decays

    def total_decays_less_than_observed(self):
        hnls = self.beam.find_instances_of_type(HNL)
        if len(hnls) == 0:
            raise Exception("No HNLs!")

        total_decays = self.get_total_decays(hnls)
        print(f"Mixing: {self.beam.mixing_squared}, Total decays: {total_decays}")
        return total_decays - self.beam.experiment.OBSERVED_EVENTS

    def __get_normalised_hnl_flux_from_DpDm_mesons(self, hnl_mass):
        return self.beam.experiment.ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DPDM_X*BR.D_TO_E_HNL(hnl_mass, self.beam.mixing_squared))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)

    def __get_normalised_tau_hnl_flux_from_Ds_mesons(self, hnl_mass):
        return self.beam.experiment.ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DSDS_X*BR.DS_TO_TAU_HNL(hnl_mass, self.beam.mixing_squared))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)

    def __get_normalised_electron_hnl_flux_from_Ds_mesons(self, hnl_mass):
        return self.beam.experiment.ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DSDS_X*BR.DS_TO_ELECTRON_HNL(hnl_mass, self.beam.mixing_squared))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)

    def __get_normalised_hnl_flux_from_tau_two_body(self, hnl_mass):
        return self.beam.experiment.ELECTRON_NU_MASSLESS_FLUX*(CS.P_TO_DSDS_X*BR.DS_TO_TAU_X*BR.TAU_TO_PI_HNL(hnl_mass, self.beam.mixing_squared))/(CS.P_TO_DPDM_X*BR.D_TO_E_NUE_X + CS.P_TO_D0D0_X*BR.D0_TO_E_NUE_X)

    def __get_normalised_hnl_flux_from_tau_three_body(self, hnl_mass):
        DS_TO_D_FLUX_RATIO = 0.2
        D_FLUX = self.beam.experiment.ELECTRON_NU_MASSLESS_FLUX/(BR.D_TO_E_NUE_X + 2*BR.D0_TO_E_NUE_X)
        DS_FLUX = DS_TO_D_FLUX_RATIO*D_FLUX
        hnl_flux = DS_FLUX*BR.DS_TO_TAU_X*BR.TAU_TO_HNL_L_NU_L(hnl_mass, self.beam.mixing_squared)
        return hnl_flux
    
    def __get_normalised_hnl_flux_from_Bpm_mesons(self, hnl_mass):
        B_FLUX = 2*self.beam.experiment.POT*self.beam.experiment.B_MESON_FRACTION
        if self.beam.mixing_type == MixingType.electron:
            hnl_flux = B_FLUX*BR.B_TO_ELL_HNL(hnl_mass, self.beam.mixing_squared, PM.ELECTRON_MASS)*(PM.ELECTRON_MASS/PM.TAU_MASS)**2
            return hnl_flux
        if self.beam.mixing_type == MixingType.tau:
            hnl_flux = B_FLUX*BR.B_TO_ELL_HNL(hnl_mass, self.beam.mixing_squared, PM.TAU_MASS)
            return hnl_flux