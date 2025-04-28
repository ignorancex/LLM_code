#!/usr/bin/env python3

# Place import files below
import copy

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d

from common_functions import (convert_Lsun_per_pc_to_mag_arcsec,
                              convert_Lsun_to_abs_mag, v_sphere)
from universal_settings import (b_mvir_ratio, d_lg, max_mstar, min_mstar,
                                min_star_particles, mu_e, mu_e2, obs_m31_r200,
                                obs_mw_r200, reff, reff2, sim_ids, sim_n_part,
                                udg_file_template)


def distance_from_dist_mod(distance_modulus):
    """Calculate distance to an object from its distance modulus.

    Args:
        distance_modulus (fl/arr): Distance modulus of an object.

    Returns:
        fl/arr: Distance to an object.
    """
    return 10.**((distance_modulus + 5.) / 5.)  # pc


class FattahiData(object):

    def __init__(self, data_file, h=0.704) -> None:
        """Reads and processes data from Fattahi+(2020)

        Args:
            data_file (str): Filepath to data.
            h (float, optional): Hubble parameter. Defaults to 0.704.

        Returns:
            None
        """
        self.data_file = data_file
        self.h = h

        self._load_data()
        return None

    def _load_data(self):
        """Loads data from the data file.

        Returns:
            None
        """
        data = np.genfromtxt(self.data_file)
        density_data = data / v_sphere(3. * self.h)
        density_data[:, 1] /= 1.5
        density_data[:, 0] *= self.h
        self.m_density = density_data[:, 0]
        self.n_density = density_data[:, 1]
        self.interp_func = interp1d(np.log10(density_data[:, 0]),
                                    np.log10(density_data[:, 1]))
        return None


class LGData(object):

    def __init__(self, data_file) -> None:
        """Reads and processes Local Group galaxy data.

        Args:
            data_file (str): Filepath to data file.

        Returns:
            None
        """
        self.lg_galaxy_data_file = data_file

        self._read_data()
        self._derived_data()
        self._data_selections()
        return None

    def _data_selections(self):
        """Helper function to select desired data. Sets class attributes

        Returns:
            None
        """
        self.select_lg_volume = self.dist <= (d_lg * 1.e3)
        self.select_mw_satellites = self.dist <= obs_mw_r200  # kpc
        self.select_m31_satellites = (
            (self.dist_m31 <= obs_m31_r200) * (self.dist_m31 > 0))  # kpc

        # Same stellar mass range as our analysis
        self.select_stellar_mass = ((self.m_star >= min_mstar) *
                                    (self.m_star <= max_mstar))

        self.select_field_galaxies = (~self.select_mw_satellites *
                                      ~self.select_m31_satellites *
                                      self.select_stellar_mass *
                                      self.select_lg_volume)
        return None

    def _derived_data(self):
        """Calculates data derived from the tabulated data.

        Returns:
            None
        """
        dm_list = np.asarray([
            np.zeros(len(self.dist_mod)), self.dist_mod_errplus,
            0. - self.dist_mod_errminus
        ]) + self.dist_mod

        # Distances relative to observer (i.e. Milky Way)
        (self.dist, self.dist_errplus,
         self.dist_errminus) = distance_from_dist_mod(dm_list) / 1.e3  # kpc

        self.abs_vmag = self.app_vmag - self.dist_mod
        self.luminosity = 10.**(-0.4 * (self.abs_vmag - 4.81))  # Lsun
        self.m_star = copy.deepcopy(self.luminosity)  # [Msun], Assumes M/L = 1

        c = SkyCoord(np.sum(self.ra * (1., 1. / 60., 1. / 3600.) * u.hourangle,
                            axis=1),
                     np.sum(self.dec * (u.deg, u.arcmin, u.arcsec), axis=1),
                     distance=self.dist * u.kpc)
        c_errplus = SkyCoord(np.sum(self.ra * (1., 1. / 60., 1. / 3600.) *
                                    u.hourangle,
                                    axis=1),
                             np.sum(self.dec * (u.deg, u.arcmin, u.arcsec),
                                    axis=1),
                             distance=self.dist_errplus * u.kpc)
        c_errminus = SkyCoord(np.sum(self.ra * (1., 1. / 60., 1. / 3600.) *
                                     u.hourangle,
                                     axis=1),
                              np.sum(self.dec * (u.deg, u.arcmin, u.arcsec),
                                     axis=1),
                              distance=self.dist_errminus * u.kpc)
        mw_c = SkyCoord(np.sum(self.ra[self.mw_idx] *
                               (1., 1. / 60., 1. / 3600.) * u.hourangle),
                        np.sum(self.dec[self.mw_idx] *
                               (u.deg, u.arcmin, u.arcsec)),
                        distance=self.dist[self.mw_idx] * u.kpc)
        m31_c = SkyCoord(np.sum(self.ra[self.m31_idx] *
                                (1., 1. / 60., 1. / 3600.) * u.hourangle),
                         np.sum(self.dec[self.m31_idx] *
                                (u.deg, u.arcmin, u.arcsec)),
                         distance=self.dist[self.m31_idx] * u.kpc)

        dists_from_mw = np.column_stack(
            (c.separation_3d(mw_c).value, c_errminus.separation_3d(mw_c).value,
             c_errplus.separation_3d(mw_c).value))  # kpc
        dists_from_m31 = np.column_stack(
            (c.separation_3d(m31_c).value,
             c_errminus.separation_3d(m31_c).value,
             c_errplus.separation_3d(m31_c).value))  # kpc

        (self.dist_errminus_rel_mw, self.dist_rel_mw,
         self.dist_errplus_rel_mw) = np.sort(dists_from_mw, axis=1).T
        (self.dist_errminus_rel_m31, self.dist_rel_m31,
         self.dist_errplus_rel_m31) = np.sort(dists_from_m31, axis=1).T

        return None

    def _read_data(self):
        """Reads data from data_file.

        Args:
            host_info (bool): Flag to read in host galaxy data. Defaults
                to False.

        Returns:
            None
        """
        obs_lg_galaxy_data = np.genfromtxt(self.lg_galaxy_data_file)
        self.ra = obs_lg_galaxy_data[:, [1, 2, 3]]
        self.dec = obs_lg_galaxy_data[:, [4, 5, 6]]
        self.dist_mod = obs_lg_galaxy_data[:, 7]
        self.dist_mod_errplus = obs_lg_galaxy_data[:, 8]
        self.dist_mod_errminus = obs_lg_galaxy_data[:, 9]
        self.app_vmag = obs_lg_galaxy_data[:, 10]
        # self.select_m31_satellites = obs_lg_galaxy_data[:, -1].astype(bool)
        self.dist_m31 = obs_lg_galaxy_data[:, -3]
        self.dist_m31[self.dist_m31 < 0] = 9999

        labels = np.genfromtxt(self.lg_galaxy_data_file,
                               delimiter=' ',
                               usecols=0,
                               dtype=str)

        self.mw_idx = np.argwhere(labels == 'MilkyWay')[0][0]
        self.m31_idx = np.argwhere(labels == 'M31')[0][0]

        return None


class UDGData(object):

    def __init__(self) -> None:
        """Reads in and processes all data concerning the UDGs in HESTIA
        """
        self.sim_ids = sim_ids
        self.udg_file_template = udg_file_template
        self.process_data()

    def process_data(self):
        """Processes the data being read from the data files.

        Returns:
            None
        """
        halo_mb_mvir_ratio = []
        halo_sim_id = []
        halo_m31_r200 = []
        halo_mw_r200 = []
        halo_ids = []
        halo_mass_res_selection = []
        halo_selection_reff1_mu1 = []
        halo_selection_reff1_mu2 = []
        halo_selection_reff2_mu1 = []
        halo_selection_reff2_mu2 = []
        halo_re_lum_Rband = []
        halo_re_lum_Vband = []
        halo_Rband_mu_mag_arcsec = []
        halo_Vband_mu_mag_arcsec = []
        halo_Vband_abs_mag = []
        stellar_mass = []
        star_particles = []
        halo_coordinates = []
        halo_coordinates_rel_mw = []
        halo_coordinates_rel_m31 = []
        halo_dist_mw = []
        halo_dist_m31 = []
        halo_dist_mid = []

        los_halo_sim_id = []
        los_halo_ids = []
        los_halo_re_lum_Rband = []
        los_halo_re_lum_Vband = []
        los_halo_mueff_Rband = []
        los_halo_mueff_Vband = []

        for sim_id in self.sim_ids:
            udg_file = self.udg_file_template.format(sim_n_part, sim_id)

            # Load UDG data
            with h5py.File(udg_file, 'r') as f:
                halo_ids_with_stars = np.uint64(f['Galaxies'][:, 0])
                halo_position = f['Galaxies'][:, [1, 2, 3]]  # Mpc
                dist_to_mw = f['Galaxies'][:, 4]  # Mpc
                dist_to_m31 = f['Galaxies'][:, 5]  # Mpc
                dist_to_midpoint = f['Galaxies'][:, 6]  # Mpc
                m_star = f['Galaxies'][:, 7]  # Msun
                r_half_lum_Rband = f['Galaxies'][:, 8]  # kpc
                r_half_lum_Vband = f['Galaxies'][:, 9]  # kpc
                Rband_lxy_rlum = f['Galaxies'][:, 10]  # Lsun
                Vband_lxy_rlum = f['Galaxies'][:, 11]  # Lsun
                mb_mvir_ratio = f['Galaxies'][:, 12]
                n_star_particles = f['Galaxies'][:, 13]

                primary_positions = f['Primaries'][:, [1, 2, 3]]
                primary_r200s = f['Primaries'][:, 4]  # kpc

                n_los = int(f['Lines of sight'].attrs['N_LOS'])
                los_halo_ids_with_stars = np.uint64(f['Lines of sight'][:, 0])
                los_rhalf_lum_rband = f['Lines of sight'][:, 1]  # kpc
                los_rhalf_lum_Vband = f['Lines of sight'][:, 2]  # kpc
                los_mueff_rband = f['Lines of sight'][:, 5]  # mag arcsec^-2
                los_mueff_Vband = f['Lines of sight'][:, 6]  # mag arcsec^-2

                los_sim_id_array = np.empty(len(los_halo_ids_with_stars),
                                            dtype='object')
                los_sim_id_array[:] = sim_id

            m31_position = primary_positions[0]
            mw_position = primary_positions[1]
            m31_r200 = primary_r200s[0]
            mw_r200 = primary_r200s[1]

            Vband_abs_mag = convert_Lsun_to_abs_mag(Vband_lxy_rlum, band='V')
            # 'Face-on'
            Rband_S_in_re = Rband_lxy_rlum / (np.pi *
                                              (r_half_lum_Rband * 1.e3)**2)
            Rband_mu_mag_arsec = convert_Lsun_per_pc_to_mag_arcsec(
                Rband_S_in_re, band='r')
            Vband_S_in_re = Vband_lxy_rlum / (np.pi *
                                              (r_half_lum_Vband * 1.e3)**2)
            Vband_mu_mag_arsec = convert_Lsun_per_pc_to_mag_arcsec(
                Vband_S_in_re, band='V')

            # Field galaxy selection
            select_mstar_nstar_data = (
                (m_star >= min_mstar) * (m_star <= max_mstar) *
                (n_star_particles >= min_star_particles) *
                (mb_mvir_ratio <= b_mvir_ratio))

            # UDG selections based on different criteria
            select_reff1_mu1_data = (select_mstar_nstar_data *
                                     (r_half_lum_Rband >= reff) *
                                     (Rband_mu_mag_arsec >= mu_e))
            select_reff1_mu2_data = (select_mstar_nstar_data *
                                     (r_half_lum_Rband >= reff) *
                                     (Rband_mu_mag_arsec >= mu_e2))
            select_reff2_mu1_data = (select_mstar_nstar_data *
                                     (r_half_lum_Rband >= reff2) *
                                     (Rband_mu_mag_arsec >= mu_e))
            select_reff2_mu2_data = (select_mstar_nstar_data *
                                     (r_half_lum_Rband >= reff2) *
                                     (Rband_mu_mag_arsec >= mu_e2))

            # Populate array with simulation ID of the given simulation
            sim_id_array = np.empty(len(halo_ids_with_stars), dtype='object')
            sim_id_array[:] = sim_id
            m31_r200_array = np.ones(len(halo_ids_with_stars),
                                     dtype=m31_r200.dtype) * m31_r200
            mw_r200_array = np.ones(len(halo_ids_with_stars),
                                    dtype=mw_r200.dtype) * mw_r200

            halo_Rband_mu_mag_arcsec.append(Rband_mu_mag_arsec)
            halo_Vband_mu_mag_arcsec.append(Vband_mu_mag_arsec)
            halo_Vband_abs_mag.append(Vband_abs_mag)
            halo_mb_mvir_ratio.append(mb_mvir_ratio)
            halo_ids.append(halo_ids_with_stars)
            halo_sim_id.append(sim_id_array)
            halo_m31_r200.append(m31_r200_array)
            halo_mw_r200.append(mw_r200_array)
            halo_re_lum_Rband.append(r_half_lum_Rband)
            halo_re_lum_Vband.append(r_half_lum_Vband)
            halo_mass_res_selection.append(select_mstar_nstar_data)
            halo_selection_reff1_mu1.append(select_reff1_mu1_data)
            halo_selection_reff1_mu2.append(select_reff1_mu2_data)
            halo_selection_reff2_mu1.append(select_reff2_mu1_data)
            halo_selection_reff2_mu2.append(select_reff2_mu2_data)
            stellar_mass.append(m_star)
            star_particles.append(n_star_particles)

            halo_coordinates.append(halo_position)
            halo_coordinates_rel_mw.append(halo_position - mw_position)
            halo_coordinates_rel_m31.append(halo_position - m31_position)
            halo_dist_mw.append(dist_to_mw)
            halo_dist_m31.append(dist_to_m31)
            halo_dist_mid.append(dist_to_midpoint)

            los_halo_sim_id.append(los_sim_id_array)
            los_halo_ids.append(los_halo_ids_with_stars)
            los_halo_re_lum_Rband.append(los_rhalf_lum_rband)
            los_halo_re_lum_Vband.append(los_rhalf_lum_Vband)
            los_halo_mueff_Rband.append(los_mueff_rband)
            los_halo_mueff_Vband.append(los_mueff_Vband)

        self.simulation_id = np.concatenate(halo_sim_id)
        self.m31_r200 = np.concatenate(halo_m31_r200)
        self.mw_r200 = np.concatenate(halo_mw_r200)
        self.halo_ids = np.concatenate(halo_ids)
        self.select_udgs_reff1_mu1 = np.concatenate(halo_selection_reff1_mu1)
        self.select_udgs_reff1_mu2 = np.concatenate(halo_selection_reff1_mu2)
        self.select_udgs_reff2_mu1 = np.concatenate(halo_selection_reff2_mu1)
        self.select_udgs_reff2_mu2 = np.concatenate(halo_selection_reff2_mu2)
        self.all_halo_mb_mvir_ratio = np.concatenate(halo_mb_mvir_ratio)
        self.rband_mu_mag_arsec = np.concatenate(halo_Rband_mu_mag_arcsec)
        self.Vband_mu_mag_arsec = np.concatenate(halo_Vband_mu_mag_arcsec)
        self.re_rband = np.concatenate(halo_re_lum_Rband)
        self.re_Vband = np.concatenate(halo_re_lum_Vband)
        self.m_star = np.concatenate(stellar_mass)
        self.select_gal_mstar_nstar = np.concatenate(halo_mass_res_selection)
        self.all_star_particles = np.concatenate(star_particles)
        self.abs_mag_Vband = np.concatenate(halo_Vband_abs_mag)
        self.position = np.row_stack(halo_coordinates) * 1.e3  # kpc
        self.position_rel_mw = np.row_stack(
            halo_coordinates_rel_mw) * 1.e3  # kpc
        self.position_rel_m31 = np.row_stack(
            halo_coordinates_rel_m31) * 1.e3  # kpc
        self.dist_from_mw = np.concatenate(halo_dist_mw) * 1.e3  # kpc
        self.dist_from_m31 = np.concatenate(halo_dist_m31) * 1.e3  # kpc
        self.dist_from_midpoint = np.concatenate(halo_dist_mid) * 1.e3  # kpc
        self.app_mag_Vband_rel_mw = self.abs_mag_Vband + 5. * np.log10(
            self.dist_from_mw * 1.e2)
        self.app_mag_Vband_rel_m31 = self.abs_mag_Vband + 5. * np.log10(
            self.dist_from_m31 * 1.e2)
        self.app_mag_Vband_rel_midpoint = self.abs_mag_Vband + 5. * np.log10(
            self.dist_from_midpoint * 1.e2)

        self.n_los = n_los
        self.los_simulation_id = np.concatenate(los_halo_sim_id)
        self.los_halo_ids = np.concatenate(los_halo_ids)
        self.los_re_rband = np.concatenate(los_halo_re_lum_Rband)
        self.los_re_Vband = np.concatenate(los_halo_re_lum_Vband)
        self.los_mue_rband = np.concatenate(los_halo_mueff_Rband)
        self.los_mue_Vband = np.concatenate(los_halo_mueff_Vband)

        return None

    def retrieve_data(self, attr_list):
        """Returns values of a set of desired class attributes.

        Args:
            attr_list (list): List of desired class attributes

        Returns:
            list: List containing the values of the requested class
                attributes.
        """
        return [getattr(self, attr_item) for attr_item in attr_list]
