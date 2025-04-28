"""
QSOLoader for DR16Q GP catalogue
"""

from json import dump
from types import BuiltinFunctionType
from typing import Dict, List, Tuple

from collections import namedtuple, Counter
from astropy.io.fits import hdu

import numpy as np
from matplotlib import pyplot as plt
import h5py

from astropy.io import fits
from numpy.lib.type_check import real_if_close

from .set_parameters import *
from .qso_loader import QSOLoader, GPLoader, conditional_mvnpdf_low_rank, make_fig, search_index_from_another
from .voigt import Voigt_absorption
from .effective_optical_depth import effective_optical_depth
from .solene_dlas import solene_eBOSS_cuts, bitget_string

class QSOLoaderDR16Q(QSOLoader):
    def __init__(
        self,
        preloaded_file: str = "preloaded_qsos.mat",
        catalogue_file: str = "catalog.mat",
        learned_file: str = "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        processed_file: str = "processed_qsos_dr12q.mat",
        dla_concordance: str = "dla_catalog",
        los_concordance: str = "los_catalog",
        snrs_file: str = "snrs_qsos_multi_meanflux_dr16q.mat",
        sub_dla: bool = True,
        sample_file: str = "dla_samples_a03_30000.mat",
        tau_sample_file: str = "tau_0_samples_30000.mat",
        occams_razor: float = 1 / 30,
        dist_file: str = "DR16Q_v4.fits",
        is_qso_final_cut: bool = True,
        class_person_cut: bool = True,
        z_source_cut: bool = True,
        zestimation_cut: bool = True,
    ):

        super().__init__(
            preloaded_file=preloaded_file,
            catalogue_file=catalogue_file,
            learned_file=learned_file,
            processed_file=processed_file,
            dla_concordance=dla_concordance,
            los_concordance=los_concordance,
            snrs_file=snrs_file,
            sub_dla=sub_dla,
            sample_file=sample_file,
            occams_razor=occams_razor,
        )

        # [meanflux] read the meanflux suppression samples
        self.tau_sample_file = h5py.File(tau_sample_file, "r")

        self.tau_0_samples = self.tau_sample_file["tau_0_samples"][:, 0]
        self.beta_samples = self.tau_sample_file["beta_samples"][:, 0]

        # read the DLA parameter samples
        self.sample_file = h5py.File(sample_file, "r")

        self.log_nhi_samples = self.sample_file["log_nhi_samples"][:, 0]

        # read the DR16Q catalogue file
        self.dist_file = dist_file
        self.hdu = fits.open(dist_file)

        # filter: from DR16Q catalog -> our QSO samples
        # 1) first bring in the test_ind and nan_ind built in QSOLoader
        self.condition = np.ones(np.sum(self.test_ind), dtype=np.bool)
        # 2) filter out the disagreement in zestimation
        if zestimation_cut:
            self.zestimation_cut()
        # 3) filter out the rest of them with eBOSS cut
        self.eBOSS_cut(is_qso_final_cut, class_person_cut, z_source_cut)

        # reload the arrays with the eBOSS cut condition
        self._load_arrays()

    def _load_arrays(self):
        """
        Load GP processed file and catalog based on the self.condition
        """
        # test_set prior inds : organise arrays into the same order using selected test_inds
        # self.test_ind = self.processed_file['test_ind'][0, :].astype(np.bool) #size: (num_qsos, )
        # Assume you already have test_ind in the self
        assert "test_ind" in dir(self)
        self.test_real_index = np.nonzero(self.test_ind)[0]

        # load processed data
        self.model_posteriors = self.processed_file["model_posteriors"][()].T

        self.p_dlas = self.processed_file["p_dlas"][0, :]
        self.p_no_dlas = self.processed_file["p_no_dlas"][0, :]
        if self.sub_dla:
            self.p_no_dlas += self.model_posteriors[:, 1]
        self.log_priors_dla = self.processed_file["log_priors_dla"][0, :]
        self.min_z_dlas = self.processed_file["min_z_dlas"][0, :]
        self.max_z_dlas = self.processed_file["max_z_dlas"][0, :]
        if "MAP_log_nhis" in self.processed_file.keys():
            self.map_log_nhis = self.processed_file["MAP_log_nhis"][()].T
            self.map_z_dlas = self.processed_file["MAP_z_dlas"][()].T

        # [partial file] if we are reading a partial processed file trim down
        # the test_ind
        if self.p_dlas.shape[0] != self.test_real_index.shape[0]:
            print("[Warning] You have less samples in processed file than test_ind,")
            print(
                "          you are reading a partial file, otherwise, check your test_ind."
            )
            size = self.p_dlas.shape[0]
            self.test_ind[self.test_real_index[size:]] = False
            self.test_real_index = np.nonzero(self.test_ind)[0]
            assert self.test_real_index.shape[0] == self.p_dlas.shape[0]
            print("[Info] Reading first {} spectra.".format(size))

        # load snrs data
        try:
            self.snrs = self.snrs_file["snrs"][0, :]
        except TypeError as e:
            # this is to read the file computed by sbird's calc_cddf.compute_all_snrs function
            print(e)
            print(
                "[Warning] You are reading the snrs file from calc_cddf.compute_all_snrs."
            )
            self.snrs = self.snrs_file["snrs"][()]
        # [partial processed file but snrs not partial]
        if self.snrs.shape[0] != self.test_real_index.shape[0]:
            assert self.snrs.shape[0] == np.sum(
                self.processed_file["test_ind"][0, :].astype(np.bool)
            )
            print(
                "[Warning] Reading only first {} snrs.".format(
                    self.test_real_index.shape[0]
                )
            )
            self.snrs = self.snrs[: self.test_real_index.shape[0]]

        # store thing_ids based on test_set prior inds
        self.thing_ids = self.catalogue_file["thing_ids"][0, :].astype(np.int)
        self.thing_ids = self.thing_ids[self.test_ind]

        # plates, mjds, fiber_ids
        self.plates = self.catalogue_file["plates"][0, :].astype(np.int)
        self.mjds = self.catalogue_file["mjds"][0, :].astype(np.int)
        self.fiber_ids = self.catalogue_file["fiber_ids"][0, :].astype(np.int)

        self.plates = self.plates[self.test_ind]
        self.mjds = self.mjds[self.test_ind]
        self.fiber_ids = self.fiber_ids[self.test_ind]

        # store small arrays
        self.z_qsos = self.catalogue_file["z_qsos"][0, :]
        self.z_qsos = self.z_qsos[self.test_ind]

        # [Occams Razor] Update model posteriors with an additional occam's razor
        # updating: 1) model_posteriors, p_dlas, p_no_dlas
        self.model_posteriors = self._occams_model_posteriors(
            self.model_posteriors, self.occams_razor
        )
        self.p_dlas = self.model_posteriors[:, 1 + self.sub_dla :].sum(axis=1)
        self.p_no_dlas = self.model_posteriors[:, : 1 + self.sub_dla].sum(axis=1)

        # build a MAP number of DLAs array
        # construct a reference array of model_posteriors in Roman's catalogue for computing ROC curve
        multi_p_dlas = self.model_posteriors  # shape : (num_qsos, 2 + num_dlas)

        dla_map_model_index = np.argmax(multi_p_dlas, axis=1)
        multi_p_dlas = multi_p_dlas[
            np.arange(multi_p_dlas.shape[0]), dla_map_model_index
        ]

        # get the number of DLAs with the highest val in model_posteriors
        dla_map_num_dla = dla_map_model_index
        if self.sub_dla:
            # (no dla, sub dla, DLA(1), DLA(2), ..., DLA(k))
            dla_map_num_dla = dla_map_num_dla - self.sub_dla
            dla_map_num_dla[dla_map_num_dla < 0] = 0

        self.multi_p_dlas = multi_p_dlas
        self.dla_map_model_index = dla_map_model_index
        self.dla_map_num_dla = dla_map_num_dla

        # construct an array based on model_posteriors, the array should be
        # [ {P(Mdla(m) | D)}j=1^m, P(Mdla(m+1) | D), ..., P(Mdla(k) | D) ]
        # This should be the reference p_dlas for multi-DLAs if we divide each los into num_dla pieces.
        model_posteriors_dla = np.copy(
            self.model_posteriors[:, 1 + self.sub_dla :].T
        )  # shape : (num_dlas, num_qsos)
        num_models = model_posteriors_dla.shape[0]
        num_qsos = model_posteriors_dla.shape[1]

        # build a mask array for assign P(Mdla(m) | D) until `m`th index
        indices = np.arange(1, num_models + 1, dtype=np.int8)[:, None] * np.ones(
            num_qsos, dtype=np.int8
        )
        multi_dla_map_num_dla = self.dla_map_num_dla * np.ones(
            (num_models, num_qsos), dtype=np.int8
        )  # shape : (num_dlas, num_qsos) )
        multi_highest_posterior = self.multi_p_dlas * np.ones(
            (num_models, num_qsos), dtype=np.int8
        )
        multi_p_no_dlas = self.p_no_dlas * np.ones(
            (num_models, num_qsos), dtype=np.int8
        )
        mask_inds = indices <= multi_dla_map_num_dla

        model_posteriors_dla[mask_inds] = multi_highest_posterior[mask_inds]
        self.model_posteriors_dla = model_posteriors_dla
        self.multi_p_no_dlas = multi_p_no_dlas

        # [filtering] Filter out the spectra based on the given condition
        self.filter_dla_spectra()

    def filter_dla_spectra(
        self,
    ):
        """
        Filter DLA based on the given condition.
        """
        # just re-write whole bunch of arrays here
        self.model_posteriors = self.model_posteriors[self.condition]
        self.p_dlas = self.p_dlas[self.condition]
        self.p_no_dlas = self.p_no_dlas[self.condition]
        self.log_priors_dla = self.log_priors_dla[self.condition]
        self.min_z_dlas = self.min_z_dlas[self.condition]
        self.max_z_dlas = self.max_z_dlas[self.condition]
        if "MAP_log_nhis" in self.processed_file.keys():
            self.map_z_dlas = self.map_z_dlas[self.condition]
            self.map_log_nhis = self.map_log_nhis[self.condition]

        # catalog file
        self.snrs = self.snrs[self.condition]
        self.thing_ids = self.thing_ids[self.condition]
        self.plates = self.plates[self.condition]
        self.mjds = self.mjds[self.condition]
        self.fiber_ids = self.fiber_ids[self.condition]
        self.z_qsos = self.z_qsos[self.condition]

        # some derived arrays
        self.multi_p_dlas = self.multi_p_dlas[self.condition]
        self.dla_map_model_index = self.dla_map_model_index[self.condition]
        self.dla_map_num_dla = self.dla_map_num_dla[self.condition]
        self.model_posteriors_dla = self.model_posteriors_dla.T[self.condition].T
        self.multi_p_no_dlas = self.multi_p_no_dlas.T[self.condition].T
        # [map] reprepare the MAP values after filtering
        self.prepare_map_vals()
        # [test_real_index] update this array to be test_ind[condition]
        # since find_this_flux is using this array
        self.test_real_index = self.test_real_index[self.condition]

    def eBOSS_cut(
        self,
        is_qso_final_cut: bool,
        class_person_cut: bool,
        z_source_cut: bool,
    ):
        """
        Apply some cuts defined in the eBOSS paper but did not implemented in build_catalog.mat
        """

        # the best redshift measurement column, only in DR16Q
        z_pca = self.hdu[1].data["Z_PCA"][self.test_ind]
        # make sure the distfile is the same as the one we used for catalog.mat
        assert np.all(np.abs(z_pca - self.z_qsos) < 1e-3)

        if is_qso_final_cut:
            is_qso_final = self.hdu[1].data["IS_QSO_FINAL"][self.test_ind]
            # IS_QSO_FINAL: -2 ~ 2
            # QSO: 1; Questionable QSO: 2
            # non QSO: -2 ~ 0
            # Details see 3.6 section of eBOSS DR16Q paper
            is_qso_condition = is_qso_final == 1

            print(
                "[Info] {} -> {} after setting IS_QSO_FINAL == 1.".format(
                    np.sum(self.condition), np.sum(is_qso_condition * self.condition)
                )
            )
            self.condition = self.condition * is_qso_condition

        if class_person_cut:
            class_person = self.hdu[1].data["CLASS_PERSON"][self.test_ind]
            # visual inspection classification of BAL
            # 0: no inspected; 1: star; 3: Quasar; 4: Galaxy
            # 30: BAL Quasar; 50: Blazar (?)
            # Details see Table 2 and Section 3.5 in DR16Q
            #
            # We aim to avoid BAL Quasars but keep those non-inspected
            human_qso_condition = ~((class_person != 3) & (class_person != 0))
            assert np.sum(human_qso_condition) == (
                np.sum(class_person == 3) + np.sum(class_person == 0)
            )

            print(
                "[Info] {} -> {} after setting class_person == 3 or 0.".format(
                    np.sum(self.condition), np.sum(human_qso_condition * self.condition)
                )
            )
            self.condition = self.condition * human_qso_condition

        if z_source_cut:
            source_z = self.hdu[1].data["SOURCE_Z"][self.test_ind]
            # Section 3.4: Z > 5 and source_z == "pipe" should be considered suspect
            z_source_condition = ~((source_z == "PIPE") & (z_pca > 5))

            print(
                "[Info] {} -> {} after filtering out SOURCE_Z: PIPE and Z > 5".format(
                    np.sum(self.condition), np.sum(z_source_condition * self.condition)
                )
            )
            self.condition = self.condition * z_source_condition

    def zestimation_cut(self, delta_z_qso: float = 0.1):
        """
        Cut spectra with zQSO disagreements
        """
        # the best redshift measurement column, only in DR16Q
        z_best = self.hdu[1].data["Z"][self.test_ind]

        z_pca = self.hdu[1].data["Z_PCA"][self.test_ind]
        z_pipe = self.hdu[1].data["Z_PIPE"][self.test_ind]
        z_vi = self.hdu[1].data["Z_VI"][self.test_ind]

        # make sure the distfile is the same as the one we used for catalog.mat
        assert np.all(np.abs(z_pca - self.z_qsos) < 1e-3)

        all_z_methods = [z_pca, z_pipe, z_vi]
        z_condition = np.ones_like(z_pca, dtype=np.bool)
        for z_method in all_z_methods:
            ind = np.abs(z_pca - z_method) < delta_z_qso

            ignore_ind = z_method == -1
            ind[ignore_ind] = True

            z_condition = ind * z_condition

        print(
            "[Info] {} -> {} after filtering out uncertain z measures.".format(
                np.sum(self.condition), np.sum(z_condition)
            )
        )
        self.z_condition = z_condition
        self.condition = self.condition * z_condition

    @staticmethod
    def prediction_parks2dict(dist_file: str, selected_thing_ids=None) -> Dict:
        """
        Get Parks' CNN predictions on DR16Q catalogue

        Split a single spectrum to several DLAs.

        No DLA for NaN in dla_confidence

        Z_PCA for zQSO.
        """
        hdu = fits.open(dist_file)

        # extract DLA information (exclude subDLA, lyb)
        ras = []
        decs = []
        plates = []
        mjds = []
        fiber_ids = []
        z_qsos = []
        dla_confidences = []
        z_dlas = []
        log_nhis = []
        thing_ids = []

        num_quasars = hdu[1].data["THING_ID"].shape[0]

        for i in range(num_quasars):
            # find number DLAs
            conf_dla = hdu[1].data["CONF_DLA"][i, :]
            num_dlas = np.sum(conf_dla != -1)

            assert num_dlas >= 0
            assert num_dlas <= 5

            # [thing_id] check if the thing_id is in the selected
            thing_id = hdu[1].data["THING_ID"][i]
            if thing_id not in selected_thing_ids:
                continue

            # has dla(s)
            if num_dlas > 0:
                for j in range(num_dlas):
                    # append basic quasar info
                    thing_ids.append(hdu[1].data["THING_ID"][i])
                    ras.append(hdu[1].data["RA"][i])
                    decs.append(hdu[1].data["DEC"][i])
                    plates.append(hdu[1].data["PLATE"][i])
                    mjds.append(hdu[1].data["MJD"][i])
                    fiber_ids.append(hdu[1].data["FIBERID"][i])
                    z_qsos.append(hdu[1].data["Z_PCA"][i])

                    # append the object (dla or lyb or subdla) info
                    dla_confidences.append(hdu[1].data["CONF_DLA"][i, j])
                    z_dlas.append(hdu[1].data["Z_DLA"][i, j])
                    log_nhis.append(hdu[1].data["NHI_DLA"][i, j])

                    assert dla_confidences[-1] >= 0
                    assert dla_confidences[-1] <= 1
                    assert log_nhis[-1] >= 20.3
                    assert z_qsos[-1] >= 0

            elif num_dlas == 0:
                # append basic quasar info
                thing_ids.append(hdu[1].data["THING_ID"][i])
                ras.append(hdu[1].data["RA"][i])
                decs.append(hdu[1].data["DEC"][i])
                plates.append(hdu[1].data["PLATE"][i])
                mjds.append(hdu[1].data["MJD"][i])
                fiber_ids.append(hdu[1].data["FIBERID"][i])
                z_qsos.append(hdu[1].data["Z_PCA"][i])

                # append the object (dla or lyb or subdla) info
                dla_confidences.append(np.nan)
                z_dlas.append(np.nan)
                log_nhis.append(np.nan)

        dict_parks = {
            "thing_ids": np.array(thing_ids).astype(np.int),
            "ras": np.array(ras),
            "decs": np.array(decs),
            "plates": np.array(plates).astype(np.int),
            "mjds": np.array(mjds).astype(np.int),
            "fiber_ids": np.array(fiber_ids).astype(np.int),
            "z_qso": np.array(z_qsos),
            "dla_confidences": np.array(dla_confidences),
            "z_dlas": np.array(z_dlas),
            "log_nhis": np.array(log_nhis),
        }

        return dict_parks

    def prediction_solene2dict(self, dist_file: str, our_sightlines: bool = False, voigt_fitter: bool = False) -> Dict:
        """
        Get Solene's CNN predictions on DR16Q catalogue

        TODO:
        Need to load both NHI_CNN and NHI_FIT in order to plot CDDFs using different NHIs.
        Need to get thingIDs from DR16Q and filtering using Solene's conditions (Given as follows).
        This is due to Solene's catalogue only includes positive DLA detections, not negative detections.
        But to calculate dX (total path length), sightlines with negative detections are required.

        DLA_CAT_SDSS_DR16.fits (Chabanier+21)
        *************************************************************************************
        Results of the DLA search using the CNN from Parks+18 on the 263,201 QSO spectra from
        the SDSS-IV quasar catalog from DR16 (Lyke+20) with 2 <= Z_QSO <= 6 and Z_WARNING !=
        SKY, LITTLE_COVERAGE, UNPLUGGED, BAD_TARGET or NODATA.

        The fits file contains an array of the 117,458 detected absorbers in sightlines with BAL_PROB =
        0 and absorbing redshift Z_CNN >= 2.

        Some additional information
        Z_QSO: using Z_LYAWG
        NHI_FIT: The logarithm of the absorber column density as found by the Voigt-profile fitter for
            absorbers in the rest-frame range $1,040 \angstrom \leq \lambda_{\rm RF} \leq 1,216
            \angstrom$ and $\log \nhi < 22$. This parameter is set to -1 for absorbers that do not
            meet the criteria
        Z_CNN: the absorber redshift as found by the CNN
        NHI_CNN: the logarithm of the absorber column density as found by the CNN
        CONF_CNN: confidence parameter, between 0 and 1. Absorbers with confidence > 0.5 are
            considered as confident absorbers.
        """
        hdu_solene = fits.open(dist_file)

        # extract DLA information: Solene's catalogue allows duplicate THING_IDs for
        # multiple DLAs at the same sightline. We will need to append non-DLA sightline
        # from DR16Q afterward.
        ras = hdu_solene[1].data["RA"]
        decs = hdu_solene[1].data["DEC"]
        plates = hdu_solene[1].data["PLATE"]
        mjds = hdu_solene[1].data["MJD"]
        fiber_ids = hdu_solene[1].data["FIBERID"]
        z_qsos = hdu_solene[1].data["Z_QSO"]
        thing_ids = hdu_solene[1].data["THING_ID"]

        dla_confidences = hdu_solene[1].data["CONF_CNN"]
        z_dlas = hdu_solene[1].data["Z_CNN"]
        log_nhis = hdu_solene[1].data["NHI_CNN"]
        log_nhis_fitter = hdu_solene[1].data["NHI_FIT"]

        assert np.all(dla_confidences >= 0)
        assert np.all(dla_confidences <= 1)
        assert np.all(log_nhis >= 19.5)
        assert np.all(z_dlas >= 2)

        # TODO: Check again here if results are not sensible. Not sure if -1 DLAs are
        #   considered to be null detections.
        #   Use Voigt fitter as NHI: only a subset of DLAs has applied Voigt fit
        #   DLAs do not fit the fitting criteria as set with -1
        if voigt_fitter:
            ind_no_fitter = (log_nhis_fitter == -1)

            ras = ras[~ind_no_fitter]
            decs = decs[~ind_no_fitter]
            plates = plates[~ind_no_fitter]
            mjds = mjds[~ind_no_fitter]
            fiber_ids = fiber_ids[~ind_no_fitter]
            z_qsos = z_qsos[~ind_no_fitter]
            thing_ids = thing_ids[~ind_no_fitter]

            dla_confidences = dla_confidences[~ind_no_fitter]
            z_dlas = z_dlas[~ind_no_fitter]
            # log_nhis = log_nhis[~ind_no_fitter] # keep the CNN one for future comparison
            log_nhis_fitter = log_nhis_fitter[~ind_no_fitter]

            log_nhis = log_nhis_fitter

            assert np.all( log_nhis_fitter >= 19.3 ) # strangely the min of fitter is ~19.3

        # only append the thingIDs that are not in Solene's DLAs
        # 1) search Solene's DLA thingIDs in DR16Q
        # ind_solene_dla = np.isin(self.hdu[1].data["THING_ID"], thing_ids)
        # not using thingID because -1 duplicated
        unique_ids_dr16q = self.make_unique_id(self.hdu[1].data["PLATE"], self.hdu[1].data["MJD"], self.hdu[1].data["FIBERID"])
        unique_ids_solene_dlas = self.make_unique_id(plates, mjds, fiber_ids)
        ind_solene_dla = np.isin(unique_ids_dr16q, unique_ids_solene_dlas)
        # 2) DR16 sightlines with filtering condition set by Solene
        ind_solene_los = solene_eBOSS_cuts(
            self.hdu[1].data["Z_PCA"], # self.hdu[1].data["Z_LYAWG"], # TODO: why some Solene's DLAs don't have Z_LYAWG?
            self.hdu[1].data["ZWARNING"],
            self.hdu[1].data["BAL_PROB"],
        )
        assert np.all([
            bitget_string(format(z, "b"), bit) == False 
            for bit in (0, 1, 7, 8, 9)
            for z in self.hdu[1].data["ZWARNING"][ind_solene_los]
        ])
        # DLA set should be a subset of LOS set
        # import pdb
        # pdb.set_trace()
        # assert np.sum(~ind_solene_los & ind_solene_dla) == 0 # TODO: it never passes due to many -1 in ZWARNING
        # 3) GP's line of sights
        if our_sightlines:
            print("[Info] Load only our sightlines from Solene's.")
            # awkward way to apply condition to test_ind 
            ind_gp_los = self.test_ind
            real_index = np.where(self.test_ind)[0]
            ind_gp_los[real_index[~self.condition]] = False
            assert ind_gp_los.sum() == self.condition.sum()

            # only take the intersection of GP LOS and Solene LOS
            ind_solene_los = ind_solene_los & ind_gp_los
            # DLA sightlines should be a subset of solene LOS
            ind_solene_dla = ind_solene_dla & ind_gp_los

            # in Solene's DLAs, only keep our sightlines
            ind = np.isin(thing_ids, self.thing_ids)
            ras = ras[ind]
            decs = decs[ind]
            plates = plates[ind]
            mjds = mjds[ind]
            fiber_ids = fiber_ids[ind]
            z_qsos = z_qsos[ind]
            thing_ids = thing_ids[ind]

            dla_confidences = dla_confidences[ind]
            z_dlas = z_dlas[ind]
            log_nhis = log_nhis[ind]
            log_nhis_fitter = log_nhis_fitter[ind]

        # 4) add (non-DLA detection) sightlines to Solene's catalog
        #    because the catalog did not have null detections.
        ind_add_los = ind_solene_los & (~ind_solene_dla)

        ras = np.append(ras, self.hdu[1].data["RA"][ind_add_los])
        decs = np.append(decs, self.hdu[1].data["DEC"][ind_add_los])
        plates = np.append(plates, self.hdu[1].data["PLATE"][ind_add_los])
        mjds = np.append(mjds, self.hdu[1].data["MJD"][ind_add_los])
        fiber_ids = np.append(fiber_ids, self.hdu[1].data["FIBERID"][ind_add_los])
        z_qsos = np.append(z_qsos, self.hdu[1].data["Z_PCA"][ind_add_los]) # TODO: why some Solene's DLAs don't have Z_LYAWG?
        thing_ids = np.append(thing_ids, self.hdu[1].data["THING_ID"][ind_add_los])
        # append NaNs for LOS (non-detections)
        dla_confidences = np.append(dla_confidences, np.full((np.sum(ind_add_los), ), fill_value=np.nan))
        z_dlas = np.append(z_dlas, np.full((np.sum(ind_add_los), ), fill_value=np.nan))
        log_nhis = np.append(log_nhis, np.full((np.sum(ind_add_los), ), fill_value=np.nan))
        log_nhis_fitter = np.append(log_nhis_fitter, np.full((np.sum(ind_add_los), ), fill_value=np.nan))

        assert np.all(z_qsos >= 2)


        dict_solene = {
            "thing_ids": thing_ids.astype(np.int),
            "ras": ras,
            "decs": decs,
            "plates": plates.astype(np.int),
            "mjds": mjds.astype(np.int),
            "fiber_ids": fiber_ids.astype(np.int),
            "z_qso": z_qsos,
            "dla_confidences": dla_confidences,
            "z_dlas": z_dlas,
            "log_nhis": log_nhis,
            "log_nhis_fitter": log_nhis_fitter,
        }

        return dict_solene


    def _get_parks_estimations(
        self,
        dla_parks: str = "DR16Q_v4.fits", # DLA_CAT_SDSS_DR16.fits (Solene's catalog)
        p_thresh: float = 0.97,
        lyb: bool = False,
        min_1040: bool = False,
        search_range_from_ours: bool = False,
        solene_dlas: bool = False,
        voigt_fitter: bool = False,
        our_sightlines: bool = True, # only works for Solene's catalog
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Get z_dlas and log_nhis from Parks' (2018) or Solene's (2021) estimations
        """
        # make sure we are reading the DR16Q file
        # assert dla_parks in self.dist_file

        # voigt fitter only works for Solene's catalog
        # also, the search range is 1040 ~ 1216
        if voigt_fitter:
            assert solene_dlas
            assert (search_range_from_ours | lyb)

        # avoid running it twice
        if solene_dlas:
            print("[Info] Loading Solene's catalog")
            if "dict_solene" not in dir(self):
                self.dict_solene = self.prediction_solene2dict(
                    dla_parks, our_sightlines=our_sightlines, voigt_fitter=voigt_fitter,
                )

            cnn_catalog = self.dict_solene

            # assume log_nhis is overwritten by NHI_FIT
            if voigt_fitter:
                ind = np.isnan(cnn_catalog["log_nhis"])
                assert np.all(cnn_catalog["log_nhis"][~ind] == cnn_catalog["log_nhis_fitter"][~ind])
            else:
                ind = np.isnan(cnn_catalog["log_nhis"])
                assert ~np.all(cnn_catalog["log_nhis"][~ind] == cnn_catalog["log_nhis_fitter"][~ind])

        else:
            assert not our_sightlines
            # Here is assumed using our sightlines
            if "dict_parks" not in dir(self):
                self.dict_parks = self.prediction_parks2dict(
                    dla_parks, selected_thing_ids=self.thing_ids
                )

            cnn_catalog = self.dict_parks

        # use thing_ids as an identifier
        # use zPCA as zQSOs, as said in DR16Q paper
        thing_ids = cnn_catalog["thing_ids"]
        z_qsos = cnn_catalog["z_qso"]

        # acquire SNR values: if it's a subset of our DLAs, take our own SNRs;
        #   Otherwise, take it from DR16Q's SN_MEDIAN_ALL
        all_snrs = np.zeros(thing_ids.shape)
        # double-check if all sightlines are in our selection function
        if our_sightlines or not solene_dlas:
            assert np.all(np.isin(thing_ids, self.thing_ids))
            # for loop to get snrs from sbird's snrs file
            for i, thing_id in enumerate(thing_ids):
                real_index = np.where(self.thing_ids == thing_id)[0][0]
                all_snrs[i] = self.snrs[real_index]
        else:
            mask_ind = search_index_from_another(cnn_catalog["thing_ids"], self.hdu[1].data["THING_ID"])

            all_snrs = self.hdu[1].data["SN_MEDIAN_ALL"][mask_ind]

            # double-check the matching
            real_index = np.where(self.hdu[1].data["THING_ID"] == thing_ids[0])[0][0]
            assert self.hdu[1].data["SN_MEDIAN_ALL"][real_index] == all_snrs[0]

        # the searching range for the DLAs.
        if lyb:
            print("Searching min_z_dla: lyb ... ")
            min_z_dlas = (1 + z_qsos) * lyb_wavelength / lya_wavelength - 1
        elif min_1040:
            print("Searching min_z_dla: 1040 ... ")
            min_z_dlas = (1 + z_qsos) * 1040 / lya_wavelength - 1
        else:
            min_z_dlas = (1 + z_qsos) * lyman_limit / lya_wavelength - 1
        print("Searching max_z_dla: lya ... ")
        max_z_dlas = (1 + z_qsos) * lya_wavelength / lya_wavelength - 1

        # get search range from the processed file
        # redo the full searching range
        if search_range_from_ours:
            print("Using our own searching range")
            min_z_dlas = np.zeros(thing_ids.shape)
            max_z_dlas = np.zeros(thing_ids.shape)

            for i, thing_id in enumerate(thing_ids):
                real_index = np.where(self.thing_ids == thing_id)[0][0]

                min_z_dlas[i] = self.min_z_dlas[real_index]
                max_z_dlas[i] = self.max_z_dlas[real_index]

        # get DLA properties
        # note: the following indices are DLA-only
        dla_inds = (
            cnn_catalog["dla_confidences"] > 0.005
        )  # use p_thresh=0.005 to filter out non-DLA spectra and
        # speed up the computation

        thing_ids = thing_ids[dla_inds]
        log_nhis = cnn_catalog["log_nhis"][dla_inds]
        z_dlas = cnn_catalog["z_dlas"][dla_inds]
        z_qsos = cnn_catalog["z_qso"][dla_inds]
        p_dlas = cnn_catalog["dla_confidences"][dla_inds]
        snrs = all_snrs[dla_inds]

        dla_inds = p_dlas > p_thresh

        thing_ids = thing_ids[dla_inds]
        log_nhis = log_nhis[dla_inds]
        z_dlas = z_dlas[dla_inds]
        z_qsos = z_qsos[dla_inds]
        p_dlas = p_dlas[dla_inds]
        snrs = snrs[dla_inds]

        # get rid of z_dlas larger than z_qsos or lower than lyman limit
        if lyb:
            z_cut_inds = z_dlas > ((1 + z_qsos) * lyb_wavelength / lya_wavelength - 1)
        elif min_1040:
            z_cut_inds = z_dlas > ((1 + z_qsos) * 1040 / lya_wavelength - 1)
        else:
            z_cut_inds = z_dlas > ((1 + z_qsos) * lyman_limit / lya_wavelength - 1)

        z_cut_inds = np.logical_and(
            z_cut_inds, (z_dlas < ((1 + z_qsos) * lya_wavelength / lya_wavelength - 1))
        )

        if search_range_from_ours:
            # we are not sure if the search range is lyb in the processed file
            assert lyb is False and min_1040 is False

            # for loop to get min z_dlas and max z_dlas search range from processed data
            _min_z_dlas = np.zeros(thing_ids.shape)
            _max_z_dlas = np.zeros(thing_ids.shape)

            for i, thing_id in enumerate(thing_ids):
                real_index = np.where(self.thing_ids == thing_id)[0][0]

                _min_z_dlas[i] = self.min_z_dlas[real_index]
                _max_z_dlas[i] = self.max_z_dlas[real_index]

            # # Parks chap 3.2: fixed range of the sightline ranging from 900A-1346A in rest-frame
            # min_z_dlas = (1 + z_qsos) *  900   / lya_wavelength - 1
            # max_z_dlas = (1 + z_qsos) *  1346  / lya_wavelength - 1

            z_cut_inds = z_dlas >= _min_z_dlas
            z_cut_inds = np.logical_and(z_cut_inds, (z_dlas <= _max_z_dlas))

        thing_ids = thing_ids[z_cut_inds]
        log_nhis = log_nhis[z_cut_inds]
        z_dlas = z_dlas[z_cut_inds]
        z_qsos = z_qsos[z_cut_inds]
        p_dlas = p_dlas[z_cut_inds]
        snrs = snrs[z_cut_inds]

        self.cnn_catalog = {}

        self.cnn_catalog["cddf_thing_ids"] = thing_ids
        self.cnn_catalog["cddf_log_nhis"] = log_nhis
        self.cnn_catalog["cddf_z_dlas"] = z_dlas
        self.cnn_catalog["min_z_dlas"] = min_z_dlas
        self.cnn_catalog["max_z_dlas"] = max_z_dlas
        self.cnn_catalog["snrs"] = snrs
        self.cnn_catalog["all_snrs"] = all_snrs
        self.cnn_catalog["cddf_p_dlas"] = p_dlas
        self.cnn_catalog["p_thresh"] = p_thresh
        self.cnn_catalog["solene_dlas"] = solene_dlas

        return (
            thing_ids,
            log_nhis,
            z_dlas,
            min_z_dlas,
            max_z_dlas,
            snrs,
            all_snrs,
            p_dlas,
        )

    def plot_this_mu(
        self,
        nspec: int,
        suppressed: bool = True,
        num_voigt_lines: int = 3,
        num_forest_lines: int = 31,
        Parks: bool = False,
        Solene: bool = False,
        voigt_fitter: bool = False, # only for Solene == True
        dla_solene: str = "DLA_CAT_SDSS_DR16.fits",
        label:str = "",
        new_fig: bool = True,
        color: str = "red",
        conditional_gp: bool = False,
        color_cond: str = "lightblue",
        nth=None,
    ):
        """
        Plot the spectrum with the dla model

        Parameters:
        ----
        nspec (int) : index of the spectrum in the catalogue
        suppressed (bool) : apply Lyman series suppression to the mean-flux or not
        num_voigt_lines (int, min=1, max=31) : how many members of Lyman series in the DLA Voigt profile
        number_forest_lines (int) : how many members of Lymans series considered in the froest
        Parks (bool) : whether to plot Parks' results
        Solene (bool) : whether to plot Solene's results
        dla_solene (str) : if Solene=True, specify the path to Solene' `DLA_CAT_SDSS_DR16.fits`
        voigt_fitter (bool) : only works if Solene=True. Plot an additional line for the DLAs from
            Voigt fitter in Solene's catalogue.

        Returns:
        ----
        rest_wavelengths : rest wavelengths for the DLA model
        this_mu : the DLA model
        map_z_dlas : MAP z_dla values
        map_log_nhis : MAP log NHI values
        """
        if voigt_fitter:
            assert Solene

        # spec id
        plate, mjd, fiber_id = (
            self.plates[nspec],
            self.mjds[nspec],
            self.fiber_ids[nspec],
        )

        # for obs data
        this_wavelengths = self.find_this_wavelengths(nspec)
        this_flux = self.find_this_flux(nspec)
        this_noise_variance = self.find_this_noise_variance(nspec)
        this_pixel_mask = self.find_this_pixel_mask(nspec)

        this_rest_wavelengths = emitted_wavelengths(
            this_wavelengths, self.z_qsos[nspec]
        )

        # [meanflux] re-assign the map of meanflux tau and beta
        self.GP.tau_0_kim = 0.00554
        self.GP.beta_kim = 3.182
        # [building GP model] interpolating onto the spectrum
        self.GP.set_data(
            this_rest_wavelengths,
            this_flux,
            this_noise_variance,
            this_pixel_mask,
            self.z_qsos[nspec],
            build_model=True,
        )
        self.GP.num_forest_lines = num_forest_lines

        rest_wavelengths = self.GP.x
        this_mu_original = self.GP.this_mu

        # get the MAP DLA values
        if nth == None:
            nth = np.argmax(self.model_posteriors[nspec]) - 1 - self.sub_dla
        # [yes DLAs] find maps of DLAs and multiply onto the GP mean
        if nth >= 0:
            map_z_dlas = self.all_z_dlas[nspec, : (nth + 1)]
            map_log_nhis = self.all_log_nhis[nspec, : (nth + 1)]
            assert np.all(~np.isnan(map_z_dlas))

            # [meanflux] find the map of meanflux tau and beta
            map_tau_0, map_beta = self.find_meanflux_map_from_dla_map(nspec)
            print("[Info] map_tau_0 = {:.4g}, map_beta = {:.4g}".format(map_tau_0, map_beta))
            # [meanflux] re-assign the map of meanflux tau and beta
            self.GP.tau_0_kim = map_tau_0
            self.GP.beta_kim = map_beta
            # [building GP model] interpolating onto the spectrum
            self.GP.set_data(
                this_rest_wavelengths,
                this_flux,
                this_noise_variance,
                this_pixel_mask,
                self.z_qsos[nspec],
                build_model=True,
            )
            this_mu = self.GP.this_mu

            for map_z_dla, map_log_nhi in zip(map_z_dlas, map_log_nhis):
                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]),
                    10 ** map_log_nhi,
                    map_z_dla,
                    num_lines=num_voigt_lines,
                )

                this_mu = this_mu * absorption

        # [no DLA] find map of meanflux and apply on the GP mean
        elif nth < 0:
            print("[Info] no map values stored currently for null model ...")
            this_mu = self.GP.this_mu

        # get parks model
        if Parks:
            if not "dict_parks" in dir(self):
                self.dict_parks = self.prediction_parks2dict(self.dist_file, self.thing_ids)

            dict_parks = self.dict_parks

            # construct an array of unique ids for los
            self.unique_ids = self.make_unique_id(
                self.plates, self.mjds, self.fiber_ids
            )
            unique_ids = self.make_unique_id(
                dict_parks["plates"], dict_parks["mjds"], dict_parks["fiber_ids"]
            )
            assert unique_ids.dtype is np.dtype("int64")
            assert self.unique_ids.dtype is np.dtype("int64")

            uids = np.where(unique_ids == self.unique_ids[nspec])[0]

            this_parks_mu = this_mu_original
            dla_confidences_parks = []
            z_dlas_parks = []
            log_nhis_parks = []

            for uid in uids:
                z_dla_parks = dict_parks["z_dlas"][uid]
                log_nhi_parks = dict_parks["log_nhis"][uid]
                dla_confidence_parks = dict_parks["dla_confidences"][uid]

                dla_confidences_parks.append(dla_confidence_parks)
                z_dlas_parks.append(z_dla_parks)
                log_nhis_parks.append(log_nhi_parks)

                if np.any(np.isnan(dict_parks["dla_confidences"][uid])):
                    continue

                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]),
                    10 ** log_nhi_parks,
                    z_dla_parks,
                    num_lines=1,
                )

                this_parks_mu = this_parks_mu * absorption

        # get Solene model
        if Solene:
            if not "dict_solene" in dir(self):
                self.dict_solene = self.prediction_solene2dict(dla_solene, our_sightlines=True, voigt_fitter=False)

            dict_solene = self.dict_solene

            # construct an array of unique ids for los
            self.unique_ids = self.make_unique_id(
                self.plates, self.mjds, self.fiber_ids
            )
            unique_ids = self.make_unique_id(
                dict_solene["plates"], dict_solene["mjds"], dict_solene["fiber_ids"]
            )
            assert unique_ids.dtype is np.dtype("int64")
            assert self.unique_ids.dtype is np.dtype("int64")

            uids = np.where(unique_ids == self.unique_ids[nspec])[0]

            this_solene_mu = this_mu_original
            dla_confidences_solene = []
            z_dlas_solene = []
            log_nhis_solene = []

            for uid in uids:
                z_dla_solene = dict_solene["z_dlas"][uid]
                log_nhi_solene = dict_solene["log_nhis"][uid]
                dla_confidence_solene = dict_solene["dla_confidences"][uid]

                dla_confidences_solene.append(dla_confidence_solene)
                z_dlas_solene.append(z_dla_solene)
                log_nhis_solene.append(log_nhi_solene)

                if np.any(np.isnan(dict_solene["dla_confidences"][uid])):
                    continue

                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]),
                    10 ** log_nhi_solene,
                    z_dla_solene,
                    num_lines=1,
                )

                this_solene_mu = this_solene_mu * absorption

        # get Solene model's voigt fitter:
        # because we want it to be overplotted with Solene's DLAs
        if Solene and voigt_fitter:
            if not "dict_solene" in dir(self):
                self.dict_solene = self.prediction_solene2dict(dla_solene, our_sightlines=True, voigt_fitter=False)

            dict_solene = self.dict_solene

            # construct an array of unique ids for los
            self.unique_ids = self.make_unique_id(
                self.plates, self.mjds, self.fiber_ids
            )
            unique_ids = self.make_unique_id(
                dict_solene["plates"], dict_solene["mjds"], dict_solene["fiber_ids"]
            )
            assert unique_ids.dtype is np.dtype("int64")
            assert self.unique_ids.dtype is np.dtype("int64")

            uids = np.where(unique_ids == self.unique_ids[nspec])[0]

            this_solene_mu_fitter = this_mu_original

            dla_confidences_solene_fitter = []
            z_dlas_solene_fitter = []
            log_nhis_solene_fitter = []

            for uid in uids:
                z_dla_solene_fitter = dict_solene["z_dlas"][uid]
                log_nhi_solene_fitter = dict_solene["log_nhis_fitter"][uid]
                dla_confidence_solene_fitter = dict_solene["dla_confidences"][uid]

                dla_confidences_solene_fitter.append(dla_confidence_solene_fitter)
                z_dlas_solene_fitter.append(z_dla_solene_fitter)
                log_nhis_solene_fitter.append(log_nhi_solene_fitter)

                if np.any(np.isnan(dict_solene["dla_confidences"][uid])):
                    continue
                
                # if this DLA has not applied the fitter, it's -1
                if log_nhi_solene_fitter == -1:
                    continue

                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]),
                    10 ** log_nhi_solene_fitter,
                    z_dla_solene_fitter,
                    num_lines=1,
                )

                this_solene_mu_fitter = this_solene_mu_fitter * absorption

        # plt.figure(figsize=(16, 5))
        if new_fig:
            make_fig()
            plt.plot(
                this_rest_wavelengths,
                this_flux,
                label="normalized flux; spec-{}-{}-{}".format(plate, mjd, fiber_id),
                color="C0",
            )

        if Parks:
            plt.plot(
                rest_wavelengths,
                this_parks_mu,
                label=r"CNN: z_dlas = ({}); lognhis=({}); p_dlas=({})".format(
                    ",".join("{:.3g}".format(z) for z in z_dlas_parks),
                    ",".join("{:.3g}".format(n) for n in log_nhis_parks),
                    ",".join("{:.3g}".format(p) for p in dla_confidences_parks),
                ),
                color="orange",
            )

        if Solene:
            plt.plot(
                rest_wavelengths,
                this_solene_mu,
                label=r"Solene (NHI_CNN): z_dlas = ({}); lognhis=({}); p_dlas=({})".format(
                    ",".join("{:.3g}".format(z) for z in z_dlas_solene),
                    ",".join("{:.3g}".format(n) for n in log_nhis_solene),
                    ",".join("{:.3g}".format(p) for p in dla_confidences_solene),
                ),
                color="magenta",
            )

        if Solene and voigt_fitter:
            plt.plot(
                rest_wavelengths,
                this_solene_mu_fitter,
                label=r"Solene (NHI_FIT): z_dlas = ({}); lognhis=({}); p_dlas=({})".format(
                    ",".join("{:.3g}".format(z) for z in z_dlas_solene_fitter),
                    ",".join("{:.3g}".format(n) for n in log_nhis_solene_fitter),
                    ",".join("{:.3g}".format(p) for p in dla_confidences_solene_fitter),
                ),
                color="greenyellow",
            )


        if nth >= 0:
            plt.plot(
                rest_wavelengths,
                this_mu,
                label=label
                + r"$\mathcal{M}$"
                + r" DLA({n})".format(n=nth + 1)
                + ": {:.3g}; ".format(
                    self.model_posteriors[nspec, 1 + self.sub_dla + nth]
                )
                + "lognhi = ({}); ".format(
                    ",".join("{:.3g}".format(n) for n in map_log_nhis)
                )
                + "z_dlas = ({})".format(
                    ",".join("{:.3g}".format(z) for z in map_z_dlas)
                ),
                color=color,
            )
        else:
            plt.plot(
                rest_wavelengths,
                this_mu,
                label=label
                + r"$\mathcal{M}$"
                + r" DLA({n})".format(n=0)
                + ": {:.3g}".format(self.p_no_dlas[nspec]),
                color=color,
            )

        # [conditional GP]
        if conditional_gp:
            ind_1 = self.GP.x <= lya_wavelength
            y2 = self.GP.y[~ind_1]
            this_mu1 = self.GP.this_mu[ind_1]
            this_mu2 = self.GP.this_mu[~ind_1]
            this_M1 = self.GP.this_M[ind_1, :]
            this_M2 = self.GP.this_M[~ind_1, :]
            d1 = self.GP.v[ind_1] + self.GP.this_omega2[ind_1]
            d2 = self.GP.v[~ind_1] + self.GP.this_omega2[~ind_1]

            mu1, Sigma11 = conditional_mvnpdf_low_rank(
                y2, this_mu1, this_mu2, this_M1, this_M2, d1, d2
            )

            plt.plot(
                self.GP.x[ind_1],
                mu1[:, 0],
                label="conditional GP (suppressed)",
                color=color_cond,
            )

        plt.xlabel(r"Rest-frame Wavelength $(\AA)$")
        plt.ylabel(r"Normalized Flux")
        plt.legend()

        if nth >= 0:
            return rest_wavelengths, this_mu, map_z_dlas, map_log_nhis
        return rest_wavelengths, this_mu

    def find_meanflux_map_from_dla_map(self, nspec: int):
        """
        Find the meanflux MAP values reversely from logNHI MAP,
        assuming logNHI samples are unique.
        """
        # use the first DLA parameter as a proxy to find the max index
        map_log_nhi = self.all_log_nhis[nspec, 0]
        assert ~np.isnan(map_log_nhi)

        maxind = np.where(self.log_nhi_samples == map_log_nhi)[0][0]

        return self.tau_0_samples[maxind], self.beta_samples[maxind]

    def load_dla_parks(
        self,
        dla_parks: str = "DR16Q_v4.fits",
        p_thresh: float = 0.98,
        release: str = "dr12",
        multi_dla: bool = True,
        num_dla: int = 2,
    ):
        """
        load Parks CNN predictions out of DR16Q catalogue

        Note: we have to consider DLAs from the same sightlines as different objects

        Parameters:
        ----
        dla_parks (str) : the filename of DR16Q catalogue
        p_thresh (float): the minimum probability to be considered as a DLA in Parks CNN
        release (str)
        multi_dla (bool): whether or not we want to construct multi-dla index
        num_dla (int)   : number of dla we want to consider if we are considering multi-dlas

        Note:
        ---
        We only load those spectra in our GP catalogue
        """
        dict_parks = self.prediction_parks2dict(
            dla_parks, selected_thing_ids=self.thing_ids
        )

        # construct an array of unique ids for los
        # Note: keep spectral ID here to avoid thing_id to be -1
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids = self.make_unique_id(
            dict_parks["plates"], dict_parks["mjds"], dict_parks["fiber_ids"]
        )
        assert unique_ids.dtype is np.dtype("int64")
        assert self.unique_ids.dtype is np.dtype("int64")

        # TODO: make the naming of variables more consistent
        raw_unique_ids = unique_ids
        raw_thing_ids = dict_parks["thing_ids"]
        raw_z_dlas = dict_parks["z_dlas"]
        raw_log_nhis = dict_parks["log_nhis"]
        raw_dla_confidences = dict_parks["dla_confidences"]

        real_index_los = np.where(np.in1d(self.unique_ids, unique_ids))[0]

        unique_ids_los = self.unique_ids[real_index_los]
        thing_ids_los = self.thing_ids[real_index_los]
        assert (
            np.unique(unique_ids_los).shape[0] == unique_ids_los.shape[0]
        )  # make sure we don't double count los

        # construct an array of unique ids for dlas
        dla_inds = dict_parks["dla_confidences"] > p_thresh

        real_index_dla = np.where(np.in1d(self.unique_ids, unique_ids[dla_inds]))[
            0
        ]  # Note that in this step we lose
        # the info about multi-DLA since
        # we are counting based on los

        unique_ids_dla = self.unique_ids[real_index_dla]
        thing_ids_dla = self.thing_ids[real_index_dla]

        # Construct a list of sub-los index and dla detection based on sub-los.
        # This is a relatively complicate loop and it's hard to understand philosophically.
        # It's better to write an explaination in the paper.
        if multi_dla:
            self.multi_unique_ids = self.make_multi_unique_id(
                num_dla, self.plates, self.mjds, self.fiber_ids
            )
            multi_unique_ids = self.make_multi_unique_id(
                num_dla,
                dict_parks["plates"],
                dict_parks["mjds"],
                dict_parks["fiber_ids"],
            )  # note here some index repeated
            # more than num_dla times

            multi_real_index_los = np.where(
                np.in1d(self.multi_unique_ids, multi_unique_ids)
            )[
                0
            ]  # here we have a real_index array
            # exactly repeat num_dla times

            multi_unique_ids_los = self.multi_unique_ids[multi_real_index_los]

            self.multi_thing_ids = self.make_array_multi(num_dla, self.thing_ids)
            multi_thing_ids_los = self.multi_thing_ids[multi_real_index_los]

            # loop over unique_ids to assign DLA detection to sub-los
            # Note: here we ignore the z_dla of DLAs.
            dla_multi_inds = np.zeros(multi_unique_ids_los.shape, dtype=bool)
            for uid in np.unique(multi_unique_ids_los):
                k_dlas = (
                    dict_parks["dla_confidences"][unique_ids == uid] > p_thresh
                ).sum()

                k_dlas_val = np.zeros(num_dla, dtype=bool)
                k_dlas_val[:k_dlas] = True  # assigning True until DLA(k)

                # assign DLA detections to the unique_ids of sub-los
                dla_multi_inds[multi_unique_ids_los == uid] = k_dlas_val
                assert (
                    multi_unique_ids_los[multi_unique_ids_los == uid].shape[0]
                    == num_dla
                )

            multi_real_index_dla = multi_real_index_los[dla_multi_inds]
            multi_unique_ids_dla = multi_unique_ids_los[dla_multi_inds]
            multi_thing_ids_dla = multi_thing_ids_los[dla_multi_inds]

            # store data in named tuple under self
            dla_catalog = namedtuple(
                "dla_catalog_parks",
                [
                    "real_index",
                    "real_index_los",
                    "thing_ids",
                    "thing_ids_los",
                    "unique_ids",
                    "unique_ids_los",
                    "multi_real_index_dla",
                    "multi_real_index_los",
                    "multi_thing_ids_dla",
                    "multi_thing_ids_los",
                    "multi_unique_ids_dla",
                    "multi_unique_ids_los",
                    "release",
                    "num_dla",
                    "raw_unique_ids",
                    "raw_z_dlas",
                    "raw_log_nhis",
                    "raw_dla_confidences",
                ],
            )
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla,
                real_index_los=real_index_los,
                thing_ids=thing_ids_dla,
                thing_ids_los=thing_ids_los,
                unique_ids=unique_ids_dla,
                unique_ids_los=unique_ids_los,
                multi_real_index_dla=multi_real_index_dla,
                multi_real_index_los=multi_real_index_los,
                multi_thing_ids_dla=multi_thing_ids_dla,
                multi_thing_ids_los=multi_thing_ids_los,
                multi_unique_ids_dla=multi_unique_ids_dla,
                multi_unique_ids_los=multi_unique_ids_los,
                release=release,
                num_dla=num_dla,
                raw_unique_ids=raw_unique_ids,
                raw_z_dlas=raw_z_dlas,
                raw_log_nhis=raw_log_nhis,
                raw_dla_confidences=raw_dla_confidences,
            )

        else:
            dla_catalog = namedtuple(
                "dla_catalog_parks",
                [
                    "real_index",
                    "real_index_los",
                    "thing_ids",
                    "thing_ids_los",
                    "unique_ids",
                    "unique_ids_los",
                    "release",
                    "raw_unique_ids",
                    "raw_z_dlas",
                    "raw_log_nhis",
                    "raw_dla_confidences",
                ],
            )
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla,
                real_index_los=real_index_los,
                thing_ids=thing_ids_dla,
                thing_ids_los=thing_ids_los,
                unique_ids=unique_ids_dla,
                unique_ids_los=unique_ids_los,
                release=release,
                raw_unique_ids=raw_unique_ids,
                raw_z_dlas=raw_z_dlas,
                raw_log_nhis=raw_log_nhis,
                raw_dla_confidences=raw_dla_confidences,
            )

    def generate_json_catalogue(self, outfile: str = "predictions_DLAs_dr16q.json", flag_tail_dlas: bool = False, tail_zone: float = 0.1):
        """
        Generate the Multi-DLA catalogue in JSON format,
        which has the same form as Parks' (2018) product.

        Parameters:
        ---
        flag_tail_dlas: move tail detection to another column
        tail_zone: the dz + blue end that we want to flag as tail dlas

        Example template :
        ----
        [
            {
                "p_dla" : 0.9,
                "p_no_dla" : 0.1,
                "max_model_posteriors": 0.7,
                "num_dlas": 1,
                "dlas": [
                    {
                        "log_nhi": 20.78598574580358,
                        "z_dla": 2.2291207038222756
                    }
                ],
                "tail_dlas" : [
                    {
                        "log_nhi": 21,
                        "z_dla": 2.3,
                    }
                ],
                "min_z_dla": 2.0,
                "max_z_dla": 2.18,
                "ra": 9.2152,
                "snr": 1.23,
                "dec": -0.1659,
                "plate": 3586,
                "mjd": 55181,
                "fiber_id": 16,
                "thing_id": ,
                "z_qso": 2.190
            },
            ...
        ]
        """
        # initialise a list to store dicts
        # we will save each spectrum into a dict.
        # if there's any dla, we will just save the dlas as a list of
        # dicts as an item in the spectrum dict.
        predictions_DLAs = []

        # query the maximum values of the model posteriors per spectrum first
        model_index = self.model_posteriors.argmax(axis=1)
        max_model_posteriors = self.model_posteriors.max(axis=1)

        if self.sub_dla:
            # now we need to combine sub-DLAs with null model posterior
            # to get the mosterior of no-DLAs
            inds = model_index < (1 + self.sub_dla)
            max_model_posteriors[inds] = self.p_no_dlas[inds]

            # prepare to use model_index as num_dlas
            num_dlas = model_index - self.sub_dla
            num_dlas[num_dlas < 0] = 0  # num_dlas should be positive

        assert len(max_model_posteriors) == len(self.thing_ids)

        # store some arrays here to query later
        # you do the test_ind indicing after storing the HDF5 array
        # into memory. Reading from numpy array from memory is much
        # faster than you do IO from the file.
        ras = self.catalogue_file["ras"][0, :][self.test_real_index]
        decs = self.catalogue_file["decs"][0, :][self.test_real_index]

        assert len(ras) == len(self.thing_ids)

        for i, thing_id in enumerate(self.thing_ids):
            spec = dict()

            # you need to put .item() to convert numpy datatype to python datatype
            # since json doesn't support serialize numpy datatype
            spec["p_dla"] = self.p_dlas[i].item()
            spec["p_no_dla"] = self.p_no_dlas[i].item()
            spec["max_model_posterior"] = max_model_posteriors[i].item()
            spec["num_dlas"] = num_dlas[i].item()
            spec["min_z_dla"] = self.min_z_dlas[i].item()
            spec["max_z_dla"] = self.max_z_dlas[i].item()
            spec["snr"] = self.snrs[i].item()
            spec["ra"] = ras[i].item()
            spec["dec"] = decs[i].item()
            spec["plate"] = self.plates[i].item()
            spec["mjd"] = self.mjds[i].item()
            spec["fiber_id"] = self.fiber_ids[i].item()
            spec["thing_id"] = thing_id.item()
            spec["z_qso"] = self.z_qsos[i].item()

            dlas = []
            tail_dlas = []

            if num_dlas[i] > 0:
                this_map_z_dlas = self.map_z_dlas[i, (num_dlas[i] - 1), : num_dlas[i]]
                this_map_log_nhis = self.map_log_nhis[
                    i, (num_dlas[i] - 1), : num_dlas[i]
                ]

                assert len(this_map_z_dlas) == num_dlas[i]

                # append individual dlas
                for j in range(num_dlas[i]):
                    # move tail detection to another section
                    if flag_tail_dlas and (this_map_z_dlas[j] < spec["min_z_dla"] + tail_zone):
                        tail_dlas.append(
                            {"log_nhi": this_map_log_nhis[j], "z_dla": this_map_z_dlas[j]}
                        )
                    else:
                        dlas.append(
                            {"log_nhi": this_map_log_nhis[j], "z_dla": this_map_z_dlas[j]}
                        )

            spec["dlas"] = dlas
            if flag_tail_dlas:
                spec["tail_dlas"] = tail_dlas

            predictions_DLAs.append(spec)

        import json

        with open(outfile, "w") as json_file:
            json.dump(predictions_DLAs, json_file, indent=2)

        return predictions_DLAs
