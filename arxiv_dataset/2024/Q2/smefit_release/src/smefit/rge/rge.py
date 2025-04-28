# -*- coding: utf-8 -*-
import pathlib
import pickle
import warnings
from copy import deepcopy
from functools import partial, wraps

import ckmutil.ckm
import jax.numpy as jnp
import numpy as np
import pandas as pd
import wilson
from numpy import ComplexWarning

from smefit import log
from smefit.loader import Loader
from smefit.rge.wcxf import inverse_wcxf_translate, wcxf_translate

### Patch of a CKM function, so that the CP violating
### phase is set to gamma and not computed explicitly
### See https://github.com/wilson-eft/wilson/issues/113#issuecomment-2179273979
### This needs to be done before the import of wilson

# copying so we keep the original function
ckm_tree = deepcopy(ckmutil.ckm.ckm_tree)
ckmutil.ckm.ckm_tree = partial(ckm_tree, delta_expansion_order=0)
### End of patch

##################### MONKEY PATCH
# switch off the SM - EFT mixing, since SMEFiT assumes that the
# RGE solution is linearised
# Keep a reference to the original beta function
original_beta = wilson.run.smeft.beta.beta


def beta_wrapper(C, HIGHSCALE=np.inf, *args, **kwargs):
    return original_beta(C, HIGHSCALE, *args, **kwargs)


wilson.run.smeft.beta.beta = beta_wrapper
wilson.run.smeft.beta.beta_array = partial(
    wilson.run.smeft.beta.beta_array, HIGHSCALE=np.inf
)
##################### END OF MONKEY PATCH

##################### MONKEY PATCH
# Patch smeftpar: we remove all dependence on SMEFT parameters
# in SM paramaters, otherwise the linear approximation is not valid
C_patch = {
    "phi": 0.0,
    "phiBox": 0.0,
    "phiD": 0.0,
    "phiWB": 0.0,
    "phiG": 0.0,
    "phiW": 0.0,
    "phiB": 0.0,
    "dphi": 0.0,
    "uphi": 0.0,
    "ephi": 0.0,
}
# Reference to the original smeftpar function
original_smeftpar = wilson.run.smeft.smpar.smeftpar


# Define the monkey-patched function
@wraps(original_smeftpar)
def patched_smeftpar(*args, **kwargs):
    # check if C is passed as a keyword argument
    if "C" in kwargs:
        kwargs["C"] = C_patch
    # otherwise, check if it is passed as a positional argument
    else:
        args = list(args)
        args[1] = C_patch

    return original_smeftpar(*args, **kwargs)


# Apply the monkey patch
wilson.run.smeft.smpar.smeftpar = patched_smeftpar
##################### END OF MONKEY PATCH


##################### MONKEY PATCH
# Monkey patch flavour rotation
# Define the new method
def _to_wcxf_no_rotation(self, C_out, scale_out):
    """Return the Wilson coefficients `C_out` as a wcxf.WC instance, without rotation."""
    # Skip the self._rotate_defaultbasis line
    d = wilson.util.smeftutil.arrays2wcxf_nonred(C_out)
    d = wilson.wcxf.WC.dict2values(d)
    wc = wilson.wcxf.WC("SMEFT", "Warsaw", scale_out, d)
    return wc


# Monkey-patch the method
wilson.run.smeft.classes.SMEFT._to_wcxf = _to_wcxf_no_rotation
##################### END OF MONKEY PATCH

# Suppress the ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)

_logger = log.logging.getLogger(__name__)

###########################
# Input parameter options #
###########################

# copying so we could use the default parameters later
default_params = wilson.run.smeft.smpar.p.copy()

top_yukawa = {
    "Vus": 0.0,
    "Vub": 0.0,
    "Vcb": 0.0,
    "gamma": 0.0,
    "m_b": 0.0,
    "m_s": 0.0,
    "m_c": 0.0,
    "m_u": 0.0,
    "m_d": 0.0,
    "m_e": 0.0,
    "m_mu": 0.0,
    "m_tau": 0.0,
}

no_yukawa = {
    "Vus": 0.0,
    "Vub": 0.0,
    "Vcb": 0.0,
    "gamma": 0.0,
    "m_b": 0.0,
    "m_s": 0.0,
    "m_c": 0.0,
    "m_u": 0.0,
    "m_d": 0.0,
    "m_e": 0.0,
    "m_mu": 0.0,
    "m_tau": 0.0,
    "m_t": 0.0,
}

QCD_only = {
    "alpha_e": 0.0,
    "m_W": 1e-20,
    "m_h": 1e-20,
}

# gs at MZ
alpha_s = 0.118
gs = np.sqrt(4 * np.pi * alpha_s)


def evolve_gs(scale):
    # evolve gs from MZ to scale
    beta0 = 11 - 2 / 3 * 6
    return gs / np.sqrt(
        1 + 2 * beta0 * gs**2 / (4 * np.pi) ** 2 * np.log(scale / 91.1876)
    )


class RGE:
    """
    Class to compute the RGE matrix for the SMEFT Wilson coefficients.
    The RGE matrix is computed at the initial scale `init_scale` and
    evolved to the scale of interest.

    Parameters
    ----------
    wc_names: list
        list of Wilson coefficient names to be included in the RGE matrix
    init_scale: float
        initial scale of the Wilson coefficients
    accuracy: str
        accuracy of the RGE integration. Options: "leadinglog" or "integrate".
        Default is 'integrate'. Inherited behaviour from wilson package.
    adm_QCD: bool
        if True, only the QCD anomalous dimension is used. Default is False.
    yukawa: str
        Yukawa parameterization to be used. Options: "top", "none" or "full".
        Default is "top".
    """

    def __init__(
        self,
        wc_names,
        init_scale,
        accuracy="integrate",
        adm_QCD=False,
        yukawa="top",
    ):
        # order the Wilson coefficients alphabetically
        self.wc_names = sorted(wc_names)
        self.init_scale = init_scale
        self.accuracy = accuracy

        # set the anomalous dimension matrix parameters
        if yukawa == "top":
            wilson.run.smeft.smpar.p.update(**top_yukawa)
        elif yukawa == "none":
            wilson.run.smeft.smpar.p.update(**no_yukawa)
        elif yukawa == "full":
            wilson.run.smeft.smpar.p.update(**default_params)
        else:
            raise ValueError(f"Yukawa parameter not supported: {yukawa}")

        _logger.info(f"Using Yukawa parameterization: {yukawa}.")

        if adm_QCD:
            wilson.run.smeft.smpar.p.update(**QCD_only)
            _logger.info("Using anomalous dimension order: QCD.")
        else:
            _logger.info("Using anomalous dimension order: full.")

        _logger.info(
            f"Initializing RGE runner with initial scale {init_scale} GeV and accuracy {accuracy}."
        )

    def RGEmatrix_dict(self, scale):
        """
        Compute the RGE solution at the scale `scale` and return it as a dictionary.
        """
        # compute the RGE matrix at the scale `scale`
        rge_matrix_dict = {}
        for wc_name, wc_vals in self.RGEbasis.items():
            _logger.info(f"Computing RGE for {wc_name} at {scale} GeV.")
            wc_init = wilson.Wilson(
                wc_vals, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
            )
            wc_init.set_option("smeft_accuracy", self.accuracy)

            wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw")

            # Remove small values
            wc_final_vals = {
                key: value for key, value in wc_final.dict.items() if abs(value) > 1e-10
            }

            # check that imaginary values are small
            if any(abs(val.imag) > 1e-10 for val in wc_final_vals.values()):
                raise ValueError(
                    f"Imaginary values in Wilson coefficient for operator {wc_name}."
                )

            rge_matrix_dict[wc_name] = self.map_to_smefit(wc_final_vals, scale)

        return rge_matrix_dict

    def RGEmatrix(self, scale):
        """
        Compute the RGE solution at the scale `scale` and return it as a pandas DataFrame.
        """
        # compute the RGE matrix dict at the scale `scale`
        rge_matrix_dict = self.RGEmatrix_dict(scale)

        # create the RGE matrix as pandas dataframe
        rge_matrix = pd.DataFrame(
            columns=self.wc_names, index=self.all_ops, dtype=float
        )

        for wc_name, wc_dict in rge_matrix_dict.items():
            for op in self.all_ops:
                rge_matrix.loc[op, wc_name] = wc_dict.get(op, 0.0)
        # if there are rows with all zeros, remove them
        rge_matrix = rge_matrix.loc[(rge_matrix != 0).any(axis=1)]

        return rge_matrix

    @property
    def RGEbasis(self):
        """
        Returns the RGE basis translated from smefit to Warsaw.
        """
        # computes the translation from the smefit basis to the Warsaw basis
        # as expected by the Wilson package
        wc_basis = {}
        for wc_name in self.wc_names:
            try:
                wcxf_dict = wcxf_translate[wc_name]
            except KeyError:
                _logger.warning(
                    f"Wilson coefficient {wc_name} not present in the WCxf translation dictionary."
                )
                _logger.warning(
                    "Assuming it is a UV coupling and associating it to the null vector."
                )
                wc_basis[wc_name] = {}
                continue

            wc_warsaw_name = wcxf_dict["wc"]
            if "value" not in wcxf_dict:
                wc_warsaw_value = [1] * len(wcxf_dict["wc"])
            else:
                # check if value is gs
                # (this is a special case for OtG)
                if wcxf_dict["value"] == ["gs"]:
                    wc_warsaw_value = [evolve_gs(self.init_scale)]
                else:
                    wc_warsaw_value = wcxf_dict["value"]

            # 1e-6 is because the Warsaw basis is in GeV^-2
            wc_value = {
                wc: val * 1e-6 for wc, val in zip(wc_warsaw_name, wc_warsaw_value)
            }
            wc_basis[wc_name] = wc_value

        return wc_basis

    def map_to_smefit(self, wc_final_vals, scale):
        """
        Map the Wilson coefficients from the Warsaw basis to the SMEFiT basis.
        """
        wc_dict = {}
        for wc_basis, wc_inv_dict in inverse_wcxf_translate.items():
            wc_warsaw_name = wc_inv_dict["wc"]
            if "coeff" not in wc_inv_dict:
                wc_warsaw_coeff = [1] * len(wc_warsaw_name)
            else:
                # check if coeff is 1/gs
                # (this is a special case for OtG)
                if wc_inv_dict["coeff"] == ["1/gs"]:
                    wc_warsaw_coeff = [1 / evolve_gs(scale)]
                else:
                    wc_warsaw_coeff = wc_inv_dict["coeff"]

            value = 0.0
            for wc, coeff in zip(wc_warsaw_name, wc_warsaw_coeff):
                if wc in wc_final_vals:
                    # 1e6 is to transform from GeV^-2 to TeV^2
                    value += 1e6 * wc_final_vals[wc].real * coeff
            wc_dict[wc_basis] = value
        return wc_dict

    @property
    def all_ops(self):
        return sorted(wcxf_translate.keys())

    def RGEevolve(self, wcs, scale):
        """
        Evolve the Wilson coefficients from the initial scale to the scale of interest.
        """
        wc_wilson = {}
        for op, values in self.RGEbasis.items():
            for key in values:
                if key not in wc_wilson:
                    wc_wilson[key] = values[key] * wcs[op]
                else:
                    wc_wilson[key] += values[key] * wcs[op]

        wc_init = wilson.Wilson(
            wc_wilson, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
        )
        wc_init.set_option("smeft_accuracy", self.accuracy)
        wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw").dict

        # remove small values
        wc_final = {key: value for key, value in wc_final.items() if abs(value) > 1e-10}

        return self.map_to_smefit(wc_final, scale)


def load_scales(
    datasets, theory_path, default_scale=1e3, cutoff_scale=None, scale_variation=1.0
):
    """
    Load the energy scales for the datasets.

    Parameters
    ----------
    datasets: list
        list of datasets
    theory_path: str
        path to the theory files
    default_scale: float
        default scale to use if the dataset does not have a scale
    cutoff_scale: float
        cutoff scale to use for the scales

    Returns
    -------
    scales: list
        list of scales for the datasets
    """
    scales = []
    for dataset in datasets:

        Loader.theory_path = pathlib.Path(theory_path)
        # dummy call just to get the scales
        _, _, _, _, dataset_scales = Loader.load_theory(
            dataset.get("name"),
            operators_to_keep={},
            order="LO",
            use_quad=False,
            use_theory_covmat=False,
            use_multiplicative_prescription=False,
        )
        # check that dataset_scales is not a list filled with None
        # otherwise, assume the initial scale
        if not all(scale is None for scale in dataset_scales):
            scales.extend(dataset_scales)
        else:
            scales.extend([default_scale] * len(dataset_scales))

        _logger.info(f"Loaded scales for dataset {dataset['name']}: {dataset_scales}")

    if cutoff_scale is not None:
        scales = [scale for scale in scales if scale < cutoff_scale]
    if scale_variation != 1.0:
        _logger.info(f"Applying scale variation of {scale_variation}.")
        scales = [scale * scale_variation for scale in scales]

    return scales


def load_rge_matrix(
    rge_dict,
    coeff_list,
    datasets=None,
    theory_path=None,
    cutoff_scale=None,
    result_path=None,
    result_ID=None,
):
    """
    Load the RGE matrix for the SMEFT Wilson coefficients.

    Parameters
    ----------
    rge_dict: dict
        dictionary with the RGE input parameter options
    coeff_list: list
        list of Wilson coefficients to be included in the RGE matrix
    datasets: list
        list of datasets
    theory_path: str
        path to the theory files

    Returns
    -------
    rgemat: numpy.ndarray
        RGE matrix
    """
    init_scale = rge_dict.get("init_scale", 1e3)
    obs_scale = rge_dict.get("obs_scale", 91.1876)
    smeft_accuracy = rge_dict.get("smeft_accuracy", "integrate")
    adm_QCD = rge_dict.get("adm_QCD", "full")
    yukawa = rge_dict.get("yukawa", "top")
    scale_variation = rge_dict.get("scale_variation", 1.0)
    rge_runner = RGE(coeff_list, init_scale, smeft_accuracy, adm_QCD, yukawa)

    # load precomputed RGE matrix if it exists
    path_to_rge_mat = rge_dict.get("rg_matrix", False)
    if path_to_rge_mat:
        with open(path_to_rge_mat, "rb") as f:
            rgemats = pickle.load(f)
        stacked_mats = jnp.stack([rgemat.values for rgemat in rgemats])
        operators_to_keep = {op: {} for op in rgemats[0].index}
        return stacked_mats, operators_to_keep

    # if it is a float, it is a static scale
    if type(obs_scale) is float or type(obs_scale) is int:
        rgemat = rge_runner.RGEmatrix(obs_scale)
        gen_operators = list(rgemat.index)
        operators_to_keep = {k: {} for k in gen_operators}

        # prepend additional dimension for consistency with the dynamic scale case
        stacked_mats = jnp.stack([rgemat.values])
        save_rg(pathlib.Path(result_path) / result_ID, rgemat=[rgemat])

        return stacked_mats, operators_to_keep

    elif obs_scale == "dynamic":
        scales = load_scales(
            datasets,
            theory_path,
            default_scale=init_scale,
            cutoff_scale=cutoff_scale,
            scale_variation=scale_variation,
        )

        operators_to_keep = {}
        rgemat = []
        rge_cache = {}
        for scale in scales:
            # Check if the RGE matrix has already been computed
            if scale in rge_cache:
                rgemat_scale = rge_cache[scale]
            else:
                rgemat_scale = rge_runner.RGEmatrix(scale)
                gen_operators = list(rgemat_scale.index)
                # Fill with the operators if not already present in the dictionary
                for op in gen_operators:
                    if op not in operators_to_keep:
                        operators_to_keep[op] = {}
                rge_cache[scale] = rgemat_scale

            rgemat.append(rgemat_scale)

        # now loop through the rgemat and if there are operators
        # that are not present in the matrix, fill them with zeros
        for mat in rgemat:
            for op in operators_to_keep:
                if op not in mat.index:
                    mat.loc[op] = np.zeros(len(mat.columns))
            # order the rows alphabetically in the index
            mat.sort_index(inplace=True)

        # now stack the matrices in a 3D array
        stacked_mats = jnp.stack([mat.values for mat in rgemat])

        # save RGE matrix to result_path
        save_rg(pathlib.Path(result_path) / result_ID, rgemat=rgemat)

        return stacked_mats, operators_to_keep

    else:
        raise ValueError(
            f"obs_scale must be either a float/int or 'dynamic'. Passed: {obs_scale}"
        )


def save_rg(path, rgemat):
    """
    Save the RGE matrix to the result folder.

    Parameters
    ----------
    path : pathlib.Path
        path to the result folder
    rgemat: list
        List of RGE matrices for each datapoint
    """
    if path is not None:
        with open(path / "rge_matrix.pkl", "wb") as f:
            pickle.dump(rgemat, f)
