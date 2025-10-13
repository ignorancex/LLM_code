"""
calibrators and adjusters of evidence processes that allow "lifting"
"""

import numpy as np

# extreme values
EPS = 10 ** -100
MAX_E = 10 ** 100


def e_to_p_calibrator(e: np.ndarray):
    """Calibrate an e-process into a p-process via f(e) = min(1/e, 1).

    This is the unique admissible e-to-p (deterministic) calibrator.
    """
    return np.minimum(np.divide(1, e), 1)


def p_to_e_calibrator(p: np.ndarray, kappa: float = None, use_zero: bool = False):
    """Calibrate a p-process into an e-process.

    Different admissible calibrators exist. The default option is given by

        C(p) = (1 - p + p * log(p)) / (p * log(1/p) ** 2).

    When kappa is specified (between 0 and 1), the function is

        C(p; kappa) = kappa * p ** (kappa - 1).
    
    If `use_zero`, then use Shafer et al. (2011)'s alternative ``zero'' calibrator (eq. 9),
        which is zero whenever 1/p is smaller than exp(1 + kappa) 
        but shrinks fast otherwise.
    """

    if use_zero:
        assert kappa is not None, "kappa must be specified for the zero calibrator"
        if kappa <= 0:
            raise ValueError("invalid value for kappa: must be positive if use_zero is True")
    elif kappa is not None and not (0 < kappa < 1):
        raise ValueError("invalid value for kappa: must be between 0 and 1 if specified")

    # handle zeros
    p = np.array(p)
    p = np.where(p > 0, p, EPS)

    # Shafer et al. (2011)'s alternative calibrator ("zero" calibrator)
    if use_zero:
        return np.where(
            p <= np.exp(-1 - kappa),
            kappa * (1 + kappa) ** kappa / np.maximum(p * np.log(1 / p) ** (1 + kappa), EPS),
            0,
        )

    # default calibrators
    if kappa:
        return kappa * p ** (kappa - 1)
    else:
        return np.where(
            p < 1,
            (1 - p + p * np.log(p)) / np.maximum(p * np.log(p) ** 2, EPS),
            0.5,
        )


def adjuster(
        e: np.ndarray, 
        use_maximum: bool = True, 
        kappa: float = None, 
        use_kv: bool = False,
        use_zero: bool = False,
):
    """Adjust the running maximum of an e-process into an e-process.

    Equivalent to:

        A(e_t*) = f(min(1, 1/e_t*))

    where e* is the running maximum up to t and f is any p-to-e calibrator.

    The default adjuster is the mixture adjuster defined in our paper:

        A(e) = (e - 1 - np.log(e)) / (np.log(e) ** 2).

    If `use_kv`, then use Koolen & Vovk (2014)'s adjuster:

        A(e) = e^2 * log(2) / ((1 + e) * log(1 + e) ** 2).

    If `use_zero`, then use Shafer et al. (2011)'s alternative ``zero'' adjuster,
        which grows linearly with e and dominates the default version
        but is zero for any value smaller than exp(1 + kappa).
    """
    if use_maximum:
        e = np.maximum.accumulate(e)

    # Koolen-Vovk (2014) adjuster
    if use_kv:
        return (e ** 2 * np.log(2)) / ((1 + e) * (np.log(1 + e)) ** 2)

    # zero adjuster (Shafer et al., 2011)
    if use_zero:
        e_clipped = np.maximum(e, np.exp(1 + kappa))
        return np.where(
            e >= np.exp(1 + kappa),
            (kappa * (1 + kappa) ** kappa) * e_clipped / np.log(e_clipped) ** (1 + kappa),
            0,
        )

    # default (mixture over kappas)
    if kappa is None:
        log_e = np.log(np.maximum(e, EPS))
        return np.where(
            e > 1,
            (e - 1 - log_e) / np.maximum(log_e ** 2, EPS),
            0.5,
        )
    
    # specific kappa in (0, 1)
    else:
        if kappa <= 0 or kappa >= 1:
            raise ValueError("invalid value for kappa: must be between 0 and 1")
        return kappa * e ** (1 - kappa)


def spine_adjuster(
        e: np.ndarray,
        kappa: float,
):
    """Adjust an e-process using the spine adjuster (Dawid et al., 2011a), 
    which takes in both the running maximum and the current value.
    
    For 0 <= kappa <= 1, the spine adjuster is given by

        A(e_t*, e_t) = kappa * (e_t*)^(1/2) + (1-kappa) * (e_t*)^(-kappa) * e_t.
    
    """
    assert 0 <= kappa < 1, "invalid value for kappa: must be between 0 and 1"

    e_max = np.maximum.accumulate(e)
    return kappa * e_max ** 0.5 + (1 - kappa) * e_max ** (-kappa) * e
