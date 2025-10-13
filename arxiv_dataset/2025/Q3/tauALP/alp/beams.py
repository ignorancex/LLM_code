import numpy as np

from scipy.stats import gamma

import vector
from . import const
from . import phase_space as ps

xsec_ccbar = {
    "120GeV": 2.2e-6,
    "400GeV": 20e-6,
    "13.6TeV": 0.5e-6,
}  # in cm^2
frag_fractions = {"Ds+": 0.081, "D+": 0.244, "D0": 0.606, "D-": 0.244}


def concatenate_vectors(p1, p2):
    """
    Concatenate two vector arrays along the first axis.
    """

    if len(p1["E"]) == 0:
        return p2
    if len(p2["E"]) == 0:
        return p1
    return vector.array(
        {
            "E": np.concatenate([p1["E"], p2["E"]]),
            "px": np.concatenate([p1["px"], p2["px"]]),
            "py": np.concatenate([p1["py"], p2["py"]]),
            "pz": np.concatenate([p1["pz"], p2["pz"]]),
        }
    )


def get_xF(p4, p_beam, CoM=False):
    n_events = p4.size
    p4_beam = vector.array(
        {
            "px": np.zeros(n_events),
            "py": np.zeros(n_events),
            "pz": np.ones(n_events) * p_beam,
            "E": np.sqrt(const.m_proton**2 + p_beam**2) * np.ones(n_events),
        }
    )

    if CoM:
        p4_CM = p4
        p4_beam_CM = p4_beam
    else:
        # Boost to the center of mass frame
        beta = (
            p_beam
            / (np.sqrt(p_beam**2 + const.m_proton**2) + const.m_proton)
            * np.ones(n_events)
        )
        p4_CM = p4.boostZ(-beta)
        p4_beam_CM = p4_beam.boostZ(-beta)

    return p4_CM["pz"] / p4_beam_CM["pz"]


def generate_Ds(p_beam=120, n_events=1000, a=2.0, b=1.0, n_exp=5):
    """
    Generate n 4-momenta of Ds mesons using the provided differential cross section shape.

    Parameters:
    - p_beam: beam energy in GeV
    - n_events: number of momenta to generate
    - a, b, n_exp: parameters of the differential cross-section for D production

    Returns:
    - List of 4-momenta [E, px, py, pz]
    """
    n_events = int(n_events)

    # u = np.random.uniform(0, 1, size=n_events)
    # xF_abs = 1 - u**(1 / (n_exp + 1))  # Inverse CDF of (1 - x)^n
    # x_F = xF_abs * np.random.choice([-1, 1], size=n_events)  # Randomly assign sign to xF

    x_F = np.random.uniform(-1, 1, size=n_events)

    p_T2 = np.random.exponential(
        scale=b, size=n_events
    )  # sample pT^2 from a rough expected distribution
    p_T = np.sqrt(p_T2)

    # Putting together the 4-momenta
    phi = np.random.uniform(0, 2 * np.pi, size=n_events)
    sqrt_s = np.sqrt(2 * np.sqrt(p_beam**2 + const.m_proton**2) * const.m_proton)
    pz_CM_max = sqrt_s / 2
    p4_CM = vector.array(
        {
            "E": np.sqrt(p_T**2 + (x_F * pz_CM_max) ** 2 + const.m_charged_Ds**2),
            "px": p_T * np.cos(phi),
            "py": p_T * np.sin(phi),
            "pz": x_F * pz_CM_max,
        }
    )
    beta = (
        p_beam
        / (np.sqrt(p_beam**2 + const.m_proton**2) + const.m_proton)
        * np.ones(n_events)
    )
    # NOTE: check minus signs here
    p4 = p4_CM.boostZ(beta)
    # Weight from the given differential cross section (unnormalized)
    weight = (1 - abs(x_F)) ** n_exp * np.exp(-(a * p_T + b * p_T2))

    return p4, weight / np.sum(weight)  # normalize the weights


def generate_taus_with_custom_method(
    params,
    p_beam=120,
    pT_max=30.0,
    CoM=False,
    as_dict=False,
    pid=15,
    n_events=1000,
    cone_force_acceptance=None,
    n_trials=None,
):
    """
    Generate n 4-momenta of Ds mesons using the provided differential cross section shape.

        cone_force_acceptance: [x0, y0, z0, R]
    Returns:
    - List of 4-momenta [E, px, py, pz]
    """
    n_events = int(n_events)

    n_generated = 0
    p4_tau = vector.array(
        {
            "E": [],
            "px": [],
            "py": [],
            "pz": [],
        }
    )
    weights = np.empty(0)
    w_total = 0
    if n_trials is None:
        n_trials = n_events

    while n_generated < n_events:
        w_trial = np.ones(n_trials)

        # sample xF from a log-uniform distribution
        log10_x_F_abs = np.random.uniform(-4, 0, size=n_trials)
        # Randomly assign sign to xF
        x_F = 10**log10_x_F_abs * np.random.choice([-1, 1], size=n_trials)

        # x_F = np.random.uniform(-1, 1, size=n_trials)

        # final desired pdf for xF
        x_F_pdf_final = params["r_1"] * np.exp(
            -params["a_1"] * abs(x_F) ** params["n_1"]
        ) + params["r_2"] * np.exp(-params["a_2"] * abs(x_F) ** params["n_2"])

        # x_F_pdf_final /= np.trapz(x_F_pdf_final, abs(x_F))  # normalize the pdf
        w_trial *= x_F_pdf_final * abs(x_F)
        # w_trial *= x_F_pdf_final

        # p_T = np.random.gamma(shape=params["mu1"], scale=params["lambda1"], size=n_trials)
        log10_p_T = np.random.uniform(-4, np.log10(pT_max), size=n_trials)
        p_T = 10**log10_p_T  # sample pT from a log-uniform distribution

        p_T_pdf_final = (
            params["g_1"]
            * gamma.pdf(
                p_T ** params["m_1"],
                a=params["mu_1"],
                scale=params["lambda_1"],
            )
            + params["g_2"]
            * gamma.pdf(
                p_T ** params["m_2"], a=params["mu_2"], scale=params["lambda_2"]
            )
            + params["g_3"]
            * gamma.pdf(
                p_T ** params["m_3"], a=params["mu_3"], scale=params["lambda_3"]
            )
        )
        # p_T_pdf_final /= np.trapz(p_T_pdf_final, p_T)  # normalize the pdf

        w_trial *= p_T_pdf_final * p_T

        # Putting together the 4-momenta
        phi = np.random.uniform(0, 2 * np.pi, size=n_trials)
        if CoM:
            sqrt_s = 2 * p_beam
        else:
            sqrt_s = np.sqrt(
                2 * np.sqrt(p_beam**2 + const.m_proton**2) * const.m_proton
            )
        pz_CM_max = sqrt_s / 2
        p4_CM = vector.array(
            {
                "pz": x_F * pz_CM_max,
                "px": p_T * np.cos(phi),
                "py": p_T * np.sin(phi),
                "E": np.sqrt(p_T**2 + (x_F * pz_CM_max) ** 2 + const.m_tau**2),
            }
        )

        if CoM:
            p4_lab = p4_CM

        else:
            beta = (
                p_beam
                / (np.sqrt(p_beam**2 + const.m_proton**2) + const.m_proton)
                * np.ones(n_trials)
            )
            # NOTE: check minus signs here
            p4_lab = p4_CM.boostZ(beta)

        # If force events in acceptance, count only accepted ones
        if isinstance(cone_force_acceptance, list):
            x0, y0, L, R = cone_force_acceptance
            v_tau = p4_lab.to_3D().unit()
            x_tau_p0 = v_tau["x"] * L
            y_tau_p0 = v_tau["y"] * L

            accepted = np.sqrt((x_tau_p0 - x0) ** 2 + (y_tau_p0 - y0) ** 2) < R
            n_generated += accepted.sum()
            w_total += w_trial.sum()

        else:
            accepted = np.full(n_trials, True)
            n_generated += n_trials
            w_total += w_trial.sum()

        weights = np.concatenate([weights, w_trial[accepted]])
        p4_tau = concatenate_vectors(p4_tau, p4_lab)
        w_accepted = weights.sum()

        efficiency = w_accepted / w_total
        print(f"Generated {n_generated} taus, efficiency: {efficiency:.4f}")

        if accepted.sum() > n_events:
            weights = weights[:n_events]
            p4_tau = p4_tau[:n_events]

    # Account for tau+ and tau- separate production
    weights = weights / np.sum(weights) / 2  # normalize the weights

    if as_dict:
        tau_dict = {
            "E": p4_tau["E"],
            "px": p4_tau["px"],
            "py": p4_tau["py"],
            "pz": p4_tau["pz"],
            "weights": weights * efficiency,
            "pid": np.full(len(p4_tau), pid, dtype=int),
        }

        return tau_dict
    else:
        return p4_tau, weights * efficiency


def generate_taus(p_beam=120, n_events=1000, a=2.0, b=1.0, n_exp=5, as_dict=False):
    """
    Generate n 4-momenta of tau leptons using the provided differential cross section shape.

    Parameters:
    - p_beam: beam energy in GeV
    - n_events: number of momenta to generate
    - a, b, n_exp: parameters of the differential cross-section for D production

    Returns:
    - List of 4-momenta [E, px, py, pz]
    """

    # Generate Ds+ and Ds- momenta (so far they're the same)
    p4_Dsp, wp = generate_Ds(p_beam, n_events, a, b, n_exp)
    p4_Dsm, wm = generate_Ds(p_beam, n_events, a, b, n_exp)

    # concatenate

    p4_Ds = concatenate_vectors(p4_Dsp, p4_Dsm)

    w = np.concatenate((wp, wm))

    # Generate tau momenta from Ds momenta
    p4_tau = ps.decay_2_body(p4_Ds, const.m_charged_Ds, const.m_tau, 0.0)
    w = w * frag_fractions["Ds+"]

    if as_dict:
        tau_dict = {}
        tau_dict["weights"] = w
        tau_dict["E_mother"] = p4_Ds["E"]
        tau_dict["px_mother"] = p4_Ds["px"]
        tau_dict["py_mother"] = p4_Ds["py"]
        tau_dict["pz_mother"] = p4_Ds["pz"]
        tau_dict["E"] = p4_tau["E"]
        tau_dict["px"] = p4_tau["px"]
        tau_dict["py"] = p4_tau["py"]
        tau_dict["pz"] = p4_tau["pz"]
        tau_dict["mother_pid"] = np.zeros(len(p4_tau), dtype=int)
        tau_dict["mother_pid"][: len(p4_Dsp)] = 431
        tau_dict["mother_pid"][len(p4_Dsp) :] = -431
        tau_dict["pid"] = np.zeros(len(p4_tau), dtype=int)
        tau_dict["pid"][: len(p4_Dsp)] = 15
        tau_dict["pid"][len(p4_Dsp) :] = -15

        return tau_dict
    else:
        return p4_tau, w
