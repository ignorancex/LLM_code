from contextlib import nullcontext
import numpy as np

from . import const

from alp import exp, models
from tqdm import tqdm


def make_rate_table(
    EXP,
    save=True,
    inv_fa_range=[0.5e-9, 1e-4],
    ma_range=[1e-2, const.m_tau - const.m_e * 1.01],
    Npoints=101,
    c_lepton=False,
    name="",
    show_progress=True,
):

    if isinstance(EXP, str):
        # Load the Experiment object with simulation events
        simulation_exp = exp.load_events_from_pickle(f"tau_events/{EXP}.pkl")
    elif isinstance(EXP, exp.Experiment):
        simulation_exp = EXP
    else:
        raise ValueError("EXP must be a string or an Experiment object.")

    # Create scan grid
    m_alps = np.geomspace(*ma_range, Npoints, endpoint=True)
    # higher resolution around the dimuon threshold
    if ma_range[0] < 2 * const.m_mu and ma_range[1] > 2 * const.m_mu:
        m_alps = np.append(m_alps, 2 * const.m_mu + np.linspace(-10e-3, 10e-3, 20))
        m_alps = np.append(
            m_alps, const.m_mu + const.m_e + np.linspace(-5e-3, 5e-3, 20)
        )
        m_alps = np.sort(m_alps)
        inv_fas = np.geomspace(*inv_fa_range, Npoints + 40, endpoint=True)
    else:
        inv_fas = np.geomspace(*inv_fa_range, Npoints, endpoint=True)

    MA, INV_FA = np.meshgrid(m_alps, inv_fas)

    # Make sure the initial experiment is set up with the right kind of ALP
    # This makes sure we can use reweighting to get the rates
    simulation_exp.alp = models.ALP(0.5, 1e7, c_lepton=c_lepton)

    z = []
    total_iterations = len(m_alps) * len(inv_fas)
    progress_ctx = (
        tqdm(total=total_iterations, desc=f"Rate table for {simulation_exp.name}")
        if show_progress
        else nullcontext()
    )

    with progress_ctx as pbar:
        for ma in m_alps:
            alp_1 = models.ALP(ma, 1e7, c_lepton=c_lepton)
            simulation_exp.get_event_rate(alp_1, generate_events=True)
            for inv_fa in inv_fas:
                alp_2 = models.ALP(ma, 1 / inv_fa, c_lepton=c_lepton)
                r = simulation_exp.reweight(alp_1, alp_2)
                z.append(r)
                if show_progress:
                    pbar.update(1)

    Z = np.reshape(z, MA.shape).T

    if save:
        np.save(f"data/{simulation_exp.name}_rates_{name}.npy", [MA, INV_FA, Z])
    return MA, INV_FA, Z
