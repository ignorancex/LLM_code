import multiprocessing as mp
import numpy as np

from alp import exp, sim_tools, const
from alp.exp_dicts import EXPERIMENTS
from tqdm.contrib.concurrent import process_map

# n_events = 100_000
# NUMI_files = f"tau_events/df_NuMI_{n_events}_custom_parametrization.parquet"
# SPS_files = f"tau_events/df_SPS_{n_events}_custom_parametrization.parquet"
# LHC_files = f"tau_events/df_LHC_{n_events}_custom_parametrization.parquet"

NUMI_files = "pythia8_events/numi_120GeV.pkl"
SPS_files = "pythia8_events/sps_400GeV.pkl"
LHC_files = "pythia8_events/lhc_13.6TeV.pkl"


NPOINTS = 30
MAX_WORKERS = mp.cpu_count() - 1


def simulate(args):
    exp_case, kwargs = args
    _ = sim_tools.make_rate_table(exp_case, save=True, **kwargs)


def run_simulations(exp_list, BP_NAME, **kwargs):
    print(f"Running simulations for {BP_NAME}...")
    args_list = [(exp_case, kwargs) for exp_case in exp_list]
    process_map(simulate, args_list, max_workers=MAX_WORKERS)


if __name__ == "__main__":

    mp.set_start_method("spawn")
    # """'
    #     LFV ANARCHY   FA vs ma
    # """
    # BP_NAME = "anarchy_jul19"
    # lamb = 1
    # c_lepton = np.ones((3, 3))

    # kwargs = {
    #     "inv_fa_range": [1e-10, 1e-3],
    #     # "ma_range": [1e-2, 0.25],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [
    #     "CHARM",
    #     "BEBC",
    #     "SHIP",
    #     "FASER",
    #     "FASER2",
    #     "NOVA",
    #     "PROTODUNE_NP02",
    #     "PROTODUNE_NP04",
    #     "DUNE",
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """
    # LFC   FA vs ma
    # """
    # BP_NAME = "LFC_universal_jul19"
    # c_lepton = np.diag([1, 1, 1])

    # kwargs = {
    #     "inv_fa_range": [0.5e-5, 1],
    #     "ma_range": [2e-3, 0.25],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # # exp_list = ["CHARM"]
    # exp_list = [
    #     "CHARM",
    #     "BEBC",
    #     "SHIP",
    #     "FASER2",
    #     "PROTODUNE_NP02",
    #     "PROTODUNE_NP04",
    #     "DUNE",
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """
    # LFC   FA vs ma -- e-tau coupled (Muonphobic)
    # """
    # BP_NAME = "LFC_etau_jul19"
    # c_lepton = np.diag([1, 0, 1])

    # kwargs = {
    #     "inv_fa_range": [1e-5, 1],
    #     "ma_range": [2e-3, const.m_tau - const.m_e],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [
    #     "CHARM",
    #     "BEBC",
    #     "SHIP",
    #     "FASER2",
    #     "PROTODUNE_NP02",
    #     "PROTODUNE_NP04",
    #     "DUNE",
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFC   FA vs ma -- tauphilic
    # """
    # BP_NAME = "LFC_tauphilic_jul19"
    # c_lepton = np.diag([0, 0, 1])

    # kwargs = {
    #     "inv_fa_range": [1e-4, 10],
    #     "ma_range": [2e-3, const.m_tau - const.m_e],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [
    #     "CHARM",
    #     "BEBC",
    #     "SHIP",
    #     "FASER2",
    #     "DUNE",
    #     "PROTODUNE_NP02",
    #     "PROTODUNE_NP04",
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    # LFV HIEREACHY  FA vs ma
    # """
    # for lam in [0.05, 0.01, 0.005, 0.001]:
    #     BP_NAME = f"hierarchy_lambda_{lam}_clep_1_jul19"
    #     lamb = lam
    #     C_LEPTON = 1
    #     c_lepton = C_LEPTON * np.array(
    #         [[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]]
    #     )

    #     kwargs = {
    #         "inv_fa_range": [1e-10, 1e-3],
    #         "name": BP_NAME,
    #         "c_lepton": c_lepton,
    #         "Npoints": NPOINTS,
    #     }

    #     exp_list = [
    #         "CHARM",
    #         "BEBC",
    #         "SHIP",
    #         "FASER",
    #         "FASER2",
    #         "NOVA",
    #         "PROTODUNE_NP02",
    #         "PROTODUNE_NP04",
    #         "DUNE",
    #     ]
    #     run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV ANARCHY   FA vs ma
    # """
    # BP_NAME = "anarchy_jul19"
    # lamb = 1
    # c_lepton = np.ones((3, 3))

    # kwargs = {
    #     "inv_fa_range": [2e-9, 1e-3],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = ["MICROBOONE", "ICARUS"]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV HIEREACHY  FA vs ma
    # """
    # for lam in [0.05, 0.01, 0.005, 0.001]:
    #     BP_NAME = f"hierarchy_lambda_{lam}_clep_1_jul19"
    #     lamb = lam
    #     C_LEPTON = 1
    #     c_lepton = C_LEPTON * np.array(
    #         [[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]]
    #     )

    #     kwargs = {
    #         "inv_fa_range": [1e-9, 1e-3],
    #         "name": BP_NAME,
    #         "c_lepton": c_lepton,
    #         "Npoints": NPOINTS,
    #     }

    #     exp_list = ["MICROBOONE", "ICARUS"]
    #     run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV Ra=5 FA vs ma
    # """
    # BP_NAME = "LFV_Ra_5_jul19"
    # c_lepton = np.array([[5, 2, 2], [2, 5, 2], [2, 2, 5]])

    # kwargs = {
    #     "inv_fa_range": [1e-9, 1e-5],
    #     "ma_range": [2.2e-1, 1.7],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # # exp_list = [ARGONEUT, ARGONEUT_absorber, CHARM, BEBC]
    # exp_list = ["ARGONEUT", "ARGONEUT_absorber"]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV Ra=1/3 FA vs ma
    # """
    # BP_NAME = "LFV_Ra_1o3_jul19"
    # c_lepton = np.array([[0.333, 2, 2], [2, 0.333, 2], [2, 2, 0.333]])

    # kwargs = {
    #     "inv_fa_range": [1e-9, 1e-5],
    #     "ma_range": [2.2e-1, 1.7],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = ["ARGONEUT", "ARGONEUT_absorber", "CHARM", "BEBC"]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """
    # LFV ANARCHY   FA vs ma
    # """
    # BP_NAME = "anarchy_vMC"
    # lamb = 1
    # c_lepton = np.ones((3, 3))

    # kwargs = {
    #     "inv_fa_range": [1e-10, 1e-3],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = []

    # for exp_name in ["SHiP", "NoVA", "BEBC", "CHARM"]:
    #     for ncase in ["nlow", "nhigh"]:
    #         for bcase in ["blow", "bhigh"]:

    #             EXPERIMENTS[exp_name]["name"] = f"{exp_name}_{ncase}_{bcase}"

    #             exp_case = exp.Experiment(
    #                 f"tau_events/df_120GeV_1e6_{ncase}_{bcase}.pkl",
    #                 exp_dic=EXPERIMENTS[exp_name],
    #                 duplicate_taus=5 if exp_name == "NoVA" else 1,
    #             )

    #             exp_list.append(exp_case)

    #     # Run with Pythia8 events:
    #     EXPERIMENTS[exp_name]["name"] = f"{exp_name}_pythia8"
    #     new_case = exp.Experiment(
    #         NUMI_files,
    #         exp_dic=EXPERIMENTS[exp_name],
    #         duplicate_taus=5 if exp_name == "NoVA" else 1,
    #     )
    #     exp_list.append(new_case)

    # run_simulations(exp_list, BP_NAME, **kwargs)

    """'
        LFV ANARCHY   FA vs ma
    """
    BP_NAME = "anarchy_jul20"
    lamb = 1
    c_lepton = np.ones((3, 3))

    kwargs = {
        "inv_fa_range": [2e-9, 1e-3],
        "name": BP_NAME,
        "c_lepton": c_lepton,
        "Npoints": NPOINTS,
    }

    exp_list = [
        "MICROBOONE_custom_parametrization",
        "ICARUS_custom_parametrization",
        "NoVA_custom_parametrization",
    ]
    run_simulations(exp_list, BP_NAME, **kwargs)

    """'
        LFV HIEREACHY  FA vs ma
    """
    for lam in [0.05, 0.01, 0.005, 0.001]:
        BP_NAME = f"hierarchy_lambda_{lam}_clep_1_jul20"
        lamb = lam
        C_LEPTON = 1
        c_lepton = C_LEPTON * np.array(
            [[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]]
        )

        kwargs = {
            "inv_fa_range": [1e-9, 1e-3],
            "name": BP_NAME,
            "c_lepton": c_lepton,
            "Npoints": NPOINTS,
        }

    exp_list = [
        "MICROBOONE_custom_parametrization",
        "ICARUS_custom_parametrization",
        "NoVA_custom_parametrization",
    ]
    run_simulations(exp_list, BP_NAME, **kwargs)
