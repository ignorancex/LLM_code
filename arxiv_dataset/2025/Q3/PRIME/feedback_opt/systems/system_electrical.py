import os

import numpy as np
import pandapower as pp
import pandas as pd

from feedback_opt.systems.system_base import SystemBase
from feedback_opt.utils import UtilsPd, get_sens_powerInjections_to_voltage


class SystemElectrical(SystemBase):
    """
    unconstrained system defined by io map:
    y = h(u)

    The class is intended to be immutable after initialization!
    """

    def __init__(self, params):
        # pandapower net
        assert isinstance(params.net_path, str)
        cwd = os.path.abspath(os.getcwd())
        cwd = cwd[: cwd.rfind("prime") + len("prime")]
        self.net = pp.from_json(f"{cwd}/{params.net_path}")

        pp.runpp(self.net)
        self.base_mva = self.net._ppc["internal"]["baseMVA"]
        self.Y_admittance = self.net._ppc["internal"]["Ybus"].todense()

        # extract bus mapping
        self.bus_tot = self.net._pd2ppc_lookups["bus"]
        self.bus_slack = self.net._ppc["internal"]["ref"]
        assert len(self.bus_slack) == 1, "multiple slack busses are currently not supported"
        assert len(self.net._ppc["internal"]["pv"]) == 0, "PV busses are currently not supported"
        self.bus_pq = self.net._ppc["internal"]["pq"]
        self.bus_sgen = self.net.sgen["bus"].to_numpy()

        # input dimension
        # [p_pu, q_pu ...] in PU stacked for each generator
        params.m = 2 * len(self.bus_sgen)

        # output dimension
        # [vm_pu ...] in PU for every PQ bus
        params.p = len(self.bus_pq)

        # system constraints
        self._load_constraint_polytopes(params)

        super().__init__(params)

    def _load_constraint_polytopes(self, params):
        # input constraints
        df_sgen = self.net.sgen.copy()
        assert {"min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"}.issubset(df_sgen.columns)
        UtilsPd.cart_to_complex(df_sgen, real="min_p_mw", imaginary="min_q_mvar", cmplx="min_u_mw")
        UtilsPd.cart_to_complex(df_sgen, real="max_p_mw", imaginary="max_q_mvar", cmplx="max_u_mw")
        df_sgen["min_u_pu"] = df_sgen["min_u_mw"] / self.base_mva
        df_sgen["max_u_pu"] = df_sgen["max_u_mw"] / self.base_mva
        UtilsPd.complex_to_cart(
            df_sgen, cmplx="min_u_pu", real="min_u_real", imaginary="min_u_imag"
        )
        UtilsPd.complex_to_cart(
            df_sgen, cmplx="max_u_pu", real="max_u_real", imaginary="max_u_imag"
        )

        params.A_u = np.vstack([np.eye(params.m), -np.eye(params.m)])
        params.b_u = np.hstack(
            [
                df_sgen["max_u_real"].T,
                df_sgen["max_u_imag"],
                -df_sgen["min_u_real"],
                -df_sgen["min_u_imag"],
            ]
        ).reshape(-1, 1)

        # output constraints
        df_bus = self.net.bus.copy()
        params.A_y = np.vstack([np.eye(params.p), -np.eye(params.p)])
        params.b_y = np.hstack(
            [df_bus["max_vm_pu"][self.bus_pq], -df_bus["min_vm_pu"][self.bus_pq]]
        ).reshape(-1, 1)

    def _apply_u(self, u: np.ndarray) -> pd.DataFrame:
        assert np.shape(u) == (self.m, 1)

        # pu to physical
        df = pd.DataFrame(data=u.reshape((-1, 2), order="F"), columns=["p_pu", "q_pu"])
        UtilsPd.cart_to_complex(df, real="p_pu", imaginary="q_pu", cmplx="complex_power_pu")
        df["complex_power"] = df["complex_power_pu"] * self.base_mva
        UtilsPd.complex_to_cart(df, cmplx="complex_power", real="p_mw", imaginary="q_mvar")

        # apply to network
        self.net.sgen["p_mw"] = df["p_mw"]
        self.net.sgen["q_mvar"] = df["q_mvar"]

        # update power flow
        pp.runpp(self.net)

    def h(self, u: np.ndarray) -> np.ndarray:
        assert np.shape(u) == (self.m, 1)

        # update u
        self._apply_u(u)

        # build y
        y = np.array(self.net.res_bus["vm_pu"][self.bus_pq]).reshape((-1, 1))

        assert np.shape(y) == (self.p, 1)
        return y

    def du_h(self, u: np.ndarray) -> np.ndarray:
        assert np.shape(u) == (self.m, 1)

        # update u
        self._apply_u(u)

        # extract complex pu bus voltages
        UtilsPd.pol_to_complex(
            self.net.res_bus, absolute="vm_pu", degree="va_degree", cmplx="v_complex"
        )
        v_complex = self.net.res_bus["v_complex"].to_numpy()

        # obtain voltage to power sensitivity (linearization)
        gamma = get_sens_powerInjections_to_voltage(self.Y_admittance, v_complex)

        # select bus_pq in bus_tot
        select_pq_in_bus = np.zeros((len(self.bus_tot), len(self.bus_pq)))
        for col, row in enumerate(self.bus_pq):
            select_pq_in_bus[row, col] = 1

        C = np.block(
            [
                [select_pq_in_bus, np.zeros_like(select_pq_in_bus)],
                [np.zeros_like(select_pq_in_bus), select_pq_in_bus],
            ]
        )

        gamma_c = C.T @ gamma @ C

        # obtain power to voltage sensitivity
        psi = np.linalg.inv(gamma_c)

        # select bus_sgen in bus_pq
        sgen_in_pq_idx = np.array([np.where(self.bus_pq == val)[0][0] for val in self.bus_sgen])
        select_sgen_in_pq = np.zeros((len(self.bus_pq), len(self.bus_sgen)))
        for col, row in enumerate(sgen_in_pq_idx):
            select_sgen_in_pq[row, col] = 1

        C = np.block(
            [
                [select_sgen_in_pq, np.zeros_like(select_sgen_in_pq)],
                [np.zeros_like(select_sgen_in_pq), select_sgen_in_pq],
            ]
        )

        sensitivity = (psi @ C)[: len(self.bus_pq), :]

        assert np.shape(sensitivity) == (self.p, self.m)
        return sensitivity
