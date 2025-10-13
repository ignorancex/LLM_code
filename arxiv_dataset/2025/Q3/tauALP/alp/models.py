import numpy as np
from scipy.integrate import quad, dblquad

from . import const
from . import phase_space as ps


LEPTON_FINAL_STATES = ["ee", "em", "me", "mm", "et", "te", "tm", "mt", "tt"]
FINAL_STATES = LEPTON_FINAL_STATES + ["gg", "NN"]

LEPTON_MASSES = [const.m_e, const.m_mu, const.m_tau, 0]
Gamma_tau = const.get_decay_rate_in_s(2.903e-13)  # GeV is the width of the Tau lepton
Gamma_mu = const.get_decay_rate_in_s(2.196e-6)  # GeV is the width of the Muon lepton
LEPTON_WIDTHS = [0, Gamma_mu, Gamma_tau]
LEPTON_INDEX = {"e": 0, "m": 1, "t": 2, "g": 3}

# def g_ps(x):
#     return 1 - 8 * x - 12 * x**2 * np.log(x) + 8 * x**3 - x**4
# Gamma_mu = (
#     const.m_mu**5 * const.Gf**2 / 192 / np.pi**3 * g_ps(const.m_e**2 / const.m_mu**2)
# )


class ALP:
    def __init__(
        self,
        m_a,
        f_a,
        c_gg=0,
        c_NN=0,
        mN=0,
        Bvis=1,
        c_lepton=None,
        force_LFC_flat=False,
    ):
        self.f_a = f_a
        self.m_a = m_a
        self.mN = mN
        self.Lambda_NP = 4 * np.pi * self.f_a
        self.force_LFC_flat = force_LFC_flat

        # NOTE: democratic couplings
        if c_lepton is None:
            self.c_lepton = np.ones((3, 3))
        else:
            self.c_lepton = c_lepton

        self.c_gg = c_gg + self.c_gg_eff()
        self.c_BB = self.c_lepton.diagonal().sum() / 2
        # NOTE: Check the factor of 1/2
        self.c_WW = -(self.c_lepton.diagonal().sum()) / 2

        # Heavy neutrino couplings
        self.c_NN = c_NN

        # Decay widths
        for daughters in LEPTON_FINAL_STATES:
            vars(self)[f"Gamma_a_to_{daughters}"] = self.get_Gamma_a_to_li_lj(
                LEPTON_INDEX[daughters[0]], LEPTON_INDEX[daughters[1]]
            )
        # Other channels
        self.Gamma_a_to_gg = self.get_Gamma_a_to_gg()
        self.Gamma_a_to_NN = self.get_Gamma_a_to_NN()

        # Visible decay widths
        self.Gamma_a_vis = self.Gamma_a_to_gg + np.sum(
            [
                vars(self)["Gamma_a_to_" + daughters]
                for daughters in LEPTON_FINAL_STATES
            ],
            axis=0,
        )

        # Invisible decay width
        self.Bvis = Bvis
        self.Binv = 1 - self.Bvis
        self.Gamma_a_inv = self.Gamma_a_vis * (1 - self.Bvis) / self.Bvis
        self.Gamma_a = self.Gamma_a_vis + self.Gamma_a_inv

        if isinstance(self.Gamma_a, np.ndarray):

            # creating branching ratios
            for daughters in FINAL_STATES:
                vars(self)[f"BR_a_to_{daughters}"] = np.empty_like(self.Gamma_a)
            self.BR_a_to_inv = np.empty_like(self.Gamma_a)

            # Assigning branching ratios
            for daughters in FINAL_STATES:
                vars(self)[f"BR_a_to_{daughters}"][self.Gamma_a > 0] = (
                    vars(self)[f"Gamma_a_to_{daughters}"][self.Gamma_a > 0]
                    / self.Gamma_a[self.Gamma_a > 0]
                )

                vars(self)[f"BR_a_to_{daughters}"][self.Gamma_a <= 0] = 0

            self.BR_a_to_inv[self.Gamma_a > 0] = (
                self.Gamma_a_inv[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_inv[self.Gamma_a <= 0] = 0

        else:
            # Assigning branching ratios
            self.BR_a_to_inv = np.empty_like(self.Gamma_a)
            for daughters in FINAL_STATES:
                vars(self)[f"BR_a_to_{daughters}"] = (
                    vars(self)[f"Gamma_a_to_{daughters}"] / self.Gamma_a
                    if self.Gamma_a > 0
                    else 0
                )
            self.BR_a_to_inv = (
                self.Gamma_a_inv / self.Gamma_a if self.Gamma_a > 0 else 0
            )

        # Decay kinematics
        self.Ea_min = {
            "tau>nu+rho+a": self.m_a,
            "tau>nu+pi+a": self.m_a,
            "tau>nu+nu+e+a": self.m_a,
            "tau>nu+nu+mu+a": self.m_a,
            "tau>e+a": (const.m_tau**2 + self.m_a**2 - const.m_e**2) / 2 / const.m_tau,
            "tau>mu+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
        }

        self.Ea_max = {
            "tau>nu+rho+a": (const.m_tau**2 + self.m_a**2 - const.m_charged_rho**2)
            / 2
            / const.m_tau,
            "tau>nu+pi+a": (const.m_tau**2 + self.m_a**2 - const.m_charged_pion**2)
            / 2
            / const.m_tau,
            "tau>nu+nu+e+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
            "tau>nu+nu+mu+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
            "tau>e+a": (const.m_tau**2 + self.m_a**2 - const.m_e**2) / 2 / const.m_tau,
            "tau>mu+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
        }

    def prob_decay(self, Ea, L, dL):
        gamma = Ea / self.m_a
        if isinstance(Ea, np.ndarray):
            beta = np.ones_like(Ea)
            beta[gamma > 1] = np.sqrt(1 - 1 / gamma[gamma > 1] ** 2)
        else:
            beta = np.sqrt(1 - 1 / gamma**2)
        ell_dec = const.get_decay_rate_in_cm(self.Gamma_a) * gamma * beta
        p = np.empty_like(ell_dec)
        # p[ell_dec > 0] = np.exp(-L / ell_dec[ell_dec > 0]) * (
        # 1 - np.exp(-dL / ell_dec[ell_dec > 0])
        # )
        p = np.exp(-L / ell_dec) * (1 - np.exp(-dL / ell_dec))
        # p[ell_dec <= 0] = 0
        return p

    def c_gg_eff(self):
        if isinstance(self.m_a, np.ndarray):
            c = np.zeros_like(self.m_a, dtype=complex)
            c[self.m_a > 0] = (
                self.c_lepton[0, 0]
                * ps.B1_loop((2 * const.m_e / self.m_a[self.m_a > 0]) ** 2)
                + self.c_lepton[1, 1]
                * ps.B1_loop((2 * const.m_mu / self.m_a[self.m_a > 0]) ** 2)
                + self.c_lepton[2, 2]
                * ps.B1_loop((2 * const.m_tau / self.m_a[self.m_a > 0]) ** 2)
            )
            return c
        else:
            if self.m_a:
                return (
                    self.c_lepton[0, 0] * ps.B1_loop((2 * const.m_e / self.m_a) ** 2)
                    + self.c_lepton[1, 1] * ps.B1_loop((2 * const.m_mu / self.m_a) ** 2)
                    + self.c_lepton[2, 2]
                    * ps.B1_loop((2 * const.m_tau / self.m_a) ** 2)
                )
            else:
                return 0

    def F_alp_2body(self, m_l1, m_l2):
        """two body phase space of alp decays"""
        term1 = self.m_a * (m_l1 + m_l2) ** 2 / 32 / np.pi / self.f_a**2
        if isinstance(self.m_a, np.ndarray):
            term2 = np.zeros_like(self.m_a)
            term2[self.m_a > m_l1 + m_l2] = (
                1 - (m_l1 - m_l2) ** 2 / self.m_a[self.m_a > m_l1 + m_l2] ** 2
            )
        else:
            if self.m_a > m_l1 + m_l2:
                term2 = 1 - (m_l1 - m_l2) ** 2 / self.m_a**2
            else:
                term2 = 0

        if isinstance(self.m_a, np.ndarray):
            Gamma = np.zeros_like(self.m_a)

            Gamma[self.m_a > m_l1 + m_l2] = (
                np.sqrt(
                    1
                    - 2 * (m_l1**2 + m_l2**2) / self.m_a[self.m_a > m_l1 + m_l2] ** 2
                    + (m_l1**2 - m_l2**2) ** 2 / self.m_a[self.m_a > m_l1 + m_l2] ** 4
                )
                * term1[self.m_a > m_l1 + m_l2]
                * term2[self.m_a > m_l1 + m_l2]
            )
        else:
            if self.m_a > m_l1 + m_l2:
                Gamma = (
                    np.sqrt(
                        1
                        - 2 * (m_l1**2 + m_l2**2) / self.m_a**2
                        + (m_l1**2 - m_l2**2) ** 2 / self.m_a**4
                    )
                    * term1
                    * term2
                )
            else:
                Gamma = 0

        return Gamma

    def get_Gamma_a_to_li_lj(self, l_i, l_j):
        """two body phase space of alp decays"""
        return self.c_lepton[l_i, l_j] ** 2 * self.F_alp_2body(
            LEPTON_MASSES[l_i], LEPTON_MASSES[l_j]
        )

    def get_BR_a_to_li_lj(self, l_i, l_j):
        """branching ratio to leptons"""
        return self.get_Gamma_a_to_li_lj(self, l_i, l_j) / self.Gamma_a

    def get_Gamma_a_to_gg(self):
        return (
            np.abs(self.c_gg) ** 2
            * self.m_a**3
            * const.alphaQED**2
            / 64
            / np.pi**3
            / self.f_a**2
        )

    def get_Gamma_a_to_NN(self):
        return self.c_NN**2 * self.F_alp_2body(self.mN, self.mN)

    def alp_BR(self, final_state):
        """Branching ratio for alp -> l+ l-"""
        if final_state == "ee":
            return self.BR_a_to_ee
        elif final_state == "em":
            return self.BR_a_to_em
        elif final_state == "me":
            return self.BR_a_to_me
        elif final_state == "mm":
            return self.BR_a_to_mm
        elif final_state == "et":
            return self.BR_a_to_et
        elif final_state == "te":
            return self.BR_a_to_te
        elif final_state == "tm":
            return self.BR_a_to_tm
        elif final_state == "mt":
            return self.BR_a_to_mt
        elif final_state == "tt":
            return self.BR_a_to_tt
        elif final_state == "gg":
            return self.BR_a_to_gg
        else:
            raise ValueError(f"Unknown final state {final_state}")

    def alp_visible_BR(self, final_states):
        return (
            self.BR_a_to_ee * ("ee" in final_states)
            + self.BR_a_to_em * ("em" in final_states)
            + self.BR_a_to_mm * ("mm" in final_states)
            + self.BR_a_to_me * ("me" in final_states)
            + self.BR_a_to_et * ("et" in final_states)
            + self.BR_a_to_te * ("te" in final_states)
            + self.BR_a_to_tm * ("tm" in final_states)
            + self.BR_a_to_mt * ("mt" in final_states)
            + self.BR_a_to_tt * ("tt" in final_states)
            + self.BR_a_to_gg * ("gg" in final_states)
        )

    def tau_BR(self, production_channel):
        """Branching ratio for tau -> a + l"""
        if production_channel == "tau>e+a":
            return self.BR_tau_to_a_e()
        elif production_channel == "tau>mu+a":
            return self.BR_tau_to_a_mu()
        elif production_channel == "tau>nu+pi+a":
            return self.BR_tau_to_pi_nu_a()
        elif production_channel == "tau>nu+rho+a":
            return self.BR_tau_to_rho_nu_a()
        elif production_channel == "tau>nu+nu+e+a":
            return self.BR_tau_to_e_nu_nu_a()
        elif production_channel == "tau>nu+nu+mu+a":
            return self.BR_tau_to_mu_nu_nu_a()
        else:
            raise ValueError(f"Unknown production channel {production_channel}")

    def tau_diff_BR(self, Ea, production_channel):
        """Differential Branching ratio for tau decay wrt Ealp"""
        if production_channel == "tau>nu+pi+a":
            return self.diff_BR_tau_to_pi_nu_a(Ea)
        elif production_channel == "tau>nu+rho+a":
            return self.diff_BR_tau_to_rho_nu_a(Ea)
        elif production_channel == "tau>nu+nu+e+a":
            return self.diff_BR_tau_to_nu_nu_e_a(Ea)
        elif production_channel == "tau>nu+nu+mu+a":
            return self.diff_BR_tau_to_nu_nu_mu_a(Ea)
        else:
            raise ValueError(f"Unknown production channel {production_channel}")

    def BR_tau_to_a_mu(self):
        return self.BR_li_to_lj_a(2, 1)

    def BR_tau_to_a_e(self):
        return self.BR_li_to_lj_a(2, 0)

    def BR_li_to_lj_a(self, l_i, l_j):
        return (
            self.c_lepton[l_i, l_j] ** 2
            / 64
            / np.pi
            * LEPTON_MASSES[l_i] ** 3
            / self.f_a**2
            * ps.F_lepton_2body(LEPTON_MASSES[l_i], self.m_a, LEPTON_MASSES[l_j])
            / LEPTON_WIDTHS[l_i]
        )

    """
    BEGIN approximate from Yohei's notes
    """

    def BR_tau_to_pi_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1.5e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_rho_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 2.3e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_e_nu_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_mu_nu_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1e-12 * (1e4 / self.f_a) ** 2

    """
    END approximate from Yohei's notes
    """

    def diff_BR_tau_to_pi_nu_a(self, Ea):
        """Differential decay rate for tau -> pi nu a"""

        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * const.fcharged_pion**2
            * np.abs(const.Vud) ** 2
            * (mtau_barSQR - const.m_charged_pion**2) ** 2
            / 256
            / np.pi**3
            / self.f_a**2
            / mtau_barSQR
            * (
                1
                - self.m_a**2
                * (const.m_tau**2 + mtau_barSQR)
                / (const.m_tau**2 - mtau_barSQR) ** 2
            )
            * ps.v_2body(const.m_tau, np.sqrt(mtau_barSQR), self.m_a)
        )
        return (
            dGammadE
            * (Ea > self.Ea_min["tau>nu+pi+a"])
            * (Ea < self.Ea_max["tau>nu+pi+a"])
        ) / Gamma_tau

    def diff_BR_tau_to_rho_nu_a(self, Ea):
        """Differential decay rate for tau -> rho nu a"""
        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * const.fcharged_rho**2
            * np.abs(const.Vud) ** 2
            * (mtau_barSQR + 2 * const.m_charged_rho**2)
            * (1 - const.m_charged_rho**2 / mtau_barSQR) ** 2
            / 256
            / np.pi**3
            / self.f_a**2
            * (
                1
                - self.m_a**2
                * (const.m_tau**2 + mtau_barSQR)
                / (const.m_tau**2 - mtau_barSQR) ** 2
            )
            * ps.v_2body(const.m_tau, np.sqrt(mtau_barSQR), self.m_a)
        )
        return (
            dGammadE
            * (Ea >= self.Ea_min["tau>nu+rho+a"])
            * (Ea <= self.Ea_max["tau>nu+rho+a"])
        ) / Gamma_tau

    def diff_BR_tau_to_nu_nu_e_a(self, Ea):
        """Differential decay rate for tau -> rho nu a"""
        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * mtau_barSQR
            / 1536
            / np.pi**5
            / self.f_a**2
            * Iv_mvv_integral(Ea, self.m_a, mlep=const.m_e)
            * (
                1
                - self.m_a**2
                * (const.m_tau**2 + mtau_barSQR)
                / (const.m_tau**2 - mtau_barSQR) ** 2
            )
            * ps.v_2body(const.m_tau, np.sqrt(mtau_barSQR), self.m_a)
        )
        return (
            dGammadE
            * (Ea > self.Ea_min["tau>nu+nu+e+a"])
            * (Ea < self.Ea_max["tau>nu+nu+e+a"])
        ) / Gamma_tau

    def diff_BR_tau_to_nu_nu_mu_a(self, Ea):
        """Differential decay rate for tau -> rho nu a"""
        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * mtau_barSQR
            / 1536
            / np.pi**5
            / self.f_a**2
            * Iv_mvv_integral(Ea, self.m_a, mlep=const.m_mu)
            * (
                1
                - self.m_a**2
                * (const.m_tau**2 + mtau_barSQR)
                / (const.m_tau**2 - mtau_barSQR) ** 2
            )
            * ps.v_2body(const.m_tau, np.sqrt(mtau_barSQR), self.m_a)
        )
        return (
            dGammadE
            * (Ea > self.Ea_min["tau>nu+nu+mu+a"])
            * (Ea < self.Ea_max["tau>nu+nu+mu+a"])
        ) / Gamma_tau

    def BR_li_to_lj_gamma(self, l_i, l_j):
        """Branching ratio for li -> lj + gamma"""

        Gamma = (
            const.m_mu**3
            / 8
            / np.pi
            * (1 - (LEPTON_MASSES[l_j] / LEPTON_MASSES[l_i]) ** 2)
        )
        FFSQR = np.abs(
            np.abs(self.F2_5(self.m_a, l_i=l_i, l_j=l_j)) ** 2
            + np.abs(self.F2(self.m_a, l_i=l_i, l_j=l_j)) ** 2
        )
        return Gamma * FFSQR / Gamma_mu

    def BR_diff_li_to_lj_lk_lk(self, s12, s23, l_i, l_j, l_k):
        """Differential decay rate for mu -> e + a + gamma
        Args:
            s12: invariant mass of the e + a system
            s23: invariant mass of the a + gamma system
        """

        F2_s23 = self.F2(s23, l_i=l_i, l_j=l_j)
        F3_s23 = self.F3(s23, l_i=l_i, l_j=l_j)

        F2_5_s23 = self.F2_5(s23, l_i=l_i, l_j=l_j)
        F3_5_s23 = self.F3_5(s23, l_i=l_i, l_j=l_j)

        s13 = (
            self.m_a**2 + LEPTON_MASSES[l_i] ** 2 + LEPTON_MASSES[l_j] ** 2 - s12 - s23
        )

        # Denominator factors like (s23 - m_a^2 + i m_a Gamma_a)
        denom_s23 = s23 - self.m_a**2 + 1j * self.m_a * self.Gamma_a
        # denom_s13_p = s13 - self.m_a**2 + 1j * self.m_a * self.Gamma_a
        denom_s13_m = s13 - self.m_a**2 - 1j * self.m_a * self.Gamma_a

        # Absolute square of denom_s23:
        abs_denom_s23_sq = abs(denom_s23) ** 2

        # Real part of something like (denom_s23 * denom_s13_m):
        re_s23_s13 = (denom_s23 * denom_s13_m).real

        #
        # --- Overall prefactors: (|ke_12|^2 + |kE_12|^2)*|c_ee|^2*(m_e^2 m_mu^2 / f_a^4)
        #
        # If ke_12 etc. are real in your notation, just do ke_12**2.
        # If they might be complex, you could do abs(ke_12)**2.
        #
        prefactor = (
            (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
            * abs(self.c_lepton[l_i, l_i]) ** 2
            * LEPTON_MASSES[l_j] ** 2
            * LEPTON_MASSES[l_i] ** 2
            / self.f_a**4
        )

        #
        # --- First big curly bracket from eq. (4.33) ---
        #
        #  2 * s23 * (s12 + s13) / |denom_s23|^2
        #  - s13*s23 / Re[...something...]
        #
        bracket_1 = 2.0 * s23 * (s12 + s13) / abs_denom_s23_sq - (
            s13 * s23 / (re_s23_s13)
        )  # etc. fill in exactly...

        part1 = prefactor * bracket_1

        #
        # --- Next big chunk: the " + 4 e^2 [...] " piece ---
        #
        # We'll just illustrate a few terms and you can fill out the rest.
        #
        # 4 * e^2 * [
        #    2*(s12 + s13) * Re[F2^*(s23)*F3(s23) + F2^5*(s23)*F3^5(s23)]
        #  + (1/s23)*(...)*(|F2(s23)|^2 + |F2^5(s23)|^2)
        #  + ...
        # ]
        #

        # Real part of ( F2_s23.conjugate()*F3_s23 + F2_5_s23.conjugate()*F3_5_s23 )
        re_F2F3 = (F2_s23.conjugate() * F3_s23 + F2_5_s23.conjugate() * F3_5_s23).real

        termA = 2.0 * (s12 + s13) * re_F2F3
        termB = (1.0 / s23) * (
            (LEPTON_MASSES[l_i] ** 2 * (s12 + s13) - 2.0 * s12 * s13)
            * (abs(F2_s23) ** 2 + abs(F2_5_s23) ** 2)
        )
        # etc.  Build them all up and sum:
        bracket_2 = termA + termB  # + the rest of your terms carefully transcribed

        part2 = 4.0 * const.eQED**2 * bracket_2

        #
        # --- Next chunk: the " + 2 e s23 me / f_a^2 c_ee Re[...] " piece ---
        #
        # For example:
        #
        # 2 * e * s23 * m_e / f_a^2 * c_ee * Re[ something with (ke_21 + kE_21)/(...) etc. ]
        #
        # Suppose you name that big bracket bracket_3:
        #
        bracket_3_real = 0.0  # fill in carefully
        part3 = (
            2.0
            * const.eQED
            * s23
            * LEPTON_MASSES[l_j]
            / (self.f_a**2)
            * self.c_lepton[l_j, l_j]
            * bracket_3_real
        )

        #
        # --- Finally, sum them up (still not adding the (1 <-> 2) piece) ---
        #
        M_sq_no_sym = part1 + part2 + part3

        return M_sq_no_sym

    # def BR_diff_li_to_lj_a_gamma(self, s12, s23, l_i, l_j):
    #     """Differential decay rate for li -> lj + a + gamma
    #     Args:
    #         s12: invariant mass of the (lj + gamma) system
    #         s23: invariant mass of the (gamma + a) system
    #     """
    #     denom = s12 * (self.m_a**2 - s12 - s23) ** 2
    #     curlyF = (
    #         self.m_a**6
    #         - s23**2 * (s12 + s23)
    #         - self.m_a**4 * (2 * LEPTON_MASSES[l_i] ** 2 + s12 + s23)
    #         + 2 * LEPTON_MASSES[l_i] ** 2 * (s12 + s23) * (2 * s12 + s23)
    #         - 2 * LEPTON_MASSES[l_i] ** 4 * (4 * s12 + s23)
    #         + self.m_a**2
    #         * (2 * LEPTON_MASSES[l_i] ** 4 + 4 * LEPTON_MASSES[l_i] ** 2 * s12 + s23**2)
    #     )
    #     curlyF /= denom

    #     return (
    #         const.alphaQED
    #         / 4
    #         / np.pi**2
    #         / 32
    #         / LEPTON_MASSES[l_i]
    #         * curlyF
    #         * (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
    #         / self.f_a**2
    #     )

    def BR_diff_li_to_lj_a_gamma_dEa(self, l_i, l_j, Ea):
        """Differential decay rate for li -> lj + a + gamma
        Args:
            x: 2 E_j/m_i
            y: 2 E_gamma/m_i
            z: 2 E_a/m_i
        """

        eta = self.m_a**2 / LEPTON_MASSES[l_i] ** 2
        z = 2 * Ea / LEPTON_MASSES[l_i]
        Integrand = (
            -0.5 * ((-10 + 4 * eta + 3 * z) * Sqrt(-4 * eta + Power(z, 2)))
            - (5 + (-2 + eta) * eta + (-4 + z) * z)
            * Log(-2 + z - Sqrt(-4 * eta + Power(z, 2)))
            + (5 + (-2 + eta) * eta + (-4 + z) * z)
            * Log(-2 + z + Sqrt(-4 * eta + Power(z, 2)))
        ) / (-1 - eta + z)

        return (
            LEPTON_MASSES[l_i] ** 3
            * const.alphaQED
            / 32
            / np.pi**2
            * Integrand
            * (2 / LEPTON_MASSES[l_i])
            * (Ea < LEPTON_MASSES[l_i] * (1 - eta))
            * (self.m_a < Ea)
            * (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
        )

    def BR_diff_li_to_lj_a_gamma_normed(self, x, y, l_i, l_j, m_a):
        """Differential decay rate for li -> lj + a + gamma
        Args:
            x: 2 E_j/m_i
            y: 2 E_gamma/m_i
        """
        eta = m_a**2 / LEPTON_MASSES[l_i] ** 2
        denom = y**2 * (1 - x - y - eta)
        num = y * (1 - x**2 - eta**2) - 2 * (1 - eta) * (1 - x - eta)

        if denom != 0:
            return (
                LEPTON_MASSES[l_i] ** 3
                * const.alphaQED
                / 32
                / np.pi**2
                * num
                / denom
                * (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
            )
        else:
            return 0

    def BR_li_to_lj_a_gamma(
        self, l_i, l_j, E_gamma_min=0.0, E_e_min=const.m_e, theta_eg_min=0.0
    ):
        """Branching ratio for l_i -> l_j + a + gamma"""

        eta = self.m_a**2 / LEPTON_MASSES[l_i] ** 2

        # if isinstance(self.m_a, np.ndarray) and self.m_a.ndim == 2:
        # Gamma_a = 0
        # else:
        Gamma_a, _ = dblquad(
            lambda x, y: self.BR_diff_li_to_lj_a_gamma_normed(x, y, l_i, l_j, self.m_a)
            * (y > 2 * E_gamma_min / LEPTON_MASSES[l_i])
            * (x > 2 * E_e_min / LEPTON_MASSES[l_i])
            * ((1 + 2 * (1 - x - y - eta) / x / y) < np.cos(theta_eg_min)),
            a=2 * E_gamma_min / LEPTON_MASSES[l_i],
            b=1 - eta,
            gfun=lambda y: max((1 - y - eta), 2 * E_e_min / LEPTON_MASSES[l_i]),
            hfun=lambda y: (1 - y - eta) / (1 - y),
        )

        return Gamma_a / self.f_a**2 / LEPTON_WIDTHS[l_i]

    # def BR_li_to_lj_a_gamma(self, l_i, l_j):
    #     """Branching ratio for mu -> e + a + gamma"""

    #     Gamma_a, _ = dblquad(
    #         lambda s23, s12: self.BR_diff_li_to_lj_a_gamma(s12, s23, l_i, l_j),
    #         a=s12_min(LEPTON_MASSES[l_i], LEPTON_MASSES[l_j], 0, self.m_a),
    #         b=s12_max(LEPTON_MASSES[l_i], LEPTON_MASSES[l_j], 0, self.m_a),
    #         gfun=lambda s12: s23_min(
    #             s12, LEPTON_MASSES[l_i], LEPTON_MASSES[l_j], 0, self.m_a
    #         ),
    #         hfun=lambda s12: s23_max(
    #             s12, LEPTON_MASSES[l_i], LEPTON_MASSES[l_j], 0, self.m_a
    #         ),
    #     )
    #     return Gamma_a / Gamma_mu

    def F2_5(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -LEPTON_MASSES[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] + self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * ps.g1(q, LEPTON_MASSES[l_i])
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * ps.g2(q, LEPTON_MASSES[l_i], self.f_a)
            )
        )

    def F2(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -LEPTON_MASSES[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] - self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * ps.g1(q, LEPTON_MASSES[l_i])
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * ps.g2(q, LEPTON_MASSES[l_i], self.f_a)
            )
        )

    def F3(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -LEPTON_MASSES[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] - self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * ps.l1(q, LEPTON_MASSES[l_i], self.m_a)
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * ps.l2(q, LEPTON_MASSES[l_i], self.f_a)
            )
        )

    def F3_5(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -LEPTON_MASSES[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] + self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * ps.l1(q, LEPTON_MASSES[l_i], self.m_a)
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * ps.l2(q, LEPTON_MASSES[l_i], self.f_a)
            )
        )

    """
    BEGIN Estimates of Onia to ALPs
    """

    def BR_Psi2S_to_li_lj_a(self, l_i, l_j):
        """Branching ratio for Psi2S -> li + lj + a"""
        Brfactor = (
            self.c_lepton[l_i, l_j] ** 2
            / 16
            / np.pi**2
            * (LEPTON_MASSES[l_i] + LEPTON_MASSES[l_j]) ** 2
            / self.f_a**2
        ) * (self.m_a < 3.686 - LEPTON_MASSES[l_i] - LEPTON_MASSES[l_j])
        if l_i == 2 or l_j == 2:
            return Brfactor * 3e-3
        else:
            return Brfactor * 8e-3

    """
    END Estimates of Onia to ALPs
    """

    def delta_a_mag_mom(self, l_i):
        """Return the coupling explaining the delta_amu for a certain mz.
        The contribution of the model to the the a_mu is given by:
        delta_amu = factor * coupling^2 * m_ratio^2 * integrand_function(m_ratio)
        """
        x = (self.m_a / LEPTON_MASSES[l_i]) ** 2 * (1.0 + 0.0j)
        prefactor = -(
            (LEPTON_MASSES[l_i] * self.c_lepton[l_i, l_i] / self.f_a / 4 / np.pi) ** 2
        )
        muSQR = (1e3) ** 2  # TeV^2, scale for the running of alphaQED
        delta_a = prefactor * (
            h1_gminus2(x)
            # + 2
            # * const.alphaQED
            # / np.pi
            # * self.c_gg
            # / self.c_lepton[l_i, l_i]
            # * (np.log(muSQR / LEPTON_MASSES[l_i] ** 2) - h2_gminus2(x))
        )
        return delta_a


# Expressions from Bauer et al (https://arxiv.org/pdf/2110.10698)
def h_gminus2(x):
    return 2 * x**2 / (x - 1) ** 3 * np.log(x) - (3 * x - 1) / (x - 1) ** 2


def j_gminus2(x):
    return 1 + 2 * x - 2 * x**2 * np.log(x / (x - 1))


def h1_gminus2(x):
    return np.where(
        x < 4,
        1
        + 2 * x
        + (1 - x) * x * np.log(x)
        - 2 * x * (3 - x) * np.sqrt(x / (4 - x)) * np.arccos(np.sqrt(x) / 2),
        1
        + 2 * x
        + (1 - x) * x * np.log(x)
        - 2 * x * (3 - x) * np.sqrt(x / (x - 4)) * np.arccosh(np.sqrt(x) / 2),
    )


def h2_gminus2(x):
    return np.where(
        x < 4,
        1
        - x / 3
        + x**2 * np.log(x)
        + (x + 2) / 3 * np.sqrt((4 - x) * x) * np.arccos(np.sqrt(x) / 2),
        1 - x / 3
        # + x**2 * np.log(x)
        - (x + 2) / 3 * np.sqrt((x - 4) * x) * np.arccosh(np.sqrt(x) / 2),
    )


def s23_max(s12, M, m1, m2, m3):
    """max of s23 = (p2+p3)^2 for given M, m1, m2, m3"""
    E2_star = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))
    E3_star = (M**2 - s12 - m3**2) / (2 * np.sqrt(s12))

    return (E2_star + E3_star) ** 2 - (
        np.sqrt(E2_star**2 - m2**2) - np.sqrt(E3_star**2 - m3**2)
    ) ** 2


def s23_min(s12, M, m1, m2, m3):
    """min of s23 = (p2+p3)^2 for given M, m1, m2, m3"""
    E2_star = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))
    E3_star = (M**2 - s12 - m3**2) / (2 * np.sqrt(s12))

    return (E2_star + E3_star) ** 2 - (
        np.sqrt(E2_star**2 - m2**2) + np.sqrt(E3_star**2 - m3**2)
    ) ** 2


def s12_max(M, m1, m2, m3):
    """max of s12 = (P - p3)^2 for given M, m1, m2, m3"""
    return (M - m3) ** 2


def s12_min(M, m1, m2, m3):
    """min of s12 = (P - p3)^2 for given M, m1, m2, m3"""
    return (m1 + m2) ** 2


def Power(x, y):
    return x**y


def Log(x):
    return np.log(x)


def Sqrt(x):
    return np.sqrt(x)


def Iv_mvv_integral(Ea, ma, mlep, mtau=const.m_tau):
    """Integral for the visible decay width of ALP decays

    Integral of (I * v) over mvvSQR

    From MATHEMATICA:

        v[a_, b_, c_] := Sqrt[1 - 2 (b^2 + c^2)/a^2 + (b^2 - c^2)^2/a^4];

        mtaubarSQR = mtau^2 + ma^2 - 2*mtau*Ea;

        mvvSQRMax = (Sqrt[mtaubarSQR] - mlep)^2;

        Ifunc[mtbarSQR_, mvvSQR_] := (1 + (mvvSQR - 2*mlep^2)/mtaubar^2 + (mlep^4 + mlep^2 mvvSQR - 2  mvvSQR^2)/mtaubar)*v[Sqrt[mtaubarSQR], Sqrt[mvvSQR], mlep];

    """

    return (
        Power(ma, 2)
        - 8 * Power(mlep, 2)
        - 2 * Ea * mtau
        + Power(mtau, 2)
        - Power(mlep, 8) / Power(Power(ma, 2) - 2 * Ea * mtau + Power(mtau, 2), 3)
        + (8 * Power(mlep, 6)) / Power(Power(ma, 2) - 2 * Ea * mtau + Power(mtau, 2), 2)
    ) / 2.0 + (
        6
        * Power(mlep, 4)
        * Log((Power(ma, 2) - 2 * Ea * mtau + Power(mtau, 2)) / Power(mlep, 2))
    ) / (
        Power(ma, 2) + mtau * (-2 * Ea + mtau)
    )


#
# Now we make a little helper that adds the (1 <-> 2) piece.
# That means we add M_squared(...) + M_squared(...) with s12<->s13 swapped.
# In many references, "1 <-> 2" indicates that one must symmetrize w.r.t.
# permutations of the final-state momenta, so you effectively add the
# expression again but with s12 and s13 swapped (and if the form-factor
# arguments likewise need swapping).
#


# def M_squared_sym(
#     s12,
#     s13,
#     s23,
#     m_e,
#     m_mu,
#     f_a,
#     m_a,
#     Gamma_a,
#     c_ee,
#     ke_12,
#     kE_12,
#     e,
#     # F2 etc. at both s23 and s13
#     F2_s23,
#     F2_5_s23,
#     F3_s23,
#     F3_5_s23,
#     F2_s13,
#     F2_5_s13,
#     F3_s13,
#     F3_5_s13,
# ):
#     """
#     Returns the *fully symmetrized* |M|^2 = expression + (1 <-> 2).
#     """
#     # The original piece:
#     val = M_squared(
#         s12,
#         s13,
#         s23,
#         m_e,
#         m_mu,
#         f_a,
#         m_a,
#         Gamma_a,
#         c_ee,
#         ke_12,
#         kE_12,
#         e,
#         F2_s23,
#         F2_5_s23,
#         F3_s23,
#         F3_5_s23,
#         F2_s13,
#         F2_5_s13,
#         F3_s13,
#         F3_5_s13,
#     )
#     # The swapped piece (swap s12 <-> s13, also swap the form-factor inputs that
#     # used to go with s13 vs. s12):
#     val_swapped = M_squared(
#         s13,
#         s12,
#         s23,
#         m_e,
#         m_mu,
#         f_a,
#         m_a,
#         Gamma_a,
#         c_ee,
#         ke_12,
#         kE_12,
#         e,
#         # notice we swap which F_*(sXX) we pass in
#         F2_s23=F2_s23,
#         F2_5_s23=F2_5_s23,
#         F3_s23=F3_s23,
#         F3_5_s23=F3_5_s23,
#         F2_s13=F2_s12,
#         F2_5_s13=F2_5_s12,
#         F3_s13=F3_s12,
#         F3_5_s13=F3_5_s12,
#     )
#     # sum them
#     return val + val_swapped
