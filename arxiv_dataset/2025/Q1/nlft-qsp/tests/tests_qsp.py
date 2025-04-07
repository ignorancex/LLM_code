
import unittest

from nlft import NonLinearFourierSequence
from qsp import gqsp_solve, nlfs_to_phase_factors, xqsp_solve
from rand import random_polynomial, random_real_polynomial, random_sequence

import numerics as bd


class QSPTestCase(unittest.TestCase):
    
    @bd.workdps(30)
    def test_gqsp_phase_factors(self):
        nlft = NonLinearFourierSequence(random_sequence(100, 16))

        pf = nlfs_to_phase_factors(nlft)
        nlft2 = pf.to_nlfs()

        self.assertAlmostEqual(pf.phase_offset(), 0, delta=bd.machine_threshold())
        for a, b in zip(nlft.coeffs, nlft2.coeffs):
            self.assertAlmostEqual(a, b, delta=1e-10)

    @bd.workdps(30)
    def test_qsp_polynomial_gen(self):
        nlft = NonLinearFourierSequence(random_sequence(1000000, 16))
        qsp = nlfs_to_phase_factors(nlft)

        a, b = nlft.transform()
        P, Q = qsp.polynomials()

        self.assertAlmostEqual((a.shift(15) - P).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((b - Q).l2_norm(), 0, delta=bd.machine_threshold())

        qsp2 = nlfs_to_phase_factors(nlft, alpha=1)
        P2, Q2 = qsp2.polynomials()
        self.assertAlmostEqual((P2 - P * bd.exp(1j)).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((Q2 - Q).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_qsp_polynomial_laurent_analytic(self):
        nlft = NonLinearFourierSequence(random_sequence(100, 17))
        qsp = nlfs_to_phase_factors(nlft)

        P, Q = qsp.polynomials()
        Pl, Ql = qsp.polynomials(mode='laurent')

        self.assertEqual(P.support_start, 0)
        self.assertEqual(Q.support_start, 0)
        n = P.effective_degree()

        self.assertEqual(Pl.support_start, -n)
        self.assertEqual(Ql.support_start, -n)
        for k in P.support():
            self.assertAlmostEqual(Pl[2*k - n], P[k], delta=bd.machine_threshold())
            self.assertAlmostEqual(Ql[2*k - n], Q[k], delta=bd.machine_threshold())
            self.assertAlmostEqual(Pl[2*k - n + 1], 0, delta=bd.machine_threshold())
            self.assertAlmostEqual(Ql[2*k - n + 1], 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_gqsp_solver(self):
        P = random_polynomial(16, eta=0.5)

        qsp = gqsp_solve(P, mode='nlft')
        Q2, P2 = qsp.polynomials()

        self.assertAlmostEqual((P - P2).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_gqsp_solver_ix(self):
        P = random_polynomial(1024, eta=0.5)

        qsp1 = gqsp_solve(P, mode='nlft')
        qsp2 = gqsp_solve(P)
        Q1, P1 = qsp1.polynomials() # (Q, P)
        P2, Q2 = qsp2.polynomials() # (P, iQ)

        self.assertAlmostEqual((P - P2).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((P1 - P2).l2_norm(), 0, delta=bd.machine_threshold())

        self.assertAlmostEqual((1j*Q1 - Q2).l2_norm(), 0, delta=bd.machine_threshold())

    @bd.workdps(30)
    def test_xqsp_solver(self):
        P = random_real_polynomial(16, eta=0.5)

        qsp = xqsp_solve(1j*P, mode='nlft')
        Q2, P2 = qsp.polynomials()

        self.assertAlmostEqual((1j*P - P2).l2_norm(), 0, delta=bd.machine_threshold())

        qsp = xqsp_solve(P)
        P2, Q2 = qsp.polynomials()

        self.assertAlmostEqual((P - P2).l2_norm(), 0, delta=bd.machine_threshold())



if __name__ == '__main__':
    unittest.main()