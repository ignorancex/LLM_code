from inflation import InflationLP, InflationProblem

import numpy as np

def prob(E1, E2):
    # We write the distribution for E3=0, because in the end E3 does not enter our calculations
    p = np.zeros((2, 2, 2, 1, 1, 1))
    for a, b, c in np.ndindex(2, 2, 2):
        p[a, b, c, 0, 0, 0] = 1/8 * (1 + ((-1)**a+(-1)**b+(-1)**c) * E1
                                       + ((-1)**(a+b)+(-1)**(b+c)+(-1)**(c+a)) * E2)
    return p

ip = InflationProblem(dag={"l1": ["A", "B"],
                           "l2": ["B", "C"],
                           "l3": ["C", "A"]},
                      outcomes_per_party=[2, 2, 2],
                      inflation_level_per_source=[2, 2, 2])

lp = InflationLP(ip)
# Easier than finding the optimal E3 is removing p(abc) from the list of
# operators that we can assign values
del lp.knowable_atoms[-1]

E1s = np.linspace(0.15, 0.5, 100)

results = []
for E1 in E1s:
    left = -1/3
    right = 0
    while right - left > 1e-4:
        E2 = (left + right) / 2
        lp.reset('all')
        lp.set_distribution(prob(E1, E2), use_lpi_constraints=True)
        lp.solve(feas_as_optim=True)
        if isinstance(lp.primal_objective, str):
            right = E2
        elif lp.primal_objective < -1e-7:
            left = E2
        else:
            right = E2
    results.append([E1, E2])

np.savetxt('e1e2_nsi_boundary_old.txt', results)