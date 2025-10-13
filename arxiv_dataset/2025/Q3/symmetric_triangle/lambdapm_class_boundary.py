from inflation import InflationLP, InflationProblem

import numpy as np

def prob(p, q):
    return np.array([[[[[[p]]], [[[(1-p-q)/6]]]],
                            [[[[(1-p-q)/6]]], [[[(1-p-q)/6]]]]],
                        [[[[[(1-p-q)/6]]], [[[(1-p-q)/6]]]],
                            [[[[(1-p-q)/6]]], [[[q]]]]]])

ip = InflationProblem(dag={"l1": ["A", "B"],
                           "l2": ["B", "C"],
                           "l3": ["C", "A"]},
                      outcomes_per_party=[2, 2, 2],
                      inflation_level_per_source=[2, 2, 2],
                    #   classical_sources='all'
                      )

lp = InflationLP(ip)

ps = np.arange(0.01, 1, 0.01)

results = []
for p in ps:
    left = 0
    right = 1 - p
    while right - left > 1e-4:
        q = (left + right) / 2
        lp.reset('all')
        lp.set_distribution(prob(p, q), use_lpi_constraints=True)
        lp.solve(feas_as_optim=True)
        if isinstance(lp.primal_objective, str):
            left = q
        elif lp.primal_objective < -1e-7:
            right = q
        else:
            left = q
    results.append([p, q])

np.savetxt('pq_nsi_boundary.txt', results)