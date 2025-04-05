import ast
import heapq

from functools import partial

import numpy as np

from matplotlib import pyplot as plt


plt.style.use('seaborn-v0_8')


def div(y, x):
    return float(y / x) if x != 0 else 0.


lines = open('sensitivity_results.txt', 'r').readlines()

datasets = []
results = []
cur_results = []

# Grab the results
for line in lines:
    if line.startswith('Running'):
        datasets.append(line.split(' ')[1][:-1])

        if cur_results:
            results.append(cur_results)
        cur_results = []
        continue

    parts = line.split('|')
    smoothness = float(parts[1].split(':')[1])
    cur_result = ast.literal_eval(parts[2].split(':')[1])

    cur_results.append((smoothness, cur_result))

results.append(cur_results)
min_len = min(len(x) for x in results if len(x) > 50)  # ignore maven
results = [x[:min_len] for x in results if len(x) >= min_len]
datasets = [x for x in datasets if x != 'maven:']

# Do the computation we would if this was online
all_regrets = []
for result_set in results:
    smoothness, perf = zip(*result_set, strict=True)

    def get_regret(perf, smoothness, k):
        best_perfs = []
        for i in range(20, len(smoothness)):
            highest = heapq.nlargest(k, enumerate(smoothness[:i]), key=lambda x: x[1])
            idx, _ = zip(*highest, strict=True)

            chosen_perfs = np.array(perf)[[idx]][0]
            best_perf = max(chosen_perfs, key=lambda x: x[0])
            best_perfs.append(best_perf)

        # Compute the normalized regret
        # 0 = pd - pf, 1 = pd, 2 = pf, 3 = prec, 4 = auc
        regrets = [div(x[0] - best_perfs[0][0], best_perfs[0][0]) for x in best_perfs]
        return regrets

    regret_fn = partial(get_regret, perf, smoothness)
    regrets = [regret_fn(k) for k in [5, 10, 15]]
    all_regrets.append(regrets)

all_regrets = np.array(all_regrets)
assert len(all_regrets.shape) == 3  # (datasets, k, evals)

all_regrets = np.swapaxes(all_regrets, 0, 1)
assert all_regrets.shape == (3, 7, 56), all_regrets.shape

fig, ax = plt.subplots(dpi=150)
for i, k in enumerate([5, 10, 15]):
    plt.plot(range(20, 76), -np.mean(all_regrets[i, ...], axis=0), label=f'$N_2 = {k}$')

plt.title('Regret over evaluations')
plt.xlabel('$N_1$ (#evals)')
plt.ylabel('Normalized regret')
plt.legend()
plt.show()

fig.savefig('sensitivity.png')
