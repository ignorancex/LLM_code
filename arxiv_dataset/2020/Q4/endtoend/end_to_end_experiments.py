import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
import seaborn as sns

from project_typing import *

EPS = 1e-6


def apportion_population_numbers(weights: np.ndarray, population_size: int) -> np.ndarray:
    sum_weights = sum(weights)
    number_copies = np.floor_divide(population_size * weights, sum_weights).astype(int)
    remainders = [(rem, i) for i, rem in enumerate(np.remainder(population_size * weights, sum_weights))]
    remainders.sort(reverse=True)
    still_missing = population_size - sum(number_copies)
    for rem, i in remainders[:still_missing]:
        number_copies[i] += 1
    return number_copies


class Population:
    def __init__(self, population_size: int, background: List[Agent], feature_values: Dict[Feature, Set[Value]],
                 background_qis: np.ndarray, background_weights: np.ndarray):
        self.num_origs = len(background)
        assert len(background_qis) == self.num_origs
        assert len(background_weights) == self.num_origs

        self.population_size = population_size
        self.background = background
        self.feature_values = feature_values
        self.background_qis = background_qis
        self.background_weights = background_weights

        self.popnum = apportion_population_numbers(background_weights, population_size)
        self.cum_counts = np.add.accumulate(self.popnum)
        assert population_size == self.cum_counts[-1]
        self.ais = 1 / self.background_qis
        self.qstar = min(background_qis)
        self.fvfractions = {}
        for feature, values in feature_values.items():
            for value in values:
                self.fvfractions[(feature, value)] = sum(num for num, orig in zip(self.popnum, self.background)
                                                         if orig[feature] == value) / population_size

    def alpha(self, r: int, k: int):
        return self.qstar * r / k

    def is_good_pool(self, pool: np.ndarray, pis: np.ndarray, num_deterministic: int, r: int, k: int):
        if num_deterministic > 0:
            # Pool not good because `num_deterministic` unclipped πs would exceed 1
            return 3
        alphaexp = self.alpha(r, k)**(-.49)
        for feature, values in self.feature_values.items():
            for value in values:
                proportional = self.fvfractions[(feature, value)] * k
                sum_pis = sum(pi * number for orig, number, pi in zip(self.background, pool, pis)
                              if orig[feature] == value)
                if (1 - alphaexp) * proportional > sum_pis:
                    # Pool not good: Σ π_i,P for `feature`:`value` below (1 - `alphaexp`) * k * nfv / n
                    return 2
                elif (1 + alphaexp) * proportional < sum_pis:
                    # Pool not good: Σ π_i,P for `feature`:`value` exceeds (1 + `alphaexp`) * k * nfv / n
                    return 2
        sum_ais = sum(num / qi for num, qi in zip(pool, self.background_qis))
        if sum_ais > r / (1 - alphaexp):
            # Pool not good because Σ aᵢ too large
            return 1
        return 0

    def sample_recipients(self, num_recipients: int) -> Counter:
        selections = rng.choice(self.population_size, num_recipients, replace=False)
        indices = np.searchsorted(self.cum_counts, selections, side='right')
        num_copies_in_recipients = Counter(indices)
        recipients = np.array([num_copies_in_recipients[i] if i in num_copies_in_recipients else 0
                               for i in range(self.num_origs)])
        assert sum(recipients) == num_recipients
        return recipients

    def sample_pool(self, recipients: Counter, r: int, k: int) -> Tuple[int, np.ndarray, np.ndarray]:
        sampled = []
        for i in range(self.num_origs):
            qi = self.background_qis[i]
            num_recipients = recipients[i]
            sampled.append(rng.binomial(num_recipients, qi))
        pool = np.array(sampled).astype(int)

        mult = 0.
        clipped = mult * self.ais > 1.
        while True:
            k2 = k - sum(pool[clipped])
            old_mult = mult
            mult = k2 / sum((~clipped) * pool * self.ais)
            assert mult >= old_mult
            old_clipped = clipped
            clipped = mult * self.ais > 1.
            if all(clipped == old_clipped):
                pis = np.clip(mult * self.ais, 0., 1.)
                #print(sum(pool * pis), k)
                assert abs(sum(pool * pis) - k) < EPS
                num_deterministic = sum(pool[clipped])
                break
        good_pool = self.is_good_pool(pool, pis, num_deterministic, r, k)

        return good_pool, pool, pis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate participation probabilities and run selection algorithm on pool.")
    parser.add_argument("--r", type=int, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("directory", type=str, help="input and output directory ./output/{directory}/")
    args = parser.parse_args()

    base_directory = Path("./output/" + args.directory)
    assert base_directory.exists()
    est_dir = base_directory.joinpath("estimation")
    assert est_dir.exists()
    directory = base_directory.joinpath("endtoend")
    directory.mkdir(parents=False, exist_ok=True)

    with open(est_dir.joinpath("background_qis.pickle"), "rb") as file:
        data = load(file)

    background = data["background"]
    feature_values = data["feature_values"]
    background_qis = np.array(data["background_qis"])
    background_weights = np.array(data["background_weights"])

    rng = np.random.default_rng(seed=args.seed)

    n = 60000000
    pop = Population(n, background, feature_values, background_qis, background_weights)

    k = 110

    timestamps = []
    for r in args.r:
        DEGREES_POOL_GOODNESS = 4
        sum_expectations = np.zeros((DEGREES_POOL_GOODNESS, len(background)))
        num_averages = args.steps
        start = datetime.now()
        print("Start: ", start)
        for i in range(num_averages):
            print(i, r)
            recipients = pop.sample_recipients(r)
            good, pool, pis = pop.sample_pool(recipients, r, k)
            sum_expectations[good] += pool * pis
        end = datetime.now()
        print("End: ", end)
        timestamps.append((str(start), str(end)))

        normalized_averages = (n / k / num_averages / pop.popnum) * sum_expectations
        up_to_normalized_averages = np.add.accumulate(normalized_averages)  # 0: good pools, i: good pools + 1-good + … + i-good
        df2 = pd.DataFrame({"qi": background_qis, "normalized good": normalized_averages[0],
                            "normalized satisfying 1&2": up_to_normalized_averages[1],
                            "normalized satisfying 1": up_to_normalized_averages[2],
                            "normalized any": up_to_normalized_averages[3]})
        df2.to_csv(directory.joinpath(f"end_to_end_r{r}_n{num_averages}.csv"))
        f = sns.relplot(x = "qi", y = "normalized good", data=df2, marker="+")
        f.savefig(directory.joinpath(f"good0_r{r}_n{num_averages}.pdf"))
        f = sns.relplot(x = "qi", y = "normalized satisfying 1&2", data=df2, marker="+")
        f.savefig(directory.joinpath(f"good1_r{r}_n{num_averages}.pdf"))
        f = sns.relplot(x = "qi", y = "normalized satisfying 1", data=df2, marker="+")
        f.savefig(directory.joinpath(f"good2_r{r}_n{num_averages}.pdf"))
        f = sns.relplot(x = "qi", y = "normalized any", data=df2, marker="+")
        f.savefig(directory.joinpath(f"good3_r{r}_n{num_averages}.pdf"))

    id = args.id if args.id is not None else ""
    log_path = directory.joinpath(f"times{id}.log")
    with open(log_path, "w") as file:
        for r, (start, end) in zip(args.r, timestamps):
            file.write(f"r={r}, start={start}, end={end}\n")