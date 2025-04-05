import time

from typing import Tuple

from smoothness.hpo.base import BaseHPO

from ConfigSpace import ConfigurationSpace
from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband


class BohbHPO(BaseHPO):
    def run(self, n_runs: int, n_iters: int) -> Tuple[list, float]:
        """
        Runs random HPO.
        :param n_runs: Number of runs to perform.
        :param n_iters: Number of iterations to explore.
        :return:
        """
        start = time.time()
        config_space = ConfigurationSpace(self.hpo_space)

        scores = []
        for _ in range(n_runs):
            scenario = Scenario(config_space, n_trials=n_iters, min_budget=100, max_budget=100)
            intensifier = Hyperband(scenario, incumbent_selection='highest_budget')
            smac = MultiFidelityFacade(scenario, self.query_fn, intensifier=intensifier)
            incumbent = smac.optimize()

            scores.append((self.query_fn(incumbent),))

        end = time.time()
        return scores, end - start
