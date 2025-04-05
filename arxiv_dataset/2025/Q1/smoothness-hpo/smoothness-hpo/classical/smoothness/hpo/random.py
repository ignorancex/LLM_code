import time
from typing import Tuple

from smoothness.hpo.base import BaseHPO
from smoothness.hpo.util import get_many_random_hyperparams


class RandomHPO(BaseHPO):
    def run(self, n_runs: int, n_iters: int) -> Tuple[list, float]:
        """
        Runs random HPO.
        :param n_runs: Number of runs to perform.
        :param n_iters: Number of iterations to explore.
        :return:
        """
        start = time.time()

        scores = []
        for _ in range(n_runs):
            configs = get_many_random_hyperparams(self.hpo_space, n_iters)

            best_score = (-1,)

            for config in configs:
                score = self.query_fn(config)

                if isinstance(score, float):
                    score = (score,)

                if score > best_score:
                    best_score = score

            scores.append(best_score)

        end = time.time()
        return scores, end - start
