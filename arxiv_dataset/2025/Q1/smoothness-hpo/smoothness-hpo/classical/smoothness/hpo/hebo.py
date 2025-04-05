import time
from typing import Tuple

from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace

from smoothness.hpo.base import BaseHPO


class HeboHPO(BaseHPO):
    def run(self, n_runs: int, n_iters: int) -> Tuple[list, float]:
        """
        Runs random HPO.
        :param n_runs: Number of runs to perform.
        :param n_iters: Number of iterations to explore.
        :return:
        """
        start = time.time()
        config_space = DesignSpace().parse(self.hpo_space)

        scores = []
        for _ in range(n_runs):
            hebo = HEBO(config_space)
            best_score = (-1,)
            for _ in range(n_iters):
                config = hebo.suggest(n_suggestions=1)
                score = self.query_fn(config)
                hebo.observe(config, score)

                if isinstance(score, float):
                    score = (score,)

                if score > best_score:
                    best_score = score

            scores.append(best_score)

        end = time.time()
        return scores, end - start
