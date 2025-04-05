from typing import Callable, Union, Tuple


class BaseHPO:
    def __init__(self, hpo_space: dict, learner: str, query_fn: Callable[[dict], Union[tuple, float]]):
        """
        Random HPO.
        :param hpo_space: A space of hyper-parameters to explore
        :param learner: The learner to use
        :param query_fn: A function that can be queried for a score. Must take in a dict as
        input, and return either a float representing a score, or a tuple representing a
        set of scores ordered in descending order of importance.
        """
        self.hpo_space = hpo_space
        self.learner = learner
        self.query_fn = query_fn

    def run(self, n_runs: int, n_iters: int) -> Tuple[list, float]:
        raise NotImplementedError
