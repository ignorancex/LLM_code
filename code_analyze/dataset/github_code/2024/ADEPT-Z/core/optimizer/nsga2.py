import random
from typing import Callable

import nevergrad.common.typing as tp
import numpy as np
from matplotlib import pyplot as plt
from nevergrad.parametrization import parameter as p
from pyutils.general import logger

# from mpl_toolkits.mplot3d import Axes3D
from .base import EvolutionarySearchBase

__all__ = ["NSGA2"]


class CrowdingDistance(object):
    """This class implements the calculation of crowding distance for NSGA-II."""

    def __init__(
        self,
        maximize: bool = True,
    ):
        self._maximize = maximize

    def accumulate_distance_per_objective(self, front: tp.List[p.Parameter], i: int):
        if isinstance(front[0].losses, np.ndarray) and front[0].losses.shape != ():
            is_multiobj: bool = (
                len(front[0].losses) > 1
            )  # isinstance(front[0].loss, np.ndarray)
        else:
            is_multiobj = False
        assert (not is_multiobj and (i == 0)) or is_multiobj

        # Sort the population by objective i
        if is_multiobj:
            # print("front before sorting:", front)
            front = sorted(front, key=lambda x: x.losses[i], reverse=self._maximize)
            # print("front after sorting:", front)

            # objective_minn = front[0].losses[i]
            # objective_maxn = front[-1].losses[i]
            objective_minn = front[-1].losses[i]
            objective_maxn = front[0].losses[i]

            # print("i: ", i)
            # print("objective_minn:", objective_minn)
            # print("objective_maxn:", objective_maxn)

            assert objective_minn <= objective_maxn

            # Set the crowding distance
            front[0]._meta["crowding_distance"] = float("inf")
            front[-1]._meta["crowding_distance"] = float("inf")

            # All other intermediate solutions are assigned a distance value equal
            # to the absolute normalized difference in the function values of two
            # adjacent solutions.
            for j in range(1, len(front) - 1):
                # distance = front[j + 1].losses[i] - front[j - 1].losses[i]
                distance = front[j - 1].losses[i] - front[j + 1].losses[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass  # undefined
                else:
                    distance = distance / float(objective_maxn - objective_minn)
                logger.debug("front[j]: %s distance: %s", front[j].uid, distance)
                # The overall crowding-distance value is calculated as the sum of
                # individual distance values corresponding to each objective.
                front[j]._meta["crowding_distance"] += distance
        else:
            raise NotImplementedError

    def compute_distance(self, front: tp.List[p.Parameter]):
        """This function assigns the crowding distance to the solutions.
        :param front: The list of solutions.
        """
        size = len(front)

        if size == 0:
            return
        # The boundary solutions (solutions with smallest and largest function values)
        # are set to an infinite (maximum) distance value
        if size == 1:
            front[0]._meta["crowding_distance"] = float("inf")
            return
        if size == 2:
            front[0]._meta["crowding_distance"] = float("inf")
            front[1]._meta["crowding_distance"] = float("inf")
            return

        for f in front:
            f._meta["crowding_distance"] = 0.0

        if isinstance(front[0].losses, np.ndarray) and front[0].losses.shape != ():
            number_of_objectives = len(front[0].losses)
        else:
            number_of_objectives = 1

        for i in range(number_of_objectives):
            self.accumulate_distance_per_objective(front, i)

    def sort(
        self, candidates: tp.List[p.Parameter], in_place: bool = True
    ) -> tp.List[p.Parameter]:
        if in_place:
            candidates.sort(
                key=lambda elem: elem._meta["crowding_distance"], reverse=True
            )  # Larger -> Less crowded
        return sorted(
            candidates, key=lambda elem: elem._meta["crowding_distance"], reverse=True
        )


class FastNonDominatedRanking(object):
    def __init__(
        self,
        compare_fn: Callable = lambda x, y: x
        > y,  # true if better, false if worse. By default is maximization. larger is better
    ):
        self._compare_fn = compare_fn

    """Non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]"""

    def compare(self, candidate1: p.Parameter, candidate2: p.Parameter) -> int:
        """Compare the domainance relation of two candidates.
        :param candidate1: Candidate.
        :param candidate2: Candidate.
        """
        one_wins = np.sum(self._compare_fn(candidate1.losses, candidate2.losses))
        two_wins = np.sum(self._compare_fn(candidate2.losses, candidate1.losses))

        # if one_wins > two_wins:
        #     return -1
        # if two_wins > one_wins:
        #     return 1
        # return 0

        # About domination: if candidate1 dominates candidate2, maximize = true
        # Then candidate1 should be larger than candidate2 in all objectives
        if one_wins == len(candidate1.losses):
            return -1
        if two_wins == len(candidate1.losses):
            return 1
        return 0

    # pylint: disable=too-many-locals
    def compute_ranking(
        self, candidates: tp.List[p.Parameter], k: tp.Optional[int] = None
    ) -> tp.List[tp.List[p.Parameter]]:
        """Compute ranking of candidates.
        :param candidates: List of candidates.
        :param k: Number of individuals.
        """
        n_cand: int = len(candidates)

        # print("Number of candidates: ", n_cand)

        # dominated_by_cnt[i]: number of candidates dominating ith candidate
        dominated_by_cnt: tp.List[int] = [
            0
        ] * n_cand  # [0 for _ in range(len(candidates))]

        # print("dominated_by_cnt: ", dominated_by_cnt)

        # candidates_dominated[i]: List of candidates dominated by ith candidate
        candidates_dominated: tp.List[tp.List[int]] = [[] for _ in range(n_cand)]

        # print("candidates_dominated: ", candidates_dominated)

        # front[i] contains the list of solutions belonging to front i
        front: tp.List[tp.List[int]] = [[] for _ in range(n_cand + 1)]

        uid2candidate = {c.uid: c for c in candidates}

        # print("uid2candidate:", uid2candidate)

        uids = [c.uid for c in candidates]

        # print("uids:", uids)

        for c1 in range(n_cand - 1):
            uid1 = uids[c1]
            for c2 in range(c1 + 1, n_cand):
                uid2 = uids[c2]
                dominance_test_result = self.compare(
                    uid2candidate[uid1], uid2candidate[uid2]
                )
                # self.number_of_comparisons += 1
                if dominance_test_result == -1:
                    # c1 wins
                    candidates_dominated[c1].append(c2)
                    dominated_by_cnt[c2] += 1
                elif dominance_test_result == 1:
                    # c2 wins
                    candidates_dominated[c2].append(c1)
                    dominated_by_cnt[c1] += 1

        # print("dominated_by_cnt after comparison: ", dominated_by_cnt)

        # Reset rank
        for cand in candidates:
            cand._meta["non_dominated_rank"] = float("inf")

        # Formation of front[0], i.e. candidates that do not dominated by others
        front[0] = [c1 for c1 in range(n_cand) if dominated_by_cnt[c1] == 0]

        # print("front[0]:", front[0])
        # print("front:", front)

        last_fronts = 0

        while len(front[last_fronts]) != 0:
            # print("length of front[last_fronts]:", len(front[last_fronts]))
            # print("front[last_fronts]: ",front[last_fronts])
            # print("candidates_dominated: ",candidates_dominated)
            # print("dominated_by_cnt: ",dominated_by_cnt)
            # print("front:", front)

            last_fronts += 1
            # Number of candidates in a frontier <= Number of candidates that dominate at least 1 candidate
            assert len(front[last_fronts - 1]) <= len(candidates_dominated)
            for c1 in front[last_fronts - 1]:
                for c2 in candidates_dominated[c1]:
                    dominated_by_cnt[c2] -= 1
                    if dominated_by_cnt[c2] == 0:
                        front[last_fronts].append(c2)

            # print("After_length of front[last_fronts]:", len(front[last_fronts]))
            # print("After_front[last_fronts]: ",front[last_fronts])
            # print("After_candidates_dominated: ",candidates_dominated)
            # print("After_dominated_by_cnt: ",dominated_by_cnt)
            # print("After_front:", front)

        # Convert index to uid
        # Trim to frontiers that contain the k candidates of interest
        ranked_sublists = []
        count = 0
        for front_i in range(last_fronts):
            count += len(front[front_i])
            for cand_i in front[front_i]:
                uid2candidate[uids[cand_i]]._meta["non_dominated_rank"] = front_i
            ranked_sublists.append([uid2candidate[uids[i]] for i in front[front_i]])
            if (k is not None) and (count >= k):
                break

        return ranked_sublists


def rank(
    population: tp.List[p.Parameter],
    n_selected: tp.Optional[int] = None,
    maximize: bool = True,
) -> tp.Dict[str, tp.Tuple[int, int, float]]:
    """implements the multi-objective ranking function of NSGA-II."""
    frontier_ranker = FastNonDominatedRanking(
        compare_fn=(lambda x, y: x > y) if maximize else (lambda x, y: x < y)
    )
    density_estimator = CrowdingDistance(maximize=maximize)
    selected_pop: tp.Dict[str, tp.Tuple[int, int, float]] = {}
    frontiers = frontier_ranker.compute_ranking(population)
    count = 0
    next_rank = 0
    for front_i, p_frontier in enumerate(frontiers):
        count += len(p_frontier)
        if n_selected is None or count > n_selected:
            density_estimator.compute_distance(p_frontier)
            density_estimator.sort(p_frontier)
            n_dist_calc = (
                n_selected - len(selected_pop)
                if n_selected is not None
                else len(p_frontier)
            )
            for c_i in range(0, n_dist_calc):
                selected_pop[p_frontier[c_i].uid] = (
                    next_rank,
                    front_i,
                    p_frontier[c_i]._meta["crowding_distance"],
                )
                next_rank += 1
            if n_selected is not None:
                break
        if n_selected is not None:
            for candidate in p_frontier:
                selected_pop[candidate.uid] = (next_rank, front_i, float("inf"))
            next_rank += 1
    return selected_pop


class Candidate(object):
    def __init__(self, losses, uid, solution) -> None:
        self.losses = np.array(losses)
        self.uid = str(uid)
        self.solution = solution
        self._meta = {}


class NSGA2(EvolutionarySearchBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Multiobjective Evolutionary search on the PIC topology.
        https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/optimization/multiobjective/nsga2.py
        """
        super().__init__(*args, **kwargs)

    def generate_offsprings(self):
        ## Step 1: (first call this at the beginning of each iter)
        # generate offsprings (double the population size) via crossover and mutation
        offsprings = []
        k = 0
        while k < len(self.population):
            parent1, parent2 = random.sample(self.population, 2)
            crossovered_gene1, crossovered_gene2 = self.crossover(parent1, parent2)
            crossovered_gene1 = self.mutate(crossovered_gene1)
            crossovered_gene2 = self.mutate(crossovered_gene2)
            if self.satisfy_constraints(crossovered_gene1) and self.satisfy_constraints(
                crossovered_gene2
            ):
                offsprings.append(crossovered_gene1)
                offsprings.append(crossovered_gene2)
                k += 2
        self.population = self.population + offsprings

    def tell(self, scores):
        """perform evo search according to the scores"""
        # this scores is a list of tuples for all populations after generating offsprings. Each tuple contains multiple objectives.

        # Step 2: reduce the population size 50% by ranking

        # for i in range(len(self.population)):
        #     print(self.population[i])

        # population = [
        #     Candidate(losses=scores[i], uid=i, solution=self.population[i])
        #     for i in range(len(self.population))
        # ]

        # Didn't check if a generated offspring is the same as one solution in self.population
        population = [
            Candidate(losses=scores[i], uid=i, solution=self.population[i])
            for i in range(len(scores))
        ]

        # print("self.population size: ",len(self.population))
        # print("population size;", len(population))
        # print("Before selection, the populations are:")
        # for i in range(len(population)):
        #     print(population[i].losses)
        #     print(population[i].uid)
        #     print(population[i].solution)

        selected_population = rank(population, maximize=True)  # list of Candidate()

        # print("Selected_population size:", len(selected_population))

        # print("After selection, the populations are:")
        # for key, value in selected_population.items():
        #     print(f"Key:{key}, Value: {value}")

        self.population = [
            population[int(uid)].solution for uid in selected_population.keys()
        ]
        self.population = self.population[
            : self.population_size
        ]  # reduce the population by 50%

        # print("After selection, the self.population size:", len(self.population))

        # print("After selection, the populations are:")
        # for i in range(len(self.population)):
        #     print(self.population[i])

        # Record Pareto Front, including all solutions in the population
        self.pareto_fronts = {}  # this dict stores all solutions (dict key) and all of their objectives(losses)
        for uid, (
            solution_rank,
            front_index,
            crowding_distance,
        ) in selected_population.items():
            solution_objectives = population[int(uid)].losses
            solution = population[int(uid)].solution
            if front_index not in self.pareto_fronts:
                self.pareto_fronts[front_index] = {}
            self.pareto_fronts[front_index][uid] = {
                "objectives": solution_objectives.tolist(),
                "solution": solution,
            }

    def draw(self):  # currently cannot be used on the server
        objective_values = np.array(
            [details["objectives"] for details in self.pareto_front.values()]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x = objective_values[:, 0]
        y = objective_values[:, 1]
        z = objective_values[:, 2]
        ax.scatter(x, y, z, c="r", marker="o")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Compute_Density")
        ax.set_zlabel("Energy Efficiency")
        fig.savefig("figures/pareto_front.png", dpi=300)
