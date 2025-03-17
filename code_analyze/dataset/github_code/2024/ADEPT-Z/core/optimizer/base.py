import copy
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from pyutils.general import ensure_dir, logger

from .utils import Converter, Evaluator, combination_sum

__all__ = ["EvolutionarySearchBase", "evo_args"]


class evo_args:
    def __init__(
        self,
        es_population_size,
        es_parent_size,
        matrix_size,
        es_mutation_size,
        es_mutation_rate_dc,
        es_mutation_rate_cr,
        es_mutation_rate_block,
        es_mutation_ops_dc,
        es_mutation_ops_cr,
        es_crossover_size,
        es_crossover_rate_dc,
        es_crossover_cr_split_ratio,
        es_crossover_rate_block,
        es_max_num_crossings_per_block,
        es_constr,
        es_n_iterations,
        es_n_global_search,
        es_score_mode,
        es_num_procs,
    ):
        self.es_population_size = es_population_size
        self.es_parent_size = es_parent_size
        self.es_mutation_size = es_mutation_size
        self.matrix_size = matrix_size
        self.es_mutation_rate_dc = es_mutation_rate_dc
        self.es_mutation_rate_cr = es_mutation_rate_cr
        self.es_mutation_rate_block = es_mutation_rate_block
        self.es_mutation_ops_dc = es_mutation_ops_dc
        self.es_mutation_ops_cr = es_mutation_ops_cr
        self.es_crossover_size = es_crossover_size
        self.es_crossover_rate_dc = es_crossover_rate_dc
        self.es_crossover_cr_split_ratio = es_crossover_cr_split_ratio
        self.es_crossover_rate_block = es_crossover_rate_block
        self.es_max_num_crossings_per_block = es_max_num_crossings_per_block
        self.es_constr = es_constr
        self.es_n_iterations = es_n_iterations
        self.es_n_global_search = es_n_global_search
        self.es_score_mode = es_score_mode
        self.es_num_procs = es_num_procs


# input: CR array
# output: number of crossings, crossing list, use 2 index to represent one crossing
def get_crossings_list(gene_cr):
    inv_list = []
    for i in range(len(gene_cr)):
        for j in range(i + 1, len(gene_cr)):
            if gene_cr[i] > gene_cr[j]:
                inv_list.append((gene_cr[i], gene_cr[j]))
    return inv_list


# input: CR array
# output: number of crossings that need to be removed
def reduce_crossings(crossing, inversions, max_num):
    while len(inversions) != max_num:
        if not inversions:
            break
        selected_pair = random.choice(inversions)

        first_index = np.where(crossing == selected_pair[0])[0][0]
        second_index = np.where(crossing == selected_pair[1])[0][0]

        crossing[first_index], crossing[second_index] = (
            crossing[second_index],
            crossing[first_index],
        )

        inversions = get_crossings_list(crossing)

        if len(inversions) == max_num:
            break

        if len(inversions) < max_num:
            np.random.shuffle(crossing)
            inversions = get_crossings_list(crossing)
            continue

    return crossing


class EvolutionarySearchBase(object):
    def __init__(
        self,
        population_size: int = 100,
        parent_size: int = 80,
        matrix_size: int = 8,
        mutation_size: int = 10,
        mutation_rate_dc: float = 0.05,
        mutation_rate_cr: float = 0.05,
        mutation_rate_block: float = 0.05,
        mutation_ops_dc: List[str] = ["op1_dc", "op2_dc"],
        mutation_ops_cr: List[str] = ["op1_cr", "op2_cr"],
        crossover_size: int = 10,
        crossover_rate_dc: int = 0.5,
        crossover_cr_split_ratio: float = 0.5,
        crossover_rate_block: int = 0.5,
        max_num_crossings_per_block: int = 6,
        super_layer: Dict[str, Any] = None,  #
        constraints: Dict[str, List] = {"area": [0, 100]},  # {"area": [1, 5], ...}
        evaluator: Evaluator = None,
        mutation_rate_scheduler: Dict[str, Any] = None,
    ):
        """Evolution search on the PIC topology. The gene is defined as
        [
            3, # number of active blocks
            [
                np.array([1,1,2,2,2]), # coupler representation as the port number
                np.array([0,3,2,1,4,5,7,6]) # crossing representation as the indices
            ], # block 0
            [], # block 1
            ...
            [], # block max_n_blocks - 1
        ]
         these blocks are for both U and V. U by default uses the first half. The rest is for V


        Args:
            population_size (int, optional): _description_. Defaults to 100.
            parent_size (int, optional): _description_. Defaults to 80.
            matrix_size (int, optional): _description_. Defaults to 4.
            mutation_rate_dc (float, optional): _description_. Defaults to 0.05.
            mutation_rate_cr (float, optional): _description_. Defaults to 0.05.
            mutation_rate_block (float, optional): _description_. Defaults to 0.05.
            crossover_cr_split_ratio (float, optional): _description_. Defaults to 0.5.
            super_layer (_type_, optional): _description_. Defaults to None.
        """
        self.population_size = population_size  # total population size
        self.parent_size = parent_size  # selected parent size
        self.max_n_blocks = int(super_layer.arch["n_blocks"])
        self.matrix_size = int(super_layer.arch["n_waveguides"])
        self.mutation_ops_dc = mutation_ops_dc
        self.mutation_ops_cr = mutation_ops_cr
        self.mutation_size = mutation_size
        self.mutation_rate_dc = mutation_rate_dc  # probability of mutation for DC array
        self.mutation_rate_cr = mutation_rate_cr  # probability of mutation for CR array
        self.mutation_rate_block = (
            mutation_rate_block  # probability of mutation for blocks
        )

        self.crossover_size = crossover_size
        self.crossover_dc_rate = crossover_rate_dc
        self.crossover_cr_split_ratio = crossover_cr_split_ratio
        self.crossover_block_rate = crossover_rate_block

        self.population = []
        self.super_layer = super_layer
        self.constraints = constraints
        self.evaluator = evaluator
        self.converter = Converter(super_layer)
        self.dc_port_candidates = super_layer.arch[
            "dc_port_candidates"
        ]  # list of DC port candidates, e.g., [2,4,6]

        self.dc_solution_space = combination_sum(
            [1] + self.dc_port_candidates, target=self.matrix_size
        )

        # set the maximum number of crossings as 6 based on butterfly design(half number of crossings in the 2nd block)
        self.max_num_crossings_per_block = max_num_crossings_per_block
        # print("max_num_crossings_per_block:", self.max_num_crossings_per_block)
        self.add_initial_population()

        self.current_generation = 1

        self.mutation_rate_scheduler = mutation_rate_scheduler

    def set_parameters(self, params):
        self.mutation_rate_dc = params[
            "mutation_rate_dc"
        ]  # probability of mutation for DC array
        self.mutation_rate_cr = params[
            "mutation_rate_cr"
        ]  # probability of mutation for CR array
        self.mutation_rate_block = params[
            "mutation_rate_block"
        ]  # probability of mutation for blocks
        self.crossover_dc_rate = params["crossover_rate_dc"]
        self.crossover_cr_split_ratio = params["cr_split_ratio"]
        self.crossover_block_rate = params["crossover_rate_block"]

    def set_operators(self, mutation_ops_dc, mutation_ops_cr):
        self.mutation_ops_dc = mutation_ops_dc
        self.mutation_ops_cr = mutation_ops_cr

    def add_initial_population(self):
        # add some good solution to initial population
        gene_b = generate_Butterfly_gene(k=self.matrix_size, n_blocks=self.max_n_blocks)
        # solution_b = self.converter.gene2solution(gene_b)
        self.population.append(gene_b)

        gene_m = generate_MZI_gene(k=self.matrix_size)
        # solution_b = self.converter.gene2solution(gene_b)
        self.population.append(gene_m)
        # print(f"Initial gene in population: {gene_b}")

        # gene_k = generate_kxk_MMI_gene(k=self.matrix_size,n_blocks=self.max_n_blocks)
        # self.population.append(gene_k)

    # function to initialize a population (updated)
    def _dummy_block(self):
        return [
            np.ones(self.matrix_size, dtype=int),
            np.arange(self.matrix_size, dtype=int),
        ]

    def initialize_population(self):
        # Function to generate a string representing one DC array
        def generate_dc():
            arr = self.dc_solution_space[
                np.random.randint(0, len(self.dc_solution_space))
            ]
            np.random.shuffle(arr)
            return np.array(arr)

        # Function to generate a string representing one CR array
        def generate_cr():
            numbers = np.random.permutation(range(self.matrix_size))
            # numbers = np.arange(self.matrix_size, dtype=int)
            return numbers

        # Function to generate one individual
        def generate_individual():
            B = np.random.randint(1, self.max_n_blocks + 1)
            gene = [B]
            for _ in range(B):
                gene.append([generate_dc(), generate_cr()])
            for _ in range(B, self.max_n_blocks):
                # identity blocks as dummy blocks
                gene.append(self._dummy_block())

            # here check the number of crossings for each CR block
            # if the number of crossings in one CR block is larger than maximum
            # then reduce the number of crossings to the maximum number
            for _, CR_array in gene[1:]:
                crossing_list = get_crossings_list(CR_array)
                num_CR = len(crossing_list)
                if num_CR > self.max_num_crossings_per_block:
                    CR_array = reduce_crossings(
                        CR_array, crossing_list, self.max_num_crossings_per_block
                    )
                # print("num_CR in initialization:", len(get_crossings_list(CR_array)))
            return gene

        # initialize the population with randomly generated individuals
        # for _ in range(self.population_size):
        while len(self.population) < self.population_size:
            gene = generate_individual()
            while not self.satisfy_constraints(gene):
                gene = generate_individual()
            self.population.append(gene)

        return self.population

    # sort the population based on their scores, select parents to do mutation and crossover
    def select_parents(self, fitness_scores):
        total_score = sum(fitness_scores)
        probabilities = [
            score / total_score for score in fitness_scores
        ]  # individual with higher score has higher probability to be selected
        selected_parents = random.choices(
            population=self.population, weights=probabilities, k=self.parent_size
        )
        return selected_parents

    # crossover operator (tested in test_es.py)
    def crossover(self, parent1, parent2):
        def crossover_dc(parent1_, parent2_, dc_rate):
            parent1_dc = copy.deepcopy(parent1_)
            parent2_dc = copy.deepcopy(parent2_)

            cum_sum1 = np.cumsum(parent1_dc)[:-1]
            cum_sum2 = np.cumsum(parent2_dc)[:-1]

            cutting_points = np.intersect1d(cum_sum1, cum_sum2)

            cutting_indices1 = np.add(np.searchsorted(cum_sum1, cutting_points), 1)
            cutting_indices2 = np.add(np.searchsorted(cum_sum2, cutting_points), 1)
            segments1 = np.split(parent1_dc, cutting_indices1)
            segments2 = np.split(parent2_dc, cutting_indices2)

            list1 = []
            list2 = []
            for i in range(len(segments1)):
                if np.random.uniform() < dc_rate:
                    list1.append(segments2[i])
                    list2.append(segments1[i])
                else:
                    list1.append(segments1[i])
                    list2.append(segments2[i])
            crossover_dc1 = np.concatenate(list1)
            crossover_dc2 = np.concatenate(list2)

            return crossover_dc1, crossover_dc2

        def crossover_cr(
            parent1_cr: np.ndarray, parent2_cr: np.ndarray, split_ratio: float = 0.5
        ):
            """Crossover CR gene block. Select a subset of indices from parent 1, \
                and the rest indices will be copied from parent 2 with the original ordering

            Args:
                parent1_cr (np.ndarray): 1D array of CR indices from parent 1
                parent2_cr (np.ndarray): 1D array of CR indices from parent 2
                split_ratio (float): ratio from parent 1. will be rounded to integer

            Returns:
                new_parent1_cr (np.ndarray): 1D array of CR indices for parent 1
                new_parent2_cr (np.ndarray): 1D array of CR indices for parent 2
            """
            assert 0 <= split_ratio <= 1, logger.error(
                f"CR gene crossover split ratio must in [0, 1], but got {split_ratio}"
            )
            length = len(parent1_cr)
            split = round(len(parent1_cr) * split_ratio)
            indices = np.random.choice(length, size=split, replace=False)
            values1 = parent1_cr[indices]
            values2 = parent2_cr[indices]
            values1_set = set(values1.tolist())
            values2_set = set(values2.tolist())

            rest_indices = np.setdiff1d(np.arange(length), indices)

            rest_values1 = np.array(
                list(filter(lambda i: i not in values1_set, parent2_cr))
            )
            rest_values2 = np.array(
                list(filter(lambda i: i not in values2_set, parent1_cr))
            )

            new_parent1_cr = copy.deepcopy(parent1_cr)
            new_parent1_cr[rest_indices] = rest_values1

            new_parent2_cr = copy.deepcopy(parent2_cr)
            new_parent2_cr[rest_indices] = rest_values2

            return new_parent1_cr, new_parent2_cr

        def crossover_block(parent1_block, parent2_block, block_rate):
            parent1 = copy.deepcopy(parent1_block)
            parent2 = copy.deepcopy(parent2_block)
            if parent1[0] > parent2[0]:
                parent1, parent2 = (
                    parent2,
                    parent1,
                )  # make sure the number of valid blocks in parent1 is smaller or equal to parent2
            crossovered_father = [parent1[0]]
            crossovered_mother = [parent2[0]]  # number of valid blocks
            for k in range(1, parent1[0] + 1):  # both blocks are valid blocks
                if np.random.uniform() < block_rate:  # probability 0.5 to crossover
                    crossovered_father.append(parent2[k])
                    crossovered_mother.append(parent1[k])
                else:
                    crossovered_father.append(parent1[k])
                    crossovered_mother.append(parent2[k])
            if (
                parent1[0] < parent2[0]
            ):  # check if dummy blocks need to crossover with valid blocks
                for m in range(parent1[0] + 1, parent2[0] + 1):
                    if np.random.uniform() < block_rate:
                        crossovered_father.append(parent2[m])
                        crossovered_father[0] = crossovered_father[0] + 1
                        crossovered_mother[0] = (
                            crossovered_mother[0] - 1
                        )  # need to change the number of valid blocks
                    else:
                        crossovered_mother.append(parent2[m])

            while len(crossovered_father) < len(parent1):
                crossovered_father.append(self._dummy_block())
            while len(crossovered_mother) < len(parent2):
                crossovered_mother.append(
                    self._dummy_block()
                )  # Make length of children equal to parents, append dummy blocks
            return crossovered_father, crossovered_mother

        crossovered_parent1 = copy.deepcopy(parent1)
        crossovered_parent2 = copy.deepcopy(parent2)

        for i in range(1, self.max_n_blocks + 1):
            crossovered_parent1[i][0], crossovered_parent2[i][0] = crossover_dc(
                crossovered_parent1[i][0],
                crossovered_parent2[i][0],
                self.crossover_dc_rate,
            )
            # stop using crossover opereator for CR array, only use mutation
            crossovered_parent1[i][1], crossovered_parent2[i][1] = crossover_cr(
                crossovered_parent1[i][1],
                crossovered_parent2[i][1],
                split_ratio=self.crossover_cr_split_ratio,
            )
        crossovered_parent1, crossovered_parent2 = crossover_block(
            crossovered_parent1, crossovered_parent2, self.crossover_block_rate
        )  # do DC and CR crossover first, then do block crossover
        return crossovered_parent1, crossovered_parent2

    # mutation Operator ##10/19: change mutation operators, should support arbitrary DC ports
    def mutate(self, individual):
        # mutation on 1 DC array
        def mutation_dc(gene):
            gene_dc = copy.deepcopy(
                gene
            )  # keep a copy of the gene_dc and perform mutation

            def max_consecutive_ones(lst):
                # This function is used to find the max port number of DC that can be added.
                # eg. [1,2,3,1,1,1,1,4,1], max_count = 4.
                count = 0
                max_count = 0
                for num in lst:
                    if num == 1:
                        count += 1
                        max_count = max(max_count, count)
                    else:
                        count = 0
                return max_count

            def r2a1(gene_dc):
                # remove 2 DC and add one DC
                gene_dc = np.array(gene_dc)
                # print("gene_dc1:", gene_dc)
                for _ in range(2):
                    indices_of_n = np.where(gene_dc != 1)[0]
                    selected_index = np.random.choice(indices_of_n)
                    gene_dc = np.hstack(
                        (
                            gene_dc[:selected_index],
                            np.full(gene_dc[selected_index], 1),
                            gene_dc[selected_index + 1 :],
                        )
                    )  # extend to n-port devices

                max_port_num = max_consecutive_ones(
                    gene_dc
                )  # Get the max port number of DC that can be added
                # print("max_port_num:", max_port_num)
                DC_list = np.array(
                    [i for i in self.dc_port_candidates if i <= max_port_num]
                )  # Get a subset of dc_port candidates, they can be added

                if len(DC_list) > 0:
                    # print("gene_dc2:", gene_dc)
                    # print("DC_list:", DC_list)
                    add_dc = np.random.choice(
                        DC_list
                    )  # choose the number of port for the DC to be added
                    indices_of_pairs = np.array(
                        [
                            i
                            for i in range(len(gene_dc) - add_dc + 1)
                            if np.all(gene_dc[i : i + add_dc] == 1)
                        ]
                    )
                    selected_index = np.random.choice(np.array(indices_of_pairs))
                    gene_dc = np.hstack(
                        (
                            gene_dc[:selected_index],
                            [add_dc],
                            gene_dc[selected_index + add_dc :],
                        )
                    )  # add the DC
                    # print("Type in r2a1:", type(gene_dc))
                    return gene_dc
                else:
                    return gene_dc

            def a2r1(gene_dc):
                # Add 2 DC and remove 1 DC
                gene_dc = np.array(gene_dc)
                for _ in range(2):
                    max_port_num = max_consecutive_ones(
                        gene_dc
                    )  # Get the max port number of DC that can be added
                    DC_list = np.array(
                        [i for i in self.dc_port_candidates if i <= max_port_num]
                    )  # Get a subset of dc_port candidates, they can be added
                    if (
                        DC_list.size == 0
                    ):  # If the second DC cannot be added, then break the loop, change to add 1 DC and remove 1 DC
                        break
                    else:
                        add_dc = np.random.choice(
                            DC_list
                        )  # choose the number of port for the DC to be added
                        indices_of_pairs = np.array(
                            [
                                i
                                for i in range(len(gene_dc) - add_dc + 1)
                                if np.all(gene_dc[i : i + add_dc] == 1)
                            ]
                        )
                        selected_index = np.random.choice(np.array(indices_of_pairs))
                        gene_dc = np.hstack(
                            (
                                gene_dc[:selected_index],
                                [add_dc],
                                gene_dc[selected_index + add_dc :],
                            )
                        )  # add the DC
                indices_of_n = np.where(gene_dc != 1)[0]
                selected_index = np.random.choice(indices_of_n)
                gene_dc = np.hstack(
                    (
                        gene_dc[:selected_index],
                        np.full(gene_dc[selected_index], 1),
                        gene_dc[selected_index + 1 :],
                    )
                )  # extend to n-port devices
                # print("Type in a2r1:", type(gene_dc))
                return gene_dc

            def swap(gene_dc):  # choose one DC device and change its position
                gene_dc = np.array(gene_dc)

                indices_of_n = np.where(gene_dc > 1)[
                    0
                ]  # get the indices of all DC devices
                selected_index = np.random.choice(indices_of_n)  # choose one DC device
                port = gene_dc[selected_index]  # get the port number of the DC device
                mask = gene_dc != port
                if not np.any(mask):
                    ## all couplers have same port counts. do not swap
                    return gene_dc

                selected_index2 = np.random.choice(np.where(mask)[0])
                gene_dc[selected_index], gene_dc[selected_index2] = (
                    gene_dc[selected_index2],
                    gene_dc[selected_index],
                )
                # ## only swap with different port numbers, e.g., 1 vs 2, or 2 vs 3. We do not swap 2 vs 2
                # print("Type in swap:", type(gene_dc))
                return gene_dc

            def random_sample(gene_dc):
                gene_dc = np.array(
                    self.dc_solution_space[
                        np.random.randint(0, len(self.dc_solution_space))
                    ]
                )
                np.random.shuffle(gene_dc)
                # print("Type in random_sample:", type(gene_dc))
                return gene_dc

            def check_legality(
                gene_dc, operator
            ):  # check if the mutation operator can be applied to the gene
                def check_op1_legality(gene_dc):  # check legality for r2a1
                    indices_of_n = np.where(gene_dc != 1)[0]
                    return (
                        len(indices_of_n) >= 2
                    )  # Legal when there are more than two DC devices in the gene_dc

                def check_op2_legality(gene_dc):  # check legality for a2r1
                    max_port_num = max_consecutive_ones(
                        gene_dc
                    )  # Get the max port number of DC that can be added
                    min_candidate = np.array(gene_dc).min()
                    return (
                        min_candidate <= max_port_num
                    )  # To make sure at least one type of DC can be added

                def check_op3_legality(gene_dc):  # check legality for swap
                    indices_of_n = np.where(gene_dc != 1)[
                        0
                    ]  # to make sure at least one DC device exists
                    indices_of_1 = np.where(
                        gene_dc == 1
                    )[
                        0
                    ]  # to make sure at least one available position to swap, eg:[8], no position to swap
                    return (len(indices_of_n) >= 1) and (len(indices_of_1) >= 1)

                def check_op4_legality(
                    gene_dc,
                ):  # no need to check legality, always add it as an option
                    return True

                if operator == "op1_dc":
                    return check_op1_legality(gene_dc)
                elif operator == "op2_dc":
                    return check_op2_legality(gene_dc)
                elif operator == "op3_dc":
                    return check_op3_legality(gene_dc)
                elif operator == "op4_dc":
                    return check_op4_legality(
                        gene_dc
                    )  # check legality based on the name of the operator

            # randomly choose one operator
            mutation_operator_set = {
                "op1_dc": r2a1,
                "op2_dc": a2r1,
                "op3_dc": swap,
                "op4_dc": random_sample,
            }
            names = [
                name for name in self.mutation_ops_dc
            ]  # get the name list of operators: ["op1", "op2", ...]
            while names:
                selected_name = np.random.choice(names)  # select one operator
                is_valid = check_legality(
                    gene_dc, selected_name
                )  # check legality of the selected operator
                if is_valid:
                    selected_operator = mutation_operator_set[selected_name]
                    mutated_dc = selected_operator(gene_dc)
                    return mutated_dc  # if legal, then use the selected operator to do the mutation
                else:
                    names.remove(
                        selected_name
                    )  # if not legal, then remove the selected operator from the name list
            return gene_dc  # If all ops are removed from name list, do not mutate, return the original gene

        # mutation on 1 CR array(updated)
        def mutation_cr(gene):
            gene_cr = copy.deepcopy(gene)

            # reduce a certain number of crossings
            def reduce_cr(gene_cr):
                # print("reduce_cr chosen")
                crossings_list = get_crossings_list(gene_cr)
                count = len(crossings_list)
                if count > 1:
                    # print("Number of cr before mutation:", count)
                    num_cr = np.random.randint(0, count)
                    # print("Reduce to number of cr:", num_cr)
                    new_gene_cr = reduce_crossings(gene_cr, crossings_list, num_cr)
                    # print("Number of cr after mutation_reduce:", len(get_crossings_list(new_gene_cr)))
                else:
                    new_gene_cr = gene_cr
                return new_gene_cr

            # add some crossings
            def add_cr(gene_cr):
                # print("add_cr chosen")
                # shuffle, and reduce number of crossings
                crossings_list = get_crossings_list(gene_cr)
                num_CR = len(crossings_list)
                # print("num_CR in mutation:", num_CR)
                while len(crossings_list) < self.max_num_crossings_per_block:
                    np.random.shuffle(gene_cr)
                    crossings_list = get_crossings_list(gene_cr)
                if num_CR >= self.max_num_crossings_per_block:
                    num_cr = self.max_num_crossings_per_block
                else:
                    num_cr = np.random.randint(
                        num_CR + 1, self.max_num_crossings_per_block + 1
                    )
                    # print("Add to number of cr:", num_cr)
                new_gene_cr = reduce_crossings(gene_cr, crossings_list, num_cr)
                # print("Number of cr after mutation_add:", len(get_crossings_list(new_gene_cr)))
                return new_gene_cr

            # randomly choose one operator
            mutation_operator_set = {
                "op1_cr": reduce_cr,
                "op2_cr": add_cr,
                # "op3_cr": random_sample,
            }

            mutation_operator_names = [
                name for name in self.mutation_ops_cr
            ]  # get the name list of operators: ["op1_cr", "op2_cr", ...]
            # mutation_operators = [reduce_cr, add_cr, to_identity]
            selected_name = np.random.choice(mutation_operator_names)
            chosen_mutation_operator = mutation_operator_set[selected_name]
            # print("number of cr before mutation:", len(get_crossings_list(gene_cr)))

            if len(get_crossings_list(gene_cr)) == 0:
                mutated_cr = add_cr(gene_cr)
            elif len(get_crossings_list(gene_cr)) >= self.max_num_crossings_per_block:
                mutated_cr = reduce_cr(gene_cr)
            else:
                mutated_cr = chosen_mutation_operator(gene_cr)
            # print("number of cr after mutation:", len(get_crossings_list(mutated_cr)))

            # mutated_cr = random_sample(gene_cr)
            return mutated_cr

        # mutation on the number of blocks of one individual(updated on Aug 17th)
        def mutation_block(gene_block):
            gene = copy.deepcopy(gene_block)

            def extend_gene(gene):
                # print(f"extend_gene chosen for {gene[0]}")
                start_index = 1
                end_index = np.random.randint(1, gene[0] + 1)
                b = gene[0]
                gene[0] = gene[0] + end_index
                l = len(gene)
                dup_gene = gene[start_index : (end_index + 1)]
                gene = gene[: (b + 1)] + dup_gene
                while len(gene) < l:
                    gene.append(self._dummy_block())
                return gene

            def reduce_gene(gene):
                # print(f"reduce_gene chosen for {gene[0]}")
                start_index = 1
                end_index = np.random.randint(gene[0] // 2, gene[0])
                gene[0] = end_index - start_index + 1
                l = len(gene)
                gene = gene[: (end_index + 1)]
                while len(gene) < l:
                    gene.append(self._dummy_block())
                return gene

            # define a naive operator
            def replace_block(gene):
                if gene[0] < 2:
                    return gene
                numbers = np.random.choice(
                    np.arange(1, gene[0] + 1), size=2, replace=False
                )
                num1 = numbers[0]
                num2 = numbers[1]
                gene[num1] = gene[num2]
                return gene

            # randomly choose one operator
            if gene[0] < ((len(gene) - 1) / 2):
                mutation_operator_set = {
                    "op1_block": reduce_gene,
                    "op2_block": extend_gene,
                    "op3_block": replace_block,
                }
            else:
                mutation_operator_set = {
                    "op1_block": reduce_gene,
                    "op3_block": replace_block,
                }

            # # comparison
            # mutation_operator_set={
            #     "op3_block": replace_block,
            # }

            mutation_operator_names = [
                name for name in mutation_operator_set
            ]  # get the name list of operators: ["op1_cr", "op2_cr", ...]
            selected_name = np.random.choice(mutation_operator_names)
            chosen_mutation_operator = mutation_operator_set[selected_name]
            # print(f"Gene before mutation_block:{gene}")
            mutated_gene = chosen_mutation_operator(gene)
            # print(f"Mutated gene:{mutated_gene}")
            return mutated_gene

        # mutated_individual = copy.deepcopy(individual)
        mutated_individual = [individual[0]]
        for i in range(1, self.max_n_blocks + 1):
            small_block = []
            if i <= individual[0]:
                if np.random.uniform() < self.mutation_rate_dc:
                    small_block_dc = mutation_dc(individual[i][0])
                    small_block.append(small_block_dc)
                else:
                    small_block.append(individual[i][0])

                if np.random.uniform() < self.mutation_rate_cr:
                    small_block_cr = mutation_cr(individual[i][1])
                    small_block.append(small_block_cr)
                else:
                    small_block.append(individual[i][1])

                mutated_individual.append(small_block)
            else:
                mutated_individual.append(individual[i])

        if np.random.uniform() < self.mutation_rate_block:
            mutated_individual = mutation_block(mutated_individual)
        else:
            mutated_individual = mutated_individual
        return mutated_individual

    def _satisfy_constraints_area(self, arch_sol: tuple, bound: tuple = (0, 1)) -> bool:
        cost = self.evaluator.evaluate_cost(arch_sol, keys=["area"])
        flag = True
        if bound[1] is not None:
            flag &= cost["area"] <= bound[1]
        if bound[0] is not None:
            flag &= bound[0] <= cost["area"]
        if not flag:
            area = cost["area"]
            print(f"Area: {area} not in {bound}")
        return flag

    def _satisfy_constraints_latency(
        self, arch_sol: tuple, bound: tuple = (0, 1)
    ) -> bool:
        cost = self.evaluator.evaluate_cost(arch_sol, keys=["latency"])
        flag = True
        if bound[1] is not None:
            flag &= cost["latency"] <= bound[1]
        if bound[0] is not None:
            flag &= bound[0] <= cost["latency"]
        if not flag:
            latency = cost["latency"]
            print(f"Latency: {latency} not in {bound}")
        return flag

    def _satisfy_constraints_power(
        self, arch_sol: tuple, bound: tuple = (0, 1)
    ) -> bool:
        cost = self.evaluator.evaluate_cost(arch_sol, keys=["power"])
        flag = True
        if bound[1] is not None:
            flag &= cost["power"] <= bound[1]
        if bound[0] is not None:
            flag &= bound[0] <= cost["power"]
        if not flag:
            power = cost["power"]
            print(f"Power: {power} not in {bound}")
        return flag

    def _satisfy_constraints_robustness(
        self, arch_sol: tuple, bound: tuple = (0, 1)
    ) -> bool:
        cost = self.evaluator.evaluate_cost(arch_sol, keys=["robustness"])
        flag = True
        if bound[1] is not None:
            flag &= cost["robustness"] <= bound[1]
        if bound[0] is not None:
            flag &= bound[0] <= cost["robustness"]
        if not flag:
            robustness = cost["robustness"]
            print(f"Robustness: {robustness} not in {bound}")
        return flag

    def satisfy_constraints(self, gene):
        ## check the constraints
        if len(self.constraints) == 0:
            return True
        arch_sol = self.converter.gene2solution(gene, to_string=False)

        flag = True
        for metric, bound in self.constraints.items():
            if metric == "area":
                flag &= self._satisfy_constraints_area(arch_sol, bound)
            elif metric == "latency":
                flag &= self._satisfy_constraints_latency(arch_sol, bound)
            elif metric == "power":
                flag &= self._satisfy_constraints_power(arch_sol, bound)
            # elif metric == "robustness":
            #     flag &= self._satisfy_constraints_robustness(arch_sol, bound)
        return flag

    def ask(self, to_solution: bool = True):
        """return the solutions"""
        if to_solution:
            return [self.converter.gene2solution(gene) for gene in self.population]
        else:
            return self.population

    def tell(self, scores):
        raise NotImplementedError

    def dump_solution_to_file(self, best_sol: dict, filename=None) -> None:
        ensure_dir(os.path.dirname(filename))

        max_solutions_per_file = 10

        best_sol_items = list(best_sol.items())

        subdicts = [
            best_sol_items[i : i + max_solutions_per_file]
            for i in range(0, len(best_sol_items), max_solutions_per_file)
        ]

        for file_index, sublist in enumerate(subdicts):
            file = f"{filename}_{file_index + 1}.yml"
            with open(file, "w") as f:
                for i, (solution_name, (scores, cost_dict)) in enumerate(sublist):
                    yaml.safe_dump(
                        {
                            "Solution Number": i,
                            "solution": solution_name,
                            "score": scores,
                            "cost": cost_dict,
                        },
                        f,
                    )
                    f.write("---\n")
            logger.info(f"Dumped best solution to {file}")

    def load_solution_from_file(self, file_path):
        best_solutions = []
        best_scores = []
        with open(file_path, "r") as file:
            all_docs = yaml.safe_load_all(file)
            for doc in all_docs:
                if doc is not None and "solution" in doc.keys():
                    best_solutions.append(doc["solution"])
                if doc is not None and "score" in doc.keys():
                    best_scores.append(doc["score"])
        return best_solutions, best_scores

    def save_checkpoint(self, checkpoint_filepath):
        # print("checkpoint filepath:",checkpoint_filepath)
        # print("In Save_checkpoint, mutation_rate_dc is:", self.mutation_rate_dc)
        checkpoint = {
            "population": self.population,
            "mutation_rate_dc": self.mutation_rate_dc,
            "mutation_rate_cr": self.mutation_rate_cr,
            "mutation_rate_block": self.mutation_rate_block,
            "mutation_ops_dc": self.mutation_ops_dc,
            "mutation_ops_cr": self.mutation_ops_cr,
            "crossover_rate_dc": self.crossover_dc_rate,
            "crossover_cr_split_ratio": self.crossover_cr_split_ratio,
            "crossover_block_rate": self.crossover_block_rate,
            "current_generation": self.current_generation,
            "mutation_rate_scheduler_state_dict": {
                key: scheduler.state_dict()
                for key, scheduler in self.mutation_rate_scheduler.items()
            },
        }
        torch.save(checkpoint, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath)
        self.population = checkpoint["population"]
        self.mutation_rate_dc = checkpoint["mutation_rate_dc"]
        self.mutation_rate_cr = checkpoint["mutation_rate_cr"]
        self.mutation_rate_block = checkpoint["mutation_rate_block"]
        self.mutation_ops_dc = checkpoint["mutation_ops_dc"]
        self.mutation_ops_cr = checkpoint["mutation_ops_cr"]
        self.crossover_dc_rate = checkpoint["crossover_rate_dc"]
        self.crossover_cr_split_ratio = checkpoint["crossover_cr_split_ratio"]
        self.crossover_block_rate = checkpoint["crossover_block_rate"]
        self.current_generation = checkpoint["current_generation"]
        for key, scheduler in self.mutation_rate_scheduler.items():
            if key in checkpoint["mutation_rate_scheduler_state_dict"]:
                scheduler.load_state_dict(
                    checkpoint["mutation_rate_scheduler_state_dict"][key]
                )


def generate_MZI_gene(k):
    if k >= 2 and k % 2 == 0:  # k is even number
        DC_array1 = np.full(shape=k // 2, fill_value=2)
        DC_array2 = np.full(shape=(k // 2) + 1, fill_value=2)
        DC_array2[0], DC_array2[-1] = 1, 1

        CR_array = np.arange(k)

        Block1 = [DC_array1, CR_array]
        Block2 = [DC_array2, CR_array]

        num_block = 4 * k
        gene = []
        gene.append(num_block)
        for _ in range(k):
            gene.append(Block1)
            gene.append(Block1)
            gene.append(Block2)
            gene.append(Block2)
    elif k >= 2 and k % 2 == 1:  # k is odd number
        DC_array1 = np.full(shape=(k + 1) // 2, fill_value=2)
        DC_array1[0] = 1
        DC_array2 = np.full(shape=(k + 1) // 2, fill_value=2)
        DC_array2[-1] = 1

        CR_array = np.arange(k)

        Block1 = [DC_array1, CR_array]
        Block2 = [DC_array2, CR_array]

        num_block = 4 * k
        gene = []
        gene.append(num_block)
        for _ in range(k):
            gene.append(Block1)
            gene.append(Block1)
            gene.append(Block2)
            gene.append(Block2)
    else:
        raise ValueError("input k must be larger than 1")

    return gene


def generate_Butterfly_gene(k, n_blocks):
    def sort_indices_by_values(lst):
        sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1])
        sorted_indices = np.array([index for index, value in sorted_pairs])
        return sorted_indices

    if k % 4 == 0 or k == 2:
        if k == 2:
            gene = [2, np.array([2]), np.array([0, 1])]
        else:
            gene = []
            # the first element: number of active blocks
            num_active_block = int(np.log2(k) * 2)
            gene.append(num_active_block)

            # generate DC array
            DC_array = np.full(shape=k // 2, fill_value=2)

            original_arr = np.arange(k)
            CR_arrays = [original_arr.copy()]
            step = 4
            while step <= k:
                new_arr = np.array([], dtype=int)
                for i in range(0, k, step):
                    group = original_arr[i : i + step]
                    evens = group[group % 2 == 0]
                    odds = group[group % 2 != 0]
                    new_arr = np.concatenate([new_arr, np.concatenate([evens, odds])])
                original_arr = new_arr
                CR_arrays.append(original_arr.copy())
                step *= 2
            CR_arrays.append(CR_arrays.pop(0))

            for _ in range(2):
                for i in range(num_active_block // 2):
                    gene.append([DC_array, CR_arrays[i]])

            while len(gene) < n_blocks + 1:
                dummy_DC_array = np.full(shape=k, fill_value=1)
                dummy_CR_array = np.arange(k)
                gene.append([dummy_DC_array, dummy_CR_array])

        gene = (
            gene[0:1]
            + gene[1 : (num_active_block // 2) + 1]
            + gene[(num_active_block // 2) + 1 : num_active_block][::-1]
            + gene[num_active_block:]
        )

        for i in range(1, ((num_active_block // 2) + 1)):
            # print(gene[i])
            gene[i][1] = sort_indices_by_values(gene[i][1])
            # print("After:", gene[i])

        return gene

    else:
        raise ValueError("Cannot generate genotype with the given k.")


def generate_kxk_MMI_gene(k, n_blocks):
    if k >= 2:
        DC_array1 = np.full(shape=1, fill_value=k)  # [k]

        CR_array = np.arange(k)  # [0,1,2, ... ,k-1]

        Block1 = [DC_array1, CR_array]  # [[k], [0,1,2, ... ,k-1]]

        num_block = 2 * k + 2
        gene = []
        gene.append(num_block)
        for _ in range(num_block):
            gene.append(Block1)
    else:
        raise ValueError("input k must be larger than 2")

    while len(gene) < n_blocks + 1:
        dummy_DC_array = np.full(shape=k, fill_value=1)
        dummy_CR_array = np.arange(k)
        gene.append([dummy_DC_array, dummy_CR_array])
    return gene


def generate_SCF_gene(k, n_blocks):
    def swap_sublists(lst, k):
        # if len(lst) % (2 * k) != 0:
        #     raise ValueError("The length of the list must be a multiple of 2k")
        # print(lst)
        # print(k)
        result = []
        for i in range(0, len(lst), 2 * k):
            sublist1 = lst[i : i + k]
            # print("sublist1:",sublist1)
            sublist2 = lst[i + k : i + 2 * k]
            # print("sublist2:", sublist2)
            result.extend(sublist2)
            result.extend(sublist1)
        # print(result)
        return np.array(result)

    if k % 4 == 0 or k == 2:
        if k == 2:
            raise ValueError("Cannot generate genotype with the given k.")
        else:
            gene = []
            # the first element: number of active blocks
            # in SCF mesh design, the depth scales as O(N)
            num_active_block = k
            gene.append(num_active_block)

            # generate DC array
            DC_array = np.full(shape=k // 2, fill_value=2)

            original_arr = np.arange(k)
            CR_arrays = [original_arr.copy()]
            # print("CR arrays:",CR_arrays)
            step = 2
            while step <= (k // 2):
                new_arr = swap_sublists(original_arr, step)
                CR_arrays.append(new_arr)
                step *= 2
            # CR_arrays.append(CR_arrays.pop(0))
            # print(CR_arrays)
            # print("lenth of CR arrays:", len(CR_arrays))

            idx = list(range(len(CR_arrays)))
            # print("idx:",idx)

            def build_list(initial_list):
                result = [initial_list[-1]]

                for i in range(len(initial_list) - 2, -1, -1):
                    current = initial_list[i]
                    new_result = [current]
                    for item in result:
                        new_result.extend([item, current])
                    result = new_result
                return result

            idx_result = build_list(idx)
            idx_result.append(idx[0])
            # print("idx_result:",idx_result)

            for i in idx_result:
                # print("i:",i)
                gene.append([DC_array, CR_arrays[i]])

            dummy_DC_array = np.full(shape=k, fill_value=1)
            dummy_CR_array = np.arange(k)

            gene[-1] = [DC_array, dummy_CR_array]

            while len(gene) < n_blocks + 1:
                gene.append([dummy_DC_array, dummy_CR_array])

        # gene = gene[0:1] + gene[1:(num_active_block//2)+1] + gene[(num_active_block//2)+1: num_active_block][::-1] + gene[num_active_block:]

        # for i in range(1,((num_active_block//2)+1)):
        #     # print(gene[i])
        #     gene[i][1] = sort_indices_by_values(gene[i][1])
        # print("After:", gene[i])

        return gene

    else:
        raise ValueError("Cannot generate genotype with the given k.")
