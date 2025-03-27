"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-23 21:38:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:36:35
"""

import random

import numpy as np

from .base import EvolutionarySearchBase

__all__ = ["EvolutionarySearch"]


class EvolutionarySearch(EvolutionarySearchBase):
    def __init__(
        self,
        *args,
        **kwargs,
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
        super().__init__(*args, **kwargs)

    def tell(self, scores):
        """perform evo search according to the scores"""
        # sorted_idx = np.array(scores).argsort()[: self.parent_size]
        sorted_idx = np.array(scores).argsort()[::-1][: self.parent_size]
        # sorted_idx = np.array([s.item() for s in scores]).argsort()[::-1][: self.parent_size]
        self.best_gene = self.population[sorted_idx[0]]
        self.best_solution = self.converter.gene2solution(self.best_gene)
        parents = [self.population[i] for i in sorted_idx]
        self.best_score = scores[sorted_idx[0]]

        # mutation
        mutate_population = []
        k = 0
        while k < self.mutation_size:
            mutated_gene = self.mutate(random.choices(parents)[0])
            if self.satisfy_constraints(mutated_gene):
                mutate_population.append(mutated_gene)
                k += 1

        # crossover
        crossover_population = []
        k = 0
        while k < self.crossover_size:
            parent1, parent2 = random.sample(parents, 2)
            crossovered_gene1, crossovered_gene2 = self.crossover(parent1, parent2)
            if self.satisfy_constraints(crossovered_gene1) and self.satisfy_constraints(
                crossovered_gene2
            ):
                crossover_population.append(crossovered_gene1)
                crossover_population.append(crossovered_gene2)
                k += 2

        self.population = parents + mutate_population + crossover_population
