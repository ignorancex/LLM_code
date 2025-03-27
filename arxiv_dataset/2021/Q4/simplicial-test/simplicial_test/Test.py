#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simplicial_test -- a python module to realize simplicial degree-size sequences
#
# Copyright (C) 2020-2021 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from . import validators
from .utils import *
from .custom_exceptions import NoMoreBalls, SimplicialSignal, GoToNextLevel, NonSimplicialSignal
from copy import deepcopy
import heapq  # note that in Standard Library, heap queue = min-priority queue

try:
    from sage.all import Combinations as combinations
except ImportError:
    from more_itertools import distinct_combinations as combinations
else:
    import sys

    sys.setrecursionlimit(100000)


class Test:
    r"""Base class for [Simpliciality]Test.

    Parameters
    ----------
    degree_list : ``iterable`` or :class:`numpy.ndarray`, required
        Sequence of vertex degree distribution.

    size_list : ``iterable`` or :class:`numpy.ndarray`, required
        Sequence of facet size distribution.

    blocked_sets : ``list`` of integer-valued ``tuple`` objects (optional, default ``None``)
        Facets that must be respected for non-inclusion.

    verbose : ``bool`` (optional, default ``False``)
        If ``True``, progress information will be shown.

    **kwargs :  keyword arguments
        Keyword arguments to be passed to base type constructor.

    Examples
    --------
    >>> from simplicial_test import *
    >>> degree_list, size_list = ([2, 1, 3, 2, 1], [3, 2, 2, 2])
    >>> st = Test(degree_list, size_list, verbose=False)
    >>> is_simplicial, facets = st.is_simplicial()
    >>> print(is_simplicial, facets)
    (True, ((0, 1, 3), (0, 2), (0, 4), (1, 2)))

    """

    def __init__(self, degree_list, size_list, verbose=False, **kwargs):
        super().__init__()
        self._level = 0
        self.s_depot = kwargs.pop("s_depot", SimplicialDepot(
            sorted(degree_list, reverse=True), sorted(size_list, reverse=True)
        ))
        self.degree_list, self.size_list = self.s_depot.degree_list, self.s_depot.size_list
        self.blocked_sets = []

        self.depth = kwargs.pop("depth", np.infty)
        self.width = kwargs.pop("width", 1e2)
        self.s_depot.cutoff = kwargs.pop("cutoff", 1e5)
        self.verbose = verbose
        if len(kwargs) > 0:
            raise ValueError(f"unrecognized keyword arguments: {str(list(kwargs.keys()))}")

    @staticmethod
    def get_distinct_selection(size, degs, blocked_sets, level_map, width):
        r"""Compute the generator for candidate facets (each of size ``size``) from a set of nodes.
        These facets are ordered according to the branching heuristic as explained in the paper,
        see https://arxiv.org/abs/2106.00185.

        Parameters
        ----------
        size : ``int``
            Number of vertices to make the facet.

        degs : ``list`` of ``int``
            Sequence of wanting (or residual) vertex degree sequence.

        blocked_sets : ``list`` of integer-valued ``tuple`` objects (optional, default ``None``)
            Facets that must be respected for non-inclusion.

        level_map : ``dict``
            This dictionary translates a vertex id in the first stage (l=1) to that of the desired stage.

        Returns
        -------
        facet : ``tuple``
            The candidate facet.

        Notes
        -----

        Note that _ = vtx degree at original view, vid = vtx index at original view,
        level_map[vid] =  vtx index at current view.

        """
        h = []
        collection_keys = []
        key2vid = defaultdict(list)
        for ind_d1, d1 in enumerate(degs):
            if level_map[ind_d1] != -1:
                key = tuple(get_indices_of_k_in_blocked_sets(blocked_sets, level_map[ind_d1]) + [d1])
                key2vid[key] += [level_map[ind_d1]]
                collection_keys += [key]
        candidates = combinations(collection_keys, size).__iter__()
        for entry_count in range(int(width)):
            facet = []
            tracker = defaultdict(int)
            weight = 0
            try:
                candidate = next(candidates)
            except StopIteration:
                break
            for key in candidate:
                facet += [key2vid[key][tracker[key]]]
                tracker[key] += 1
                weight += key[-1]
            # heapq.heappush(h, (entry_count, tuple(facet)))  # arXiv:2106.00185v1
            heapq.heappush(h, (- weight, - sum(facet), entry_count, tuple(facet)))  # resubmitted version

        return h

    def sample_candidate_facet(self, size, valid_trials=None):
        r"""

        Parameters
        ----------
        size : ``int``
            Number of vertices to make the facet.

        valid_trials

        Returns
        -------

        """
        if not valid_trials:
            self.s_depot.valid_trials[self._level - 1] = self.get_distinct_selection(
                size, self.s_depot.prev_d[1], self.blocked_sets, self.s_depot.level_map[self._level - 1], self.width)
        counter = 0
        while True:
            try:
                facet = heapq.heappop(self.s_depot.valid_trials[self._level - 1])[-1]  # candidate_facet
            except IndexError:
                self.s_depot.add_to_time_counter(self._level - 1, reason="backtrack")
            else:
                if validators.a_issubset_any_b(facet, self.blocked_sets):
                    continue
                counter += 1
                if not self.validate(facet):
                    self.s_depot.add_to_time_counter(self._level - 1, reason="reject")

    def validate(self, facet) -> (bool, str):
        r"""This function must return True in order for the candidate facet to be considered.

        Parameters
        ----------
        facet : ``tuple``
            The candidate facet.

        Returns
        -------
        token : ``bool``
            A tuple of the form ``(ind, reason)``, where ``ind`` is the indicator for whether
            the candidate facet passes the validations and ``reason`` explains why they fail.

        Raises
        ------
        NoMoreBalls : :class:`simplicial_test.custom_exceptions.NoMoreBalls`
            if we depleted this level and wanted to go up (i.e., from `lv` to `lv - 1`).

        SimplicialSignal : :class:`simplicial_test.custom_exceptions.SimplicialSignal`
            Signal that indicates a simplicial instance has been found.

        Notes
        -----

        """
        residual_sizes = self.size_list
        if np.sum(residual_sizes) == 0:
            raise SimplicialSignal([facet])  # Last facet explored.

        # Validating the candidate facet.
        residual_degs, non_shielding_q = validators.get_residual_data(self.degree_list, facet)
        simplified_blocked_sets = simplify_blocked_sets(self.blocked_sets)
        if not validators.apply(residual_degs, residual_sizes, simplified_blocked_sets, non_shielding_q):
            return False

        # Reducing the forced sets of nodes and facets.
        safe, data = validators.reduce(residual_degs, residual_sizes, [facet] + simplified_blocked_sets)
        if not safe:
            return False
        residual_degs, residual_sizes, residual_blocked_sets, collected_facets, exempt_vids = data
        if np.sum(residual_sizes) == 0:
            raise SimplicialSignal([facet] + [exempt_vids] + collected_facets)

        self.s_depot.maps[self._level] = get_seq2seq_mapping(residual_degs)  # (mapping_forward, mapping_backward)
        self.s_depot.compute_lmap(self._level, self.s_depot.prev_d[self._level], self.s_depot.maps[self._level][0])

        b_new = transform_facets(residual_blocked_sets, self.s_depot.maps[self._level][0], to="l+1")
        blocked_sets = sort_facets(b_new)

        # We can make the algorithm faster by checking whether a branch is marked before we recursively explore it.
        sorted_residual_degs = sorted(residual_degs, reverse=True)
        _ = (tuple(sorted_residual_degs), tuple(residual_sizes), tuple(blocked_sets))
        if self.s_depot.explored[_]:
            self.s_depot.add_to_time_counter(self._level - 1, reason="reject")
            return False
        self.s_depot.explored[_] = True
        self.degree_list, self.size_list, self.blocked_sets = sorted_residual_degs, residual_sizes, blocked_sets
        raise GoToNextLevel(facet, exempt_vids, collected_facets)

    def is_simplicial(self):
        if not validators.check_preconditions(self.degree_list, self.size_list):
            return False, self.__mark(False, tuple())
        self.degree_list, self.size_list, num_ones = pair_one_by_one(self.degree_list, self.size_list)
        self.s_depot.collects[0] = [(len(self.degree_list) + _,) for _ in range(num_ones)]
        if np.sum(self.size_list) == 0:
            return True, self.__mark(True, self._assemble_simplicial_facets([]))
        while True:
            self._level += 1
            self._verbose_logging()
            self._stage_state()
            s = self.size_list.pop(0)
            try:
                self.sample_candidate_facet(s, valid_trials=self.s_depot.valid_trials[self._level - 1])
            except SimplicialSignal as e:
                return True, self.__mark(True, self._assemble_simplicial_facets(e.message))
            except GoToNextLevel as e:
                sigma, delta, sigma_c = e.message
                self.s_depot.candidates[self._level] = sigma
                self.s_depot.exempts[self._level] = delta
                self.s_depot.collects[self._level] = sigma_c
            except NoMoreBalls:
                if self._level > self.depth:
                    self._rollback(self.depth)
                else:
                    self._rollback(self._level - 1)
            except NonSimplicialSignal:
                return False, self.__mark(False, tuple())

    def _clean_valid_trials(self, level):
        for ind in self.s_depot.valid_trials.keys():
            if ind >= level:
                self.s_depot.valid_trials[ind] = None

    def _assemble_simplicial_facets(self, seed_facets):
        seed_facets = [_ for _ in seed_facets if len(_) != 0]
        if self._level > 1:
            for lv in np.arange(self._level - 1, 0, -1):
                seed_facets = transform_facets(seed_facets, self.s_depot.maps[lv][1], to="l-1")
                seed_facets = [self.s_depot.exempts[lv] + list(s) for s in seed_facets]
                seed_facets = seed_facets + [self.s_depot.candidates[lv]] + self.s_depot.collects[lv]
        return tuple(sort_callback(seed_facets) + self.s_depot.collects[0])

    def __mark(self, simplicial, facets):
        self.s_depot.simplicial = simplicial
        self.s_depot.facets = sort_facets(facets)
        del self.s_depot.valid_trials
        del self.s_depot.explored
        return self.s_depot.facets

    def _rollback(self, level):
        self._clean_valid_trials(level)
        self.size_list = self.s_depot.prev_s[level]
        self.degree_list = self.s_depot.prev_d[level]
        self.blocked_sets = self.s_depot.prev_b[level]
        self._level = level - 1

    def _stage_state(self):
        self.s_depot.prev_s[self._level] = deepcopy(self.size_list)
        self.s_depot.prev_d[self._level] = deepcopy(self.degree_list)
        self.s_depot.prev_b[self._level] = deepcopy(self.blocked_sets)

    def _verbose_logging(self):
        if self.verbose:
            print(f"========== (l={self._level}) ==========\n"
                  f"Size list: {[_ for _ in self.size_list if _ > 0]}\n"
                  f"Degree list: {[_ for _ in self.degree_list if _ > 0]}\n"
                  f"blocked_sets = {self.blocked_sets}\n"
                  # f"currents = {self.s_depot.candidates}\n"
                  # f"exempts = {self.s_depot.exempts}\n"
                  # f"collects = {self.s_depot.collects}"
                  )
