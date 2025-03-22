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

from .utils import *


def check_preconditions(sorted_d, sorted_s):
    cardinality_s = len(sorted_s)
    cardinality_d = np.count_nonzero(sorted_d)
    if np.max(sorted_d) > cardinality_s:
        return False
    if cardinality_d < np.max(sorted_s):
        return False
    if np.sum(sorted_d) != np.sum(sorted_s):
        return False

    counter_s = Counter(sorted_s)
    counter_d = Counter(sorted_d)
    if counter_s[1] > counter_d[1]:
        return False

    # if np.max(sorted_d) - (cardinality_s - counter_s[2]) > cardinality_d - 1:
    #     return False

    return True


def get_residual_data(degs, facet):
    r"""Used in the greedy case only.

    Parameters
    ----------
    degs: wanting degrees
    facet: candidate facet

    Returns
    -------

    """
    n = len(degs)
    residual_degs = [degs[_] - 1 if _ in set(facet) else degs[_] for _ in range(n)]
    q = [0 if _ in set(facet) else degs[_] for _ in range(n)]
    return np.array(residual_degs, dtype=np.int_), np.array(q, dtype=np.int_)


def apply(residual_degs, residual_sizes, blocked_sets, non_shielding_q):
    if not rule_1(residual_degs, residual_sizes):
        return False
    if not rule_2(residual_degs, residual_sizes, non_shielding_q):
        return False
    if not rule_3(residual_degs, blocked_sets):
        return False
    return True


def rule_1(residual_degs, residual_sizes):
    """

    Parameters
    ----------
    residual_degs
    residual_sizes

    Returns
    -------

    """
    cardinality_s = len(residual_sizes)
    cardinality_d = np.count_nonzero(residual_degs)
    if np.max(residual_degs) > cardinality_s:
        return False
    if cardinality_d < residual_sizes[0]:
        return False
    elif cardinality_d == residual_sizes[0]:
        if cardinality_s != 1:
            return False

    # With some memory of the previous stage, the following rule can throw NoMoreBalls,
    # if we believe that no more alternation of proposal facet would pass the validation rule.
    # We did not apply the rule in the paper.
    #
    # counter_s = Counter(residual_sizes)
    # if np.max(residual_degs) - (cardinality_s - counter_s[2]) > cardinality_d - 1:
    #     return False
    return True


def rule_2(residual_degs, residual_sizes, non_shielding_q):
    r"""[Validate non-shielding nodes (i.e., Q)]
    Verify that the sum of non-shielding degrees not be less than the number of the remaining facets.
    If this function returns True, then we pass the validation rule. Otherwise, it returns false.

    Parameters
    ----------
    residual_sizes
    residual_degs
    non_shielding_q

    Returns
    -------


    """
    shielding = residual_degs - non_shielding_q  # only shielding
    cardinality_s = len(residual_sizes)
    if cardinality_s > np.sum(non_shielding_q):
        return False
    elif cardinality_s == np.sum(non_shielding_q):
        # if cardinality_s <= 1:  # only the last to-be-chosen facet remains
        #     return True
        if residual_sizes[0] - 1 > np.count_nonzero(shielding):
            return False
    return True


def rule_3(residual_degs, blocked_sets):
    for blocked_set in blocked_sets:
        if set(np.nonzero(residual_degs)[0]).issubset(set(blocked_set)):
            return False
    return True


def reduce(residual_degs, residual_sizes, blocked_sets) -> tuple:
    r"""

    Parameters
    ----------
    residual_degs
    residual_sizes
    blocked_sets
    verbose

    Returns
    -------

    """
    collected_facets, exempt_vids = [], []
    counter_degs = Counter(residual_degs)

    while counter_degs[len(residual_sizes)] > 0 and np.sum(residual_sizes) != 0:
        residual_sizes = [_ - counter_degs[len(residual_sizes)] for _ in residual_sizes]
        if (len(residual_sizes) > 1 and Counter(residual_sizes)[0] > 0) or np.min(residual_sizes) < 0:
            return False, None
        exempt_vids += np.where(residual_degs == len(residual_sizes))[0].tolist()  # exempt vids

        if np.sum(residual_sizes) == 0:
            if a_issubset_any_b(exempt_vids, blocked_sets):
                return False, None
        residual_degs[residual_degs == len(residual_sizes)] = 0

        counter_sizes = Counter(residual_sizes)
        counter_degs[len(residual_sizes)] = 0

        if counter_sizes[1] > 0:
            if counter_sizes[1] > counter_degs[1]:
                return False, None
            cannot_choose = set()
            for bset in blocked_sets:
                if set(exempt_vids).issubset(bset):
                    cannot_choose = cannot_choose.union(set(bset).difference(exempt_vids))
            sites_1s = set(np.where(residual_degs == 1)[0])
            sites_1s.difference_update(cannot_choose)

            if len(sites_1s) < counter_sizes[1]:
                return False, None

            residual_degs, removed_sites = remove_ones(residual_sizes, residual_degs, choose_from=sites_1s)
            [residual_sizes.remove(1) for _ in range(len(removed_sites))]

            collected_facets += [exempt_vids + [_] for _ in removed_sites]  # collected_facets immer from removed_sites
            if np.sum(residual_sizes) == 0:
                exempt_vids = []
            counter_degs[1] -= len(removed_sites)
            counter_sizes[1] -= len(removed_sites)

    residual_blocked_sets = filter_blocked_facets(blocked_sets, exempt_vids)
    return True, (residual_degs, residual_sizes, residual_blocked_sets, collected_facets, exempt_vids)


def a_issubset_any_b(a, b=None):
    if b is not None:
        for _b in b:
            if set(a).issubset(set(_b)):
                return True
    return False
