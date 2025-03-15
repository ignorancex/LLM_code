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

import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from functools import partial
from typing import List
from copy import deepcopy
from .custom_exceptions import NonSimplicialSignal, NoMoreBalls


@dataclass(frozen=False)
class SimplicialDepot:
    degree_list: List
    size_list: List
    simplicial: bool = False
    facets: tuple = ()
    prev_d: dict = field(repr=False, default_factory=dict)
    prev_s: dict = field(repr=False, default_factory=dict)
    prev_b: dict = field(repr=False, default_factory=dict)
    maps: dict = field(repr=False, default_factory=dict)
    exempts: dict = field(repr=False, default_factory=dict)
    collects: dict = field(repr=False, default_factory=dict)
    candidates: dict = field(repr=False, default_factory=dict)
    valid_trials: dict = field(repr=False, default_factory=dict)
    levels_traj: List = field(repr=False, default_factory=list)
    conv_time: int = field(repr=False, default=0)
    cutoff: int = field(repr=False, init=False)

    def __post_init__(self):
        self.explored = defaultdict(set)
        self.time = np.zeros(len(self.size_list), np.int_)
        self.level_map = defaultdict(partial(np.ndarray, len(self.degree_list), int))
        for ind, _ in enumerate(self.degree_list):
            self.level_map[0][ind] = ind
        for ind, _ in enumerate(self.size_list):
            self.valid_trials[ind] = None
        self.level_map[1].fill(-1)

    def compute_lmap(self, level, degree_list, mapping_forward):
        for _ in range(len(degree_list)):
            vtx_current_view = self.level_map[level - 1][_]
            if vtx_current_view == -1 or vtx_current_view not in mapping_forward:
                self.level_map[level][_] = -1
            else:
                self.level_map[level][_] = mapping_forward[vtx_current_view]

    def add_to_time_counter(self, level, reason="reject"):
        self.time[int(level - 1)] += 1
        self.conv_time += 1
        self.levels_traj += [level - 1]
        if self.conv_time >= self.cutoff:
            raise NonSimplicialSignal
        if reason == "reject":
            pass
        elif reason == "backtrack":
            if level == 0:
                raise NonSimplicialSignal
            raise NoMoreBalls


def flatten(nested_list):
    f"""flattens out nested lists."""
    return [item for sublist in nested_list for item in sublist]


def simplify_blocked_sets(bsets):
    r"""No more inclusive blocked sets.

    Parameters
    ----------
    bsets : ``iterable`` of integer-valued ``tuple`` objects
        Facets that must be respected for non-inclusion, or "blocked sets (of facets)".

    Returns
    -------
    simplified_bsets : ``list`` of integer-valued ``tuple`` objects
        Simplified blocked sets, of which redundancy is removed.

    Examples
    --------
    >>> bsets = ((7, 44, 109, 273), (7, 273), (44, 273), (109, 273), (2,), (4,), (44,), (56,))
    >>> simplified_bsets = simplify_blocked_sets(bsets)
    >>> print(simplified_bsets)
    [(7, 44, 109, 273), (2,), (4,), (56,)]

    """
    simplified_bsets = []
    for _bsets in sorted(bsets, key=lambda _: -len(_)):
        if not tuple(set(_bsets)) in simplified_bsets:
            if len(simplified_bsets) > 0:
                to_add = True
                for _ in simplified_bsets:
                    if set(_bsets).issubset(set(_)):
                        to_add = False
                        break
                if to_add:
                    simplified_bsets += [tuple(set(_bsets))]
            else:
                simplified_bsets += [tuple(set(_bsets))]
    return simplified_bsets


def sort_facets(facets):
    sorted_facets = []
    for facet in facets:
        sorted_facets += [tuple(sorted(facet, reverse=False))]
    sorted_facets = set(sorted_facets)
    return tuple(sorted(sorted_facets, key=lambda _: [-len(_)] + list(_)))


def prune_included_facets(bsets):
    r"""Alias of simplify_blocked_sets."""
    return simplify_blocked_sets(bsets)


def get_indices_of_k_in_blocked_sets(blocked_sets, k):
    indices = []
    for idx, _ in enumerate(blocked_sets):
        if k in _:
            indices += [idx]
    return indices


def transform_facets(facets, mapping, to="l+1") -> set:
    if to == "l-1":  # mapping_backward
        return set(map(lambda x: tuple(map(lambda y: mapping[y], x)), facets))
    elif to == "l+1":  # mapping_forward
        _facets = set()
        for facet in facets:
            transformed_facet = []
            for vtx in facet:
                try:
                    transformed_facet += [mapping[vtx]]
                except KeyError:
                    pass
            if len(transformed_facet) > 0:
                _facets.add(tuple(sorted(transformed_facet)))
        return _facets
    else:
        raise ValueError(f"data content to={to} not understood")


def remove_ones(sizes, residual_degs, choose_from=None):
    # todo, to figure out: you cannot do [-Counter(s)[1]:]
    removed_vtx_sites = np.array(list(choose_from), dtype=np.int_)[:Counter(sizes)[1]]
    for site in removed_vtx_sites:
        residual_degs[site] = 0
    return residual_degs, removed_vtx_sites.tolist()


def pair_one_by_one(degree_list, size_list) -> (list, list, int):
    """
    This function is used to pair up the ones in size/deg lists, since this is the only plausible situation.
    Note that it is only applied when level=0.

    Parameters
    ----------
    degree_list
    size_list

    Returns
    -------

    """
    degree_list = sorted(degree_list, reverse=True)
    size_list = sorted(size_list, reverse=True)
    _ = min(Counter(degree_list)[1], Counter(size_list)[1])
    for __ in range(_):
        degree_list.remove(1)
        size_list.remove(1)
    return degree_list, size_list, _


def sort_helper(st) -> list:
    d = defaultdict()
    for idx, _ in enumerate(st.DEGREE_LIST):
        d[idx] = _

    inv_map = defaultdict(list)

    for _ in d:
        inv_map[d[_]] += [_]
    joint_seq = st.compute_joint_seq_from_identifier(sorted_deg=False)
    if len(joint_seq[0]) == 0:
        raise ValueError("Invalid input instance. Perhaps execute SimplicialTest.is_simplicial() first?")
    old2new = dict()
    for idx, _ in enumerate(st.compute_joint_seq_from_identifier(sorted_deg=False)[1]):
        old2new[idx] = inv_map[_].pop(0)

    translated = []
    for facet in st.identifier2facets():
        translated += [sorted([old2new[_] for _ in facet])]
    return translated


def sort_callback(facets):
    t = []
    for facet in facets:
        t += [tuple(sorted(facet, reverse=False))]
    return t


# def get_seq2seq_mapping(degs):
#     mapping_backward = dict()
#     nonzero_degs = np.count_nonzero(degs)
#     argsort = np.argsort(degs)[::-1]
#     for idx in np.arange(0, nonzero_degs, 1):
#         mapping_backward[idx] = argsort[idx]
#     mapping_forward = {v: k for k, v in mapping_backward.items()}
#     return mapping_forward, mapping_backward

def get_seq2seq_mapping(degs):
    old2new = dict()
    for idx, _deg in enumerate(degs):
        old2new[idx] = _deg
    inv_old2new = defaultdict(list)
    for key in old2new.keys():
        if old2new[key] != 0:
            inv_old2new[old2new[key]] += [key]
    _idx = 0
    _inv_old2new = deepcopy(inv_old2new)

    keys = sorted(inv_old2new.keys(), reverse=True)
    for key in keys:
        for idx, _ in enumerate(inv_old2new[key]):
            inv_old2new[key][idx] = _idx
            _idx += 1

    mapping_backward = dict()
    for key in inv_old2new.keys():
        for idx, new_key in enumerate(inv_old2new[key]):
            mapping_backward[new_key] = _inv_old2new[key][idx]

    mapping_forward = {v: k for k, v in mapping_backward.items()}
    return mapping_forward, mapping_backward


def filter_blocked_facets(blocked_facets, exempt_vids):
    r"""Find the effective subset of blocked_facets that affects future node selection at the presence of exempt_vids.

    Note that not all blocked_facets (B) collected at a certain level will be carried over to the next level.
    When we reduce the sequence and collect a number of exempt vids, these vids are shared among all remaining facets.
    It is possible that any of the exempt vids does not intersect with all existing blocked_facets,
    thus preventing the remaining facets to be a subset of the selected facets.

    Parameters
    ----------
    blocked_facets
    exempt_vids

    Returns
    -------
    filtered

    """
    if len(exempt_vids) == 0:  # only to make the function faster without sweeping through blocked_facets
        return blocked_facets
    filtered = []
    for facet in blocked_facets:
        if set(exempt_vids).issubset(facet):
            filtered += [facet]
    return filtered


def compute_joint_seq(facets) -> (list, list):
    if len(facets) == 0:
        raise ValueError("Empty input facets. Are you sure the input is correct?")
    flattened_facets = [item for sublist in facets for item in sublist]
    n = max(flattened_facets) - min(flattened_facets) + 1
    degs = np.zeros(n, dtype=np.int_)
    sizes = []

    for facet in facets:
        sizes += [len(facet)]
        for vertex_id in facet:
            degs[vertex_id] += 1

    return sorted(degs, reverse=True), sorted(sizes, reverse=True)


def if_facets_simplicial(facets) -> bool:
    for idx1, facet1 in enumerate(sorted(facets, key=len)):
        for idx2, facet2 in enumerate(sorted(facets, key=len)):
            if idx2 > idx1:
                if set(list(facet1)).issubset(set(list(facet2))):
                    return False
    return True


def accel_asc(n):
    """source: http://jeromekelleher.net/generating-integer-partitions.html"""
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def get_partition(n=1, sortby="asc"):
    gen = accel_asc(n)
    partitions = []
    while True:
        try:
            partitions += [sorted(next(gen), reverse=True)]
        except StopIteration:
            break
    if sortby == "asc":
        return partitions
    else:
        return sorted(partitions, key=lambda x: max(x) - len(x), reverse=True)


def read_hyperedge_list(path, delimiter=","):
    with open(path, 'r') as f:
        edge_list = set()
        for line in f:
            if not line.lstrip().startswith("%"):
                e = line.strip().split(delimiter)
                _edge_list = list()
                for _e in e:
                    _edge_list += [int(_e) - 1]
                edge_list.add(tuple(_edge_list))
        return list(edge_list)


def write_simplicial_list(el, path):
    with open(path, 'w') as f:
        for _el in el:
            for __el in _el:
                f.write(str(__el) + " ")
            f.write("\n")
