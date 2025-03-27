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
import random
from .utils import *

try:
    from pysat.examples.hitman import Hitman
except ModuleNotFoundError:
    # Windows(TM) will fail
    pass


def gen_joint_sequence(m, poisson_lambda, n_max=0, size_seq=None):
    if size_seq is None:
        _size_list = [0]
        while min(_size_list) == 0:
            _size_list = sorted(np.random.poisson(lam=poisson_lambda, size=m), reverse=True)
    else:
        _size_list = sorted(size_seq, reverse=True)

    if n_max == 0:
        # print(n := np.random.randint(max(_size_list) + 1, np.sum(_size_list) + 1))
        n_max = np.sum(_size_list)
    if n_max <= max(_size_list):
        n_max = max(_size_list) + 1

    facets = []
    for facet_size in _size_list:
        candidate_facet = random.sample(range(0, n_max), k=facet_size)
        qualified_draw = False

        count = 0
        while not qualified_draw:
            count += 1
            if count > 1e4:
                # print("Re-sample joint sequence at a higher n_max!")
                n_max += 1
                return gen_joint_sequence(m, poisson_lambda, n_max, size_seq=_size_list)
            qualified_draw = True
            for facet in facets:
                if set(candidate_facet).issubset(facet):
                    candidate_facet = random.sample(range(0, n_max), k=facet_size)
                    qualified_draw = False
                    break
        facets += [candidate_facet]
    _degree_list = sorted(list(Counter(flatten(facets)).values()), reverse=True)
    facets = relabel(facets)
    return sorted(_size_list, reverse=True), _degree_list, facets


def get_hitting_sets(facets, created_vids):
    ns_slots = []
    for facet in facets:
        ns_slots += [list(created_vids.difference(set(facet)))]

    hs_list = []
    with Hitman(bootstrap_with=ns_slots, htype='sorted') as hitman:
        try:
            for hs in hitman.enumerate():
                hs_list += [hs]
        except TypeError:
            # object of type 'NoneType' has no len()
            # potentially a bug in Hitman; reproducible when
            # ns_slots = [
            # [12, 13, 14, 15, 16, 17, 18],
            # [4, 5, 6, 7, 8, 9, 10, 11, 17, 18],
            # [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
            # [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 18]
            # ]
            pass
    return hs_list


def gen_joint_sequence_from_sizes(size_seq, p=0.2):
    size_seq = sorted(size_seq, reverse=True)
    _size_list = sorted(size_seq, reverse=True)
    vid_registar = dict()
    s = size_seq.pop(0)
    vid = 0
    facets = []
    facet = []
    for _ in range(s):
        vid_registar[_] = vid
        facet += [vid]
        vid += 1

    facets += [facet]
    _facet = []
    while len(size_seq) > 0:
        token = False
        _s = size_seq.pop(0)
        created_vids = set(list(vid_registar.keys()))
        if random.random() < p:  # create a new group
            token = True
        else:
            hs_list = get_hitting_sets(facets, created_vids)
            # print(f"hs_list = {hs_list}")
            if len(hs_list) == 0:
                token = True
            else:
                random.shuffle(hs_list)
                while True:
                    if len(hs_list) == 0:
                        token = True
                        break
                    ns = hs_list.pop(0)

                    if _s - len(ns) < 0:
                        continue
                    _facet = ns + random.sample(created_vids.difference(set(ns)), _s - len(ns))
                    # print(f"2 adding {facet}")
                    break
        if token:
            _facet = []
            num_of_new_groups = random.randint(1, _s)
            # print(f"_s = {_s}, num_of_new_groups={num_of_new_groups}")
            for _ in range(num_of_new_groups):
                vid_registar[len(created_vids)] = vid
                _facet += [vid]
                vid += 1
            _facet += random.sample(created_vids, _s - num_of_new_groups)
            # print(f"1 adding {_facet}...  take {_s - num_of_new_groups} from {created_vids}")

        facets += [_facet]
    _degree_list = sorted(list(Counter(flatten(facets)).values()), reverse=True)
    facets = relabel(facets)
    return sorted(_size_list, reverse=True), _degree_list, facets


def relabel(facets):
    d = {}
    i = 0
    for facet in facets:
        for vid in facet:
            try:
                d[vid]
            except KeyError:
                d[vid] = i
                i += 1
            else:
                pass
    _facets = []
    for facet in facets:
        _facet = []
        for vid in facet:
            _facet += [d[vid]]
        _facets += [_facet]
    return _facets
