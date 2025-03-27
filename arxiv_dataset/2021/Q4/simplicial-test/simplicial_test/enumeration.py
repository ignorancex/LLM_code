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
from . import validators
from .sample import get_hitting_sets
from .custom_exceptions import SimplicialSignal
from copy import deepcopy
from itertools import combinations_with_replacement, starmap, product
from collections import defaultdict


def compute_dpv(facets, n=None, is_sorted=True):
    if n is None:
        dpv = defaultdict(int)
    else:
        dpv = np.zeros([n], dtype=np.int_)

    for facet in facets:
        for vid in facet:
            dpv[vid] += 1
    if n is not None:
        return dpv

    if is_sorted:
        return tuple(sorted(list(dpv.values()), reverse=True))
    else:
        _dpv = []
        for _ in range(len(dpv.keys())):
            _dpv += [dpv[_]]
        return tuple(_dpv)


def get_relabeled_facets(facets):
    equiv = defaultdict(list)
    deg_list = compute_dpv(facets, is_sorted=False)

    for idx, _ in enumerate(deg_list):
        equiv[_] += [idx]
    index = 0
    old2new = dict()
    for _ in sorted(equiv.items(), key=lambda kv: kv[0], reverse=True):
        for _v in _[1]:
            if _v not in old2new:
                old2new[_v] = index
                index += 1
    new_facets = []
    for facet in facets:
        _facet = []
        for vtx in facet:
            _facet += [old2new[vtx]]
        new_facets += [tuple(sorted(_facet))]
    return sort_facets(new_facets)


def groupby_vtx_equiv_class(d_input):
    res = {}
    for i, v in d_input.items():
        res[v] = [i] if v not in res.keys() else res[v] + [i]
    return res


class EnumRegistrar(object):
    def __init__(self):
        self.m = 0
        self.pointer = 0
        self.facets_pointer = 0
        self.facet_count_per_facets = defaultdict(int)
        self.ns_vtx = 0
        self.dfs_locator = None
        self.deg_seq = self.size_seq = None

        self.facet2id = dict()
        self.id2facet = dict()

        self.facets2id = dict()
        self.id2facets = dict()

        self.states = dict()

        self.facet_size_per_id = np.array([], dtype=np.int_)
        self.logbook = dict()

        self.facet_book = dict()
        self.dpv_m = defaultdict(set)

        self.dpv2fid = dict()
        self.dpv2fids = defaultdict(set)

        self.dpv_pointer = 0
        self.dpv2id = dict()

    def register_facet(self, facet) -> (tuple, int):
        facet = tuple(sorted(facet, reverse=False))
        if facet not in self.facet2id:
            self.facet2id[facet] = self.pointer
            self.id2facet[self.pointer] = facet
            self.pointer += 1
        return facet, self.facet2id[facet]

    def register_facets(self, facets):
        if facets not in self.facets2id:
            self.facets2id[facets] = self.facets_pointer
            self.id2facets[self.facets_pointer] = facets
            self.facets_pointer += 1
            # print(f"registering {sorted_facets} with id={self.facets_pointer - 1}")
        return facets, self.facets2id[facets]

    def register_state(self, facets, previous):
        """
        dpv: `degree per vertex`
        
        vpf: `vertex id per facet`

        vsc: `vertex symmetry class`

        n: `number of facets`

        Parameters
        ----------
        facets

        Returns
        -------

        """
        fid = self.facets2id[facets]
        dpv = compute_dpv(facets, is_sorted=True)

        # for plot use
        if dpv not in self.dpv2id:
            self.dpv2id[dpv] = self.dpv_pointer
            self.dpv_pointer += 1

        # if dpv in self.dpv_m[len(facets)]:
        #     return

        if fid not in self.states:
            vpf = self.compute_vpf(facets)
            vsc = groupby_vtx_equiv_class(self.dict2tuple(vpf))
            self.states[fid] = {
                "m": len(facets),
                "dpv": dpv,
                "vpf": vpf,
                "vsc": vsc,
                "loc": (len(facets), self.facet_count_per_facets[len(facets)], self.ns_vtx),
                "dfs": tuple(self.dfs_locator),
                "p": previous
            }
            self.dpv_m[len(facets)].add(dpv)
            self.dpv2fids[dpv].add(fid)
            self.dpv2fid[dpv] = fid
            self.facet_count_per_facets[len(facets)] += 1

    @staticmethod
    def get_created_vids(facets):
        return set(flatten(facets))

    def compute_vpf(self, facets) -> dict:
        vpf = defaultdict(list)
        for facet in facets:
            for vid in facet:
                vpf[vid] += [self.facet2id[tuple(facet)]]
        return vpf

    @staticmethod
    def dict2tuple(d_input):
        d_input_tupled = dict()
        for d in d_input:
            d_input_tupled[d] = tuple(d_input[d])
        return d_input_tupled

    def log_forbidden(self, name, reason_id) -> None:
        self.logbook[tuple(name)] = {
            "is_simplicial": False,
            "reason": reason_id
        }

    def update_incrementally(self, facet, facets):
        if self.deg_seq is not None:
            if max(facet) >= len(self.deg_seq):
                return
            if len(facets) > 0 and max(flatten(facets)) >= len(self.deg_seq):
                return
            self.size_seq = np.array(self.size_seq, dtype=np.int_)
            residual_degs, non_shielding_q = validators.get_residual_data(self.deg_seq, facet)
            if not validators.apply(residual_degs, self.size_seq[len(facets):], None, non_shielding_q):
                return

        if len(facets) == 0:
            _facets = [tuple(facet)]
            previous = None
        else:
            _facets = deepcopy(facets)
            previous = self.facets2id[tuple(_facets)]
            _facets += [tuple(facet)]

        relabeled_facets = get_relabeled_facets(_facets)
        for facet in relabeled_facets:
            self.register_facet(facet)
        self.register_facets(relabeled_facets)
        self.register_state(relabeled_facets, previous)

        dpv = compute_dpv(relabeled_facets, is_sorted=True)
        if self.deg_seq is not None and dpv == tuple(self.deg_seq):
            print(f"WeCanStopSignal:{dpv} == {self.deg_seq}")
            raise SimplicialSignal


class Enum(EnumRegistrar):
    def __init__(self, size_seq, deg_seq=None):
        super().__init__()
        self.size_seq = sorted(size_seq, reverse=True)
        self.m = len(self.size_seq)
        self.created_vids = set()
        self.deg_seq = deg_seq
        pass

    @staticmethod
    def get_dfs_navigator(size_seq):
        m = len(size_seq)
        if m == 1:
            return product([0])
        dummy1 = np.zeros([m], dtype=np.int_)
        dummy1[1] = 1
        dummy2 = [_ + 1 for _ in size_seq]
        dummy2[0] = 1
        s_map = starmap(range, zip(dummy1, dummy2))
        return product(*s_map)

    def compute(self):
        s = self.size_seq.pop(0)
        facets = []
        facet = []
        for _ in range(s):
            facet += [_]
        self.dfs_locator = [0]
        self.update_incrementally(facet, facets)
        facets_count = 0
        while len(self.size_seq) > 0:
            facets_count += 1
            next_size = self.size_seq.pop(0)
            pool = self.get_fids_per_m(facets_count)

            for fid in pool:
                self.dfs_locator = list(self.states[fid]["dfs"])
                facets = self.id2facets[fid]
                for ns_vtx in range(0, next_size + 1):
                    self.ns_vtx = ns_vtx
                    self.created_vids = self.get_created_vids(facets)
                    self.dfs_locator += [ns_vtx]
                    if ns_vtx == 0:  # must check with hitting set routine
                        self.fill_wo_creating_new_vertices(next_size, facets)
                    else:  # freedom in choosing slots in the shielded region, must consider identical vertices
                        self.fill_w_creating_new_vertices(next_size, facets, ns_vtx)
                    self.dfs_locator.pop(-1)

    @staticmethod
    def get_hs_identifier(vsc, hs):
        vsc_inv = {tuple(v): k for k, v in vsc.items()}
        _id = []
        for _hs in hs:
            for k, v in vsc_inv.items():
                if _hs in k:
                    _id += [v[0]]
        return tuple(sorted(_id))

    def fill_wo_creating_new_vertices(self, next_size, facets):
        hs_list = get_hitting_sets(facets, self.created_vids)
        vsc = self.states[self.facets2id[tuple(facets)]]["vsc"]
        vsc_deepcopy = deepcopy(vsc)

        if len(hs_list) == 0:
            return
        else:
            hs_ids = dict()
            for _hs_list in hs_list:
                try:
                    hs_ids[self.get_hs_identifier(vsc, _hs_list)]
                except KeyError:
                    hs_ids[self.get_hs_identifier(vsc, _hs_list)] = _hs_list
                else:
                    if sum(_hs_list) < sum(hs_ids[self.get_hs_identifier(vsc, _hs_list)]):
                        hs_ids[self.get_hs_identifier(vsc, _hs_list)] = _hs_list
            for _hs in hs_ids.values():
                if len(_hs) <= next_size:
                    vsc = deepcopy(vsc_deepcopy)
                    _next_size = next_size - len(_hs)
                    _iter = combinations_with_replacement([_ for _ in vsc.keys() if _ not in _hs], _next_size)
                    for _iter_combn in _iter:
                        facet = deepcopy(_hs)
                        tracker = defaultdict(int)
                        for symm_class in _iter_combn:
                            try:
                                while vsc[symm_class][tracker[symm_class]] in facet:
                                    tracker[symm_class] += 1
                                facet += [vsc[symm_class][tracker[symm_class]]]
                                tracker[symm_class] += 1
                            except IndexError:
                                break
                        if len(set(facet)) == next_size:
                            self.update_incrementally(facet, facets)

    def fill_w_creating_new_vertices(self, next_size, facets, ns_vtx):
        if ns_vtx == next_size:
            return self.fill_with_only_ones(next_size, facets)
        facet = []
        for _ in range(ns_vtx):
            facet += [_ + len(self.created_vids)]
        facet_deepcopy = deepcopy(facet)
        vsc = self.states[self.facets2id[tuple(facets)]]["vsc"]
        vsc_deepcopy = deepcopy(vsc)
        _next_size = next_size - ns_vtx
        _iter = combinations_with_replacement(vsc.keys(), _next_size)
        for _iter_combn in _iter:
            facet = deepcopy(facet_deepcopy)
            vsc = deepcopy(vsc_deepcopy)
            tracker = defaultdict(int)
            for symm_class in _iter_combn:
                try:
                    while vsc[symm_class][tracker[symm_class]] in facet:
                        tracker[symm_class] += 1
                    facet += [vsc[symm_class][tracker[symm_class]]]
                    tracker[symm_class] += 1
                except IndexError:
                    break
            if len(set(facet)) == next_size:
                self.update_incrementally(facet, facets)

    def fill_with_only_ones(self, next_size, facets):
        facet = []
        for _ in range(next_size):
            facet += [_ + len(self.created_vids)]
        self.update_incrementally(facet, facets)

    def get_fids_per_m(self, m):
        pool = []
        for _ in self.states:
            if self.states[_]["m"] == m:
                pool += [_]
        return pool

    def get_dpvs_per_m(self, m):
        pool = set()
        for _ in self.states:
            if self.states[_]["m"] == m:
                pool.add(self.states[_]["dpv"])
        return pool

    def get_fids_per_dfs_locator(self, dfs):
        pool = set()
        for _ in self.states:
            if self.states[_]["dfs"] == dfs:
                pool.add(_)
        return pool

    def compute_dfs(self):
        try:
            nav = self.get_dfs_navigator(self.size_seq)
            s = self.size_seq.pop(0)
            facets = []
            facet = []
            for _ in range(s):
                facet += [_]

            self.dfs_locator = tuple([0])
            self.update_incrementally(facet, facets)

            for _nav in nav:
                _nav = list(_nav)
                _nav.pop(0)
                idx = 0
                dfs_locator = [0]
                self.dfs_locator = tuple(dfs_locator)
                while len(_nav) > 0:
                    ns_vtx = _nav.pop(0)
                    next_size = self.size_seq[idx]
                    self.ns_vtx = ns_vtx

                    pool = self.get_fids_per_dfs_locator(self.dfs_locator)
                    dfs_locator += [ns_vtx]
                    self.dfs_locator = tuple(dfs_locator)

                    for facets_id in pool:
                        facets = list(self.id2facets[facets_id])
                        self.created_vids = self.get_created_vids(facets)
                        if ns_vtx == 0:
                            self.fill_wo_creating_new_vertices(next_size, facets)
                        elif ns_vtx == next_size:
                            self.fill_with_only_ones(next_size, facets)
                        else:
                            self.fill_w_creating_new_vertices(next_size, facets, ns_vtx)
                    idx += 1
        except SimplicialSignal:
            return

    def traceback(self, p):
        ps = [p]
        while p is not None:
            ps += [self.states[p]["p"]]
            p = self.states[p]["p"]
        return ps

    # def get_placed_order_per_fid(self, _id):
    #     flen = len(self.id2facets[_id])
    #     codes = []
    #     for c in range(flen):
    #         fid = tuple(self.id2facets[_id][:c + 1])
    #         codes += [self.states[self.facets2id[fid]]["loc"][2]]
    #     return codes
