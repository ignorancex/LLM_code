# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf)
# and GPU4PySCF (https://github.com/bytedance/gpu4pyscf),
# which are licensed under the Apache License 2.0.
# The original copyright:
#     Copyright 2014-2020 The GPU4PySCF/PySCF Developers. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy
import cupy
import ctypes
from byteqc.cuobc.lib.int3c import ind2pair
from byteqc.cuobc.scf.hf import _VHFOpt, _make_s_index_offsets
from byteqc.cuobc.lib import load_library
from byteqc.lib import Mg, empty_from_buf


libgint = load_library('libgint')


def get_int4c(mol, blksize=None, vhfopt=None, batch=1, buf_size=0, nbins=1):
    if vhfopt is None:
        vhfopt = _VHFOpt(mol, 'int2e')
        vhfopt.build(group_size=blksize, buf_size=buf_size)

    nao = vhfopt.mol.nao

    def task(int4c, cp_ij_id, cp_kl_id):
        if int4c is None:
            int4c = cupy.zeros((nao, nao, nao, nao))
        si, sj, sk, sl, int4blk = get_int4c_slice(
            cp_ij_id, cp_kl_id, vhfopt, nbins=nbins)
        int4c[si, sj, sk, sl] = int4blk
        int4c[sj, si, sk, sl] = int4blk.transpose((1, 0, 2, 3))
        int4c[si, sj, sl, sk] = int4blk.transpose((0, 1, 3, 2))
        int4c[sj, si, sl, sk] = int4blk.transpose((1, 0, 3, 2))

        int4c[sk, sl, si, sj] = int4blk.transpose((2, 3, 0, 1))
        int4c[sk, sl, sj, si] = int4blk.transpose((2, 3, 1, 0))
        int4c[sl, sk, si, sj] = int4blk.transpose((3, 2, 0, 1))
        int4c[sl, sk, sj, si] = int4blk.transpose((3, 2, 1, 0))

        return int4c

    int4c = Mg.reduce(
        task, *numpy.tril_indices(len(vhfopt.log_qs)),
        batch=batch)
    return int4c


def get_int4c_slice(cp_ij_id, cp_kl_id, vhfopt, buf=None, nbins=1):
    cpi, cpj = ind2pair(cp_ij_id)
    cpk, cpl = ind2pair(cp_kl_id)

    i0, i1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpi:cpi + 2]]
    j0, j1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpj:cpj + 2]]
    k0, k1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpk:cpk + 2]]
    l0, l1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpl:cpl + 2]]

    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0
    nl = l1 - l0

    ao_offsets = numpy.array([i0, j0, k0, l0], dtype=numpy.int32)
    strides = numpy.array([nj * nk * nl, nk * nl, nl, 1], dtype=numpy.int32)
    int4c_blk = empty_from_buf(buf, [ni, nj, nk, nl])
    int4c_blk[:] = 0.0

    err = _get_int4c(int4c_blk, cp_ij_id, cp_kl_id, vhfopt, ao_offsets,
                     strides, nbins)
    if (err != 0):
        raise RuntimeError('GINT_fill_int2e failed')
    return slice(i0, i1), slice(j0, j1), slice(k0, k1), \
        slice(l0, l1), int4c_blk


def _get_int4c(eri, cp_ij_id, cp_kl_id, vhfopt, ao_offsets, strides, nbins=1):
    bpcache = vhfopt.bpcaches[Mg.getgid()]
    nao = vhfopt.mol.nao

    log_q_ij = vhfopt.log_qs[cp_ij_id]
    log_q_kl = vhfopt.log_qs[cp_kl_id]

    bins_locs_ij = _make_s_index_offsets(
        log_q_ij, nbins, vhfopt.direct_scf_tol)
    bins_locs_kl = _make_s_index_offsets(
        log_q_kl, nbins, vhfopt.direct_scf_tol)

    err = libgint.GINTfill_int2e(
        bpcache,
        ctypes.cast(eri.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao),
        strides.ctypes.data_as(ctypes.c_void_p),
        ao_offsets.ctypes.data_as(ctypes.c_void_p),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbins),
        ctypes.c_int(cp_ij_id),
        ctypes.c_int(cp_kl_id),
        ctypes.cast(cupy.cuda.get_current_stream().ptr, ctypes.c_void_p))
    return err
