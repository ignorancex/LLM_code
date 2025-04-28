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
import pyscf
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import _vhf
from byteqc.cuobc.scf.hf import (
    basis_seg_contraction, _make_s_index_offsets, _VHFOpt)
from byteqc.cuobc.lib import load_library
from byteqc.lib import Mg, gemm, cholesky

libgint = load_library('libgint')


def make_fake_mol():
    # fake mol for pairing with auxiliary basis
    fake_mol = pyscf.M(
        atom='He 0 0 0 ',
        verbose=1,
        basis={'He': [[0, (100, 2.00)]]})
    fake_mol, coeff = basis_seg_contraction(fake_mol)
    ptr_coeff = fake_mol._bas[0, gto.PTR_COEFF]
    ptr_exp = fake_mol._bas[0, gto.PTR_EXP]
    # due to the common factor
    # https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g1e.c
    fake_mol._env[ptr_coeff] = 1.0 / 0.282094791773878143
    fake_mol._env[ptr_exp] = 0.0
    return fake_mol


def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size):
    '''Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
    l = uniq_l_ctr[:, 0]
    nf = l * (l + 1) // 2
    _l_ctrs = []
    _l_ctr_counts = []
    for l_ctr, counts in zip(uniq_l_ctr, l_ctr_counts):
        l = l_ctr[0]
        nf = (l + 1) * (l + 2) // 2
        max_shells = max(group_size // nf, 2)
        if counts <= max_shells:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(counts)
            continue

        nsubs, rests = counts.__divmod__(max_shells)
        _l_ctrs.extend([l_ctr] * nsubs)
        _l_ctr_counts.extend([max_shells] * nsubs)
        if rests > 0:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(rests)
    uniq_l_ctr = numpy.vstack(_l_ctrs)
    l_ctr_counts = numpy.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts


def sort_mol(mol, cart=True, group_size=None):
    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    l_ctrs = mol._bas[:, [gto.ANG_OF, gto.NPRIM_OF]]

    uniq_l_ctr, _, inv_idx, l_ctr_counts = numpy.unique(
        l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)
    # Limit the number of AOs in each group
    if group_size is not None:
        uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
            uniq_l_ctr, l_ctr_counts, group_size)

    if mol.verbose >= logger.DEBUG3:
        logger.debug3(mol, 'Number of shells for each [l, nctr] group')
        for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
            logger.debug(mol, '    %s : %s', l_ctr, n)

    sorted_idx = numpy.argsort(inv_idx, kind='stable').astype(numpy.int32)
    # Sort contraction coefficients before updating self.mol
    ao_loc = mol.ao_loc_nr(cart=cart)
    nao = ao_loc[-1]

    # Some addressing problems in GPU kernel code
    assert nao < 32768
    ao_idx = numpy.array_split(numpy.arange(nao), ao_loc[1:-1])
    ao_idx = numpy.hstack([ao_idx[i] for i in sorted_idx])
    # Sort basis inplace
    mol._bas = mol._bas[sorted_idx]
    return mol, ao_idx, uniq_l_ctr, l_ctr_counts


def get_pairing(p_offsets, q_offsets, q_cond, cutoff=1e-11,
                diag_block_with_triu=True):
    '''
    pair shells and return pairing indices
    '''
    log_qs = []
    pair2bra = []
    pair2ket = []
    for p0, p1 in zip(p_offsets[:-1], p_offsets[1:]):
        for q0, q1 in zip(q_offsets[:-1], q_offsets[1:]):
            if q0 < p0:
                q_sub = q_cond[p0:p1, q0:q1].ravel()
                idx = q_sub.argsort(axis=None)[::-1]
                q_sorted = q_sub[idx]
                mask = q_sorted > cutoff
                idx = idx[mask]
                ishs, jshs = numpy.unravel_index(idx, (p1 - p0, q1 - q0))
                ishs += p0
                jshs += q0
                pair2bra.append(ishs)
                pair2ket.append(jshs)

                log_q = numpy.log(q_sorted[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)
            elif p0 == q0 and p1 == q1:
                q_sub = q_cond[p0:p1, p0:p1].ravel()
                idx = q_sub.argsort(axis=None)[::-1]
                q_sorted = q_sub[idx]
                ishs, jshs = numpy.unravel_index(idx, (p1 - p0, p1 - p0))
                mask = q_sorted > cutoff
                if not diag_block_with_triu:
                    # Drop the shell pairs in the upper triangle for diagonal
                    # blocks
                    mask &= ishs >= jshs

                ishs = ishs[mask]
                jshs = jshs[mask]
                ishs += p0
                jshs += p0
                if (len(ishs) == 0 and len(jshs) == 0):
                    continue

                pair2bra.append(ishs)
                pair2ket.append(jshs)

                log_q = numpy.log(q_sorted[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)

    return log_qs, pair2bra, pair2ket


class VHFOpt3c(_VHFOpt):
    def __init__(self, mol, auxmol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None, aosym=True):
        self.mol, self.coeff = basis_seg_contraction(mol)
        self.auxmol, self.auxcoeff = basis_seg_contraction(auxmol)
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

    def build(self, cutoff=1e-13, group_size=None,
              aux_group_size=None, diag_block_with_triu=True, gpus=None):
        if gpus is not None:
            Mg.set_gpus(gpus)
        cput0 = (logger.process_clock(), logger.perf_counter())
        sorted_mol, ao_idx, uniq_l_ctr, l_ctr_counts = sort_mol(
            self.mol, True, group_size)
        self.coeff = cupy.asarray(self.coeff[ao_idx])

        # sort fake mol
        fake_mol = make_fake_mol()
        fake_mol, _, fake_uniq_l_ctr, fake_l_ctr_counts = sort_mol(fake_mol)

        # sort auxiliary mol
        sorted_auxmol, aux_ao_idx, aux_uniq_l_ctr, aux_l_ctr_counts = sort_mol(
            self.auxmol, True, aux_group_size)
        self.auxcoeff = cupy.asarray(self.auxcoeff[aux_ao_idx])

        tot_mol = gto.mole.conc_mol(fake_mol, sorted_auxmol)
        tot_mol = gto.mole.conc_mol(sorted_mol, tot_mol)

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, tot_mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        q_cond = self.get_q_cond()
        cput1 = logger.timer(sorted_mol, 'Initialize q_cond', *cput0)

        l_ctr_offsets = numpy.append(0, numpy.cumsum(l_ctr_counts))
        log_qs, pair2bra, pair2ket = get_pairing(
            l_ctr_offsets, l_ctr_offsets, q_cond,
            diag_block_with_triu=diag_block_with_triu)
        self.log_qs = log_qs.copy()

        fake_l_ctr_offsets = numpy.append(0, numpy.cumsum(fake_l_ctr_counts))
        fake_l_ctr_offsets += l_ctr_offsets[-1]
        aux_l_ctr_offsets = numpy.append(0, numpy.cumsum(aux_l_ctr_counts))
        self.aux_l_ctr_offsets = aux_l_ctr_offsets.copy()

        aux_l_ctr_offsets += fake_l_ctr_offsets[-1]
        aux_log_qs, aux_pair2bra, aux_pair2ket = get_pairing(
            aux_l_ctr_offsets, fake_l_ctr_offsets, q_cond,
            diag_block_with_triu=diag_block_with_triu)
        self.aux_log_qs = aux_log_qs.copy()

        pair2bra += aux_pair2bra
        pair2ket += aux_pair2ket

        # self.uniq_l_ctr = numpy.concatenate([uniq_l_ctr, fake_uniq_l_ctr, aux_uniq_l_ctr])
        # self.aux_uniq_l_ctr = aux_uniq_l_ctr
        self.l_ctr_offsets = numpy.concatenate([
            l_ctr_offsets,
            fake_l_ctr_offsets[1:],
            aux_l_ctr_offsets[1:]])

        self.bas_pair2shls = numpy.hstack(
            pair2bra + pair2ket).astype(numpy.int32).reshape(2, -1)
        self.bas_pairs_locs = numpy.append(
            0, numpy.cumsum([x.size for x in pair2bra])).astype(numpy.int32)

        log_qs = log_qs + aux_log_qs
        ao_loc = tot_mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)
        # self.pmol = tot_mol
        if diag_block_with_triu:
            scale_shellpair_diag = 1.
        else:
            scale_shellpair_diag = 0.5

        self.initBpcache(scale_shellpair_diag, ao_loc, ncptype, tot_mol)

        logger.timer(tot_mol, 'Initialize GPU cache', *cput1)
        self.ao_loc = self.mol.ao_loc_nr(cart=True)
        self.aux_ao_loc = self.auxmol.ao_loc_nr(cart=True)
        return self


def get_int2c(vhfopt, j2c_eig_always=False, linear_dep_threshold=1e-13):
    unao = vhfopt.mol.nao
    unaux = vhfopt.auxmol.nao
    norb = unao + unaux + 1

    int2c = cupy.zeros([unaux, unaux], order='F')

    nbins = 1
    ao_offsets = numpy.array(
        [unao + 1, unao, unao + 1, unao],
        dtype=numpy.int32)
    strides = numpy.array([1, unaux, unaux, unaux * unaux], dtype=numpy.int32)

    for k_id, log_q_k in enumerate(vhfopt.aux_log_qs):
        bins_locs_k = _make_s_index_offsets(log_q_k, nbins)
        cp_k_id = k_id + len(vhfopt.log_qs)
        for l_id, log_q_l in enumerate(vhfopt.aux_log_qs):
            bins_locs_l = _make_s_index_offsets(log_q_l, nbins)
            cp_l_id = l_id + len(vhfopt.log_qs)
            err = libgint.GINTfill_int2e(
                vhfopt.bpcaches[Mg.getgid()],
                ctypes.cast(int2c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_k.ctypes.data_as(ctypes.c_void_p),
                bins_locs_l.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_k_id),
                ctypes.c_int(cp_l_id),
                ctypes.cast(cupy.cuda.get_current_stream().ptr,
                            ctypes.c_void_p))
            if (err != 0):
                raise RuntimeError("int2c2e failed\n")

    int2c = gemm(vhfopt.auxcoeff, int2c, transa='T')
    int2c = gemm(int2c, vhfopt.auxcoeff)
    try:
        if j2c_eig_always:
            raise cupy.linalg.LinAlgError
        v = cholesky(int2c)  # do not overwrite int2c here
    except cupy.linalg.LinAlgError:
        e, v = cupy.linalg._eigenvalue._syevd(
            int2c, 'U', with_eigen_vector=True, overwrite_a=True)
        e, v = cupy.linalg.eigh(int2c)
        mask = e > linear_dep_threshold
        v = v[:, mask] * cupy.sqrt(1 / e[mask]).reshape(1, -1)
    return v


def get_int3c(cp_ij_id, cp_aux_id, vhfopt, bpcache=None, buf=None,
              order='F'):
    if bpcache is None:
        bpcache = vhfopt.bpcaches[Mg.getgid()]
    unao = vhfopt.mol.nao
    unaux = vhfopt.auxmol.nao
    unorb = unao + unaux + 1

    cpi, cpj = ind2pair(cp_ij_id)
    cp_kl_id = cp_aux_id + len(vhfopt.log_qs)
    log_q_ij = vhfopt.log_qs[cp_ij_id]
    log_q_kl = vhfopt.aux_log_qs[cp_aux_id]

    nbins = 1
    bins_locs_ij = numpy.array([0, len(log_q_ij)], dtype=numpy.int32)
    bins_locs_kl = numpy.array([0, len(log_q_kl)], dtype=numpy.int32)

    i0, i1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpi:cpi + 2]]
    j0, j1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpj:cpj + 2]]
    k0, k1 = vhfopt.auxmol.ao_loc[vhfopt.aux_l_ctr_offsets
                                  [cp_aux_id: cp_aux_id + 2]]

    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0

    ao_offsets = numpy.array([i0, j0, unao + 1 + k0, unao], dtype=numpy.int32)
    if order == 'C':
        strides = numpy.array([nj * nk, nk, 1, 1], dtype=numpy.int32)
    else:
        strides = numpy.array([1, ni, ni * nj, 1], dtype=numpy.int32)
    if buf is None:
        int3c_blk = cupy.zeros([ni, nj, nk], order=order)
    else:
        int3c_blk = cupy.ndarray([ni, nj, nk], order=order, memptr=buf.data)

    int3c_blk[:] = 0.0
    err = libgint.GINTfill_int2e(
        bpcache,
        ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
        ctypes.c_int(unorb),
        strides.ctypes.data_as(ctypes.c_void_p),
        ao_offsets.ctypes.data_as(ctypes.c_void_p),
        bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
        bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbins),
        ctypes.c_int(cp_ij_id),
        ctypes.c_int(cp_kl_id),
        ctypes.cast(cupy.cuda.get_current_stream().ptr, ctypes.c_void_p))
    if (err != 0):
        raise RuntimeError('GINT_fill_int2e failed')
    return slice(i0, i1), slice(j0, j1), slice(k0, k1), int3c_blk


def ind2pair(ind):
    a = int(numpy.sqrt(2 * ind + 0.25) - 0.5)
    return a, ind - a * (a + 1) // 2
