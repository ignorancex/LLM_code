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
# This project includes code adapted from PySCF (https://github.com/pyscf/pyscf)
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

import ctypes
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import _vhf
from byteqc.lib import Mg
from byteqc.cupbc import lib as cupbclib
from pyscf.gto.mole import cart2sph
import bisect

LMAX_ON_GPU = 10
FREE_CUPY_CACHE = True

libgint = cupbclib.load_library('libgint')


class _VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol, self.coeffs = basis_seg_contraction(mol, isstack=False)
        self.coeff = scipy.linalg.block_diag(*self.coeffs)
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

    def build_aux(self, group_size=None, diag_block_with_triu=True):
        self.build(group_size, diag_block_with_triu, True)

    def build(self, group_size=None, diag_block_with_triu=True,
              isaux=False, iL=0, jL=0, Ls=np.zeros(3)):
        cput0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = mol._bas[:, [gto.ANG_OF, gto.NPRIM_OF]]
        uniq_l_ctr, _, inv_idx, l_ctr_counts = np.unique(
            l_ctrs, return_index=True, return_inverse=True, return_counts=True,
            axis=0)

        # Limit the number of AOs in each group
        if group_size is not None:
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
                uniq_l_ctr, l_ctr_counts, group_size)

        if mol.verbose >= logger.DEBUG:
            logger.debug1(mol, 'Number of shells for each [l, nctr] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug(mol, '    %s : %s', l_ctr, n)
        sorted_idx = np.argsort(inv_idx, kind='stable').astype(np.int32)
        # Sort contraction coefficients before updating self.mol
        ao_loc = mol.ao_loc_nr(cart=True)
        self.ao_loc = ao_loc
        nao = ao_loc[-1]
        # Some addressing problems in GPU kernel code
        # assert nao < 32768
        ao_idx = np.array_split(np.arange(nao), ao_loc[1:-1])
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        self.ao_idx = ao_idx
        self.coeff = self.coeff[ao_idx]
        # Sort basis inplace
        mol._bas = mol._bas[sorted_idx]
        self.sorted_idx = sorted_idx
        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        lmax = uniq_l_ctr[:, 0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:, 0] == l].sum()
                     for l in range(lmax + 1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax >= LMAX_ON_GPU:
            self.g_shls = l_slices[LMAX_ON_GPU:LMAX_ON_GPU + 2].tolist()
        else:
            self.g_shls = []
        if lmax > LMAX_ON_GPU:
            self.h_shls = l_slices[LMAX_ON_GPU + 1:].tolist()
        else:
            self.h_shls = []

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        # q_cond = self.get_q_cond()
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        pair2bra, pair2ket = _get_pair_bra_ket(
            l_ctr_offsets, uniq_l_ctr, isaux)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = l_ctr_offsets
        self.bas_pair2shls = np.hstack(
            pair2bra + pair2ket).astype(np.int32).reshape(-1)
        self.bas_pairs_locs = np.append(
            0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        ao_loc = mol.ao_loc_nr(cart=True)
        self.ao_loc = ao_loc
        ncptype = len(pair2bra)
        self.ncptype = ncptype
        self.set_coeffs()
        self.bpcache = ctypes.POINTER(BasisProdCache)()
        if diag_block_with_triu:
            scale_shellpair_diag = 1.
        else:
            scale_shellpair_diag = 0.5
        if isaux:
            self.initauxBpcache(scale_shellpair_diag, ao_loc, ncptype, mol)
        else:
            libgint.GINTinit_basis_prod(
                ctypes.byref(self.bpcache), ctypes.c_double(
                    scale_shellpair_diag),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                self.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                self.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ncptype),
                mol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(iL),
                ctypes.c_int(jL), Ls.ctypes.data_as(ctypes.c_void_p))
        logger.timer(mol, 'Initialize GPU cache', *cput0)
        return self

    def initauxBpcache(self, scale_shellpair_diag, ao_loc, ncptype, mol):
        nbas = mol.nbas
        bas_coords = np.empty((nbas * 3,))
        bpcache = ctypes.POINTER(BasisProdCache)()
        libgint.GINTinit_basis_prod_aux_cpu(
            ctypes.byref(bpcache), ctypes.c_double(scale_shellpair_diag),
            bas_coords.ctypes.data_as(ctypes.c_void_p),
            self.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            self.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ncptype),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p))
        aexyz = np.ctypeslib.as_array(
            bpcache.contents.aexyz,
            (bpcache.contents.primitive_pairs_locs[ncptype] * 5,))

        self.memory = Mg.broadcast(
            ao_loc, bas_coords, aexyz, self.bas_pair2shls)
        self.bpcaches = [None] * Mg.ngpu
        for i in range(Mg.ngpu):
            self.bpcaches[i] = ctypes.POINTER(BasisProdCache)()
            libgint.GINTinit_basis_prod_aux_gpu(
                ctypes.byref(bpcache), ctypes.byref(self.bpcaches[i]),
                ctypes.cast(self.memory[0][i].data.ptr, ctypes.c_void_p),
                ctypes.cast(self.memory[1][i].data.ptr, ctypes.c_void_p),
                ctypes.cast(self.memory[2][i].data.ptr, ctypes.c_void_p),
                ctypes.cast(self.memory[3][i].data.ptr, ctypes.c_void_p))
        self.bpcache = self.bpcaches[0]

    def set_coeffs(self):
        ao_loc = self.mol.ao_loc_nr(cart=True)
        naoSlices = ao_loc[self.l_ctr_offsets]
        coeff_off = np.cumsum([0] + [c.shape[0] for c in self.coeffs])
        coeff_off2 = np.cumsum([0] + [c.shape[1] for c in self.coeffs])
        coeffs = []
        coeffs_inds = []
        for i in range(len(naoSlices) - 1):
            inds = self.ao_idx[naoSlices[i]:naoSlices[i + 1]]
            j = 0
            coeff = []
            coeff_ind = []
            while j < len(inds):
                icoeff = bisect.bisect_right(coeff_off, inds[j]) - 1
                m = min(len(inds) - j, coeff_off[icoeff + 1] - inds[j])
                if m != 1:
                    for m in range(j + 1, j + m):
                        if inds[m] - inds[m - 1] != 1:
                            break
                    m += 1 - j
                coeff.append(
                    self.coeffs[icoeff][inds[j] - coeff_off[icoeff]:inds[j]
                                        + m - coeff_off[icoeff]])
                coeff_ind.append(
                    range(coeff_off2[icoeff], coeff_off2[icoeff + 1]))
                j += m
            coeffs.append(scipy.linalg.block_diag(*coeff))
            coeffs_inds.append(np.hstack(coeff_ind))
        self.coeffs = coeffs
        self.coeffs_inds = coeffs_inds

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        libgint.GINTdel_basis_prod(ctypes.byref(self.bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass


class ContractionProdType(ctypes.Structure):
    _fields_ = [
        ("l_bra", ctypes.c_int),
        ("l_ket", ctypes.c_int),
        ("nprim_12", ctypes.c_int),
        ("npairs", ctypes.c_int),
    ]


class BasisProdCache(ctypes.Structure):
    _fields_ = [
        ("nbas", ctypes.c_int),
        ("ncptype", ctypes.c_int),
        ("cptype", ctypes.POINTER(ContractionProdType)),
        ("bas_pairs_locs", ctypes.POINTER(ctypes.c_int)),
        ("primitive_pairs_locs", ctypes.POINTER(ctypes.c_int)),
        ("bas_pair2shls", ctypes.POINTER(ctypes.c_int)),
        ("aexyz", ctypes.POINTER(ctypes.c_double)),
        ("bas_coords", ctypes.POINTER(ctypes.c_double)),
        ("bas_pair2bra", ctypes.POINTER(ctypes.c_int)),
        ("bas_pair2ket", ctypes.POINTER(ctypes.c_int)),
        ("ao_loc", ctypes.POINTER(ctypes.c_int)),
        ("a12", ctypes.POINTER(ctypes.c_double)),
        ("e12", ctypes.POINTER(ctypes.c_double)),
        ("x12", ctypes.POINTER(ctypes.c_double)),
        ("y12", ctypes.POINTER(ctypes.c_double)),
        ("z12", ctypes.POINTER(ctypes.c_double)),
    ]


def basis_seg_contraction(mol, allow_replica=False, isstack=True):
    '''transform generally contracted basis to segment contracted basis

    Kwargs:
        allow_replica:
            transform the generally contracted basis to replicated
            segment-contracted basis
    '''
    bas_templates = {}
    _bas = []
    _env = mol._env.copy()
    contr_coeff = []

    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:, :2]):
        key = tuple(mol._bas[ib0:ib1, gto.PTR_EXP])
        if key in bas_templates:
            bas_of_ia, coeff = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:, gto.ATOM_OF] = ia
        else:
            # Generate the template for decontracted basis
            coeff = []
            bas_of_ia = []
            for shell in mol._bas[ib0:ib1]:
                l = shell[gto.ANG_OF]
                nf = (l + 1) * (l + 2) // 2
                nctr = shell[gto.NCTR_OF]
                if nctr == 1:
                    bas_of_ia.append(shell)
                    coeff.append(np.eye(nf))
                    continue

                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[gto.NPRIM_OF]
                pcoeff = shell[gto.PTR_COEFF]
                if allow_replica:
                    coeff.extend([np.eye(nf)] * nctr)
                    bs = np.repeat(shell[np.newaxis], nctr, axis=0)
                    bs[:, gto.NCTR_OF] = 1
                    bs[:, gto.PTR_COEFF] = np.arange(
                        pcoeff, pcoeff + nprim * nctr, nprim)
                    bas_of_ia.append(bs)
                else:
                    pexp = shell[gto.PTR_EXP]
                    exps = _env[pexp:pexp + nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    c = _env[pcoeff:pcoeff + nprim * nctr].reshape(nctr, nprim)
                    c = np.einsum('ip,p,ef->iepf', c, 1 / norm, np.eye(nf))
                    coeff.append(c.reshape(nf * nctr, nf * nprim).T)

                    _env[pcoeff:pcoeff + nprim] = norm
                    bs = np.repeat(shell[np.newaxis], nprim, axis=0)
                    bs[:, gto.NPRIM_OF] = 1
                    bs[:, gto.NCTR_OF] = 1
                    bs[:, gto.PTR_EXP] = np.arange(pexp, pexp + nprim)
                    bs[:, gto.PTR_COEFF] = np.arange(pcoeff, pcoeff + nprim)
                    bas_of_ia.append(bs)

            bas_of_ia = np.vstack(bas_of_ia)
            bas_templates[key] = (bas_of_ia, coeff)

        _bas.append(bas_of_ia)
        contr_coeff.extend(coeff)

    pmol = gto.copy(mol)
    pmol._mesh = mol._mesh
    pmol.cart = True
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env
    if mol.cart:
        contr_coeff = scipy.linalg.block_diag(*contr_coeff)
    else:
        contr_coeff = merge_diag(contr_coeff, cart2sph_coeff(mol))
        if isstack:
            contr_coeff = scipy.linalg.block_diag(*contr_coeff)
    return pmol, contr_coeff


def merge_diag(block1, block2):
    i = j = 1
    di = block1[0].shape[1]
    dj = block2[0].shape[0]
    r = []
    while True:
        if di == dj:
            r.append(scipy.linalg.block_diag(
                *block1[0:i]) @ scipy.linalg.block_diag(*block2[0:j]))
            block1 = block1[i:]
            block2 = block2[j:]
            if len(block1) == 0:
                assert len(block2) == 0, "Two blocks size is not the same"
                break
            i = 1
            j = 1
            di = block1[0].shape[1]
            dj = block2[0].shape[0]
        elif dj < di:
            dj += block2[j].shape[0]
            j += 1
        else:
            di += block1[i].shape[1]
            i += 1
    return r


def cart2sph_coeff(mol, normalized='sp'):
    '''Transformation matrix that transforms Cartesian GTOs to spherical
    GTOs for all basis functions

    Kwargs:
        normalized : string or boolean
            How the Cartesian GTOs are normalized.  Except s and p functions,
            Cartesian GTOs do not have the universal normalization coefficients
            for the different components of the same shell.  The value of this
            argument can be one of 'sp', 'all', None.  'sp' means the Cartesian
            s and p basis are normalized.  'all' means all Cartesian functions
            are normalized.  None means none of the Cartesian functions are
            normalized. The default value 'sp' is the convention used by
            libcint library.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvtz')
    >>> c = mol.cart2sph_coeff()
    >>> s0 = mol.intor('int1e_ovlp_sph')
    >>> s1 = c.T.dot(mol.intor('int1e_ovlp_cart')).dot(c)
    >>> print(abs(s1-s0).sum())
    >>> 4.58676826646e-15
    '''
    c2s_l = [cart2sph(l, normalized=normalized) for l in range(12)]
    c2s = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for n in range(mol.bas_nctr(ib)):
            c2s.append(c2s_l[l])
    return c2s


def _get_pair_bra_ket(l_ctr_offsets, uniq_l_ctr, isaux=False):
    pair2bra = []
    pair2ket = []
    if isaux:
        for i, (p0, p1) in enumerate(
                zip(l_ctr_offsets[:-1], l_ctr_offsets[1:])):
            if uniq_l_ctr[i, 0] > LMAX_ON_GPU:
                # no integrals with g functions should be evaluated on GPU
                continue
            pair2bra.append(np.arange(p0, p1))
    else:
        for i, (p0, p1) in enumerate(
                zip(l_ctr_offsets[:-1], l_ctr_offsets[1:])):
            if uniq_l_ctr[i, 0] > LMAX_ON_GPU:
                print("no integrals with g functions should be evaluated on "
                      "GPU")
                continue
            for q0, q1 in zip(l_ctr_offsets[:i], l_ctr_offsets[1:i + 1]):
                idx = np.arange(0, (p1 - p0) * (q1 - q0))
                ishs, jshs = np.unravel_index(idx, (p1 - p0, q1 - q0))
                ishs += p0
                jshs += q0
                pair2bra.append(ishs)
                pair2ket.append(jshs)
            idx = np.arange(0, (p1 - p0) * (p1 - p0))
            ishs, jshs = np.unravel_index(idx, (p1 - p0, p1 - p0))
            ishs += p0
            jshs += p0
            pair2bra.append(ishs)
            pair2ket.append(jshs)
    return pair2bra, pair2ket


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
        assert group_size >= nf, "group_size to small to handle even one "
        "orbital!!"
        max_shells = group_size // nf
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
    uniq_l_ctr = np.vstack(_l_ctrs)
    l_ctr_counts = np.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts
