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

import time
import copy
import ctypes
import numpy
import cupy
import scipy.linalg
import warnings
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from byteqc.cuobc.lib import load_library
from byteqc.lib import Mg, contraction, elementwise_binary

LMAX_ON_GPU = 5

libgint = load_library('libgint')
libgvhf = load_library('libgvhf')


def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None, group_size=None, gpus=None, prescreen=True,
           nbins=None):
    '''Compute J, K matrices with CPU-GPU hybrid algorithm
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if hermi != 1:
        raise NotImplementedError(
            'JK-builder only supports hermitian density matrix')

    if vhfopt is None:
        vhfopt = _VHFOpt(mol, 'int2e').build(gpus=gpus, group_size=group_size)

    coeff = cupy.asarray(vhfopt.coeff)
    nao, nao0 = coeff.shape
    dm0 = dm
    dm = dm.reshape(-1, nao0, nao0)
    n_dm = dm.shape[0]
    dms = cupy.asarray(dm)
    dms = contraction('nij', dms, 'qj', coeff, 'niq')
    dms = contraction('niq', dms, 'pi', coeff, 'npq')
    dmss = Mg.broadcast(dms)
    dms = dms.get()
    scripts = []

    if with_j:
        vj = Mg.mapgpu(lambda: cupy.zeros(dms.shape).transpose(0, 2, 1))
        vj_ptr = [ctypes.cast(x.data.ptr, ctypes.c_void_p) for x in vj]
        scripts.append('ji->s2kl')
    if with_k:
        vk = Mg.mapgpu(lambda: cupy.zeros(dms.shape).transpose(0, 2, 1))
        vk_ptr = [ctypes.cast(x.data.ptr, ctypes.c_void_p) for x in vk]
        if hermi == 1:
            scripts.append('jk->s2il')
        else:
            scripts.append('jk->s1il')

    l_symb = lib.param.ANGULAR
    bas_pairs_locs = vhfopt.bas_pairs_locs
    log_qs = vhfopt.log_qs
    direct_scf_tol = vhfopt.direct_scf_tol

    if nbins is None:
        # adjust nbins according to the size of the system
        pairs_max = (bas_pairs_locs[1:] - bas_pairs_locs[:-1]).max()
        nbins = max(10, int(pairs_max // 100000))
        log.debug('Set the number of buckets for s_index to %d', nbins)

    ncptype = len(log_qs)
    cp_idx, cp_jdx = numpy.tril_indices(ncptype)
    shell_locs_for_l_ctr_offsets = vhfopt.l_ctr_offsets
    l_ctr_ao_loc = vhfopt.mol.ao_loc[shell_locs_for_l_ctr_offsets]
    dm_ctr_cond = numpy.max(
        [lib.condense('absmax', x, l_ctr_ao_loc) for x in dms], axis=0)
    if hermi != 1:
        dm_ctr_cond = (dm_ctr_cond + dm_ctr_cond.T) * .5

    def task(n):
        igpu = Mg.getgid()
        cp_ij_id = int(numpy.floor(numpy.sqrt(2 * n)))
        if cp_ij_id * (cp_ij_id + 1) // 2 > n:
            cp_ij_id -= 1
        cp_kl_id = n - cp_ij_id * (cp_ij_id + 1) // 2
        # print("\nGPU%d-%d/%d-%d" % (Mg.gpus[igpu], cp_ij_id, len(log_qs) - 1,
        #                             cp_kl_id))
        log_q_ij = log_qs[cp_ij_id]
        log_q_kl = log_qs[cp_kl_id]
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        cpk = cp_idx[cp_kl_id]
        cpl = cp_jdx[cp_kl_id]
        li = vhfopt.uniq_l_ctr[cpi, 0]
        lj = vhfopt.uniq_l_ctr[cpj, 0]
        lk = vhfopt.uniq_l_ctr[cpk, 0]
        ll = vhfopt.uniq_l_ctr[cpl, 0]
        assert li <= LMAX_ON_GPU and lj <= LMAX_ON_GPU and lk <= LMAX_ON_GPU \
            and ll <= LMAX_ON_GPU, 'Angular momentem is too large' \
            '  %d-%d-%d-%d' % (
                li, lj, lk, ll)
        if log_q_ij.size == 0 or log_q_kl.size == 0:
            return

        sub_dm_cond = max(dm_ctr_cond[cpi, cpj], dm_ctr_cond[cpk, cpl],
                          dm_ctr_cond[cpi, cpk], dm_ctr_cond[cpj, cpk],
                          dm_ctr_cond[cpi, cpl], dm_ctr_cond[cpj, cpl])
        if prescreen and sub_dm_cond < direct_scf_tol:
            return

        t0 = time.perf_counter()
        cutoff = direct_scf_tol / sub_dm_cond
        bins_locs_ij = _make_s_index_offsets(log_q_ij, nbins, cutoff)
        bins_locs_kl = _make_s_index_offsets(log_q_kl, nbins, cutoff)

        err = libgvhf.GINTbuild_jk(
            vhfopt.bpcaches[igpu], vj_ptr[igpu], vk_ptr[igpu],
            ctypes.cast(dmss[igpu].data.ptr, ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(n_dm),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins), ctypes.c_int(cp_ij_id),
            ctypes.c_int(cp_kl_id),
            ctypes.cast(cupy.cuda.get_current_stream().ptr,
                        ctypes.c_void_p))
        if err != 0:
            detail = f'CUDA Error for '\
                '({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
            raise RuntimeError(detail)
        log.debug1('(%s%s|%s%s) on GPU %.3fs',
                   l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                   time.perf_counter() - t0)

    ntask = len(log_qs) * (len(log_qs) + 1) // 2
    Mg.map(task, range(ntask))
    dmss = None
    vj = Mg.sum(vj)
    vk = Mg.sum(vk)

    if with_j:
        vj = contraction('nqp', vj, 'qj', coeff, 'njp')
        vj = contraction('njp', vj, 'pi', coeff, 'nij')
        tmp = vj.copy()
        vj = elementwise_binary('nji', vj, 'nij', tmp,
                                alpha=2.0, gamma=2.0).get()
    if with_k:
        vk = contraction('nqp', vk, 'qj', coeff, 'njp')
        vk = contraction('njp', vk, 'pi', coeff, 'nij')
        tmp = vk.copy()
        vk = elementwise_binary('nji', vk, 'nij', tmp, gamma=1.0).get()
    tmp = None
    cput0 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)

    h_shls = vhfopt.h_shls
    if h_shls:
        warnings.warn('There are angular momentums larger than h!!!')
        log.debug3('Integrals for %s functions on CPU',
                   l_symb[LMAX_ON_GPU + 1])
        pmol = vhfopt.mol
        shls_excludes = [0, h_shls[0]] * 4
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms.get(), 1, pmol._atm, pmol._bas, pmol._env,
                                 vhfopt=vhfopt, shls_excludes=shls_excludes)
        coeff = vhfopt.coeff
        idx, idy = numpy.tril_indices(nao, -1)
        for (i, vs) in enumerate(vs_h):
            if vs.ndim == 2:
                vs_h[i] = vs.reshape((1,) + vs.shape)

        if with_j and with_k:
            vj1 = vs_h[0].reshape(n_dm, nao, nao)
            vk1 = vs_h[1].reshape(n_dm, nao, nao)
        elif with_j:
            vj1 = vs_h[0].reshape(n_dm, nao, nao)
        else:
            vk1 = vs_h[0].reshape(n_dm, nao, nao)

        if with_j:
            vj1[:, idy, idx] = vj1[:, idx, idy]
            for i, v in enumerate(vj1):
                vj[i] += coeff.T.dot(v).dot(coeff)
        if with_k:
            if hermi:
                vk1[:, idy, idx] = vk1[:, idx, idy]
            for i, v in enumerate(vk1):
                vk[i] += coeff.T.dot(v).dot(coeff)
        cput0 = log.timer_debug1('get_jk pass 2 for l>4 basis on cpu', *cput0)

    coeff = dms = None

    if dm0.ndim == 2:
        if with_j:
            vj = vj[0]
        if with_k:
            vk = vk[0]
    return vj, vk


def _get_jk(mf, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    if omega is not None:
        raise NotImplementedError('Range separated Coulomb integrals')
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mf)
    log.debug3('apply get_jk on gpu')
    if hasattr(mf, '_opt_gpu'):
        vhfopt = mf._opt_gpu
    else:
        vhfopt = _VHFOpt(
            mol, getattr(mf.opt, '_intor', 'int2e'),
            getattr(mf.opt, 'prescreen', 'CVHFnrs8_prescreen'),
            getattr(mf.opt, '_qcondname', 'CVHFsetnr_direct_scf'),
            getattr(mf.opt, '_dmcondname', 'CVHFsetnr_direct_scf_dm'))
        vhfopt.build(mf.direct_scf_tol)
        mf._opt_gpu = vhfopt
    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega,
                    verbose=log)
    log.timer('vj and vk on gpu', *cput0)
    return vj, vk


def eig(h, s):
    '''Solver for generalized eigenvalue problem
    '''
    seig, t = cupy.linalg.eigh(cupy.asarray(s))
    seig_inv = 1. / cupy.sqrt(seig)
    seig_inv[seig < 1e-15] = 0
    t *= seig_inv
    heff = t.conj().T.dot(cupy.asarray(h)).dot(t)
    e, c = cupy.linalg.eigh(heff)
    c = t.dot(c)
    e = e.get()
    c = c.get()
    return e, c


def _eigh(mf, h, s):
    return eig(h, s)


class RHF(hf.RHF):
    get_jk = _get_jk
    _eigh = _eigh


class _VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol, self.coeff = basis_seg_contraction(mol)
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

    def build(self, cutoff=1e-13, group_size=None, diag_block_with_triu=False,
              gpus=None):
        if gpus is not None:
            Mg.set_gpus(gpus)
        cput0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = mol._bas[:, [gto.ANG_OF, gto.NPRIM_OF]]
        uniq_l_ctr, _, inv_idx, l_ctr_counts = numpy.unique(
            l_ctrs, return_index=True, return_inverse=True,
            return_counts=True, axis=0)

        # Limit the number of AOs in each group
        if group_size is not None:
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
                uniq_l_ctr, l_ctr_counts, group_size)

        if mol.verbose >= logger.DEBUG:
            logger.debug1(mol, 'Number of shells for each [l, nctr] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug(mol, '    %s : %s', l_ctr, n)

        sorted_idx = numpy.argsort(inv_idx, kind='stable').astype(numpy.int32)
        # Sort contraction coefficients before updating self.mol
        ao_loc = mol.ao_loc_nr(cart=True)
        nao = ao_loc[-1]
        # Some addressing problems in GPU kernel code
        ao_idx = numpy.array_split(numpy.arange(nao), ao_loc[1:-1])
        ao_idx = numpy.hstack([ao_idx[i] for i in sorted_idx])
        self.coeff = self.coeff[ao_idx]
        # Sort basis inplace
        mol._bas = mol._bas[sorted_idx]

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        lmax = uniq_l_ctr[:, 0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:, 0] == l].sum()
                     for l in range(lmax + 1)]
        l_slices = numpy.append(0, numpy.cumsum(nbas_by_l))
        if lmax >= LMAX_ON_GPU:
            self.g_shls = l_slices[LMAX_ON_GPU:LMAX_ON_GPU + 2].tolist()
        else:
            self.g_shls = []
        if lmax > LMAX_ON_GPU:
            self.h_shls = l_slices[LMAX_ON_GPU + 1:].tolist()
        else:
            self.h_shls = []

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        q_cond = self.get_q_cond()
        cput1 = logger.timer(mol, 'Initialize q_cond', *cput0)
        log_qs = []
        pair2bra = []
        pair2ket = []
        l_ctr_offsets = numpy.append(0, numpy.cumsum(l_ctr_counts))
        for i, (p0, p1) in enumerate(
                zip(l_ctr_offsets[:-1], l_ctr_offsets[1:])):
            if uniq_l_ctr[i, 0] > LMAX_ON_GPU:
                # no integrals with g functions should be evaluated on GPU
                continue

            for q0, q1 in zip(l_ctr_offsets[:i], l_ctr_offsets[1:i + 1]):
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
            pair2bra.append(ishs)
            pair2ket.append(jshs)

            log_q = numpy.log(q_sorted[mask])
            log_q[log_q > 0] = 0
            log_qs.append(log_q)

        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = l_ctr_offsets
        self.bas_pair2shls = numpy.hstack(
            pair2bra + pair2ket).astype(numpy.int32).reshape(2, -1)
        self.bas_pairs_locs = numpy.append(
            0, numpy.cumsum([x.size for x in pair2bra])).astype(numpy.int32)
        self.log_qs = log_qs

        ao_loc = mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)
        if diag_block_with_triu:
            scale_shellpair_diag = 1.
        else:
            scale_shellpair_diag = 0.5

        self.initBpcache(scale_shellpair_diag, ao_loc, ncptype, mol)
        logger.timer(mol, 'Initialize GPU cache', *cput1)
        return self

    def initBpcache(self, scale_shellpair_diag, ao_loc, ncptype, mol):
        nbas = mol.nbas
        bas_coords = numpy.empty((nbas * 3,))
        bpcache = ctypes.POINTER(BasisProdCache)()
        libgint.GINTinit_basis_prod_cpu(
            ctypes.byref(bpcache), ctypes.c_double(scale_shellpair_diag),
            bas_coords.ctypes.data_as(ctypes.c_void_p),
            self.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            self.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ncptype), mol._atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm), mol._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.nbas), mol._env.ctypes.data_as(ctypes.c_void_p))
        aexyz = numpy.ctypeslib.as_array(
            bpcache.contents.aexyz,
            (bpcache.contents.primitive_pairs_locs[ncptype] * 5,))

        self.memory = Mg.broadcast(ao_loc, bas_coords, aexyz,
                                   self.bas_pair2shls)
        self.bpcaches = [None] * Mg.ngpu
        for i in range(Mg.ngpu):
            self.bpcaches[i] = ctypes.POINTER(BasisProdCache)()
            libgint.GINTinit_basis_prod_gpu(
                ctypes.byref(bpcache), ctypes.byref(self.bpcaches[i]),
                ctypes.cast(self.memory[0][i].data.ptr, ctypes.c_void_p),
                ctypes.cast(self.memory[1][i].data.ptr, ctypes.c_void_p),
                ctypes.cast(self.memory[2][i].data.ptr, ctypes.c_void_p),
                ctypes.cast(self.memory[3][i].data.ptr, ctypes.c_void_p))
        self.bpcache = self.bpcaches[0]

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


def basis_seg_contraction(mol, allow_replica=False):
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
                    coeff.append(numpy.eye(nf))
                    continue

                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[gto.NPRIM_OF]
                pcoeff = shell[gto.PTR_COEFF]
                if allow_replica:
                    coeff.extend([numpy.eye(nf)] * nctr)
                    bs = numpy.repeat(shell[numpy.newaxis], nctr, axis=0)
                    bs[:, gto.NCTR_OF] = 1
                    bs[:, gto.PTR_COEFF] = numpy.arange(
                        pcoeff, pcoeff + nprim * nctr, nprim)
                    bas_of_ia.append(bs)
                else:
                    pexp = shell[gto.PTR_EXP]
                    exps = _env[pexp:pexp + nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    c = _env[pcoeff:pcoeff + nprim * nctr].reshape(nctr, nprim)
                    c = numpy.einsum(
                        'ip,p,ef->iepf', c, 1 / norm, numpy.eye(nf))
                    coeff.append(c.reshape(nf * nctr, nf * nprim).T)

                    _env[pcoeff:pcoeff + nprim] = norm
                    bs = numpy.repeat(shell[numpy.newaxis], nprim, axis=0)
                    bs[:, gto.NPRIM_OF] = 1
                    bs[:, gto.NCTR_OF] = 1
                    bs[:, gto.PTR_EXP] = numpy.arange(pexp, pexp + nprim)
                    bs[:, gto.PTR_COEFF] = numpy.arange(pcoeff, pcoeff + nprim)
                    bas_of_ia.append(bs)

            bas_of_ia = numpy.vstack(bas_of_ia)
            bas_templates[key] = (bas_of_ia, coeff)

        _bas.append(bas_of_ia)
        contr_coeff.extend(coeff)

    pmol = copy.copy(mol)
    pmol.cart = True
    pmol._bas = numpy.asarray(numpy.vstack(_bas), dtype=numpy.int32)
    pmol._env = _env
    if mol.cart:
        contr_coeff = scipy.linalg.block_diag(*contr_coeff)
    else:
        c2s_coeff = mol.cart2sph_coeff()
        dtype = contr_coeff[0].dtype
        cols = numpy.cumsum([0] + [c.shape[0] for c in contr_coeff])
        rows = numpy.cumsum([0] + [c.shape[1] for c in contr_coeff])
        contr_coeffs = contr_coeff
        contr_coeff = numpy.empty((cols[-1], c2s_coeff.shape[1]), dtype=dtype)
        for i in range(len(contr_coeffs)):
            contr_coeff[cols[i]:cols[i + 1], :] = contr_coeffs[i].dot(
                c2s_coeff[rows[i]:rows[i + 1], :])
    return pmol, contr_coeff


def _make_s_index_offsets(log_q, nbins=10, cutoff=1e-12):
    '''Divides the shell pairs to "nbins" collections down to "cutoff"'''
    scale = nbins / numpy.log(min(cutoff, .1))
    s_index = numpy.floor(scale * log_q).astype(numpy.int32)
    bins = numpy.bincount(s_index)
    if bins.size < nbins:
        bins = numpy.append(bins, numpy.zeros(nbins - bins.size,
                                              dtype=numpy.int32))
    else:
        bins = bins[:nbins]
    assert bins.max() < 65536 * 8
    return numpy.append(0, numpy.cumsum(bins)).astype(numpy.int32)


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
        if l > LMAX_ON_GPU or counts <= max_shells:
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
