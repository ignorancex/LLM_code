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
# This project includes code adapted from PySCF (https://github.com/pyscf/pyscf,
# https://github.com/hongzhouye/pyscf/tree/rsdf_direct),
# which is licensed under the Apache License 2.0. The original copyright:
#     Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
#     Author: Qiming Sun <osirpt.sun@gmail.com>
#     Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import gamma_point, member
import numpy
import scipy
import cupy
from pyscf import __config__
from byteqc.lib import empty_pinned, multigpu, contraction, gemm, free_all_blocks
from byteqc.cupbc import lib as cupbclib
from pyscf.pbc.df.df_jk import (_format_dms, _format_jks, _format_kpts_band,
                                _ewald_exxdiv_for_G0)
from byteqc.cupbc.df.rsdf_direct_helper import (
    get_kptij_lst, loop_uniq_q, get_tril_indices)
from byteqc.cupbc.df.int3c import Int3cCal

libgpbc = cupbclib.load_library('libgpbc')

EIGH_DM_THRESH = getattr(
    __config__, 'pbc_gto_df_rsdf_jk_direct_eigh_dm_thresh', 1e-10)
REAL = numpy.float64
COMPLEX = numpy.complex128


def _safe_member(q, qs):
    idxs = member(q, qs)
    if len(idxs) != 1:
        raise RuntimeError
    return idxs[0]


def _balance_blksize(ntot, nblksize):
    nblk = ntot // nblksize + (ntot % nblksize > 0)
    nblksize = ntot // nblk + (ntot % nblk > 0)
    return nblksize


def _eigh_rdm1(dm_kpts, thr_nonzero=EIGH_DM_THRESH):
    assert (dm_kpts.ndim == 3)
    nkpts = len(dm_kpts)
    mo_coeff = [None] * nkpts
    mo_occ = [None] * nkpts
    for k, dm in enumerate(dm_kpts):
        e, u = scipy.linalg.eigh(dm)
        if numpy.any(e < -thr_nonzero):
            raise RuntimeError('Input dm is not PSD.')
        idx1 = numpy.where(e > thr_nonzero)[0]
        idx2 = numpy.asarray(
            [i for i in range(len(e)) if i not in idx1], dtype=int)
        mo_occ[k] = numpy.concatenate([e[idx1], numpy.zeros_like(e[idx2])])
        mo_coeff[k] = numpy.hstack([u[:, idx1], u[:, idx2]])
    return mo_coeff, mo_occ


def get_j_kpts(
        mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)),
        kpts_band=None, bvk_kmesh=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    t1 = (logger.process_clock(), logger.perf_counter())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts = dms.shape[:2]

    if mydf.auxcell is None:
        mydf.build()
    mydf.int3c.config(kpts, 'j')
    vhfopt = mydf.int3c.vhfopt
    nao_cart = vhfopt.mol.nao_nr(cart=True)
    naux = mydf.auxcell.nao_nr()
    naux_cart = mydf.int3c.auxvhfopt.mol.nao_nr(cart=True)
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if isinstance(mydf.int3c, Int3cCal):
        # dms vj coeff auxcoeff
        mem_c = nset * nkpts * nao_cart * nao_cart + \
            nset * nband * nao_cart * nao_cart + \
            nao_cart * mydf.cell.nao + naux * naux_cart
        mem_c *= 8 if gamma_point(kpts_band) else 16
        mydf.int3c.build(bvk_kmesh, mem_c=mem_c)
        naux3c = naux_cart
    else:
        mydf.int3c.build(bvk_kmesh, auxflag=2)
        naux3c = naux

    j_real = gamma_point(kpts_band) and not numpy.iscomplexobj(dms)
    vj_cupy = multigpu.Mg.mapgpu(
        lambda: cupy.zeros(
            (nset, nband, nao_cart, nao_cart),
            dtype=COMPLEX if not j_real else REAL))
    coeff_cupy = cupy.asarray(vhfopt.coeff)

    rho_cupy = multigpu.Mg.mapgpu(lambda: cupy.zeros(
        (nset, naux3c), dtype=numpy.float64 if j_real else numpy.complex128))
    dms_cupy = cupy.ndarray((nset, nkpts, nao_cart, nao_cart), dtype=dms.dtype)
    tmp = contraction("pi", coeff_cupy, "nkij", cupy.asarray(dms), "nkpj")
    contraction("nkpj", tmp, "qj", coeff_cupy, "nkqp", dms_cupy)
    tmp = None
    dms_cupy = multigpu.Mg.broadcast(dms_cupy)

    l_ctr_offsets = vhfopt.l_ctr_offsets
    ao_loc = vhfopt.ao_loc
    ncptype_ij = vhfopt.ncptype

    def j_pass1(cp_ij_id):
        t2 = (logger.process_clock(), logger.perf_counter())
        kcpqL_cupy = mydf.int3c(cp_ij_id)
        t2 = log.timer_debug1(
            'get_j pass1[%d/%d] intj3c' % (cp_ij_id, ncptype_ij), *t2)

        t2 = (logger.process_clock(), logger.perf_counter())
        cpi, cpj = get_tril_indices(cp_ij_id)
        ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
        jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]

        kcpqL_cupy = kcpqL_cupy.reshape((nkpts, 1, i1 - i0, j1 - j0, naux3c))
        gid = multigpu.Mg.getgid()
        aosymm = False
        if i0 != j0:
            aosymm = True
        contraction(
            "nkij", dms_cupy[gid][:, :, i0: i1, j0: j1],
            "kijL", kcpqL_cupy[:, 0],
            "nL", rho_cupy[gid],
            alpha=1, beta=1)
        if (aosymm):
            contraction(
                "nkij", dms_cupy[gid][:, :, j0: j1, i0: i1],
                "kjiL", kcpqL_cupy[:, 0],
                "nL", rho_cupy[gid],
                alpha=1, beta=1, opb='CONJ')
        kcpqL_cupy = None
        t2 = log.timer_debug1(
            'get_j pass1[%d/%d] get_rho' % (cp_ij_id, ncptype_ij), *t2)

    multigpu.Mg.map(j_pass1, range(ncptype_ij))
    dms_cupy = None
    multigpu.Mg.mapgpu(free_all_blocks)
    if isinstance(mydf.int3c, Int3cCal):
        multigpu.Mg.sum(rho_cupy, coeff=1. / nkpts)
        auxcoeffs = multigpu.Mg.broadcast(mydf.int3c.j2c_coeff)

        def task_rho(rho, coeff):
            rho = rho @ coeff @ coeff.T
            return rho

        rho_cupy = multigpu.Mg.mapgpu(task_rho, rho_cupy, auxcoeffs)
        auxcoeffs = None
        t1 = log.timer_debug1('get_j solve j2c', *t1)
    else:
        multigpu.Mg.sum(rho_cupy, coeff=1. / nkpts)

    def j_pass2(cp_ij_id):
        t2 = (logger.process_clock(), logger.perf_counter())
        kcpqL_cupy = mydf.int3c(cp_ij_id)
        t2 = log.timer_debug1(
            'get_j pass2[%d/%d] intj3c' % (cp_ij_id, ncptype_ij), *t2)
        cpi, cpj = get_tril_indices(cp_ij_id)
        ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
        jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]

        gid = multigpu.Mg.getgid()
        aosymm = False
        if i0 != j0:
            aosymm = True
        contraction(
            "kpqL", kcpqL_cupy.reshape((nkpts, i1 - i0, j1 - j0, naux3c)),
            "nL", rho_cupy[gid],
            "nkpq", vj_cupy[gid][:, :, i0: i1, j0: j1],
            alpha=1, beta=1)
        if (aosymm):
            contraction(
                "kqpL", kcpqL_cupy.reshape(
                    (nkpts, i1 - i0, j1 - j0, naux3c)),
                "nL", rho_cupy[gid],
                "nkpq", vj_cupy[gid][:, :, j0: j1, i0: i1],
                alpha=1, beta=1, opa='CONJ')
        kcpqL_cupy = None
        t2 = log.timer_debug1(
            'get_j pass2[%d/%d]' % (cp_ij_id, ncptype_ij), *t2)
    multigpu.Mg.mapgpu(free_all_blocks)
    multigpu.Mg.map(j_pass2, range(ncptype_ij))
    mydf.int3c.clean()
    vj_cupy = multigpu.Mg.sum(vj_cupy)
    rho_cupy = None
    multigpu.Mg.mapgpu(free_all_blocks)
    t1 = log.timer_debug1('get_j pass2', *t1)

    vj_cupy = contraction('ip', coeff_cupy, 'nkij', vj_cupy, 'nkpj')
    vj_cupy = contraction('jq', coeff_cupy, 'nkpj', vj_cupy, 'nkpq')
    vj = vj_cupy.get()
    vj_cupy = coeff_cupy = None
    multigpu.Mg.mapgpu(free_all_blocks)
    t1 = log.timer_debug1('get_j get_vj', *t1)
    t0 = log.timer_debug1('get_j', *t0)
    return _format_jks(vj, dm_kpts, input_band, kpts)


def _get_k_kpts(mydf, skmo, kpts, isreal, bvk_kmesh=[None, None]):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    t1 = (logger.process_clock(), logger.perf_counter())

    datatype = REAL if isreal else COMPLEX
    datasize = 8 if isreal else 16

    if mydf.auxcell is None:
        mydf.build()
    mydf.int3c.config(kpts, 'k')
    vhfopt = mydf.int3c.vhfopt
    nset = len(skmo)
    nkpts = len(kpts)
    nband = nkpts
    naux = mydf.auxcell.nao_nr()
    naux_cart = mydf.auxcell.nao_nr(cart=True)
    nao_cart = vhfopt.mol.nao_nr(cart=True)

    kptij_lst = get_kptij_lst(kpts, ksym='s1')
    nkptij = len(kptij_lst)
    uniq_q_loop = [x for x in loop_uniq_q(
        mydf, kptij_lst=kptij_lst, verbose=0)]

    # buffer for kAiX
    nmomax = numpy.max([mo.shape[1] for kmo in skmo for mo in kmo])
    mem_avail = (mydf.max_memory - lib.current_memory()[0]) * 0.9
    kAiXsize = nset * nkptij * naux * nao_cart
    mem_kAiX = kAiXsize * datasize / (1024**2)
    nmoblksize = min(nmomax, int(numpy.floor(mem_avail / mem_kAiX)))
    nmoblksize = _balance_blksize(nmomax, nmoblksize)
    log.debug1("Slicing %d into %d with %.2fMB for get_k CPU memory" %
               (nmomax, nmoblksize, mem_avail))
    if nmoblksize < 1:
        mem_need = mem_kAiX
        log.error(
            'Caching (L|[p]q) and (L|p[i]) needs at least %.2fMB of memory, '
            'which exceeds the available memory %.2fMB', mem_need, mem_avail)
        raise MemoryError
    buf_kAiX = empty_pinned(
        (nset, nkptij, nao_cart, nmoblksize * naux), dtype=datatype)

    nkpts_uniq = len(uniq_q_loop)
    if isinstance(mydf.int3c, Int3cCal):
        # skmo j2c_coeff
        mem_c = (nset * nband * nao_cart**2
                 + nkpts_uniq * naux * naux_cart) * datasize
        mem_b = nset * (nkptij + 1) * nmoblksize * naux * 2 * datasize
        mydf.int3c.build(bvk_kmesh, mem_b=mem_b, mem_c=mem_c)
    else:
        mydf.int3c.build(bvk_kmesh, auxflag=0)
    auxcoeff = multigpu.Mg.broadcast(mydf.int3c.j2c_coeff)
    naux3c = naux_cart

    skmo = cupy.asarray(lib.einsum("pq,skqo->skpo", vhfopt.coeff, skmo))
    skmo = multigpu.Mg.broadcast(skmo)

    vk = cupy.zeros((nset, nband, nao_cart, nao_cart), dtype=datatype)
    t1 = log.timer_debug1('get_k preperation skmo', *t1)

    l_ctr_offsets = vhfopt.l_ctr_offsets
    ao_loc = vhfopt.ao_loc
    ncptype_ij = vhfopt.ncptype

    lock = multigpu.LabelLock()
    t1 = log.timer_debug1('get_k preperation j3c', *t1)

    for i0, i1 in lib.prange(0, nmomax, nmoblksize):
        di = i1 - i0
        kAiX = numpy.ndarray((nset, nkptij, nao_cart, di * naux),
                             dtype=datatype, buffer=buf_kAiX)
        kAiX[:] = 0

        def k_pass1(cp_ij_id):
            t2 = (logger.process_clock(), logger.perf_counter())
            # Step 1:Get the W from the
            # formula:\sum_{sigma}V_{lambda,sigma,L}C_{sigma,i}->W_{lambda,i}
            kcpqL_cupy = mydf.int3c(cp_ij_id)
            t2 = log.timer_debug1(
                "get_k[%d-%d] pass1[%d] intj3c" % (i0, i1, cp_ij_id), *t2)
            cpi, cpj = get_tril_indices(cp_ij_id)
            ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
            jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
            p0, p1 = ao_loc[ish0], ao_loc[ish1]
            q0, q1 = ao_loc[jsh0], ao_loc[jsh1]
            dp = p1 - p0
            dq = q1 - q0
            aosymm = p0 != q0
            # memory need (nkpts**2 * blksize**2 * naux）+ (nset * nkpts**2 *
            # blksize * nmomax * naux) + (len(uniq_kpts) * naux**2)
            kcpqL_cupy = kcpqL_cupy.reshape(
                (nkpts * nkpts, 1, dp, dq, naux3c))
            kpiX_cupy = None
            kqiX_cupy = cupy.zeros(
                (nset, nkptij, q1 - q0, di * naux), dtype=datatype)
            ji_all = numpy.concatenate([elem[2] for elem in uniq_q_loop])
            ji_all_index = -1
            kq = -1

            gid = multigpu.Mg.getgid()
            for kpt, adapted_kptjs, adapted_ji_idx in uniq_q_loop:
                kq += 1
                for kptj, ji in zip(adapted_kptjs, adapted_ji_idx):
                    ji_all_index += 1
                    ki = _safe_member(kptj - kpt, kpts)
                    for iset in range(nset):
                        mo = skmo[gid][iset][ki][p0:p1, i0:i1]
                        qiX = contraction(
                            'pi', mo, 'pqL',
                            kcpqL_cupy[ji_all[ji_all_index], 0], 'qiL',
                            opa='CONJ').reshape(-1, naux_cart)
                        gemm(
                            qiX, auxcoeff[gid][kq],
                            kqiX_cupy[iset, ji].reshape(dq * di, -1),
                            alpha=1.0, beta=1.0)
                        qiX = None

            with lock((q0, q1)):
                kAiX[:, :, q0:q1] += kqiX_cupy.get(
                    out=numpy.ndarray(kqiX_cupy.shape, dtype=kqiX_cupy.dtype))
            kqiX_cupy = None
            if (aosymm):
                kpiX_cupy = cupy.zeros(
                    (nset, nkptij, p1 - p0, di * naux), dtype=datatype)
                ji_all_index = -1
                kq = -1
                for kpt, adapted_kptjs, adapted_ji_idx in uniq_q_loop:
                    kq += 1
                    for kptj, ji in zip(adapted_kptjs, adapted_ji_idx):
                        ji_all_index += 1
                        ki = _safe_member(kptj - kpt, kpts)

                        k1 = ji_all[ji_all_index] // nkpts
                        k2 = ji_all[ji_all_index] % nkpts
                        k12 = k2 * nkpts + k1

                        for iset in range(nset):
                            mo = skmo[gid][iset][ki][q0:q1, i0:i1]
                            piX = contraction(
                                'qi', mo, 'pqL', kcpqL_cupy[k12, 0],
                                'piL', opa='CONJ', opb='CONJ').reshape(
                                -1, naux_cart)
                            gemm(
                                piX, auxcoeff[gid][kq],
                                kpiX_cupy[iset, ji].reshape(dp * di, -1),
                                alpha=1.0, beta=1.0)
                            piX = None

                with lock((p0, p1)):
                    kAiX[:, :, p0:p1] += kpiX_cupy.get(out=numpy.ndarray(
                        kpiX_cupy.shape, dtype=kpiX_cupy.dtype))
            kpiX_cupy = None
            kcpqL_cupy = mo = None
            t2 = log.timer_debug1(
                "get_k[%d-%d] pass1[%d] get kAiX" % (i0, i1, cp_ij_id), *t2)
        multigpu.Mg.mapgpu(free_all_blocks)
        multigpu.Mg.map(k_pass1, range(ncptype_ij))
        if (i1 == nmomax):
            mydf.int3c.clean()
        multigpu.Mg.mapgpu(free_all_blocks)
        cupy.cuda.profiler.stop()
        t1 = log.timer_debug1('get_k[%d-%d] pass1' % (i0, i1), *t1)

        for _, adapted_kptjs, adapted_ji_idx in uniq_q_loop:
            for kptj, ji in zip(adapted_kptjs, adapted_ji_idx):
                kj = _safe_member(kptj, kpts)
                for iset in range(nset):
                    for p0, p1 in lib.prange(0, nao_cart, mydf.int3c.blksize):
                        piX = cupy.asarray(kAiX[iset, ji, p0:p1])
                        for q0, q1 in lib.prange(
                                0, nao_cart, mydf.int3c.blksize):
                            if p0 > q0:
                                continue
                            if p0 == q0:
                                qiX = piX
                            else:
                                qiX = cupy.asarray(kAiX[iset, ji, q0:q1])
                            contraction(
                                'ab', piX, 'cb', qiX, 'ac',
                                vk[iset, kj, p0:p1, q0:q1],
                                beta=1, opa='CONJ')
                            if p0 != q0:
                                contraction(
                                    'ab', piX, 'cb', qiX, 'ca',
                                    vk[iset, kj, q0:q1, p0:p1],
                                    beta=1, opb='CONJ')
                        piX = qiX = None

        t1 = log.timer_debug1('get_k[%d-%d] pass2' % (i0, i1), *t1)
    skmo = None
    multigpu.Mg.mapgpu(free_all_blocks)

    coeff = cupy.asarray(vhfopt.coeff)
    vk = contraction('ip', coeff, 'skij', vk, 'skpj')
    vk = contraction('skpj', vk, 'jq', coeff, 'skpq', alpha=1. / nkpts).get()
    coeff = None
    multigpu.Mg.mapgpu(free_all_blocks)
    t1 = log.timer_debug1('get_j get_vk', *t1)
    t0 = log.timer_debug1('get_k_kpts', *t0)
    return vk


def get_k_kpts(
        mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)),
        kpts_band=None, exxdiv=None, bvk_kmesh=None, semidirect=False,
        ksym='s1'):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('RSDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('RSGDF does not support exxdiv %s' % exxdiv)

    if kpts_band is not None:
        log.warn('RSDF get_k_kpts for band calculations is not implemented.')
        raise NotImplementedError

    kpts = lib.asarray(kpts).reshape(-1, 3)
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    if mydf.auxcell is None:
        mydf.build(with_j3c=False)

# set up mo_coeff, mo_occ
    dm_kpts_ = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts_, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ = dm_kpts.mo_occ
        if nset == 1 and len(mo_coeff) == nkpts and len(mo_coeff[0]) == nao:
            mo_coeff = [mo_coeff]
            mo_occ = [mo_occ]
        elif not (len(mo_coeff) == nset and len(mo_coeff[0]) == nkpts
                  and len(mo_occ) == nset and len(mo_occ[0]) == nkpts):
            log.error('Input mo_coeff or mo_occ has wrong dim.')
            raise ValueError
        log.debug1('Input mo_coeff and mo_occ found via tagged dm_kpts.')
    else:
        log.debug1('Diagonalizing dm_kpts to generate mo_coeff and mo_occ')
        xs = [_eigh_rdm1(dms[i]) for i in range(nset)]
        mo_coeff = [x[0] for x in xs]
        mo_occ = [x[1] for x in xs]
        xs = None

    skmo = [_format_mo_coeff(mo_coeff[i], mo_occ[i], order='F')
            for i in range(nset)]
    mo_isreal = skmo[0][0].dtype == REAL
    j3c_isreal = gamma_point(kpts) and gamma_point(kpts_band)
    if mo_isreal and j3c_isreal and nkpts == 1:  # gamma point
        # smo = [kmoR[0] for kmoR in skmo]
        if semidirect:
            raise NotImplementedError
        else:
            vk_kpts = _get_k_kpts(mydf, skmo, kpts, True)
    else:
        if ksym == 's1':
            if semidirect:
                raise NotImplementedError
            else:
                fgetk = lambda *args, **kwargs: _get_k_kpts(
                    *args, isreal=False, **kwargs)
        else:
            if semidirect:
                raise NotImplementedError
            else:
                raise NotImplementedError
        log.debug1('Using kernel %s for K-build', fgetk)
        vk_kpts = fgetk(mydf, skmo, kpts, bvk_kmesh=bvk_kmesh)
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)
    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


def get_j(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpts_band=None):
    kpts = numpy.asarray(kpt).reshape(1, 3)
    dms = numpy.asarray(dm)
    vjs = get_j_kpts(mydf, dm, hermi=hermi, kpts=kpts, kpts_band=kpts_band,
                     bvk_kmesh=None)
    if kpts_band is None:
        vjs = vjs.reshape(dms.shape)
    return vjs


def get_k(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpts_band=None, exxdiv=None,
          bvk_kmesh=None, semidirect=False, ksym='s1'):
    kpts = numpy.asarray(kpt).reshape(1, 3)
    dms = numpy.asarray(dm)
    vks = get_k_kpts(
        mydf, dm, hermi=hermi, kpts=kpts, kpts_band=kpts_band, exxdiv=exxdiv,
        bvk_kmesh=bvk_kmesh, semidirect=semidirect, ksym=ksym)
    if kpts_band is None:
        vks = vks.reshape(dms.shape)
    return vks


def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpts_band=None, exxdiv=None,
           with_j=True, with_k=True, bvk_kmesh=None, semidirect=False):
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi=hermi, kpt=kpt, kpts_band=kpts_band)
    if with_k:
        vk = get_k(mydf, dm, hermi=hermi, kpt=kpt, kpts_band=kpts_band,
                   exxdiv=exxdiv, bvk_kmesh=bvk_kmesh, semidirect=semidirect)
    return vj, vk


def _format_mo_coeff(mo_coeff, mo_occ, order='C'):
    nkpts = len(mo_coeff)
    nao = mo_coeff[0].shape[0]
    # padding mo_coeff using the maximum nmo of all kpts
    nmo = numpy.max([numpy.count_nonzero(occ > 0)
                    for k, occ in enumerate(mo_occ)])
    kmo = [numpy.zeros((nao, nmo), dtype=mo_coeff[0].dtype, order=order)
           for k in range(nkpts)]
    for k, occ in enumerate(mo_occ):
        mask = occ > 0
        nmok = numpy.count_nonzero(mask)
        mo = mo_coeff[k][:, mask] * numpy.sqrt(occ[mask])
        kmo[k][:, :nmok] = mo
    return kmo
