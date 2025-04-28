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
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf),
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
#

from byteqc.cucc import culib
import numpy
import cupy
from pyscf.lib import logger, prange
from pyscf.cc.ccsd import BLKMIN
from byteqc import lib


def make_rdm1(mycc, t1, t2, l1, l2, ao_repr=False):
    r'''
    Spin-traced one-particle density matrix in MO basis (the occupied-virtual
    blocks from the orbital response contribution are not included).

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)


def _gamma1_intermediates(mycc, t1, t2, l1, l2):
    pool = mycc.pool
    nocc, nvir = t1.shape
    pool.memory_status(nocc, nvir, which=2)

    doo = pool.empty((nocc, nocc), 'f8')
    culib.contraction('ja', t1, 'ia', l1, 'ij', doo, alpha=-1.0)
    dvv = pool.empty((nvir, nvir), 'f8')
    culib.contraction('ia', t1, 'ib', l1, 'ab', dvv)
    xtv = pool.empty((nocc, nocc), 'f8')
    culib.contraction('ie', t1, 'me', l1, 'im', xtv)
    dvo = pool.empty((nvir, nocc), 'f8')
    culib.contraction('im', xtv, 'ma', t1, 'ai', dvo, alpha=-1.0)
    xtv = None
    lib.elementwise_binary('ba', t1, 'ab', dvo, gamma=1.0)

    memory = pool.free_memory
    blksize = min(nocc, int(memory / 8 / nocc / nvir**2 / 2))

    buf = lib.ArrayBuffer(
        pool.empty(
            (2 * blksize * nocc * nvir**2 + 1024),
            'f8'))
    buf1 = buf.empty((blksize, nocc, nvir, nvir), 'f8')
    buf2 = buf.empty((blksize, nocc, nvir, nvir), 'f8')

    theta = pool.new('theta', (nocc, nocc, nvir, nvir), 'f8')

    for p0, p1 in prange(0, nocc, blksize):
        t = theta[p0:p1]
        arr = t.enter(buf=buf1)
        t2s = t2[p0:p1].ascupy(buf=buf2)
        lib.elementwise_binary('abdc', t2s, 'abcd', t2s, arr,
                               alpha=-1.0, gamma=2.0)
        t.exit()
    buf = buf1 = buf2 = t2s = arr = None

    culib.contraction('jkab', theta, 'ikab', l2, 'ij', doo,
                      alpha=-1.0, beta=1.0)
    culib.contraction('jica', theta, 'jicb', l2, 'ab', dvv, beta=1.0)
    culib.contraction('imae', theta, 'me', l1, 'ai', dvo, beta=1.0)
    xt1 = pool.empty((nocc, nocc), 'f8')
    culib.contraction('mnef', l2, 'inef', theta, 'mi', xt1)
    culib.contraction('mi', xt1, 'ma', t1, 'ai', dvo,
                      alpha=-1.0, beta=1.0)
    xt1 = None
    xt2 = pool.empty((nvir, nvir), 'f8')
    culib.contraction('mnaf', l2, 'mnef', theta, 'ea', xt2)
    theta = None
    culib.contraction('ie', t1, 'ae', xt2, 'ai', dvo,
                      alpha=-1.0, beta=1.0)
    xt2 = None

    lib.free_all_blocks()
    dov = l1
    return doo, dov, dvo, dvv


def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False):
    r'''dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    pool = mycc.pool
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir
    dm1 = pool.empty((nmo, nmo), dtype=doo.dtype)
    dm1[:nocc, :nocc] = doo
    dm1[:nocc, :nocc] += doo.conj().T
    dm1[:nocc, nocc:] = dov
    dm1[:nocc, nocc:] += dvo.conj().T
    dm1[nocc:, :nocc] = dm1[:nocc, nocc:].conj().T
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, nocc:] += dvv.conj().T
    ker = cupy.ElementwiseKernel(
        'int64 nd', 'raw T out', '''
            out[i * nd + i] += 2.0;
        ''', 'dm1_add_diag_2')
    ker(nmo, dm1, size=nocc)

    if with_frozen and mycc.frozen is not None:
        assert False, 'not implement'

    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = culib.contraction('pi', mo, 'ij', dm1, 'pj')
        dm1 = culib.contraction('pj', dm1, 'qj', mo, 'pq', opb='CONJ')
    return dm1


def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s]=\sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = _gamma2(mycc, t1, t2, l1, l2)
    return _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True,
                      ao_repr=ao_repr)


def _gamma2(mycc, t1, t2, l1, l2):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    pool = mycc.pool
    pool.memory_status(nocc, nvir, which=3)

    dtype = cupy.result_type(t1, t2, l1, l2).char

    dvvvv = pool.new('dvvvv', (nvir, nvir, nvir, nvir), dtype)
    dovvo = pool.new('dovvo', (nocc, nvir, nvir, nocc), dtype)

    fswap = {}
    one = cupy.array(1.0, 'f8')
    time1 = logger.process_clock(), logger.perf_counter()

    mvOOv = pool.new('mvOOv', (nvir, nocc, nocc, nvir), 'f8')
    culib.contraction('ikca', l2, 'jkcb', t2, 'aijb', mvOOv)
    moo = pool.empty((nocc, nocc), 'f8')
    culib.contraction('dljd', mvOOv, '', one, 'jl', moo, alpha=2.0)
    mvv = pool.empty((nvir, nvir), 'f8')
    culib.contraction('blld', mvOOv, '', one, 'db', mvv, alpha=2.0)

    gooov = culib.contraction('kc', t1, 'cija', mvOOv, 'jkia')

    pvoOV = culib.contraction('ikca', l2, 'jkbc', t2, 'aijb', alpha=-1)
    theta = t2.ascupy() * 2
    theta -= t2.transpose(0, 1, 3, 2)
    pvoOV = culib.contraction(
        'ikac', l2, 'jkbc', theta, 'aijb', pvoOV, beta=1.0)
    moo += cupy.einsum('dljd->jl', pvoOV)
    mvv += cupy.einsum('blld->db', pvoOV)
    gooov = culib.contraction('jc', t1, 'cika', pvoOV, 'jkia', gooov,
                              alpha=-1.0, beta=1.0)
    fswap['mvoOV'] = pool.add('mvoOV', pvoOV)
    pvoOV = None

    mia = culib.contraction('kc', l1, 'ikac', t2, 'ia', alpha=2)
    mia = culib.contraction('kc', l1, 'ikca', t2, 'ia', mia,
                            alpha=-1.0, beta=1.0)
    mab = culib.contraction('kc', l1, 'kb', t1, 'cb')
    mij = culib.contraction('kc', l1, 'jc', t1, 'jk')
    mij += moo * .5

    tau = culib.contraction('ia', t1, 'jb', t1, 'ijab')
    tau += t2.ascupy()
    goooo = culib.contraction('ijab', tau, 'klab', l2, 'ijkl', alpha=0.5)
    tmp = goooo.transpose(0, 2, 1, 3).conj() * 2
    tmp -= goooo.transpose(0, 3, 1, 2).conj()
    doooo = pool.add('doooo', tmp)

    gooov = culib.contraction('ji', moo, 'ka', t1,
                              'jkia', gooov, alpha=-0.5, beta=1)
    gooov = culib.contraction(
        'la', t1, 'jkil', goooo, 'jkia', gooov, alpha=2, beta=1)
    gooov = culib.contraction('ib', l1, 'jkba', tau,
                              'jkia', gooov, alpha=-1, beta=1)
    gooov = culib.contraction('jkba', l2, 'ib', t1,
                              'jkia', gooov, alpha=-1, beta=1, opc='CONJ')
    tmp = gooov.transpose(0, 2, 1, 3) * 2
    tmp -= gooov.transpose(1, 2, 0, 3)
    dooov = pool.add('dooov', tmp)
    tau = gooov = None

    time1 = log.timer_debug1('rdm intermediates pass1', *time1)
    goovv = culib.contraction('ia', mia, 'jb', t1,
                              'ijab', opa='CONJ', opb='CONJ')
    max_memory = max(0, mycc.max_memory - culib.current_memory()[0])
    unit = nocc**2 * nvir * 6
    blksize = min(nocc, nvir, max(BLKMIN, int(
        max_memory * 1e6 / 8 / unit)))
    doovv = pool.new('doovv', (nocc, nocc, nvir, nvir), dtype)

    log.debug1(
        'rdm intermediates pass 2: block size = %d, nvir = %d in %d blocks',
        blksize, nvir, int((nvir + blksize - 1) / blksize))
    for p0, p1 in prange(0, nvir, blksize):
        tau = culib.contraction('ia', t1[:, p0:p1], 'jb', t1, 'ijab')
        tau += t2[:, :, p0:p1].ascupy()
        tmpoovv = culib.contraction('ijkl', goooo, 'klab', tau, 'ijab')
        tmpoovv = culib.contraction(
            'jk', mij, 'ikab', tau, 'ijab', tmpoovv, alpha=-1, beta=1)
        tmpoovv = culib.contraction(
            'cb', mab, 'ijac', t2[:, :, p0:p1],
            'ijab', tmpoovv, alpha=-1, beta=1)
        tmpoovv = culib.contraction(
            'bd', mvv, 'ijad', tau, 'ijab', tmpoovv, alpha=-0.5, beta=1)
        tmpoovv *= 2
        tmpoovv += tau
        tmpoovv *= 0.5
        tmpoovv = tmpoovv.conj()
        tmpoovv *= 2
        tmpoovv += l2[:, :, p0:p1].ascupy()
        tmpoovv *= 0.5
        goovv[:, :, p0:p1] += tmpoovv
        tmpoovv = None

        pvOOv = mvOOv[p0:p1]
        pvoOV = fswap['mvoOV'][p0:p1]
        gOvvO = cupy.einsum('kiac,jc,kb->iabj',
                            l2[:, :, p0:p1].ascupy(), t1.ascupy(), t1.ascupy())
        gOvvO += pvOOv.transpose(1, 0, 3, 2)
        govVO = culib.contraction(
            'ia', l1[:, p0:p1].ascupy(), 'jb', t1.ascupy(), 'iabj')
        govVO -= cupy.einsum(
            'ikac,jc,kb->iabj', l2[:, :, p0:p1].ascupy(), t1.ascupy(),
            t1.ascupy())
        govVO += pvoOV.transpose(1, 0, 3, 2)
        dovvo[:, p0:p1] = gOvvO
        dovvo[:, p0:p1] += 2 * govVO
        tmp = govVO.conj()
        tmp *= 0.5
        tmp += gOvvO.conj()
        tmp *= -2
        doovv[:, :, p0:p1] = tmp.transpose(3, 0, 1, 2)
        gOvvO = govVO = None

        tau -= t2[:, :, p0:p1].ascupy() * .5
        for q0, q1 in prange(0, nvir, blksize):
            goovv[:, :, q0:q1, :] += culib.contraction(
                'dlib', pvOOv.ascupy(), 'jlda', tau[:, :, :, q0:q1],
                'ijab').conj()
            goovv[:, :, :, q0:q1] -= culib.contraction(
                'dlia', pvoOV.ascupy(), 'jldb', tau[:, :, :, q0:q1],
                'ijab').conj()
            tmp = pvOOv[:, :, :, q0:q1].ascupy() * .5
            tmp += pvoOV[:, :, :, q0:q1].ascupy()
            goovv[:, :, q0:q1, :] += culib.contraction(
                'dlia', tmp, 'jlbd', t2[:, :, :, p0:p1].ascupy(),
                'ijab').conj()
        pvOOv = pvoOV = tau = None
        time1 = log.timer_debug1(
            'rdm intermediates pass2 [%d:%d]' % (p0, p1), *time1)
    tmp = goovv.transpose(0, 2, 1, 3) * 2
    tmp -= goovv.transpose(1, 2, 0, 3)
    dovov = pool.add('dovov', tmp)
    goovv = goooo = None

    max_memory = max(0, mycc.max_memory - culib.current_memory()[0])
    unit = max(nocc**2 * nvir * 2 + nocc * nvir**2 * 3,
               nvir**3 * 2 + nocc * nvir**2 * 2 + nocc**2 * nvir * 2)
    blksize = min(nvir, max(BLKMIN, int(
        max_memory * 1e6 / 8 / unit)))
    log.debug1(
        'rdm intermediates pass 3: block size = %d, nvir = %d in %d blocks',
        blksize, nocc, int((nvir + blksize - 1) / blksize))
    dovvv = pool.new('dovvv', (nocc, nvir, nvir, nvir), dtype)
    time1 = logger.process_clock(), logger.perf_counter()
    for istep, (p0, p1) in enumerate(prange(0, nvir, blksize)):
        l2tmp = l2[:, :, p0:p1].ascupy()
        gvvvv = culib.contraction('ijab', l2tmp, 'ijcd', t2.ascupy(), 'abcd')
        jabc = culib.contraction('ijab', l2tmp, 'ic', t1.ascupy(), 'jabc')
        gvvvv = culib.contraction(
            'jabc', jabc, 'jd', t1.ascupy(), 'abcd', gvvvv, beta=1)
        l2tmp = jabc = None

        for i in range(p0, p1):
            vvv = gvvvv[i - p0].conj()
            dvvvv[i] = vvv.transpose(1, 0, 2)
            dvvvv[i] *= 2
            dvvvv[i] -= vvv.transpose(2, 0, 1)
            dvvvv[i] *= 0.5

        gvovv = culib.contraction(
            'adbc', gvvvv, 'id', t1.ascupy(), 'aibc', alpha=-1)
        gvvvv = None

        gvovv = culib.contraction('akic', fswap['mvoOV'][p0:p1].ascupy(
        ), 'kb', t1.ascupy(), 'aibc', gvovv, beta=1)
        gvovv = culib.contraction('akib', mvOOv[p0:p1].ascupy(
        ), 'kc', t1.ascupy(), 'aibc', gvovv, alpha=-1, beta=1)

        gvovv = culib.contraction('ja', l1[:, p0:p1].ascupy(
        ), 'jibc', t2.ascupy(), 'aibc', gvovv, beta=1)
        gvovv += cupy.einsum('ja,jb,ic->aibc',
                             l1[:, p0:p1].ascupy(), t1.ascupy(), t1.ascupy())
        gvovv = culib.contraction('ba', mvv[:, p0:p1], 'ic', t1.ascupy(),
                                  'aibc', gvovv, alpha=0.5, beta=1)
        gvovv = gvovv.conj()
        gvovv = culib.contraction('ja', t1[:, p0:p1].ascupy(), 'jibc',
                                  l2.ascupy(), 'aibc', gvovv, beta=1)

        dovvv[:, :, p0:p1] = gvovv.transpose(1, 3, 0, 2) * 2
        dovvv[:, :, p0:p1] -= gvovv.transpose(1, 2, 0, 3)
        gvovv = None
        time1 = log.timer_debug1(
            'rdm intermediates pass3 [%d:%d]' % (p0, p1), *time1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv,
            dovvo, dvvov, dovvv, dooov)

# Note vvvv part of 2pdm have been symmetrized.  It does not correspond to
# vvvv part of CI 2pdm


def _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=False):
    r'''
    dm2[p,q,r,s]=\sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    dm2 = cupy.empty((nmo, nmo, nmo, nmo), dtype=doovv.dtype)

    dm2[:nocc, nocc:, :nocc, nocc:] = dovov.ascupy()
    dm2[:nocc, nocc:, :nocc, nocc:] += dovov.ascupy().transpose(2, 3, 0, 1)
    dovov = None
    dm2[nocc:, :nocc, nocc:, :nocc] = dm2[
        :nocc, nocc:, :nocc, nocc:].transpose(1, 0, 3, 2).conj()

    dm2[:nocc, :nocc, nocc:, nocc:] = doovv.ascupy()
    dm2[:nocc, :nocc, nocc:, nocc:] += doovv.ascupy().transpose(1,
                                                                0, 3, 2).conj()
    doovv = None
    dm2[nocc:, nocc:, :nocc, :nocc] = dm2[
        :nocc, :nocc, nocc:, nocc:].transpose(2, 3, 0, 1)

    dm2[:nocc, nocc:, nocc:, :nocc] = dovvo.ascupy()
    dm2[:nocc, nocc:, nocc:, :nocc] += dovvo.ascupy().transpose(3,
                                                                2, 1, 0).conj()
    dovvo = None
    dm2[nocc:, :nocc, :nocc, nocc:] = dm2[
        :nocc, nocc:, nocc:, :nocc].transpose(1, 0, 3, 2).conj()

    if dvvvv.ndim == 2:
        # To handle the case of compressed vvvv, which is used in nuclear
        # gradients
        assert False, 'not implement'
    else:
        dm2[nocc:, nocc:, nocc:, nocc:] = dvvvv.ascupy()
        dm2[nocc:, nocc:, nocc:, nocc:] += dvvvv.ascupy(
        ).transpose(1, 0, 3, 2).conj()
        dvvvv = None
        dm2[nocc:, nocc:, nocc:, nocc:] *= 2

    dm2[:nocc, :nocc, :nocc, :nocc] = doooo.ascupy()
    dm2[:nocc, :nocc, :nocc, :nocc] += doooo.ascupy().transpose(1,
                                                                0, 3, 2).conj()
    doooo = None
    dm2[:nocc, :nocc, :nocc, :nocc] *= 2

    dm2[:nocc, nocc:, nocc:, nocc:] = dovvv.ascupy()
    dm2[nocc:, nocc:, :nocc, nocc:] = dovvv.ascupy().transpose(2, 3, 0, 1)
    tmp = dovvv.ascupy().conj()
    dm2[nocc:, nocc:, nocc:, :nocc] = tmp.transpose(3, 2, 1, 0)
    dm2[nocc:, :nocc, nocc:, nocc:] = tmp.transpose(1, 0, 3, 2)
    tmp = None
    dovvv = None

    dm2[:nocc, :nocc, :nocc, nocc:] = dooov.ascupy()
    dm2[:nocc, nocc:, :nocc, :nocc] = dooov.ascupy().transpose(2, 3, 0, 1)
    tmp = dooov.ascupy().conj()
    dm2[:nocc, :nocc, nocc:, :nocc] = tmp.transpose(1, 0, 3, 2)
    dm2[nocc:, :nocc, :nocc, :nocc] = tmp.transpose(3, 2, 1, 0)
    tmp = None
    dooov = None

    if with_frozen and mycc.frozen is not None:
        assert False, 'not implement'

    if with_dm1:
        dm1 = _make_rdm1(mycc, d1, with_frozen)
        dm1[numpy.diag_indices(nocc)] -= 2

        for i in range(nocc):
            dm2[i, i, :, :] += dm1 * 2
            dm2[:, :, i, i] += dm1 * 2
            dm2[:, i, i, :] -= dm1
            dm2[i, :, :, i] -= dm1.T

        for i in range(nocc):
            for j in range(nocc):
                dm2[i, i, j, j] += 4
                dm2[i, j, j, i] -= 2

    # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
    # above. Transposing it so that it be contracted with ERIs (in Chemist's
    # notation):
    #   E = einsum('pqrs,pqrs', eri, rdm2)
    dm2 = dm2.transpose(1, 0, 3, 2)

    if ao_repr:
        mo = mycc.mo_coeff
        if not numpy.allclose(mo, numpy.eye(mo.shape[0]), atol=1e-12):
            moC = mycc.mo_coeff.conj()
            dm2 = cupy.einsum('ijkl,pi,qj,rk,sl->pqrs',
                              dm2.transpose(1, 0, 3, 2), mo, moC, mo, moC)
            return dm2
        else:
            return cupy.asarray(dm2.transpose(1, 0, 3, 2))
    return dm2
