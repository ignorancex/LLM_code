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

import cupy
from pyscf.lib import logger, prange
from byteqc.cucc import culib
from pyscf.cc.ccsd import BLKMIN
from byteqc.cucc import diis
from byteqc import lib
from byteqc.cucc.ccsd import diffabs2_t2


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO,
           fintermediates=None, fupdate=None, callback=None):
    if eris is None:
        eris = mycc.ao2mo()
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)

    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = t1
    if l2 is None:
        l2 = t2
    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)

    if isinstance(mycc.diis, diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = diis.DIIS(
            mycc, mycc.diis_file, incore=mycc.incore_complete,
            pool=mycc.pool)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        if callback is not None:
            callback(locals())
        normt = cupy.sqrt(culib.diffabs2_t1(l1, l1new)
                          + diffabs2_t2(mycc, l2, l2new))
        l1, l1new = l1new, None
        l2, l2new = l2new, None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep + 1, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    pool = mycc.pool
    if getattr(eris, 'Loo', None) is None:
        naux = None
    else:
        naux = eris.Loo.shape[0]
    pool.memory_status(nocc, nvir, naux, which=1)
    w1 = pool.asarray(eris.fock[nocc:, nocc:])
    w2 = pool.asarray(eris.fock[:nocc, :nocc])
    w4 = pool.asarray(eris.fock[:nocc, nocc:])
    w3 = w4.T.ascupy(copy=True)

    class _IMDS:
        pass
    imds = _IMDS()
    # TODO: mycc.incore_complete
    imds.w1 = w1
    imds.w2 = w2
    imds.w3 = w3
    imds.w4 = w4

    namx = max(nocc, nvir)
    unit = nocc * namx**2 * 3
    max_memory = mycc.pool.free_memory
    blksize = min(nvir, max(BLKMIN, int(max_memory / 8 / unit)))
    log.debug1('ccsd lambda make_intermediates: '
               'block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir + blksize - 1) // blksize))
    buf = lib.ArrayBuffer(
        pool.empty((blksize * nocc * namx * namx * 3 + 1024), 'f8'))
    buf1 = buf.empty((blksize, nocc, namx, namx), 'f8')
    buf2 = buf.empty((blksize, nocc, namx, namx), 'f8')
    buf.tag('buf3')
    buf3 = buf.empty((blksize, nocc, namx, namx), 'f8')
    buf.untag('buf3')
    buf3_1 = buf.empty((blksize, nocc), 'f8')
    buf3_2 = buf.empty((blksize, nvir), 'f8')

    culib.contraction('ja', w4, 'jb', t1, 'ba', w1, alpha=-1.0, beta=1.0)
    culib.contraction('ib', w4, 'jb', t1, 'ij', w2, beta=1.0)
    culib.contraction('kc', w4, 'jkbc', t2, 'bj', w3, alpha=2.0, beta=1.0)
    culib.contraction('kc', w4, 'kjbc', t2, 'bj', w3, beta=1.0, alpha=-1.0)

    tmp = culib.contraction('kc', w4, 'kb', t1, 'cb', buf=buf.arr)
    culib.contraction('cb', tmp, 'jc', t1, 'bj', w3, beta=1.0)
    tmp = None

    imds.woooo = pool.new('woooo', (nocc, nocc, nocc, nocc), 'f8')
    imds.wvooo = pool.new('wvooo', (nvir, nocc, nocc, nocc), 'f8')
    imds.wVOov = pool.new('wVOov', (nvir, nocc, nocc, nvir), 'f8')
    imds.wvOOv = pool.new('wvOOv', (nvir, nocc, nocc, nvir), 'f8')
    imds.wvvov = pool.new('wvvov', (nvir, nvir, nocc, nvir), 'f8')

    woooo = imds.woooo
    woooo[:] = 0
    wvooo = imds.wvooo
    wvooo[:] = 0

    for p0, p1 in prange(0, nvir, blksize):
        wVOov = imds.wVOov[p0:p1]
        wvOOv = imds.wvOOv[p0:p1]
        wvvov = imds.wvvov[p0:p1]
        eris_ovoo = eris.ovoo.getitem(cupy.s_[:, p0:p1], order='c', buf=buf1)
        culib.contraction('kbij', eris_ovoo, 'kb', t1[:, p0:p1], 'ij', w2,
                          alpha=2.0, beta=1.0)
        culib.contraction('ibkj', eris_ovoo, 'kb', t1[:, p0:p1], 'ij', w2,
                          alpha=-1.0, beta=1.0)

        t2slice = t2[:, :, p0:p1].ascupy(order='c', buf=buf2)
        culib.contraction('lckj', eris_ovoo, 'lkcb', t2slice, 'bj', w3,
                          alpha=-2.0, beta=1.0)
        culib.contraction('lckj', eris_ovoo, 'klcb', t2slice, 'bj', w3,
                          beta=1.0)

        culib.contraction('lc', t1[:, p0:p1], 'jcik', eris_ovoo, 'ijkl', woooo,
                          beta=1.0)
        culib.contraction('lc', t1[:, p0:p1], 'jcik', eris_ovoo, 'jilk', woooo,
                          beta=1.0)
        # First set wvOOv
        culib.contraction('lbjk', eris_ovoo, 'lc', t1, 'bjkc', wvOOv)
        # First set wVOov
        culib.contraction('jbkl', eris_ovoo, 'lc', t1, 'bjkc', wVOov,
                          alpha=-1)
        culib.contraction('klbc', t2slice, 'iblj', eris_ovoo, 'ckij',
                          wvooo, alpha=-1.5, beta=1.0)
        t2slice = None

        tmp = lib.empty_from_buf(buf2, wvooo[p0:p1].shape, wvooo.dtype)
        lib.elementwise_trinary('dacb', eris_ovoo, 'bacd', eris_ovoo, 'abcd',
                                tmp, alpha=2.0, beta=-1.0)
        wvooo[p0:p1] += tmp
        tmp = None

        g2ovoo = lib.empty_from_buf(buf2, eris_ovoo.shape, eris_ovoo.dtype)
        lib.elementwise_binary('cbad', eris_ovoo, 'abcd', eris_ovoo,
                               g2ovoo, alpha=-1.0, gamma=2.0)
        eris_ovoo = None
        t2slice = t2[:, :, :, p0:p1].ascupy(order='c', buf=buf3)
        culib.contraction('jlcb', t2slice, 'lbki', g2ovoo, 'cikj',
                          wvooo, alpha=2.0, beta=1.0)
        culib.contraction('jlcb', t2slice, 'lbki', g2ovoo, 'cjki',
                          wvooo, alpha=-1.0, beta=1.0)
        culib.contraction('ljcb', t2slice, 'lbki', g2ovoo, 'cikj',
                          wvooo, alpha=-1.0, beta=1.0)
        culib.contraction('ljcb', t2slice, 'lbki', g2ovoo, 'cjki',
                          wvooo, alpha=0.5, beta=1.0)
        t2slice = None
        culib.contraction('ia', t1, 'jb', t1, 'ijab', t2, beta=1.0)
        # First set wvvov
        culib.contraction('laki', g2ovoo, 'klbc', t2, 'abic', wvvov)
        g2ovoo = None

        # all buf released

        eris_ovvv = eris.ovvv.getitem(cupy.s_[:, :, p0:p1], copy=True,
                                      buf=buf1)
        culib.contraction('idcb', eris_ovvv, 'jkbd', t2,
                          'ckij', wvooo[p0:p1], alpha=2.0, beta=1.0)
        culib.contraction('ibcd', eris_ovvv, 'jkbd', t2,
                          'ckij', wvooo[p0:p1], alpha=-1.0, beta=1.0)
        culib.contraction('ia', t1, 'jb', t1, 'ijab', t2, alpha=-1.0, beta=1.0)
        culib.contraction('jdcb', eris_ovvv, 'kd', t1, 'cjkb', wvOOv,
                          alpha=-1.0, beta=1.0)

        eris_ovvv2 = eris.ovvv.getitem(cupy.s_[:, p0:p1], order='c', buf=buf2)
        culib.contraction('jcba', eris_ovvv2, 'jc', t1[:, p0:p1],
                          'ba', w1, alpha=2.0, beta=1.0)
        culib.contraction('jabc', eris_ovvv2, 'jc', t1, 'ba', w1[:, p0:p1],
                          alpha=-1.0, beta=1.0)

        t2slice = t2[:, :, :, p0:p1].ascupy(order='c', buf=buf3)
        culib.contraction('jkcd', t2slice, 'kdcb', eris_ovvv2, 'bj',
                          w3, alpha=2.0, beta=1.0)
        culib.contraction('kjcd', t2slice, 'kdcb', eris_ovvv2, 'bj',
                          w3, alpha=-1.0, beta=1.0)
        t2slice = None

        culib.contraction('jbcd', eris_ovvv2, 'kd', t1, 'bjkc', wVOov,
                          beta=1.0)
        culib.contraction('jabd', eris_ovvv2, 'jkcd', t2, 'abkc', wvvov,
                          alpha=-1.5, beta=1.0)

        g2vvov = lib.elementwise_binary('acbd', eris_ovvv2, 'abcd', eris_ovvv,
                                        alpha=-1.0, gamma=2.0)
        vackb = culib.contraction('jdac', g2vvov, 'kjbd', t2, 'ackb',
                                  alpha=2.0, buf=buf3)
        lib.elementwise_trinary('cadb', eris_ovvv2, 'cdab', g2vvov, 'abcd',
                                vackb, gamma=1.0)
        eris_ovvv2 = None
        culib.contraction('jdac', g2vvov, 'kjdb', t2, 'ackb',
                          vackb, alpha=-1.0, beta=1.0)
        g2vvov = eris_ovvv = None

        tmp = lib.empty_from_buf(buf1, vackb.shape, vackb.dtype)
        lib.elementwise_binary('adcb', vackb, 'abcd', vackb, tmp,
                               alpha=1.0, gamma=-0.5)
        vackb = None
        wvvov += tmp
        tmp = None

        # all buf released

        eris_ovov = eris.ovov.getitem(cupy.s_[:, p0:p1], order='c', buf=buf1)
        culib.contraction('ia', t1[:, p0:p1], 'jb', t1, 'ijab',
                          t2[:, :, p0:p1], beta=1.0)
        culib.contraction('icjd', eris_ovov, 'klcd', t2[:, :, p0:p1], 'ijkl',
                          woooo, beta=1.0)
        culib.contraction('ia', t1[:, p0:p1], 'jb', t1, 'ijab',
                          t2[:, :, p0:p1], alpha=-1.0, beta=1.0)
        tmp = culib.contraction('lbjd', eris_ovov, 'kd', t1, 'bljk', buf=buf2)
        culib.contraction('bljk', tmp, 'lc', t1, 'bjkc', wvOOv,
                          alpha=1.0, beta=1.0)
        tmp = culib.contraction('jbld', eris_ovov, 'kd', t1, 'bjlk', buf=buf2)
        culib.contraction('bjlk', tmp, 'lc', t1, 'bjkc', wVOov,
                          alpha=-1.0, beta=1.0)
        tmp = None

        g2ovov = lib.empty_from_buf(buf2, eris_ovov.shape, eris_ovov.dtype)
        lib.elementwise_binary('cbad', eris_ovov, 'abcd', eris_ovov,
                               g2ovov, alpha=-1.0, gamma=2.0)

        tmpw4 = culib.contraction('kcld', g2ovov, 'ld', t1, 'kc', buf=buf3_1)
        culib.contraction('kcja', g2ovov, 'kjcb', t2[:, :, p0:p1], 'ba', w1,
                          alpha=-1.0, beta=1.0)
        culib.contraction('ja', tmpw4, 'jb', t1, 'ba', w1[:, p0:p1],
                          alpha=-1.0, beta=1.0)
        culib.contraction('jkbc', t2[:, :, p0:p1], 'ibkc', g2ovov, 'ij', w2,
                          beta=1.0)
        culib.contraction('ib', tmpw4, 'jb', t1[:, p0:p1], 'ij', w2, beta=1.0)

        tmp = culib.gemm('T', 'N', t1, tmpw4, buf=buf3_2)
        w4[:, p0:p1] += tmpw4
        tmpw4 = None
        culib.contraction('ab', tmp, 'cb', t1[:, p0:p1], 'ac', w3, beta=1.0)
        tmp = None

        VOov = culib.contraction('jbld', g2ovov, 'klcd', t2, 'bjkc', buf=buf3)
        g2ovov = None
        culib.contraction('jbld', eris_ovov, 'kldc', t2, 'bjkc', VOov,
                          alpha=-1.0, beta=1.0)
        lib.elementwise_binary('bacd', eris_ovov, 'abcd', VOov, gamma=1.0)
        vOOv = culib.contraction('lbjd', eris_ovov, 'kldc', t2, 'bjkc',
                                 buf=buf2)
        eris_ovov = None
        eri_oovv = eris.oovv.getitem(cupy.s_[:, :, p0:p1], order='c', buf=buf1)
        lib.elementwise_binary('cbad', eri_oovv, 'abcd', vOOv,
                               alpha=-1.0, gamma=1.0)
        eri_oovv = None
        wVOov += VOov
        wvOOv += vOOv

        # vOOv, VOov = 2*vOOv+VOov, 2*VOov+vOOv
        lib.elementwise_binary('abcd', VOov, 'abcd', vOOv,
                               alpha=1.0, gamma=2.0)
        lib.elementwise_binary('abcd', vOOv, 'abcd', VOov,
                               alpha=0.5, gamma=1.5)

        culib.contraction('jb', t1[:, p0:p1], 'bikc', vOOv, 'ckij',
                          wvooo, alpha=-1.0, beta=1.0)
        culib.contraction('kb', t1[:, p0:p1], 'bijc', VOov, 'ckij',
                          wvooo, beta=1.0)
        culib.contraction('ckjb', VOov, 'kc', t1[:, p0:p1], 'bj', w3, beta=1.0)

        culib.contraction('ajkc', vOOv, 'jb', t1, 'abkc', wvvov, beta=1.0)
        culib.contraction('ajkb', VOov, 'jc', t1, 'abkc', wvvov, alpha=-1.0,
                          beta=1.0)
        vOOv = VOov = None

    buf1 = buf2 = buf3 = buf3_1 = buf3_2 = None
    blksize = min(nocc, int(buf.nbytes / 8 / nocc ** 3))
    for p0, p1 in prange(0, nocc, blksize):
        with woooo[p0:p1] as arr:
            eris_oooo = eris.oooo.getitem(cupy.s_[p0:p1], buf=buf.arr)
            lib.elementwise_binary('acbd', eris_oooo, 'abcd', arr, gamma=1.0)
    buf = eris_oooo = None

    culib.contraction('bc', w1, 'jc', t1, 'bj', w3, beta=1.0)
    culib.contraction('kj', w2, 'kb', t1, 'bj', w3, alpha=-1.0, beta=1.0)

    lib.free_all_blocks()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if imds is None:
        imds = make_intermediates(mycc, t1, t2, eris)
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    pool = mycc.pool
    mo_e_o = pool.asarray(eris.mo_energy[:nocc])
    mo_e_v = pool.asarray(eris.mo_energy[nocc:])
    mo_e_v += mycc.level_shift

    mba = pool.empty((nvir, nvir), 'f8')
    culib.contraction('klca', l2, 'klcb', t2, 'ba', mba, alpha=2.0)
    culib.contraction('klca', l2, 'klbc', t2, 'ba', mba, alpha=-1.0, beta=1.0)
    mij = pool.empty((nocc, nocc), 'f8')
    culib.contraction('ikcd', l2, 'jkcd', t2, 'ij', mij, alpha=2.0)
    culib.contraction('ikcd', l2, 'jkdc', t2, 'ij', mij, alpha=-1.0, beta=1.0)
    mba1 = pool.empty((nvir, nvir), 'f8')
    culib.contraction('jc', l1, 'jb', t1, 'bc', mba1)
    mba1 += mba
    mij1 = pool.empty((nocc, nocc), 'f8')
    culib.contraction('kb', l1, 'jb', t1, 'kj', mij1)
    mij1 += mij
    mia1 = pool.empty((nocc, nvir), 'f8')
    culib.contraction('kc', l1, 'jkbc', t2, 'jb', mia1, alpha=2.0)
    mia1 += t1
    culib.contraction('kc', l1, 'jkcb', t2, 'jb', mia1, alpha=-1.0, beta=1.0)

    tmp = pool.empty((nocc, nocc), 'f8')
    culib.gemm('N', 'T', t1, l1, tmp)
    culib.gemm('N', 'N', tmp, t1, mia1, alpha=-1.0, beta=1.0)
    tmp = None
    culib.contraction('bd', mba, 'jd', t1, 'jb', mia1, alpha=-1.0, beta=1.0)
    culib.contraction('lj', mij, 'lb', t1, 'jb', mia1, alpha=-1.0, beta=1.0)

    l2new = mycc.pool.new('', l2.shape)
    mycc._add_vvvv(None, l2, eris, out=l2new, with_ovvv=False, t2sym='jiba')
    l1new = mycc.pool.new('t1', l1.shape)
    culib.contraction('ijab', l2new, 'jb', t1, 'ia', l1new, alpha=2.0)

    culib.contraction('jiab', l2new, 'jb', t1, 'ia', l1new, alpha=-1.0,
                      beta=1.0)

    # *.5 because of l2+l2.transpose(1,0,3,2) in the end
    l2new *= .5

    w1 = imds.w1 - cupy.diag(mo_e_v)
    w2 = imds.w2 - cupy.diag(mo_e_o)

    l1new += pool.asarray(eris.fock[:nocc, nocc:])
    culib.contraction('ib', l1, 'ba', w1, 'ia', l1new, beta=1.0)
    culib.contraction('ja', l1, 'ij', w2, 'ia', l1new, alpha=-1.0, beta=1.0)
    culib.contraction('ik', mij, 'ka', imds.w4,
                      'ia', l1new, alpha=-1.0, beta=1.0)
    culib.contraction('ca', mba, 'ic', imds.w4,
                      'ia', l1new, alpha=-1.0, beta=1.0)
    culib.contraction('ijab', l2, 'bj', imds.w3,
                      'ia', l1new, alpha=2.0, beta=1.0)
    culib.contraction('ijba', l2, 'bj', imds.w3,
                      'ia', l1new, alpha=-1.0, beta=1.0)

    culib.contraction('ia', l1, 'jb', imds.w4, 'ijab', l2new, beta=1.0)
    culib.contraction('jibc', l2, 'ca', w1, 'jiba', l2new, beta=1.0)
    w1 = None
    culib.contraction('jk', w2, 'kiba', l2, 'jiba', l2new,
                      alpha=-1.0, beta=1.0)
    w2 = None

    culib.contraction('ckij', imds.wvooo, 'jkca', l2,
                      'ia', l1new, alpha=-1.0, beta=1.0)
    culib.contraction('abkc', imds.wvvov, 'kibc', l2, 'ia',
                      l1new, beta=1.0)

    culib.contraction('jkca', l2, 'bikc', imds.wvOOv, 'jiba',
                      l2new, beta=1.0)
    culib.contraction('jkca', l2, 'bikc', imds.wvOOv, 'ijba',
                      l2new, alpha=0.5, beta=1.0)

    culib.contraction('ikca', l2, 'bjkc', imds.wVOov,
                      'jiba', l2new, alpha=-1.0, beta=1.0)
    culib.contraction('ikac', l2, 'bjkc', imds.wVOov,
                      'jiba', l2new, alpha=2.0, beta=1.0)
    culib.contraction('ikca', l2, 'bjkc', imds.wvOOv,
                      'jiba', l2new, alpha=-0.5, beta=1.0)
    culib.contraction('ikac', l2, 'bjkc', imds.wvOOv,
                      'jiba', l2new, beta=1.0)

    nmax = max(nocc, nvir)
    max_memory = pool.free_memory
    unit = nocc**2 * nvir + nocc * nvir * nmax + nmax**2 * nvir
    blksize = min(nocc, int(max_memory / 8 / unit))
    log.debug1('block size = %d, nvir = %d is divided into %d blocks',
               blksize, nvir, int((nvir + blksize - 1) / blksize))

    buf = lib.ArrayBuffer(pool.empty((unit * blksize + 1024), 'f8'))
    buf1 = buf.empty((blksize, nocc, nocc, nvir), 'f8')
    buf2 = buf.empty((blksize, nvir, nmax, nmax), 'f8')
    buf3 = buf.empty((blksize, nocc, nvir, nmax), 'f8')

    blk = min(nocc, int(buf.nbytes / 8 / nocc**2 / nvir))
    for p0, p1 in prange(0, nocc, blk):
        eris_ovoo = eris.ovoo.getitem(cupy.s_[p0:p1], buf=buf.arr)
        culib.contraction('iajk', eris_ovoo, 'kj', mij1, 'ia', l1new[p0:p1],
                          alpha=-2.0, beta=1.0)
        culib.contraction('jaik', eris_ovoo, 'kj', mij1[:, p0:p1], 'ia',
                          l1new, beta=1.0)
        culib.contraction('jbki', eris_ovoo, 'ka', l1, 'jiba', l2new[p0:p1],
                          alpha=-1.0, beta=1.0)
    eris_ovoo = None

    blk = min(nocc, int(buf.nbytes / 8 / nocc / nvir**2))
    for p0, p1 in prange(0, nocc, blk):
        eris_oovv = eris.oovv.getitem(cupy.s_[p0:p1], buf=buf.arr)
        culib.contraction('jb', l1[p0:p1], 'jiab', eris_oovv, 'ia', l1new,
                          alpha=-1.0, beta=1.0)
    eris_oovv = None

    for p0, p1 in prange(0, nvir, blksize):
        eris_ovvv = eris.ovvv.getitem(cupy.s_[:, p0:p1], buf=buf3)
        culib.contraction('iabc', eris_ovvv, 'bc', mba1, 'ia',
                          l1new[:, p0:p1], alpha=2.0, beta=1.0)
        culib.contraction('ibca', eris_ovvv, 'bc', mba1[p0:p1], 'ia', l1new,
                          alpha=-1.0, beta=1.0)
        culib.contraction('jbac', eris_ovvv, 'ic', l1, 'jiba',
                          l2new[:, :, p0:p1], beta=1.0)

        tmp = culib.contraction('kc', t1, 'kadb', eris_ovvv, 'dcab', buf=buf2)
        m4 = culib.contraction('jidc', l2, 'dcab', tmp, 'ijab', buf=buf1)

        l2new[:, :, p0:p1] -= m4
        culib.contraction('ijab', m4, 'jb', t1, 'ia', l1new[:, p0:p1],
                          alpha=-2.0, beta=1.0)
        culib.contraction('ijab', m4, 'ia', t1[:, p0:p1], 'jb', l1new,
                          alpha=-2.0, beta=1.0)
        culib.contraction('jiab', m4, 'jb', t1, 'ia', l1new[:, p0:p1],
                          beta=1.0)
        culib.contraction('jiab', m4, 'ia', t1[:, p0:p1], 'jb', l1new,
                          beta=1.0)
    eris_ovvv = tmp = m4 = None

    for p0, p1 in prange(0, nvir, blksize):
        eris_ovov = eris.ovov.getitem(cupy.s_[:, p0:p1], buf=buf1)
        culib.contraction('jb', l1, 'iajb', eris_ovov, 'ia',
                          l1new[:, p0:p1], alpha=2.0, beta=1.0)
        culib.contraction('iajb', eris_ovov, 'jb', mia1, 'ia',
                          l1new[:, p0:p1], alpha=2.0, beta=1.0)
        culib.contraction('ibja', eris_ovov, 'jb', mia1[:, p0:p1], 'ia', l1new,
                          alpha=-1.0, beta=1.0)
        tmp = culib.contraction('klcd', t2, 'kalb', eris_ovov, 'cdab',
                                buf=buf2)
        tmp2 = culib.contraction('kc', t1, 'kalb', eris_ovov, 'clab', buf=buf3)
        culib.contraction('ld', t1, 'clab', tmp2, 'cdab', tmp, beta=1.0)
        tmp2 = None
        m4 = culib.contraction('ijcd', l2, 'cdab', tmp, 'ijab', buf=buf3)
        tmp = None
        culib.contraction('ijab', m4, 'jb', t1, 'ia',
                          l1new[:, p0:p1], alpha=2, beta=1)
        culib.contraction('ijba', m4, 'jb', t1[:, p0:p1], 'ia', l1new,
                          alpha=-1.0, beta=1.0)

        l2new_s = l2new[:, :, p0:p1]
        l2nslice = l2new_s.enter(buf=buf2)
        lib.elementwise_binary('abcd', m4, 'abcd', l2nslice,
                               alpha=0.5, gamma=1.0)
        m4 = None
        lib.elementwise_binary('acbd', eris_ovov, 'abcd', l2nslice,
                               alpha=0.5, gamma=1.0)
        culib.contraction('jbic', eris_ovov, 'ca', mba1, 'jiba', l2nslice,
                          alpha=-1.0, beta=1.0)
        culib.contraction('jbka', eris_ovov, 'ik', mij1, 'jiba', l2nslice,
                          alpha=-1.0, beta=1.0)
        eris_ovov = None
        l2new_s.exit()

    mba = mij = mba1 = mij1 = mia1 = None

    blk = min(nocc, int(buf.nbytes / 8 / nocc / nvir**2))
    for p0, p1 in prange(0, nocc, blk):
        m3 = culib.contraction('ijkl', imds.woooo[p0:p1], 'klab', l2, 'ijab',
                               alpha=0.5, buf=buf.arr)
        l2new[p0:p1] += m3
        culib.contraction('ijab', m3, 'jb', t1, 'ia', l1new[p0:p1],
                          alpha=4.0, beta=1.0)
        culib.contraction('ijba', m3, 'jb', t1, 'ia', l1new[p0:p1],
                          alpha=-2.0, beta=1.0)
    m3 = None
    time0 = log.timer_debug1('lambda pass [%d:%d]' % (p0, p1), *time0)
    buf = buf1 = buf2 = buf3 = None

    ker = cupy.ElementwiseKernel(
        'int64 m, raw T eo, raw T ev', 'T out', '''
            size_t a = i / m;
            size_t b = i % m;
            out /= eo[a] - ev[b];
        ''', 't1new_mo_e')
    with l1new as arr:
        ker(nvir, mo_e_o, mo_e_v, arr)

#    l2new = l2new + l2new.transpose(1,0,3,2)
#    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
#    l2new += l2
    ker1 = cupy.ElementwiseKernel(
        'int64 m, int64 ind, raw T eo, raw T ev', 'T out', '''
            size_t c = i % m;
            size_t a = i / m;
            size_t b = a % m;
            a /= m;
            out /= eo[ind]  + eo[a] - ev[b] - ev[c];
        ''', 't2new_eia_1')
    ker2 = cupy.ElementwiseKernel(
        'int64 m, T eo, raw T ev', 'T out', '''
            size_t a = i / m;
            size_t b = i % m;
            out /= 2*eo - ev[a] - ev[b];
        ''', 't2new_eia_2')
    for i in range(nocc):
        if i > 0:
            with l2new[i, :i] as arr:
                arr += l2new[:i, i].ascupy().transpose(0, 2, 1)
                ker1(nvir, i, mo_e_o, mo_e_v, arr)
                l2new[:i, i] = arr.transpose(0, 2, 1)
        with l2new[i, i] as arr:
            arr += arr.T
            ker2(nvir, mo_e_o[i], mo_e_v, arr)

    lib.free_all_blocks()
    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new
