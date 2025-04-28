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

from byteqc import cucc, lib
from pyscf import df
from pyscf.lib import logger, prange, param
from byteqc.cucc import culib
import cupy
import numpy
from byteqc.cucc.ccsd import EriWrapper


class RCCSD(cucc.ccsd.CCSD):
    def __init__(
            self, mf, frozen=None, mo_coeff=None, mo_occ=None, gpulim=None,
            cpulim=None, pool=None, path=param.TMPDIR, mem_ratio=0.7):
        cucc.ccsd.CCSD.__init__(
            self, mf, frozen, mo_coeff, mo_occ, gpulim, cpulim, pool=pool,
            path=path, mem_ratio=mem_ratio)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)


def _contract_vvvv_t2(mycc, mol, Lvv, t1, t2, out=None, log=None):
    if log is None:
        log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = logger.process_clock(), logger.perf_counter()

    nocc, nvir = t2.shape[1:3]
    naux = Lvv.shape[0]

    memory = mycc.pool.free_memory
    blk1 = min(nvir, int(memory / 8 / (nvir**3)))
    if blk1 < 4:
        blk1 = 4
        blk2 = min(nvir, int(memory / 8 / (nvir**2) / blk1))
        log.info('nvir(%d) is sliced to %d and %d with memory %.2fGB' % (
            nvir, blk1, blk2, memory / 1e9))
    else:
        blk2 = nvir
        log.info('nvir(%d) is sliced to %d with memory %.2fGB' % (
            nvir, blk1, memory / 1e9))
    buf = mycc.pool.empty((blk1, blk2, nvir, nvir), t2.dtype)

    for p0, p1 in prange(0, nvir, blk1):
        for q0, q1 in prange(0, nvir, blk2):
            vvvv = culib.contraction('Lac', Lvv[:, p0:p1], 'Ldb',
                                     Lvv[:, q0:q1], 'cdab', buf=buf)
            culib.contraction('cdab', vvvv, 'ijcd', t2, 'ijab',
                              out[:, :, p0:p1, q0:q1], issync=True)
            if blk2 != nvir:
                time1 = log.timer_debug1(
                    'vvvv t2 [%d:%d]-[%d:%d]' % (p0, p1, q0, q1), *time1)
        if blk2 == nvir:
            time1 = log.timer_debug1('vvvv t2 [%d:%d]' % (p0, p1), *time1)

    buf = vvvv = None
    lib.free_all_blocks()

    if t1 is not None:
        ovL = mycc.pool.empty((nocc, nvir, naux), t2.dtype)
        culib.contraction('Lac', Lvv, 'ic', t1, 'iaL', ovL)
        if out.dev != 0:  # cutensorMG Bug
            memory = mycc.pool.free_memory
            blksize = min(nvir, int(memory / 8 / (nvir**2 * nocc)))
            for p0, p1 in prange(0, nocc, blksize):
                with out[p0:p1] as arr:
                    culib.contraction('iaL', ovL[p0:p1], 'jbL', ovL, 'ijab',
                                      arr, beta=1.0)
                arr = None
        else:
            culib.contraction('iaL', ovL, 'jbL', ovL, 'ijab', out, beta=1.0)
        time1 = log.timer_debug1('vvvv t1', *time1)
        ovL = None
    lib.free_all_blocks()
    return out


class _ChemistsERIs(cucc.ccsd._ChemistsERIs):
    def _contract_vvvv_t2(self, mycc, t1, t2, direct=False,
                          out=None, log=None):
        assert (not direct)
        return _contract_vvvv_t2(
            mycc, self.mol, self.Lvv, t1, t2, out, log)


def nr_e2(eri, mo_coeff, orbs_slice, aosym='s1',
          mosym='s1', out=None, ao_loc=None):
    assert (eri.flags.c_contiguous)
    assert (mo_coeff.dtype == numpy.double)
    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    kl_count = kc * lc

    nrow = eri.shape[0]
    if nrow * kl_count == 0:
        return cupy.empty((nrow, kl_count))
    assert aosym == 's2' and mosym == 's1' and kc == lc and ao_loc is None, (
        "Not implemented")
    mat = lib.unpack_tril(eri, out=out)
    tmp = culib.contraction('ij', mo_coeff[k0:], 'mjk', mat, 'mik')
    return culib.gemm(
        'N', 'T', tmp.reshape(-1, tmp.shape[-1]), mo_coeff[l0:], buf=out
    ).reshape((tmp.shape[0], tmp.shape[1], -1))


def _make_df_eris(mycc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = mycc.with_df
    naux = eris.naux = with_df.get_naoaux()
    status = mycc.pool.memory_status(nocc, nvir, naux,
                                     mem_ratio=mycc.mem_ratio)

    Loo = mycc.pool.new('Loo', (naux, nocc, nocc), 'f8')
    Lov = mycc.pool.new('Lov', (naux, nocc, nvir), 'f8')
    Lvv = eris.Lvv = mycc.pool.new('Lvv', (naux, nvir, nvir), 'f8')
    d_mo_coeff = mycc.pool.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for _, eri1 in enumerate(with_df.loop()):
        eri1 = cupy.asarray(eri1)
        Lpq = nr_e2(eri1, d_mo_coeff.T, ijslice,
                    aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1 - p0, nmo, nmo)
        Loo[p0:p1] = Lpq[:, :nocc, :nocc]
        Lov[p0:p1] = Lpq[:, :nocc, nocc:]
        Lvv[p0:p1] = Lpq[:, nocc:, nocc:]
        eri1 = None
    Lpq = d_mo_coeff = None
    lib.free_all_blocks()

    _Loo = Loo.reshape(naux, nocc**2)
    _Lov = Lov.reshape(naux, nocc * nvir)
    _Lvv = Lvv.reshape(naux, nvir**2)

    if status['oooo'] != 0:
        eris.oooo = EriWrapper(Loo, Loo)
    else:
        eris.oooo = EriWrapper(mycc.pool.new(
            'oooo', (nocc, nocc, nocc, nocc), 'f8'))
        culib.gemm('T', 'N', _Loo, _Loo, eris.oooo.reshape(nocc**2, -1))

    if status['ovoo'] != 0:
        eris.ovoo = EriWrapper(Lov, Loo)
    else:
        eris.ovoo = EriWrapper(mycc.pool.new(
            'ovoo', (nocc, nvir, nocc, nocc), 'f8'))
        culib.gemm('T', 'N', _Lov, _Loo, eris.ovoo.reshape(nocc * nvir, -1))

    if status['ovov'] != 0:
        eris.ovov = EriWrapper(Lov, Lov)
    else:
        eris.ovov = EriWrapper(mycc.pool.new(
            'ovov', (nocc, nvir, nocc, nvir), 'f8'))
        culib.gemm('T', 'N', _Lov, _Lov, eris.ovov.reshape(nocc * nvir, -1))

    if status['oovv'] != 0:
        eris.oovv = EriWrapper(Loo, Lvv)
    else:
        eris.oovv = EriWrapper(mycc.pool.new(
            'oovv', (nocc, nocc, nvir, nvir), 'f8'))
        culib.gemm('T', 'N', _Loo, _Lvv, eris.oovv.reshape(nocc**2, -1))

    if status['ovvv'] != 0:
        eris.ovvv = EriWrapper(Lov, Lvv)
    else:
        eris.ovvv = EriWrapper(mycc.pool.new(
            'ovvv', (nocc, nvir, nvir, nvir), 'f8'))
        culib.gemm('T', 'N', _Lov, _Lvv, eris.ovvv.reshape(nocc * nvir, -1))
    Loo = Lov = _Loo = _Lov = None
    lib.free_all_blocks()
    return eris
