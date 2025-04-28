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
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf,
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

import numpy
import cupy
import h5py
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import gto as mol_gto
from pyscf.pbc.lib.kpts_helper import (is_zero, unique)
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.rsdf_helper import KPT_DIFF_TOL, gamma_point, _get_bvk_data
from byteqc.cupbc.df import ft_ao as gint_ft_ao
from byteqc.cupbc.int3c2e import _VHFOpt
from byteqc.lib import contraction, gpu_avail_bytes, ArrayBuffer, gemm, Mg
from byteqc.cupbc import lib as cupbclib
from byteqc.cupbc.df.rsdf_direct_helper import (
    get_j2c, cholesky_decomposed_metric, get_kptij_lst, loop_uniq_q,
    get_prescreening_data, get_aux_chg, j3c_g0_kernel, j3c_g0_real_kernel)
import scipy


libgpbc = cupbclib.load_library('libgpbc')
libgaft = cupbclib.load_library('libgaft')
libcgto = lib.load_library('libcgto')


class Int3cEngine:
    def __init__(self, df, verbose=None):
        self.df = df
        self.isconfig = False
        self.isbuild = False
        self.int3c = lambda *x, **y: None
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = df.verbose

    @property
    def j2c(self):
        if self.jktype == 'j':
            if hasattr(self, 'j2c_k'):
                self.j2c_j = (self.j2c_k[0][0], self.j2c_k[1][0],
                              self.j2c_k[2][0])
            if not hasattr(self, 'j2c_j'):
                j2c = get_j2c(self.df, kpts=numpy.zeros((1, 3)),
                              verbose=self.verbose - 2, omega=self.df.omega,
                              jktype="j-j2c")[0]
                j2c, j2c_negative, j2ctag = cholesky_decomposed_metric(
                    self.df, j2c)
                self.j2c_j = j2c, j2c_negative, j2ctag
            return self.j2c_j
        else:
            if not hasattr(self, 'j2c_k'):
                uniq_kpts = [x[0] for x in loop_uniq_q(
                    self.df, kptij_lst=self.kptij_lst, verbose=0)]
                nkpts_uniq = len(uniq_kpts)
                kj2c = get_j2c(self.df, kpts=uniq_kpts,
                               verbose=self.verbose - 2, omega=self.df.omega,
                               jktype="k-j2c")

                kj2c_neg = [None] * nkpts_uniq
                kj2ctag = [None] * nkpts_uniq
                knauxcd = [None] * nkpts_uniq

                for k, _ in enumerate(uniq_kpts):
                    j2c, kj2c_neg[k], kj2ctag[k] = cholesky_decomposed_metric(
                        self.df, kj2c[k])
                    if kj2ctag[k] == 'CD':
                        kj2c[k] = j2c
                    else:
                        kj2c[k] = j2c
                    knauxcd[k] = j2c.shape[0]
                    j2c = None
                kj2c = numpy.asarray(kj2c)
                self.j2c_k = (kj2c, kj2c_neg, kj2ctag)
            return self.j2c_k

    @property
    def j2c_coeff(self):
        assert self.isbuild, "cannot get j2c_coeff before build"
        coeff = self.auxvhfopt.coeff
        if self.jktype == 'j':
            if not hasattr(self, 'j2c_coeff_j'):
                j2cs, _, j2ctags = self.j2c
                if j2ctags == 'CD':
                    j2c_coeff = scipy.linalg.solve_triangular(
                        j2cs, coeff.T, lower=True).T
                else:
                    j2c_coeff = coeff @ j2cs.T
                self.j2c_coeff_j = j2c_coeff
            return self.j2c_coeff_j
        else:
            if not hasattr(self, 'j2c_coeff_k'):
                j2cs, _, j2ctags = self.j2c
                uniq_q_loop = [x for x in loop_uniq_q(
                    self.df, kptij_lst=self.kptij_lst, verbose=0)]
                nkpts_uniq = len(uniq_q_loop)
                j2c_coeffs = [None] * len(j2cs)
                for kq in range(nkpts_uniq):
                    j2c = j2cs[kq]
                    j2ctag = j2ctags[kq]
                    if j2ctag == 'CD':
                        j2c_coeff = scipy.linalg.solve_triangular(
                            j2c, coeff.T, lower=True).T
                    else:
                        j2c_coeff = coeff @ j2c.T
                    j2c_coeffs[kq] = j2c_coeff
                self.j2c_coeff_k = numpy.asarray(j2c_coeffs)
            return self.j2c_coeff_k

    def apply_aux_coeff(self, j3c, auxcoeff, buf=None):
        naux = self.auxvhfopt.mol.nao_nr(cart=False)
        naux_cart = self.auxvhfopt.mol.nao_nr(cart=True)
        if buf is None:
            out = cupy.ndarray((*j3c.shape[:-1], naux), dtype=j3c.dtype)
        else:
            out = cupy.ndarray(
                (*j3c.shape[:-1], naux), dtype=j3c.dtype, memptr=buf.data)
        if auxcoeff.ndim == 2:
            if j3c.dtype == auxcoeff.dtype:
                return gemm(j3c.reshape((-1, naux_cart)), auxcoeff,
                            out.reshape((-1, naux)), alpha=1.0)
            else:
                n = j3c.reshape(-1, naux_cart).shape[0]
                j3c_real = cupy.ndarray(
                    (n, naux_cart, 2), dtype='f8', memptr=j3c.data)
                out_real = cupy.ndarray(
                    (n, naux, 2), dtype='f8', memptr=out.data)
                contraction('abi', j3c_real, 'bB', auxcoeff, 'aBi', out_real)
                return out

        assert self.jktype == 'k', (
            '`auxcoeff` with 3-dimensions is for `k` type only')
        kq = -1
        for _, _, adapted_ji_idx in loop_uniq_q(
                self.df, kptij_lst=self.kptij_lst, verbose=0):
            kq += 1
            for ji in adapted_ji_idx:
                if j3c.dtype == auxcoeff.dtype:
                    gemm(j3c[ji].reshape(-1, naux_cart), auxcoeff[kq],
                         out[ji].reshape(-1, naux), alpha=1.0)
                else:
                    n = j3c[ji].reshape(-1, naux_cart).shape[0]
                    j3c_real = cupy.ndarray(
                        (n, naux_cart, 2), dtype='f8', memptr=j3c[ji].data)
                    out_real = cupy.ndarray(
                        (n, naux, 2), dtype='f8', memptr=out[ji].data)
                    contraction('abi', j3c_real, 'bB',
                                auxcoeff[kq], 'aBi', out_real)
        return out

    def config(self, kpts, jktype):
        if self.isbuild or self.isconfig:
            self.clean()
        self.isconfig = True
        self.isbuild = False

        self.kpts = kpts
        self.jktype = jktype
        df = self.df
        nkpts = len(kpts)
        if jktype == 'j':
            kptij_lst = numpy.repeat(kpts, 2, axis=0).reshape(nkpts, 2, 3)
        elif jktype == 'k':
            kptij_lst = get_kptij_lst(kpts, ksym='s1')
        else:
            raise ValueError("Invalid jktype(%s) is passed!" % str(jktype))
        self.kptij_lst = kptij_lst

        self.vhfopt = _VHFOpt(df.cell, 'int2e')
        self.auxvhfopt = _VHFOpt(df.auxcell, 'int2e')
        omega = df.omega
        self.vhfopt.mol._env[mol_gto.PTR_RANGE_OMEGA] = abs(omega)
        self.auxvhfopt.mol._env[mol_gto.PTR_RANGE_OMEGA] = abs(omega)

    def build(self):
        assert self.isconfig, "Not config yet!"
        self.isbuild = True

    def __call__(self, *args, **kargs):
        return self.int3c(*args, **kargs)

    def __del__(self):
        self.clean()


class Int3cCal(Int3cEngine):
    def __init__(self, df, verbose=None):
        Int3cEngine.__init__(self, df, verbose=verbose)
        self.prescreen_data = get_prescreening_data(
            df.cell, df.auxcell, df.omega, precision=df.precision_R,
            verbose=self.verbose)

        self.b = df.cell.reciprocal_vectors()
        self.Gv, self.Gvbase, _ = df.cell.get_Gv_weights(df.mesh_compact)
        self.gxyz = lib.cartesian_prod(
            [numpy.arange(len(x)) for x in self.Gvbase])

    def config(self, kpts, jktype):
        Int3cEngine.config(self, kpts, jktype)
        df = self.df
        Ls = self.vhfopt.mol.get_lattice_Ls()
        self.Ls = Ls[numpy.linalg.norm(Ls, axis=1).argsort()]  # lr
        self.wcoulG_lr = []  # lr
        self.adapted_ji_idx = []  # g0
        for uniq_q in loop_uniq_q(df, kptij_lst=self.kptij_lst, verbose=0):
            kpt, _, adapted_ji_idx = uniq_q
            wcoulG_lr = df.weighted_coulG(
                omega=df.omega, kpt=kpt, exx=False, mesh=df.mesh_compact)
            self.wcoulG_lr.append(wcoulG_lr)
            if is_zero(kpt):
                self.adapted_ji_idx.append(adapted_ji_idx)

    def build(self, bvk_kmesh, prescreen_mask=None, j3c_mask=3,
              mem_a=0, mem_b=0, mem_c=0, outfrombuf=False):
        Int3cEngine.build(self)
        vhfopt = self.vhfopt
        auxvhfopt = self.auxvhfopt
        df = self.df

        if prescreen_mask is None:
            prescreen_mask = df.prescreen_mask
        log = logger.new_logger(df, verbose=self.verbose)

        if hasattr(bvk_kmesh, '__len__') and len(bvk_kmesh) == 2:
            bvk_kmesh_R, bvk_kmesh_G = bvk_kmesh
        else:
            bvk_kmesh_R = bvk_kmesh_G = bvk_kmesh
        log.debug1('Using bvk_kmesh_R= %s  bvk_kmesh_G= %s',
                   bvk_kmesh_R, bvk_kmesh_G)

        nao_cart = vhfopt.mol.nao_nr(cart=True)
        naux_cart = auxvhfopt.mol.nao_nr(cart=True)
        nkpts = len(self.kpts)

        mem_avail = gpu_avail_bytes(0.9)

        a = float(mem_a)
        b = float(mem_b)
        # nkpts*nao_cart*nao_cart*16/2 for self.ovlp_mg
        mem_c += nkpts * nao_cart * nao_cart * 8
        c = float(mem_c - mem_avail)

        if gamma_point(self.kptij_lst):  # g
            a += 8 * 1.5 * (naux_cart)
        else:
            if self.jktype == 'j':  # k
                bvk_nimgs = numpy.prod(bvk_kmesh_R)
                a += ((naux_cart)
                      * nkpts * 2 + bvk_nimgs * naux_cart) * 8
            else:  # kk
                bvk_nimgs = numpy.prod(bvk_kmesh_R)
                a += ((naux_cart) * nkpts * nkpts * 2 + min(bvk_nimgs, 80)
                      * naux_cart * nkpts * 2 + bvk_nimgs * naux_cart) * 8
        blksize = (numpy.sqrt(b * b - 4 * a * c) - b) / 2 / a
        blksize = min(int(blksize * 0.8), naux_cart)
        self.mem_avail = mem_avail - int(mem_a * blksize * blksize)
        self.mem_avail -= int(mem_b * blksize + mem_c)
        self.blksize = blksize
        vhfopt.build(group_size=blksize)
        auxvhfopt.build_aux(diag_block_with_triu=True)

        # sr
        gint3c = self.get_sr(prescreen_mask, bvk_kmesh_R,
                             diag_block_with_triu=True)
        # g0
        g0 = numpy.pi / df.omega**2. / df.cell.vol
        qaux = get_aux_chg(auxvhfopt.mol)
        self.vbar = qaux * g0
        ind = vhfopt.ao_loc[vhfopt.l_ctr_offsets]
        self.ovlp = []
        for q, adapted_kptjs, _ in loop_uniq_q(df, kptij_lst=self.kptij_lst,
                                               verbose=0):
            if is_zero(q):
                ovlp = vhfopt.mol.pbc_intor(
                    'int1e_ovlp', hermi=1, kpts=adapted_kptjs)
                ovlp_ = []
                for i in range(len(ind) - 1):
                    for j in range(i + 1):
                        ovlp_.append(numpy.asarray(
                            [s[ind[i]:ind[i + 1], ind[j]:ind[j + 1]].reshape(
                                -1) for s in ovlp]))
                self.ovlp.append(ovlp_)
        # lr
        mem_avail = (self.mem_avail - self.out_size
                     - nkpts * nao_cart * nao_cart * 8) * 0.9
        Gblksize = max(16, int(numpy.floor(mem_avail / (
            blksize * blksize * len(self.kptij_lst) + naux_cart) / 16)))
        self.Gblksize = Gblksize
        if bvk_kmesh_R is not None:
            bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(
                vhfopt.mol, bvk_kmesh_R)
            ovlp_mask = gint_ft_ao._estimate_overlap(
                self.vhfopt.mol, self.Ls) > self.df.cell.precision
            self.ovlp_mask = numpy.asarray(
                ovlp_mask, dtype=numpy.int8, order='C')
            translations = numpy.linalg.solve(
                vhfopt.mol.lattice_vectors().T, self.Ls.T)
            t_mod = translations.round(3).astype(
                int) % numpy.asarray(bvk_kmesh_R)[:, None]
            self.cell_loc_bvk = numpy.ravel_multi_index(
                t_mod, bvk_kmesh_R).astype(numpy.int32)
        ngrids = self.gxyz.shape[0]
        self.GvTq = []
        self.bq = []
        self.expkL = []
        for q, adapted_kptjs, _ in loop_uniq_q(df, kptij_lst=self.kptij_lst,
                                               verbose=0):
            self.GvTq.append([numpy.asarray(self.Gv[p0:p1].T, order='C')
                              + q.reshape(-1, 1) for p0, p1 in lib.prange(
                0, ngrids, Gblksize)])
            self.bq.append(numpy.hstack((self.b.ravel(), q) + self.Gvbase))
            if bvk_kmesh_R is None:
                self.expkL.append(
                    numpy.exp(1j * adapted_kptjs.dot(self.Ls.T)))
            else:
                self.expkL.append(
                    numpy.exp(1j * adapted_kptjs.dot(bvkmesh_Ls.T)))
        self.gxyzT = [numpy.asarray(
            self.gxyz[p0:p1].T, order='C', dtype=numpy.int32)
            for p0, p1 in lib.prange(0, ngrids, Gblksize)]
        self.atm_bas_env_lr = gto.conc_env(
            vhfopt.mol._atm, vhfopt.mol._bas, vhfopt.mol._env, vhfopt.mol._atm,
            vhfopt.mol._bas, vhfopt.mol._env)
        self.ao_loc_lr = gto.moleintor.make_loc(
            self.atm_bas_env_lr[1], 'GTO_ft_ovlp_cart')
        ghost_atm = numpy.array([[0, 0, 0, 0, 0, 0]], dtype=numpy.int32)
        ghost_bas = numpy.array(
            [[0, 0, 1, 1, 0, 0, 3, 0]], dtype=numpy.int32)
        ghost_env = numpy.asarray([0, 0, 0, numpy.sqrt(4 * numpy.pi)])
        mol = self.auxvhfopt.mol
        self.atm_bas_env_lr_ghost = gto.conc_env(
            mol._atm, mol._bas, mol._env, ghost_atm, ghost_bas, ghost_env)
        ao_loc = mol.ao_loc_nr()
        self.ao_loc_lr_ghost = numpy.asarray(numpy.hstack(
            (ao_loc, [ao_loc[-1] + 1])), dtype=numpy.int32)

        self.initGPU(bvk_kmesh_R, outfrombuf=outfrombuf)

        def f(cp_ij_id):
            log = logger.new_logger(df, self.verbose)
            t0 = (logger.process_clock(), logger.perf_counter())
            cupy.cuda.nvtx.RangePush("get_j3c[%s] sr" % (cp_ij_id))
            j3c_cupy = gint3c(cp_ij_id)
            cupy.cuda.nvtx.RangePop()
            t0 = log.timer_debug1("get_j3c[%s] sr" % (cp_ij_id), *t0)

            if j3c_mask & 1 != 0:
                cupy.cuda.nvtx.RangePush("get_j3c[%s] g0" % (cp_ij_id))
                self.remove_G0(cp_ij_id, j3c_cupy)
                cupy.cuda.nvtx.RangePop()
                t0 = log.timer_debug1("get_j3c[%s] g0" % (cp_ij_id), *t0)
            if j3c_mask & 2 != 0:
                cupy.cuda.nvtx.RangePush("get_j3c[%s] lr" % (cp_ij_id))
                self.add_lr(cp_ij_id, j3c_cupy, bvk_kmesh_G)
                cupy.cuda.nvtx.RangePop()
                t0 = log.timer_debug1("get_j3c[%s] lr" % (cp_ij_id), *t0)

            return j3c_cupy
        self.int3c = f

    def initGPU(self, bvk_kmesh, outfrombuf=False):
        # sr
        if outfrombuf:
            self.buf_mg = Mg.mapgpu(lambda: ArrayBuffer(mem=self.mem_avail))
            self.out_mg = Mg.mapgpu(
                lambda buf: buf.empty(self.out_size, dtype=cupy.int8),
                self.buf_mg)
        else:
            self.buf_mg = Mg.mapgpu(lambda: ArrayBuffer(
                mem=self.mem_avail - self.out_size))
            self.out_mg = Mg.mapgpu(lambda: cupy.empty(
                self.out_size, dtype=cupy.int8))
        self.expkL_sr_mg = Mg.mapgpu(
            lambda buf: buf.asarray(self.expkL_sr), self.buf_mg)
        # g0
        self.vbar_mg = Mg.mapgpu(
            lambda buf: buf.asarray(self.vbar), self.buf_mg)
        self.adapted_ji_idx_mg = Mg.mapgpu(
            lambda buf: [buf.asarray(x) for x in self.adapted_ji_idx],
            self.buf_mg)
        self.ovlp_mg = Mg.mapgpu(
            lambda buf: [[buf.asarray(x) for x in xs] for xs in self.ovlp],
            self.buf_mg)
        # lr
        self.Ls_mg = Mg.mapgpu(lambda buf: buf.asarray(self.Ls), self.buf_mg)
        self.wcoulG_lr_mg = Mg.mapgpu(
            lambda buf: [buf.asarray(x) for x in self.wcoulG_lr], self.buf_mg)
        if bvk_kmesh is not None:
            self.ovlp_mask_mg = Mg.mapgpu(
                lambda buf: buf.asarray(self.ovlp_mask), self.buf_mg)
            self.cell_loc_bvk_mg = Mg.mapgpu(
                lambda buf: buf.asarray(self.cell_loc_bvk), self.buf_mg)
        self.GvTq_mg = Mg.mapgpu(
            lambda buf: [[buf.asarray(x) for x in xs] for xs in self.GvTq],
            self.buf_mg)
        self.bq_mg = Mg.mapgpu(
            lambda buf: [buf.asarray(x) for x in self.bq], self.buf_mg)
        self.expkL_lr_mg = Mg.mapgpu(
            lambda buf: [buf.asarray(x) for x in self.expkL], self.buf_mg)
        self.gxyzT_mg = Mg.mapgpu(
            lambda buf: [buf.asarray(x) for x in self.gxyzT], self.buf_mg)
        self.ao_loc_lr_mg = Mg.mapgpu(
            lambda buf: buf.asarray(self.ao_loc_lr), self.buf_mg)
        self.ao_loc_lr_ghost_mg = Mg.mapgpu(
            lambda buf: buf.asarray(self.ao_loc_lr_ghost), self.buf_mg)

    def get_sr(self, prescreen_mask, bvk_kmesh, diag_block_with_triu):
        vhfopt = self.vhfopt
        auxvhfopt = self.auxvhfopt
        cell = vhfopt.mol
        auxcell = auxvhfopt.mol

        refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp, dcut2s, \
            dstep_BOHR, Rcut2s, dijs_loc, Ls = self.prescreen_data
        refuniqshl_map = refuniqshl_map[vhfopt.sorted_idx]
        auxuniqshl_map = auxuniqshl_map[auxvhfopt.sorted_idx]

        kptij_lst = self.kptij_lst
        nimgs = len(Ls)
        nbas = cell.nbas
        auxnbas = auxcell.nbas
        naux_cart = auxcell.nao_nr(cart=True)
        blksize = self.blksize
        kpti = kptij_lst[:, 0]
        kptj = kptij_lst[:, 1]
        ao_loc = cell.ao_loc
        auxao_loc = auxcell.ao_loc
        if bvk_kmesh is None:
            Ls_ = Ls
        else:
            Ls, Ls_, cell_loc_bvk = _get_bvk_data(cell, Ls, bvk_kmesh)
            bvk_nimgs = Ls_.shape[0]

        if gamma_point(kptij_lst):
            kk_type = 'g'
            dtype = numpy.double
            nkpts = nkptij = 1
            kptij_idx = numpy.array([0], dtype=numpy.int32)
            expkL = numpy.ones(1)
        elif is_zero(kpti - kptj):  # j_only
            kk_type = 'k'
            dtype = numpy.complex128
            kpts = kptij_idx = numpy.asarray(kpti, order='C')
            expkL = numpy.exp(1j * numpy.dot(kpts, Ls_.T))
            nkpts = nkptij = len(kpts)
        else:
            kk_type = 'kk'
            dtype = numpy.complex128
            kpts = unique(numpy.vstack([kpti, kptj]))[0]
            expkL = numpy.exp(1j * numpy.dot(kpts, Ls_.T))
            wherei = numpy.where(abs(kpti.reshape(-1, 1, 3) - kpts).sum(axis=2)
                                 < KPT_DIFF_TOL)[1]
            wherej = numpy.where(abs(kptj.reshape(-1, 1, 3) - kpts).sum(axis=2)
                                 < KPT_DIFF_TOL)[1]
            nkpts = len(kpts)
            kptij_idx = numpy.asarray(
                wherei * nkpts + wherej, dtype=numpy.int32)
            nkptij = len(kptij_lst)

        cfunc_prefix = "GINTPBCsr3c"
        if not (gamma_point(kptij_lst) or bvk_kmesh is None):
            cfunc_prefix += "_bvk"
        drv = getattr(libgpbc, "%s_%s_drv" % (cfunc_prefix, kk_type))

        if diag_block_with_triu:
            scale_shellpair_diag = 1.
        else:
            scale_shellpair_diag = 0.5
        l_ctr_offsets = vhfopt.l_ctr_offsets
        _atm, _bas, _env = mol_gto.conc_env(
            cell._atm, cell._bas, cell._env, auxcell._atm, auxcell._bas,
            auxcell._env)

        maxref = max(refuniqshl_map)
        maxref2 = maxref * (maxref + 3) // 2
        aux_id_range = auxvhfopt.ncptype
        blksize2 = blksize * blksize
        self.expkL_sr = expkL
        shlpr_mask = numpy.ones((cell.nbas, cell.nbas),
                                dtype=numpy.int8, order="C")
        if gamma_point(kptij_lst):
            buf_size = [vhfopt.bas_pairs_locs[-1] * 8,  # d_bas_pair2bra(ket)
                        (nbas + auxnbas) * 32,  # d_bas
                        cell.natm * 48,  # d_atm
                        _env.size * 8,  # d_env
                        Ls.size * 8,  # d_Ls
                        nbas * 4,  # d_refuniqshl_map
                        auxnbas * 4,  # d_auxuniqshl_map
                        maxref * 8 + 8,  # d_uniqexp
                        (maxref2 + 1) * 8,  # d_uniq_dcut2s
                        # d_uniq_Rcut2s
                        nbasauxuniq * dijs_loc[maxref2 + 1] * 8,
                        (maxref2 + 2) * 4,  # d_uniqshlpr_dij_loc
                        (nbas + 1) * 4,  # d_ao_loc
                        aux_id_range * 6 * (66**3)  # tmp_idx4c
                        ]
            buf_size = sum(buf_size) + 256 * (13 + aux_id_range)
            self.out_size = blksize2 * naux_cart * 8

            def int3c(cp_ij_id):
                cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
                cpj = cp_ij_id - cpi * (cpi + 1) // 2
                ish0 = l_ctr_offsets[cpi]
                ish1 = l_ctr_offsets[cpi + 1]
                jsh0 = l_ctr_offsets[cpj]
                jsh1 = l_ctr_offsets[cpj + 1]
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                dij = (i1 - i0) * (j1 - j0)
                shls_slice = (ish0, ish1, jsh0, jsh1, 0, auxnbas)
                gid = Mg.getgid()
                out = cupy.ndarray((1, 1, dij, naux_cart),
                                   dtype='f8', memptr=self.out_mg[gid].data)
                self.buf_mg[gid].check(buf_size)
                assert out.dtype == dtype
                drv(ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    ctypes.cast(self.buf_mg[gid].data.ptr, ctypes.c_void_p),
                    ctypes.c_ulong(self.buf_mg[gid].bufsize),
                    ctypes.c_int(1), ctypes.c_int(nimgs),
                    Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(Ls.size),
                    (ctypes.c_int * 6)(*shls_slice),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    auxao_loc.ctypes.data_as(ctypes.c_void_p),
                    shlpr_mask.ctypes.data_as(ctypes.c_void_p),
                    refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(max(refuniqshl_map)),
                    ctypes.c_int(nbasauxuniq),
                    uniqexp.ctypes.data_as(ctypes.c_void_p),
                    dcut2s.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(dstep_BOHR),
                    Rcut2s.ctypes.data_as(ctypes.c_void_p),
                    dijs_loc.ctypes.data_as(ctypes.c_void_p),
                    _atm.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(cell.natm),
                    _bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbas),
                    _env.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(_env.size), ctypes.c_int(cp_ij_id),
                    ctypes.c_int(aux_id_range),
                    vhfopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                    vhfopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(vhfopt.ncptype), auxvhfopt.bpcaches[gid],
                    ctypes.c_double(scale_shellpair_diag),
                    ctypes.c_int(prescreen_mask))
                return out

        else:
            if kk_type == 'k':
                buf_size = [bvk_nimgs * nkpts * 16,  # d_bufexp
                            bvk_nimgs * blksize2 * naux_cart * 8,  # !!!d_bufL
                            # d_bas_pair2bra
                            vhfopt.bas_pairs_locs[-1] * 8,
                            (nbas + auxnbas) * 32,  # d_bas
                            cell.natm * 48,  # d_atm
                            _env.size * 8,  # d_env
                            Ls.size * 8,  # d_Ls
                            nbas * 4,  # d_refuniqshl_map
                            auxnbas * 4,  # d_auxuniqshl_map
                            maxref * 8 + 8,  # d_uniqexp
                            (maxref2 + 1) * 8,  # d_uniq_dcut2s
                            # d_uniq_Rcut2s
                            nbasauxuniq * dijs_loc[maxref2 + 1] * 8,
                            (maxref2 + 2) * 4,  # d_uniqshlpr_dij_loc
                            (nbas + 1) * 4,  # d_ao_loc
                            aux_id_range * 6 * (66**3)  # tmp_idx4c
                            ]
                buf_size = sum(buf_size) + 256 * (15 + aux_id_range)
                self.out_size = nkpts * blksize2 * naux_cart * 16
            else:
                buf_size = [min(bvk_nimgs, 80) * nkpts * blksize2
                            * naux_cart * 16,  # !!!d_bufkL
                            bvk_nimgs * blksize2 * naux_cart * 8,  # !!!d_bufL
                            # d_bas_pair2bra
                            vhfopt.bas_pairs_locs[-1] * 8,
                            (nbas + auxnbas) * 32,  # d_bas
                            cell.natm * 48,  # d_atm
                            _env.size * 8,  # d_env
                            Ls.size * 8,  # d_Ls
                            nbas * 4,  # d_refuniqshl_map
                            auxnbas * 4,  # d_auxuniqshl_map
                            maxref * 8 + 8,  # d_uniqexp
                            (maxref2 + 1) * 8,  # d_uniq_dcut2s
                            # d_uniq_Rcut2s
                            nbasauxuniq * dijs_loc[maxref2 + 1] * 8,
                            (maxref2 + 2) * 4,  # d_uniqshlpr_dij_loc
                            (nbas + 1) * 4,  # d_ao_loc
                            aux_id_range * 6 * (66**3)]  # tmp_idx4c
                buf_size = sum(buf_size) + 256 * (15 + aux_id_range)
                self.out_size = nkpts * nkpts * blksize2 * naux_cart * 16

            def int3c(cp_ij_id):
                cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
                cpj = cp_ij_id - cpi * (cpi + 1) // 2
                ish0 = l_ctr_offsets[cpi]
                ish1 = l_ctr_offsets[cpi + 1]
                jsh0 = l_ctr_offsets[cpj]
                jsh1 = l_ctr_offsets[cpj + 1]
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                dij = (i1 - i0) * (j1 - j0)
                shls_slice = (ish0, ish1, jsh0, jsh1, 0, auxnbas)
                gid = Mg.getgid()
                if kk_type == 'k':
                    out = cupy.ndarray(
                        (nkpts, 1, dij, naux_cart), dtype='complex128',
                        memptr=self.out_mg[gid].data)
                else:
                    out = cupy.ndarray(
                        (nkpts * nkpts, 1, dij, naux_cart), dtype='complex128',
                        memptr=self.out_mg[gid].data)
                self.buf_mg[gid].check(buf_size)

                drv(ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    ctypes.cast(self.buf_mg[gid].data.ptr, ctypes.c_void_p),
                    ctypes.c_ulong(self.buf_mg[gid].bufsize),
                    ctypes.c_int(nkptij), ctypes.c_int(nkpts),
                    ctypes.c_int(1), ctypes.c_int(nimgs),
                    ctypes.c_int(bvk_nimgs),
                    Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(Ls.size),
                    ctypes.cast(
                        self.expkL_sr_mg[gid].data.ptr, ctypes.c_void_p),
                    kptij_idx.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int * 6)(*shls_slice),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    auxao_loc.ctypes.data_as(ctypes.c_void_p),
                    cell_loc_bvk.ctypes.data_as(
                        ctypes.c_void_p),  # cell_loc_bvk
                    shlpr_mask.ctypes.data_as(ctypes.c_void_p),   # shlpr_mask
                    refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(max(refuniqshl_map)),
                    ctypes.c_int(nbasauxuniq),
                    uniqexp.ctypes.data_as(ctypes.c_void_p),
                    dcut2s.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(dstep_BOHR),
                    Rcut2s.ctypes.data_as(ctypes.c_void_p),
                    dijs_loc.ctypes.data_as(ctypes.c_void_p),
                    _atm.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(cell.natm),
                    _bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbas),
                    _env.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(_env.size),
                    ctypes.c_int(cp_ij_id), ctypes.c_int(aux_id_range),
                    vhfopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                    vhfopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(vhfopt.ncptype), auxvhfopt.bpcaches[gid],
                    ctypes.c_double(scale_shellpair_diag),
                    ctypes.c_int(prescreen_mask))
                return out
        return int3c

    def remove_G0(self, cp_ij_id, j3c):
        gid = Mg.getgid()
        vbar = self.vbar_mg[gid]
        for i, adapted_ji_idx in enumerate(self.adapted_ji_idx_mg[gid]):
            ovlp = self.ovlp_mg[gid][i][cp_ij_id]
            nij = ovlp.shape[1]
            if j3c.dtype == cupy.float64:
                j3c_g0_real_kernel((int((adapted_ji_idx.size * nij)),), (256,),
                                   (j3c, vbar, ovlp, adapted_ji_idx,
                                   adapted_ji_idx.size, nij, vbar.size))
            else:
                j3c_g0_kernel((int((adapted_ji_idx.size * nij)),), (256,),
                              (j3c, vbar, ovlp, adapted_ji_idx,
                              adapted_ji_idx.size, nij, vbar.size))
        return j3c

    def add_lr(self, cp_ij_id, j3c, bvk_kmesh):
        gid = Mg.getgid()
        ngrids = self.gxyz.shape[0]
        df = self.df
        vhfopt = self.vhfopt
        l_ctr_offsets = vhfopt.l_ctr_offsets
        naux_cart = self.auxvhfopt.mol.nao_nr(cart=True)
        cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
        cpj = cp_ij_id - cpi * (cpi + 1) // 2
        ish0 = l_ctr_offsets[cpi]
        ish1 = l_ctr_offsets[cpi + 1]
        jsh0 = l_ctr_offsets[cpj]
        jsh1 = l_ctr_offsets[cpj + 1]
        shls_slice = (ish0, ish1, jsh0, jsh1, 0, df.auxcell.nbas)
        ao_loc = vhfopt.ao_loc
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        ncol = ni * nj

        if abs(self.b - numpy.diag(self.b.diagonal())).sum() < 1e-8:
            eval_gz_fg = 1  # GTO_Gv_orth
        else:
            eval_gz_fg = 2  # GTO_Gv_nonorth

        kq = 0

        buf = self.buf_mg[gid]
        buf.tag('lr')
        out = buf.empty((len(self.kptij_lst), self.blksize, self.blksize,
                         self.Gblksize), dtype='complex128')
        out2 = buf.empty((naux_cart, self.Gblksize), dtype='complex128')
        for kpt, adapted_kptjs, adapted_ji_idx in loop_uniq_q(
                self.df, kptij_lst=self.kptij_lst, verbose=0):
            count = 0
            for p0, p1 in lib.prange(0, ngrids, self.Gblksize):
                dat_cupy = self.ft_aopair_kpts(
                    self.GvTq_mg[gid][kq][count], shls_slice=shls_slice,
                    b=self.bq_mg[gid][kq], gxyzT=self.gxyzT_mg[gid][count],
                    kptjs=adapted_kptjs, bvk_kmesh=bvk_kmesh, kq=kq,
                    eval_gz_fg=eval_gz_fg, out=out, bufptr=buf.data)
                nG = p1 - p0

                gL_T = self.ft_ao_trans(
                    self.GvTq_mg[gid][kq][count], kpt,
                    gxyzT=self.gxyzT_mg[gid][count], b=self.bq_mg[gid][kq],
                    eval_gz=eval_gz_fg, out=out2, bufptr=buf.data)
                gL_T *= self.wcoulG_lr_mg[gid][kq][p0:p1].reshape(1, -1)
                for k, ji in enumerate(adapted_ji_idx):
                    pqg = dat_cupy[k].reshape(ncol, nG)

                    _j3c = j3c[ji][0]
                    if not cupy.isrealobj(j3c):
                        contraction('ig', pqg, 'ag',
                                    gL_T, 'ia', _j3c, beta=1, opb='CONJ')
                    else:
                        pqgreal = cupy.ndarray(
                            (ncol, nG, 2), dtype='f8', memptr=pqg.data)
                        gL_T_real = cupy.ndarray(
                            (naux_cart, nG, 2), dtype='f8', memptr=gL_T.data)
                        blksize = min(
                            nG, (2**31) // max(naux_cart, ncol) - 1)
                        for i in range(0, nG, blksize):
                            s = slice(i, i + blksize)
                            contraction(
                                'ig', pqgreal[:, s, 0], 'ag',
                                gL_T_real[:, s, 0], 'ia', _j3c, beta=1)
                            contraction(
                                'ig', pqgreal[:, s, 1], 'ag',
                                gL_T_real[:, s, 1], 'ia', _j3c, beta=1)
                        _j3c = None
                    pqg = pqgreal = None
                gL_T = gL_T_real = None
                count += 1
            kq += 1
        buf.untag('lr')
        return j3c

    def ft_aopair_kpts(self, GvTq, eval_gz_fg, out, shls_slice,
                       b, gxyzT,
                       kptjs=numpy.zeros((1, 3)),
                       bvk_kmesh=None, kq=0, bufptr=None):
        r'''
        Fourier transform AO pair for a group of k-points
        \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

        The return array holds the AO pair
        corresponding to the kpoints given by kptjs
        '''
        gid = Mg.getgid()
        cell = self.vhfopt.mol
        kptjs = numpy.asarray(kptjs, order='C').reshape(-1, 3)
        nGv = GvTq.shape[1]
        GvT = GvTq
        p_gxyzT = ctypes.cast(gxyzT.data.ptr, ctypes.c_void_p)
        p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
        p_mesh = (ctypes.c_int * 3)(*[len(x) for x in self.Gvbase])
        expkL = self.expkL_lr_mg[gid][kq]
        Ls = self.Ls_mg[gid]
        nimgs = len(Ls)
        nkpts = len(kptjs)
        nbas = cell.nbas
        atm, bas, env = self.atm_bas_env_lr
        ao_loc = self.ao_loc_lr
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas + shls_slice[2], nbas + shls_slice[3])
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        shape = (nkpts, ni, nj, nGv)

        out = cupy.ndarray(shape, dtype=numpy.complex128, memptr=out.data)
        bufsize = env.nbytes + bas.nbytes + atm.nbytes + 1600
        gpu_buf = cupy.ndarray(bufsize, dtype=numpy.int8, memptr=bufptr)
        ao_loc = self.ao_loc_lr_mg[gid]

        if bvk_kmesh is None:
            drv = libgaft.PBC_ft_latsum_drv
            drv(ctypes.c_int(eval_gz_fg),
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
                ctypes.c_ulong(bufsize),
                ctypes.c_int(nkpts), ctypes.c_int(1), ctypes.c_int(nimgs),
                ctypes.cast(Ls.data.ptr, ctypes.c_void_p),
                ctypes.cast(expkL.data.ptr, ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p), p_b, p_gxyzT,
                p_mesh, ctypes.c_int(nGv),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
                env.ctypes.data_as(ctypes.c_void_p))
        else:
            ovlp_mask = self.ovlp_mask_mg[gid]
            cell_loc_bvk = self.cell_loc_bvk_mg[gid]
            drv = libgaft.PBC_ft_bvk_drv
            drv(ctypes.c_int(eval_gz_fg),
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
                ctypes.c_ulong(bufsize),
                ctypes.c_int(nkpts), ctypes.c_int(1), ctypes.c_int(nimgs),
                ctypes.c_int(expkL.shape[1]),
                ctypes.cast(Ls.data.ptr, ctypes.c_void_p),
                ctypes.cast(expkL.data.ptr, ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(cell_loc_bvk.data.ptr, ctypes.c_void_p),
                ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p), p_b, p_gxyzT,
                p_mesh, ctypes.c_int(nGv),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
                env.ctypes.data_as(ctypes.c_void_p))
        return out

    def ft_ao_trans(self, GvTq, kpt, gxyzT, b, eval_gz, out, bufptr=None):
        gid = Mg.getgid()
        auxvhfopt = self.auxvhfopt
        mol = auxvhfopt.mol
        nGv = GvTq.shape[1]
        GvT = GvTq

        if gamma_point(kpt):
            p_gxyzT = ctypes.cast(gxyzT.data.ptr, ctypes.c_void_p)
            p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
            p_gs = (ctypes.c_int * 3)(*[len(x) for x in self.Gvbase])
        else:
            p_gxyzT = lib.c_null_ptr()
            p_gs = (ctypes.c_int * 3)(0, 0, 0)
            b = cupy.asarray([0], dtype='f8')
            p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
            eval_gz = 0  # 'GTO_Gv_general'

        fn = libgaft.GTO_ft_fill_drv
        assert mol.cart  # GTO_ft_ovlp_cart

        atm, bas, env = self.atm_bas_env_lr_ghost
        nao = mol.nao_nr()
        shape = (nao, nGv)
        phase = 0
        l_ctr_offsets = auxvhfopt.l_ctr_offsets
        bufsize = env.nbytes + bas.nbytes + atm.nbytes + 1600
        gpu_buf = cupy.ndarray(bufsize, dtype=numpy.int8, memptr=bufptr)
        mat = cupy.ndarray(shape, order='C',
                           dtype=numpy.complex128, memptr=out.data)
        mat[:] = 0
        ao_loc = self.ao_loc_lr_ghost
        ao_loc_cupy = self.ao_loc_lr_ghost_mg[gid]

        for i in range(auxvhfopt.ncptype):
            ish0 = l_ctr_offsets[i]
            ish1 = l_ctr_offsets[i + 1]
            shls_slice = (ish0, ish1, mol.nbas, mol.nbas + 1)
            fn(eval_gz,
               ctypes.cast(mat[ao_loc[ish0]:].data.ptr, ctypes.c_void_p),
               ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
               ctypes.c_ulong(bufsize), ctypes.c_int(1),
               (ctypes.c_int * 4)(*shls_slice),
               ctypes.cast(ao_loc_cupy.data.ptr, ctypes.c_void_p),
               ctypes.c_double(phase),
               ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
               p_b, p_gxyzT, p_gs, ctypes.c_int(nGv),
               atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
               bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
               env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return mat

    def clean(self):
        self.expkL_sr_mg = None
        self.out_mg = None
        self.auxcoeff_mg = None
        self.vbar_mg = None
        self.wcoulG_lr_mg = None
        self.adapted_ji_idx_mg = None
        self.ovlp_mg = None
        self.GvTq_mg = None
        self.gxyzT_mg = None
        self.bq_mg = None
        self.Ls_mg = None
        self.ao_loc_lr_mg = None
        self.ao_loc_lr_ghost_mg = None
        self.expkL_lr_mg = None
        self.ovlp_mask_mg = None
        self.cell_loc_bvk_mg = None
        self.buf_mg = None
        self.isbuild = False
        self.isconfig = False
        self.vhfopt = None
        self.auxvhfopt = None
        self.int3c = lambda *x, **y: None


class Int3cFile(Int3cEngine):
    def __init__(self, df, verbose=None, cderi=None):
        Int3cEngine.__init__(self, df, verbose=verbose)
        self.cderi = cderi
        self.feri = h5py.File(self.cderi, 'r')
        fjktype, self.fauxflag, self.fblksize = self.feri['j3c-config']
        self.fjktype = ['j', 'k'][fjktype]
        assert self.fauxflag != 1, (
            'The ERI in file is calculated with `auxflag==1`, which is '
            'not suitable for out-core calculaitons currently.')

    def config(self, kpts, jktype):
        Int3cEngine.config(self, kpts, jktype)
        if self.fjktype == 'j' and self.jktype == 'k':
            assert False, ('cderi:%s is for "j" type, which cannot be used to '
                           'calculate "j" type')

    def build(self, *args, auxflag=2, **kwargs):
        Int3cEngine.build(self)
        assert auxflag >= self.fauxflag, (
            'The ERI in file is calculated with `auxflag==%d`, which is not '
            'suitable for out-core calculaitons with `auxflag==%d`.' % (
                self.fauxflag, auxflag))
        self.auxflag = auxflag
        vhfopt = self.vhfopt
        auxvhfopt = self.auxvhfopt
        nkpts = len(self.kpts)

        self.blksize = self.fblksize
        vhfopt.build(group_size=self.blksize)
        auxvhfopt.build_aux(diag_block_with_triu=True)

        if self.fauxflag == 0 and auxflag > 0:
            if auxflag == 1:
                self.auxcoeff_mg = Mg.broadcast(self.auxvhfopt.coeff)
            else:
                self.auxcoeff_mg = Mg.broadcast(self.j2c_coeff)

        self.feri = h5py.File(self.cderi, "r")

        def f(cp_ij_id):
            if self.fjktype == 'k' and self.jktype == 'j':
                inds = numpy.asarray([i * (nkpts + 1) for i in range(nkpts)])
                j3c_cupy = cupy.asarray(
                    self.feri['j3c/%d' % cp_ij_id][inds])
            else:
                j3c_cupy = cupy.asarray(self.feri['j3c/%d' % cp_ij_id])
            if self.fauxflag == 0 and auxflag > 0:
                gid = Mg.getgid()
                j3c_cupy = self.apply_aux_coeff(
                    j3c_cupy, self.auxcoeff_mg[gid])
            return j3c_cupy
        self.int3c = f

    def clean(self):
        self.auxcoeff_mg = None
        if hasattr(self, 'feri'):
            self.feri.close()
            del self.feri
