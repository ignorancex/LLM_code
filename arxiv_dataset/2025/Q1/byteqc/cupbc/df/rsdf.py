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
from pyscf.pbc import tools as pbctools
from pyscf.pbc.df import aft
from pyscf.pbc.df.aft import _sub_df_jk_
from pyscf.pbc.df import rsdf
from pyscf.pbc.df.df_jk import _format_kpts_band
from byteqc.cupbc.df.rsdf_direct_helper import (kpts_to_kmesh, get_kptij_lst,
                                                get_tril_indices)
from byteqc.lib import contraction, Mg
from byteqc.cupbc.df.int3c import Int3cCal, Int3cFile
from pyscf.df.outcore import _guess_shell_ranges
import threading
import time

LONGRANGE_AFT_TURNOVER_THRESHOLD = 2.5
REAL = numpy.float64
COMPLEX = numpy.complex128


class RSGDF(rsdf.RSGDF):
    '''Range Separated Gaussian Density Fitting
    '''

    def __init__(self, *args, prescreen_mask=2, **kwargs):
        rsdf.RSGDF.__init__(self, *args, **kwargs)
        self.direct = True
        self.semidirect = False
        self.ksym = 's1'
        self.prescreen_mask = prescreen_mask
        self.omega = 0.1
        self.int3c = None

    def build(self, with_j3c=False, cderi=None, jktype='k', *args):
        rsdf.RSGDF.build(self, *args, with_j3c=False)
        self.int3c = Int3cCal(self)
        if with_j3c:
            if cderi is None:
                if isinstance(self._cderi_to_save, str):
                    cderi = self._cderi_to_save
                else:
                    cderi = self._cderi_to_save.name
                self._cderi = cderi
                self.save_j3c(cderi, jktype=jktype,
                              auxflag=0 if jktype == 'k' else '2')
            else:
                self._cderi = cderi
            self.int3c = Int3cFile(self, cderi=self._cderi)

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):

        if not self.direct:
            from pyscf.pbc.df import df
            return df.GDF.get_jk(
                self, dm, hermi=hermi, kpts=kpts, kpts_band=kpts_band,
                with_j=with_j, with_k=with_k, omega=omega, exxdiv=exxdiv)

        from byteqc.cupbc.df import rsdf_direct_jk
        if omega is not None:
            cell = self.cell
            if (omega < LONGRANGE_AFT_TURNOVER_THRESHOLD
                and cell.dimension >= 2
                    and cell.low_dim_ft_type != 'inf_vacuum'):
                mydf = aft.AFTDF(cell, self.kpts)
                mydf.ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = pbctools.cutoff_to_mesh(
                    cell.lattice_vectors(), mydf.ke_cutoff)
            else:
                mydf = self
            return _sub_df_jk_(mydf, dm, hermi, kpts, kpts_band,
                               with_j, with_k, omega, exxdiv)

        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        kpts = numpy.asarray(kpts)

        if isinstance(self.use_bvk, bool):
            use_bvk_R = use_bvk_G = self.use_bvk
        else:
            use_bvk_R, use_bvk_G = self.use_bvk
        if use_bvk_R or use_bvk_G:
            bvk_kmesh0 = kpts_to_kmesh(self.cell, kpts)
            bvk_kmesh = [bvk_kmesh0 if use_bvk_R else None,
                         bvk_kmesh0 if use_bvk_G else None]
        else:
            bvk_kmesh = None

        semidirect = self.semidirect
        ksym = self.ksym

        if kpts.shape == (3,):
            bvk_kmesh_ = None if kpts_band is None else bvk_kmesh
            return rsdf_direct_jk.get_jk(
                self, dm, hermi, kpts, kpts_band, exxdiv, with_j, with_k,
                bvk_kmesh_, semidirect)

        vj = vk = None

        if with_k:
            cupy.cuda.nvtx.RangePush("get_k", 1)
            vk = rsdf_direct_jk.get_k_kpts(
                self, dm, hermi, kpts, kpts_band, exxdiv, bvk_kmesh=bvk_kmesh,
                semidirect=semidirect, ksym=ksym)
            cupy.cuda.nvtx.RangePop()

        if with_j:
            cupy.cuda.nvtx.RangePush("get_j", 3)
            vj = rsdf_direct_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band,
                                           bvk_kmesh=bvk_kmesh)
            cupy.cuda.nvtx.RangePop()

        return vj, vk

    def j3c_func(self, kpts, jktype='k',
                 kpts_band=None, auxflag=2, j3c_mask=3):
        if kpts is None:
            kpts = self.kpts
        nkpts = len(kpts)
        datasize = 8 if nkpts == 1 else 16
        datatype = 'f8' if nkpts == 1 else 'complex128'
        if self.auxcell is None:
            self.build(with_j3c=False)

        kpts_band = _format_kpts_band(kpts_band, kpts)
        if jktype == 'j':
            kptij_lst = numpy.repeat(kpts, 2, axis=0).reshape(nkpts, 2, 3)
        elif jktype == 'k':
            kptij_lst = get_kptij_lst(kpts, ksym='s1')
        nkij = len(kptij_lst)

        if isinstance(self.use_bvk, bool):
            use_bvk_R = use_bvk_G = self.use_bvk
        else:
            use_bvk_R, use_bvk_G = self.use_bvk
        bvk_kmesh = None
        if use_bvk_R or use_bvk_G:
            bvk_kmesh0 = kpts_to_kmesh(self.cell, kpts)
            bvk_kmesh = [bvk_kmesh0 if use_bvk_R else None,
                         bvk_kmesh0 if use_bvk_G else None]

        self.int3c.config(kpts, jktype)
        naux = self.int3c.auxvhfopt.mol.nao_nr()
        naux_cart = self.int3c.auxvhfopt.mol.nao_nr(cart=True)
        if auxflag != 0:
            self.int3c.build(
                bvk_kmesh, j3c_mask=j3c_mask, mem_a=nkij * naux * datasize,
                mem_c=self.int3c.auxvhfopt.coeff.nbytes)
            blksize = self.int3c.blksize
            out_buf = Mg.mapgpu(lambda: cupy.zeros(
                (nkij, blksize, blksize, naux), dtype=datatype))
            if auxflag == 1:
                auxcoeff = Mg.broadcast(self.int3c.auxvhfopt.coeff)
            elif auxflag == 2:
                auxcoeff = Mg.broadcast(self.int3c.j2c_coeff)
        else:
            self.int3c.build(bvk_kmesh, j3c_mask=j3c_mask)
        vhfopt = self.int3c.vhfopt

        l_ctr_offsets = vhfopt.l_ctr_offsets
        ao_loc = vhfopt.ao_loc

        def j3c(cp_ij_id):
            cpi, cpj = get_tril_indices(cp_ij_id)
            ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
            jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            kcpqL = self.int3c(cp_ij_id).reshape(
                -1, i1 - i0, j1 - j0, naux_cart)

            if auxflag == 0:
                return slice(i0, i1), slice(j0, j1), kcpqL
            else:
                gid = Mg.getgid()
                return slice(i0, i1), slice(j0, j1), \
                    self.int3c.apply_aux_coeff(
                    kcpqL, auxcoeff[gid], buf=out_buf[gid])
        return j3c

    def clean_j3c(self):
        self.int3c.clean()

    def dump_eri(self, cderi_file, kpts=None, buflen=None):
        '''
            Dump eri into file. Follow the format of CPU's.
        '''
        feri = h5py.File(cderi_file, 'w')
        if kpts is None:
            kpts = self.kpts
        nkpts = len(kpts)
        kptij_lst = get_kptij_lst(kpts, ksym='s1')
        feri['j3c-kptij'] = numpy.asarray(kptij_lst)

        nao = self.cell.nao

        j3c_func = self.j3c_func(kpts, auxflag=2)
        vhfopt = self.int3c.vhfopt
        naux = self.auxcell.nao_nr(cart=False)
        coeffs = Mg.broadcast(*vhfopt.coeffs)
        inds = vhfopt.coeffs_inds

        def with_coeff(x, cpi, cpj, issym=False):
            gid = cupy.cuda.runtime.getDevice()
            if issym:
                tmp = contraction(
                    'KkjiL', x, 'iI', coeffs[cpi][gid], 'kKLIj',
                    opa='CONJ' if x.dtype != 'f8' else 'IDENTITY')
            else:
                tmp = contraction('kKijL', x, 'iI', coeffs[cpi][gid], 'kKLIj')
            return contraction('kKLIj', tmp, 'jJ', coeffs[cpj][gid], 'kKLIJ')

        if buflen is None:
            buflen = nao**2
        shranges = _guess_shell_ranges(self.cell, buflen, 's1')
        cols = [sh_range[2] for sh_range in shranges]
        locs = numpy.append(0, numpy.cumsum(cols))
        for istep in range(len(locs) - 1):
            for i in range(nkpts):
                for j in range(nkpts):
                    i0 = locs[istep] // nao
                    i1 = locs[istep + 1] // nao
                    ij = i * nkpts + j
                    feri.create_dataset(
                        'j3c/%d/%d' % (ij, istep),
                        shape=(naux, (i1 - i0) * nao if i != j else (
                            i0 + i1 + 1) * (i1 - i0) // 2),
                        dtype='complex128' if nkpts != 1 else 'f8')

        def task_file(cpi, cpj, j3c):
            indi = inds[cpi]
            indj = inds[cpj]
            for istep in range(len(locs) - 1):
                i0 = locs[istep] // nao
                i1 = locs[istep + 1] // nao
                indi_mask = (indi >= i0) & (indi < i1)
                if indi_mask.any().item():
                    tmp = j3c[:, :, :, indi_mask]
                    real_ind = (
                        (indi[indi_mask] - i0).reshape(-1, 1) * nao
                        + indj.reshape(1, -1)).ravel()
                    diag_mask = indi[indi_mask].reshape(
                        -1, 1) >= indj.reshape(1, -1)
                    real_ind_diag = indi[indi_mask].reshape(-1, 1)
                    real_ind_diag = (
                        real_ind_diag * (real_ind_diag + 1) // 2
                        + indj.reshape(1, -1))[diag_mask].ravel() \
                        - i0 * (i0 + 1) // 2
                    for i in range(nkpts):
                        for j in range(nkpts):
                            ij = i * nkpts + j
                            if i == j:
                                ind = real_ind_diag
                                mask = diag_mask.ravel()
                            else:
                                ind = real_ind
                                mask = slice(None)
                            with Mg.lock((ij, istep)):
                                pass
                                feri['j3c/%d/%d' % (ij, istep)][:, ind] += \
                                    tmp[i, j, :].reshape(naux, -1)[:, mask]

        def task(cp_ij_id):
            cpi, cpj = get_tril_indices(cp_ij_id)
            si, sj, kcpqL = j3c_func(cp_ij_id)
            di = si.stop - si.start
            dj = sj.stop - sj.start
            kcpqL = kcpqL.reshape(nkpts, nkpts, di, dj, naux)
            j3c = with_coeff(kcpqL, cpi, cpj, False).get()
            p = threading.Thread(target=task_file, args=(cpi, cpj, j3c))
            p.start()
            if si != sj:
                j3c = with_coeff(kcpqL, cpj, cpi, True).get()
                p2 = threading.Thread(target=task_file, args=(cpj, cpi, j3c))
                p2.start()
                return (p, p2)
            return p

        start = time.time()
        ps = Mg.map(task, range(vhfopt.ncptype))
        for p in ps:
            if isinstance(p, tuple):
                p[0].join()
                p[1].join()
            else:
                p.join()
        print("Time:", time.time() - start)
        feri.close()

    def save_j3c(self, cderi_file, kpts=None, jktype='k', auxflag=2):
        '''
            Save eri into file. The format is incompatible with Cpu's..
        '''
        feri = h5py.File(cderi_file, 'w')
        if kpts is None:
            kpts = self.kpts

        j3c_func = self.j3c_func(kpts, jktype=jktype, auxflag=auxflag)

        def savetofile(x, cp_ij_id, ):
            x = x.get()
            feri['j3c/%d' % (cp_ij_id,)] = x

        ps = []

        def task(cp_ij_id):
            si, sj, kcpqL = j3c_func(cp_ij_id)
            p = threading.Thread(
                target=savetofile, args=(kcpqL, cp_ij_id))
            ps.append(p)
            p.start()
            return si, sj

        ncptype = self.int3c.vhfopt.ncptype
        indices = Mg.map(task, range(ncptype))
        self.int3c.clean()
        iind = numpy.empty((ncptype, 2), dtype=numpy.int32)
        jind = numpy.empty((ncptype, 2), dtype=numpy.int32)
        for i, (si, sj) in enumerate(indices):
            iind[i] = [si.start, si.stop]
            jind[i] = [sj.start, sj.stop]
        for p in ps:
            p.join()
        feri['j3c-i'] = iind
        feri['j3c-j'] = jind
        feri['j3c-config'] = numpy.asarray(
            [jktype == 'k', auxflag, self.int3c.blksize], dtype=numpy.int32)
        feri.close()


RSDF = RSGDF
