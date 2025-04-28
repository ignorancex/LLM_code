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

import numpy as np
from pyscf.pbc import scf
from pyscf.lib import logger


class KsymAdaptedKRHF(scf.khf_ksymm.KsymAdaptedKRHF):
    """ Nearly-symmetric KSCF class"""

    def rs_density_fit(self, auxbasis=None, with_df=None):
        from byteqc.cupbc.df import rsdf_jk
        return rsdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None,
               kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
        kpts_band = None
        if isinstance(kpts, np.ndarray):
            return super().get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
                                  with_j, with_k, omega, **kwargs)
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        # get dms for each kpt in BZ
        if isinstance(dm_kpts[0], np.ndarray) and dm_kpts[0].ndim == 3:
            ndm = len(dm_kpts[0])
        else:
            ndm = len(dm_kpts)
        if ndm != kpts.nkpts_ibz:
            raise RuntimeError("Number of input density matrices does not \
                               match the number of IBZ kpts: %d vs %d."
                               % (ndm, kpts.nkpts_ibz))
        dm_kpts = kpts.transform_dm(dm_kpts)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            raise NotImplementedError('rsjk with k-points symmetry')
        else:
            vj, vk = self.with_df.get_jk(
                dm_kpts, hermi, kpts.kpts, kpts_band, with_j, with_k, omega,
                exxdiv=self.exxdiv)
        vj = vj[kpts.ibz2bz]
        if vk is not None:
            vk = vk[kpts.ibz2bz]

        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def to_khf(self):
        '''transform to non-symmetry object
        '''
        from pyscf.pbc.scf import kghf_ksymm
        from byteqc.cupbc.scf import khf
        from byteqc.cupbc.dft import krks, krks_ksymm
        from pyscf.scf import addons as mol_addons

        def update_mo_(mf, mf1):
            kpts = mf.kpts
            if mf.mo_energy is not None:
                mo_energy = kpts.transform_mo_energy(mf.mo_energy)
                mo_occ = kpts.transform_mo_occ(mf.mo_occ)

                if isinstance(mf, kghf_ksymm.KGHF):
                    raise NotImplementedError
                else:
                    mo_coeff = kpts.transform_mo_coeff(mf.mo_coeff)

                mf1.mo_coeff = mo_coeff
                mf1.mo_occ = mo_occ
                mf1.mo_energy = mo_energy
            return mf1

        known_cls = {KsymAdaptedKRHF: khf.KRHF,
                     krks_ksymm.KRKS: krks.KRKS}

        out = mol_addons._object_without_soscf(self, known_cls, False)
        out.__dict__.pop('kpts', None)
        return update_mo_(self, out)


KRHF = KsymAdaptedKRHF
