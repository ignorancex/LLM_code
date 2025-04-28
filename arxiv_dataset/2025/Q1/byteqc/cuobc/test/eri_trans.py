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

import numpy
import cupy
from pyscf import gto
from pyscf.df import addons
from byteqc.cuobc.lib import int3c
from byteqc.lib import Mg, gemm


def eri_trans(mol, mo_coeff, blksize=None, auxbasis='weigend+etb', auxmol=None,
              gpus=None):
    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    vhfopt = int3c.VHFOpt3c(mol, auxmol, 'int2e')

    vhfopt.build(group_size=blksize, gpus=gpus)
    j2c = int3c.get_int2c(vhfopt)
    naux = j2c.shape[1]
    mo = gemm(vhfopt.coeff, mo_coeff)
    m = mo.shape[1]

    mos, j2cs = Mg.broadcast(mo, j2c)

    def task(result, cp_ij_id):
        igpu = cupy.cuda.runtime.getDevice()
        mo = mos[igpu]
        print("\rGPU%d-%d/%d" % (igpu, cp_ij_id, len(vhfopt.log_qs)), end="")
        cpi, cpj = int3c.ind2pair(cp_ij_id)
        i0, i1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpi:cpi + 2]]
        j0, j1 = vhfopt.mol.ao_loc[vhfopt.l_ctr_offsets[cpj:cpj + 2]]
        ni = i1 - i0
        nj = j1 - j0
        if result is None:
            result = cupy.zeros((m, m, naux))
        for cp_aux_id, _ in enumerate(vhfopt.aux_log_qs):
            _, _, sk, int3c_blk = int3c.get_int3c(
                cp_ij_id, cp_aux_id, vhfopt, bpcache=vhfopt.bpcaches[igpu])
            nk = sk.stop - sk.start
            int3c_blk = gemm(int3c_blk.reshape(ni, nj * nk),
                             mo[i0: i1], transa='T')
            int3c_blk = gemm(int3c_blk.reshape(nj, m * nk),
                             mo[j0: j1], transa='T')
            int3c_blk = gemm(int3c_blk.reshape(nk, m * m),
                             j2cs[igpu][sk], result, beta=1.0, transa='T')
        return result

    from time import time
    start = time()
    buf = Mg.reduce(task, range(len(vhfopt.log_qs)), batch=1)
    print("")
    print("Trans time", time() - start)
    return buf


numatom = 2  # 88 -> nao=10620
atom = 6
basis = 'ccpvtz'

mol = gto.Mole()
numatom = numatom * 4 + 2
bondlen = 1.2 / numpy.sin(numpy.pi / numatom) / 2
mol.atom = [
    [atom, (bondlen * numpy.cos(theta), bondlen * numpy.sin(theta), 0.)]
    for theta in numpy.arange(numatom) * numpy.pi / numatom * 2]
mol.basis = {
    atom: basis,
}
mol.build(verbose=0)

nao = mol.nao
nmo = 200 if nao > 200 else nao
numpy.random.seed(1231)
mo = numpy.random.rand(nao, nmo)
print(mo.sum())
mo_coeff = cupy.asarray(mo)
eri_trans(mol, mo_coeff, blksize=300, gpus=1)
