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
from byteqc.cuobc.lib import int4c
from byteqc.cuobc.scf.hf import _VHFOpt

numatom = 1  # 88 -> nao=10620
atom = 6
basis = 'ccpvdz'

mol = gto.Mole()
numatom = numatom * 4 + 2
bondlen = 1.2 / numpy.sin(numpy.pi / numatom) / 2
mol.atom = [[atom, (bondlen * numpy.cos(theta), bondlen * numpy.sin(theta),
                    0.)]
            for theta in numpy.arange(numatom) * numpy.pi / numatom * 2]
mol.basis = {
    atom: basis,
}
mol.build(verbose=0)

vhfopt = _VHFOpt(mol, 'int2e')
# This test only supports single GPU
vhfopt.build(diag_block_with_triu=True)

gpu = int4c.get_int4c(mol, vhfopt=vhfopt, batch=1)
cpu = gto.moleintor.getints('int2e_cart', vhfopt.mol._atm, vhfopt.mol._bas,
                            vhfopt.mol._env)
a = cupy.where(True ^ cupy.isclose(cpu, gpu))
a = zip(*a)
for ind, i in enumerate(a):
    print(i)
    if ind > 5:
        break
print(cupy.allclose(cpu, gpu))
print(cupy.abs(cupy.asarray(cpu) - gpu).max())
