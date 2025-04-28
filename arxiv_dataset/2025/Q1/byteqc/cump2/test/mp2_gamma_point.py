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

import time
import numpy
from byteqc.cump2 import DFKMP2
from pyscf.pbc import mp as pbcmp
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from byteqc.lib import Mg


def get_cell(basis='ccpvdz', k_mesh=None):

    cell = pbcgto.Cell()
    cell.atom = '''
    H        -3.387259         4.313406        -1.424778
    H        -3.181837         5.369208        -0.311470
    H         2.963168         1.468906         1.932833
    H         3.285969         1.385243         3.475835
    H         0.841667         0.204178         5.599864
    H         2.129677         1.043538         5.781331
    H         1.465538         3.210193        -1.490078
    H         2.777609         3.402557        -2.346553
    H         4.997562         0.773964        -2.555873
    H         5.088937         2.299233        -2.804018
    H        -0.780920         3.193296         1.966265
    H        -1.843975         4.318694         1.657300
    H         5.905500         1.121165         3.087808
    H         5.700078         0.065363         1.974500
    H        -0.444928         3.965666        -0.269803
    H        -0.767728         4.049329        -1.812805
    H         1.676574         5.230393        -3.936835
    H         0.388563         4.391033        -4.118300
    H         1.052703         2.224379         3.153108
    H        -0.259368         2.032014         4.009582
    H        -2.479321         4.660607         4.218903
    H        -2.570696         3.135338         4.467048
    H         3.299161         2.241275        -0.303236
    H         4.362216         1.115877         0.005730
    O        -3.805203         4.649237        -0.585459
    O         2.541021         1.354868         2.825656
    O         1.820288         0.102491         5.754385
    O         2.359668         2.814461        -1.669750
    O         4.860804         1.454814        -3.269776
    O        -1.291680         3.619532         1.227485
    O         6.323444         0.785334         2.248489
    O        -0.022780         4.079704        -1.162626
    O         0.697952         5.332080        -4.091355
    O         0.158573         2.620111         3.332779
    O        -2.342563         3.979757         4.932805
    O         3.809921         1.815039         0.435545
    '''
    cell.basis = basis
    cell.unit = 'A'
    cell.a = numpy.asarray([
        [7.12333918, 0.0, -3.04115415],
        [-4.60506582, 5.43464184, -3.04115415],
        [0.0, 0.0, 7.74535894],
    ])

    cell.verbose = 6
    cell.exp_to_discard = 0.1
    cell.build()

    if k_mesh:
        from pyscf.pbc.tools.pbc import super_cell
        cell = super_cell(cell, k_mesh)
        cell.verbose = 6
        cell.exp_to_discard = 0.1
        cell.build()

    return cell


for basis in ['cc-pVDZ', ]:
    auxbasis = f'{basis}-ri'
    for k_mesh in [[1, 1, 1]]:
        print("\n supercell k_mesh:", k_mesh, "basis:", basis)
        cell = get_cell(basis, k_mesh)
        cell.build()
        rhf = pbcscf.RHF(cell, exxdiv=None).density_fit(auxbasis=auxbasis)
        rhf.max_cycle = 50
        rhf.with_df._cderi_to_save = '/mnt/bn/zigeng-big/TBD_test/SIE_test2/rhf_df_cderi_111.h5'
        rhf.kernel()

        for ngpu in range(1, 2):
            Mg.set_gpus(ngpu)
            print(
                "\n supercell k_mesh:",
                k_mesh,
                "basis:",
                basis,
                "ngpu",
                ngpu)
            start = time.time()
            e1, e2, rdm1 = DFKMP2.kernel(cell, rhf, with_rdm1=True)
            tg = time.time() - start
            mp = pbcmp.RMP2(rhf)

            mp.max_memory = 240000
            mp.run()
            rdm1_ref = mp.make_rdm1()

            print('GPU gamma point energy:', e1)
            print('CPU gamma point energy:', mp.e_corr)

            print('GPU/CPU gamma point 1-RDM difference:', abs(rdm1 - rdm1_ref).sum())
