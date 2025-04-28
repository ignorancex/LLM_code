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

from pyscf.pbc import scf
from byteqc.cupbc import scf as pscf
import sys
from pyscf.lib import logger
from byteqc.cupbc.test.getconfig import getconfig
from pyscf.pbc import gto

atom, basis, a, kmesh, omega, pseudo, name, _ = getconfig()
cell = gto.Cell(atom=atom, basis=basis, a=a, pseudo=pseudo)
cell.verbose = 0
cell.build()

print("nbas:%s nao:%d" % (cell.nbas, cell.nao))
scaled_center = None
# scaled_center = np.random.rand(3)
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
log = logger.Logger(cell.stdout, 6)
log.info("kmesh= %s", kmesh)
log.info("kpts = %s", kpts)
max_cycle = 0


def run_rsdf(scf, max_cycle=0, verbose=3, **kargs):
    mf = scf.KRHF(cell, kpts=kpts).rs_density_fit(**kargs)
    mf.verbose = verbose
    mf.max_memory = 700000
    mf.with_df.max_memory = 700000
    mf.max_cycle = max_cycle
    mf.with_df.omega = omega
    mf.with_df.direct = True
    mf.with_df.ksym = 's1'
    mf.with_df.use_bvk = [True, True]
    # mf.with_df.build(with_j3c=True)
    mf.kernel()
    return mf.e_tot


RSDF_gpu_Enabled = True
RSDF_cpu_Enabled = True
if len(sys.argv) >= 2:
    if (sys.argv[1]) == 'cpu':
        RSDF_gpu_Enabled = False
    if (sys.argv[1]) == 'gpu':
        RSDF_cpu_Enabled = False


if (RSDF_gpu_Enabled):
    egpu = run_rsdf(pscf, max_cycle)
if (RSDF_cpu_Enabled):
    ecpu = run_rsdf(scf, max_cycle)

if (RSDF_gpu_Enabled):
    log.info("HF/GPU-RSDF energy   : % .10f", egpu)
if (RSDF_cpu_Enabled):
    log.info("HF/CPU-RSDF energy   : % .10f", ecpu)
