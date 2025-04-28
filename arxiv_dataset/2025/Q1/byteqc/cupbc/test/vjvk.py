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

import sys
import numpy
from pyscf.lib import logger
from byteqc.cupbc.test.getconfig import getconfig
from pyscf.pbc import gto
from byteqc.cupbc.df.rsdf_direct_helper import get_kptij_lst
from byteqc.cupbc import scf as scf_gpu
from pyscf.pbc import scf as scf_cpu
from byteqc.cupbc.df.rsdf_direct_jk import get_j_kpts as get_j_kpts_gpu
from pyscf.pbc.df.df_jk import get_j_kpts as get_j_kpts_cpu
from byteqc.cupbc.df.rsdf_direct_jk import get_k_kpts as get_k_kpts_gpu
from pyscf.pbc.df.df_jk import get_k_kpts as get_k_kpts_cpu
from byteqc.cupbc.df.rsdf_direct_helper import kpts_to_kmesh

atom, basis, a, kmesh, omega, pseudo, name, _ = getconfig()
cell = gto.Cell(atom=atom, basis=basis, a=a, pseudo=pseudo)
cell.verbose = 0
cell.build()

print("nbas:%s nao:%d" % (cell.nbas, cell.nao))
scaled_center = None
# scaled_center = np.random.rand(3)
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
kptij_lst = get_kptij_lst(kpts, ksym='s1')
log = logger.Logger(cell.stdout, 6)
log.info("kmesh= %s", kmesh)
log.info("kpts = %s", kpts)
max_cycle = 0

test_vj = True
test_vk = True


def get_vjk(which, get_j_kpts, get_k_kpts, dm):
    if which == 'gpu':
        mf = scf_gpu.KRHF(cell, kpts=kpts).rs_density_fit()
    else:
        mf = scf_cpu.KRHF(cell, kpts=kpts).rs_density_fit()
    mf.verbose = 5
    mf.max_memory = 700000
    mf.with_df.max_memory = 700000
    mf.max_cycle = max_cycle
    mf.with_df.omega = omega
    mf.with_df.direct = True
    mf.with_df.ksym = 's1'
    use_bvk = [True, True]
    mf.with_df.use_bvk = use_bvk
    mf.with_df.build(with_j3c=False if which == 'gpu' else True)
    if isinstance(use_bvk, bool):
        use_bvk_R = use_bvk_G = use_bvk
    else:
        use_bvk_R, use_bvk_G = use_bvk
    if use_bvk_R or use_bvk_G:
        bvk_kmesh0 = kpts_to_kmesh(mf.with_df.cell, kpts)
        bvk_kmesh = [bvk_kmesh0 if use_bvk_R else None,
                     bvk_kmesh0 if use_bvk_G else None]
    else:
        bvk_kmesh = None
    if which == 'gpu':
        vj = get_j_kpts(mf.with_df, dm, kpts=kpts,
                        bvk_kmesh=bvk_kmesh) if test_vj else None
        vk = get_k_kpts(mf.with_df, dm, hermi=1, kpts=kpts,
                        bvk_kmesh=bvk_kmesh) if test_vk else None
    else:
        vj = get_j_kpts(mf.with_df, dm, kpts=kpts) if test_vj else None
        vk = get_k_kpts(mf.with_df, dm, hermi=1,
                        kpts=kpts) if test_vk else None
    return vj, vk


nkpts = len(kpts)
nao = cell.nao
dm = numpy.random.rand(nkpts, nao, nao)
dm += dm.transpose(0, 2, 1)
for k in range(nkpts):
    idm = dm[k]
    e, u = numpy.linalg.eigh(idm)
    dm[k] = u @ numpy.eye(len(e)) @ u.T

ENABLE_CPU = True
ENABLE_GPU = True
if len(sys.argv) > 1:
    if (sys.argv[1]) == 'cpu':
        ENABLE_GPU = False
    if (sys.argv[1]) == 'gpu':
        ENABLE_CPU = False

if ENABLE_GPU:
    vjg, vkg = get_vjk('gpu', get_j_kpts_gpu, get_k_kpts_gpu, dm)
if ENABLE_CPU:
    vjc, vkc = get_vjk('cpu', get_j_kpts_cpu, get_k_kpts_cpu, dm)
sys.stdout = sys.__stdout__
if ENABLE_GPU and ENABLE_CPU:
    if test_vj:
        print("Is vj match?", numpy.allclose(vjg, vjc),
              "max error:", abs(vjg - vjc).max())
    if test_vk:
        print("Is vk match?", numpy.allclose(vkg, vkc),
              "max error:", abs(vkg - vkc).max())
