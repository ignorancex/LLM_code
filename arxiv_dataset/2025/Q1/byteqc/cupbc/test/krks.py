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
import time


class DoubleOutput:
    def __init__(self, name="tmp.txt", isinit=True):
        self.name = name
        self.encoding = sys.stdout.encoding
        if isinit:
            self.init()

    def __del__(self):
        self.restore

    def init(self):
        self.file = open(self.name, "w")
        self.stdout_old = sys.stdout
        sys.stdout = self
        self.time = time.time()
        self.write(time.strftime(
            "Start logging at %m%d-%H:%M:%S\n", time.localtime()))

    def restore(self):
        sys.stdout = self.stdout_old
        self.file.close()

    def difftime(self):
        t2 = time.time()
        dt = t2 - self.time
        if dt < 1:
            return "%.3fs" % dt
        if dt < 60:
            return "%.1fs" % dt
        dt = round(dt)
        strtime = '%ds' % (dt % 60)
        dt //= 60
        strtime = '%dm' % (dt % 60) + strtime
        if dt < 60:
            return strtime
        dt //= 60
        strtime = '%dh' % (dt % 24) + strtime
        if dt < 24:
            return strtime
        dt //= 24
        return '%dd' % dt + strtime

    def write(self, data):
        if data and data.isspace():
            self.file.write(data)
        else:
            str = "[%s]" % self.difftime()
            self.file.write(f"{str}{data}")
        self.stdout_old.write(data)

    def flush(self):
        self.stdout_old.flush()
        self.file.flush()


logout = DoubleOutput(isinit=True)

atom, basis, a, kmesh, omega, pseudo, name, xc = getconfig()
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


def run_rsdf(scf, max_cycle=0, verbose=4, **kargs):
    mf = scf.KRKS(cell, kpts=kpts).rs_density_fit(**kargs)
    mf.xc = xc
    mf.verbose = verbose
    mf.with_df.verbose = verbose
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
    log.info("KS/GPU-RSDF energy   : % .10f", egpu)
if (RSDF_cpu_Enabled):
    log.info("KS/CPU-RSDF energy   : % .10f", ecpu)
