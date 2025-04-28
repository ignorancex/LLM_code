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
import pickle
import os
from time import time
from byteqc import cucc
from pyscf import gto
from pyscf import scf
import sys


basis = '6-31g'
numatom = 3
isdiis = False
lambda_damp = 0.6
verbo = 5
cpulim = 400e9
gpulim = 70e9
device = eval(os.getenv('DEFAULT_DEVICE', '0'))
type_df = 2
maxcycle = 10
i = 1
atom = 6
while i < len(sys.argv):
    if sys.argv[i] in ('-b', '-basis'):
        basis = sys.argv[i + 1]
    elif sys.argv[i] in ('-n', '-numatom'):
        numatom = int(sys.argv[i + 1])
    elif sys.argv[i] in ('-i', '-isdiis'):
        isdiis = bool(sys.argv[i + 1])
    elif sys.argv[i] in ('-l', '-lambda_damp'):
        lambda_damp = float(sys.argv[i + 1])
    elif sys.argv[i] in ('-v', '-verbo'):
        verbo = int(sys.argv[i + 1])
    elif sys.argv[i] in ('-c', '-cpulim'):
        cpulim = float(sys.argv[i + 1])
    elif sys.argv[i] in ('-g', '-gpulim'):
        gpulim = float(sys.argv[i + 1])
    elif sys.argv[i] in ('-d', '-device'):
        device = int(sys.argv[i + 1])
    elif sys.argv[i] in ('-t', '-type_df'):
        type_df = int(sys.argv[i + 1])
    elif sys.argv[i] in ('-m', '-maxcycle'):
        maxcycle = int(sys.argv[i + 1])
    elif sys.argv[i] in ('-a', '-atom'):
        atom = int(sys.argv[i + 1])
    i += 2
    pass
print('\n\nBasis:%s Atom:%d Natom:%d DIIS:%s' %
      (basis, atom, numatom * 4 + 2, isdiis))
print('Device:%s TypeDF:%d Maxcycle:%d verbo:%d' %
      ('GPU' if device == 0 else 'CPU', type_df, maxcycle, verbo))
print('CPUlim:%.2fG GPUlim:%.2fG\n' % (cpulim / 1e9, gpulim / 1e9))

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
rhf = scf.RHF(mol)  # TODO 尝试UHF， GHF
rhf.verbose = 3
path = "data%d%s.dat" % (
    numatom, basis) if atom == 6 else "data%d%s%d.dat" % (
    numatom, basis, atom)
if os.path.isfile(path):
    print("Reading data from file")
    with open(path, 'rb') as f:
        rhf.mo_energy, rhf.mo_coeff, rhf.mo_occ, rhf.e_tot, \
            rhf.scf_summary, rhf._eri, rhf._t0, rhf._w0 = pickle.load(f)
else:
    print("Calculating scf")
    rhf.scf()
    with open(path, 'wb') as f:
        pickle.dump((rhf.mo_energy, rhf.mo_coeff, rhf.mo_occ, rhf.e_tot,
                     rhf.scf_summary, rhf._eri, rhf._t0, rhf._w0), f, 1)

if type_df == 1:
    mf = rhf.density_fit(auxbasis='weigend')
else:
    mf = rhf
mcc = cucc.CCSD(mf, cpulim=cpulim, gpulim=gpulim)
if type_df == 2:
    mcc = mcc.density_fit(auxbasis='weigend')
mcc.verbose = verbo
if getattr(mcc, 'pool', None) is not None:
    mcc.pool.setverbose(verbo)
mcc.diis = isdiis
if not mcc.diis:
    mcc.iterative_damping = lambda_damp
mcc.max_memory = int(400000)
mcc.max_cycle = maxcycle
start = time()
mcc.ccsd()
if device == 0:
    dm = mcc.make_rdm2()
else:
    dm = mcc.tocpu().make_rdm2()
# print(dm)
print("\ntime = ", time() - start)
