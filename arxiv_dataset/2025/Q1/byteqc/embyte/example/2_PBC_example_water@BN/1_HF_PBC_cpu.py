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
import os
import time
from pyscf.pbc.scf import RHF as pbcRHF


def water_on_BN_PBC(k_mesh, basis_set, mol_type):

    water = numpy.array([
        [6.66443, 3.73248, 5.20000],
        [5.88487, 3.46404, 5.71700],
        [6.39397, 3.62736, 4.26700],
    ])

    related_N = numpy.asarray([6.31815, 3.59667, 2.00438])
    water -= related_N

    from pyscf.pbc import gto as pbcgto
    Lz = 16

    cell = pbcgto.Cell()
    bond = 1.4502
    length_cell = bond / numpy.tan(numpy.pi / 6)
    cell.unit = 'A'
    a = numpy.asarray([length_cell, 0, 0])
    b = numpy.asarray([length_cell * numpy.cos(numpy.pi / 3),
                      length_cell * numpy.sin(numpy.pi / 3), 0])
    c = numpy.asarray([0, 0, Lz])
    C1 = a * 1 / 3 + b * 1 / 3
    C2 = a * 2 / 3 + b * 2 / 3
    related_N = C2 + 1 * a + 1 * b + c / 8
    water += related_N
    atom_list = water.tolist()
    for a_ind in range(k_mesh[0]):
        for b_ind in range(k_mesh[1]):
            atom_list.append((C1 + a_ind * a + b_ind * b + c / 8).tolist())

    for a_ind in range(k_mesh[0]):
        for b_ind in range(k_mesh[1]):
            atom_list.append((C2 + a_ind * a + b_ind * b + c / 8).tolist())

    a *= k_mesh[0]
    b *= k_mesh[1]
    atom_list = numpy.asarray(atom_list)

    symbols = {
        0: {'O': 'O', 'H': 'H', 'B': 'ghost-B', 'N': 'ghost-N'},
        1: {'O': 'ghost-O', 'H': 'ghost-H', 'B': 'B', 'N': 'N'},
        2: {'O': 'O', 'H': 'H', 'B': 'B', 'N': 'N'},
    }

    k_total = k_mesh[0] * k_mesh[1] * k_mesh[2]

    atoms = []

    for atom_ind, axis in enumerate(atom_list):
        if atom_ind == 0:
            atoms.append([symbols[mol_type]['O'], axis])
            print(round(axis[0], 5), round(axis[1], 5), round(axis[2], 5))
        elif atom_ind > 0 and atom_ind < 3:
            atoms.append([symbols[mol_type]['H'], axis])
            print(round(axis[0], 5), round(axis[1], 5), round(axis[2], 5))
        elif atom_ind < 3 + k_total:
            atoms.append([symbols[mol_type]['B'], axis])
            print(round(axis[0], 5), round(axis[1], 5), round(axis[2], 5))
        elif atom_ind < 3 + 2 * k_total:
            atoms.append([symbols[mol_type]['N'], axis])
            print(round(axis[0], 5), round(axis[1], 5), round(axis[2], 5))
        else:
            assert False

    cell.atom = atoms
    cell.unit = 'A'
    cell.basis = basis_set
    cell.a = numpy.asarray([a, b, c])

    cell.exp_to_discard = 0.1
    cell.verbose = 6
    cell.build()

    return cell


if __name__ == '__main__':

    mol_basis = 'cc-pVTZ'

    km = [4, 4, 1]

    init_guess = 'minao'
    mol_type = 2
    cell = water_on_BN_PBC(km, mol_basis, mol_type)

    logdir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        f'result/')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    chkfile = os.path.join(logdir, 'HF_chkfile_cpu.chk')
    JK_file = os.path.join(logdir, 'JK_file_cpu.npy')
    eri_file = os.path.join(logdir, 'cderi_cpu')
    logfile = os.path.join(logdir, 'HF_cpu.log')

    cell.output = logfile
    cell.max_memory = 200e3

    cell.build()
    cell.stdout.writelines('The orbitals number is %s \n' % (cell.nao))
    t = time.time()

    mf = pbcRHF(cell, exxdiv=None).density_fit(auxbasis=f'{mol_basis}-ri')
    mf.with_df._cderi_to_save = eri_file
    mf.with_df.build()
    mf.chkfile = chkfile

    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-6

    ehf = mf.kernel()
    numpy.save(JK_file, mf.get_veff())

    oei_path = os.path.join(logdir, 'oei.npy')
    numpy.save(oei_path, mf.get_hcore())
    cell.stdout.writelines(
        'Total time cost is %s mins \n' %
        ((time.time() - t) / 60))
