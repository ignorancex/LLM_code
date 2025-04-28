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

from byteqc.embyte.Solver import GPU_CCSDSolver, GPU_MP2Solver
from byteqc.embyte.Localization import iao
from byteqc.embyte.Tools.fragment import Fragment
from byteqc import embyte
import pyscf
import numpy
import os
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def get_atom_frag_list_eq_list_water_on_BN(atom_Num):
    atom_frag_list = [[0, 1, 2]] + [[i + 3] for i in range(atom_Num - 3)]
    eq_list = list(range(atom_Num - 2))
    return atom_frag_list, eq_list


def get_fragments(orb_list, equi_list):
    fragments = [
        Fragment(
            orb_list[i],
            equi_list[i],
            ['main']).to_dict() for i in range(
            len(equi_list))]
    return fragments


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
    cell.build()

    return cell


if __name__ == '__main__':

    print('++++++++++++++++++++++++++++====================++++++++++++++++++++++++++++====================')
    print(pyscf.__version__)
    print(pyscf.__file__)

    mol_type = 2
    mol_basis = 'cc-pVTZ'
    threshold = [8.0]
    threshold = [10 ** -th for th in threshold]
    if_MP2 = True

    logdir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        f'result/')
    chkfile = os.path.join(logdir, 'HF_chkfile_cpu.chk')
    jk_file = os.path.join(logdir, 'JK_file_cpu.npy')
    eri_path = os.path.join(logdir, 'cderi_cpu')
    oei_path = os.path.join(logdir, 'oei.npy')
    logfile = os.path.join(logdir, 'SIE_result')

    assert os.path.exists(eri_path) and os.path.exists(oei_path) \
        and os.path.exists(chkfile) and os.path.exists(jk_file)

    km = [4, 4, 1]
    init_guess = 'minao'
    cell = water_on_BN_PBC(km, mol_basis, mol_type)
    cell.build()

    SIE_class = embyte.Framework.SIE.SIE_kernel(logfile, chkfile)

    if if_MP2:
        SIE_class.electronic_structure_solver = GPU_MP2Solver
    else:
        SIE_class.electronic_structure_solver = GPU_CCSDSolver

    SIE_class.electron_localization_method = iao

    SIE_class.RDM = True
    SIE_class.in_situ_T = False

    SIE_class.aux_basis = f'{mol_basis}-ri'

    SIE_class.jk_file = jk_file
    SIE_class.eri_file = eri_path
    SIE_class.oei_file = oei_path

    atom_list_frag, equi_list = get_atom_frag_list_eq_list_water_on_BN(
        cell.natm)
    SIE_class.threshold = threshold

    orb_list = embyte.Tools.fragment.from_atom_to_orb_iao(cell, atom_list_frag)
    fragments = get_fragments(orb_list, equi_list)

    SIE_class.simulate(cell, chkfile, fragments)
