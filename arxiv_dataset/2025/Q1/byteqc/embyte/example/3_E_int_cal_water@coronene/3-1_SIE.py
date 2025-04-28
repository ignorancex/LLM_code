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

from pyscf import gto
from byteqc.embyte.Solver import GPU_CCSDSolver, GPU_MP2Solver
from byteqc.embyte.Localization import iao
from byteqc.embyte.Tools.fragment import Fragment
from byteqc import embyte
import pyscf
import time
import numpy
import os
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def get_fragments(orb_list, equi_list):
    fragments = [
        Fragment(
            orb_list[i],
            equi_list[i],
            ['main']).to_dict() for i in range(
            len(equi_list))]
    return fragments


def get_atom_frag_list_eq_list(mol, n=2):
    atom_frag_list = [[0, 1, 2]]
    atom_frag_list += [[i + 3] for i in range(6 * n ** 2)]
    atom_coords = mol.atom_coords(unit='A')
    eq_list = numpy.asarray([i for i in range(len(atom_frag_list))])
    max_y = atom_coords[:, 1].max()

    H_atom_coords = atom_coords[numpy.arange(6 * n) + 3 + 6 * n ** 2]
    H_list = [[] for _ in range(6)]
    for H_i, tmp_H_coord in enumerate(H_atom_coords):
        if tmp_H_coord[0] > 0 and tmp_H_coord[1] > 0 and tmp_H_coord[1] < max_y:
            H_list[0].append(H_i)
        elif numpy.isclose(tmp_H_coord[1], max_y):
            H_list[1].append(H_i)
        elif tmp_H_coord[0] < 0 and tmp_H_coord[1] > 0 and tmp_H_coord[1] < max_y:
            H_list[2].append(H_i)
        elif tmp_H_coord[0] < 0 and tmp_H_coord[1] < 0 and tmp_H_coord[1] > -max_y:
            H_list[3].append(H_i)
        elif numpy.isclose(tmp_H_coord[1], -max_y):
            H_list[4].append(H_i)
        elif tmp_H_coord[0] > 0 and tmp_H_coord[1] < 0 and tmp_H_coord[1] > -max_y:
            H_list[5].append(H_i)

    H_list = numpy.asarray(H_list)
    tmp_H_coord = H_atom_coords[:, 1][H_list[0]]
    new_ind = tmp_H_coord.argsort()
    H_list[0] = H_list[0][new_ind]

    tmp_H_coord = H_atom_coords[:, 0][H_list[1]]
    new_ind = tmp_H_coord.argsort()
    H_list[1] = H_list[1][new_ind]

    tmp_H_coord = H_atom_coords[:, 1][H_list[2]]
    new_ind = tmp_H_coord.argsort()
    H_list[2] = H_list[2][new_ind]

    tmp_H_coord = H_atom_coords[:, 1][H_list[3]]
    new_ind = (-1 * tmp_H_coord).argsort()
    H_list[3] = H_list[3][new_ind]

    tmp_H_coord = H_atom_coords[:, 0][H_list[4]]
    new_ind = tmp_H_coord.argsort()
    H_list[4] = H_list[4][new_ind]

    tmp_H_coord = H_atom_coords[:, 1][H_list[5]]
    new_ind = (-1 * tmp_H_coord).argsort()
    H_list[5] = H_list[5][new_ind]

    H_list += 3 + 6 * n ** 2
    H_list = H_list.tolist()
    atom_frag_list += H_list

    eq_list_H = [i + 1 + max(eq_list) for i in range(6)]
    eq_list = numpy.asarray(eq_list.tolist() + eq_list_H)

    return atom_frag_list, eq_list


def rotate_vector(v, k, theta):
    k = k / numpy.linalg.norm(k)
    v_rot = v * numpy.cos(theta) \
        + numpy.cross(k, v) * numpy.sin(theta) \
        + k * numpy.dot(k, v) * (1 - numpy.cos(theta))
    v_rot = numpy.where(numpy.abs(v_rot) < 1e-14, 0.0, v_rot)
    return v_rot


def water_on_coronene_mol(basis_set, mol_type, ad_type='2-leg', Z_t=0):
    # Rotate around x axis
    V = numpy.asarray([1, 0, 0])

    water = numpy.array([
        [0.00000000, 0.00000000, 3.15500000],
        [0.00000000, 0.75668992, 2.56910806],
        [0.00000000, -0.75668992, 2.56910806],
    ])
    Z = numpy.asarray([0., 0., float(Z_t)])
    mol_shift = water[0] + numpy.asarray([0, 0, 0.175])
    water += numpy.asarray([0, 0, 0.35])
    water -= mol_shift

    if ad_type == '2-leg':
        angle = 0
    elif ad_type == '0-leg':
        angle = 180 / 180 * numpy.pi

    water_rotate = rotate_vector(water, V, angle) + Z + mol_shift

    mol_ref = gto.M()
    mol_ref.atom = '''
        O1  0  0  0
        H1  0  0  0
        H1  0  0  0
        C   1.22975607   0.71000000   0.00000000
        C   0.00000000   1.42000000   0.00000000
        C   -1.22975607   0.71000000   0.00000000
        C   -1.22975607   -0.71000000   0.00000000
        C   -0.00000000   -1.42000000   0.00000000
        C   1.22975607   -0.71000000   0.00000000
        C   3.68926822   0.71000000   0.00000000
        C   2.45951215   1.42000000   0.00000000
        C   2.45951215   -1.42000000   0.00000000
        C   3.68926822   -0.71000000   0.00000000
        C   2.45951215   2.84000000   0.00000000
        C   1.22975607   3.55000000   0.00000000
        C   0.00000000   2.84000000   0.00000000
        C   -1.22975607   3.55000000   0.00000000
        C   -2.45951215   2.84000000   0.00000000
        C   -2.45951215   1.42000000   0.00000000
        C   -3.68926822   0.71000000   0.00000000
        C   -3.68926822   -0.71000000   0.00000000
        C   -2.45951215   -1.42000000   0.00000000
        C   -2.45951215   -2.84000000   0.00000000
        C   -1.22975607   -3.55000000   0.00000000
        C   -0.00000000   -2.84000000   0.00000000
        C   1.22975607   -3.55000000   0.00000000
        C   2.45951215   -2.84000000   0.00000000
        H   4.63236988   1.25450000   0.00000000
        H   4.63236988   -1.25450000   0.00000000
        H   3.40261381   3.38450000   0.00000000
        H   1.22975607   4.63900000   0.00000000
        H   -1.22975607   4.63900000   0.00000000
        H   -3.40261381   3.38450000   0.00000000
        H   -4.63236988   1.25450000   0.00000000
        H   -4.63236988   -1.25450000   0.00000000
        H   -3.40261381   -3.38450000   0.00000000
        H   -1.22975607   -4.63900000   0.00000000
        H   1.22975607   -4.63900000   0.00000000
        H   3.40261381   -3.38450000   0.00000000
    '''

    if mol_type == 0:
        basis = {'default': basis_set, 'ghost-H': 'sto-3g'}
    else:
        basis = {'default': basis_set, 'H': 'sto-3g'}

    mol_ref.basis = basis
    mol_ref.verbose = 4
    mol_ref.build()

    mol_return = gto.M()
    if mol_type == 2:
        mol_return.atom \
            = [(mol_ref.atom_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 1:
        mol_return.atom \
            = [(mol_ref.atom_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else ('ghost-' + mol_ref.atom_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol_return.atom \
            = [('ghost-' + mol_ref.atom_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    mol_return.unit = 'angstrom'
    mol_return.basis = basis
    mol_return.build()
    if 'ccecp' in basis_set:
        mol_return.ecp = 'ccecp'
        mol_return.build()
    mol_return.verbose = 6
    return mol_return


if __name__ == '__main__':

    print('++++++++++++++++++++++++++++====================++++++++++++++++++++++++++++====================')
    print(pyscf.__version__)
    print(pyscf.__file__)

    threshold = [8.0]
    threshold = [10 ** -th for th in threshold]

    if_MP2 = True

    mol_basis = 'ccecp-cc-pVDZ'
    mol_type = 2
    # mol_type = 2 for water + coronene
    # mol_type = 1 for ghost-water + coronene
    # mol_type = 0 for water + ghost-coronene
    mol = water_on_coronene_mol(mol_basis, mol_type)

    if mol_type == 2:
        logdir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            f'result/Full')
    elif mol_type == 1:
        logdir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            f'result/ghost-water')
    elif mol_type == 0:
        logdir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            f'result/ghost-coronene')
    else:
        assert False

    # HF_chkfile is generated by using pyscf.scf.RHF without density fiting.
    # JK_file is generated by using pyscf.scf.RHF.get_veff without density
    # fiting.

    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    jk_file = os.path.join(logdir, 'JK_file.npy')
    eri_path = None
    assert os.path.exists(chkfile) and os.path.exists(jk_file)

    if if_MP2:
        logfile = os.path.join(logdir, f'SIE+MP2')
    else:
        logfile = os.path.join(logdir, f'SIE+CCSD(T)')

    tot_t = time.time()

    SIE_class = embyte.Framework.SIE.SIE_kernel(logfile, chkfile)

    if if_MP2:
        SIE_class.electronic_structure_solver = GPU_MP2Solver
    else:
        SIE_class.electronic_structure_solver = GPU_CCSDSolver
        SIE_class.in_situ_T = True

    SIE_class.electron_localization_method = iao

    SIE_class.RDM = True

    SIE_class.aux_basis = 'weigend+etb'

    SIE_class.jk_file = jk_file
    SIE_class.eri_file = eri_path

    # The 'get_atom_frag_list_eq_list' function is used to determine which
    # atoms are included in each fragment and the equivalence relationships
    # between fragments. Therefore, this part needs to be customized based
    # on the actual situation.

    # 'atom_list_frag' like [frag_list_0, frag_list_1, ...],
    # 'frag_list_i' includes atoms in fragment i, like [atom_0, atom_1, ...].

    # 'equi_list' is a int list which repersents the equivalence relationships
    # between fragments, like [0, 1, 0, ...] shows that the fragment 0 and
    # fragment 2 at the same equalence level. Notably, equalence level showing up
    # order must from 0 to the largest. A wrong example is like [1, 0, 1, ...].

    atom_list_frag, equi_list = get_atom_frag_list_eq_list(mol)

    SIE_class.threshold = threshold

    # 'SIE_class.cheat_th' is used to set a different threshold for some specific fragments.
    # 'cheat_th' should be a dict like {frag_id : [threshold_0, threshold_1, ...]}.
    # Note that the length of the specific threhold should be the same with the
    # default thershould set 'SIE_class.threshold'.

    if mol_type != 1:
        SIE_class.cheat_th = {
            0: [10 ** -6.5]
        }

    orb_list = embyte.Tools.fragment.from_atom_to_orb_iao(mol, atom_list_frag)

    # The get_fragments function is used to register each fragment, including the indices of
    # localized orbitals within the fragment, the fragment's equivalence level, and
    # the equivalence operator. This registration facilitates grouping fragments with
    # the same equivalence level together and clarifies the equivalence relationships
    # between fragments. For more detailed usage, please see more in
    # 1. byteqc.embyte.Tools.fragment.Fragment
    # 2. byteqc.embyte.Tools.fragment.Fragment_group
    # 3. ./3-2_SIE_with_symm.py

    fragments = get_fragments(orb_list, equi_list)

    if mol_type == 1:
        fragments_tmp = []
        for frag in fragments[1:]:
            frag['equivalent_level'] -= 1
            fragments_tmp.append(frag)
        fragments = fragments_tmp

    elif mol_type == 0:
        fragments = fragments[:1]

    SIE_class.simulate(mol, chkfile, fragments)
