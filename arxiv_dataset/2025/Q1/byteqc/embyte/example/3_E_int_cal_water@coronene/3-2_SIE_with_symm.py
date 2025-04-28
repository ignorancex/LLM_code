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


def get_fragments(mol, orb_list, atom_list, equi_list):
    fragments = [None] * len(orb_list)
    fragments[0] = Fragment(orb_list[0], 0, ['main']).to_dict()
    atom_coords = mol.atom_coords(unit='A')
    for clu_i in range(1, len(orb_list)):
        if fragments[clu_i] is None:
            fragments[clu_i] = Fragment(
                orb_list[clu_i], equi_list[clu_i], ['main']).to_dict()
            for clu_i_tmp in range(1, len(orb_list)):
                if equi_list[clu_i_tmp] == equi_list[clu_i] and clu_i_tmp != clu_i:
                    atom_i = atom_list[clu_i][0]
                    atom_i_tmp = atom_list[clu_i_tmp][0]
                    if numpy.allclose(
                            atom_coords[atom_i] * numpy.asarray([-1, 1, 1]), atom_coords[atom_i_tmp]):
                        fragments[clu_i_tmp] = Fragment(orb_list[clu_i_tmp], equi_list[clu_i_tmp], [
                                                        ['Reflection', [1, 0, 0]]]).to_dict()
                    elif numpy.allclose(atom_coords[atom_i] * numpy.asarray([1, -1, 1]), atom_coords[atom_i_tmp]):
                        fragments[clu_i_tmp] = Fragment(orb_list[clu_i_tmp], equi_list[clu_i_tmp], [
                                                        ['Reflection', [0, 1, 0]]]).to_dict()
                    elif numpy.allclose(atom_coords[atom_i] * numpy.asarray([-1, -1, 1]), atom_coords[atom_i_tmp]):
                        fragments[clu_i_tmp] = Fragment(orb_list[clu_i_tmp], equi_list[clu_i_tmp], [
                                                        ['Reflection', [1, 0, 0]], ['Reflection', [0, 1, 0]]]).to_dict()
                    else:
                        raise KeyboardInterrupt
    if None in fragments:
        raise KeyboardInterrupt

    return fragments


def get_atom_frag_list_eq_list(mol, n=2, ad_type='0-leg'):
    atom_frag_list = [[0, 1, 2]]
    atom_frag_list += [[i + 3] for i in range(6 * n ** 2)]
    atom_coords = mol.atom_coords(unit='A')
    C_atom_coords = atom_coords[numpy.arange(6 * n ** 2) + 3]
    eq_list = numpy.asarray([0] * len(atom_frag_list))
    turn_i = 1
    start_point = 1
    max_y = atom_coords[:, 1].max()

    for C_i, _ in enumerate(C_atom_coords):
        tmp_C_coord = C_atom_coords[C_i].copy()

        if eq_list[C_i + start_point] != 0:
            continue

        eq_list[C_i + start_point] = turn_i

        tmp_C_coord[0] = -tmp_C_coord[0]
        C_inds = numpy.where(
            numpy.all(
                numpy.isclose(
                    tmp_C_coord,
                    C_atom_coords),
                axis=1))[0]
        eq_list[C_inds + start_point] = turn_i
        if ad_type != '1-leg':
            tmp_C_coord[1] = -tmp_C_coord[1]
            C_inds = numpy.where(
                numpy.all(
                    numpy.isclose(
                        tmp_C_coord,
                        C_atom_coords),
                    axis=1))[0]
            eq_list[C_inds + start_point] = turn_i

            tmp_C_coord[0] = -tmp_C_coord[0]
            C_inds = numpy.where(
                numpy.all(
                    numpy.isclose(
                        tmp_C_coord,
                        C_atom_coords),
                    axis=1))[0]
            eq_list[C_inds + start_point] = turn_i

        turn_i += 1

    H_atom_coords = atom_coords[numpy.arange(6 * n) + 3 + 6 * n ** 2]
    H_list = [[] for _ in range(6)]
    start_point += 6 * n ** 2
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

    if ad_type != '1-leg':
        eq_list_H = numpy.asarray([0, 1, 0, 0, 1, 0]) + turn_i
    else:
        eq_list_H = numpy.asarray([0, 1, 0, 2, 3, 2]) + turn_i
    eq_list = numpy.asarray(eq_list.tolist() + eq_list_H.tolist())

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

    if_MP2 = False

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
        logfile = os.path.join(logdir, f'SIE+MP2_with_symm')
    else:
        logfile = os.path.join(logdir, f'SIE+CCSD(T)_with_symm')

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

    atom_list_frag, equi_list = get_atom_frag_list_eq_list(
        mol, n=2, ad_type='2-leg')
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

    fragments = get_fragments(mol, orb_list, atom_list_frag, equi_list)

    if mol_type == 1:
        fragments_tmp = []
        for frag in fragments[1:]:
            frag['equivalent_level'] -= 1
            fragments_tmp.append(frag)
        fragments = fragments_tmp

    elif mol_type == 0:
        fragments = fragments[:1]

    SIE_class.simulate(mol, chkfile, fragments)
