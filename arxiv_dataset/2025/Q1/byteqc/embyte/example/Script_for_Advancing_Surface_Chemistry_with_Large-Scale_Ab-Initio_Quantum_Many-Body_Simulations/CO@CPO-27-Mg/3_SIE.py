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

from pathlib import Path
from pyscf import gto
from byteqc import embyte
from byteqc.embyte.Tools.fragment import Fragment
from byteqc.embyte.Localization import iao
from byteqc.embyte.Solver import GPU_CCSDSolver, GPU_MP2Solver
import time
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


def get_atom_frag_list_eq_list(mol):
    atom_frag_list = [[i] for i in range(mol.natm)]
    eq_list = [i for i in range(len(atom_frag_list))]

    return atom_frag_list, eq_list


def CO_on_CPO_27_Mg(basis_set, mol_type):
    current = Path(__file__).resolve().parent
    mol_path = os.path.join(current, f'CO+6B_cluster.xyz')

    mol_ref = gto.M()
    mol_ref.atom = mol_path

    mol_ref.basis = basis_set
    mol_ref.verbose = 6
    mol_ref.build()

    mol = mol_ref.copy()

    if mol_type == 1:
        mol.atom = [(mol_ref.atom_pure_symbol(atom_ind),
                     mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 1
                    else ('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                          mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                    for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol.atom = [('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                     mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 1
                    else (mol_ref.atom_pure_symbol(atom_ind),
                          mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                    for atom_ind in range(mol_ref.natm)]
    elif mol_type == 2:
        pass
    else:
        assert False

    if 'ccecp' in basis_set:
        mol.ecp = 'ccecp'

    mol.build()

    return mol


if __name__ == '__main__':

    threshold = [8.0]
    threshold = [10 ** -th for th in threshold]

    if_MP2 = False

    basis = 'aug-cc-pVDZ'
    # basis = 'aug-cc-pVTZ'

    mol_type = 2
    # mol_type = 1
    # mol_type = 0

    mol = CO_on_CPO_27_Mg(basis, mol_type)

    current = Path(__file__).resolve().parent
    if mol_type == 2:
        logdir = os.path.join(current, f'{basis}/Full')
    elif mol_type == 1:
        logdir = os.path.join(current, f'{basis}/ghost-CO')
    elif mol_type == 0:
        logdir = os.path.join(current, f'{basis}/ghost-MOF')
    else:
        assert False

    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    jk_file = os.path.join(logdir, 'JK_file.npy')
    oei_file = None
    eri_path = None
    assert os.path.exists(chkfile) and os.path.exists(jk_file)

    if if_MP2:
        logfile = os.path.join(logdir, f'SIE+MP2')
    else:
        logfile = os.path.join(logdir, f'SIE+CCSD(T)')

    tot_t = time.time()

    SIE_class = embyte.Framework.SIE.SIE_kernel(logfile, chkfile)

    if if_MP2:
        # Using MP2 as the high-level solver.
        SIE_class.electronic_structure_solver = GPU_MP2Solver
    else:
        # Using CCSD as the high-level solver.
        SIE_class.electronic_structure_solver = GPU_CCSDSolver
        # Using in-situ perturbative (T) correction for achvieving CCSD(T)
        # accuracy.
        SIE_class.in_situ_T = True

    SIE_class.electron_localization_method = iao

    SIE_class.RDM = True

    SIE_class.aux_basis = 'weigend+etb'

    SIE_class.jk_file = jk_file
    SIE_class.oei_file = oei_file
    SIE_class.eri_file = eri_path

    atom_list_frag, equi_list = get_atom_frag_list_eq_list(mol)
    SIE_class.threshold = threshold

    orb_list = embyte.Tools.fragment.from_atom_to_orb_iao(mol, atom_list_frag)

    fragments = get_fragments(orb_list, equi_list)

    natm_ad = 2
    if mol_type == 1:
        fragments_tmp = []
        for frag in fragments[natm_ad:]:
            frag['equivalent_level'] -= natm_ad
            fragments_tmp.append(frag)
        fragments = fragments_tmp

    elif mol_type == 0:
        fragments = fragments[:natm_ad]

    SIE_class.simulate(mol, chkfile, fragments)
