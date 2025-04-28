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
from byteqc.cuobc import scf
import numpy
import os


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
    mol_return.verbose = 4
    return mol_return


if __name__ == '__main__':
    basis = 'ccecp-cc-pVDZ'
    mol_type = 2
    # mol_type = 2 for water + coronene
    # mol_type = 1 for ghost-water + coronene
    # mol_type = 0 for water + ghost-coronene
    mol = water_on_coronene_mol(basis, mol_type=mol_type)

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

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    JK_file = os.path.join(logdir, 'JK_file.npy')
    logfile = os.path.join(logdir, 'HF.log')
    mol.output = logfile
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.diis_space = 12

    mf.kernel()

    veff = mf.get_veff()
    numpy.save(JK_file, veff)
