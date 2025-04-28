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
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf.dft import rks
from pyscf import gto
import numpy
import os
# Please install gpu4pyscf before running this script.


def rotate_vector(v, k, theta):
    k = k / numpy.linalg.norm(k)
    v_rot = v * numpy.cos(theta) \
        + numpy.cross(k, v) * numpy.sin(theta) \
        + k * numpy.dot(k, v) * (1 - numpy.cos(theta))
    v_rot = numpy.where(numpy.abs(v_rot) < 1e-14, 0.0, v_rot)
    return v_rot


def water_on_PAH_OBC(basis_set, mol_type, n, ad_type='2-leg', Z_t=0):
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
    else:
        assert isinstance(
            ad_type, int) or isinstance(
            ad_type, float), 'The degree of the water rotation angle must be a number or a str 0-leg or 2-leg.'
        angle = ad_type

    water_rotate = rotate_vector(water, V, angle) + Z + mol_shift

    geom_path = Path(__file__).resolve().parent.parent
    mol_ref = gto.M()
    mol_ref.atom = os.path.join(geom_path, f'Geometry/PAH{n}_0-leg.xyz')

    mol_ref.basis = basis_set
    mol_ref.verbose = 4
    mol_ref.build()

    mol_return = gto.M()
    if mol_type == 2:
        mol_return.atom \
            = [(mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_pure_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 1:
        mol_return.atom \
            = [(mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else ('ghost-' + mol_ref.atom_pure_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol_return.atom \
            = [('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_pure_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    mol_return.unit = 'angstrom'
    mol_return.basis = basis_set
    mol_return.build()
    if 'ccecp' in basis_set:
        mol_return.ecp = 'ccecp'
        mol_return.build()
    mol_return.verbose = 4
    return mol_return


if __name__ == '__main__':
    # Set the GTO basis
    basis = 'cc-pVQZ'

    mol_type = 2

    boundary_condition = 'OBC'

    max_scf_cycles = 200
    screen_tol = 1e-14
    scf_tol = 1e-10

    n = 4

    xc_list = [
        'PBE',
        'PBE0',
        'B3LYP',
        'BLYP',
        'REVPBE',
        'REVPBE0',
        'WB97M-V',
        'B97M-V',
    ]

    for ad_type in ['0-leg', '2-leg']:
        for xc in xc_list:

            if xc not in ['WB97M-V', 'B97M-V']:
                D3 = True
                xc_name = f'{xc}-D3'
            else:
                D3 = False
                xc_name = xc

            mol = water_on_PAH_OBC(basis, mol_type, n, ad_type=ad_type)
            current = Path(__file__).resolve().parent
            logdir = os.path.join(current, f'{xc}/{ad_type}')

            if not os.path.exists(logdir):
                os.makedirs(logdir)

            logfile = os.path.join(logdir, 'DFT_opt.log')

            mol.output = logfile
            mol.build()
            mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='weigend+etb')
            mf.conv_tol = scf_tol
            mf.max_cycle = max_scf_cycles
            mf.screen_tol = screen_tol
            mf.diis_space = 12

            if D3:
                mf.disp = 'd3bj'

            gradients = []

            def callback(envs):
                gradients.append(envs['gradients'])

            mol_opt = optimize(
                mf,
                constraints=os.path.join(current, 'constraints'),
                maxsteps=max_scf_cycles * 10,
                callback=callback)

            geom_file = os.path.join(logdir, 'geom_opt.xyz')
            with open(geom_file, 'w') as f:
                f.write(f'{3 + n ** 2 * 6 + n * 6}\n')
                f.write(f'optimized by {xc_name} by using {basis} in PAH{n}\n')
                for atom_ind in range(mol_opt.natm):
                    coord = mol_opt.atom_coord(atom_ind, 'angstrom').tolist()
                    coord = [f'{c:12.8f}' for c in coord]
                    f.write(f'{mol_opt.atom_pure_symbol(atom_ind)}{coord[0]}{coord[1]}{coord[2]}\n')
