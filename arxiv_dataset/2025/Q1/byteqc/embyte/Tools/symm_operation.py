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
#
# ByteQC includes code adapted from Vayesta
# (https://github.com/BoothGroup/Vayesta) which are licensed under the Apache
# License 2.0. The original copyright:
#     Vayesta is a Python package for performing correlated wave function-based
#     quantum embedding in ab initio molecules and solids, as well as lattice
#     models. Copyright 2022 The Vayesta Developers. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import scipy
import numpy
import pyscf


class SymmetryOperation:
    '''
    The class for symmetry operation.
    Only SymmetryReflection operator is supported for now.
    All the symmetry operation will be supported in the future.
    '''

    def __init__(self, mol):
        self.mol = mol

    @property
    def xtol(self):
        return 1e-6

    @property
    def natom(self):
        return self.mol.natm

    @property
    def nao(self):
        return self.mol.nao

    def __call__(self, a, *args, axis=0, **kwargs):
        return self.call_wrapper(a, *args, axis=axis, **kwargs)

    def call_wrapper(self, a, *args, axis=0, **kwargs):
        """Common pre- and post-processing for all symmetries.

        Symmetry specific processing is performed in call_kernel."""
        if hasattr(axis, '__len__'):
            for ax in axis:
                a = self(a, *args, axis=ax, **kwargs)
            return a
        if isinstance(a, (tuple, list)):
            return tuple([self(x, *args, axis=axis, **kwargs) for x in a])
        a = numpy.moveaxis(a, axis, 0)
        # Reorder AOs according to new atomic center
        a = a[self.ao_reorder]
        a = self.call_kernel(a, *args, **kwargs)
        a = numpy.moveaxis(a, 0, axis)
        return a

    def get_closest_atom(self, coords):
        """pos in internal coordinates."""
        dists = numpy.linalg.norm(self.mol.atom_coords() - coords, axis=1)
        idx = numpy.argmin(dists)
        return idx, dists[idx]

    def compare_atoms(self, atom1, atom2, check_basis=None, check_label=None):
        """Compare atom symbol and (optionally) basis between atom1 and atom2."""
        if check_label:
            type1 = self.mol.atom_symbol(atom1)
            type2 = self.mol.atom_symbol(atom2)
        else:
            type1 = self.mol.atom_pure_symbol(atom1)
            type2 = self.mol.atom_pure_symbol(atom2)
        if (type1 != type2):
            return False
        if not check_basis:
            return True
        bas1 = self.mol._basis[self.mol.atom_symbol(atom1)]
        bas2 = self.mol._basis[self.mol.atom_symbol(atom2)]
        return (bas1 == bas2)

    def call_kernel(self, *args, **kwargs):
        raise KeyboardInterrupt

    def apply_to_point(self, r0):
        raise KeyboardInterrupt

    def get_atom_reorder(self):
        """Reordering of atoms for a given rotation.

        Parameters
        ----------

        Returns
        -------
        reorder: list
        inverse: list
        """
        reorder = numpy.full((self.natom,), -1, dtype=int)
        inverse = numpy.full((self.natom,), -1, dtype=int)

        def assign():
            success = True
            for atom0, r0 in enumerate(self.mol.atom_coords()):
                r1 = self.apply_to_point(r0)
                atom1, dist = self.get_closest_atom(r1)
                if dist > self.xtol:
                    success = False
                elif not self.compare_atoms(atom0, atom1):
                    success = False
                else:
                    reorder[atom1] = atom0
                    inverse[atom0] = atom1
            return success

        if not assign():
            return None, None

        assert (not numpy.any(reorder == -1))
        assert (not numpy.any(inverse == -1))
        assert numpy.all(
            numpy.arange(
                self.natom)[reorder][inverse] == numpy.arange(
                self.natom))
        return reorder, inverse

    def get_ao_reorder(self, atom_reorder):
        if atom_reorder is None:
            return None, None
        aoslice = self.mol.aoslice_by_atom()[:, 2:]
        reorder = numpy.full((self.mol.nao,), -1)
        inverse = numpy.full((self.mol.nao,), -1)
        for atom0 in range(self.natom):
            atom1 = atom_reorder[atom0]
            aos0 = list(range(aoslice[atom0, 0], aoslice[atom0, 1]))
            aos1 = list(range(aoslice[atom1, 0], aoslice[atom1, 1]))
            reorder[aos0[0]:aos0[-1] + 1] = aos1
            inverse[aos1[0]:aos1[-1] + 1] = aos0
        assert not numpy.any(reorder == -1)
        assert not numpy.any(inverse == -1)
        assert numpy.all(
            numpy.arange(
                self.nao)[reorder][inverse] == numpy.arange(
                self.nao))
        return reorder, inverse

    def rotate_angular_orbitals(self, a, rotmats):
        """Rotate between orbitals in p,d,f,... shells."""
        ao_loc = self.mol.ao_loc
        ao_start = ao_loc[0]
        b = numpy.asarray(a.copy())
        for bas, ao_end in enumerate(ao_loc[1:]):
            l = self.mol.bas_angular(bas)
            # s orbitals do not require rotation:
            if l == 0:
                ao_start = ao_end
                continue
            rot = rotmats[l]
            size = ao_end - ao_start

            # It is possible that multiple shells are contained in a single
            # 'bas'!
            nl = rot.shape[0]
            assert (size % nl == 0)
            for shell0 in range(0, size, nl):
                shell = numpy.s_[ao_start + shell0:ao_start + shell0 + nl]
                b[shell] = numpy.einsum('x...,xy->y...', a[shell], rot)
            ao_start = ao_end
        return b


class SymmetryIdentity(SymmetryOperation):

    def call_kernel(self, a):
        return a

    def apply_to_point(self, r0):
        return r0

    def get_atom_reorder(self):
        reorder = list(range(self.mol.natm))
        return reorder, reorder


class SymmetryRotation(SymmetryOperation):

    def __init__(self, mol, rotvec, center=(0, 0, 0)):
        self.rotvec = numpy.asarray(rotvec, dtype=float)
        self.center = numpy.asarray(center, dtype=float)
        super().__init__(mol)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]
        try:
            self.angular_rotmats = pyscf.symm.basis._momentum_rotation_matrices(
                self.mol, self.as_matrix())
        except AttributeError:
            self.angular_rotmats = pyscf.symm.basis._ao_rotation_matrices(
                self.mol, self.as_matrix())

    def as_matrix(self):
        return scipy.spatial.transform.Rotation.from_rotvec(
            self.rotvec).as_matrix()

    def apply_to_point(self, r0):
        rot = self.as_matrix()
        return numpy.dot(rot, (r0 - self.center)) + self.center

    def call_kernel(self, a):
        a = self.rotate_angular_orbitals(a, self.angular_rotmats)
        return a


class SymmetryReflection(SymmetryOperation):

    def __init__(self, mol, axis, center=(0, 0, 0)):
        center = numpy.asarray(center, dtype=float)
        self.center = center
        self.axis = numpy.asarray(axis) / numpy.linalg.norm(axis)
        super().__init__(mol)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]

        # A reflection can be decomposed into a C2-rotation + inversion
        # We use this to derive the angular transformation matrix:
        rot = scipy.spatial.transform.Rotation.from_rotvec(
            self.axis * numpy.pi).as_matrix()
        try:
            angular_rotmats = pyscf.symm.basis._momentum_rotation_matrices(
                self.mol, rot)
        except AttributeError:
            angular_rotmats = pyscf.symm.basis._ao_rotation_matrices(
                self.mol, rot)
        # Inversion of p,f,h,... shells:
        self.angular_rotmats = [(-1)**i * x for (i, x)
                                in enumerate(angular_rotmats)]

    def as_matrix(self):
        """Householder matrix. Does not account for shifted origin!"""
        return numpy.eye(3) - 2 * numpy.outer(self.axis, self.axis)

    def apply_to_point(self, r0):
        """Householder transformation."""
        r1 = r0 - 2 * numpy.dot(numpy.outer(self.axis,
                                self.axis), r0 - self.center)
        return r1

    def call_kernel(self, a):
        a = self.rotate_angular_orbitals(a, self.angular_rotmats)
        return a


class SymmetryInversion(SymmetryOperation):

    def __init__(self, mol, center=(0, 0, 0)):
        center = numpy.asarray(center, dtype=float)
        self.center = center

        super().__init__(mol)

        self.atom_reorder = self.get_atom_reorder()[0]
        if self.atom_reorder is None:
            raise RuntimeError("Symmetry %s not found" % self)
        self.ao_reorder = self.get_ao_reorder(self.atom_reorder)[0]

    def apply_to_point(self, r0):
        return 2 * self.center - r0

    def call_kernel(self, a):
        rotmats = [(-1)**i * numpy.eye(n)
                   for (i, n) in enumerate(range(1, 19, 2))]
        a = self.rotate_angular_orbitals(a, rotmats)
        return a
