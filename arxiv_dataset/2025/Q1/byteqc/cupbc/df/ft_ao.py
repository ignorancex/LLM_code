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
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf,
# https://github.com/hongzhouye/pyscf/tree/rsdf_direct),
# which is licensed under the Apache License 2.0. The original copyright:
#     Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
#     Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import cupy

from pyscf import lib
from pyscf import gto
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from byteqc.cupbc import lib as cupbclib

libgaft = cupbclib.load_library('libgaft')


def ft_ao_trans(auxvhfopt, Gv, b=None,
                gxyz=None, Gvbase=None, kpt=numpy.zeros(3), verbose=None):
    if gamma_point(kpt):
        return mol_ft_ao_trans(auxvhfopt, Gv, b, gxyz, Gvbase, verbose)
    else:
        kG = Gv + kpt
        return mol_ft_ao_trans(auxvhfopt, kG, None, None, None, verbose)


def mol_ft_ao_trans(auxvhfopt, Gv, b=numpy.eye(3),
                    gxyz=None, Gvbase=None, verbose=None):
    r'''Analytical FT transform AO
    \int mu(r) exp(-ikr) dr^3

    The output tensor has the shape [nGv, nao]
    '''
    mol = auxvhfopt.mol
    nGv = Gv.shape[0]
    GvT = cupy.asarray(Gv.T, order='C')

    if (gxyz is None or b is None or Gvbase is None
        # backward compatibility for pyscf-1.2, in which the argument Gvbase is
        # gs
            or (Gvbase is not None and isinstance(Gvbase[0], (
                int, numpy.integer)))):
        p_gxyzT = lib.c_null_ptr()
        p_gs = (ctypes.c_int * 3)(0, 0, 0)
        b = cupy.asarray([0], dtype='f8')
        p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
        eval_gz = 0  # 'GTO_Gv_general'
    else:
        if abs(b - numpy.diag(b.diagonal())).sum() < 1e-8:
            eval_gz = 1  # 'GTO_Gv_orth'
        else:
            eval_gz = 2  # 'GTO_Gv_nonorth'
        gxyzT = cupy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = ctypes.cast(gxyzT.data.ptr, ctypes.c_void_p)
        b = cupy.asarray(numpy.hstack((b.ravel(), numpy.zeros(3)) + Gvbase))
        p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
        p_gs = (ctypes.c_int * 3)(*[len(x) for x in Gvbase])

    fn = libgaft.GTO_ft_fill_drv
    assert mol.cart  # GTO_ft_ovlp_cart

    ghost_atm = numpy.array([[0, 0, 0, 0, 0, 0]], dtype=numpy.int32)
    ghost_bas = numpy.array([[0, 0, 1, 1, 0, 0, 3, 0]], dtype=numpy.int32)
    ghost_env = numpy.zeros(4)
    ghost_env[3] = numpy.sqrt(4 * numpy.pi)  # s function spherical norm
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,
                                 ghost_atm, ghost_bas, ghost_env)
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[mol.nbas]
    ao_loc = numpy.asarray(numpy.hstack(
        (ao_loc, [nao + 1])), dtype=numpy.int32)
    shape = (nao, nGv)
    mat = cupy.zeros(shape, order='C', dtype=numpy.complex128)
    phase = 0
    l_ctr_offsets = auxvhfopt.l_ctr_offsets

    bufsize = env.nbytes + bas.nbytes + atm.nbytes + 1600
    gpu_buf = cupy.empty(bufsize, dtype=numpy.int8)
    ao_loc_cupy = cupy.asarray(ao_loc)

    for i in range(auxvhfopt.ncptype):
        ish0 = l_ctr_offsets[i]
        ish1 = l_ctr_offsets[i + 1]
        shls_slice = (ish0, ish1, mol.nbas, mol.nbas + 1)
        fn(eval_gz, ctypes.cast(mat[ao_loc[ish0]:].data.ptr, ctypes.c_void_p),
           ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
           ctypes.c_ulong(bufsize), ctypes.c_int(1),
           (ctypes.c_int * 4)(*shls_slice),
           ctypes.cast(ao_loc_cupy.data.ptr, ctypes.c_void_p),
           ctypes.c_double(phase), ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
           p_b, p_gxyzT, p_gs, ctypes.c_int(nGv),
           atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
           bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
           env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
    return mat


def ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                   b=None, gxyz=None, Gvbase=None, q=numpy.zeros(3),
                   kptjs=numpy.zeros((1, 3)), intor='GTO_ft_ovlp', comp=1,
                   bvk_kmesh=None, out=None, ovlp_mask=None, kq=0, Ls=None):
    r'''
    Fourier transform AO pair for a group of k-points
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The return array holds the AO pair
    corresponding to the kpoints given by kptjs
    '''
    intor = cell._add_suffix(intor)

    q = numpy.reshape(q, 3)
    kptjs = numpy.asarray(kptjs, order='C').reshape(-1, 3)
    Gv = numpy.asarray(Gv, order='C').reshape(-1, 3)
    nGv = Gv.shape[0]
    GvT = cupy.asarray(numpy.asarray(Gv.T, order='C') + q.reshape(-1, 1))

    if (gxyz is None or b is None or Gvbase is None or (abs(q).sum() > 1e-9)
        # backward compatibility for pyscf-1.2, in which the argument Gvbase is
        # gs
            or (Gvbase is not None and isinstance(Gvbase[0], (
                int, numpy.integer)))):

        gxyzT = cupy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = ctypes.cast(gxyzT.data.ptr, ctypes.c_void_p)
        p_mesh = (ctypes.c_int * 3)(0, 0, 0)
        b = cupy.asarray([0], dtype='f8')
        p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
        eval_gz_fg = 0  # GTO_Gv_general
    else:
        if abs(b - numpy.diag(b.diagonal())).sum() < 1e-8:
            eval_gz_fg = 1  # GTO_Gv_orth
        else:
            eval_gz_fg = 2  # GTO_Gv_nonorth
        gxyzT = cupy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = ctypes.cast(gxyzT.data.ptr, ctypes.c_void_p)
        b = cupy.asarray(numpy.hstack((b.ravel(), q) + Gvbase))
        p_b = ctypes.cast(b.data.ptr, ctypes.c_void_p)
        p_mesh = (ctypes.c_int * 3)(*[len(x) for x in Gvbase])

    Ls = cell.get_lattice_Ls()
    Ls = Ls[numpy.linalg.norm(Ls, axis=1).argsort()]
    nkpts = len(kptjs)
    nimgs = len(Ls)
    nbas = cell.nbas

    if bvk_kmesh is None:
        expkL = cupy.exp(cupy.asarray(1j * numpy.dot(kptjs, Ls.T)))
    else:
        translations = numpy.linalg.solve(cell.lattice_vectors().T, Ls.T)
        if ovlp_mask is None:
            ovlp_mask = _estimate_overlap_cupy(
                cell, cupy.asarray(Ls)) > cell.precision
        ovlp_mask = cupy.asarray(ovlp_mask, dtype=numpy.int8)
        # t_mod is the translations inside the BvK cell
        t_mod = translations.round(3).astype(
            int) % numpy.asarray(bvk_kmesh)[:, None]
        cell_loc_bvk = cupy.asarray(numpy.ravel_multi_index(
            t_mod, bvk_kmesh).astype(numpy.int32))
        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh)
        expkL = cupy.asarray(numpy.exp(1j * numpy.dot(kptjs, bvkmesh_Ls.T)))

    Ls = cupy.asarray(Ls)
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    if shls_slice is None:
        shls_slice = (0, nbas, nbas, nbas * 2)
    else:
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas + shls_slice[2], nbas + shls_slice[3])
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    shape = (nkpts, comp, ni, nj, nGv)

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# hermi operation needs reordering the axis-0.  It is inefficient.
    if aosym == 's1hermi':  # Symmetry for Gamma point
        assert (is_zero(q) and is_zero(kptjs) and ni == nj)
    elif aosym == 's2':
        i0 = ao_loc[shls_slice[0]]
        i1 = ao_loc[shls_slice[1]]
        nij = i1 * (i1 + 1) // 2 - i0 * (i0 + 1) // 2
        shape = (nkpts, comp, nij, nGv)

    if out is None:
        out = cupy.ndarray(shape, dtype=numpy.complex128)
    else:
        out = cupy.ndarray(shape, dtype=numpy.complex128, memptr=out.data)
    bufsize = env.nbytes + bas.nbytes + atm.nbytes + 1600
    gpu_buf = cupy.empty(bufsize, dtype=numpy.int8)
    ao_loc = cupy.asarray(ao_loc)

    if bvk_kmesh is None:
        drv = libgaft.PBC_ft_latsum_drv
        drv(ctypes.c_int(eval_gz_fg),
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
            ctypes.c_ulong(bufsize), ctypes.c_int(nkpts), ctypes.c_int(comp),
            ctypes.c_int(nimgs), ctypes.cast(Ls.data.ptr, ctypes.c_void_p),
            ctypes.cast(expkL.data.ptr, ctypes.c_void_p),
            (ctypes.c_int * 4)(*shls_slice),
            ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p), p_b, p_gxyzT, p_mesh,
            ctypes.c_int(nGv), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.natm), bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.nbas), env.ctypes.data_as(ctypes.c_void_p))
    else:
        drv = libgaft.PBC_ft_bvk_drv
        drv(ctypes.c_int(eval_gz_fg),
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
            ctypes.c_ulong(bufsize), ctypes.c_int(nkpts), ctypes.c_int(comp),
            ctypes.c_int(nimgs), ctypes.c_int(expkL.shape[1]),
            ctypes.cast(Ls.data.ptr, ctypes.c_void_p),
            ctypes.cast(expkL.data.ptr, ctypes.c_void_p),
            (ctypes.c_int * 4)(*shls_slice),
            ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(cell_loc_bvk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p), p_b, p_gxyzT, p_mesh,
            ctypes.c_int(nGv), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.natm), bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.nbas), env.ctypes.data_as(ctypes.c_void_p))

    gpu_buf = None
    if aosym == 's1hermi':
        for i in range(1, ni):
            out[:, :, :i, i] = out[:, :, i, :i]
    if comp == 1:
        out = out[:, 0]
    return out


def _estimate_overlap_module(module, cell, Ls):
    '''Consider the lattice sum in overlap when estimating the ss-type overlap
    integrals for each traslation vector
    '''
    exps = module.array([cell.bas_exp(ib).min() for ib in range(cell.nbas)])

    atom_coords = cell.atom_coords()
    bas_coords = atom_coords[cell._bas[:, gto.ATOM_OF]]
    rij = module.asarray(bas_coords[:, None, :] - bas_coords)

    aij = exps[:, None] * exps / (exps[:, None] + exps)

    dijL = module.linalg.norm(rij[:, :, None, :] - Ls, axis=-1)
    rij = None

    vol = cell.vol
    vol_rad = vol**(1. / 3)

    fac = (4 * aij / (exps[:, None] + exps))**.75
    s = fac[:, :, None] * module.exp(-aij[:, :, None] * dijL**2)
    s_cum = fac[:, :, None] * \
        module.exp(-aij[:, :, None] * (dijL - vol_rad / 2)**2)
    fac = 2 * module.pi / vol / aij[:, :, None] * abs(dijL - vol_rad / 2)

    return module.maximum(fac * s_cum, s)


def _estimate_overlap(cell, Ls):
    return _estimate_overlap_module(numpy, cell, Ls)


def _estimate_overlap_cupy(cell, Ls):
    return _estimate_overlap_module(cupy, cell, Ls)
