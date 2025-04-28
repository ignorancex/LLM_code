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
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf),
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

import numpy
import multiprocessing
from pyscf import lib
from byteqc.cupbc.df import rsdf_jk
from pyscf.pbc.dft import gen_grid
from pyscf import __config__
from pyscf.pbc import dft as pdft
from pyscf.pbc.dft import rks, multigrid
from pyscf.pbc.dft.gen_grid import get_becke_grids
from pyscf.lib import logger

# from pyscf.dft import libxc
# libxc._itrf = lib.load_library('libgxc_itrf')
# libxc._itrf.LIBXC_is_lda.restype = ctypes.c_int
# libxc._itrf.LIBXC_is_gga.restype = ctypes.c_int
# libxc._itrf.LIBXC_is_meta_gga.restype = ctypes.c_int
# libxc._itrf.LIBXC_needs_laplacian.restype = ctypes.c_int
# libxc._itrf.LIBXC_needs_laplacian.argtypes = [ctypes.c_int]
# libxc._itrf.LIBXC_is_hybrid.restype = ctypes.c_int
# libxc._itrf.LIBXC_is_nlc.restype = ctypes.c_int
# libxc._itrf.LIBXC_is_cam_rsh.restype = ctypes.c_int
# libxc._itrf.LIBXC_max_deriv_order.restype = ctypes.c_int
# libxc._itrf.LIBXC_number_of_functionals.restype = ctypes.c_int
# libxc._itrf.LIBXC_functional_numbers.argtypes = (
#     numpy.ctypeslib.ndpointer(dtype=numpy.intc, ndim=1,
#     flags=("W", "C", "A")), )
# libxc._itrf.LIBXC_functional_name.argtypes = [ctypes.c_int]
# libxc._itrf.LIBXC_functional_name.restype = ctypes.c_char_p
# libxc._itrf.LIBXC_hybrid_coeff.argtypes = [ctypes.c_int]
# libxc._itrf.LIBXC_hybrid_coeff.restype = ctypes.c_double
# libxc._itrf.LIBXC_nlc_coeff.argtypes = [
#     ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
# libxc._itrf.LIBXC_rsh_coeff.argtypes = [
#     ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
# libxc._itrf.LIBXC_version.restype = ctypes.c_char_p
# libxc._itrf.LIBXC_reference.restype = ctypes.c_char_p
# libxc._itrf.LIBXC_reference_doi.restype = ctypes.c_char_p
# libxc._itrf.LIBXC_xc_reference.argtypes = [
#     ctypes.c_int, (ctypes.c_char_p * 8)]
# libxc._itrf.xc_functional_get_number.argtypes = (ctypes.c_char_p, )
# libxc._itrf.xc_functional_get_number.restype = ctypes.c_int


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : (nkpts, nao, nao) or (*, nkpts, nao, nao) ndarray
        Veff = J + Vxc.
    '''
    if cell is None:
        cell = ks.cell
    if dm is None:
        dm = ks.make_rdm1()
    if kpts is None:
        kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if isinstance(kpts, numpy.ndarray):
        _kpts, _kpts_band, _dm = kpts, kpts_band, dm
        weight = 1. / len(kpts)
    else:
        if len(dm) != kpts.nkpts_ibz:
            raise KeyError('Shape of the input density matrix does not '
                           'match the number of IBZ k-points: '
                           f'{len(dm)} vs {kpts.nkpts_ibz}.')
        _dm = kpts.transform_dm(dm)
        _kpts, _kpts_band = kpts.kpts, kpts.kpts_ibz
        weight = kpts.weights_ibz

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, _dm, hermi,
                                       _kpts, _kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 3
                    and kpts_band is None)

    # For UniformGrids, grids.coords does not indicate whehter grids are
    # initialized
    def setup_grids():
        ks.grids.coords, ks.grids.weights = get_becke_grids(
            ks.grids.cell, ks.grids.atom_grid,
            radi_method=ks.grids.radi_method, level=ks.grids.level,
            prune=ks.grids.prune)
        ks.grids.non0tab = ks.grids.make_mask(ks.grids.cell, ks.grids.coords)
        logger.info(ks.grids, 'tot grids = %d', len(ks.grids.weights))
        logger.info(ks.grids, 'cell vol = %.9g  sum(weights) = %.9g',
                    cell.vol, ks.grids.weights.sum())
        if (isinstance(ks.grids, gen_grid.BeckeGrids)
                and ks.small_rho_cutoff > 1e-20 and ground_state):
            rho = ks.get_rho(dm, ks.grids, kpts)
            n = numpy.dot(rho, ks.grids.weights)
            if abs(n - cell.nelectron) < rks.NELEC_ERROR_TOL * n:
                rho *= ks.grids.weights
                idx = abs(rho) > ks.small_rho_cutoff / ks.grids.weights.size
                logger.debug(ks, 'Drop ks.grids %d',
                             ks.grids.weights.size - numpy.count_nonzero(idx))
                ks.grids.coords = numpy.asarray(
                    ks.grids.coords[idx], order='C')
                ks.grids.weights = numpy.asarray(
                    ks.grids.weights[idx], order='C')
                ks.grids.non0tab = ks.grids.make_mask(
                    ks.grids.cell, ks.grids.coords)

    def get_vxc(conn):
        if ks.grids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            setup_grids()
            t0 = logger.timer(ks, 'setting up grids', *t0)
            init_para = (ks.grids.coords,
                         ks.grids.weights, ks.grids.non0tab)
        else:
            init_para = ()

        t0 = (logger.process_clock(), logger.perf_counter())
        if hermi == 2:  # because rho = 0
            n, exc, vxc = 0, 0, 0
        else:
            max_memory = ks.max_memory - lib.current_memory()[0]
            n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, _dm, 0, hermi,
                                    _kpts, _kpts_band, max_memory=max_memory)
            logger.debug(ks, 'nelec by numeric integration = %s', n)
            t0 = logger.timer(ks, 'vxc', *t0)
        if conn is not None:
            conn.send((*init_para, n, exc, vxc))
        else:
            return *init_para, n, exc, vxc
    conn1, conn2 = multiprocessing.Pipe()
    if ks.parallel_vxc:
        multiprocessing.Process(target=get_vxc, args=(conn1, )).start()
    else:
        r = get_vxc(None, )

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        if ks.parallel_vxc:
            r = conn2.recv()
        if len(r) == 6:
            ks.grids.coords, ks.grids.weights, ks.grids.non0tab, \
                n, exc, vxc = r
        else:
            n, exc, vxc = r
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            ks.xc, spin=cell.spin)
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        if omega != 0:
            vklr = ks.get_k(cell, dm, hermi, kpts,
                            kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        if ks.parallel_vxc:
            r = conn2.recv()
        if len(r) == 6:
            ks.grids.coords, ks.grids.weights, ks.grids.non0tab, \
                n, exc, vxc = r
        else:
            n, exc, vxc = r
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        vxc += vj - vk * .5

        if ground_state:
            if isinstance(kpts, numpy.ndarray):
                exc -= numpy.einsum('Kij,Kji', dm, vk).real * .5 * .5 * weight
            else:
                exc -= numpy.einsum('K,Kij,Kji', weight, dm, vk).real * .5 * .5

    if ground_state:
        if isinstance(kpts, numpy.ndarray):
            ecoul = numpy.einsum('Kij,Kji', dm, vj).real * .5 * weight
        else:
            ecoul = numpy.einsum('K,Kij,Kji', weight, dm, vj).real * .5
    else:
        ecoul = None
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc


class KRKS(pdft.krks.KRKS):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    get_veff = get_veff
    parallel_vxc = True

    def rs_density_fit(self, auxbasis=None, with_df=None, *args, **kwargs):
        mf = rsdf_jk.density_fit(self, auxbasis, with_df=with_df, **kwargs)
        mf.with_df._j_only = True
        mf.grids = gen_grid.BeckeGrids(self.cell)
        mf.grids.level = getattr(__config__, 'pbc_dft_rks_RKS_grids_level',
                                 mf.grids.level)
        return mf
