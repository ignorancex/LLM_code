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
#     Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

from pyscf.pbc.df.rsdf_helper import (
    _get_refuniq_map, BOHR, _get_schwartz_data, _get_schwartz_dcut,
    _make_dijs_lst, _get_atom_Rcuts_3c, _get_Lsmin, KPT_DIFF_TOL,
    gamma_point, MoleNoBasSort)
import numpy
import ctypes
from pyscf import gto as mol_gto
from byteqc.cupbc.df import ft_ao as gint_ft_ao
from pyscf.pbc.df import ft_ao
from pyscf.pbc.lib.kpts_helper import (is_zero, member, unique)
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.df.rsdf_helper import _get_bvk_data, intor_j2c
from byteqc.cupbc.int3c2e import basis_seg_contraction
import cupy
import cupyx
from byteqc.lib import Mg, contraction, gpu_avail_bytes
from byteqc.cupbc import lib as cupbclib

KINT_MAX = getattr(__config__, 'pbc_df_rsdf_direct_helper_kint_max', 30)
J3C_ORDER = getattr(__config__, 'pbc_df_rsdf_direct_helper_j3c_order', 'ijL')

libgpbc = cupbclib.load_library('libgpbc')
libcgto = lib.load_library('libcgto')

''' These functions exist in previous implementations but are updated here.
'''


def get_aux_chg(auxcell, shls_slice=None):
    r""" Compute charge of the auxiliary basis, \int_Omega dr chi_P(r)

    Returns:
        The function returns a 1d numpy array of size auxcell.nao_nr().
    """
    G0 = numpy.zeros((1, 3))
    return ft_ao.ft_ao(auxcell, G0, shls_slice=shls_slice)[0].real


def get_aux_chg_cart(auxvhfopt):
    r""" Compute charge of the auxiliary basis, \int_Omega dr chi_P(r)

    Returns:
        The function returns a 1d numpy array of size auxcell.nao_nr().
    """
    G0 = numpy.zeros((1, 3))
    return gint_ft_ao.ft_ao_trans(auxvhfopt, G0)[:, 0].real


def kpts_to_kmesh(cell, kpts, kint_max=KINT_MAX):
    """ Check if kpt mesh includes the Gamma point. Generate the bvk kmesh
    only if it does.
    """
    kpts = numpy.reshape(kpts, (-1, 3))
    nkpts = len(kpts)
    if nkpts == 1:  # single-kpt (either Gamma or shifted)
        return None

    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    ksums = abs(scaled_k).sum(axis=1)
    mask = numpy.zeros_like(ksums, dtype=bool)
    found = False
    for kint in numpy.arange(1, kint_max):
        tmp = ksums * kint
        # mask = numpy.logical_or(mask, abs(numpy.round(tmp) - tmp) < 1e-6)
        mask |= abs(numpy.round(tmp) - tmp) < 1e-6
        if numpy.all(mask):
            found = True
            break
    if found:
        kmesh = (len(numpy.unique(scaled_k[:, 0])),
                 len(numpy.unique(scaled_k[:, 1])),
                 len(numpy.unique(scaled_k[:, 2])))
    else:
        kmesh = None
    return kmesh


def get_j2c_sr(mydf, auxcell=None, kpts=None, omega=None, verbose=None):
    r''' Calculate SR part of j2c for given kpts.
    '''
    if auxcell is None:
        auxcell = mydf.auxcell
    if kpts is None:
        kpts = numpy.zeros((1, 3))
    if verbose is None:
        verbose = mydf.verbose
    if omega is None:
        omega = mydf.omega_j2c
    omega_j2c = abs(omega)
    j2c = intor_j2c(auxcell, omega_j2c, kpts=kpts)
    return j2c


def remove_j2c_sr_G0_(mydf, j2c, kpts, auxcell=None, omega=None, exxdiv=None):
    if auxcell is None:
        auxcell = mydf.auxcell
    if omega is None:
        omega = mydf.omega_j2c
    omega_j2c = abs(omega)

    if not exxdiv:
        if auxcell.dimension == 3:
            qaux = get_aux_chg(auxcell)

            qaux2 = None
            g0_j2c = numpy.pi / omega_j2c**2. / auxcell.vol

            for k, kpt in enumerate(kpts):
                if is_zero(kpt):
                    if qaux2 is None:
                        qaux2 = numpy.outer(qaux, qaux)
                    j2c[k] -= qaux2 * g0_j2c
    return j2c


def get_j2c_lr(mydf, auxcell=None, kpts=None, omega=None, mesh=None, out=None,
               verbose=None):
    r''' Calculate LR part of j2c for given kpts.
    '''
    log = logger.new_logger(mydf, verbose)

    if auxcell is None:
        auxcell = mydf.auxcell
    if kpts is None:
        kpts = numpy.zeros((1, 3))
    omega_j2c = abs(mydf.omega_j2c) if omega is None else abs(omega)
    mesh_j2c = mydf.mesh_j2c if mesh is None else mesh

    naoaux = auxcell.nao_nr()
    if out is None:
        out = [
            numpy.zeros((naoaux, naoaux), dtype=numpy.float64) if gamma_point(
                kpt) else numpy.zeros((naoaux, naoaux), dtype=numpy.complex128)
            for kpt in kpts]
    j2c = out

    Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh_j2c)
    b = auxcell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory * .5e6 / 16 / auxcell.nao_nr()))
    log.debug2('j2c_lr: max_memory %s (MB)  blocksize %s', max_memory, blksize)

    for k, kpt in enumerate(kpts):
        coulG_lr = mydf.weighted_coulG(
            omega=omega_j2c, kpt=kpt, exx=False, mesh=mesh_j2c)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                                Gvbase, kpt).T

            if is_zero(kpt):  # kpti == kptj
                LkR = numpy.asarray(aoaux.real, order='C')
                LkI = numpy.asarray(aoaux.imag, order='C')
                j2c[k] += (LkR * coulG_lr[p0:p1]) @ LkR.T
                j2c[k] += (LkI * coulG_lr[p0:p1]) @ LkI.T
            else:
                j2c[k] += (aoaux.conj() * coulG_lr[p0:p1]) @ aoaux.T

            aoaux = None

    return out


def get_j2c(mydf, auxcell=None, kpts=None, omega=None, mesh=None, exxdiv=None,
            verbose=None, jktype="j2c"):
    log = logger.new_logger(mydf, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    t1 = (logger.process_clock(), logger.perf_counter())
    if auxcell is None:
        auxcell = mydf.auxcell
    if kpts is None:
        kpts = numpy.zeros((1, 3))
    if verbose is None:
        verbose = mydf.verbose
    if omega is None:
        omega = mydf.omega_j2c

    j2c = get_j2c_sr(mydf, auxcell=auxcell, kpts=kpts,
                     omega=omega, verbose=verbose)
    t1 = log.timer_debug1('get_j2c sr', *t1)

    j2c = remove_j2c_sr_G0_(mydf, j2c, kpts=kpts, auxcell=auxcell, omega=omega,
                            exxdiv=exxdiv)
    t1 = log.timer_debug1('get_j2c G0', *t1)

    j2c = get_j2c_lr(
        mydf, auxcell=auxcell, kpts=kpts, omega=omega, mesh=mesh, out=j2c,
        verbose=verbose)
    t1 = log.timer_debug1('get_j2c lr', *t1)
    t0 = log.timer_debug1('get_j2c', *t0)
    return j2c


'''
Transform the C ._bas to Python .basis
'''


def _bas2_basis(mol):
    uniq_atms = []
    uniq_atms_id = []
    basis = {}
    for i in range(mol.natm):
        atm = mol.atom_symbol(i)
        if atm not in uniq_atms:
            uniq_atms.append(atm)
            uniq_atms_id.append(i)
            basis[atm] = []
    for bas in mol._bas:
        ptr_exp, ptr_coeff, atm_id = bas[mol_gto.PTR_EXP], \
            bas[mol_gto.PTR_COEFF], bas[mol_gto.ATOM_OF]
        if atm_id not in uniq_atms_id:
            continue
        sym = mol.atom_symbol(atm_id)
        angl, nctr, nprim = bas[mol_gto.ANG_OF], bas[mol_gto.NCTR_OF], \
            bas[mol_gto.NPRIM_OF]
        coeff_shape = (nctr, nprim)
        es = numpy.zeros(nprim)
        cs = numpy.zeros(numpy.prod(coeff_shape))
        for prim in range(nprim):
            es[prim] = mol._env[ptr_exp + prim]
        for i in range(nprim * nctr):
            cs[i] = mol._env[ptr_coeff + i]
        cs = cs.reshape(coeff_shape).T
        cs = numpy.einsum('pi,p->pi', cs, 1 / mol_gto.gto_norm(angl, es))
        info = [int(angl)]
        for i in range(nprim):
            info.append([es[i]] + list(cs[i, :]))
        basis[sym].append(info)
    return basis


def cholesky_decomposed_metric(
        mydf, j2c, j2c_eig_always=None, linear_dep_threshold=None):
    import scipy.linalg

    log = logger.new_logger(mydf)

    if j2c_eig_always is None:
        j2c_eig_always = mydf.j2c_eig_always
    if linear_dep_threshold is None:
        linear_dep_threshold = mydf.linear_dep_threshold
    cell = mydf.cell

    j2c_negative = None
    try:
        if j2c_eig_always:
            raise scipy.linalg.LinAlgError
        j2c = scipy.linalg.cholesky(j2c, lower=True)
        j2ctag = 'CD'
    except scipy.linalg.LinAlgError:
        # msg =('===================================\n'
        #      'J-metric not positive definite.\n'
        #      'It is likely that mesh is not enough.\n'
        #      '===================================')
        # log.error(msg)
        # raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
        w, v = scipy.linalg.eigh(j2c)
        ndrop = numpy.count_nonzero(w < linear_dep_threshold)
        if ndrop > 0:
            # log.debug('DF metric linear dependency for kpt %s',
            #           uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns', w[-1] / w[0], ndrop)
        w = 1 / w
        w[w > 1 / linear_dep_threshold] = 0
        w[w < 0] = 0
        w = numpy.sqrt(w)
        v1 = v.conj().T * w.reshape(-1, 1)
        # v1 = v[:, w > linear_dep_threshold].conj().T
        # v1 /= numpy.sqrt(w[w > linear_dep_threshold]).reshape(-1, 1)
        j2c = v1
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            idx = numpy.where(w < -linear_dep_threshold)[0]
            if len(idx) > 0:
                j2c_negative = (v[:, idx] / numpy.sqrt(-w[idx])).conj().T
        w = v = None
        j2ctag = 'eig'
    return j2c, j2c_negative, j2ctag


def get_3c2e_Rcuts(bas_lst_or_mol, auxbas_lst_or_auxmol, dijs_lst, omega,
                   precision, estimator, Qijs_lst):
    """ Given a list of basis ("bas_lst") and auxiliary basis ("auxbas_lst"),
    determine the cutoff radius for
        2-norm( (k|v_SR(omega)|ij) ) < precision
    where i and j shls are separated by d specified by "dijs_lst".
    """

    if isinstance(bas_lst_or_mol, mol_gto.mole.Mole):
        mol = bas_lst_or_mol
    else:
        bas_lst = bas_lst_or_mol
        mol = MoleNoBasSort()
        mol.build(dump_input=False, parse_arg=False,
                  atom="H 0 0 0", basis=bas_lst, spin=None)

    if isinstance(auxbas_lst_or_auxmol, mol_gto.mole.Mole):
        auxmol = auxbas_lst_or_auxmol
    else:
        auxbas_lst = auxbas_lst_or_auxmol
        auxmol = MoleNoBasSort()
        auxmol.build(dump_input=False, parse_arg=False,
                     atom="H 0 0 0", basis=auxbas_lst, spin=None)

    dijs = cupy.asarray(numpy.concatenate(dijs_lst))
    Qijs = cupy.asarray(numpy.concatenate(Qijs_lst))
    dijs_ij = numpy.zeros(dijs.shape, dtype=numpy.int32)
    count = 0
    for ind, dij in enumerate(dijs_lst):
        dijs_ij[count:count + dij.shape[0]] = ind
        count += dij.shape[0]
    dijs_ij = cupy.asarray(dijs_ij)
    nbasaux = auxmol.nbas
    nbas = mol.nbas
    Rcuts = cupy.empty(len(Qijs) * nbasaux)

    eks = cupy.asarray([auxmol.bas_exp(ksh)[-1] for ksh in range(nbasaux)])
    lks = cupy.asarray([int(auxmol.bas_angular(ksh))
                       for ksh in range(nbasaux)], dtype=numpy.int32)
    cks = cupy.asarray([abs(auxmol._libcint_ctr_coeff(ksh)[-1]).max()
                        for ksh in range(nbasaux)])

    ls = cupy.asarray(mol._bas[:, mol_gto.ANG_OF])
    es = [mol.bas_exp(sh) for sh in range(nbas)]
    imin = [es[sh].argmin() for sh in range(nbas)]
    es = cupy.asarray([es[sh][imin[sh]] for sh in range(nbas)])
    cs = cupy.asarray([abs(mol._libcint_ctr_coeff(sh)[imin[sh]]).max()
                       for sh in range(nbas)])
    assert estimator == "ME", "Not implemented"
    libgpbc.GINT_get_3c2e_Rcuts(
        ctypes.c_int(len(dijs)), ctypes.c_int(nbasaux),
        ctypes.cast(Rcuts.data.ptr, ctypes.c_void_p),
        ctypes.cast(dijs.data.ptr, ctypes.c_void_p),
        ctypes.cast(Qijs.data.ptr, ctypes.c_void_p),
        ctypes.cast(dijs_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(ls.data.ptr, ctypes.c_void_p),
        ctypes.cast(es.data.ptr, ctypes.c_void_p),
        ctypes.cast(cs.data.ptr, ctypes.c_void_p),
        ctypes.cast(lks.data.ptr, ctypes.c_void_p),
        ctypes.cast(eks.data.ptr, ctypes.c_void_p),
        ctypes.cast(cks.data.ptr, ctypes.c_void_p),
        ctypes.c_double(omega), ctypes.c_double(precision), ctypes.c_int(0))
    return Rcuts


def get_prescreening_data(cell, auxcell, omega, precision=None, estimator='ME',
                          verbose=None):
    log = logger.new_logger(cell, verbose)
    _cell = mol_gto.copy(cell)
    _auxcell = mol_gto.copy(auxcell)
    cell, _ = basis_seg_contraction(cell)
    auxcell, _ = basis_seg_contraction(auxcell)
    cell._basis = _bas2_basis(cell)
    auxcell._basis = _bas2_basis(auxcell)

    # prescreening data
    t1 = (logger.process_clock(), logger.perf_counter())
    if precision is None:
        precision = cell.precision
    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = _get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
        _get_refuniq_map(auxcell)
    _refuniqshl_map, _uniq_atms, _uniq_bas, _uniq_bas_loc = _get_refuniq_map(
        _cell)
    _auxuniqshl_map, _uniq_atms, _uniq_basaux, _uniq_basaux_loc = \
        _get_refuniq_map(_auxcell)

    nuniq_bas = len(uniq_bas)
    inds = []
    for i in range(_cell.nbas):
        inds.extend([i] * (_cell._bas[i, mol_gto.NPRIM_OF]
                    if _cell._bas[i, mol_gto.NCTR_OF] != 1 else 1))
    inds = [_refuniqshl_map[inds[numpy.where(refuniqshl_map == i)[
        0][0]]] for i in range(nuniq_bas)]
    nuniq_basaux = len(uniq_basaux)
    _nuniq_basaux = len(_uniq_basaux)

    dstep = 1  # 1 Ang bin size for shl pair
    dstep_BOHR = dstep / BOHR
    _Qauxs = _get_schwartz_data(_uniq_basaux, omega, keep1ctr=False, safe=True)
    _dcuts = _get_schwartz_dcut(_uniq_bas, omega, precision / _Qauxs.max(),
                                r0=_cell.rcut)
    dcuts = numpy.empty((nuniq_bas * (nuniq_bas + 1) // 2,), _dcuts.dtype)
    for i in range(nuniq_bas):
        for j in range(i + 1):
            dcuts[i * (i + 1) // 2 + j] = \
                _dcuts[inds[i] * (inds[i] + 1) // 2 + inds[j]]
    dijs_lst = _make_dijs_lst(dcuts, dstep_BOHR)
    dijs_loc = numpy.cumsum([0] + [len(dijs)
                            for dijs in dijs_lst]).astype(numpy.int32)

    _dijs_lst = _make_dijs_lst(_dcuts, dstep_BOHR)
    _dijs_loc = numpy.cumsum([0] + [len(dijs)
                                    for dijs in _dijs_lst]).astype(numpy.int32)
    if estimator.upper() in ["ISFQ0", "ISFQL"]:
        _Qs_lst = _get_schwartz_data(
            _uniq_bas, omega, _dijs_lst, keep1ctr=True, safe=True)
    else:
        _Qs_lst = [numpy.zeros_like(dijs) for dijs in _dijs_lst]
    _Rcuts = get_3c2e_Rcuts(
        _uniq_bas, _uniq_basaux, _dijs_lst, omega, precision, estimator,
        _Qs_lst).get()
    Rcuts = numpy.empty(
        (sum(dijs.shape[0] for dijs in dijs_lst) * nuniq_basaux,),
        _Rcuts.dtype)
    for i in range(nuniq_bas):
        for j in range(i + 1):
            ind = slice(dijs_loc[i * (i + 1) // 2 + j] * nuniq_basaux,
                        dijs_loc[i * (i + 1) // 2 + j + 1] * nuniq_basaux)
            _ind = slice(
                _dijs_loc[inds[i] * (inds[i] + 1) // 2 + inds[j]]
                * _nuniq_basaux,
                _dijs_loc[inds[i] * (inds[i] + 1) // 2 + inds[j] + 1]
                * _nuniq_basaux)
            Rcuts[ind] = _Rcuts[_ind]
    _bas_exps = numpy.array([numpy.asarray(b[1:])[:, 0].min()
                            for b in _uniq_bas])
    _atom_Rcuts = _get_atom_Rcuts_3c(
        _Rcuts, _dijs_lst, _bas_exps, _uniq_bas_loc, _uniq_basaux_loc)
    cell_rcut = _atom_Rcuts.max()
    _uniqexp = numpy.array([numpy.asarray(b[1:])[:, 0].min()
                           for b in _uniq_bas])
    uniqexp = _uniqexp[inds]
    dcut2s = dcuts**2.
    Rcut2s = Rcuts**2.
    Ls = _get_Lsmin(cell, _atom_Rcuts, uniq_atms)
    prescreening_data = (refuniqshl_map, auxuniqshl_map, nuniq_basaux, uniqexp,
                         dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls)
    log.debug("j3c prescreening: cell rcut %.2f Bohr  keep %d imgs",
              cell_rcut, Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
    return prescreening_data


def get_j3c_pre(mydf, vhfopt, kptij_lst=numpy.zeros((1, 2, 3)), aosym='s1',
                j3c_order=J3C_ORDER, verbose=None, prescreen_mask=None,
                jktype='j', bvk_kmesh=None):
    if prescreen_mask is None:
        prescreen_mask = mydf.prescreen_mask
    log = logger.new_logger(mydf, verbose=verbose)

    if aosym[:2] not in ['s1', 's2']:
        log.error('Invalid aosym %s (must start with "s1" or "s2").', aosym)
        raise ValueError

    if hasattr(bvk_kmesh, '__len__') and len(bvk_kmesh) == 2:
        bvk_kmesh_R, bvk_kmesh_G = bvk_kmesh
    else:
        bvk_kmesh_R = bvk_kmesh_G = bvk_kmesh
    log.debug1('Using bvk_kmesh_R= %s  bvk_kmesh_G= %s',
               bvk_kmesh_R, bvk_kmesh_G)

    cell = mydf.cell
    auxcell = mydf.auxcell
    naoaux = auxcell.nao_nr()

    ncptype_ij = vhfopt.ncptype
    log.debug1("ncptype:%s" % ncptype_ij)

    xs = [x for x in loop_uniq_q(mydf, kptij_lst=kptij_lst, verbose=0)]
    uniq_kpts = numpy.asarray([x[0] for x in xs])
    xs = None

    if aosym[:2] == 's2':
        if not is_j_only(kptij_lst):
            log.error(
                'aosym = "s2" must be used with kpti = kptj, '
                'i.e., j-only mode.')
            raise RuntimeError

        if j3c_order == 'Lij':
            log.error(
                's2 symmetry for j3c_order = "Lij" is not implemented (yet). '
                'Use j3c_order = "ijL" instead.')
            raise NotImplementedError

    omega = mydf.omega
    mesh = mydf.mesh_compact
    if 'get_j3c_pre' in mydf.rtcache:
        Ls, ovlp_mask, prescreen_data, b, Gv, Gvbase, gxyz = mydf.rtcache[
            'get_j3c_pre']
    else:
        # precompute ovlp_mask
        Ls = vhfopt.mol.get_lattice_Ls()
        Ls = Ls[numpy.linalg.norm(Ls, axis=1).argsort()]
        ovlp_mask = gint_ft_ao._estimate_overlap(
            vhfopt.mol, Ls) > vhfopt.mol.precision
        ovlp_mask = numpy.asarray(ovlp_mask, dtype=numpy.int8, order='C')
        # precompute gint3c
        prescreen_data = get_prescreening_data(
            cell, auxcell, omega, precision=mydf.precision_R, verbose=verbose)
        # precompute kgLR/I

        b = cell.reciprocal_vectors()
        Gv, Gvbase, _ = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        mydf.rtcache['get_j3c_pre'] = Ls, ovlp_mask, prescreen_data, b, Gv, \
            Gvbase, gxyz

    if 'get_j3c_pre_kgL_%s' % jktype in mydf.rtcache:
        kgL = mydf.rtcache['get_j3c_pre_kgL_%s' % jktype]
    else:
        auxshls_slice = (0, auxcell.nbas)
        ngrids = gxyz.shape[0]
        kgL = cupyx.empty_pinned(
            (len(uniq_kpts), ngrids, naoaux), dtype=numpy.complex128)
        for k, kpt in enumerate(uniq_kpts):
            Gaux = ft_ao.ft_ao(auxcell, Gv, auxshls_slice,
                               b, gxyz, Gvbase, kpt)
            wcoulG_lr = mydf.weighted_coulG(
                omega=omega, kpt=kpt, exx=False, mesh=mesh)
            Gaux *= wcoulG_lr.reshape(-1, 1)
            kgL[k] = Gaux
            Gaux = None
        mydf.rtcache['get_j3c_pre_kgL_%s' % jktype] = kgL

    gint3c = get_int3c(
        cell, auxcell, omega, vhfopt=vhfopt, precision=mydf.precision_R,
        kptij_lst=kptij_lst, verbose=verbose, bvk_kmesh=bvk_kmesh_R,
        aosym=aosym, j3c_order=j3c_order, prescreen_mask=prescreen_mask,
        prescreen_data=prescreen_data)
    return gint3c, (Ls, ovlp_mask, prescreen_data, b, Gv, Gvbase, gxyz, kgL)


def get_j3c(mydf, cp_ij_id, vhfopt, kptij_lst=None,
            omega=None, aosym='s2ij', j3c_order=J3C_ORDER, comp=None,
            bvk_kmesh_R=None, bvk_kmesh_G=None,
            precision=None, mesh=None, estimator='ME', exxdiv=None,
            gint3c=None, out=None, verbose=None, j3c_mask=3,
            prescreen_mask=None, j3c_pre_data=None):
    if prescreen_mask is None:
        prescreen_mask = mydf.prescreen_mask
    cell = mydf.cell
    auxcell = mydf.auxcell
    if kptij_lst is None:
        numpy.zeros((1, 2, 3))  # gamma point only
    if omega is None:
        omega = mydf.omega
    if precision is None:
        precision = mydf.precision_R
    if mesh is None:
        mesh = mydf.mesh_compact
    if verbose is None:
        verbose = mydf.verbose

    # determine aosym
    if aosym[:2] == 's2' and is_j_only(kptij_lst):
        aosym = 's2'
    else:
        aosym = 's1'

    log = logger.new_logger(mydf, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    j3c_cupy = gint3c(cp_ij_id, out)
    t0 = log.timer_debug1("get_j3c[%s] sr" % (cp_ij_id), *t0)

    if j3c_mask & 1 != 0:
        remove_j3c_sr_G0_q_(
            mydf, j3c_cupy, cp_ij_id, vhfopt, kptij_lst, cell=cell,
            auxcell=auxcell, omega=omega, aosym=aosym, j3c_order=j3c_order,
            exxdiv=exxdiv)
        t0 = log.timer_debug1("get_j3c[%s] g0" % (cp_ij_id), *t0)

    if j3c_mask & 2 != 0:
        add_j3c_lr_(
            mydf, j3c_cupy, cp_ij_id, vhfopt, cell=vhfopt.mol, auxcell=auxcell,
            kptij_lst=kptij_lst, omega=omega, mesh=mesh, aosym=aosym,
            j3c_order=j3c_order, comp=comp, bvk_kmesh=bvk_kmesh_G,
            j3c_pre_data=j3c_pre_data, verbose=verbose)
        t0 = log.timer_debug1("get_j3c[%s] lr" % (cp_ij_id), *t0)

    return j3c_cupy


def get_int3c(
        cell, auxcell, omega, vhfopt, auxvhfopt, precision=None,
        kptij_lst=numpy.zeros((1, 2, 3)), intor='int3c2e', comp=None,
        estimator='ME', verbose=None, bvk_kmesh=None, aosym='s1',
        j3c_order=J3C_ORDER, prescreen_mask=2, prescreen_data=None):
    if prescreen_data is None:
        prescreen_data = get_prescreening_data(cell, auxcell, omega,
                                               precision=precision,
                                               estimator=estimator,
                                               verbose=verbose)
    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp, dcut2s, \
        dstep_BOHR, Rcut2s, dijs_loc, Ls = prescreen_data
    seg_cell = vhfopt.mol
    seg_auxcell = auxvhfopt.mol

    sorted_prescreening_data = refuniqshl_map[vhfopt.sorted_idx], \
        auxuniqshl_map[auxvhfopt.sorted_idx], nbasauxuniq, uniqexp, \
        dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls
    intor, comp = mol_gto.moleintor._get_intor_and_comp(
        seg_cell._add_suffix(intor), comp)

    shlpr_mask = numpy.ones((seg_cell.nbas, seg_cell.nbas),
                            dtype=numpy.int8, order="C")
    aux_id_range = auxvhfopt.ncptype
    gint3c = wrap_gint3c_nospltbas(
        seg_cell, seg_auxcell, omega, aux_id_range, vhfopt, auxvhfopt,
        shlpr_mask, sorted_prescreening_data, intor, aosym, comp, kptij_lst,
        bvk_kmesh=bvk_kmesh, order=j3c_order, verbose=verbose,
        prescreen_mask=prescreen_mask)
    return gint3c


def wrap_gint3c_nospltbas(
        cell, auxcell, omega, aux_id_range, vhfopt, auxvhfopt, shlpr_mask,
        sorted_prescreening_data, intor='int3c2e', aosym='s1', comp=1,
        kptij_lst=numpy.zeros((1, 2, 3)), bvk_kmesh=None, order='ijL',
        verbose=None, diag_block_with_triu=True, prescreen_mask=2):
    log = logger.new_logger(cell, verbose)

    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp, dcut2s, dstep_BOHR, \
        Rcut2s, dijs_loc, Ls = sorted_prescreening_data

    intor = cell._add_suffix(intor)

    cell._env[mol_gto.PTR_RANGE_OMEGA] = abs(omega)
    auxcell._env[mol_gto.PTR_RANGE_OMEGA] = abs(omega)
    nimgs = len(Ls)
    nbas = cell.nbas
    auxnbas = auxcell.nbas
    kpti = kptij_lst[:, 0]
    kptj = kptij_lst[:, 1]
    ao_loc = cell.ao_loc
    auxao_loc = auxcell.ao_loc
    if bvk_kmesh is None:
        Ls_ = Ls
    else:
        Ls, Ls_, cell_loc_bvk = _get_bvk_data(cell, Ls, bvk_kmesh)
        bvk_nimgs = Ls_.shape[0]

    if gamma_point(kptij_lst):
        # assert(aosym[:2] == "s2")
        kk_type = 'g'
        dtype = numpy.double
        nkpts = nkptij = 1
        kptij_idx = numpy.array([0], dtype=numpy.int32)
        expkL = numpy.ones(1)
    elif is_zero(kpti - kptj):  # j_only
        kk_type = 'k'
        dtype = numpy.complex128
        kpts = kptij_idx = numpy.asarray(kpti, order='C')
        expkL = numpy.exp(1j * numpy.dot(kpts, Ls_.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = numpy.complex128
        kpts = unique(numpy.vstack([kpti, kptj]))[0]
        expkL = numpy.exp(1j * numpy.dot(kpts, Ls_.T))
        wherei = numpy.where(abs(kpti.reshape(-1, 1, 3) - kpts).sum(axis=2)
                             < KPT_DIFF_TOL)[1]
        wherej = numpy.where(abs(kptj.reshape(-1, 1, 3) - kpts).sum(axis=2)
                             < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = numpy.asarray(wherei * nkpts + wherej, dtype=numpy.int32)
        nkptij = len(kptij_lst)

    cfunc_prefix = "GINTPBCsr3c"
    if not (gamma_point(kptij_lst) or bvk_kmesh is None):
        cfunc_prefix += "_bvk"
    fill = "%s_%s%s" % (cfunc_prefix, kk_type, aosym[:2])
    if order == 'Lij':
        fill += '_Lij'
    drv = getattr(libgpbc, "%s_%s_drv" % (cfunc_prefix, kk_type))

    log.debug1("Using %s to evaluate SR integrals", fill)
    if diag_block_with_triu:
        scale_shellpair_diag = 1.
    else:
        scale_shellpair_diag = 0.5
    l_ctr_offsets = vhfopt.l_ctr_offsets
    _atm, _bas, _env = mol_gto.conc_env(
        cell._atm, cell._bas, cell._env, auxcell._atm, auxcell._bas,
        auxcell._env)
    expkLs = Mg.broadcast(expkL)

    if gamma_point(kptij_lst):
        def int3c(cp_ij_id, out):
            gpu_id = cupy.cuda.runtime.getDevice()
            cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
            cpj = cp_ij_id - cpi * (cpi + 1) // 2
            ish0 = l_ctr_offsets[cpi]
            ish1 = l_ctr_offsets[cpi + 1]
            jsh0 = l_ctr_offsets[cpj]
            jsh1 = l_ctr_offsets[cpj + 1]
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            shls_slice = (ish0, ish1, jsh0, jsh1, 0, auxnbas)

            dij = (i1 - i0) * (j1 - j0)
            naux_cart, _ = auxvhfopt.coeff.shape
            numThreads = 8
            maxref = max(refuniqshl_map)
            maxref2 = maxref * (maxref + 3) // 2
            npair = vhfopt.bas_pairs_locs[cp_ij_id + 1] - \
                vhfopt.bas_pairs_locs[cp_ij_id]
            nprim = vhfopt.bpcache.contents.primitive_pairs_locs[
                cp_ij_id + 1] - vhfopt.bpcache.contents.primitive_pairs_locs[
                    cp_ij_id]
            buf_size = [vhfopt.bas_pairs_locs[-1] * 8,  # d_bas_pair2bra(ket)
                        (nbas + auxnbas) * 32,  # d_bas
                        cell.natm * 48,  # d_atm
                        _env.size * 8,  # d_env
                        Ls.size * 8,  # d_Ls
                        nbas * 4,  # d_refuniqshl_map
                        auxnbas * 4,  # d_auxuniqshl_map
                        maxref * 8 + 8,  # d_uniqexp
                        (maxref2 + 1) * 8,  # d_uniq_dcut2s
                        # d_uniq_Rcut2s
                        nbasauxuniq * dijs_loc[maxref2 + 1] * 8,
                        (maxref2 + 2) * 4,  # d_uniqshlpr_dij_loc
                        (nbas + 1) * 4,  # d_ao_loc
                        (auxvhfopt.ncptype + 1) * numThreads * 8,
                        # d_prescreen_n_bas_pairs
                        nbas * numThreads * 48,  # d_bas_coords
                        (2 * npair + 1) * numThreads * 4,
                        # d_prescreen_bas_pair2bra
                        npair * auxnbas * numThreads * 8,
                        # !!!d_prescreen_bas_aux
                        nprim * numThreads * 40,  # d_aexyz
                        auxnbas * 4,  # d_auxbas_pairs_locs
                        auxvhfopt.ncptype * 6 * (66**3)  # tmp_idx4c
                        ]
            if out is None:
                out = cupy.ndarray((1, 1, dij, naux_cart), dtype='f8')
            buf_size = sum(buf_size)
            buf_size += 256 * (14 + 5 * numThreads + auxvhfopt.ncptype)
            log.debug1("Alloc %.2fMB for sr buffing" % (buf_size / (1024**2)))
            gpu_buf = cupy.ndarray(shape=(buf_size, ), dtype='int8')

            assert out.dtype == dtype
            drv(ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
                ctypes.c_ulong(buf_size),
                ctypes.c_int(comp), ctypes.c_int(nimgs),
                Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(Ls.size),
                (ctypes.c_int * 6)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                auxao_loc.ctypes.data_as(ctypes.c_void_p),
                shlpr_mask.ctypes.data_as(ctypes.c_void_p),
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(max(refuniqshl_map)),
                ctypes.c_int(nbasauxuniq),
                uniqexp.ctypes.data_as(ctypes.c_void_p),
                dcut2s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(dstep_BOHR),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                _atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                _bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas), _env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(_env.size), ctypes.c_int(cp_ij_id),
                ctypes.c_int(aux_id_range),
                vhfopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                vhfopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(vhfopt.ncptype), auxvhfopt.bpcaches[gpu_id],
                ctypes.c_double(scale_shellpair_diag),
                ctypes.c_int(prescreen_mask))
            gpu_buf = None
            return out

    else:
        if bvk_kmesh is None:
            raise NotImplementedError

        def int3c(cp_ij_id, out):
            gpu_id = cupy.cuda.runtime.getDevice()
            cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
            cpj = cp_ij_id - cpi * (cpi + 1) // 2
            ish0 = l_ctr_offsets[cpi]
            ish1 = l_ctr_offsets[cpi + 1]
            jsh0 = l_ctr_offsets[cpj]
            jsh1 = l_ctr_offsets[cpj + 1]
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            shls_slice = (ish0, ish1, jsh0, jsh1, 0, auxnbas)

            dij = (i1 - i0) * (j1 - j0)
            naux_cart, naux = auxvhfopt.coeff.shape
            numThreads = 8
            maxref = max(refuniqshl_map)
            maxref2 = maxref * (maxref + 3) // 2
            npair = vhfopt.bas_pairs_locs[cp_ij_id + 1] - \
                vhfopt.bas_pairs_locs[cp_ij_id]
            nprim = vhfopt.bpcache.contents.primitive_pairs_locs[
                cp_ij_id + 1] - vhfopt.bpcache.contents.primitive_pairs_locs[
                    cp_ij_id]
            if kk_type == 'k':
                buf_size = sum([bvk_nimgs * nkpts * 16,  # d_bufexp
                                bvk_nimgs * dij * naux_cart * 8,  # !!!d_bufL
                                # d_bas_pair2bra
                                vhfopt.bas_pairs_locs[-1] * 8,
                                (nbas + auxnbas) * 32,  # d_bas
                                cell.natm * 48,  # d_atm
                                _env.size * 8,  # d_env
                                Ls.size * 8,  # d_Ls
                                nbas * 4,  # d_refuniqshl_map
                                auxnbas * 4,  # d_auxuniqshl_map
                                maxref * 8 + 8,  # d_uniqexp
                                (maxref2 + 1) * 8,  # d_uniq_dcut2s
                                # d_uniq_Rcut2s
                                nbasauxuniq * dijs_loc[maxref2 + 1] * 8,
                                (maxref2 + 2) * 4,  # d_uniqshlpr_dij_loc
                                (nbas + 1) * 4,  # d_ao_loc
                                (auxvhfopt.ncptype + 1) * numThreads * 8,
                                # d_prescreen_n_bas_pairs
                                nbas * numThreads * 48,  # d_bas_coords
                                # d_prescreen_bas_pair2bra
                                (2 * npair + 1) * numThreads * 4,
                                npair * auxnbas * numThreads * 8,
                                # !!!d_prescreen_bas_aux
                                nprim * numThreads * 40,  # d_aexyz
                                auxnbas * 4,  # d_auxbas_pairs_locs
                                auxvhfopt.ncptype * 6 * (66**3)  # tmp_idx4c
                                ])
                buf_size += 256 * (16 + 5 * numThreads + auxvhfopt.ncptype)
                if out is None:
                    out = cupy.ndarray(
                        (nkpts, 1, dij, naux_cart), dtype='complex128')
            else:
                buf_size = sum([min(bvk_nimgs, 80) * nkpts * dij
                                * naux_cart * 16,  # !!!d_bufkL
                                bvk_nimgs * dij * naux_cart * 8,  # !!!d_bufL
                                # d_bas_pair2bra
                                vhfopt.bas_pairs_locs[-1] * 8,
                                (nbas + auxnbas) * 32,  # d_bas
                                cell.natm * 48,  # d_atm
                                _env.size * 8,  # d_env
                                Ls.size * 8,  # d_Ls
                                nbas * 4,  # d_refuniqshl_map
                                auxnbas * 4,  # d_auxuniqshl_map
                                maxref * 8 + 8,  # d_uniqexp
                                (maxref2 + 1) * 8,  # d_uniq_dcut2s
                                # d_uniq_Rcut2s
                                nbasauxuniq * dijs_loc[maxref2 + 1] * 8,
                                (maxref2 + 2) * 4,  # d_uniqshlpr_dij_loc
                                (nbas + 1) * 4,  # d_ao_loc
                                (auxvhfopt.ncptype + 1) * numThreads * 8,
                                # d_prescreen_n_bas_pairs
                                nbas * numThreads * 48,  # d_bas_coords
                                # d_prescreen_bas_pair2bra
                                (2 * npair + 1) * numThreads * 4,
                                npair * auxnbas * numThreads * 8,
                                # !!!d_prescreen_bas_aux
                                nprim * numThreads * 40,  # d_aexyz
                                auxnbas * 4,  # d_auxbas_pairs_locs
                                auxvhfopt.ncptype * 6 * (66**3)])  # tmp_idx4c
                buf_size += 256 * (16 + 5 * numThreads + auxvhfopt.ncptype)
                if out is None:
                    out = cupy.ndarray(
                        (nkpts * nkpts, 1, dij, naux_cart), dtype='complex128')
            log.debug1("Alloc %.2fMB for sr buffing" % (buf_size / (1024**2)))
            gpu_buf = cupy.ndarray(shape=(buf_size, ), dtype='int8')

            assert out.dtype == dtype
            drv(ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.cast(gpu_buf.data.ptr, ctypes.c_void_p),
                ctypes.c_ulong(buf_size),
                ctypes.c_int(nkptij), ctypes.c_int(nkpts),
                ctypes.c_int(comp), ctypes.c_int(nimgs),
                ctypes.c_int(bvk_nimgs),
                Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(Ls.size),
                ctypes.cast(expkLs[gpu_id].data.ptr, ctypes.c_void_p),
                kptij_idx.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 6)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                auxao_loc.ctypes.data_as(ctypes.c_void_p),
                cell_loc_bvk.ctypes.data_as(ctypes.c_void_p),  # cell_loc_bvk
                shlpr_mask.ctypes.data_as(ctypes.c_void_p),   # shlpr_mask
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(max(refuniqshl_map)),
                ctypes.c_int(nbasauxuniq),
                uniqexp.ctypes.data_as(ctypes.c_void_p),
                dcut2s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(dstep_BOHR),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                _atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                _bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas), _env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(_env.size), ctypes.c_int(cp_ij_id),
                ctypes.c_int(aux_id_range),
                vhfopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                vhfopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(vhfopt.ncptype), auxvhfopt.bpcaches[gpu_id],
                ctypes.c_double(scale_shellpair_diag),
                ctypes.c_int(prescreen_mask))
            gpu_buf = None

            return out
    return int3c


j3c_g0_kernel = cupy.RawKernel(
    r'''
            # include "cuComplex.h"
            extern "C" __global__
            void j3c_g0_kernel(cuDoubleComplex* j3c, const double* vbar,
                const cuDoubleComplex* ovlp, const long* adapted_ji_idx,
                const long adapted_ji_idx_len, const long nij, const long nl)
            {
                const long bid = blockIdx.x;
                const long k  = bid / nij;
                const long ij = bid % nij;
                const long idx = adapted_ji_idx[k];
                const cuDoubleComplex ovlp_ij = ovlp[k*nij + ij];
                for (long l = threadIdx.x; l < nl; l += blockDim.x) {
                    const double vbar_l = vbar[l];
                    if (abs(vbar_l) < 1e-10) continue;
                    cuDoubleComplex j3c_sub = make_cuDoubleComplex(vbar_l *
                        ovlp_ij.x, vbar_l * ovlp_ij.y);
                    j3c[l + ij * nl + idx * nl * nij] = cuCsub(j3c[l + ij * nl
                        + idx * nl * nij], j3c_sub);
                }
            }
        ''',
    'j3c_g0_kernel')

j3c_g0_real_kernel = cupy.RawKernel(
    r'''
            extern "C" __global__
            void j3c_g0_real_kernel(double* j3c, const double* vbar,
                const double* ovlp, const long* adapted_ji_idx,
                const long adapted_ji_idx_len, const long nij, const long nl)
            {
                const long bid = blockIdx.x;
                const long k  = bid / nij;
                const long ij = bid % nij;
                const long idx = adapted_ji_idx[k];
                const double ovlp_ij = ovlp[k*nij + ij];
                for (long l = threadIdx.x; l < nl; l += blockDim.x) {
                    const double vbar_l = vbar[l];
                    if (abs(vbar_l) < 1e-10) continue;
                    double j3c_sub = vbar_l * ovlp_ij;
                    j3c[l + ij * nl + idx * nl * nij] -= j3c_sub;
                }
            }
        ''',
    'j3c_g0_real_kernel')


def remove_j3c_sr_G0_q_(
        mydf, j3c, cp_ij_id, vhfopt, kptij_lst, cell=None, auxcell=None,
        omega=None, aosym='s2ij', j3c_order=J3C_ORDER, exxdiv=None, ovlps=None,
        vbar=None, adapted_ji_idxs=None):
    r''' Calculate G=0 correction to SR j3c. This will modify the input j3c
         in situ.
    Args:
        aosym (str):
            's1' or 's2'. This has to be consistent with the input j3c,
            shls_slice, and kptij_lst.
    '''
    if cell is None:
        cell = mydf.cell
    if auxcell is None:
        auxcell = mydf.auxcell
    if cp_ij_id is None:
        raise ValueError("cp_ij_id must be specified!")
    if omega is None:
        omega = mydf.omega

    l_ctr_offsets = vhfopt.l_ctr_offsets
    cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
    cpj = cp_ij_id - cpi * (cpi + 1) // 2
    ish0 = l_ctr_offsets[cpi]
    ish1 = l_ctr_offsets[cpi + 1]
    jsh0 = l_ctr_offsets[cpj]
    jsh1 = l_ctr_offsets[cpj + 1]
    ao_loc = vhfopt.ao_loc
    if not exxdiv:
        if cell.dimension == 3:
            ni = ao_loc[ish1] - ao_loc[ish0]
            nj = ao_loc[jsh1] - ao_loc[jsh0]
            nij = ni * nj

            assert aosym[:2] == 's1'
            nao_pair = nij

            # check if aosym is consistent with j3c shape
            assert j3c_order == 'ijL'
            nao_pair_j3c = j3c.shape[-2]
            assert (nao_pair_j3c == nao_pair)

            if vbar is None:
                g0 = numpy.pi / omega**2. / cell.vol
                qaux = get_aux_chg(mydf.int3c.auxvhfopt.mol,
                                   shls_slice=(0, auxcell.nbas))
                vbar = qaux * g0
                vbar = cupy.asarray(vbar)
            verbose_loop = mydf.verbose - 2  # print only if verbose>=8
            count = 0
            for q, adapted_kptjs, adapted_ji_idx in loop_uniq_q(
                    mydf, kptij_lst=kptij_lst, verbose=verbose_loop):
                if not is_zero(q):
                    continue
                if adapted_ji_idxs is None:
                    adapted_ji_idx = cupy.asarray(adapted_ji_idx)
                else:
                    adapted_ji_idx = adapted_ji_idxs[count]

                # TODO: calculate ovlp for shls_slice only
                # no need. FWIW this function seems to NEVER take >1% of the
                # time

                if ovlps is None:
                    ovlp = vhfopt.mol.pbc_intor(
                        'int1e_ovlp', hermi=1, kpts=adapted_kptjs)
                else:
                    ovlp = ovlps[count]
                count += 1

                ovlp = [s[ao_loc[ish0]:ao_loc[ish1],
                          ao_loc[jsh0]:ao_loc[jsh1]].reshape(-1)
                        for s in ovlp]

                ovlp_cupy = cupy.asarray(ovlp)
                if j3c.dtype == cupy.float64:
                    j3c_g0_real_kernel(
                        (int((adapted_ji_idx.size * nij)),), (256,),
                        (j3c, vbar, ovlp_cupy, adapted_ji_idx,
                         adapted_ji_idx.size, nij, vbar.size))
                else:
                    j3c_g0_kernel(
                        (int((adapted_ji_idx.size * nij)),), (256,),
                        (j3c, vbar, ovlp_cupy, adapted_ji_idx,
                         adapted_ji_idx.size, nij, vbar.size))
                ovlp_cupy = None
                adapted_ji_idx = None
                count += 1
                ###############################################################
            vbar = None
    else:
        raise NotImplementedError
    return j3c


def add_j3c_lr_(
        mydf, j3c, cp_ij_id, vhfopt, j3c_pre_data, kptij_lst=None, cell=None,
        auxcell=None, omega=None, mesh=None, aosym='s2ij', j3c_order=J3C_ORDER,
        comp=None, bvk_kmesh=None, verbose=None):
    r''' Add the LR part of j3c to input j3c.
    '''
    if cell is None:
        cell = mydf.cell
    if auxcell is None:
        auxcell = mydf.auxcell
    if omega is None:
        omega = mydf.omega
    if mesh is None:
        mesh = mydf.mesh_compact

    if aosym[:2] == 's2' and is_j_only(kptij_lst):
        aosym = 's2'
    else:
        aosym = 's1'

    if kptij_lst is None:
        kptij_lst = numpy.zeros((1, 2, 3))

    verbose_loop = verbose - 2
    kq = 0
    kgL_T_func = j3c_pre_data[0]
    for kpt, adapted_kptjs, adapted_ji_idx in loop_uniq_q(
            mydf, kptij_lst=kptij_lst, verbose=verbose_loop):
        add_j3c_lr_q_(
            mydf, j3c, cp_ij_id, vhfopt, kpt, adapted_kptjs, adapted_ji_idx,
            kgL_T_func=kgL_T_func, cell=cell, auxcell=auxcell, omega=omega,
            mesh=mesh, aosym=aosym, j3c_order=j3c_order, comp=comp,
            bvk_kmesh=bvk_kmesh, j3c_pre_data=j3c_pre_data[1:],
            verbose=verbose, kq=kq)
        kq += 1

    return j3c


def add_j3c_lr_q_(
        mydf, j3c, cp_ij_id, vhfopt, kpt, adapted_kptjs, adapted_ji_idx,
        kgL_T_func, j3c_pre_data, cell=None, auxcell=None, omega=None,
        mesh=None, aosym='s2ij', j3c_order=J3C_ORDER, comp=None,
        bvk_kmesh=None, kq=0, verbose=None):
    r''' Add LR part of j3c to input j3c
    '''
    log = logger.new_logger(mydf, verbose)

    if comp is None:
        comp = 1
    if comp != 1:
        raise NotImplementedError

    if cell is None:
        cell = mydf.cell
    if auxcell is None:
        auxcell = mydf.auxcell
    if omega is None:
        omega = mydf.omega
    if mesh is None:
        mesh = mydf.mesh_compact
    l_ctr_offsets = vhfopt.l_ctr_offsets

    cpi = int(numpy.floor((numpy.sqrt(1 + 8 * cp_ij_id) - 1) / 2))
    cpj = cp_ij_id - cpi * (cpi + 1) // 2
    ish0 = l_ctr_offsets[cpi]
    ish1 = l_ctr_offsets[cpi + 1]
    jsh0 = l_ctr_offsets[cpj]
    jsh1 = l_ctr_offsets[cpj + 1]
    shls_slice = (ish0, ish1, jsh0, jsh1, 0, auxcell.nbas)
    nkptj = len(adapted_kptjs)

    # determine nao_pair
    ao_loc = cell.ao_loc_nr()
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nii_start = ao_loc[shls_slice[0]] * (ao_loc[shls_slice[0]] + 1) // 2
    nii_end = ao_loc[shls_slice[1]] * (ao_loc[shls_slice[1]] + 1) // 2
    nii = nii_end - nii_start
    nij = ni * nj

    if is_zero(kpt):
        aosym_ = aosym[:2]
    else:
        aosym_ = 's1'

    ncol = nii if aosym_ == 's2' else nij

    # check j3c dtype and shape
    if j3c.dtype == numpy.double and not (
            is_zero(kpt) and is_zero(adapted_kptjs)):
        log.error(
            'input j3c is real but input kpt/adapted_kptjs are not '
            'gamma point')
        raise ValueError

    naoaux_cart = mydf.int3c.auxvhfopt.mol.nao_nr(cart=True)
    j3c_shape = (naoaux_cart, ncol) if j3c_order == 'Lij' else (
        ncol, naoaux_cart)
    if j3c.shape[-2:] != j3c_shape:
        log.error(
            'Input j3c has a wrong shape. Expecting j3c.shape[-2:]= %s, '
            'getting %s.', j3c_shape, j3c.shape[-2:])
        raise ValueError

    # useful constants
    naoaux_cart = mydf.int3c.auxvhfopt.mol.nao_nr(cart=True)

    mesh = mydf.mesh_compact
    ovlp_mask, b, Gv, Gvbase, gxyz, Ls = j3c_pre_data

    ngrids = gxyz.shape[0]
    # kLpqbuf = cupy.zeros((nkptj,*j3c_shape), dtype=j3c[0].dtype)
    # (pq|G;{kj}) + (pq|G) ==> ncol*(nkptj+1)*Gblksize
    mem_avail = gpu_avail_bytes()
    Gblksize = max(
        16, int(numpy.floor(mem_avail / (ncol * nkptj + naoaux_cart) / 16)))
    Gblksize = min(Gblksize, ngrids)
    log.debug1("Slicing %d into %d with %.2fGB for LR cal" %
               (ngrids, Gblksize, mem_avail / (1024**3)))

    dat_cupy = None
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        # shape: nkptj, nG, ncol
        # add bvk_kmesh may cause issue:
        # https://github.com/pyscf/pyscf/issues/1335
        dat_cupy = gint_ft_ao.ft_aopair_kpts(
            cell, Gv[p0:p1], shls_slice, aosym_, b, gxyz[p0:p1], Gvbase, kpt,
            adapted_kptjs, bvk_kmesh=bvk_kmesh, ovlp_mask=ovlp_mask, kq=kq,
            out=dat_cupy, Ls=Ls)
        nG = p1 - p0

        gL_T = kgL_T_func(kq, p0, p1)
        for k, ji in enumerate(adapted_ji_idx):
            pqg = dat_cupy[k].reshape(ncol, nG)
            if j3c_order == 'Lij':
                raise NotImplementedError
            else:
                _j3c = j3c[ji][0]
                if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                    contraction('ig', pqg, 'ag',
                                gL_T, 'ia', _j3c, beta=1, opb='CONJ')
                else:
                    pqgreal = cupy.ndarray(
                        (ncol, nG, 2), dtype='f8', memptr=pqg.data)
                    gL_T_real = cupy.ndarray(
                        (naoaux_cart, nG, 2), dtype='f8', memptr=gL_T.data)
                    if not cupy.isrealobj(j3c):
                        _j3c = cupy.ndarray(
                            (ncol, naoaux_cart, 2), dtype='f8',
                            memptr=_j3c.data)[:, :, 0]
                    blksize = min(nG, (2**31) // max(naoaux_cart, ncol) - 1)
                    for i in range(0, nG, blksize):
                        s = slice(i, i + blksize)
                        contraction(
                            'ig', pqgreal[:, s, 0], 'ag', gL_T_real[:, s, 0],
                            'ia', _j3c, beta=1)
                        contraction(
                            'ig', pqgreal[:, s, 1], 'ag', gL_T_real[:, s, 1],
                            'ia', _j3c, beta=1)
                    _j3c = None
            pqg = pqgreal = None
        gL_T = gL_T_real = None
    return j3c


def get_kptij_lst(kpts, kpts_band=None, j_only=False, ksym='s2'):
    uniq_idx = unique(kpts)[1]
    kpts = numpy.asarray(kpts)[uniq_idx]
    if kpts_band is None:
        kband_uniq = numpy.zeros((0, 3))
    else:
        kband_uniq = [k for k in kpts_band if len(member(k, kpts)) == 0]
    if j_only:
        kall = numpy.vstack([kpts, kband_uniq])
        kptij_lst = numpy.hstack((kall, kall)).reshape(-1, 2, 3)
    else:
        if ksym == 's2':
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts)
                         for j in range(i + 1)]
        else:
            kptij_lst = [(ki, kj) for ki in kpts for kj in kpts]
        kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
        kptij_lst.extend([(ki, ki) for ki in kband_uniq])
        kptij_lst = numpy.asarray(kptij_lst)
    return kptij_lst


def is_j_only(kptij_lst):
    kpti = kptij_lst[:, 0]
    kptj = kptij_lst[:, 1]
    aosym_ks2 = abs(kpti - kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    return j_only


def loop_uniq_q(mydf, kptij_lst=None, verbose=None):
    r''' Loop over uniq q = kptj-kpti, yielding q, adapted_kptjs,
                            adapted_ji_idx
    '''
    log = logger.new_logger(mydf, verbose)

    if kptij_lst is None:
        # kpts = mydf.kpts
        # kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts)
        #              for j in range(i+1)]
        # kptij_lst = numpy.asarray(kptij_lst)
        kptij_lst = get_kptij_lst(mydf.kpts)
    else:
        kptij_lst = numpy.asarray(kptij_lst).reshape(-1, 2, 3)
    kptis = kptij_lst[:, 0]
    kptjs = kptij_lst[:, 1]
    uniq_kpts, uniq_index, uniq_inverse = unique(kptjs - kptis)

    ared = mydf.cell.lattice_vectors() / (2 * numpy.pi)

    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = numpy.einsum('wx,ix->wi', ared, uniq_kpts + kpt)
        kdif_int = numpy.rint(kdif)
        mask = numpy.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = numpy.where(mask)[0]
        return uniq_kptji_ids

    done = numpy.zeros(len(uniq_kpts), dtype=bool)
    for k, kpt in enumerate(uniq_kpts):
        if done[k]:
            continue

        uniq_kptji_ids = kconserve_indices(-kpt)
        log.debug1("Symmetry pattern (k - %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                q = uniq_kpts[uniq_kptji_id]
                adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
                adapted_kptjs = kptjs[adapted_ji_idx]
                log.debug1('adapted_ji_idx = %s', adapted_ji_idx)
                yield q, adapted_kptjs, adapted_ji_idx
        done[uniq_kptji_ids] = True

        uniq_kptji_ids = kconserve_indices(kpt)
        log.debug1("Symmetry pattern (k + %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                q = uniq_kpts[uniq_kptji_id]
                adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
                adapted_kptjs = kptjs[adapted_ji_idx]
                log.debug1('adapted_ji_idx = %s', adapted_ji_idx)
                yield q, adapted_kptjs, adapted_ji_idx
        done[uniq_kptji_ids] = True


def get_tril_indices(cp_ij_id):
    i = int(-0.5 + numpy.sqrt(8 * cp_ij_id + 1) / 2)
    return i, cp_ij_id - i * (i + 1) // 2
