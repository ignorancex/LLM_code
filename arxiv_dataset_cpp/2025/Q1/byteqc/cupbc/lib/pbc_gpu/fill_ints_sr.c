/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf,
https://github.com/hongzhouye/pyscf/tree/rsdf_direct),
which is licensed under the Apache License 2.0. The original copyright:
    Copyright 2014-2020 The PySCF Developers. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Author: Qiming Sun <osirpt.sun@gmail.com>
    Author: Hong-Zhou Ye <hzyechem@gmail.com>

*/

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "config.h"
#include "fill_ints_sr.h"

#define OF_CMPLX 2

#define GINT_NR3C_BVK_FUNC(TYPE)                                               \
    void GINT_nr3c_bvk_##TYPE(double complex *out, void *gpu_buf,              \
        unsigned long buf_size, int nkpts_ij, int nkpts, int comp, int nimgs,  \
        int bvk_nimgs, int *cell_loc_bvk, double *Ls, int nLs,                 \
        double complex *d_expkL, int *kptij_idx, int *shls_slice, int *ao_loc, \
        int *auxao_loc, int *refuniqshl_map, int *auxuniqshl_map,              \
        int uniqmaxshell, int nbasauxuniq, double *uniqexp,                    \
        double *uniq_dcut2s, double dcut_binsize, double *uniq_Rcut2s,         \
        int *uniqshlpr_dij_loc, int *atm, int natm, int *bas, int nbas,        \
        double *env, int nenv, int cp_ij_id, int aux_id_range,                 \
        int *bas_pair2shls, int *bas_pairs_locs, int ncptype,                  \
        BasisProdCache *auxbpcache, double diag_fac, int prescreen_mask) {     \
        const int ish0 = shls_slice[0];                                        \
        const int ish1 = shls_slice[1];                                        \
        const int jsh0 = shls_slice[2];                                        \
        const int jsh1 = shls_slice[3];                                        \
        const int ksh0 = shls_slice[4];                                        \
        const int ksh1 = shls_slice[5];                                        \
                                                                               \
        const int di = ao_loc[ish1] - ao_loc[ish0];                            \
        const int dj = ao_loc[jsh1] - ao_loc[jsh0];                            \
        const size_t dij = di * dj;                                            \
        size_t dm = auxao_loc[ksh1] - auxao_loc[ksh0];                         \
                                                                               \
        const double omega = fabs(env[PTR_RANGE_OMEGA]);                       \
        size_t strides[3];                                                     \
        int ao_offsets[3];                                                     \
        strides[0] = 1;                                                        \
        strides[1] = dj * dm;                                                  \
        strides[2] = dm;                                                       \
                                                                               \
        ao_offsets[0] = 0;                                                     \
        ao_offsets[1] = ao_loc[ish0];                                          \
        ao_offsets[2] = ao_loc[jsh0];                                          \
                                                                               \
        GPBC_nr_fill_ao_ints_bvk_##TYPE(out, bvk_nimgs, dij, dm, nkpts,        \
            d_expkL, cell_loc_bvk, cp_ij_id, aux_id_range, gpu_buf, buf_size,  \
            nkpts_ij, comp, kptij_idx, shls_slice, ao_loc, auxao_loc, atm,     \
            natm, bas, nbas, env, nenv, bas_pair2shls, bas_pairs_locs,         \
            ncptype, auxbpcache, diag_fac, Ls, nLs, strides, ao_offsets,       \
            omega, refuniqshl_map, auxuniqshl_map, uniqmaxshell, nbasauxuniq,  \
            uniqexp, uniq_dcut2s, dcut_binsize, uniq_Rcut2s,                   \
            uniqshlpr_dij_loc, prescreen_mask);                                \
    }
#define PBCSR3C_BVK_FUNC(S1TYPE, TYPE)                                         \
    void GINTPBCsr3c_bvk_##S1TYPE(double complex *out, void *gpu_buf,          \
        unsigned long buf_size, int nkpts_ij, int nkpts, int comp, int nimgs,  \
        int bvk_nimgs, int *cell_loc_bvk, double *Ls, int nLs,                 \
        double complex *d_expkL, int *kptij_idx, int *shls_slice, int *ao_loc, \
        int *auxao_loc, int *refuniqshl_map, int *auxuniqshl_map,              \
        int uniqmaxshell, int nbasauxuniq, double *uniqexp,                    \
        double *uniq_dcut2s, double dcut_binsize, double *uniq_Rcut2s,         \
        int *uniqshlpr_dij_loc, int *atm, int natm, int *bas, int nbas,        \
        double *env, int nenv, int cp_ij_id, int aux_id_range,                 \
        int *bas_pair2shls, int *bas_pairs_locs, int ncptype,                  \
        BasisProdCache *auxbpcache, double diag_fac, int prescreen_mask) {     \
        GINT_nr3c_bvk_##TYPE(out, gpu_buf, buf_size, nkpts_ij, nkpts, comp,    \
            nimgs, bvk_nimgs, cell_loc_bvk, Ls, nLs, d_expkL, kptij_idx,       \
            shls_slice, ao_loc, auxao_loc, refuniqshl_map, auxuniqshl_map,     \
            uniqmaxshell, nbasauxuniq, uniqexp, uniq_dcut2s, dcut_binsize,     \
            uniq_Rcut2s, uniqshlpr_dij_loc, atm, natm, bas, nbas, env, nenv,   \
            cp_ij_id, aux_id_range, bas_pair2shls, bas_pairs_locs, ncptype,    \
            auxbpcache, diag_fac, prescreen_mask);                             \
    }
#define PBCSR3C_BVK_DRV_FUNC(DRVTYPE, S1TYPE)                                  \
    void GINTPBCsr3c_bvk_##DRVTYPE(double complex *out, void *gpu_buf,         \
        unsigned long buf_size, int nkpts_ij, int nkpts, int comp, int nimgs,  \
        int bvk_nimgs, double *Ls, int nLs, double complex *d_expkL,           \
        int *kptij_idx, int *shls_slice, int *ao_loc, int *auxao_loc,          \
        int *cell_loc_bvk, int8_t *shlpr_mask, int *refuniqshl_map,            \
        int *auxuniqshl_map, int uniqmaxshell, int nbasauxuniq,                \
        double *uniqexp, double *uniq_dcut2s, double dcut_binsize,             \
        double *uniq_Rcut2s, int *uniqshlpr_dij_loc, int *atm, int natm,       \
        int *bas, int nbas, double *env, int nenv, int cp_ij_id,               \
        int aux_id_range, int *bas_pair2shls, int *bas_pairs_locs,             \
        int ncptype, BasisProdCache *auxbpcache, double diag_fac,              \
        int prescreen_mask) {                                                  \
        GINTPBCsr3c_bvk_##S1TYPE(out, gpu_buf, buf_size, nkpts_ij, nkpts,      \
            comp, nimgs, bvk_nimgs, cell_loc_bvk, Ls, nLs, d_expkL,           \
            kptij_idx, shls_slice, ao_loc, auxao_loc, refuniqshl_map,          \
            auxuniqshl_map, uniqmaxshell, nbasauxuniq, uniqexp, uniq_dcut2s,   \
            dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc, atm, natm, bas,      \
            nbas, env, nenv, cp_ij_id, aux_id_range, bas_pair2shls,            \
            bas_pairs_locs, ncptype, auxbpcache, diag_fac, prescreen_mask);    \
    }

// kk
GINT_NR3C_BVK_FUNC(kk)
PBCSR3C_BVK_FUNC(kks1, kk)
PBCSR3C_BVK_DRV_FUNC(kk_drv, kks1)

// k
GINT_NR3C_BVK_FUNC(k)
PBCSR3C_BVK_FUNC(ks1, k)
PBCSR3C_BVK_DRV_FUNC(k_drv, ks1)

// gamma
void GINTPBCsr3c_gs1(double *out, void *gpu_buf, unsigned long buf_size,
    int comp, int nimgs, double *Ls, int nLs, int *shls_slice, int *ao_loc,
    int *auxao_loc, int *refuniqshl_map, int *auxuniqshl_map, int uniqmaxshell,
    int nbasauxuniq, double *uniqexp, double *uniq_dcut2s, double dcut_binsize,
    double *uniq_Rcut2s, int *uniqshlpr_dij_loc, int *atm, int natm, int *bas,
    int nbas, double *env, int nenv, int cp_ij_id, int aux_id_range,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
    BasisProdCache *auxbpcache, double diag_fac, int prescreen_mask) {
    GINT_nr3c_g(out, gpu_buf, buf_size, comp, nimgs, Ls, nLs, shls_slice,
        ao_loc, auxao_loc, refuniqshl_map, auxuniqshl_map, uniqmaxshell,
        nbasauxuniq, uniqexp, uniq_dcut2s, dcut_binsize, uniq_Rcut2s,
        uniqshlpr_dij_loc, atm, natm, bas, nbas, env, nenv, cp_ij_id,
        aux_id_range, bas_pair2shls, bas_pairs_locs, ncptype, auxbpcache,
        diag_fac, prescreen_mask);
}
void GINTPBCsr3c_g_drv(double *out, void *gpu_buf, unsigned long buf_size,
    int comp, int nimgs, double *Ls, int nLs, int *shls_slice, int *ao_loc,
    int *auxao_loc, int8_t *shlpr_mask, int *refuniqshl_map,
    int *auxuniqshl_map, int uniqmaxshell, int nbasauxuniq, double *uniqexp,
    double *uniq_dcut2s, double dcut_binsize, double *uniq_Rcut2s,
    int *uniqshlpr_dij_loc, int *atm, int natm, int *bas, int nbas, double *env,
    int nenv, int cp_ij_id, int aux_id_range, int *bas_pair2shls,
    int *bas_pairs_locs, int ncptype, BasisProdCache *auxbpcache,
    double diag_fac, int prescreen_mask) {
    GINTPBCsr3c_gs1(out, gpu_buf, buf_size, comp, nimgs, Ls, nLs, shls_slice,
        ao_loc, auxao_loc, refuniqshl_map, auxuniqshl_map, uniqmaxshell,
        nbasauxuniq, uniqexp, uniq_dcut2s, dcut_binsize, uniq_Rcut2s,
        uniqshlpr_dij_loc, atm, natm, bas, nbas, env, nenv, cp_ij_id,
        aux_id_range, bas_pair2shls, bas_pairs_locs, ncptype, auxbpcache,
        diag_fac, prescreen_mask);
}
