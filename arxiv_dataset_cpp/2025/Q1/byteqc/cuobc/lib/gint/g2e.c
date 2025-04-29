/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf)
and GPU4PySCF (https://github.com/bytedance/gpu4pyscf),
which are licensed under the Apache License 2.0.
The original copyright:
    Copyright 2014-2020 The GPU4PySCF/PySCF Developers. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "g2e.h"

void GINTinit_EnvVars(GINTEnvVars *envs, ContractionProdType *cp_ij,
    ContractionProdType *cp_kl) {
    int i_l = cp_ij->l_bra;
    int j_l = cp_ij->l_ket;
    int k_l = cp_kl->l_bra;
    int l_l = cp_kl->l_ket;
    int nfi = (i_l + 1) * (i_l + 2) / 2;
    int nfj = (j_l + 1) * (j_l + 2) / 2;
    int nfk = (k_l + 1) * (k_l + 2) / 2;
    int nfl = (l_l + 1) * (l_l + 2) / 2;
    int nroots = (i_l + j_l + k_l + l_l) / 2 + 1;
    double fac = (M_PI * M_PI * M_PI) * 2 / SQRTPI;

    envs->i_l = i_l;
    envs->j_l = j_l;
    envs->k_l = k_l;
    envs->l_l = l_l;
    envs->nfi = nfi;
    envs->nfj = nfj;
    envs->nfk = nfk;
    envs->nfl = nfl;
    envs->nf = nfi * nfj * nfk * nfl;
    envs->nrys_roots = nroots;
    envs->fac = fac;

    int ibase = i_l >= j_l;
    int kbase = k_l >= l_l;
    envs->ibase = ibase;
    envs->kbase = kbase;

    int li1 = i_l + 1;
    int lj1 = j_l + 1;
    int lk1 = k_l + 1;
    int ll1 = l_l + 1;
    int di = nroots;
    int dj = di * li1;
    int dk = dj * lj1;
    int dl = dk * lk1;
    envs->g_size_ij = dk;
    envs->g_size = dl * ll1;

    if (ibase) {
        envs->ijmin = j_l;
        envs->ijmax = i_l;
        envs->stride_ijmax = nroots;
        envs->stride_ijmin = nroots * li1;
    } else {
        envs->ijmin = i_l;
        envs->ijmax = j_l;
        envs->stride_ijmax = nroots;
        envs->stride_ijmin = nroots * lj1;
    }

    if (kbase) {
        envs->klmin = l_l;
        envs->klmax = k_l;
        envs->stride_klmax = dk;
        envs->stride_klmin = dk * lk1;
    } else {
        envs->klmin = k_l;
        envs->klmax = l_l;
        envs->stride_klmax = dk;
        envs->stride_klmin = dk * ll1;
    }

    envs->nprim_ij = cp_ij->nprim_12;
    envs->nprim_kl = cp_kl->nprim_12;
}
