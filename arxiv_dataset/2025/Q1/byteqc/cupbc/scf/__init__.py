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

from byteqc.cupbc.scf import khf as krhf
from byteqc.cupbc.scf import khf_ksymm
from pyscf.pbc.lib import kpts as libkpts


def KRHF(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            # raise NotImplementedError('rsjk with k-points symmetry')
            return khf_ksymm.KRHF(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            # raise NotImplementedError('rsjk with k-points symmetry')
            return khf_ksymm.KRHF(cell, *args, **kwargs)
    return krhf.KRHF(cell, *args, **kwargs)


def KRKS(cell, *args, **kwargs):
    from byteqc.cupbc.dft import krks, krks_ksymm
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return krks_ksymm.KRKS(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return krks_ksymm.KRKS(cell, *args, **kwargs)
    return krks.KRKS(cell, *args, **kwargs)
