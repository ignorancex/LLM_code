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

import numpy
import ctypes
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def take_pointer_data_from_array(array):
    assert array.flags.c_contiguous
    assert array.itemsize == 8
    return array.ctypes.data, array.size, array.shape


def get_array_from_pointer_data(pointer_data):
    array_ctypes_data, array_size, array_shape = pointer_data
    pointer = ctypes.cast(
        array_ctypes_data,
        ctypes.POINTER(ctypes.c_double * array_size))
    array = numpy.ndarray(
        array_shape,
        dtype='float64',
        buffer=pointer.contents)
    return array


def fix_orbital_sign(mo_coeff):
    absmax = numpy.argmax(abs(mo_coeff), axis=0)
    nmo = mo_coeff.shape[-1]
    swap = mo_coeff[absmax, numpy.arange(nmo)] < 0
    mo_coeff[:, swap] *= -1
    return mo_coeff
