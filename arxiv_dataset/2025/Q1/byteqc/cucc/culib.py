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

import cupy
from pyscf.lib import HERMITIAN, prange
import cupy_backends.cuda.libs.cutensor as cutensorlib
from byteqc.lib import gpu_avail_bytes
from byteqc.lib import contraction as _contraction
from byteqc.lib import gemm as _gemm


def current_memory(device=None):
    r = cupy.cuda.Device(device).mem_info
    return (r[1] - r[0]) / 1e6, r[1] / 1e6


def damp(alpha, a, b):
    '''In-place version of a = a*alpha + (1-alpha)*b'''
    memory = gpu_avail_bytes()
    itemsize = a.dtype.itemsize
    size = a.size
    ld = a.shape[0]
    blksize = min(ld, int(memory / itemsize / (size / ld)))
    ker = cupy.ElementwiseKernel('T alpha, T b', 'T a', '''
                   a = a*alpha + (1-alpha)*b
                ''', 'damp')
    for p0, p1 in prange(0, ld, blksize):
        with a[p0:p1] as arr:
            ker(alpha, b[p0:p1].ascupy(), arr)


def hermi_sum(a, axes=None, hermi=HERMITIAN, inplace=False, out=None):
    '''GPU version of pyscf.lib.numpy_helper.hermi_sum'''
    assert inplace is False, 'Not implement'
    if out is None:
        out = cupy.empty(a.shape, a.dtype)
    else:
        out = cupy.ndarray(a.shape, a.dtype, memptr=out.data)

    if a.ndim == 2:
        assert (a.shape[0] == a.shape[1])
        s = a.shape[0]
    elif a.ndim == 3 and axes == (0, 2, 1):
        assert (a.shape[1] == a.shape[2])
        s = a.shape[1]
    else:
        raise NotImplementedError

    ker = cupy.ElementwiseKernel(
        'int64 s, raw T t', 'T out', '''
            int c  = i % s;
            int a = i / s;
            int b = a % s;
            a /= s;
            out = t[a*s*s+b*s+c]+t[a*s*s+c*s+b]
        ''', 'hermi_sum')
    ker(s, a, out)
    return out


def toarray(x):
    '''
    Convert the BufArr to cupy/numpy array.
    '''
    if getattr(x, 'dev', 0) == 2:
        return x.arr[:]
    else:
        return x


def contraction(
        inda, a, indb, b, indc, c=None, alpha=True, beta=False, opa='IDENTITY',
        opb='IDENTITY', opc='IDENTITY', buf=None,
        alg=cutensorlib.ALGO_DEFAULT, ws_pref=cutensorlib.WORKSPACE_MIN,
        issync=False):
    '''A warpper of ByteQC.lib.contraction with extra support of BufArr.'''
    a = toarray(a)
    b = toarray(b)
    if getattr(c, 'dev', 0) == 2:
        carr = c.arr[:]
    else:
        carr = c
    carr = _contraction(
        inda, a, indb, b, indc, carr, alpha, beta, opa, opb, opc,
        buf=buf, alg=alg, ws_pref=ws_pref, issync=issync)
    if getattr(c, 'dev', 0) == 2:
        c.arrback(carr)
    return carr if c is None else c


def gemm(transa, transb, a, b, c=None, alpha=1.0, beta=0.0, buf=None):
    '''A warpper of ByteQC.lib.gemm with extra support of BufArr.'''
    a = toarray(a)
    b = toarray(b)
    if getattr(c, 'dev', 0) == 2:
        carr = c.arr[:]
    else:
        carr = c
    carr = _gemm(a, b, carr, alpha, beta, buf, transa, transb)
    if getattr(c, 'dev', 0) == 2:
        c.arrback(carr)
    return carr if c is None else c


def diffabs2_t1(x, y):
    '''Return (x-y)**2 without extra memory allocation.'''
    ker = cupy.ReductionKernel(
        'T x, T y',  # input params
        'T out',  # output params
        '(x - y) * (x - y)',  # map
        'a + b',  # reduce
        'out = a',  # post-reduction map
        '0.0',  # identity value
        'diffabs2_t1'  # kernel name
    )
    out = cupy.asarray(0.0)
    ker(x.ascupy(), y.ascupy(), out)
    return out.item()


# (x-y)**2 for only triangular part of t2
diffabs2_t2 = cupy.ReductionKernel(
    'int64 off, int64 n1, int64 n2, int64 n3, T x, T y',  # input params
    'T out',  # output params
    '(_j+off) / n1 > (_j+off) % n1 / n2  || '
    '((_j+off) / n1 == (_j+off) % n1 / n2 '
    '&& (_j+off) % n2 / n3 >= (_j+off) % n3) ? '
    '(x - y) * (x - y) : 0.0',  # map
    'a + b',  # reduce
    'out += a',  # post-reduction map
    '0.0',  # identity value
    'diffabs2_t2'  # kernel name
)
