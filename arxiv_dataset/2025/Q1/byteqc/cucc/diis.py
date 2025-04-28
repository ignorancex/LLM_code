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

from pyscf.lib import diis, prange, logger
from pyscf import __config__
import cupy
import scipy
from byteqc import lib

MAXSIZE = 5000000000


class DIIS(diis.DIIS):
    __doc__ = diis.DIIS.__doc__

    def __init__(self, dev=None, filename=None, incore=getattr(
            __config__, 'lib_diis_DIIS_incore', False), pool=None):
        diis.DIIS.__init__(self, dev, filename, incore)
        self.pool = pool
        self.dict = dict()

    def _store(self, key, value):
        self.dict[key] = value.asnumpy()

    def push_vec(self, x):
        while len(self._bookkeep) >= self.space:
            self._bookkeep.pop(0)

        if self._err_vec_touched:
            self._bookkeep.append(self._head)
            key = 'x%d' % (self._head)
            self._store(key, x)
            self._head += 1

        elif self._xprev is None:
            self._store('xprev', x)
            self._xprev = self.dict['xprev']

        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            ekey = 'e%d' % self._head
            xkey = 'x%d' % self._head
            self._store(xkey, x)
            if ekey in self.dict:
                err = self.dict[ekey]
            else:
                err = self.pool.new(ekey, x.shape, x.dtype, pin=1)
                self.dict[ekey] = err
            for p0, p1 in prange(0, err.size, MAXSIZE):
                err[p0:p1] = x[p0:p1]
                lib.axpy(self._xprev[p0:p1], err[p0:p1], a=-1.0)
            self._head += 1

    def get_err_vec(self, idx):
        return self.dict['e%d' % idx]

    def get_vec(self, idx):
        return self.dict['x%d' % idx]

    def update(self, x, xerr=None):
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        dt = self.get_err_vec(self._head - 1)
        if self._H is None:
            self._H = cupy.zeros((self.space + 1, self.space + 1), dt.dtype)
            self._H[0, 1:] = self._H[1:, 0] = 1
        for i in range(nd):
            tmp = 0
            dti = self.get_err_vec(i)
            for p0, p1 in prange(0, dt.size, MAXSIZE):
                tmp += dt[p0:p1].dot(dti[p0:p1]).item()
            self._H[self._head, i + 1] = tmp
            self._H[i + 1, self._head] = tmp.conjugate()
        dt = None

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev[:] = 0
            xnew = self.extrapolate(nd)
        return xnew

    def extrapolate(self, nd=None):
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError('No vector found in DIIS object.')

        h = self._H[:nd + 1, :nd + 1]
        g = cupy.zeros(nd + 1, h.dtype)
        g[0] = 1

        try:
            w, v = cupy.linalg.eigh(h)
        except BaseException:
            w, v = scipy.linalg.eigh(h.get())
            w = cupy.asarray(w)
            v = cupy.asarray(v)
        if cupy.any(abs(w) < 1e-14):
            logger.debug(
                self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w) > 1e-14
            c = cupy.dot(v[:, idx] * (1. / w[idx]),
                         cupy.dot(v[:, idx].T.conj(), g))
        else:
            try:
                c = cupy.linalg.solve(h, g)
            except cupy.linalg.linalg.LinAlgError as e:
                logger.warn(self, ' diis singular, eigh(h) %s', w)
                raise e
        logger.debug1(self, 'diis-c %s', c)

        xnew = self._xprev
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            for p0, p1 in prange(0, xi.size, MAXSIZE):
                lib.axpy(xi[p0:p1], xnew[p0:p1], a=ci)
        return xnew
