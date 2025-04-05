# Copyright (c) Dominic Else 2019
#
# This file is part of equichain.
# equichain is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# equichain is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with equichain.  If not, see <https://www.gnu.org/licenses/>.

import copy
import numpy
from scipy import sparse
from functools import reduce
import operator

def product(factors, starting=1):
    return reduce(operator.mul, factors, starting)

class IndexedSet(object):
    def __init__(self):
        self.el_to_index = dict()
        self.index_to_el = list()

    def subset_by_indices(self, indices):
        l = IndexedSet()
        for i in indices:
            l.append(self.index_to_el[i])

    def append(self, o):
        if o not in self.el_to_index:
            self.index_to_el.append(o)
            index = len(self.index_to_el)-1
            self.el_to_index[o] = index

    def index(self, o):
        return self.el_to_index[o]

    def __getitem__(self, i):
        return self.index_to_el[i]

    def __len__(self):
        return len(self.index_to_el)

    def __iter__(self):
        return iter(self.index_to_el)


class FormalIntegerSum(object):
    def __init__(self,coeffs={}):
        if not isinstance(coeffs,dict):
            self.coeffs = { coeffs : 1 }
        else:
            self.coeffs = copy.copy(coeffs)

    def __eq__(a,b):
        return a.coeffs == b.coeffs

    def __ne__(a,b):
        return not a.__eq__(b)

    def __add__(a,b):
        ret = FormalIntegerSum(a.coeffs)
        for o,coeff in b.coeffs.iteritems():
            if o in ret.coeffs:
                ret.coeffs[o] += coeff
            else:
                ret.coeffs[o] = coeff
        return ret

    def __iter__(self):
        return self.coeffs.iteritems()

    def itervectors(self):
        return self.coeffs.iterkeys()

    def act_with(self,action):
        ret = FormalIntegerSum({})
        for o,coeff in self.coeffs.iteritems():
            ret[o.act_with(action)] = coeff
        return ret

    def __str__(self):
        if len(self.coeffs) == 0:
            return "0"
        else:
            s = ""
            items = self.coeffs.items()
            for i in xrange(len(items)):
                s += str(items[i][1]) + "*" + str(items[i][0])
                if i < len(items)-1:
                    s += " + "
            return s

    def __repr__(self):
        return str(self)

class MultiIndexer(object):
    def __init__(self, *dims):
        self.dims = tuple(dims)

    @staticmethod
    def tensor(dim, ntimes):
        return MultiIndexer(*( (dim,)*ntimes ))

    def to_index(self, *indices):
        index = 0
        stride = 1
        for i in xrange(len(indices)):
            index += stride*indices[i]
            stride *= self.dims[i]

        return index

    def from_index(self,I):
        assert I >= 0 and I < self.total_dim()
        I0 = I
        ret = numpy.zeros(len(self.dims), dtype=int)
        for i in xrange(len(self.dims)):
            ret[i] = I % self.dims[i]
            I = (I - ret[i])//self.dims[i]
        assert self.to_index(*ret) == I0
        return tuple(ret)

    def __call__(self, *indices):
        return self.to_index(*indices)

    def total_dim(self):
        return numpy.prod(self.dims)

class COOMatrixHelperItem(object):
    def __init__(self, parent, i, j):
        self.parent = parent
        self.i = i
        self.j = j

    def __iadd__(self, x):
        self.parent.iadd_at(self.i,self.j,x)
        return self

class COOMatrixHelper(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

        self.data = []
        self.i = []
        self.j = []

    def iadd_at(self,i,j,x):
        self.data.append(x)
        self.i.append(i)
        self.j.append(j)

    def __getitem__(self, at):
        return COOMatrixHelperItem(self, at[0], at[1])

    def __setitem__(self,at,val):
        if not isinstance(val,COOMatrixHelperItem):
            raise NotImplementedError
        if (val.i,val.j) != at:
            raise NotImplementedError
        if val.parent is not self:
            raise NotImplementedError

    def coomatrix(self):
        return sparse.coo_matrix( (self.data, (self.i,self.j)), shape=self.shape, dtype=self.dtype )

class MatrixIndexingWrapper(object):
    def __init__(self, A, out_indexer, in_indexer):
        assert len(A.shape) == 2
        assert A.shape[0] == out_indexer.total_dim()
        assert A.shape[1] == in_indexer.total_dim()

        self.A = A
        self.out_indexer = out_indexer
        self.in_indexer = in_indexer

    @staticmethod
    def from_factory(factory, dtype, out_indexer, in_indexer):
        A = factory( (out_indexer.total_dim(), in_indexer.total_dim()), dtype=dtype)
        return MatrixIndexingWrapper(A, out_indexer, in_indexer)

    def _convert_index(self, i):
        return (self.out_indexer.to_index(*i[0]), self.in_indexer.to_index(*i[1]))

    def __getitem__(self,i):
        return self.A[self._convert_index(i)]

    def __setitem__(self, i, value):
        self.A[self._convert_index(i)] = value

    def raw_access(self):
        return self.A

    def set_raw(self,A):
        self.A = A
