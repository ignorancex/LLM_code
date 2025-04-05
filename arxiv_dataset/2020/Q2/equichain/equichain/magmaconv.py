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

from sage.all import *
import itertools

import magmaconv_cython

from magmaconv_cython import scipy_sparse_matrix_from_magma, numpy_matrix_from_magma,\
        numpy_vector_from_magma

from sage.rings.finite_rings.integer_mod_ring import IntegerModRing_generic

def convert_sage_ring_to_magma(ring):
    if ring == ZZ:
        return magma.IntegerRing()
    elif isinstance(ring, IntegerModRing_generic):
        return magma.IntegerRing(ring.order())
    else:
        raise NotImplementedError

class AutoIncrList(object):
    def __init__(self, totallen):
        self.data = [None]*totallen
        self.totallen = totallen
        self.i = 0

    def append(self,val):
        self.data[self.i] = val
        self.i += 1

    def finished():
        assert self.i == self.totallen

def magma_sparse_matrix_from_scipy(A, ring):
    ring = convert_sage_ring_to_magma(ring)

    A = A.tocsr()

    magmadata = AutoIncrList(2*A.nnz + A.shape[0])
    for i in xrange(A.shape[0]):
        magmadata.append(A.indptr[i+1]-A.indptr[i])
        for indx,val in itertools.izip(
                A.indices[A.indptr[i]:A.indptr[i+1]]+1,
                A.data[A.indptr[i]:A.indptr[i+1]]):
            magmadata.append(indx)
            magmadata.append(val)

    s = str(magmadata.data)
    return magma.SparseMatrix(ring, A.shape[0], A.shape[1], magma(s))

def magma_dense_matrix_from_numpy(A, ring):
    ring = convert_sage_ring_to_magma(ring)
    return magmaconv_cython.numpy_int_matrix_to_magma(A,ring)

def magma_vector_from_numpy(A, ring):
    ring = convert_sage_ring_to_magma(ring)
    return magmaconv_cython.numpy_int_vector_to_magma(A,ring)
