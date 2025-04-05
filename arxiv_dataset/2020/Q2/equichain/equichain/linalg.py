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

from __future__ import division
from sage.all import *
import numpy
from equichain import magmaconv
from scipy import sparse
import functools

try:
    magma('1')
    use_magma = True
except TypeError:
    warnings.warn("Could not load Magma. Falling back to Sage for all functionality.")
    use_magma = False

# Check if patched sage
A = matrix(ZZ, 1,1)
if hasattr(A, 'set_unsafe_from_numpy_int_array'):
    patched_sage = True
else:
    patched_sage = False
    print "WARNING: patched Sage not found; performance will be much slower"

def coefficients_of_quotient(A,B):
    """ Let V and W be the column spaces of A and B respectively, and assume that V <= W.
    We require that the columns of B be linearly independent (not necessary for A).
    This function returns the torsion coefficients of W/V.

    If B is None, then W is the free module of dimension A.nrows().
    """

    if B is not None:
        # Given the assumptions, the columns of A can be expressed as linear combinations of the columns of B.
        # The matrix X encodes these coefficients. A = BX
        X = B.solve_right(A)
    else:
        X = A

    ret = [int(n) for n in X.elementary_divisors() if n != 1]

    # For rings of order n, "0" in the elementary divisors really means
    # a Z_n factor, not Z factor, in W/V, so we correct for this.
    ring_order = X.ring.order()
    if ring_order < Infinity:
        for i in xrange(len(ret)):
            if ret[i] == 0:
                ret[i] = int(ring_order)

    return ret

def kernel_mod_image(d1,d2, return_module_obj=False):
    # Returns the torsion coefficients of (ker d1)/(im d2), where d1 d2 = 0
    #   or, if return_module_obj=True, return (ker d1)/(im d2) as a Sage module object.

    if return_module_obj:
        if d1 is not None:
            K = d1.right_kernel_sage()
            return K.quotient(d2.column_space_sage())
        else:
            d2 = d2.to_sagedense().A
            return (d2.base_ring()**d2.nrows()).quotient(d2.column_module())
    else:
        if d1 is not None:
            kernel = d1.right_kernel_matrix()
        else:
            kernel = None
        return coefficients_of_quotient(d2,kernel)


def isiterable_or_slice(i):
    try:
        iter(i)
        return True
    except TypeError:
        return isinstance(i,slice)

if use_magma:
    right_kernel_use = 'magma'
else:
    right_kernel_use = 'sage'

class GenericMatrix(object):
    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.A)
    
    def __repr__(self):
        return repr(self.A)

    def to_scipy(self):
        if self.density == 'dense':
            return self.to_numpydense()
        else:
            return self.to_scipysparse()

    def dot(self, v):
        ret = self.to_scipy().dot(v)
        return self.convert_to_like_self_preserve_density(ret)

    @property
    def backend_obj(self):
        return self.A

    def to_magma(self):
        if self.density == 'dense':
            return self.to_magmadense()
        else:
            return self.to_magmasparse()

    def column_space_sage(self):
        return self.to_sagedense().column_space_sage()

    def right_kernel_sage(self):
        return self.right_kernel_matrix().column_space_sage()

    def right_kernel_matrix(self):
        if right_kernel_use == 'sage':
            conv = self.to_sagedense()
        elif right_kernel_use == 'magma':
            conv = self.to_magma()
        else:
            raise NotImplementedError
        return self.convert_to_like_self_preserve_density(conv.right_kernel_matrix_())

    def pivots(self):
        return self.to_sagedense().pivots()

    def column_space_matrix(self):
        ret = self._constructor(self.A[:,self.pivots()])
        return ret

    def elementary_divisors(self):
        return self.to_sagedense().elementary_divisors()

    def solve_right(self, v):
        #if use_magma:
        #    return v.convert_to_like_self_preserve_density(self.to_magmadense().solve_right(v))
        #else:
        try:
            ret = v.convert_to_like_self(self.to_sagedense().solve_right(v.to_sagedense()))
        except ValueError as e:
            if e.args[0] == "matrix equation has no solutions":
                raise NoSolutionError
            else:
                raise e
        return ret

    def __getitem__(self, i):
        if ( (isinstance(i, tuple) and any(isiterable_or_slice(x) for x in i)) or
                isiterable_or_slice(i) ):
            return self._constructor(self.A[i])
        else:
            return self.A[i]

    def __setitem__(self, i, x):
        if isinstance(x, GenericMatrix):
            self.A[i] = x.A
        elif isinstance(x, GenericVector):
            self.A[i] = x.v
        else:
            self.A[i] = x

    def __neg__(self):
        return self._constructor(-self.A)

class NumpyMatrixFactoryBase(object):
    def __init__(self, constructor, vector_constructor, numpy_dtype, base_module):
        self.constructor = constructor
        self.vector_constructor = vector_constructor
        self.numpy_dtype = numpy_dtype
        self.base_module = base_module

    def bmat(self, block):
        ring = None

        if not isinstance(block[0], list):
            block = [block]

        for i in xrange(len(block)):
            for j in xrange(len(block[i])):
                if block[i][j] is None:
                    continue

                if ring is None:
                    ring = block[i][j].ring
                else:
                    assert block[i][j].ring == ring

                block[i][j] = block[i][j].A
        return self.constructor(self.base_module.bmat(block))

    def eye(self, n):
        return self.constructor(self.base_module.eye(n, dtype=self.numpy_dtype))

    def zeros(self, shape):
        return self.constructor(self.base_module.zeros(shape, dtype=self.numpy_dtype))

    def zero_vector(self, n):
        return self.vector_constructor(numpy.zeros(n,
            dtype=self.numpy_dtype))

    def concatenate_vectors(self, a, b):
        return self.vector_constructor(numpy.concatenate((a.v,b.v)))

class ScipySparseMatrixFactory(NumpyMatrixFactoryBase):
    def __init__(self, constructor, vector_constructor, numpy_dtype):
        super(ScipySparseMatrixFactory,self).__init__(constructor,
                vector_constructor, numpy_dtype, sparse)

    def zeros(self, shape):
        return self.constructor(sparse.csc_matrix(shape, dtype=self.numpy_dtype))

class NumpyMatrixFactory(NumpyMatrixFactoryBase):
    def __init__(self, constructor, vector_constructor, numpy_dtype):
        super(NumpyMatrixFactory,self).__init__(constructor, vector_constructor, numpy_dtype, numpy)

class ScipyOrNumpyMatrixOverRingGeneric(GenericMatrix):
    def __init__(self):
        raise NotImplementedError

    def nrows(self):
        return self.A.shape[0]

    def ncols(self):
        return self.A.shape[1]

    @property
    def shape(self):
        return self.A.shape

    def convert_to_like_self_preserve_density(self,a):
        return a.to_scipy()

class ScipySparseMatrixOverRingGeneric(ScipyOrNumpyMatrixOverRingGeneric):
    def to_sagedense(self):
        return self.to_numpydense().to_sagedense()

    def dot(self, b):
        b = b.to_scipy()
        assert self.ring == b.ring

        if isinstance(b, GenericVector):
            return self._vector_constructor(self.A.dot(b.v))
        elif isinstance(b, GenericMatrix):
            b = b.to_scipy()
            if isinstance(b, ScipySparseMatrixOverRingGeneric):
                return self._constructor(self.A.dot(b.A))
            elif isinstance(b, NumpyMatrixOverRingGeneric):
                return self.to_numpydense().dot(b)
            else:
                assert False
        else:
            raise NotImplementedError

    def to_numpydense(self):
        return NumpyMatrixOverRing(self.A.toarray(), self.ring)

    def to_scipysparse(self):
        return self

    def to_magmasparse(self):
        return MagmaSparseMatrix(magmaconv.magma_sparse_matrix_from_scipy(self.A,
            self.ring),
            self.ring)

    def to_magmadense(self):
        return self.to_magmasparse().to_magmadense()

    def convert_to_like_self(self,a):
        return a.to_scipysparse()

    def factory(self):
        return ScipySparseMatrixFactory(numpy_dtype=self.A.dtype,
                constructor=self._constructor,
                vector_constructor=self._vector_constructor)

    def count_nonzero(self):
        return len(self.A.nonzero()[0])

    def equals(self, b):
        return self.shape == b.shape and len((self.A != b.A).nonzero()[0]) == 0

    @property
    def density(self):
        return "sparse"

class NumpyMatrixOverRingGeneric(ScipyOrNumpyMatrixOverRingGeneric):
    def __init__(self):
        raise NotImplementedError

    def factory(self):
        return NumpyMatrixFactory(numpy_dtype=self.A.dtype,
                constructor=self._constructor, vector_constructor=self._vector_constructor)

    def to_sagedense(self):
        if patched_sage:
            nrows,ncols = self.A.shape
            B = matrix(self.ring, nrows, ncols)
            B.set_unsafe_from_numpy_int_array(self.A)
            return SageDenseMatrix(B)
        else:
            return SageDenseMatrix(matrix(self.ring, self.A))

    def to_numpydense(self):
        return self

    def to_magmadense(self):
        return MagmaDenseMatrix(magmaconv.magma_dense_matrix_from_numpy(self.A,
            self.ring), self.ring)

    def to_scipysparse(self):
        return ScipySparseMatrixOverRing(sparse.csc_matrix(self.A), ring=self.ring)

    def convert_to_like_self(self,a):
        return a.to_numpydense()

    def equals(self, b):
        return numpy.array_equal(self.A, self.b.A)

    @property
    def density(self):
        return "dense"

    def dot(self, b):
        assert self.ring == b.ring

        b = b.to_numpydense()

        if isinstance(b, NumpyVectorOverRingGeneric):
            return NumpyVectorOverRing(numpy.dot(self.A, b.v), self.ring)
        else:
            return NumpyMatrixOverRing(numpy.dot(self.A, b.A), self.ring)

from sage.rings.finite_rings.integer_mod_ring import IntegerModRing_generic

class NumpyMatrixOverZ(NumpyMatrixOverRingGeneric):
    def __init__(self, A):
        self.ring = ZZ
        self._constructor = NumpyMatrixOverZ
        self._vector_constructor = NumpyVectorOverZ
        self.A = numpy.asarray(A)

    def change_ring(self, new_ring):
        if isinstance(new_ring, IntegerModRing_generic):
            return self.NumpyMatrixOverZn(self.A, new_ring.order())
        else:
            raise NotImplementedError

class NumpyMatrixOverZn(NumpyMatrixOverRingGeneric):
    def __init__(self, A, n):
        self.ring = Integers(n)
        self.A = numpy.asarray(A) % n
        self.n = n
        self._constructor = functools.partial(NumpyMatrixOverZn, n=n)
        self._vector_constructor = functools.partial(NumpyVectorOverZn, n=n)

def _scipy_sparse_mod(A, n):
    return A - ((A/n).floor()*n).astype(int)

class ScipySparseMatrixOverZ(ScipySparseMatrixOverRingGeneric):
    def __init__(self, A):
        self.ring = ZZ
        self._constructor = ScipySparseMatrixOverZ
        self._vector_constructor = NumpyVectorOverZ
        if sparse.issparse(A):
            self.A = A
        else:
            self.A = sparse.csc_matrix(A)

    def change_ring(self, new_ring):
        if new_ring is ZZ:
            return self
        elif isinstance(new_ring, IntegerModRing_generic):
            return ScipySparseMatrixOverZn(self.A, new_ring.order())
        else:
            raise NotImplementedError


class ScipySparseMatrixOverZn(ScipySparseMatrixOverRingGeneric):
    def __init__(self, A, n):
        self.ring = Integers(n)
        self._constructor = functools.partial(ScipySparseMatrixOverZn, n=n)
        self._vector_constructor = functools.partial(NumpyVectorOverZn, n=n)
        if sparse.issparse(A):
            self.A = A
        else:
            self.A = sparse.csc_matrix(A)
        self.A = _scipy_sparse_mod(A,2)

class GenericVector(object):
    def __getitem__(self, i):
        if  isiterable_or_slice(i):
            return self._constructor(self.v[i])
        else:
            return self.v[i]

    @property
    def backend_obj(self):
        return self.v

    def __setitem__(self, i, x):
        if isinstance(x, GenericVector):
            self.v[i] = x.v
        else:
            self.v[i] = x

    def __neg__(self):
        return self._constructor(-self.v)

    def __len__(self):
        return len(self.v)

    def to_scipy(self):
        return self.to_numpydense()

    def to_magma(self):
        return self.to_magmadense()

    def to_magmadense(self):
        return self.to_numpydense().to_magmadense()

    def convert_to_like_self_preserve_density(self,x):
        return self.convert_to_like_self(x)

class SageVector(GenericVector):
    def __init__(self, v):
        self.v = v
        self._constructor = SageVector

    def to_sagedense(self):
        return self

    def sageobj(self):
        return self.v

    def to_numpydense(self):
        return NumpyVectorOverRing(self.v.numpy(dtype=numpy_dtype_for_ring(self.ring)),
            ring=self.ring)

    @property
    def ring(self):
        return self.v.base_ring()

class MagmaVector(GenericVector):
    def __init__(self, v, ring):
        self.v = v
        self.ring = ring

    def to_numpydense(self):
        return NumpyVectorOverRing(magmaconv.numpy_vector_from_magma(self.v), self.ring)

    def convert_to_like_self(self):
        return self.to_magmadense()

class NumpyVectorOverRingGeneric(GenericVector):
    def __init__(self):
        raise NotImplementedError

    def to_sagedense(self):
        return SageVector(vector(self.ring, self.v))

    def to_numpydense(self):
        return self

    def to_magmadense(self):
        return MagmaVector(
                magmaconv.magma_vector_from_numpy(self.v, self.ring),
                self.ring)

    def convert_to_like_self(self,a):
        return a.to_numpydense()

    def count_nonzero(self):
        return numpy.count_nonzero(self.v)

class NumpyVectorOverZ(NumpyVectorOverRingGeneric):
    def __init__(self, v):
        self.ring = ZZ
        self.v = numpy.array(v)
        self._constructor = NumpyVectorOverZ

class NumpyVectorOverZn(NumpyVectorOverRingGeneric):
    def __init__(self, v, n):
        self.ring = Integers(n)
        self.v = numpy.array(v) % n
        self.n = n
        self._constructor = functools.partial(NumpyVectorOverZn, n = n)

#class ScipyVectorOverRingTemplate(object):
#    def __init__(self, v, ring, numpy_dtype):
#        self.ring = ring
#        self.numpy_dtype = numpy_dtype
#        self.v = v
#
#    def tosage(self,A):
#        return vector(self.v, self.ring)
#
#    def toscipy(self,A):
#        return numpy.array(A.numpy(dtype=self.numpy_dtype))

def solve_matrix_equation_with_constraint(A, subs_indices, xconstr):
    """ Solves the equation Ax = 0, given the constraint that x[i] = xconstr[i]
    for i in subs_indices. """ 

    subs_indices_list = list(subs_indices)
    subs_indices_set = frozenset(subs_indices)

    notsubs_indices_set = frozenset(xrange(A.shape[1])) - subs_indices_set
    notsubs_indices_list = list(notsubs_indices_set)

    assert len(xconstr) == A.shape[1]

    xconstr_reduced = xconstr[subs_indices_list]

    A_reduced_1 = A[:,notsubs_indices_list]
    A_reduced_2 = A[:,subs_indices_list]

    b = -A_reduced_2.dot(xconstr_reduced)

    sol = A_reduced_1.solve_right(b)

    sol_full = A.factory().zero_vector(A.shape[1])
    sol_full[notsubs_indices_list] = sol
    sol_full[subs_indices_list] = xconstr[subs_indices_list]

    assert(A.dot(sol_full).count_nonzero() == 0)
    return sol_full

def solve_simultaneous_matrix_equation(A, a, B, b):
    """ Solves the simultaneous equations Ax = a, Bx = b. """

    assert A.ncols() == B.ncols()
    n = A.ncols()

    if a is None:
        a = A.factory().zero_vector(A.nrows())

    if b is None:
        b = A.factory().zero_vector(B.nrows())

    stack = A.factory().bmat([[A],[B]])
    ab = A.factory().concatenate_vectors(a,b)

    return stack.solve_right(ab)

from sage.matrix.matrix_dense import Matrix_dense
class SageDenseMatrix(GenericMatrix):
    def __init__(self,A):
        if not isinstance(A, Matrix_dense):
            raise TypeError, "Input to SageDenseMatrix constructor must be a sage dense matrix object."
        self.A = A
        self._constructor = SageDenseMatrix

    def to_numpydense(self):
        return NumpyMatrixOverRing(numpy.array(self.A.numpy(dtype=numpy_dtype_for_ring(self.ring))),
            ring=self.ring)

    def to_scipysparse(self):
        return self.to_numpydense().to_scipysparse()

    def to_sagedense(self):
        return self

    def convert_to_like_self(self):
        return self.to_sagedense()

    @property
    def ring(self):
        return self.A.base_ring()

    def solve_right(self, b):
        return b._constructor(self.backend_obj.solve_right(b.backend_obj).change_ring(b.ring))

    @property
    def density(self):
        return "dense"

    def right_kernel_matrix_(self):
        # Flint is a lot faster than the default algorithm when working over the
        # integers.
        if self.ring is ZZ:
            kwargs = dict(algorithm='flint')
        else:
            kwargs = dict()

        ret = SageDenseMatrix(self.A.right_kernel_matrix(basis='computed', **kwargs).transpose())

        return ret

    def column_space_sage(self):
        return self.A.column_module()

    def pivots(self):
        return self.A.pivots()

    def elementary_divisors(self):
        return self.A.elementary_divisors()

class MagmaMatrix(GenericMatrix):
    def __init__(self):
        raise NotImplementedError

    def right_kernel_matrix_(self):
        # Note that Magma uses the opposite ordering, hence all the transposes
        ret = MagmaDenseMatrix( 
                magma.Transpose(magma.KernelMatrix(magma.Transpose(self.A))), self.ring)

        # Although Magma always returns a dense matrix, empirically the number of nonzero entries is
        # often quite low. So it can be more efficient to convert back to a sparse matrix.
        if self.density == 'sparse' and ret.count_nonzero() < 0.1 * ret.nrows() * ret.ncols():
            return ret.to_magmasparse()
        else:
            return ret

    def count_nonzero(self):
        return magma.NumberOfNonZeroEntries(self.A)

class NoSolutionError(Exception):
    pass

class MagmaDenseMatrix(MagmaMatrix):
    def __init__(self,A,ring):
        self.A = A
        self.ring = ring
        self._constructor = functools.partial(MagmaDenseMatrix, ring=ring)

    def nrows(self):
        return magma.Nrows(self.A)

    def ncols(self):
        return magma.Ncols(self.A)

    def to_numpydense(self):
        return NumpyMatrixOverRing(magmaconv.numpy_matrix_from_magma(self.A),
            self.ring)

    def to_sagedense(self):
        return self.to_numpydense().to_sagedense()

    def to_magmasparse(self):
        return MagmaSparseMatrix(magma.SparseMatrix(self.A), self.ring)

    def convert_to_like_self(self,obj):
        return obj.to_magmadense()

    def solve_right(self, v):
        v = v.to_magmadense()

        if self.ring != v.ring:
            raise TypeError

        try:
            if isinstance(v, MagmaMatrix):
                ret = magma.Transpose(magma.Solution(magma.Transpose(self.A), magma.Transpose(v.A)))
                return MagmaDenseMatrix(ret, self.ring)
            elif isinstance(v, MagmaVector):
                ret = magma.Solution(magma.Transpose(self.A), v.v)
                return MagmaVector(ret, self.ring)
            else:
                assert False
        except TypeError as e:
            if e[0].find("No solution exists") != -1:
                raise NoSolutionError
            else:
                raise

    @property
    def density(self):
        return "dense"

class MagmaSparseMatrix(MagmaMatrix):
    def __init__(self,A,ring):
        self.A = A
        self.ring = ring
        self._constructor = functools.partial(MagmaSparseMatrix, ring=ring)

    def to_scipysparse(self):
        return ScipySparseMatrixOverRing(
                magmaconv.scipy_sparse_matrix_from_magma(self.A),
                self.ring)

    def to_magmadense(self):
        return MagmaDenseMatrix(magma.Matrix(self.A), self.ring)

    @property
    def density(self):
        return "sparse"


def numpy_dtype_for_ring(ring):
    if ring is ZZ:
        return int
    elif isinstance(ring, IntegerModRing_generic):
        return int
    else:
        raise NotImplementedError, ring

def NumpyMatrixOverRing(A, ring):
    if ring is ZZ:
        return NumpyMatrixOverZ(A)
    elif isinstance(ring, IntegerModRing_generic):
        return NumpyMatrixOverZn(A, ring.order())
    else:
        raise NotImplementedError

def NumpyVectorOverRing(v, ring):
    if ring is ZZ:
        return NumpyVectorOverZ(v)
    elif isinstance(ring, IntegerModRing_generic):
        return NumpyVectorOverZn(v, ring.order())
    else:
        raise NotImplementedError

def ScipySparseMatrixOverRing(A, ring):
    if ring is ZZ:
        return ScipySparseMatrixOverZ(A)
    elif isinstance(ring, IntegerModRing_generic):
        return ScipySparseMatrixOverZn(A, ring.order())
    else:
        raise NotImplementedError

def image_of_constrained_subspace(A,B, basis=True):
    """ Finds the image V of ker B under A.

    If basis=True, then returns a matrix whose columns are a basis for V. (not implemented yet)
    If basis=False, then returns a matrix whose columns span V, but might
      not be linearly independent.
    """

    if A.shape[1] != B.shape[1]:
        raise ValueError

    if basis:
        raise NotImplementedError

    K = B.right_kernel_matrix()
    AK = A.dot(K)

    return AK
