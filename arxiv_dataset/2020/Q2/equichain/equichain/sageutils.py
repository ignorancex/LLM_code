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

class NotIntegerMatrixError(Exception):
    pass

def sage_matrix_to_numpy_int(A):
    if not A in MatrixSpace(ZZ,A.nrows()):
        print A
        raise NotIntegerMatrixError

    return matrix(ZZ,A).numpy(dtype=int)

def sage_vector_to_numpy_int(v):
    if not v in FreeModule(ZZ,len(v)):
        raise ValueError, "Not an integer vector." + str(v)

    return vector(ZZ,v).numpy(dtype=int)

def fracpart(x):
    return x-sage.all.floor(x)
