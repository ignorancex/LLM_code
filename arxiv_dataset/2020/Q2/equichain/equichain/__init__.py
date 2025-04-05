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
import itertools
import numpy
from scipy import sparse

from equichain.utils import *
from equichain.grps import *
from equichain.sageutils import *
import equichain.resolutions as resolutions
from equichain.linalg import *
from sage.all import *



from sage.modules.vector_rational_dense import Vector_rational_dense

class Point(object):
    def __init__(self, coords):
        if isinstance(coords, Vector_rational_dense):
            self.coords = copy(coords)
        else:
            self.coords = copy(vector(QQ, coords))
        self.coords.set_immutable()

    def __truediv__(self, m):
        return self.__div__(m)

    def __div__(self, m):
        return Point(self.coords / m)

    def __add__(a,b):
        return Point(a.coords + b.coords)

    def __hash__(self):
        return hash(self.coords)

    def __eq__(x,y):
        return x.coords == y.coords

    def __gt__(x,y):
        return x.coords > y.coords

    def __ge__(x,y):
        return x.coords >= y.coords

    def __lt__(x,y):
        return x.coords < y.coords

    def __le__(x,y):
        return x.coords <= y.coords

    def __ne__(x,y):
        return x.coords != y.coords

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.coords)

    def act_with(self, action):
        if isinstance(action, TranslationAction):
            return action(self)

        if isinstance(action, MatrixQuotientGroupElement):
            action = action.as_matrix_representative()

        v = vector(tuple(self.coords) + (1,))
        vout = action*v
        ret = Point(vout[:-1])

        return ret


class TranslationAction(object):
    def __init__(self, trans):
        if isinstance(trans, Vector_rational_dense):
            self.trans = copy(trans)
        else:
            self.trans = vector(QQ, trans)

    def __call__(self, pt):
        return Point(pt.coords + self.trans)

    def __mul__(x, y):
        return TranslationAction(x.trans + y.trans)

    def __pow__(self,n):
        return TranslationAction(n*self.trans)

    @staticmethod
    def get_translation_basis(d):
        ret = []
        for i in xrange(d):
            trans = [0]*d
            trans[i] = 1
            ret.append(TranslationAction(trans))
        return ret

class ConvexHullCell(object):
    def __init__(self,points,orientation):
        # The orientation is an anti-symmetric matrix implementing the
        # projected determinant. Pass None for an unoriented cell.
        self.points = frozenset(points)
        self._orientation = orientation

    def midpoint(self):
        i = (pt.coords for pt in self.points)
        return (i.next() + sum(i)) / len(self.points)

    def center(self):
        points = list(self.points)
        sm = points[0]
        for p in points[1:]:
            sm = sm + p
        return sm / len(points)


    def forget_orientation(self):
        return ConvexHullCell(self.points, None)

    def act_with(self,action):
        if self._orientation is None:
            new_orientation = None
        else:
            new_orientation = transform_levi_civita(self._orientation,action)

        ret = ConvexHullCell([p.act_with(action) for p in self.points],
                orientation=new_orientation)

        return ret

    def __eq__(a,b):
        return a.points == b.points

    def __ne__(a,b):
        return not a == b

    def __hash__(self):
        return hash(self.points)

    def __iter__(self):
        return iter(self.points)

    def __str__(self):
        return "CELL" + str(list(self.points))

    def __repr__(self):
        return str(self)

    def orientation(self):
        return self._orientation


class QuotientCell(object):
    def __init__(self, representative_cell, equivalence_relation):
        self.representative_cell = representative_cell
        self.equivalence_relation = equivalence_relation

    def __eq__(a,b):
        return a.equivalence_relation.are_equivalent(a.representative_cell, b.representative_cell)

    def __ne__(a,b):
        return not a == b

    def __hash__(self):
        return self.equivalence_relation.hash_equivalence_class(self.representative_cell)

    def act_with(self, action):
        return QuotientCell(self.representative_cell.act_with(action),
                self.equivalence_relation)

    def boundary(self):
        return FormalIntegerSum( dict( (QuotientCell(k,self.equivalence_relation),v) for k,v in
                self.representative_cell.boundary().coeffs.iteritems() ) )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "QUOTIENT:" + str(self.representative_cell)

class EquivalenceRelationFromCommutingActionGenerators(object):
    def __init__(self, gens, representatives, default_max_order=7,
            reduce_order=None, precompute_representatives_for=None,
            representatives_helper=None):
        self.gens = gens
        self.default_max_order = default_max_order
        self.representatives_helper = representatives_helper

        # If reduce_order is not None, then the set of representatives might be
        # overcomplete, and we try to reduce it (but we only look for
        # relations generated by powers <= reduce_order.)
        if reduce_order is not None:
            self.map_to_representatives = {}
            self.representatives = set()


            for cell in representatives:
                if not self.canonical_representative(cell,max_order=reduce_order,return_bool=True):
                    self.representatives.add(cell)
                    self.map_to_representatives[cell] = cell
        else:
            self.map_to_representatives = dict( (cell,cell) for cell in
                    representatives )
            self.representatives = frozenset(representatives)

        if precompute_representatives_for is not None:
            for cell in precompute_representatives_for:
                self.canonical_representative(cell)

    def canonical_representative(self,cell, max_order=None, return_bool=False):
        if cell in self.map_to_representatives:
            return self.map_to_representatives[cell]

        # If the cell is not already cached, use the representatives helper to
        # try to get a better representative
        if self.representatives_helper is not None:
            cell = self.representatives_helper(cell)

            # Now try again
            if cell in self.map_to_representatives:
                return self.map_to_representatives[cell]
        
        #max(max(numpy.max(numpy.abs(pt.coords)) for pt in basecell.points)
                #for basecell in cell.vertices)

        if max_order is None:
            max_order = self.default_max_order

        for morder in xrange(max_order+1):
            for k in itertools.product(range(-morder,morder+1), repeat=len(self.gens)):
                g = product( (self.gens[i]**k[i] for i in xrange(1,len(self.gens))),
                        self.gens[0]**k[0] )
                acted_cell = cell.act_with(g)
                if acted_cell in self.representatives:
                    self.map_to_representatives[cell] = acted_cell

                    if return_bool:
                        return True
                    else:
                        return acted_cell

        if return_bool:
            return False
        else:
            raise RuntimeError, "Cell doesn't appear to have a representative."

    def are_equivalent(self, cell1, cell2):
        return self.canonical_representative(cell1) == self.canonical_representative(cell2)

    def hash_equivalence_class(self, cell):
        return hash(self.canonical_representative(cell))

def get_relative_orientation_cells(cell1, cell2):
    if isinstance(cell1, ConvexHullCell):
        return get_relative_orientation(cell1.orientation(),cell2.orientation())
    else:
        return 1
    
def cell_complex_from_polytope(p, coord_subset, remember_orientation=True):
    if remember_orientation:
        raise NotImplementedError

    def conv_vertex(v):
        return Point([ sage_eval(str(v[i])) for i in coord_subset ])

    vertices = [ conv_vertex(v) for v in p.VERTICES ]

    def conv_cell(c):
        vertices_in_cell = [ vertices[int(i)] for i in c ]
        cell = ConvexHullCell(vertices_in_cell, orientation=None)
        return cell

    cells = [ conv_cell(c) for c in p.HASSE_DIAGRAM.FACES() ]

    d = int(p.CONE_DIM)-1
    cplx = CellComplex(d)
    for k in xrange(d+1):
        for face in p.HASSE_DIAGRAM.nodes_of_dim(k):
            i = int(face)
            if k > 0:
                boundary = FormalIntegerSum(dict( (cells[int(bi)],1) for 
                    bi in p.HASSE_DIAGRAM.ADJACENCY.in_adjacent_nodes(i) ))
            else:
                boundary = FormalIntegerSum()
            cplx.add_cell(k, cells[int(face)], boundary)

    return cplx

class TrivialPermutee(object):
    def act_with(self,g):
        return self

    def __eq__(a,b):
        return True

    def __ne__(a,b):
        return False

    def __hash__(self):
        return 0

    def orientation(self):
        return 1


def get_group_coboundary_matrix(cells, n,G, twist, resolution):
    if isinstance(resolution, resolutions.ZGResolution):
        mapped_cell_indices, mapping_parities = get_group_action_on_cells(cells,G,
                twist=twist,inverse=True)

        return resolution.dual_d_matrix(n, len(cells),
                mapped_cell_indices, mapping_parities, raw=True)
    else:
        raise NotImplementedError, "Unidentified resolution."

class ComplexNotInvariantError(Exception):
    pass

def get_action_on_cells(cells,action):
    mapped_cell_indices = numpy.empty( len(cells), dtype=int)
    mapping_parities = numpy.empty( len(cells), dtype=int)

    for i in xrange(len(cells)):
        acted_cell = cells[i].act_with(action)
        try:
            acted_ci = cells.index(acted_cell)
        except ValueError:
            raise ComplexNotInvariantError
        parity = get_relative_orientation_cells(acted_cell, cells[acted_ci])

        mapped_cell_indices[i] = acted_ci
        mapping_parities[i] = parity
        
    return mapped_cell_indices, mapping_parities

def get_group_action_on_cells(cells, G, twist, inverse=False):
    mapped_cell_indices = numpy.empty( (G.size(), len(cells)), dtype=int)
    mapping_parities = numpy.empty( (G.size(), len(cells)), dtype=int)
    
    for (i,g) in enumerate(G):
        if twist is None:
            anti_unitary_factor = 1
        else:
            anti_unitary_factor = twist.action_on_Z(g)
        mapped_cell_indices_g, mapping_parities_g = get_action_on_cells(cells,
                g**(-1) if inverse else g)
        mapped_cell_indices[i,:] = mapped_cell_indices_g
        mapping_parities[i,:] = mapping_parities_g*anti_unitary_factor

    return mapped_cell_indices, mapping_parities

class OrderedSimplex(object):
    """ This class represents a (combinatorial) simplex, i.e. a tuple of
    vertices (v_1, ... , v_n). Note that it is assumed that there is a partial
    ordering on vertices and that the vertices are input such that v_1 < v_2 <
    v_3 < ... < v_n. This ordering must also be preserved under action on the
    vertices. """

    def __init__(self, vertices):
        self.vertices = tuple(vertices)
        self._hash = hash(self.vertices)

    def __eq__(a,b):
        return a.vertices == b.vertices

    def __ne__(a,b):
        return a.vertices != b.vertices

    def __hash__(self):
        return self._hash

    def boundary(self):
        n = len(self.vertices)
        if n == 1:
            return FormalIntegerSum({})

        coeffs = {}
        for i in xrange(n):
            reduced_word = self.vertices[0:i] + self.vertices[(i+1):]
            coeffs[OrderedSimplex(reduced_word)] = (-1)**i

        return FormalIntegerSum(coeffs)

    def act_with(self,action):
        return OrderedSimplex(v.act_with(action) for v in self.vertices)

    def center(self):
        sm = self.vertices[0].center()
        for v in self.vertices[1:]:
            sm = sm + v.center()
        return sm / len(self.vertices)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "SIMPLEX" + str(self.vertices)

def group_orbits(cells,  G):
    cell_indices = set(xrange(len(cells)))
    
    orbits = []

    while len(cell_indices) > 0:
        orbit = set()

        i = iter(cell_indices).next()

        for g in G:
            j = _get_action_on_cell_index(cells,g,i)
            orbit.add(j)
            cell_indices.discard(j)

        orbits.append(list(orbit))

    return orbits

def _get_action_on_cell_index(cells,action,i):
    acted_cell = cells[i].act_with(action)
    return cells.index(acted_cell)

class CellComplex(object):
    def merge(cplx1, cplx2, check_not_disjoint_dimensions=()):
        assert cplx1.ndims == cplx2.ndims
        ndims = cplx1.ndims

        ret = CellComplex(cplx1.ndims)
        for k in xrange(ndims+1):
            cells1 = set(cplx1.cells[k])
            cells2 = set(cplx2.cells[k])

            common_cells = cells1.intersection(cells2)
            if k in check_not_disjoint_dimensions and len(common_cells) == 0:
                raise RuntimeError, "No common cells of dimension " + str(k) + "!"

            for common_cell in cells1.intersection(cells2):
                if cplx1.boundary_data[common_cell] != cplx2.boundary_data[common_cell]:
                    raise ValueError, "Can't merge cells -- they don't have the same boundary."

            for cell in cells1.difference(cells2):
                 ret.add_cell(k, cell, cplx1.boundary_data[cell])
            for cell in cells2:
                 ret.add_cell(k, cell, cplx2.boundary_data[cell])

        return ret

    def all_cells_iterator(self):
        for cells_k in self.cells:
            for cell in cells_k:
                yield cell

    def all_cells_iterator_unoriented(self):
        for cells_k in self.cells:
            for cell in cells_k:
                yield cell.forget_orientation()

    def get_group_orbits(self, k, G):
        return group_orbits(self.cells[k], G)

    def quotient(self, equiv_relation):
        cplx = CellComplex(self.ndims)

        for k in xrange(self.ndims+1):
            quotient_cells = set(QuotientCell(cell,equiv_relation) for cell in self.cells[k])
            for q in quotient_cells:
                cplx.add_cell(k, q)

        return cplx

    def _all_contained_cells_iterator_unoriented(self, cell):
        for boundary_cell in self.boundary_data[cell].coeffs.keys():
            yield boundary_cell.forget_orientation()
            for c in self._all_contained_cells_iterator_unoriented(boundary_cell):
                yield c

    def _barycentric_word_iterator(self, base_cell=None):
        if base_cell is None:
            for cell in self.all_cells_iterator_unoriented():
                for word in self._barycentric_word_iterator(cell):
                    yield word
        else:
            yield (base_cell,)
            for cell in self._all_contained_cells_iterator_unoriented(base_cell):
                for word in self._barycentric_word_iterator(cell):
                    yield word + (base_cell,)

    def barycentric_subdivision(self):
        cplx = CellComplex(self.ndims)
        for word in self._barycentric_word_iterator():
            cplx.add_cell(len(word)-1, OrderedSimplex(word))
        return cplx

    def get_group_coboundary_matrix(self, n,G,k, twist=None, resolution='cython_bar'):
        return get_group_coboundary_matrix(self.cells[k],n,G, twist=twist, resolution=resolution)

    #def get_action_matrix(self, k, action):
    #    return CellComplex._get_action_matrix(self.cells[k], action)

    def get_boundary_matrix(self, k):
        cells_km1 = self.cells[k-1]
        cells_k = self.cells[k]

        A = sparse.dok_matrix( (len(cells_km1), len(cells_k)), dtype=int )
        for i in xrange(len(cells_k)):
            for boundary_cell,coeff in self.boundary_data[cells_k[i]]:
                j = cells_km1.index(boundary_cell)
                A[j,i] += coeff
        return A

    def get_boundary_matrix_group_cochain(self, k,n,G, resolution='cython_bar'):
        A = self.get_boundary_matrix(k)
        rank = resolution.rank(n)
        return ScipySparseMatrixOverZ(sparse.kron(A,
            sparse.eye(rank,dtype=int)).tocsc())

    def add_cell(self, ndim, cell, boundary=None):
        self.cells[ndim].append(cell)
        if boundary is None:
            self.boundary_data[cell] = cell.boundary()
        else:
            self.boundary_data[cell] = boundary

    def __init__(self,ndims):
        self.ndims = ndims
        self.cells = [None]*(ndims+1)
        self.boundary_data = {}
        for i in xrange(ndims+1):
            self.cells[i] = IndexedSet()

def test_has_solution(fn):
    try:
        fn()
    except NoSolutionError:
        return False
    return True

def Enpage_helper(img,  cplx,n,k,G,twist,ring,resolution, return_module_obj=False):
    if n > 0:
        delta0_in = cplx.get_group_coboundary_matrix(n=(n-1), k=k, G=G, twist=twist, resolution=resolution)

        delta0_in = delta0_in.to_numpydense()
        img = img.to_numpydense()
        img = img.factory().bmat([img,delta0_in])

        delta0_out = cplx.get_group_coboundary_matrix(n=n, k=k, G=G, twist=twist, resolution=resolution)

        return kernel_mod_image(delta0_out, img, return_module_obj)
    else:
        return kernel_mod_image(None, img, return_module_obj)

def E3page(cplx,n,k,G,twist,ring, resolution):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G, resolution=resolution)
    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G, resolution=resolution)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G, twist=twist, resolution=resolution)
    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G, twist=twist, resolution=resolution)

    d1,d2,delta1,delta2 = (x.change_ring(ring) for x in (d1,d2,delta1,delta2))
    factory = d1.factory()

    z = factory.zeros((d1.shape[0], delta2.shape[1]))
    A = factory.bmat([[d1,z]])

    B = factory.bmat([[None, delta2],
                     [delta1, -d2]])

    img = image_of_constrained_subspace(A,B, False)

    return Enpage_helper(img, cplx,n,k,G,twist,ring,resolution)

def group_cohomology(G,n, resolution, ring, twist=None):
    d1 = get_group_coboundary_matrix([TrivialPermutee()], n, G, twist=twist, resolution=resolution)
    d2 = get_group_coboundary_matrix([TrivialPermutee()], n-1, G, twist=twist, resolution=resolution)

    d1,d2 = (x.change_ring(ring) for x in (d1, d2))

    return kernel_mod_image(d1,d2)

def E1page(cplx,n,k,G,twist,ring,resolution, return_module_obj=False):
    if n == 0:
        if return_module_obj:
            raise NotImplementedError

        if ring.order() == oo:
            order = 0
        else:
            order = ring.order()
        return [order]*len(cplx.cells[k])
    else:
        delta0_in = cplx.get_group_coboundary_matrix(n=(n-1), k=k, G=G, twist=twist, resolution=resolution)
        delta0_out = cplx.get_group_coboundary_matrix(n=n, k=k, G=G, twist=twist, resolution=resolution)
        return kernel_mod_image(delta0_out, delta0_in, return_module_obj)

def E2page(cplx,n,k,G,twist,ring,resolution, return_module_obj=False):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G, resolution=resolution)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G, twist=twist, resolution=resolution)

    d1,delta1 = (x.change_ring(ring) for x in (d1,delta1))

    img = image_of_constrained_subspace(d1, delta1, basis=False)

    return Enpage_helper(img, cplx,n,k,G,twist,ring,resolution, return_module_obj=return_module_obj)

def gap_space_group_translation_subgroup(G,n):
    gap_fn = gap("""function(G,n)
    local T,basis,translation_to_affine;

    translation_to_affine := function(T)
        local n,A,i;
        n := Length(T);
        A := NullMat(n+1,n+1);
        for i in [1..n] do
            A[i][n+1] := T[i];
        od;
        for i in [1..(n+1)] do
            A[i][i] := 1;
        od;
        return A;
    end;

    basis := TranslationBasis(G)*n;
    T := List(basis,translation_to_affine);
    return Subgroup(G, T);
    end;
    """)

    return gap_fn(G,n)

