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

from equichain.utils import *
from equichain.sageutils import *
from equichain import wigner_seitz
from sage.all import *
import scipy.linalg
import string
gap.load_package("Cryst")
gap.SetCrystGroupDefaultAction(gap.LeftAction)

class MatrixQuotientGroupElement(object):
    pass

class MagneticTorusTranslationGroupElement(MatrixQuotientGroupElement):
    def __init__(self, G, gaprepr):
        self.G = G
        self.gaprepr = gaprepr
        self._matrix_representative = None

    def __prod__(a, b):
        assert a.G is b.G
        return MagneticTorusTranslationGroupElement(a.G, a.gaprepr*b.gaprepr)

    def __pow__(a, n):
        return MagneticTorusTranslationGroupElement(a.G, a.gaprepr**n)

    def __eq__(a,b):
        assert a.G is b.G
        return a.gaprepr == b.gaprepr

    def __ne__(a, b):
        return not (a == b)

    def to_translation_vector(self):
        transgap = gap.Image(self.G.homo_to_gap_transgrp, self.gaprepr)
        if transgap == gap.Identity(self.G.gap_transgrp):
            return vector([0,0])

        s = str(transgap)
        components = s.split('*')
        t = vector(ZZ,2)
        for comp in components:
            if '^' in comp:
                splitted = comp.split('^')
                genname = splitted[0]
                power = eval(splitted[1])
            else:
                genname = comp
                power = 1

            if genname == 'f1':
                cur_t = vector([0,1])
            elif genname == 'f2':
                cur_t = vector([1,0])
            else:
                print "unexpected genname:", genname
                assert False

            t += cur_t*power

        return t

    def as_matrix_representative(self):
        if self._matrix_representative is None:
            self._matrix_representative = affine_transformation_from_translation_vector_sage(self.to_translation_vector())
        return self._matrix_representative

    def as_matrix_representative_numpy_int(self):
        raise NotImplementedError

    def toindex(self):
        raise NotImplementedError
        return self.G.element_to_index(self)

class MagneticTorusTranslationGroup(object):
    def __iter__(self):
        return iter(self.elts)

    def element_from_gap(self, g):
        return MagneticTorusTranslationGroupElement(self, g)
    
    def element_to_index(self, e):
        strrepr = str(e.gaprepr)
        try:
            return self.elt_index_by_strrepr[strrepr]
        except KeyError:
            for (i,g) in enumerate(self.elts):
                if g == e:
                    self.elt_index_by_strrepr[strrepr] = i
                    return i
            raise KeyError

    def element_by_index(self, i):
        return self.elts[i]

    def size(self):
        return len(self.elts)

    def identity(self):
        return self.elts[0]**0

    def gens(self):
        return self._gens

    def __len__(self):
        return self.size()

    def __init__(self, Lx, Ly, A_torsion_coeffs, fluxcoeffs):
        self.Lx = Lx
        self.Ly = Ly
        self.A_torsion_coeffs = A_torsion_coeffs
        self.fluxcoeffs = fluxcoeffs

        An = len(A_torsion_coeffs)
        assert An <= 26
        assert len(fluxcoeffs) == An

        for L in Lx,Ly:
            assert all(fluxcoeffs[i]*L % A_torsion_coeffs[i] == 0 for i in xrange(An))

        A_gennames = string.ascii_lowercase[0:An]

        gapcmd = ('FreeGroup("Tx","Ty"' +
                ''.join(',"' + genname + '"' for genname in A_gennames)
                  + ')')
        F = gap(gapcmd)

        gens = gap.GeneratorsOfGroup(F)
        Tx = gens[1]
        Ty = gens[2]
        Agens = [ gens[i+3] for i in xrange(An) ]

        flux = Agens[0]**fluxcoeffs[0]
        for i in xrange(1,An):
            flux *= Agens[i]**fluxcoeffs[i]

        relations = [ gap.Comm(Tx,Ty)*flux ]
        for i in xrange(An):
            relations.append(Agens[i] ** A_torsion_coeffs[i])
            relations.append(gap.Comm(Tx, Agens[i]))
            relations.append(gap.Comm(Ty, Agens[i]))
        relations.append(Tx**Lx)
        relations.append(Ty**Ly)

        for i in xrange(An):
            for j in xrange(1,An):
                relations.append(gap.Comm(Agens[i], Agens[j]))

        relations = gap(relations)
        self.gapgrp = gap(F.name() + ' / ' + relations.name())
        self.elts = list(MagneticTorusTranslationGroupElement(self, g) for g in gap.List(self.gapgrp))
        self._gens = list(MagneticTorusTranslationGroupElement(self,g) for g in gap.GeneratorsOfGroup(self.gapgrp))

        self.gap_transgrp = gap.FreeAbelianGroup(2)
        gap_transgrp_gens = gap.GeneratorsOfGroup(self.gap_transgrp)
        self.homo_to_gap_transgrp = gap.GroupHomomorphismByImagesNC(self.gapgrp, self.gap_transgrp,
                gap.GeneratorsOfGroup(self.gapgrp),
                [ gap_transgrp_gens[1], gap_transgrp_gens[2] ] + [ gap.Identity(self.gap_transgrp) ]*An
                )
        self.elt_index_by_strrepr = {}

class GapAffineQuotientGroup(object):
    def _base_init(self):
        self.stored_coset_representatives = dict()
        for g in self.sage_quotient_grp:
            self.stored_coset_representatives[g] = self._coset_representative(g)

        self.els = [g for g in self.sage_quotient_grp]
        self.els_reverse_lookup = dict()
        for i in xrange(len(self.els)):
            self.els_reverse_lookup[self.els[i]] = i

        self.stored_coset_representatives_numpy_int = None
        self.stored_coset_representatives_numpy = None

    def identity(self):
        return GapAffineQuotientGroupElement(self, self.sage_quotient_grp.identity())

    def __init__(self, G, N=None, scale=1, d=3):
        if N is None:
            quotient_group = gap.PointGroup(G)
            self.homo_to_factor = gap.PointHomomorphism(G)
        else:
            self.homo_to_factor = gap.NaturalHomomorphismByNormalSubgroup(G,N)
            quotient_group = gap.ImagesSource(self.homo_to_factor)

        self.d = d

        iso_to_perm = gap.IsomorphismPermGroup(quotient_group)
        self.iso_to_perm_inverse = gap.InverseGeneralMapping(iso_to_perm)
        self._gap_quotient_grp = gap.Image(iso_to_perm)
        self.sage_quotient_grp = PermutationGroup(gap_group = self._gap_quotient_grp)
        self.basegrp = self
        self.scale = scale

        Q = wigner_seitz.space_group_orthogonality_matrix(d,G).numpy(dtype=float)
        A = scipy.linalg.sqrtm(Q)
        self.orthogonalizing_basis_change = A

        self._base_init()

        #self.multiplication_table = numpy.array(shape=(len(self.els),len(self.els)), dtype=int)
        #for i in len(self.els):
        #    for j in len(self.els):

    @property
    def gap_quotient_grp(self):
        if self._gap_quotient_grp is None:
            self._gap_quotient_grp = gap.Group(
                    [ g for g in self.sage_quotient_grp.gens()]
                )

        return self._gap_quotient_grp

    def element_from_gap(self, gapg):
        return GapAffineQuotientGroupElement(self.basegrp,
                self.basegrp.sage_quotient_grp(gapg))

    def subgroup(self, gens):
        G = GapAffineQuotientGroup.__new__(GapAffineQuotientGroup)
        G.sage_quotient_grp = self.sage_quotient_grp.subgroup([g.sageperm for g in gens])
        G.iso_to_perm_inverse = self.iso_to_perm_inverse
        G.homo_to_factor = self.homo_to_factor
        G.basegrp = self.basegrp
        G._gap_quotient_grp = None
        G.scale = self.scale
        G.orthogonalizing_basis_change = self.orthogonalizing_basis_change
        G.d = self.d
        #G.basegrp = G
        G._base_init()
        return G

    def __iter__(self):
        return iter(self.elements())

    def gens(self):
        return [GapAffineQuotientGroupElement(self.basegrp,g) for g in self.sage_quotient_grp.gens()]

    def elements(self):
        return [GapAffineQuotientGroupElement(self.basegrp, g) for g in self.sage_quotient_grp]

    def element_to_index(self, g):
        return self.els_reverse_lookup[g.sageperm]

    def element_by_index(self,i):
        return GapAffineQuotientGroupElement(self.basegrp,self.els[i])

    def _coset_representative(self,g):
        g = gap(g)
        A = matrix(gap.PreImagesRepresentative(
                self.homo_to_factor,
                gap.Image(self.iso_to_perm_inverse, g)).sage())
        B = affine_transformation_rescale(A,self.scale)
        return B

    def coset_representative(self,g):
        return self.stored_coset_representatives[g.sageperm]

    def _compute_coset_representatives_numpy_int(self):
        self.stored_coset_representatives_numpy_int = dict(
                (g, sage_matrix_to_numpy_int(A))
                for g,A in self.stored_coset_representatives.iteritems()
                )

    def coset_representative_numpy_int(self,g):
        if self.stored_coset_representatives_numpy_int is None:
            self._compute_coset_representatives_numpy_int()
        return self.stored_coset_representatives_numpy_int[g.sageperm]

    def _compute_coset_representatives_numpy(self):
        self.stored_coset_representatives_numpy = dict(
                (g, A.numpy(dtype=float))
                for g,A in self.stored_coset_representatives.iteritems()
                )

    def coset_representative_numpy(self,g):
        if self.stored_coset_representatives_numpy is None:
            self._compute_coset_representatives_numpy()
        return self.stored_coset_representatives_numpy[g.sageperm]

    def orthogonal_point_group_element(self,g):
        if self.scale != 1:
            raise NotImplementedError

        A = self.orthogonalizing_basis_change
        O = g.as_matrix_representative_numpy()[0:self.d,0:self.d]
        return numpy.dot(numpy.dot(A, O), scipy.linalg.inv(A))

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.els)

class TrivialAffineMatrixGroupElement(MatrixQuotientGroupElement):
    def __init__(self,d):
        self.d = d

    def __mul__(self, b):
        return self

    def __eq__(a,b):
        return True

    def __ne__(a,b):
        return False

    def __pow__(self, n):
        return self

    def as_matrix_representative(self):
        return matrix.identity(ZZ,self.d+1)

    def as_matrix_representative_numpy_int(self):
        return numpy.eye(self.d+1, dtype=int)

    def toindex(self):
        return 0

class TrivialAffineMatrixGroup(object):
    def __init__(self, d):
        self.d = d

    def identity(self):
        return TrivialAffineMatrixGroupElement(self.d)

    def subgroup(self, gens):
        return self

    def __iter__(self):
        return iter([self.identity()])

    def gens(self):
        return [self.identity()]

    def elements(self):
        return [self.identity()]

    def element_by_index(self,i):
        return self.identity()

    def __len__(self):
        return 1

    def size(self):
        return 1

class TwistedIntegers(object):
    def __init__(self, G, action_on_Z_fn):
        self.G = G
        self.factors = [ action_on_Z_fn(g) for (i,g) in enumerate(G) ]

    @staticmethod
    def untwisted(G):
        return TwistedIntegers(G, lambda g: 1)

    @staticmethod
    def from_orientation_reversing(G):
        def action_on_Z_fn(g):
            det = gap.Determinant(g.as_matrix_representative())
            if int(det) not in [1,-1]:
                raise ValueError, "Determinant was not +/- 1!"
            return det
        return TwistedIntegers(G, action_on_Z_fn)

    def group(self):
        return self.G

    def action_on_Z(self, g):
        return self.factors[self.G.element_to_index(g)]

class GapAffineQuotientGroupElement(MatrixQuotientGroupElement):
    def __init__(self, G, sageperm):
        self.G = G
        self.sageperm = sageperm

    def __mul__(a, b):
        assert a.G is b.G
        return GapAffineQuotientGroupElement(a.G, a.sageperm*b.sageperm)

    def __eq__(a,b):
        return a.sageperm == b.sageperm

    def __ne__(a,b):
        return not a == b

    def __pow__(self, n):
        return GapAffineQuotientGroupElement(self.G, self.sageperm**(-1))

    def as_matrix_representative(self):
        return self.G.coset_representative(self)

    def as_matrix_representative_numpy(self):
        return self.G.coset_representative_numpy(self)

    def orthogonal_point_group_element(self):
        return self.G.orthogonal_point_group_element(self)

    def as_matrix_representative_numpy_int(self):
        return self.G.coset_representative_numpy_int(self)

    def toindex(self):
        raise NotImplementedError
        return self.G.element_to_index(self)
        #if self._index is None:
        #    return self.G.element_to_index(self)
        #else:
        #    return self._index

    def toindex_basegrp(self):
        raise NotImplementedError
        return self.G.basegrp.element_to_index(self)

def affine_transformation_rescale(A,scale):
    A = copy(A)
    d = A.nrows()-1
    A[0:d,d] *= scale
    return A

def affine_transformation_from_translation_vector_sage(v):
    d = len(v)
    A = matrix(ZZ, d+1, d+1)
    for i in xrange(d):
        A[i,i] = 1
    A[0:d,d] = v
    return A

def affine_transformation_from_translation_vector_numpy(v):
    d = len(v)
    A = numpy.eye(d+1, dtype=int)
    A[0:d,d] = v
    return numpy.asmatrix(A)

#def toroidal_space_group(d,n,L):
#    G = gap.SpaceGroupIT(d,n)
#    trans = gap.translation_subgroup(G,L)
#    return GapQuotientGroup(G,trans)
