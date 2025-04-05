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
import equichain
import equichain.grps
import equichain.wigner_seitz
import equichain.resolutions

def check_space_group(i, soc=False, d=3):
    """ Compute LSM quantities for a given space group.

    The idea is that we work on the torus T^d, as described in Appendix G of
    [1]. The torus is acted upon by the point group G_pt = Gspace/Z^d. We first
    construct a cell decomposition of T^d that satisfies the regularity
    condition discussed in Section III.A of [1] with respect to the action of
    G_pt.

    The function returns the tuple (sp_0, sp_1, ..., sp_d), where
    sp_r is the torsion cofficients of the finite Abelian group A_r, defined as:

    * For the case where the input parameter soc = False:
        A_0 := h_0^{G_pt}(X_0, Z_2),
        where h_*^{G_pt} denotes equivariant homology, and X_r denotes the
        r-skeleton of the cell decomposition of T^d.

    * For the case where the input parameter soc = True:
        A_0 := h_{-3}^{G_pt}(X_0, Z),

    * In both cases, A_r is the image of A_0 under the map in equivariant homology induced by
        the embedding X_0 -> X_r.

    For the full details of this computed data to LSM results, see [1].
    The basic idea is that the soc=False case is relevant to spin systems
    without spin-orbit coupling, or with spin-orbit coupling and unbroken
    time-reversal symmetry, whereas the soc=True case is relevant to spin
    systems with spin-orbit coupling and broken time-reversal symmetry.

    Specifically, 
    * A_0 corresponds to the classification of anomalous textures on
    the vertices of the cell decomposition.
    * A_1 corresponds to the classification of anomalous textures, modded out by
    lattice homotopy equivalence.
    * A_r corresponds to the classification of anomalous textures, modded out by
    anomalous textures which are cancellable by an r-skeletal defect network.

    References:
    [1] Dominic V. Else and Ryan Thorngren, "Topological theory of
    Lieb-Schultz-Mattis theorems in quantum spin systems", arXiv:1907.08204

    Parameters:
    i (int): The space group number
          (according to the International Tables for Crystallography)
    soc (bool): See description above.
    d (int): The space dimension (must be either 2 or 3).

    Return values:
    The tuple (sp_0, ..., sp_d) [see description above].
    """

    gap.load_package("hap")

    if d not in (2,3):
        raise NotImplementedError, \
                "Currently the space dimension must be either 2 or 3."

    G = gap.SpaceGroupIT(d,i)
    Gq = equichain.grps.GapAffineQuotientGroup(G,d=d)

    cplx = equichain.wigner_seitz.space_group_wigner_seitz_barycentric_subdivision(d,G)

    if soc:
        n=3
        k=0
        ring = ZZ
    else:
        n=0
        k=0
        ring = Integers(2)

    #twist = equichain.grps.TwistedIntegers.from_orientation_reversing(Gq)
    twist = equichain.grps.TwistedIntegers.untwisted(Gq)

    resolution = equichain.resolutions.HapResolution(
            gap.ResolutionFiniteGroup(Gq.gap_quotient_grp, n+(d-k)-1),
            Gq
            )

    sp = [None]*(d+1)

    page_fns = [equichain.E1page,equichain.E2page,equichain.E3page]
    for i in xrange(d):
        sp[i] = page_fns[i](cplx,n,k,Gq,ring=ring,twist=twist,resolution=resolution)

    sp[d] = sp[d-1] # This only works because G_pt does not leave any
                    # top-dimensional cells invariant!

    return tuple(sp)
