r"""

Script for checking for smooth quartics, deduping output, and choosing
nicest looking forms for pointless quartic output of quartic_search C code.

TABLE OF CONTENTS

- get_q
- constructfield_elements_list
- construct_special_field_elements
- quartic_polynomial_basis
- transformation_matrix

- Quartic (class)
  - is_smooth
  - is_equivalent_to
  - best_form

"""

import itertools
import math
import search_tools
import sys

from sage.all import *

################################################################################

def get_q(FF):
  r""" Return q, where FF = GF(q^2) """
  qq = FF.cardinality()
  p = FF.characteristic()
  rbound = ZZ(math.floor(math.log(qq,p)+1))
  for r in range(1,rbound):
    if qq != p**r: continue
    if r % 2 != 0:
      raise ValueError('Expected field cardinality to be a square, not {}'.format(qq))
    return p**(r//2)
  raise RuntimeError('Should never get to here with FF = {}'.format(FF))

################################################################################

def construct_field_elements_list(FF):
  r""".

  Construct the list of elements of ``FF`` in the order used in the C code

  INPUT:

  - ``FF`` -- a finite field of square order q^2

  OUTPUT:

  - A list of length `q^2` consisting of the elements of ``FF``, given in the
    order described below.

  NOTE:

  - Let `t` be a generator of ``FF`` over the prime field `GF(p)`. Then we order
    the elements of `FF` according to the following rules:

    * If `x \in GF(q)` and `y \in GF(q^2) - GF(q)`, then `x < y`.

    * If `x,y \in GF(q)` or `x,y \in GF(q^2) - GF(q)`, write 
      `x = x_0 + x_1 t + ... + x_j t^j` and `y = y_0 + ... + y_j t^j`. 
      Then `x < y` if `\sum x_i p^i < \sum y_i p^i`.

  """
  q = get_q(FF)
  p = FF.characteristic()
  t = FF.gen()
  rats, quads = [], []
  for i in range(q**2):
    elt = sum( a*t**j for j,a in enumerate(ZZ(i).digits(p)))
    if elt**q == elt:
      rats.append(elt)
    else:
      quads.append(elt)
  return rats + quads

################################################################################

def construct_special_field_elements(FF):
  r""".

  Construct the special elements `a` and `c` used in the C code

  INPUT:

  - ``FF`` -- a finite field of square order q^2

  OUTPUT:

  - ``a`` -- The smallest element (in the order defined in
    construct_field_elements_list) of GF(q) such that `T^2 + aT + 1` is
    irreducible over GF(q)

  - ``c`` -- The smallest element (in the order defined in
    construct_field_elements_list) of GF(q^2) that is a root of `T^2 + aT + 1`

  """
  elts = construct_field_elements_list(FF)
  R = PolynomialRing(FF,'T')
  T = R.gen()
  q = get_q(FF)
  for b in elts[:q]:
    pol = T**2+b*T+1
    pol_roots = pol.roots(multiplicities=False)
    if len(pol_roots) == 1: continue
    if any(c**q == c for c in pol_roots): continue
    a = b
    break
  for c in elts[q:]:
    if c**2 + a*c + 1 == 0: return a,c
  raise RuntimeError('Could not find a suitable pair a,c')

################################################################################

def quartic_polynomial_basis(polring, a):
  r"""
  Return the special basis of quartic polynomials used in the paper
  """
  x,y,z = polring.gens()
  basis_pols = [z**4 + x**2*z**2 + a*x*z**3 + y**2*z**2 + a*y*z**3,
    x**4 + a*x**3*z + x**2*z**2,
    x**3*z + a*x**2*z**2 + x*z**3,
    y**4 + a*y**3*z + y**2*z**2,
    y**3*z + a*y**2*z**2 + y*z**3,
    x**3*y,
    x**2*y**2,
    x*y**3,
    x**2*y*z,
    x*y**2*z,
    x*y*z**2]
  return basis_pols

################################################################################

def transformation_matrix(P, Q, c):
  r"""
  Return the matrix moving special quadratic points to P,Q

  INPUT:

  - ``P,Q`` -- quadratic points of the plane such that P,Q, and their 
    conjugates are in general position

  - ``c`` -- an element of GF(q^2) - GF(q)

  OUTPUT:

  - The matrix ``M`` with coefficients in GF(q) such that `M.(c,0,1) = P` and
    `M.(0,c,1) = Q`.
  """
  FF = c.parent()
  q = get_q(FF)
  cbar = c**q
  assert cbar != c
  Pbar = tuple(u**q for u in P)
  Qbar = tuple(u**q for u in Q)

  # Find matrix A mapping
  #   (1,0,0) to (c,0,1)
  #   (0,1,0) to (cbar,0,1)
  #   (0,0,1) to (0,c,1)
  #   (1,1,1) to (0,cbar,1)
  M = matrix(FF,[(c,0,1),(cbar,0,1),(0,c,1)]).transpose()
  vv = M.solve_right(vector((0,cbar,1)))
  assert vv is not None
  u,v,w = vv
  A = matrix(FF,[(c*u,0,u),(v*cbar,0,v),(0,w*c,w)]).transpose()

  # Find matrix B mapping
  #   (1,0,0) to P
  #   (0,1,0) to Pbar
  #   (0,0,1) to Q
  #   (1,1,1) to Qbar
  M = matrix(FF,[P,Pbar,Q]).transpose()
  vv = M.solve_right(vector(Qbar))
  assert vv is not None
  u,v,w = vv
  B = matrix(FF,[u*vector(P),v*vector(Pbar),w*vector(Q)]).transpose()

  return B*A.inverse()

################################################################################

class Quartic(object):
  r"""
  Class for manipulating a pointless plane quartic over GF(q)

  INPUT:

  - ``pol`` -- a polynomial in GF(q^2)[x,y,z] whose coefficients are in GF(q)
    and that does not vanish at any point of the plane over GF(q)

  """
  def __init__(self, pol):
    self._pol = pol
    self._polring = pol.parent()
    self._quad_points = None     # Points over GF(q^2)
    self._all_forms = None       # Forms PGL(3,q)-equivalent to pol
    self._best_form = None       # Smallest form in _all_forms with fewest monomials
    self._c = None               # Store the special element c so as not to recompute


  def __repr__(self):
    return 'Plane Quartic with equation {}'.format(self._pol)

  def is_smooth(self):
    r""" Return True if self is a smooth plane curve """
    polring = self._polring
    pol = self._pol
    x,y,z = polring.gens()
    I = polring.ideal([pol] + [pol.derivative(u) for u in [x,y,z]])
    P2 = ProjectiveSpace(polring)
    return len(P2.subscheme(I).irreducible_components()) == 0

  def quadratic_points(self):
    r"""
    Search for points on self defined over GF(q^2)

    OUTPUT:

    - A complete list of point of the plane over GF(q^2) that lie on self

    """
    if self._quad_points is not None:
      return self._quad_points
    self._quad_points = search_tools.proj_point_search(self._pol)
    return self._quad_points

  def all_forms(self):
    r"""
    Store and return the set of all forms equivalent to self that pass through
    (c:0:1), (0:1:c), and their conjugates

    OUTPUT: 

    - The set of all quartic polynomials in self._polring that vanish at the
      quadratic points (c:0:1) and (0:c:1) as well as their conjugates, 
      where c is the special element as computed in the C code

    NOTE:

    - We do this by finding all pairs of quadratic points P,Q on self such that
      P, Q, and their conjugates are in general position. Then we find the
      matrix M that moves P,Q to the points (c:0:1) and (0:c:1) and transform
      our quartic by M.

    """
    if self._all_forms is not None:
      return self._all_forms
    polring = self._polring
    _,_,z = polring.gens()
    FF = polring.base_ring()
    q = get_q(FF)
    c = self._c
    if c is None:
      _,c = construct_special_field_elements(FF)
      self._c = c

    quad_pts = self.quadratic_points()
    F = self._pol
    all_forms = set()
    for P in quad_pts:
      for Q in quad_pts:
        if P == Q: continue
        Pbar = tuple(u**q for u in P)
        Qbar = tuple(u**q for u in Q)
        # Check if pts in general position
        A = matrix(FF,[P,Q,Pbar,Qbar])
        if A.right_kernel().dimension() > 0: continue
        M = transformation_matrix(P,Q,c)
        FM = search_tools.matrix_act_on_poly(M,F)
        lc = FM.monomial_coefficient(z**4)
        assert lc != 0
        FM *= (1/lc)
        all_forms.add(FM)
    self._all_forms = all_forms
    return self._all_forms

  def best_form(self):
    r"""
    Return the lex-minimal form equivalent to self with the smallest number of monomials
    """
    if self._best_form is not None:
      return self._best_form
    all_forms = self.all_forms()
    min_terms = len(self._pol.dict())
    min_forms = []
    for F in all_forms:
      num_terms = len(F.dict())
      if num_terms > min_terms:
        continue
      elif num_terms == min_terms:
        min_forms.append(F)
      else:
        min_forms = [F]
        min_terms = num_terms
    min_forms.sort()
    self._best_form = min_forms[0]
    return self._best_form

  def is_equivalent_to(self, other):
    r""" Return True if other Quartic is equivalent to self """
    return other._pol in self.all_forms()

  
################################################################################

if __name__ == '__main__':
  if len(sys.argv) != 3:
    raise IOError('Usage: sage quartic_dedupe.py [q] [filename]')
  q = ZZ(sys.argv[1])
  filename = sys.argv[2]
  fp = open(filename,'r')

  # Construct finite field using Conway polynomials
  (p,r), = q.factor()
  print('Quartics over GF({})'.format(q))
  FF = GF(q**2,name='u')
  conway_coefs = list(ConwayPolynomials()[(p,2*r)])
  sage_coefs = FF.polynomial().list()
  assert conway_coefs == sage_coefs, 'sage pol: {}, Conway pol: {}'.format(sage_coefs,conway_coefs)

  polring = PolynomialRing(FF,names=('X','Y','Z'))
  elts = construct_field_elements_list(FF)
  a,c = construct_special_field_elements(FF)
  basis_pols = quartic_polynomial_basis(polring, a)

  coef_vec = lambda s : [ZZ(u) for u in s.strip().split(' ')]
  vecs = fp.readlines()
  fp.close()

  # Determine if progress module is present, for printing
  progress_bar = False
  if verbose:
    try:
      import progress
      progress_bar = True
    except ModuleNotFoundError:
      pass
  
  if progress_bar: prog = progress.Progress(len(vecs))
  quartics = []
  for line in vecs:
    if progress_bar: prog()  
    inds = coef_vec(line)
    coefs = [elts[ind] for ind in inds]
    pol = sum( c*f for c,f in zip(coefs,basis_pols))
    Q = Quartic(pol)
    # Need to determine if pol gives a smooth curve only when q < 7
    if q < 7 and not Q.is_smooth(): continue
    if any(C.is_equivalent_to(Q) for C in quartics): continue
    quartics.append(Q)
  if progress_bar: prog.finalize()

  # Coerce to subfield GF(q) in order to print nicely
  EE = GF(q,'t')
  t = EE.gen()
  R = PolynomialRing(EE,names=('x','y','z'))
  x,y,z = R.gens()
  FF_to_EE = {}
  # Identify the image of the generator first
  for elt in FF:
    minpol = elt.minimal_polynomial()
    if minpol(t) != 0: continue
    image_t = elt
    break
  for elt in EE:
    if q.is_prime():
      FFelt = FF(elt)
    else:
      v = vector(elt)
      FFelt = sum(a*image_t**i for i,a in enumerate(v))
    FF_to_EE[FFelt] = elt
  def pol_res(pol):
    mm = pol.monomials()
    rv = R(0)
    for exps,u in pol.dict().items():
      i,j,k = exps
      rv += FF_to_EE[u] * x**i * y**j * z**k
    return rv

  print('Found {} smooth pointless quartics'.format(len(quartics)))
  fp = open(filename+'.excessive','w')
  for Q in quartics:
    bf = Q.best_form()
    fp.write(str(pol_res(bf))+'\n')
  fp.close()
