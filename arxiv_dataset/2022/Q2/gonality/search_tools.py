r"""

Tools for finding and studying curves of small genus over small 
finite fields with various properties. 

TABLE OF CONTENTS

- FIELD AND POINT ITERATORS
  - field_element
  - field_elements
  - affine_space_point_iterator
  - proj_space_point_iterator

- UTILITIES
  - poly_iterator
  - choose_poly
  - poly_index
  - flatten_matrix
  - write_list_of_mats_to_file
  - restore_matrices
  - compare_matrix_lists
  - int_list_file_to_singular_file
  - ideal_dimension
  - ideal_is_prime
  - evaluate_poly
  - matrix_act_on_poly
  - compare_quadric_files
  - start_and_stop_work
  - cat_files

- PROJECTIVE SCHEME FUNCTIONS
  - normalize_point
  - proj_point_search
  - proj_variety_has_n_points
  - hypersurface_singularities
  - homogeneous_monomials
  - restore_complete_intersections
  - quadric_singularity_dimension
  - quadric_point_count
  - allowed_invariant_quadric_mask
  - complete_intersection_tangent_line
  - point_reps_of_bounded_degree

- ORTHOGONAL GROUPS
  - naive_orthogonal_group
  - orthogonal_group
  - orthogonal_transitive_set
  - orthogonal_stabilizer
  - new_orthogonal_group  
  - char2_dim4_nonsplit_smooth_orthogonal_group
  - char2_dim4_nonsplit_quadric_automorphism_group
  - odd_char_dim4_nonsplit_smooth_orthogonal_group
  - odd_char_dim4_nonsplit_quadric_automorphism_group
  - char2_dim5_smooth_orthogonal_group
  - char2_dim5_nonsplit_rank4_orthogonal_group
  - odd_char_dim5_smooth_orthogonal_group
  - odd_char_dim5_nonsplit_rank4_orthogonal_group

- GENUS-3 CURVES
  - genus3_canonical_curves  

- GENUS-4 CURVES
  - genus4_canonical_curves
  - count_genus4_points

- GENUS-5 CURVES
  - second_quadric_class_reps
  - genus5_nontrigonal_curve_search
  - count_genus5_points
  - genus5_g14_invariants
  - has_gonality_four
  - test_curves_for_gonality_five
  - genus5_trigonal_curves
  - IntersectionOfThreeQuadrics (class)

- HYPERELLIPTIC CURVES
  - even_characteristic_hyperelliptic_search
  - odd_characteristic_hyperelliptic_search
  - hyperelliptic_search

"""

import itertools
import progress
from collections import defaultdict
from sage.all import *

from sage.rings.polynomial.multi_polynomial_ideal import MPolynomialIdeal
import sage.libs.singular.function_factory

################################################################################
#########################  FIELD AND POINT ITERATORS  ##########################
################################################################################

def field_element(field, index):
  r"""
  Return a particular element ``field`` (lex order for power basis)

  INPUT:

  - ``field`` -- a finite field

  - ``index`` -- nonnegative integer that must be less than the field
    cardinality

  NOTE:

  - We use the standard ordering on a prime field `0 < 1 < ... < p-1`, and we
    extend this to non-prime fields using the lexicographic ordering on the
    coefficients with respect to the power basis, where `1 < t < ... < t^{d-1}`,
    `t` being a generator of the degree-`d` extension over the prime field.

  EXAMPLES::

    sage: R.<x> = GF(2)[]
    sage: F.<t> = GF(4,modulus=x^2+x+1)
    sage: print([field_element(F,i) for i in range(4)])
    [0, 1, t, t + 1]

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  p = field.characteristic()
  deg = field.degree()
  t = field(1) if q.is_prime() else field.gen()
  coefs = [0]*deg
  i = -1
  while index > 0:
    rem = index % p
    coefs[i] = rem
    index = (index - rem)//p
    i -= 1
  return sum(field(a)*t**i for a,i in zip(coefs,range(deg-1,-1,-1)))

################################################################################

def field_elements(field):
  r"""
  Return elements of ``field`` in lex order relative to the power basis

  INPUT:

  - ``field`` -- a finite field

  - ``index`` -- optional nonnegative integer that must be less than the field
    cardinality

  EXAMPLES::

    sage: print(list(field_elements(GF(3))))
    [0, 1, 2]
    sage: R.<x> = GF(2)[]
    sage: F.<t> = GF(4,modulus=x^2+x+1)
    sage: print(list(field_elements(F)))
    [0, 1, t, t + 1]
    sage: S.<y> = F[]
    sage: E.<T> = F.extension(y^2 + t*y + 1)
    sage: for u in field_elements(E):
    ....:   print(u)
    0
    1
    t
    t + 1
    T
    T + 1
    T + t
    T + t + 1
    t*T
    t*T + 1
    t*T + t
    t*T + t + 1
    (t + 1)*T
    (t + 1)*T + 1
    (t + 1)*T + t
    (t + 1)*T + t + 1

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  p = field.characteristic()
  # deg = field.degree()
  # t = field(1) if q.is_prime() else field.gen()
  # for coefs in itertools.product(range(p),repeat=deg):
  #   yield sum(field(a)*t**i for a,i in zip(coefs,range(deg-1,-1,-1)))
  # return
  if p == q:
    for c in range(p):
      yield field(c)
  else:
    deg = field.degree()
    t = field.gen()
    B = field.base_ring()
    for coefs in itertools.product(field_elements(B),repeat=deg):
      yield sum(field(a)*t**i for a,i in zip(coefs,range(deg-1,-1,-1)))
  return

################################################################################

def affine_space_point_iterator(field, n, start=None, stop=None):
  r"""
  An iterator over points of `A^n(field)`

  INPUT:

  - ``field`` -- a finite field, `GF(q)`

  - ``n`` -- a positive integer

  - ``start`` -- an optional nonnegative integer

  - ``stop`` -- an optional nonnegative integer

  OUTPUT:

  - iterator over  the `q^n` points  of `A^n(field), interpreted as  n-tuples of
    elements of ``field``; if ``start`` or ``stop`` is specified, then they will
    be used as indices to determine where to start or stop iteration

  NOTE:

  - The order is lexicographic, using the ordering on ``field`` given by the
    function field_element. It agrees with the ordering in itertools.product.

  EXAMPLES::

    sage: R.<x> = GF(2)[]
    sage: F.<t> = GF(4,modulus=x^2+x+1)
    sage: F_list = field_elements(F)
    sage: L = list(affine_space_point_iterator(F,2))
    sage: M = list(itertools.product(F_list,repeat=2))
    sage: L == M
    True

    sage: for a in range(16):
    ....:   for b in range(a,16):
    ....:     L = list(affine_space_point_iterator(F,2,start=a,stop=b))
    ....:     assert L == M[a:b]

  """
  if not field.is_finite():
    raise ValueError('{} is not finite'.format(field))
  q = field.cardinality()
  N = q**n
  FF = field

  if start is None or start < 0:
    start = 0
  if stop is None or stop > N:
    stop = N

  remaining_yield = stop - start
  if remaining_yield <= 0: return  

  # Easy case
  if start == 0:
    for aa in itertools.product(field_elements(FF),repeat=n):
      if remaining_yield == 0: return
      yield aa
      remaining_yield -= 1
    return

  # Now start > 0, so we have to iterate manually.
  coefs = [0]*n
  i = -1
  while start > 0:
    rem = start % q
    coefs[i] = rem
    start = (start - rem)//q
    i -= 1

  FF_list = list(field_elements(FF))
  while remaining_yield > 0:
    yield tuple(FF_list[c] for c in coefs)
    remaining_yield -= 1
    j = n - 1
    while j >= 0:
      if coefs[j] < q-1:
        coefs = coefs[:j] + [coefs[j]+1] + [0]*(n-j-1)
        j = -1
        break
      else:
        j -= 1
  return

################################################################################

def proj_space_point_iterator(field, n, start=None, stop=None):
  r"""
  An iterator over points of P^n(field)

  INPUT:

  - ``field`` -- a finite field, GF(q)

  - ``n`` -- a positive integer

  - ``start`` -- an optional nonnegative integer

  - ``stop`` -- an optional nonnegative integer

  OUTPUT:

  - iterator over the `1 + q + ... + q^n` points of P^n(field), interpreted as
    (n+1)-tuples of elements of field; if ``start`` or ``stop`` is specified,
    then they will be used as indices to determine where to start or stop
    iteration

  NOTE:

  - The order is lexicographic, using the ordering on ``field`` given by the
    function field_element. The first nonzero entry in each tuple will always be
    a 1.

  EXAMPLES::

    sage: R.<x> = GF(2)[]
    sage: F.<t> = GF(4,modulus=x^2+x+1)
    sage: L = list(proj_space_point_iterator(F,4))
    sage: a,b = 12,79
    sage: L[12:79] == list(proj_space_point_iterator(F,4,start=a,stop=b))
    True

  """
  if not field.is_finite():
    raise ValueError('{} is not finite'.format(field))
  q = field.cardinality()
  N = (q**(n+1) - 1) // (q-1)  
  FF = field

  if start is None or start < 0:
    start = 0
  if stop is None or stop > N:
    stop = N

  remaining_yield = stop - start
  if remaining_yield < 0: return
  

  # Easy case
  if start == 0:
    for i in range(n+1):
      for aa in affine_space_point_iterator(FF,i):
        if remaining_yield == 0: return
        yield (FF(0),)*(n-i) + (FF(1),) + aa
        remaining_yield -= 1
    return

  # Now start > 0. Determine which piece of PP^n to start with.
  for i in range(n+1):
    this_tier = (q**i - 1)//(q-1)
    next_tier = (q**(i+1) - 1)//(q-1)
    if start >= next_tier: continue
    this_start = start-this_tier
    this_stop = stop-this_tier
    # print(i,this_start,this_stop,this_tier,next_tier)
    for aa in affine_space_point_iterator(FF,i,start=this_start,stop=this_stop):
      if remaining_yield == 0: return
      yield (FF(0),)*(n-i) + (FF(1),) + aa
      remaining_yield -= 1
  return
  

################################################################################
#################################  UTILITIES  ##################################
################################################################################

def poly_iterator(monomials, start=None, stop=None, projective=False):
  r"""
  Iterator over polynomials in the given monomials

  INPUT:

  - ``monomials`` -- a list of monomials a_0, a_1, ..., a_{n-1} over GF(q),
    where q is prime

  - ``start, stop`` -- optional nonnegative integers used to determine where in 
    the sequence of polynomials to start and stop

  - ``projective`` -- boolean (default: False); whether to use homogeneous
    coordinates for coefficients

  OUTPUT:

  - An iterator that generates the `N` polynomials in the given monomials, where

    * `N = q**len(monomials)` if ``projective`` is False,

    * `N = (q**len(monomials) - 1)//(q-1)` if ``projective`` is True.

    If ``start`` or ``stop`` is specified, it will generated an appropriate
    subsequence of these polynomials.     

  EXAMPLES::

    sage: R.<u> = GF(2)[]
    sage: F.<t> = GF(4,modulus=u^2+u+1)
    sage: polring.<x,y> = F[]
    sage: for pol in poly_iterator([x,y]):
    ....:   print(pol)
    0
    y
    (t)*y
    (t + 1)*y
    x
    x + y
    x + (t)*y
    x + (t + 1)*y
    (t)*x
    (t)*x + y
    (t)*x + (t)*y
    (t)*x + (t + 1)*y
    (t + 1)*x
    (t + 1)*x + y
    (t + 1)*x + (t)*y
    (t + 1)*x + (t + 1)*y

    sage: for pol in poly_iterator([x,y],start=1,stop=4,projective=True):
    ....:  print(pol)
    x
    x + y
    x + (t)*y

  """
  n = len(monomials)
  FF = monomials[0].base_ring()
  q = FF.cardinality()

  N = (q**n - 1) // (q-1) if projective else q**n
  if start is None or start < 0:
    start = 0
  if stop is None or stop > N:
    stop = N

  if projective:
    coefs = proj_space_point_iterator(FF,n-1,start=start,stop=stop)
  else:
    coefs = affine_space_point_iterator(FF,n,start=start,stop=stop)
  for cc in coefs:
    yield sum( c * monomial for c,monomial in zip(cc,monomials))

################################################################################  

def choose_poly(index, monomials, projective=False):
  r"""
  Construct the sum of monomials corresponding to index

  INPUT:

  - ``index`` -- a nonnegative integer

  - ``monomials`` -- a list of monomials a_0, a_1, ..., a_{n-1} over GF(q),
    where q is prime

  - ``projective`` -- boolean (default: False); whether to use homogeneous
    coordinates for coefficients; if False, ``index`` must be less than
    `q^n`. If True, ``index`` must be less than `1 + q + \cdots + q^{n-1}`.

  OUTPUT:

  - The polynomial

    ..math:: \sum_{i = 0}^{n-1} c_i a_i,

    where `(c_0, ..., c_{n-1})` is the ``index``-th point of affine
    `n`-space or projective `n`-space, depending on whether ``projective`` is
    False. 

  NOTE:

  - The order on the coefficient vectors `(c_0, ..., c_{n-1})` agrees with the
    order that points are returned in 

    * affine_space_point_iterator(GF(q),n) if ``projective`` is False

    * proj_space_point_iterator(GF(q),n-1) if ``projective`` is True

  - This yields polynomials in the same order as poly_iterator

  EXAMPLES::

    sage: R.<u> = GF(2)[]
    sage: F.<t> = GF(4,modulus=u^2+u+1)
    sage: polring.<x,y> = F[]
    sage: for i in range(16):
    ....:   print(choose_poly(i,[x,y]))
    0
    y
    (t)*y
    (t + 1)*y
    x
    x + y
    x + (t)*y
    x + (t + 1)*y
    (t)*x
    (t)*x + y
    (t)*x + (t)*y
    (t)*x + (t + 1)*y
    (t + 1)*x
    (t + 1)*x + y
    (t + 1)*x + (t)*y
    (t + 1)*x + (t + 1)*y

    sage: for i in range(5):
    ....:  print(choose_poly(i,[x,y],projective=True))
    y
    x
    x + y
    x + (t)*y
    x + (t + 1)*y

  """
  n = len(monomials)
  FF = monomials[0].base_ring()
  q = FF.cardinality()

  N = (q**n - 1) // (q-1) if projective else q**n
  if index >= N:
    raise ValueError('Index {} out of range'.format(index))

  if projective:
    iter = proj_space_point_iterator(FF,n-1,start=index,stop=index+1)
  else:
    iter = affine_space_point_iterator(FF,n,start=index,stop=index+1)
  coefs = next(iter)
  return sum( c * monomial for c,monomial in zip(coefs,monomials))

################################################################################

_elts_to_inds = {} # key = field, val = dict(keys=field elements, values=indices)
def poly_index(pol, monomials, projective=False):
  r"""
  Inverse of the function choose_poly

  INPUT:

  - ``pol`` -- a polynomial 

  - ``monomials`` -- a list of monomials that includes all of those appearing in
    ``pol``

  - ``projective`` -- bool (defaults: False); whether to use projective
    coordinates for the monomial coefficients

  OUTPUT:

  - The positive integer `i` such that choose_poly(i,monomials,projective) == pol

  NOTE:

  - The function returns the same index for pol as it does for c*pol when
    projective=True

  EXAMPLES::

    sage: R.<x,y,z> = GF(3)[]
    sage: quads = homogeneous_monomials(R,2)
    sage: for i,u in enumerate(poly_iterator(quads,projective=False)):
    ....:   if poly_index(u,quads) != i:
    ....:     raise RuntimeError

    sage: R.<x,y,z> = GF(3)[]
    sage: quads = homogeneous_monomials(R,2)
    sage: for i,u in enumerate(poly_iterator(quads,projective=True)):
    ....:   if poly_index(u,quads,projective=True) != i:
    ....:     raise RuntimeError
    
  """
  F = pol.base_ring()
  elts_to_inds = _elts_to_inds.get(F)
  if elts_to_inds is None:
    elts_to_inds = {}
    for i,u in enumerate(field_elements(F)):
      elts_to_inds[u] = i
    _elts_to_inds[F] = elts_to_inds

  q = F.cardinality()
  n = len(monomials)
  coef_list = [pol.monomial_coefficient(term) for term in monomials]
  
  if not projective:
    index = sum(elts_to_inds[coef_list[n-i-1]] * q**i for i in range(n))
    return index

  # projective case - normalize coefficient list
  if pol == 0:
    raise ValueError('Polynomial must be nonzero in projective case')
  lc_inv = None
  lc_ind = None
  for i in range(n):
    if coef_list[i] != 0:
      if lc_inv is None:
        lc_inv = F(1) / coef_list[i]
        # coef_list[i] = F(1) 
        lc_ind = i
      else:
        coef_list[i] *= lc_inv

  # index = sum( q**i for i in range(n-1-lc_ind))
  index = (q**(n-1-lc_ind) - 1)//(q-1)
  coef_list = coef_list[lc_ind+1:]
  m = len(coef_list)
  index += sum(elts_to_inds[coef_list[m-i-1]] * q**i for i in range(m))
  return index
    
    
################################################################################

def flatten_matrix(M):
  r"""
  Return a flattened tuple of entries of M
  """
  return tuple(flatten([list(r) for r in M.rows()]))

def write_list_of_mats_to_file(mats, filename):
  r"""
  Write a list of matrices in flatted form to a file

  INPUT:

  - ``mats`` -- a list of Matrix objects

  - ``filename`` -- a string or None

  OUTPUT:

  - None

  EFFECT:

  - If ``filename`` is not None, then the entries of each matrix in ``mats``
    will be written to the given filename as a comma-separated list, one list
    per row

  """
  if filename is None: return
  try:
    fp = open(filename,'w')
  except IOError:
    print('Unable to open {} for writing'.format(filename))    
  for g in mats:
    entries = flatten_matrix(g)
    for i,j in enumerate(entries):
      if i==0:
        fp.write('{}'.format(j))
      else:
        fp.write(', {}'.format(j))
    fp.write('\n')
  fp.close()
  return    

################################################################################  

def restore_matrices(field, filename, filename2=None, normalize=False):
  r"""
  Construct a list of square matrices from given file data 

  INPUT:

  - ``field`` -- a coefficient field for the matrices

  - ``filename`` -- a string; the given file must exist and contain one 
    comma-separated line with a square number of entries for each matrix

  - ``filename2`` -- an optional string; if given, the matrices in ``filename``
    will be treated as a subgroup and the matrices in ``filename2`` will be
    treated as right coset representatives. 

  - ``normalize`` -- bool (default: False); if True, multiply all matrices 
    by an appropriate scalar so that the first nonzero entry is 1

  OUTPUT:

  - A list of valid matrices constructed from the given file(s) data. 

  """
  fp = open(filename,'r')
  data = fp.readlines()
  fp.close()

  mats = []
  num_bad_lines = 0
  for line in data:
    s = line.strip().split(',')
    n2 = ZZ(len(s))
    if not n2.is_square():
      continue
    n = n2.sqrt()
    try:
      M = matrix(field,n,n,s)
      mats.append(M)
    except:
      num_bad_lines += 1
      continue
  if num_bad_lines > 0:
    msg = 'WARNING: Found {} bad lines in output'
    print(msg.format(num_bad_lines))

  if filename2 is None:
    return mats

  reps = restore_matrices(field,filename2)
  all_mats = mats[:]
  for h,g in itertools.product(reps,mats):
    A = h*g
    if normalize:
      Acoefs = normalize_point(flatten_matrix(A))
      A = matrix(field,n,n,Acoefs)
    all_mats.append(A)
  # print(len(all_mats))
  return all_mats
  
################################################################################

def compare_matrix_lists(first, second):
  r"""
  Determine if two lists of matrices agree as sets

  INPUT:

  - ``first, second`` -- list of matrices

  OUTPUT:

  - True if the set of elements of ``first`` agrees with that of ``second``

  NOTE:

  - Matrices are not hashable, and so it is slightly annoying to construct sets
    of matrices for comparison

  """  
  # S0 = set(flatten_matrix(M) for M in first)
  # S1 = set(flatten_matrix(M) for M in second)
  return sorted(list(first)) == sorted(list(second))

################################################################################

def int_list_file_to_singular_file(infile, outfile, singular_listname):
  r"""
  Translate a file with integers to a list for Singular consumption

  INPUT:

  - ``infile`` -- string; filename describing a file containing integers, one
    per line

  - ``outfile`` -- string; filename to use for writing Singular output

  - ``singular_listname`` -- string; the name to use for the list to be read by
    Singular

  EFFECT:

  - Write to ``outfile``. First line will be 

    "list singular_listname;\n" 

    For i > 0, line i will be 

    "singular_listname[i] = x;\n"

    where x is the integer on the (i-1)-th line of infile.

  """
  fp = open(infile,'r')
  gp = open(outfile,'w')
  gp.write('list {};\n'.format(singular_listname))
  for i,line in enumerate(fp,1):
    x = line.strip()
    gp.write('{}[{}] = {};\n'.format(singular_listname,i,x))
  gp.close()
  fp.close()

################################################################################

def ideal_dimension(ideal):
  r"""
  Compute dimension of ideal without all the try/except in Sage

  INPUT:

  - ``ideal`` -- an ideal of a polynomial ring

  OUTPUT:

  - The dimension of `I`; i.e., the maximum length of a chain of prime ideals
    containing `I`

  """
  dim = sage.libs.singular.function_factory.ff.dim
  v = MPolynomialIdeal(ideal.ring(),ideal.groebner_basis())
  return ZZ(dim(v, attributes={v:{'isSB':1}}))


################################################################################

def ideal_is_prime(ideal):
  r"""
  Determine if ideal is prime without all the try/except in Sage

  INPUT:

  - ``ideal`` -- an ideal of a polynomial ring over a field

  OUTPUT:

  - Determine if the primary decomposition of ``ideal`` has length 1

  """
  if ideal == 0: return True
  R = ideal.ring()
  if len(R.gens()) == 1:
    # univariate case is fine
    return ideal.is_prime()  
  primdecGTZ =  sage.libs.singular.function_factory.ff.primdec__lib.primdecGTZ
  P = primdecGTZ(ideal)
  V = [(R.ideal(X[0]), R.ideal(X[1])) for X in P]
  if len(V) != 1: return False
  _,Q = V[0]
  return ideal == Q

################################################################################

def evaluate_poly(pol, pt):
  r"""
  Evaluation of pol(pt) because usual polynomial evaluation is slow
  """
  R = pol.parent()
  if len(R.gens()) == 1:
    if pt in R:
      return pol(pt)
    else:
      return pol(*pt)
  val = R.base_ring()(0)
  for term, c in pol.dict().items():
    val += c * prod(a**i for a,i in zip(pt,term) if i != 0)
  return val

################################################################################

def matrix_act_on_poly(g, pol):
  r"""
  Return pol(g(x1,...,xn)) (usual approach is slow)

  INPUT:

  - ``g`` -- an nxn matrix with coefficients in a ring `A`

  - ``pol`` -- a polynomial in A[x1,...,xn]

  OUTPUT:
  
  - The evaluation pol(*(g*xx)), where xx = (x1,...,xn) is the vector of
    generators of the parent of pol 

  """
  polring = pol.parent()
  XX = polring.gens()
  if len(XX) != 5:
    new_vars = [sum(c*u for c,u in zip(row,XX) if c != 0) for row in g.rows()]
  else:
    # This is faster when it applies
    v,w,x,y,z = XX
    new_vars = [r[0]*v + r[1]*w + r[2]*x + r[3]*y + r[4]*z for r in g.rows()]
  return evaluate_poly(pol,new_vars)
       
  
################################################################################

def compare_quadric_files(polring, file1, file2):
  r"""
  Give simple stats comparing pairs of quadrics in file1, file2

  INPUT:

  - ``polring`` -- polynomial ring in 5 variables v,w,x,y,z

  - ``file1, file2`` -- strings; each file must contain pairs of quadratic
    forms, one pair on each line, in the form "form1, form2\n".

  OUTPUT:

  - Set of elements in ``file1``

  - Set of elements in ``file2``

  EFFECT:

  - Print various details comparing file1, file2

  """
  R = polring

  data1 = []
  fp = open(file1,'r')
  for line in fp:
    f,g = line.strip().split(',')
    data1.append((R(f),R(g)))
  fp.close()

  data2 = []
  fp = open(file2,'r')
  for line in fp:
    f,g = line.strip().split(',')
    data2.append((R(f),R(g)))
  fp.close()

  S1, S2 = set(data1), set(data2)

  S1_intersect_S2 = S1.intersection(S2)
  S1_minus_S2 = S1.difference(S2)
  S2_minus_S1 = S2.difference(S1)

  print('')
  print('File 1 = {}'.format(file1))
  print('File 2 = {}'.format(file2))
  print('')
  print('                 {:>10}{:>10}'.format('File 1','File 2'))
  print('        elements:{:10}{:10}'.format(len(data1),len(data2)))
  print(' unique elements:{:10}{:10}'.format(len(S1),len(S2)))
  print('only in this set:{:10}{:10}'.format(len(S1_minus_S2),len(S2_minus_S1)))
  print('    intersection:{:10}'.format(len(S1_intersect_S2)))
  return S1,S2

################################################################################

def start_and_stop_work(total_work, num_jobs, job):
  r"""
  Compute start and stop index for this job's work, for parallelizing

  INPUT:

  - ``total_work`` -- a positive integer

  - ``num_jobs`` -- total number of jobs that work is divided into

  - ``job`` -- index of this job; must be in range(num_jobs)

  OUTPUT:

  - ``start, stop`` -- beginning and ending index (in pythonic sense) for this
    job's work

  NOTE:

  - Write `total_work = q*num_jobs + r`. The first r jobs get `q+1` work, while
    the remaining jobs get `q` work. 
  
  """
  if job < 0 or job >= num_jobs:
    raise ValueError('job = {} is invalid'.format(job))
  q,r = ZZ(total_work).quo_rem(num_jobs)
  if job < r:
    start = (q+1)*job
    stop = (q+1)*(job+1)
  else:
    start = (q+1)*r + q*(job-r)
    stop = (q+1)*r + q*(job-r+1)
  return start, stop

################################################################################

def cat_files(num_files, infile_prefix, outfile):
  r"""
  Concatenate files in correct order

  INPUT:

  - ``num_files`` -- number of files to concatenate

  - ``infile_prefix`` -- path and file prefix

  - ``outfile`` -- filename into which to put concatenated data

  EFFECT:
  
  - Take the files 'infile_prefix.0', ..., 'infile_prefix.r' with 
    `r = num_files-1` and concatenate them into ``outfile`` in 
    this order.

  """
  gp = open(outfile,'w')
  for i in range(num_files):
    fp = open(infile_prefix + '.{}'.format(i),'r')
    for line in fp:
      gp.write(line)
    fp.close()
  gp.close()

    
################################################################################
#########################  PROJECTIVE SCHEME FUNCTIONS  ########################
################################################################################

def normalize_point(point):
  r"""
  Return the same projective point, but with first nonzero coordinate 1

  INPUT:

  - ``point`` -- a tuple of field elements `(a0, ..., an)`, not all zero

  OUTPUT:

  - The tuple `(a0/u, ..., an/u)`, where u is the left-most nonzero enetry of
    the original point

  """
  n = len(point)
  u = None
  for i in range(n):
    if point[i] != 0:
      u = point[i]
      break
  if u is None:
    raise ValueError
  return tuple(a/u for a in point)

################################################################################


def proj_point_search(homogeneous_poly_or_polys, points=None, status=False, **kwargs):
  r"""
  Search for rational solutions on a projective scheme

  INPUT:

  - ``homogeneous_poly_or_polys`` -- a single homogeneous polynomial or an
    iterator of such in FF[x0,...,xn]

  - ``points`` -- optional list of projective points to use as a search space;
    if not specified, then FF must be finite

  - ``status`` -- boolean (default: False); if True, print status using the
    progress module

  - Remaining arguments are passed to progress.Progress

  OUTPUT:

  - a complete list of FF-valued projective points satisfying all of the given
    polynomials; if ``points`` is specified, then only those points will be
    searched.

  NOTE:

  - We do not check if the input polynomials are homogeneous.

  """
  if isinstance(homogeneous_poly_or_polys,(list,tuple,set)):
    polys = homogeneous_poly_or_polys
  else:
    polys = [homogeneous_poly_or_polys]
  R = polys[0].parent()
  FF = R.base_ring()
  n = R.ngens()
  
  if points is not None:
    search_space = points
    search_size = len(points)
  else:
    search_space = proj_space_point_iterator(FF,n-1)
    q = FF.cardinality()    
    search_size = (q**n - 1) / (q-1)

  FF_pts = []
  if status: prog = progress.Progress(search_size,**kwargs)
  for pt in search_space:
    if status: prog()
    if all(evaluate_poly(f,pt)==0 for f in polys): FF_pts.append(pt)
  if status: prog.finalize()
  return FF_pts

################################################################################

def hypersurface_point_search(poly, status=False, **kwargs):
  r"""
  Search for rational solutions on a projective hypersurface

  INPUT:

  - ``poly`` -- a single homogeneous polynomial

  - ``status`` -- boolean (default: False); if True, print status using the
    progress module

  - Remaining arguments are passed to progress.Progress

  OUTPUT:

  - a complete list of FF-valued projective points satisfying this homogeneous
    polynomial

  EXAMPLES::

    sage: F.<t> = GF(4)
    sage: R.<x,y,z> = F[]
    sage: f = x*y + z^2
    sage: pts = hypersurface_point_search(f)
    sage: other_pts = proj_point_search(f)
    sage: set(pts) == set(other_pts)
    True
    sage: S.<T> = F[]
    sage: minpol = T^3 + T^2 + t*T + 1
    sage: E.<u> = F.extension(minpol)
    sage: Epts = hypersurface_point_search(f.change_ring(E))
    sage: Eother_pts = proj_point_search(f.change_ring(E))
    sage: set(Epts) == set(Eother_pts)
    True

  """
  R = poly.parent()
  n = len(R.gens())
  FF = R.base_ring()
  q = FF.cardinality()

  if n == 1:
    return list(proj_point_search(poly))

  S = PolynomialRing(FF,names=('T',))
  T = S.gen()
  pts = []
  easy_pt = (FF(0),)*(n-1) + (FF(1),)
  if evaluate_poly(poly,easy_pt) == 0:
    pts.append(easy_pt)
  if status: prog = progress.Progress((q**(n-1) - 1)//(q-1))
  for aa in proj_space_point_iterator(FF,n-2):
    if status: prog()
    subs = aa + (T,)
    simple_pol = S(evaluate_poly(poly,subs))
    if simple_pol == 0:
      for r in field_elements(FF):
        pts.append(aa + (r,))
    else:
      for r in simple_pol.roots(multiplicities=False):
        pts.append(aa + (r,))
  if status: prog.finalize()
  return pts  

################################################################################

def proj_variety_has_n_points(homogeneous_polys, n, points=None):
  r"""
  Determine if variety cut out by ``homogeneous_polys` has ``n`` rational points

  INPUT:

  - ``homogeneous_polys`` -- a list of homogeneous polynomials in FF[x0,...,xm]

  - ``n`` -- a nonnegative integer or an iterable of such

  - ``points`` -- optional list of projective points to use as a search space;
    if not specified, then FF must be finite

  OUTPUT:

  - True if the variety cut out by ``homogeneous_polys`` has ``n`` rational points;
    if ``points`` is specified, then only those points will be
    searched.

  NOTE:

  - We do not check if the input polynomials are homogeneous.

  EXAMPLES::

    sage: F = GF(3)
    sage: R.<x,y,z> = F[]
    sage: len(proj_point_search(x^2 + y*z))
    4
    sage: proj_variety_has_n_points([x^2+y*z],4)
    True
    sage: proj_variety_has_n_points([x^2+y*z],5)
    False
    sage: proj_variety_has_n_points([x^2+y*z],3)
    False
    sage: proj_variety_has_n_points([x^2+y*z],[1,4,9])
    True
    sage: proj_variety_has_n_points([x^2+y*z],[4,9])
    True
    sage: proj_variety_has_n_points([x^2+y*z],[1,4])
    True

  """
  polys = homogeneous_polys
  R = polys[0].parent()
  FF = R.base_ring()
  m = R.ngens()
  
  if points is not None:
    search_space = points
  else:
    search_space = proj_space_point_iterator(FF,m-1)

  FF_pts = []
  if isinstance(n,(tuple,list,set)):
    max_num_pts = max(n)
    num_pts = set(n)
  else:
    max_num_pts = n
    num_pts = set([n])
  for pt in search_space:
    if all(evaluate_poly(f,pt)==0 for f in polys): FF_pts.append(pt)
    if len(FF_pts) > max_num_pts: return False
  return len(FF_pts) in num_pts

################################################################################

def hypersurface_is_nonsingular(poly):
  r"""
  Determine if the hypersurface cut out by poly is nonsingular

  INPUT:

  - ``poly`` -- a homogeneous polynomial in `k[x0,...,xn]`

  OUTPUT:

  - True if `poly` is nonsingular; False otherwise

  """
  R = poly.parent()
  I = [poly]
  for x in R.gens():
    I.append(poly.derivative(x))
  sing_locus_dim = ideal_dimension(R.ideal(I))
  return sing_locus_dim == 0
  
################################################################################

def hypersurface_singularities(poly, points=None):
  r"""
  Determine the rational singularities on a projective hypersurface
  
  INPUT:

  - ``poly`` -- a homogeneous polynomial

  - ``points`` -- optional list of projective points to be passed to
    proj_point_search

  OUTPUT:

  - a complete list of rational singularities of the hypersurface

  """
  if not poly.is_homogeneous():
    raise ValueError('{} is not homogeneous'.format(poly))

  R = poly.parent()
  I = [poly] + [poly.derivative(t) for t in R.gens()]
  return proj_point_search(I)

################################################################################

def homogeneous_monomials(polring, degree):
  r"""
  Return a list of monomials in polring of given degree

  INPUT:

  - ``polring`` -- a polynomial ring in 1 or more variables

  - ``degree`` -- a positive integer

  OUTPUT:

  - a complete list of monomials in standard lexicographic ordering of given
    degree

  """
  monomials = []
  for xx in itertools.combinations_with_replacement(polring.gens(),degree):
    monomials.append(prod(xx))
  return monomials
  
################################################################################

def restore_complete_intersections(filename, polring_or_form):
  r"""
  Construct a list of complete intersections from a file

  INPUT:

  - ``filename`` -- string; name of file in which data resides, in the format of
    one comma-separated sequence of homogeneous polynomials on a line
    representing a single complete intersection

  - ``polring_or_form`` -- either a polynomial ring R = k[x0,...,xn] or a homogenous
    form in this ring; all data in ``filename`` must lie in this ring

  OUTPUT:

  - a list of tuples (f1, ..., fm), one for each complete intersection in the
    data file. If a form is specified for the second argument, it will occur as
    the first entry in each tuple of the output list.

  """
  if hasattr(polring_or_form,'is_homogeneous'):
    R = polring_or_form.parent()
    f = (polring_or_form,)
  else:
    R = polring_or_form
    f = ()

  schemes = []
  fp = open(filename,'r')
  L = fp.readlines()
  prog = progress.Progress(len(L),steps_until_print=1024)
  for line in L:
    strs = line.strip().split(',')
    schemes.append(f + tuple(R(u) for u in strs))
    prog()
  prog.finalize()
  fp.close()
  return schemes

################################################################################

def radical_locus(form):
  r"""
  Compute the rational points of the radical locus of a quadratic form

  INPUT:

  - ``form`` -- a quadratic form over GF(q)

  OUTPUT:

  - The list of projective radical points of the quadratic form; i.e., points of
    projective space at which all partials of the form vanish

  """
  if not form.is_homogeneous():
    raise ValueError('{} is not a quadratic form'.format(form))
  R = form.parent()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()
  n = R.ngens()-1 # dimension of ambient projective space
  rows = []
  for t in R.gens():
    deriv = form.derivative(t)
    if deriv == 0: continue
    v = sum( c * vector(FF,exps) for exps,c in deriv.dict().items())
    rows.append(v)
  M = matrix(FF,rows)
  BB = M.right_kernel().basis()
  pts = []
  for cc in proj_space_point_iterator(FF,len(BB)-1):
    pt = sum(c * b for c,b in zip(cc,BB))
    pts.append(tuple(pt))
  return pts

################################################################################

def quadric_singularities(form):
  r"""
  Compute the rational points of the singular locus of a quadratic form

  INPUT:

  - ``form`` -- a quadratic form over GF(q)

  OUTPUT:

  - The list of projective singular points of the quadratic form

  """
  rad = radical_locus(form)
  pts = [pt for pt in rad if evaluate_poly(form,pt) == 0]
  return pts
  
################################################################################


def quadric_singularity_dimension(polring_or_form, filename=None,
                                     projective=True, status=False,
                                     num_jobs=1, job=0, **kwargs):
  r"""
  Compute singularity dimension of quadrics

  INPUT:

  - ``polring_or_form`` -- a polynomial ring R = GF(q)[x0,...,xn] or a quadratic
    form in R

  - ``filename`` -- optional string to use for writing to file

  - ``projective`` -- bool (default: True); if polring_or_form is a polynomial
    ring and projective is True, only information about quadrics modulo scalars
    will be returned/printed. (The False case is for backward compatibility.)

  - ``status`` -- boolean (default: False); if True, print status using progress
    module

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - Remaining keyword arguments are passed to progress.Progress

  OUTPUT:

  - If ``polring_or_form`` is a quadratic form, return the (projective)
    dimension of its singularity locus. We use -1 for the dimension of an empty
    singular locus, and n for the dimension of V(0). Ignore ``filename``
    argument in this case.

  - If ``polring_or_form`` is a polynomial ring:

    * If filename is None, return a list whose i-th entry is the dimension of
      the (projective) singular locus of choose_poly(i, quads), where quads is a
      list of all quadratic monomials in R as output by homogeneous_monomials.  We
      use -1 for the dimension of an emtpy singular locus, and n for the
      dimension of V(0).

    * If filename is not None, no output is given. Instead, the i-th line of
      filename is the dimension of the singular locus of choose_poly(i,quads))
      as above.

  """  
  if hasattr(polring_or_form,'is_homogeneous'):
    Q = polring_or_form
    R = Q.parent()
    if Q != 0:
      if not Q.is_homogeneous() or Q.degree() !=2:
        raise ValueError('{} is not a quadratic form'.format(Q))
  else:
    Q = None
    R = polring_or_form
  
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()
  n = R.ngens()-1 # dimension of ambient projective space

  def sing_locus_dim(pol):
    r""" Compute Singular locus dimension """
    if pol == 0: return n
    rows = []
    for t in R.gens():
      deriv = pol.derivative(t)
      if deriv == 0: continue
      v = sum( c * vector(FF,exps) for exps,c in deriv.dict().items())
      rows.append(v)
    if len(rows) == 0: # pol is a square and characteristic is 2
      return n-1
    M = matrix(FF,rows)
    BB = M.right_kernel().basis()
    # The radical and singular locus have same dimension when q is odd.
    # They have same dimension or are off by 1 When q is even.
    if q % 2 == 1: return len(BB)-1
    for cc in proj_space_point_iterator(FF,len(BB)-1):
      pt = sum(c * b for c,b in zip(cc,BB))
      if evaluate_poly(pol,pt) != 0: return len(BB) - 2 
    return len(BB) - 1

  # Handle case where we only have one form
  if Q is not None:
    return sing_locus_dim(Q)

  # Prepare to process all forms
  quads = homogeneous_monomials(R,2)
  if projective:
    num_pols = (q**len(quads) - 1)//(q-1)
  else:
    num_pols = q**len(quads)

  start, stop = start_and_stop_work(num_pols,num_jobs,job)
  print('Computing singular locus dimension for {} quadratic forms'.format(stop-start))

  # Set up output type
  if filename is None:
    dims = [0]*num_pols
    fp = None
  else:
    dims = [0]*1024 # for batching output
    fp = open(filename,'w')

  def batch_write(fp,L,num_to_write):
    msg = ''
    for j,val in zip(range(num_to_write),L):
      msg += '{}\n'.format(val)
    fp.write(msg)
        
  def write_dim(fp,L,index,val):
    r""" Add element to list or write to file"""
    if fp is None:
      L[index] = val
    else:
      index_mod = index % 1024
      L[index_mod] = val
      if index_mod == 1023:
        batch_write(fp,L,1024)

  if status: prog = progress.Progress(stop-start, **kwargs)
  for i,pol in enumerate(poly_iterator(quads,projective=projective,start=start,stop=stop)):
    dim = sing_locus_dim(pol)
    write_dim(fp,dims,i,dim)
    if status: prog()
  if status: prog.finalize()

  if filename is None:
    return dims
  else:
    batch_write(fp,dims,(stop-start)%1024) # Finish batched writing
    fp.close()
    return

################################################################################  

def quadric_point_count(polring_or_form, filename=None,
                          projective=True, status=False,
                          singularity_dimension_data=None, 
                          num_jobs=1, job=0, **kwargs):
  r"""
  Compute number of points on quadrics

  INPUT:

  - ``polring_or_form`` -- a polynomial ring R = GF(q)[x0,...,xn] or a quadratic
    form in R

  - ``filename`` -- optional string to use for writing to file

  - ``projective`` -- bool (default: True); if polring_or_form is a polynomial
    ring and projective is True, only information about quadrics modulo scalars
    will be returned/printed. (The False case is for backward compatibility.)

  - ``status`` -- boolean (default: False); if True, print status using progress
    module

  - ``singularity_dimension_data`` -- optional filename in which the output of
    quadric_singularity_dimension has been written

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - Remaining keyword arguments are passed to progress.Progress

  OUTPUT:

  - If ``polring_or_form`` is a quadratic form, return the number of projective
    points on it.  Ignore all other arguments in this case.

  - If ``polring_or_form`` is a polynomial ring:

    * If filename is None, return a list whose i-th entry is the number of projective points
      on the quadric cut out by choose_poly(i, quads), where quads is a
      list of all quadratic monomials in R as output by homogeneous_monomials.  We

    * If filename is not None, no output is given. Instead, the i-th line of
      filename is the point count for choose_poly(i,quads)) as above.

  NOTE:

  - If ``filename`` is not None, output it batched into 1024 lines at a time.

  - If ``singularity_dimension_data`` is not None, and if ``polring`` has 5
    variables, singularity info will be used to expedite the calculation of the
    point count using the following facts:
   
    * If Q is equivalent to vw + x^2, there are q^3 + q^2 + q + 1 rational points. 
      This happens when the singularity dimension is 1.

    * If Q is equivalent to vw + xy + z^2, there are q^3 + q^2 + q + 1 rational
      points. This happens when the singularity dimension is -1.

    * In the remaining cases the singularity dimension is 0. If Q is equivalent
      to vw + xy, then it has q^3 + 2q^2 + q + 1 rational points, while if Q is
      equivalent to vw + N(x,y), then it has q^3 + q + 1 ratinal points.

  """  
  if hasattr(polring_or_form,'is_homogeneous'):
    Q = polring_or_form
    R = Q.parent()
    if Q != 0:
      if not Q.is_homogeneous() or Q.degree() !=2:
        raise ValueError('{} is not a quadratic form'.format(Q))
  else:
    Q = None
    R = polring_or_form
  
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()

  # Handle case where we only have one form
  if Q is not None:
    return len(proj_point_search(Q,status=status))

  # Prepare to process all forms
  quads = homogeneous_monomials(R,2)
  if projective:
    num_pols = (q**len(quads) - 1)//(q-1)
  else:
    num_pols = q**len(quads)
  start, stop = start_and_stop_work(num_pols,num_jobs,job)
  if num_jobs > 1:
    print('Job = {}, Start = {}, Stop = {}'.format(job,start,stop))
  print('Computing number of points on {} quadrics'.format(stop-start))

  if singularity_dimension_data is not None and len(R.gens())==5:
    sp = open(singularity_dimension_data,'r')
    sing_dims = [0]*(stop-start)
    for i,line in enumerate(sp):
      if i < start: continue
      if i >= stop: break
      sing_dims[i-start] = int(line)
    sp.close()
    print('Finished loading singularity info.')
  else:
    sing_dims = None

  # Set up output type
  if filename is None:
    num_pts = [0]*num_pols
    fp = None
  else:
    num_pts = [0]*1024 # for batching output
    fp = open(filename,'w')

  def batch_write(fp,L,num_to_write):
    msg = ''
    for j,val in zip(range(num_to_write),L):
      msg += '{}\n'.format(val)
    fp.write(msg)

  def write_num_pts(fp,L,index,val):
    r""" Add element to list or write to file"""
    if fp is None:
      L[index] = val
    else:
      index_mod = index % 1024
      L[index_mod] = val
      if index_mod == 1023:
        batch_write(fp,L,1024)

  if status: prog = progress.Progress(stop-start, **kwargs)
  for i,pol in enumerate(poly_iterator(quads,projective=projective,start=start,stop=stop)):
    if sing_dims is None:
      val = len(proj_point_search(pol))
      continue
    dim = sing_dims[i]
    if dim == 4:
      val = (q**5 - 1)//(q-1)
    elif dim in [-1,1,3]:
      val = q**3 + q**2 + q + 1
    elif dim == 2:
      val = 0
      thresh = q**2 + q + 1
      for aa in proj_space_point_iterator(FF,4):
        if evaluate_poly(pol,aa)==0: val += 1
        if val <= thresh: continue
        val = 2*q**3 + q**2 + q + 1
        break                   
    else:
      val = 0
      thresh = q**3 + q + 1
      for aa in proj_space_point_iterator(FF,4):
        if evaluate_poly(pol,aa)==0: val += 1
        if val <= thresh: continue
        val = q**3 + 2*q**2 + q + 1
        break             
    write_num_pts(fp,num_pts,i,val)    
    if status: prog()
  if status: prog.finalize()

  if filename is None:
    return num_pts
  else:
    batch_write(fp,num_pts,(stop-start)%1024) # Finish batched writing
    fp.close()
    return

################################################################################

def allowed_invariant_quadric_mask(polring, invariants_file, outfile,
                                      singularity_dimension_file, point_count_file):
  r"""
  Create a mask to determine when a quadric surface is allowable

  - ``polring`` -- The polynomial ring from which the quadratic forms arise

  - ``invariants_file`` -- file giving allowed invariants; each line is of the
    form 'a, b\n', where the pairs of integers `(a,b)` correspond to allowed
    values of (dimension of singular locus, number of rational points)

  - ``singularity_dimension_file`` -- filename giving classification of quadric
    singularities as written by quadric_singularity_dimension (with
    projective=True flag set)

  - ``point_count_file`` -- filename giving the number of points on each quadric
    hypersurface as written by quadric_point_count (with projective=True flag
    set)

  - ``outfile`` -- file in which to write data in format to be read by Sage

  EFFECT:

  - Write a single long line of (ascii) 0's and 1's to ``outfile`` of length
    `N`, where `N` is the common length (in lines) of the input files. If `a\n`
    and `b\n` are the `i`th lines of the files ``singularity_dimension_file``
    and ``point_count_file``, respectively, then the `i`th binary digit of the
    ``outfile`` will be 1 if and only if `(a,b)` is in the list
    ``allowed_invariants``.

  NOTE:

  - This takes around 20-30s for q = 3 and around 15m for q = 4.

  """
  sdp = open(singularity_dimension_file,'r')
  pcp = open(point_count_file,'r')

  allowed_invariants = set()
  fp = open(invariants_file,'r')
  for line in fp:
    a,b = [int(t) for t in line.strip().split(',')]
    allowed_invariants.add((a,b))
  fp.close()
  print('Allowing invariants {} for quadratic_forms.'.format(list(allowed_invariants)))
  
  fp = open(outfile,'w')

  R = polring
  FF = R.base_ring()
  q = FF.cardinality()
    
  quads = homogeneous_monomials(R,2)
  num_pols = (q**len(quads) - 1)//(q-1)
  allowed_polys = ['0']*num_pols # 1 means allowed
  num_allowed = 0
  print('Constructing array of allowed quadrics.')
  prog = progress.Progress(num_pols,steps_until_print=int(num_pols / 100))
  for i in range(num_pols):
    prog()
    a,b = int(sdp.readline()), int(pcp.readline())
    if (a,b) in allowed_invariants:
      allowed_polys[i] = '1'
      num_allowed += 1
  prog.finalize()

  print('Found {} allowed quadrics'.format(num_allowed))

  # Clean up files
  sdp.close()
  pcp.close()

  s = ''.join(allowed_polys) + '\n'
  fp.write(s)
  fp.close()
  return  

################################################################################

def complete_intersection_tangent_line(pols, point):
  r"""
  Return a point on the tangent line to a complete intersection
  
  INPUT:

  - ``pols`` -- a list of homogeneous polynomials in GF(q)[x0,...,xn] whose intersection
    is a curve `C` in PP^n
  
  - ``point`` -- a smooth point in PP^n(GF(q^m)) satisfying all polynomials in pols

  OUTPUT:
  
  - A point of PP^n(GF(q^m)), distinct from ``point``, lying on the tangent line to `C`

  """
  R = pols[0].parent()
  FF = R.base_ring()
  pnt = normalize_point(point)

  rows = []
  for pol in pols:
    rows.append([pol.derivative(u)(*pnt) for u in R.gens()])
  M = matrix(rows)
  BB = M.right_kernel().basis()
  if len(BB) != 2:
    raise ValueError('{} is not a smooth point'.format(point))
  for u,v in proj_space_point_iterator(FF,1):
    new_pnt = normalize_point(u*vector(BB[0]) + v*vector(BB[1]))
    if new_pnt != pnt: return new_pnt
  raise RuntimeError('No new point found on the tangent line')

################################################################################

def point_reps_of_bounded_degree(pol, degree):
  r"""
  Return minimum information to construct rational points of bounded degree

  INPUT:

  - ``pol`` -- a homogeneous polynomial in GF(q)[x0,...,xn]

  OUTPUT:

  - a dict whose keys are positive integers `1, 2, ..., degree` and whose i-th
    value is a complete list of pairwise non-conjugate projective points of
    exact degree-i satisfying pol

  NOTE:

  - I had to code this a little weird because there's a bug in finite field
    coercion. I was finding that elements of GF(4) and GF(16) that Sage declares
    are equal do not give equality when stuck into a tuple. I'm not sure what
    that's about.

  """
  if not pol.is_homogeneous():
    raise ValueError
  R = pol.parent()
  FF = R.base_ring()
  q = FF.cardinality()

  if degree == 0:
    return {}
  
  pts = {}
  pts[1] = proj_point_search(pol)
  pts_seen = set(pts[1])  
  for d in range(2,degree+1):
    print('Computing degree-{} points of {} = 0'.format(d,pol))
    pts[d] = []
    ipts = proj_point_search(pol.change_ring(GF(q**d)))
    for pt in ipts:
      if pt in pts_seen: continue
      orbit = set()
      for i in range(d):
        orbit.add(tuple(u**(q**i) for u in pt))
      if len(orbit) < d: continue
      pts_seen.update(orbit)
      pts[d].append(pt)
  return pts

################################################################################
##############################  ORTHOGONAL GROUPS  #############################
################################################################################

def naive_orthogonal_group(quadratic_form, projective=True, filename=None,
                             num_jobs=1, job=0, work_estimate=False, **kwargs):
  r"""
  Compute all elements of the orthogonal group of the given quadratic form over GF(q)

  INPUT:

  - ``quadratic_form`` -- a quadratic form in GF(q)[x1,x2,...,xn].

  - ``projective`` -- boolean (default: True); return only classes of elements
    modulo the action of scalar multiplication

  - ``filename`` -- optional file in which to write the output

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - ``work_estimate`` -- bool (default: False); If True, don't do any work, but
    instead return how many total matrices will be checked in the computation.

  - Additional keyword arguments are passed to progress.Progress.

  OUTPUT:

  - a list of all elements of O(quadratic_form) \subset GL_n(GF(q)), or
    PO(quadratic_form) \subset PGL_n(GF(q)) if projective is True

  EFFECT:

  - If ``filename`` is not None, then the entries of each matrix in the output
    will be written to the given filename as a comma-separated list, one list
    per row

  NOTE:

  - We search using matrices that start with a 1 in the earliest nonzero
    position. If Abar is such a matrix, and A = c*Abar is an element of the
    orthogonal group, then Q(x) = Q(A*xx) = Q(c*Abar*x) = c^2 * Q(Abar*x). 
    Thus, we need to check whether Q(Abar*x) agrees with Q(x) up to
    multiplication by a square, and then we need to divide by the square-root.

  """
  Q = quadratic_form
  if Q == 0:
    raise ValueError('Q = 0 is not a quadratic form')
  R = quadratic_form.parent()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()  
  grp_elts = []
  n = len(R.gens())
  XX = vector(R.gens())

  # divvy up the search work
  search_size = (q**(n**2) - 1)//(q-1)
  start,stop = start_and_stop_work(search_size, num_jobs, job)
  N = stop-start
  if work_estimate:
    return N

  if num_jobs > 1:
    print('Start index = {}, Stop index = {}'.format(start,stop))
  search_space = proj_space_point_iterator(FF,n**2-1,start=start,stop=stop)
  scalars = {}
  for c in FF:
    if c == 0: continue
    if c**2 in scalars: continue
    scalars[c**2] = c
    
  print('Searching {} = 2^{:.2f} matrices'.format(N,float(log(N,2))))
  prog = progress.Progress(N,**kwargs)    
  for aa in search_space:
    prog()
    A = matrix(FF,n,n,aa)
    if not A.is_invertible(): continue
    for cc,c in scalars.items():
      if matrix_act_on_poly(A,Q) == cc*Q:
        g = (1/c)*A
        grp_elts.append(g)
        if not projective and q % 2 != 0:
          grp_elts.append(-g)
        break
  prog.finalize()
  write_list_of_mats_to_file(grp_elts,filename)  
  return grp_elts

################################################################################

def orthogonal_group(quadratic_form, projective=True, filename=None,
                       num_jobs=1, job=0, work_estimate=False, **kwargs):
  r"""
  Compute all elements of the orthogonal group of the given quadratic form over GF(q)

  INPUT:

  - ``quadratic_form`` -- a quadratic form in GF(q)[x1,x2,...,xn]

  - ``projective`` -- boolean (default: True); return only classes of elements
    modulo the action of scalar multiplication

  - ``filename`` -- optional file in which to write the output

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - ``work_estimate`` -- bool (default: False); If True, don't do any work, but
    instead return how many total matrices will be checked in the computation.

  - Additional keyword arguments are passed to progress.Progress.

  OUTPUT:

  - a list of all elements of O(quadratic_form) \subset GL_n(GF(q)), or
    PO(quadratic_form) \subset PGL_n(GF(q)) if projective is True

  EFFECT:

  - If ``filename`` is not None, then the entries of each matrix in the output
    will be written to the given filename as a comma-separated list, one list
    per row

  NOTE:

  - We search using matrices that start with a 1 in the earliest nonzero
    position. If Abar is such a matrix, and A = c*Abar is an element of the
    orthogonal group, then Q(x) = Q(A*xx) = Q(c*Abar*x) = c^2 * Q(Abar*x). 
    Thus, we need to check whether Q(Abar*x) agrees with Q(x) up to
    multiplication by a square, and then we need to divide by the square-root.

  """
  Q = quadratic_form
  if Q == 0:
    raise ValueError('Q = 0 is not a quadratic form')
  R = quadratic_form.parent()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()  
  grp_elts = []
  n = len(R.gens())
  XX = vector(R.gens())

  # Use more clever strategy. Q is nontrivial and n > 1.
  QQ = proj_point_search(Q)
  RR = proj_point_search([Q.derivative(t) for t in XX])
  SS = [a for a in RR if a in QQ]
  QQ_minus_RR = [pt for pt in QQ if pt not in RR]
  RR_minus_SS = [pt for pt in RR if pt not in SS]
  targets = []
  base_pts = []
  for L in [QQ_minus_RR,RR_minus_SS,SS]:
    if len(L) > 0:
      targets.append(L)
      base_pts.append(L[0])
  num_sets = len(targets)
  if num_sets == 0:
    # default back to naive approach
    kwds = kwargs.copy()
    kwds['projective'] = projective
    kwds['num_jobs'] = num_jobs
    kwds['job'] = job
    kwds['work_estimate'] = work_estimate
    kwds['filename'] = filename
    return naive_orthogonal_group(Q,**kwds)
  num_targets = prod(len(L) for L in targets)
  N = n**2 - (n-1)*num_sets

  print('Using transitivity on {} elements.'.format(num_targets))
  print('Searching vector spaces of dimension {}.'.format(N))
  
  # divvy up the search work
  search_size = (q**N - 1)//(q-1)
  start,stop = start_and_stop_work(search_size, num_jobs, job)
  if work_estimate:
    return num_targets * (stop-start)

  if num_jobs > 1:
    print('Start index = {}, Stop index = {}'.format(start,stop))

  search_space = lambda : proj_space_point_iterator(FF,N-1,start=start,stop=stop)
  scalars = {}
  for c in FF:
    if c == 0: continue
    if c**2 in scalars: continue
    scalars[c**2] = c
    
  # For each pt in each of the sets in targets, we get a
  # system of linear equations that must be satisfied.
  # There are n^2 + num_sets variables and n*num_sets equations.
  # We search over the remaining n^2 - (n-1)*num_sets dimensions.
  prog = progress.Progress(num_targets*(stop-start), **kwargs)
  for pp in itertools.product(*targets):
    # Find matrices M satisfying M*p0 = u*p1 for p0 in basepoints, p1 in pp
    entries = {}
    num_rows = 0
    for k,p0,p1 in zip(range(num_sets),base_pts, pp):
      for i in range(n):
        for j in range(n):
          entries[(num_rows,i*n + j)] = p0[j]
        entries[(num_rows,n**2 + k)] = -p1[i]
        num_rows += 1
    M = matrix(FF,entries)
    BB = M.right_kernel().basis()
    if len(BB) != N:
      raise RuntimeError('Expected kernel dimension to be {}, but got {}'.format(N,len(BB)))
    # prog.profile('linear algebra')
    for aa in search_space():
      prog()
      entries = sum(u*b for u,b in zip(aa,BB))
      A = matrix(FF,n,n,entries[:-num_sets])
      # prog.profile('matrix construction')
      if not A.is_invertible():
        # prog.profile('invertibility testing')
        continue
      # prog.profile('invertibility testing')      
      for cc,c in scalars.items():
        if matrix_act_on_poly(A,Q) == cc*Q:
          g = (1/c)*A
          grp_elts.append(g)
          if not projective and q % 2 != 0:
            grp_elts.append(-g)
          break
      # prog.profile('orthogonal group testing')
  prog.finalize()    
  write_list_of_mats_to_file(grp_elts,filename)
  return grp_elts

################################################################################

def orthogonal_transitive_set(quadratic_form):
  r"""
  Compute the set ``Y`` from Theorem A.8 of Faber/Grantham

  INPUT:

  - ``quadratic_form`` -- a quadratic form in GF(q)[x0,...,xn]

  OUTPUT:

  - A list of tuples of points of projecive `n`-space that describes the product
    of all of the nonempty sets among `(\cQ - \cS)`, `(\cR - \cS)`, and `\cS`.

  """
  Q = quadratic_form
  R = Q.parent()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  XX = vector(R.gens())  

  QQ = proj_point_search(Q)
  RR = proj_point_search([Q.derivative(t) for t in XX])
  SS = [a for a in RR if a in QQ]
  QQ_minus_RR = [pt for pt in QQ if pt not in RR]
  RR_minus_SS = [pt for pt in RR if pt not in SS]
  targets = []
  for L in [QQ_minus_RR,RR_minus_SS,SS]:
    if len(L) > 0:
      targets.append(L)
  return list(itertools.product(*targets))

################################################################################

def orthogonal_stabilizer(quadratic_form, points, projective=True, filename=None,
                            num_jobs=1, job=0, work_estimate=False, **kwargs):
  r"""
  Compute stabilizer of ``points`` inside orthogonal group 

  INPUT:

  - ``quadratic_form`` -- a quadratic form in GF(q)[x1,x2,...,xn]

  - ``points`` -- a tuple of points of projective (n-1)-space

  - ``projective`` -- boolean (default: True); return only classes of elements
    modulo the action of scalar multiplication

  - ``filename`` -- optional file in which to write the output

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - ``work_estimate`` -- bool (default: False); If True, don't do any work, but
    instead return how many total matrices will be checked in the computation.

  - Additional keyword arguments are passed to progress.Progress.

  OUTPUT:

  - a list of all elements of O(quadratic_form) \subset GL_n(GF(q)), or
    PO(quadratic_form) \subset PGL_n(GF(q)) if projective is True, that fix
    ``points``

  EFFECT:

  - If ``filename`` is not None, then the entries of each matrix in the output
    will be written to the given filename as a comma-separated list, one list
    per row

  NOTE:

  - We search using matrices that start with a 1 in the earliest nonzero
    position. If Abar is such a matrix, and A = c*Abar is an element of the
    orthogonal group, then Q(x) = Q(A*xx) = Q(c*Abar*x) = c^2 * Q(Abar*x). 
    Thus, we need to check whether Q(Abar*x) agrees with Q(x) up to
    multiplication by a square, and then we need to divide by the square-root.

  """
  Q = quadratic_form  
  if Q == 0:
    raise ValueError('Q = 0 is not a quadratic form')
  R = quadratic_form.parent()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()
  grp_elts = []
  n = len(R.gens())
  XX = vector(R.gens())
  num_pts = len(points)

  # Compute basis for space of matrices satisfying M*pt = c*pt
  entries = {}
  num_rows = 0
  for k,pt in zip(range(num_pts),points):
    for i in range(n):
      for j in range(n):
        entries[(num_rows,i*n + j)] = pt[j]
      entries[(num_rows,n**2 + k)] = -pt[i]
      num_rows += 1
  M = matrix(FF,entries)
  BB = M.right_kernel().basis()
  N = len(BB)

  # divvy up the search work
  search_size = (q**N - 1)//(q-1)
  start,stop = start_and_stop_work(search_size, num_jobs, job)
  if work_estimate:
    return stop-start

  if num_jobs > 1:
    print('Start index = {}, Stop index = {}'.format(start,stop))

  search_space = proj_space_point_iterator(FF,N-1,start=start,stop=stop)
  scalars = {}
  for c in FF:
    if c == 0: continue
    if c**2 in scalars: continue
    scalars[c**2] = c                          

  # For each pt in each of the sets in targets, we get a
  # system of linear equations that must be satisfied.
  prog = progress.Progress(stop-start, **kwargs)
  for aa in search_space:
    prog()
    entries = sum(u*b for u,b in zip(aa,BB))
    A = matrix(FF,n,n,entries[:-num_pts])
    if not A.is_invertible(): continue
    for cc,c in scalars.items():
      if matrix_act_on_poly(A,Q) == cc*Q:
        g = (1/c)*A
        grp_elts.append(g)
        if not projective and q % 2 != 0:
          grp_elts.append(-g)
        break
  prog.finalize()    
  write_list_of_mats_to_file(grp_elts,filename)
  return grp_elts

################################################################################

def orthogonal_transit_matrix(quadratic_form, src_points, targ_points,
                                 projective=True, filename=None,
                                 num_jobs=1, job=0, work_estimate=False, **kwargs):
  r"""
  Compute a single orthogonal group element that moves ``src_points`` to ``targ_points``

  INPUT:

  - ``quadratic_form`` -- a quadratic form in GF(q)[x1,x2,...,xn]

  - ``src_points`` -- a tuple of points of projective (n-1)-space

  - ``targ_points`` -- a tuple of points of projective (n-1)-space with same
    number of elements as ``src_points``

  - ``projective`` -- boolean (default: True); return only classes of elements
    modulo the action of scalar multiplication

  - ``filename`` -- optional file in which to write the output

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - ``work_estimate`` -- bool (default: False); If True, don't do any work, but
    instead return how many total matrices will be checked in the computation.

  - Additional keyword arguments are passed to progress.Progress.

  OUTPUT:

  - a single element of O(quadratic_form) \subset GL_n(GF(q)), or
    PO(quadratic_form) \subset PGL_n(GF(q)) if projective is True, that map
    ``src_points`` to ``targ_points`` (in given order)

  EFFECT:

  - If ``filename`` is not None, then the entries of the matrix in the output
    will be written to the given filename as a comma-separated list

  - We search using matrices that start with a 1 in the earliest nonzero
    position. If Abar is such a matrix, and A = c*Abar is an element of the
    orthogonal group, then Q(x) = Q(A*xx) = Q(c*Abar*x) = c^2 * Q(Abar*x). 
    Thus, we need to check whether Q(Abar*x) agrees with Q(x) up to
    multiplication by a square, and then we need to divide by the square-root.

  """
  Q = quadratic_form  
  R = quadratic_form.parent()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()
  grp_elts = []
  n = len(R.gens())
  XX = vector(R.gens())
  num_pts = len(src_points)
  if len(targ_points) != num_pts:
    raise ValueError('Number of elements of src_points does not agree with number in targ_points')

  # Compute basis for space of matrices satisfying M*pt0 = c*pt1
  entries = {}
  num_rows = 0
  for k,pt0,pt1 in zip(range(num_pts),src_points,targ_points):
    for i in range(n):
      for j in range(n):
        entries[(num_rows,i*n + j)] = pt0[j]
      entries[(num_rows,n**2 + k)] = -pt1[i]
      num_rows += 1
  M = matrix(FF,entries)
  BB = M.right_kernel().basis()
  N = len(BB)

  # divvy up the search work
  search_size = (q**N - 1)//(q-1)
  start,stop = start_and_stop_work(search_size, num_jobs, job)
  if work_estimate:
    return stop-start

  if num_jobs > 1:
    print('Start index = {}, Stop index = {}'.format(start,stop))

  search_space = proj_space_point_iterator(FF,N-1,start=start,stop=stop)
  scalars = {}
  for c in FF:
    if c == 0: continue
    if c**2 in scalars: continue
    scalars[c**2] = c

  # For each pt in each of the sets in targets, we get a
  # system of linear equations that must be satisfied.
  prog = progress.Progress(stop-start, **kwargs)
  for aa in search_space:
    prog()
    entries = sum(u*b for u,b in zip(aa,BB))
    A = matrix(FF,n,n,entries[:-num_pts])
    if not A.is_invertible(): continue
    for cc,c in scalars.items():
      if matrix_act_on_poly(A,Q) == cc*Q:
        g = (1/c)*A
        grp_elts.append(g)
      break
    if len(grp_elts) > 0: break
  prog.finalize()
  if len(grp_elts) == 0:
    raise RuntimeError('Did not find a transit matrix.')
  write_list_of_mats_to_file(grp_elts,filename)
  return grp_elts[0]

################################################################################

def new_orthogonal_group(quadratic_form, projective=True, filename=None, **kwargs):
  r"""
  Compute all elements of the orthogonal group of the given quadratic form over GF(q)

  INPUT:

  - ``quadratic_form`` -- a quadratic form in GF(q)[x1,x2,...,xn].

  - ``projective`` -- boolean (default: True); return only classes of elements
    modulo the action of scalar multiplication

  - ``filename`` -- optional file in which to write the output

  - Additional keyword arguments are passed to progress.Progress.

  OUTPUT:

  - a list of all elements of O(quadratic_form) \subset GL_n(GF(q)), or
    PO(quadratic_form) \subset PGL_n(GF(q)) if projective is True

  EFFECT:

  - If ``filename`` is not None, then the entries of each matrix in the output
    will be written to the given filename as a comma-separated list, one list
    per row

  NOTE:

  - We search using matrices that start with a 1 in the earliest nonzero
    position. If Abar is such a matrix, and A = c*Abar is an element of the
    orthogonal group, then Q(x) = Q(A*xx) = Q(c*Abar*x) = c^2 * Q(Abar*x). 
    Thus, we need to check whether Q(Abar*x) agrees with Q(x) up to
    multiplication by a square, and then we need to divide by the square-root.

  """
  Q = quadratic_form
  if Q == 0:
    raise ValueError('Q = 0 is not a quadratic form')    
  FF = Q.parent().base_ring()
  Y = orthogonal_transitive_set(Q)
  print('Using transitivity on {} elements'.format(len(Y)))

  base_points = Y[0]
  print('Computing stabilizer group for {}'.format(base_points))
  args = (Q,base_points)
  stabilizer = orthogonal_stabilizer(*args,projective=projective,**kwargs)

  orbit_mats = []
  print('Computing transit matrices for {} sets of points'.format(len(Y)-1))
  for i,other_points in enumerate(Y[1:],1):
    print('Computing transit matrix for point {}'.format(i))
    args = (Q,base_points,other_points)
    h = orthogonal_transit_matrix(*args,projective=projective,**kwargs)
    orbit_mats.append(h)

  grp_elts = stabilizer[:]
  m = grp_elts[0].nrows()
  for h in orbit_mats:
    for g in stabilizer:
      grp_elts.append(h*g)
  write_list_of_mats_to_file(grp_elts,filename)
  return grp_elts

################################################################################

def char2_dim4_nonsplit_smooth_orthogonal_group(field, u, **kwargs):
  r"""
  Compute the orthogonal group of vw + x^2 + uxy + y^2 over field

  INPUT:

  - ``field`` -- a field of characteristic 2

  - ``u`` -- an element of ``field`` such that x^2 + uxy + y^2 is irreducible

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of elements in GL_4(field) of the group
    O(vw + x^2 + uxy + y^2). 

  NOTE:

  - L.E. Dickson showed that O(vw + x^2 + uxy + y^2) has cardinality
    2*(q**2)*(q**2+1)*(q**2-1) on p.206 of his book, where we view this
    quadratic form as being in dimension 4. Moreover, if `Q = vw + x^2
    + uxy + y^2`, then `Q(A(v,w,x,y))` has coefficients

      (v^2, a20*a30*u + a00*a10 + a20^2 + a30^2)
      (v*w, a21*a30*u + a20*a31*u + a01*a10 + a00*a11)
      (v*x, a22*a30*u + a20*a32*u + a02*a10 + a00*a12)
      (v*y, a23*a30*u + a20*a33*u + a03*a10 + a00*a13)
      (w^2, a21*a31*u + a01*a11 + a21^2 + a31^2)
      (w*x, a22*a31*u + a21*a32*u + a02*a11 + a01*a12)
      (w*y, a23*a31*u + a21*a33*u + a03*a11 + a01*a13)
      (x^2, a22*a32*u + a02*a12 + a22^2 + a32^2)
      (x*y, a23*a32*u + a22*a33*u + a03*a12 + a02*a13)
      (y^2, a23*a33*u + a03*a13 + a23^2 + a33^2)

  - The 3-fold vw + x^2 + uxy + y^2 = 0 has q^3 - q^2 + q affine solutions. (We
    drop the all-zeros solution because it gives a non-invertible matrix.) We
    loop over two copies of this set of solutions.  The 3-fold vw + x^2 + uxy +
    y^2 = 1 has q^3 + q solutions.

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  if not q % 2 == 0:
    raise ValueError('{} does not have characteristic 2'.format(field))
  R = PolynomialRing(field,'T')
  T = R.gen()
  if not (T**2 + u*T + 1).is_irreducible():
    raise ValueError('Quadratic form vw + x^2 + ({})xy + y^2 is split'.format(u))
  pts0 = [None]*(q**3 - q**2 + q - 1)
  pts1 = [None]*(q**3+q)
  pt0_index = 0
  pt1_index = 0
  for a,b,c,d in affine_space_point_iterator(field,4):
    if (a,b,c,d) == (0,0,0,0): continue
    quad_val = a*b + c**2 + u*c*d + d**2
    if quad_val == 0:
      pts0[pt0_index] = (a,b,c,d)
      pt0_index += 1
    elif quad_val == 1:
      pts1[pt1_index] = (a,b,c,d)
      pt1_index += 1
  assert pts0[-1] is not None
  assert pts1[-1] is not None  
  
  # Use Dickson's count to alot the right amount of memory
  N = 2*(q**2)*(q**2+1)*(q**2-1)
  mats = [None]*N
  mat_index = 0
  prog = progress.Progress(len(pts0)**2*len(pts1)**2, **kwargs)
  for a00, a10, a20, a30 in pts0: # v^2
    for a01, a11, a21, a31 in pts0: # w^2
      if a01*a10 + a00*a11 + u*a21*a30 + u*a20*a31 != 1: # vw
        prog(len(pts1)**2)
        continue
      for a02, a12, a22, a32 in pts1: # x^2
        if a02*a10 + a00*a12 + u*a22*a30 + u*a20*a32 != 0: # vx
          prog(len(pts1))
          continue
        if a02*a11 + a01*a12 + u*a22*a31 + u*a21*a32 != 0: # wx
          prog(len(pts1))
          continue
        for a03, a13, a23, a33 in pts1: # y^2
          prog()
          if a03*a10 + a00*a13 + u*a23*a30 + u*a20*a33 != 0: # vy
            continue
          if a03*a11 + a01*a13 + u*a23*a31 + u*a21*a33 != 0: # wy
            continue
          if a03*a12 + a02*a13 + u*a23*a32 + u*a22*a33 != u: # xy
            continue
          entries = [[a00,a01,a02,a03],[a10,a11,a12,a13],[a20,a21,a22,a23],[a30,a31,a32,a33]]
          mats[mat_index] = matrix(field,entries)
          mat_index += 1
  prog.finalize()
  assert mats[-1] is not None
  return mats

################################################################################

def char2_dim4_nonsplit_quadric_automorphism_group(field, u, **kwargs):
  r"""
  Compute the automorphism group of the quadric surface V(vw + x^2 + uxy + y^2) over field

  INPUT:

  - ``field`` -- a field of characteristic 2

  - ``u`` -- an element of ``field`` such that `Q = x^2 + uxy + y^2` is irreducible

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of elements `g \in GL_4(field)` such that `Q(g) = cQ` for
    some nonzero `c \in field`.

  NOTE:

  - If `g` is an element of the orthogonal group of `Q` and `a` is a nonzero
    element of ``field``, then `Q(ag) = a^2 Q`. Conversely, since every nonzero
    element of ``field`` is a square, any automorphism of the variety `V(Q)`
    arises as a multiple of an element of the orthogonal group of Q
  
  """
  OQ = char2_dim4_nonsplit_smooth_orthogonal_group(field,u,**kwargs)
  auts = []
  for c in field:
    if c == 0: continue
    auts += [c*g for g in OQ]
  return auts

################################################################################
          
def odd_char_dim4_nonsplit_smooth_orthogonal_group(field, u, s=None, **kwargs):
  r"""
  Compute the (twisted) orthogonal group of vw + x^2 + uy^2 over field

  INPUT:

  - ``field`` -- a field of odd characteristic

  - ``u`` -- an element of ``field`` such that `Q = x^2 + uy^2` is irreducible

  - ``s`` -- optional element for twisting the orthogonal group (default: 1)

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of elements `g \in GL_4(field))` such that `Q(g) = sQ`; when
    `s = 1`, this is the usual orthogonal group of `Q`.

  NOTE:

  - L.E. Dickson showed that SO(vw + x^2 + uy^2) has cardinality
    (q**2)*(q**2+1)*(q**2-1) on p.160 of his book (case m=4). Moreover, if `Q =
    vw + x^2 + uy^2`, then `Q(A(v,w,x,y))` has coefficients
      
      (v^2, a00*a10 + a20^2 + u*a30^2)
      (v*w, 2*a30*a31*u + a01*a10 + a00*a11 + 2*a20*a21)
      (v*x, 2*a30*a32*u + a02*a10 + a00*a12 + 2*a20*a22)
      (v*y, 2*a30*a33*u + a03*a10 + a00*a13 + 2*a20*a23)
      (w^2, a01*a11 + a21^2 + u*a31^2)
      (w*x, 2*a31*a32*u + a02*a11 + a01*a12 + 2*a21*a22)
      (w*y, 2*a31*a33*u + a03*a11 + a01*a13 + 2*a21*a23)
      (x^2, a02*a12 + a22^2 + u*a32^2)
      (x*y, 2*a32*a33*u + a03*a12 + a02*a13 + 2*a22*a23)
      (y^2, a03*a13 + a23^2 + u*a33^2)

  - The 3-fold vw + x^2 + uy^2 = 0 has q^3 - q^2 + q affine solutions. (We drop
    the all-zeros solution because it gives a non-invertible matrix.) We loop
    over two copies of this set of solutions.  The 3-fold vw + x^2 + uy^2 = 1
    has q^3 + q solutions, as does the 3-fold vw + x^2 + uy^2 = u.

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  if not q % 2 == 1:
    raise ValueError('{} does not have odd characteristic'.format(field))
  R = PolynomialRing(field,'T')
  T = R.gen()
  if not (T**2 + u).is_irreducible():
    raise ValueError('Quadratic form vw + x^2 + ({})y^2 is split'.format(u))
  if s is None:
    s = field(1)
  elif s == 0:
    raise ValueError('s = 0 is not an allowed value')
  else:
    s = field(s)
    
  pts0 = [None]*(q**3 - q**2 + q - 1)
  pts1 = [None]*(q**3+q)
  ptsu = [None]*(q**3+q)
  pt0_index = 0
  pt1_index = 0
  ptu_index = 0
  for a,b,c,d in affine_space_point_iterator(field,4):
    if (a,b,c,d) == (0,0,0,0): continue
    quad_val = a*b + c**2 + u*d**2
    if quad_val == 0:
      pts0[pt0_index] = (a,b,c,d)
      pt0_index += 1
    elif quad_val == s:
      pts1[pt1_index] = (a,b,c,d)
      pt1_index += 1
    if quad_val == s*u: # This must be an if statement in case u = 1
      ptsu[ptu_index] = (a,b,c,d)
      ptu_index += 1
      
  assert pts0[-1] is not None
  assert pts1[-1] is not None
  assert ptsu[-1] is not None    
  
  # Use Dickson's count to alot the right amount of memory
  N = 2*(q**2)*(q**2+1)*(q**2-1)
  mats = [None]*N
  mat_index = 0
  prog = progress.Progress(len(pts0)**2*len(pts1)*len(ptsu), **kwargs)
  for a00, a10, a20, a30 in pts0: # v^2
    for a01, a11, a21, a31 in pts0: # w^2
      if 2*a30*a31*u + a01*a10 + a00*a11 + 2*a20*a21 != s: # vw
        prog(len(pts1)*len(ptsu))
        continue
      for a02, a12, a22, a32 in pts1: # x^2
        if 2*a30*a32*u + a02*a10 + a00*a12 + 2*a20*a22 != 0: # vx
          prog(len(pts1))
          continue
        if 2*a31*a32*u + a02*a11 + a01*a12 + 2*a21*a22 != 0: # wx
          prog(len(pts1))
          continue
        for a03, a13, a23, a33 in ptsu: # y^2
          prog()
          if 2*a30*a33*u + a03*a10 + a00*a13 + 2*a20*a23 != 0: # vy
            continue
          if 2*a31*a33*u + a03*a11 + a01*a13 + 2*a21*a23 != 0: # wy
            continue
          if 2*a32*a33*u + a03*a12 + a02*a13 + 2*a22*a23 != 0: # xy
            continue
          entries = [[a00,a01,a02,a03],[a10,a11,a12,a13],[a20,a21,a22,a23],[a30,a31,a32,a33]]
          mats[mat_index] = matrix(field,entries)
          mat_index += 1
  prog.finalize()
  assert mats[-1] is not None
  return mats

################################################################################

def odd_char_dim4_nonsplit_quadric_automorphism_group(field, u, **kwargs):
  r"""
  Compute the automorphism group of the quadric surface V(vw + x^2 + uy^2) over field

  INPUT:

  - ``field`` -- a field of characteristic 2

  - ``u`` -- an element of ``field`` such that `Q = x^2 + uy^2` is irreducible

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of elements `g \in GL_4(field)` such that `Q(g) = cQ` for
    some nonzero `c \in field`. 

  NOTE:

  - If `g` is an element of the orthogonal group of `Q` and `a` is a nonzero
    element of ``field``, then `Q(ag) = a^2 Q`. And if `t` is any non-square in
    ``field`` such that `Q(g) = tQ`, then `Q(ag) = at^2Q. Conversely, we can
    obtain any automorphism of `V(Q)` in one of these two ways.
  
  """
  # Find squares and the "least" nonsquare
  sqrts = set()
  squares = set()
  for a in field:
    if a == 0: continue
    if a**2 in squares: continue
    sqrts.add(a)
    squares.add(a**2)
  s = None
  for a in field:
    if a == 0: continue
    if a in squares: continue
    s = a
    break
  sqrts = sorted(list(sqrts))

  auts = []
  OQ = odd_char_dim4_nonsplit_smooth_orthogonal_group(field,u,**kwargs)
  for a in sqrts:
    auts += [a*g for g in OQ]
  OQs = odd_char_dim4_nonsplit_smooth_orthogonal_group(field,u,s,**kwargs)
  for a in sqrts:
    auts += [a*g for g in OQs]
  return auts

################################################################################

def char2_dim5_smooth_orthogonal_group(field, **kwargs):
  r"""
  Compute the projective  orthogonal group of vw + xy + z^2 over field

  INPUT:

  - ``field`` -- a field of characteristic 2

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of representatives in GL_5(field) of the group
    PO(vw + xy + z^2)

  NOTES:

  - L.E. Dickson showed that O(vw + xy + z^2) agrees with the "Special Abelian
    Group", and on p.94 of his book he argues that this group has cardinality
    q^4 * (q^2-1)^2 * (q^2+1), where q is the cardinality of the field. If ``A``
    is an orthogonal matrix with coefficient `aij`, then the final column has
    the form (0,0,0,0,*), since `A` must fix the line generated by this
    vector. (It generates the radical.) If `Q = vw + xy + z^2`, then
    `Q(A(v,w,x,y,z))` has coefficients

      (v^2, a00*a10 + a20*a30 + a40^2)
      (v*w, a01*a10 + a00*a11 + a21*a30 + a20*a31)
      (v*x, a02*a10 + a00*a12 + a22*a30 + a20*a32)
      (v*y, a03*a10 + a00*a13 + a23*a30 + a20*a33)
      (v*z, 0)
      (w^2, a01*a11 + a21*a31 + a41^2)
      (w*x, a02*a11 + a01*a12 + a22*a31 + a21*a32)
      (w*y, a03*a11 + a01*a13 + a23*a31 + a21*a33)
      (w*z, 0)
      (x^2, a02*a12 + a22*a32 + a42^2)
      (x*y, a03*a12 + a02*a13 + a23*a32 + a22*a33)
      (x*z, 0)
      (y^2, a03*a13 + a23*a33 + a43^2)
      (y*z, 0)
      (z^2, a44^2)

  - The quadric hypersurface vw + xy + z^2 = 0 has q^4 affine rational points.

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  if not q % 2 == 0:
    raise ValueError('{} does not have characteristic 2'.format(field))
  pts = [None]*(q**4-1)
  pt_index = 0
  for a,b,c,d,e in affine_space_point_iterator(field,5):
    if (a,b,c,d,e) == (0,0,0,0,0): continue
    if a*b + c*d + e**2 == 0:
      pts[pt_index] = (a,b,c,d,e)
      pt_index += 1
  assert pts[-1] is not None

  # Use Dickson's count to alot the right amount of memory
  N = q**4 * (q**2 - 1)**2 * (q**2 + 1)
  # Does every element have determinant 1? 
  mats = [None]*N
  mat_index = 0
  prog = progress.Progress(len(pts)**4, **kwargs)
  for a00, a10, a20, a30, a40 in pts: # v^2
    for a01, a11, a21, a31, a41 in pts: # w^2
      if a01*a10 + a00*a11 + a21*a30 + a20*a31 != 1: # v*w
        prog(len(pts)**2)
        continue
      for a02, a12, a22, a32, a42 in pts: # x^2
        if a02*a10 + a00*a12 + a22*a30 + a20*a32 != 0: # v*x
          prog(len(pts))
          continue
        if a02*a11 + a01*a12 + a22*a31 + a21*a32 != 0: # w*x
          prog(len(pts))
          continue
        for a03, a13, a23, a33, a43 in pts: # y^2
          prog()
          if a03*a10 + a00*a13 + a23*a30 + a20*a33 != 0: # v*y
            continue
          if a03*a11 + a01*a13 + a23*a31 + a21*a33 != 0: # w*y
            continue
          if a03*a12 + a02*a13 + a23*a32 + a22*a33 != 1: # x*y
            continue
          entries = [[a00,a01,a02,a03,0],[a10,a11,a12,a13,0],[a20,a21,a22,a23,0],[a30,a31,a32,a33,0],[a40,a41,a42,a43,1]]
          mats[mat_index] = matrix(field,entries)
          mat_index += 1
  prog.finalize()
  assert mats[-1] is not None
  return mats
        
################################################################################

def char2_dim5_nonsplit_rank4_orthogonal_group(field, u, **kwargs):
  r"""
  Compute the projective orthogonal group of vw + x^2 + uxy + y^2 over field

  INPUT:

  - ``field`` -- a field of characteristic 2

  - ``u`` -- an element of ``field`` such that x^2 + uxy + y^2 is irreducible

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of representatives in GL_5(field) of the group
    PO(vw + x^2 + uxy + y^2). 

  NOTE:

  - L.E. Dickson showed that O(vw + x^2 + uxy + y^2) has cardinality
    2*(q**2)*(q**2+1)*(q**2-1) on p.206 of his book, where we view this
    quadratic form as being in dimension 4. To extend to dimension 5, we use
    Lemma A.11 in Faber/Grantham. This shows that if ``A`` is an orthogonal
    matrix with coefficient `aij`, then the final column has the form
    (0,0,0,0,*), and the final row has the form (*,*,*,*,*). Thus we may take
    the bottom right entry of `A` to be 1 and the full cardinality of PO(vw +
    x^2 + uxy + y^2) is 2*(q**6)*(q**2+1)*(q**2-1).  Moreover, if `Q = vw + x^2
    + uxy + y^2`, then `Q(A(v,w,x,y,z))` has coefficients

      (v^2, a20*a30*u + a00*a10 + a20^2 + a30^2)
      (v*w, a21*a30*u + a20*a31*u + a01*a10 + a00*a11)
      (v*x, a22*a30*u + a20*a32*u + a02*a10 + a00*a12)
      (v*y, a23*a30*u + a20*a33*u + a03*a10 + a00*a13)
      (v*z, 0)
      (w^2, a21*a31*u + a01*a11 + a21^2 + a31^2)
      (w*x, a22*a31*u + a21*a32*u + a02*a11 + a01*a12)
      (w*y, a23*a31*u + a21*a33*u + a03*a11 + a01*a13)
      (w*z, 0)
      (x^2, a22*a32*u + a02*a12 + a22^2 + a32^2)
      (x*y, a23*a32*u + a22*a33*u + a03*a12 + a02*a13)
      (x*z, 0)
      (y^2, a23*a33*u + a03*a13 + a23^2 + a33^2)
      (y*z, 0)
      (z^2, 0)
  

  - The 3-fold vw + x^2 + uxy + y^2 = 0 has q^3 - q^2 + q affine solutions. (We
    drop the all-zeros solution because it gives a non-invertible matrix.) We
    loop over two copies of this set of solutions.  The 3-fold vw + x^2 + uxy +
    y^2 = 1 has q^3 + q solutions.

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  if not q % 2 == 0:
    raise ValueError('{} does not have characteristic 2'.format(field))
  pts0 = [None]*(q**3 - q**2 + q - 1)
  pts1 = [None]*(q**3+q)
  pt0_index = 0
  pt1_index = 0
  for a,b,c,d in affine_space_point_iterator(field,4):
    if (a,b,c,d) == (0,0,0,0): continue
    quad_val = a*b + c**2 + u*c*d + d**2
    if quad_val == 0:
      pts0[pt0_index] = (a,b,c,d)
      pt0_index += 1
    elif quad_val == 1:
      pts1[pt1_index] = (a,b,c,d)
      pt1_index += 1
  assert pts0[-1] is not None
  assert pts1[-1] is not None  
  
  # Use Dickson's count to alot the right amount of memory
  N = 2*(q**6)*(q**2+1)*(q**2-1)
  mats = [None]*N
  mat_index = 0
  prog = progress.Progress(len(pts0)**2*len(pts1)**2, **kwargs)
  for a00, a10, a20, a30 in pts0: # v^2
    for a01, a11, a21, a31 in pts0: # w^2
      if a01*a10 + a00*a11 + u*a21*a30 + u*a20*a31 != 1: # vw
        prog(len(pts1)**2)
        continue
      for a02, a12, a22, a32 in pts1: # x^2
        if a02*a10 + a00*a12 + u*a22*a30 + u*a20*a32 != 0: # vx
          prog(len(pts1))
          continue
        if a02*a11 + a01*a12 + u*a22*a31 + u*a21*a32 != 0: # wx
          prog(len(pts1))
          continue
        for a03, a13, a23, a33 in pts1: # y^2
          prog()
          if a03*a10 + a00*a13 + u*a23*a30 + u*a20*a33 != 0: # vy
            continue
          if a03*a11 + a01*a13 + u*a23*a31 + u*a21*a33 != 0: # wy
            continue
          if a03*a12 + a02*a13 + u*a23*a32 + u*a22*a33 != u: # xy
            continue
          for a40, a41, a42, a43 in affine_space_point_iterator(field,4):
            entries = [[a00,a01,a02,a03,0],[a10,a11,a12,a13,0],[a20,a21,a22,a23,0],[a30,a31,a32,a33,0],[a40,a41,a42,a43,1]]
            mats[mat_index] = matrix(field,entries)
            mat_index += 1
  prog.finalize()
  assert mats[-1] is not None
  return mats

################################################################################

def odd_char_dim5_smooth_orthogonal_group(field, **kwargs):
  r"""Compute the projective orthogonal group of vw + xy + z^2 over field

  INPUT:

  - ``field`` -- a field of odd characteristic

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of representatives in GL_5(field) of the group
    PO(vw + xy + z^2) = SO(vw + xy + z^2).

  NOTES:

  - L.E. Dickson showed on p.160 of his book that this group has cardinality q^4
    * (q^2-1)^2 * (q^2+1), where q is the cardinality of the field. If ``A`` is
    a matrix with coefficient `aij`, and if `Q = vw + xy + z^2`, then
    `Q(A(v,w,x,y,z))` has coefficients

      (v^2, a00*a10 + a20*a30 + a40^2)
      (v*w, a01*a10 + a00*a11 + a21*a30 + a20*a31 + 2*a40*a41)
      (v*x, a02*a10 + a00*a12 + a22*a30 + a20*a32 + 2*a40*a42)
      (v*y, a03*a10 + a00*a13 + a23*a30 + a20*a33 + 2*a40*a43)
      (v*z, a04*a10 + a00*a14 + a24*a30 + a20*a34 + 2*a40*a44)
      (w^2, a01*a11 + a21*a31 + a41^2)
      (w*x, a02*a11 + a01*a12 + a22*a31 + a21*a32 + 2*a41*a42)
      (w*y, a03*a11 + a01*a13 + a23*a31 + a21*a33 + 2*a41*a43)
      (w*z, a04*a11 + a01*a14 + a24*a31 + a21*a34 + 2*a41*a44)
      (x^2, a02*a12 + a22*a32 + a42^2)
      (x*y, a03*a12 + a02*a13 + a23*a32 + a22*a33 + 2*a42*a43)
      (x*z, a04*a12 + a02*a14 + a24*a32 + a22*a34 + 2*a42*a44)
      (y^2, a03*a13 + a23*a33 + a43^2)
      (y*z, a04*a13 + a03*a14 + a24*a33 + a23*a34 + 2*a43*a44)
      (z^2, a04*a14 + a24*a34 + a44^2)

  - The quadric hypersurface vw + xy + z^2 = 0 has q^4 affine rational points,
    while vw + xy + z^2 = 1 has q^4 + q^2 affine rational points.

  - To see that PO(vw + xy + z^2) = SO(vw + xy + z^2), we are using the fact
    that every orthogonal group element in this case has determinant \pm 1, and
    that the negative identity matrix is an orthogonal group element.    

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  if not q % 2 == 1:
    raise ValueError('{} does not have odd characteristic'.format(field))
  pts0 = [None]*(q**4 - 1) # number of nontrivial affine points on vw + xy + z^2 = 0
  pts1 = [None]*(q**4+q**2) # number of affine points on vw + xy + z^2 = 1
  pt0_index = 0
  pt1_index = 0
  for a,b,c,d,e in affine_space_point_iterator(field,5):
    if (a,b,c,d,e) == (0,0,0,0,0): continue
    quad_val = a*b + c*d + e**2
    if quad_val == 0:
      pts0[pt0_index] = (a,b,c,d,e)
      pt0_index += 1
    elif quad_val == 1:
      pts1[pt1_index] = (a,b,c,d,e)
      pt1_index += 1
  assert pts0[-1] is not None
  assert pts1[-1] is not None  

  # Use Dickson's count to alot the right amount of memory  
  N = q**4 * (q**2-1)**2 * (q**2 + 1)
  mats = [None]*N
  mat_index = 0
  prog = progress.Progress(len(pts0)**4*len(pts1), **kwargs)
  for a00, a10, a20, a30, a40 in pts0: # v^2
    for a01, a11, a21, a31, a41 in pts0: # w^2
      if a01*a10 + a00*a11 + a21*a30 + a20*a31 + 2*a40*a41 != 1: # v*w
        prog(len(pts0)**2 * len(pts1))
        continue
      for a02, a12, a22, a32, a42 in pts0: # x^2
        if a02*a10 + a00*a12 + a22*a30 + a20*a32 + 2*a40*a42 != 0: # v*x
          prog(len(pts0) * len(pts1))
          continue
        if a02*a11 + a01*a12 + a22*a31 + a21*a32 + 2*a41*a42 != 0: # w*x
          prog(len(pts0) * len(pts1))
          continue
        for a03, a13, a23, a33, a43 in pts0: # y^2
          if a03*a10 + a00*a13 + a23*a30 + a20*a33 + 2*a40*a43 != 0: # v*y
            prog(len(pts1))
            continue
          if a03*a11 + a01*a13 + a23*a31 + a21*a33 + 2*a41*a43 != 0: #w*y
            prog(len(pts1))
            continue
          if a03*a12 + a02*a13 + a23*a32 + a22*a33 + 2*a42*a43 != 1: # x*y
            prog(len(pts1))
            continue
          for a04, a14, a24, a34, a44 in pts1: # z^2
            if a04*a10 + a00*a14 + a24*a30 + a20*a34 + 2*a40*a44 != 0: # v*z
              prog()
              continue
            if a04*a11 + a01*a14 + a24*a31 + a21*a34 + 2*a41*a44 != 0: # w*z
              prog()
              continue
            if a04*a12 + a02*a14 + a24*a32 + a22*a34 + 2*a42*a44 != 0: # x*z
              prog()
              continue
            if a04*a13 + a03*a14 + a24*a33 + a23*a34 + 2*a43*a44 != 0: # y*z
              prog()
              continue
            entries = [[a00,a01,a02,a03,a04],[a10,a11,a12,a13,a14],[a20,a21,a22,a23,a24],[a30,a31,a32,a33,a34],[a40,a41,a42,a43,a44]]
            A = matrix(field,entries)
            if A.determinant() != 1: continue
            mats[mat_index] = A
            mat_index += 1
  prog.finalize()
  assert mats[-1] is not None
  return mats

################################################################################
          
def odd_char_dim5_nonsplit_rank4_orthogonal_group(field, u, **kwargs):
  r"""
  Compute the projective orthogonal group of vw + x^2 + uy^2 over field

  INPUT:

  - ``field`` -- a field of odd characteristic

  - ``u`` -- an element of ``field`` such that x^2 + uy^2 is irreducible

  - Remaining keyword arguments are passed to Progress class

  OUTPUT:

  - A complete list of representatives in GL_5(field) of the group
    PO(vw + x^2 + uy^2). 

  NOTE:

  - L.E. Dickson showed that SO(vw + x^2 + uy^2) has cardinality
    (q**2)*(q**2+1)*(q**2-1) on p.160 of his book (case m=4), where we view this
    quadratic form as being in dimension 4. To extend to dimension 5, we use
    Lemma A.11 in Faber/Grantham. This shows that if ``A`` is an orthogonal
    matrix with coefficient `aij`, then the final column has the form
    (0,0,0,0,*), and the final row has the form (*,*,*,*,*). Thus we may take
    the bottom right entry of `A` to be 1 and the full cardinality of PO(vw +
    x^2 + uy^2) is 2(q**6)*(q**2+1)*(q**2-1). (The extra factor of 2 comes from
    the fact that we need to use the full orthogonal group of the quadratic form
    in dimension 4, not just the elements with determinant 1. Moreover, if `Q =
    vw + x^2 + uy^2`, then `Q(A(v,w,x,y,z))` has coefficients
      
      (v^2, a00*a10 + a20^2 + u*a30^2)
      (v*w, 2*a30*a31*u + a01*a10 + a00*a11 + 2*a20*a21)
      (v*x, 2*a30*a32*u + a02*a10 + a00*a12 + 2*a20*a22)
      (v*y, 2*a30*a33*u + a03*a10 + a00*a13 + 2*a20*a23)
      (v*z, 0)
      (w^2, a01*a11 + a21^2 + u*a31^2)
      (w*x, 2*a31*a32*u + a02*a11 + a01*a12 + 2*a21*a22)
      (w*y, 2*a31*a33*u + a03*a11 + a01*a13 + 2*a21*a23)
      (w*z, 0)
      (x^2, a02*a12 + a22^2 + u*a32^2)
      (x*y, 2*a32*a33*u + a03*a12 + a02*a13 + 2*a22*a23)
      (x*z, 0)
      (y^2, a03*a13 + a23^2 + u*a33^2)
      (y*z, 0)
      (z^2, 0)
  

  - The 3-fold vw + x^2 + uy^2 = 0 has q^3 - q^2 + q affine solutions. (We drop
    the all-zeros solution because it gives a non-invertible matrix.) We loop
    over two copies of this set of solutions.  The 3-fold vw + x^2 + uy^2 = 1
    has q^3 + q solutions, as does the 3-fold vw + x^2 + uy^2 = u.

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  q = field.cardinality()
  if not q % 2 == 1:
    raise ValueError('{} does not have odd characteristic'.format(field))
  pts0 = [None]*(q**3 - q**2 + q - 1)
  pts1 = [None]*(q**3+q)
  ptsu = [None]*(q**3+q)
  pt0_index = 0
  pt1_index = 0
  ptu_index = 0
  for a,b,c,d in affine_space_point_iterator(field,4):
    if (a,b,c,d) == (0,0,0,0): continue
    quad_val = a*b + c**2 + u*d**2
    if quad_val == 0:
      pts0[pt0_index] = (a,b,c,d)
      pt0_index += 1
    elif quad_val == 1:
      pts1[pt1_index] = (a,b,c,d)
      pt1_index += 1
    if quad_val == u: # This must be an if statement in case u = 1
      ptsu[ptu_index] = (a,b,c,d)
      ptu_index += 1
      
  assert pts0[-1] is not None
  assert pts1[-1] is not None
  assert ptsu[-1] is not None    
  
  # Use Dickson's count to alot the right amount of memory
  N = 2*(q**6)*(q**2+1)*(q**2-1)
  mats = [None]*N
  mat_index = 0
  prog = progress.Progress(len(pts0)**2*len(pts1)*len(ptsu), **kwargs)
  for a00, a10, a20, a30 in pts0: # v^2
    for a01, a11, a21, a31 in pts0: # w^2
      if 2*a30*a31*u + a01*a10 + a00*a11 + 2*a20*a21 != 1: # vw
        prog(len(pts1)*len(ptsu))
        continue
      for a02, a12, a22, a32 in pts1: # x^2
        if 2*a30*a32*u + a02*a10 + a00*a12 + 2*a20*a22 != 0: # vx
          prog(len(pts1))
          continue
        if 2*a31*a32*u + a02*a11 + a01*a12 + 2*a21*a22 != 0: # wx
          prog(len(pts1))
          continue
        for a03, a13, a23, a33 in ptsu: # y^2
          prog()
          if 2*a30*a33*u + a03*a10 + a00*a13 + 2*a20*a23 != 0: # vy
            continue
          if 2*a31*a33*u + a03*a11 + a01*a13 + 2*a21*a23 != 0: # wy
            continue
          if 2*a32*a33*u + a03*a12 + a02*a13 + 2*a22*a23 != 0: # xy
            continue
          for a40, a41, a42, a43 in affine_space_point_iterator(field,4):
            entries = [[a00,a01,a02,a03,0],[a10,a11,a12,a13,0],[a20,a21,a22,a23,0],[a30,a31,a32,a33,0],[a40,a41,a42,a43,1]]
            mats[mat_index] = matrix(field,entries)
            mat_index += 1
  prog.finalize()
  assert mats[-1] is not None
  return mats
  
################################################################################
###############################  GENUS-3 CURVES  ###############################
################################################################################

def genus3_canonical_curves(polring, extra_tests=None, **kwargs):
  r"""
  Return a list of smooth plane quartics over GF(q), up to isomorphism

  INPUT:

  - ``polring`` -- a polynomial ring over a finite field in 3 variables, 
    say GF(q)[x,y,z]

  - ``extra_tests`` -- a list of boolean-valued function of a quartic polynomial, 
    used to filter the the curves returned

  - Remaining keyword arguments are passed to progress
  
  OUTPUT:

  - a list of homogeneous quartic polynomials passing all tests in
    ``extra_tests` with the property that any canonically embedded genus-3 curve
    is given by one of them, and no two polynomials yield isomorphic curves

  """
  FF = polring.base_ring()
  if not FF.is_finite():
    raise ValueError
  q = FF.cardinality()
  
  monomials = homogeneous_monomials(polring,4)
  N = len(monomials)
  assert N == 15
  monomials.sort(reverse=True)

  if extra_tests is None:
    extra_tests = []

  print('Storing automorphisms of P^2 ...')
  PGL3 = []
  for aa in proj_space_point_iterator(FF,8):
    M = matrix(FF,3,3,aa)
    if M.is_invertible():
      PGL3.append(M)

  num_pols = (q**N - 1)//(q-1)
  print('Searching {} quartics ...'.format(num_pols))
  
  seen = set()
  quartics = []
  prog = progress.Progress(num_pols,**kwargs)
  for pol in poly_iterator(monomials,projective=True):
    prog()
    if pol in seen: continue
    if not all(test(pol) for test in extra_tests): continue
    for g in PGL3:
      gpol = matrix_act_on_poly(g,pol)
      lc = gpol.lc()
      if lc != 1:
        gpol *= FF(1)/lc
      seen.add(gpol)
    if hypersurface_is_nonsingular(pol):
      quartics.append(pol)
  prog.finalize()
  return quartics

################################################################################
###############################  GENUS-4 CURVES  ###############################
################################################################################

def genus4_canonical_curves(quadratic_form, filename=None, **kwargs):
  r"""
  Return a list of cubic forms yielding smooth genus-4 curves in P^3, up to isomorphism

  INPUT:

  - ``quadratic_form`` -- a quadratic form Q in GF(q)[x,y,z,w]

  - ``filename`` -- optional string; describes file in which to write the output

  - Remaining keyword arguments are passed to progress
  
  OUTPUT:

  - A list of cubic forms F such that each curve {F = Q = 0} in P^3 is smooth of
    genus~4, and such that no two cubic forms are in the same orbit of the
    orthogonal group O(Q)

  EFFECT:

  - If ``filename`` is not None, then the forms will be written as output, one per line

  """
  Q = quadratic_form
  polring = Q.parent()
  xx = vector(polring.gens())
  FF = polring.base_ring()
  if not FF.is_finite():
    raise ValueError
  if len(xx) != 4:
    raise ValueError
  q = FF.cardinality()
  
  monomials = homogeneous_monomials(polring,3)
  N = len(monomials)
  assert N == 20

  # Open the file before any serious computation is done
  fp = open(filename,'w') if filename is not None else None

  print('Computing orthogonal group of {}'.format(Q))
  Oq = orthogonal_group(Q,projective=True)
  print('Orthogonal group has {} elements'.format(len(Oq)))

  num_pols = (q**N - 1)//(q-1)
  print('Searching {} cubic forms ...'.format(num_pols))

  
  seen = set()
  cubics = []
  first_row = [Q.derivative(x) for x in xx] # Jacobian matrix row
  prog = progress.Progress(num_pols, **kwargs)
  for aa in proj_space_point_iterator(FF,N-1):
    F = sum(c*term for c,term in zip(aa,monomials))
    prog()
    if F in seen:
      continue
    # Look at F(g(x)) + Q*L for g \in O(Q) and L a line
    for g in Oq:
      for coefs in affine_space_point_iterator(FF,3):
        L = sum(c*x for c,x in zip(coefs,xx))
        gFL = F(*(g*xx)) + Q*L
        if gFL == 0:
          seen.add(gFL)
          continue
        lc = gFL.lc()
        if lc != 1: gFL *= FF(1)/lc
        seen.add(gFL)
    I = polring.ideal(Q,F)
    # Check that Q = F = 0 defines a curve
    if ideal_dimension(I) != 2: continue
    # check for a singularity
    second_row = [F.derivative(x) for x in xx]
    J = matrix(polring,2,4,[first_row,second_row])
    sing_ideal = polring.ideal([F,Q] + J.minors(2))
    if ideal_dimension(sing_ideal) != 0: continue
    # Now Q = F = 0 has genus~4
    if fp is not None:
      fp.write(str(F) + '\n')
    else:
      cubics.append(F)
  prog.finalize()
  return cubics

################################################################################

def count_genus4_points(quadratic_form, cubicdata, return_points=False):
  r"""
  Count points on genus 4 curves specified as intersection of a quadric and cubic surface

  INPUT:

  - ``quadratic_form`` -- a quadratic_form in GF(q)[x,y,z,w]

  - ``cubicdata`` -- Either a homogeneous cubic polynomial, a list of such, or a
    string giving a file containing cubic polynomials in GF(q)[x,y,z,w]

  - ``return_points`` -- boolean (default: False); if True, return the set of
    points in the dict value as well

  OUTPUT:

  - a dict with keys given by the polynomials in cubicfile:

    * if return_points is False, the values are the number of GF(q)-rational points 
      in the intersection

    * if return_points is True, the values are the sets of GF(q)-rational points

  """
  Q = quadratic_form
  R = Q.parent()
  FF = R.base_ring()
  assert FF.is_finite()
  assert len(R.gens()) == 4

  if isinstance(cubicdata,str):
    fp = open(cubicdata,'r')
    cubics = [R(pol) for pol in fp]
    fp.close()
  elif any(isinstance(cubicdata,u) for u in [list,set,tuple]):
    cubics = list(cubicdata)
  elif cubicdata in R:
    cubics = [cubicdata]
  else:
    raise ValueError('Unsure how to handle cubicdata = {}'.format(cubicdata))

  quadric_pts = set()
  # find points with first coordinate 1
  for aa in proj_space_point_iterator(FF,3):
    if Q(*aa) == 0:
      quadric_pts.add(aa)
        
  D = {}
  for pol in cubics:
    pts = [pt for pt in quadric_pts if evaluate_poly(pol,pt)==0]
    if return_points:
      D[pol] = pts
    else:
      D[pol] = len(pts)
  return D

  
################################################################################
###############################  GENUS-5 CURVES  ###############################
################################################################################

def second_quadric_class_reps(quadratic_form, allowed_invariants,
                              quadric_sing_data=None, quadric_point_count_data=None,
                              orthog_grp_data=None, 
                              sing_outfile=None, sage_outfile=None, **kwargs):
  r""" 
  Construct class representatives of nonzero quadratic forms
  in 5 variables modulo the orthogonal group of a fixed quadratic form

  INPUT:

  - ``quadratic_form`` -- an element of GF(q)[v,w,x,y,z] that gives rise to a
    geometrically irreducible quadric in PP^4

  - ``allowed_invariants`` -- list of pairs of integers `(a,b)`, corresponding
    to allowed values of (dimension of singular locus, number of rational
    points)

  - ``quadric_sing_data`` -- optional filename giving classification of quadric
    singularities as written by quadric_singularity_dimension (with
    projective=True flag set)

  - ``quadric_point_count_data`` -- optional filename giving the number of
    points on each quadric hypersurface as written by quadric_point_count (with
    projective=True flag set)

  - ``orthog_grp_data`` -- optional filename in which matrix data is stored for
    the projective orthogonal group of quadratic_form; the output should be in
    the form given by orthogonal_group(quadratic_form, projective=True)

  - ``sing_outfile`` -- optional file in which to write data in format to be
    read by Singular; data will be written as a list with name `q2s`

  - ``sage_outfile`` -- optional file in which to write data in format to be
    read by Sage

  - Remaining arguments are passed to the progress.Progress class

  OUTPUT:

  - A list of nonzero quadratic forms `Q` with the following properties:

    * For each nonzero `Q'` in the linear span of `Q` and `Q1`, if `a` is the
      dimensions of the singular locus of `V(Q')` and `b` is the number of
      rational points on `V(Q')`, then `(a,b)` appears in the list
      ``allowed_invariants``. 

    * No element of the list is equivalent to `Q1`

    * Each quadric in PP^4 whose invariants `(a,b)` as above appear on the list
      ``allowed_invariants`` is equivalent modulo the orthogonal group action to
      exactly one `\{Q = 0\}` with Q in this list, aside from `Q1`.
  
  EFFECT:

  - If sing_outfile is specified, the list of quadratic forms is written to a
    file with first line "list q2s;", and then each successive line in the form
    "q2s[*] = Q2;", where * is an index and Q2 is a quadratic form on the
    list. This file can be loaded by Singular.

  - If sage_outfile is specified, the list of quadratic forms is written to a
    file with each line being the string representation of a polynomial. 

  NOTES:

  - A non-hyperelliptic, non-trigonal curve of genus 5 can be canonically
    embedded as a complete intersection of 3 geometrically irreducible quadrics
    in PP^4. Let Q1, Q2, Q3 be quadratic forms giving equations for these
    quadrics. After rearranging if necessary, we may assume the singular locus
    of Q1 has maximum dimension among Q1, Q2, Q3. Fixing Q1, we may now replace
    Q2 with any member of its orbit under the orthogonal group O(Q1). So in
    order to search for genus-5 curves of this shape, we use the classification
    of quadratic forms to get a very small list of standard forms for Q1. This
    function computes O(Q1)-orbit representatives for Q2.

  - Over GF(2), we can take Q1 among 
   
    ..math:: vw + x^2, vw + xy, vw + x^2 + xy + y^2, or vw + xy + z^2.

    The invariant pairs for these four quadrics are (1,15), (0,19), (0,11),
    (-1,15).

  """
  Q1 = quadratic_form  
  R = Q1.parent()
  FF = R.base_ring()
  FFcross = [t for t in FF if t != 0]
  q = FF.cardinality()
  if R.ngens() != 5:
    raise ValueError('{} does not describe a quadric in PP^4'.format(Q1))
  XX = vector(R.gens())

  # Open outfiles early
  if sage_outfile is not None:
    fp = open(sage_outfile,'w')
  if sing_outfile is not None:
    gp = open(sing_outfile,'w')
    gp.write('list q2s;\n')

  # Get other invariant data
  sdp = open(quadric_sing_data,'r')
  pcp = open(quadric_point_count_data,'r')

  # Construct an array of correct length and
  # set the bit for each quadric with allowed invariants
  quads = homogeneous_monomials(R,2)
  num_pols = (q**len(quads) - 1)//(q-1)
  allowed_polys = [0]*num_pols # 1 means allowed
  num_allowed = 0
  q = FF.cardinality()
  print('Constructing array of allowed quadrics.')
  prog = progress.Progress(num_pols,**kwargs)
  for i in range(num_pols):
    prog()
    a,b = int(sdp.readline()), int(pcp.readline())
    if (a,b) in allowed_invariants:
      allowed_polys[i] = 1
      num_allowed += 1
  prog.finalize()
  from sys import getsizeof
  num_megs = getsizeof(allowed_polys)*1.0 / 2**20
  msg = 'Found {} allowed quadrics ({:.2f}MB)'
  print(msg.format(num_allowed,num_megs))

  # Clean up files
  sdp.close()
  pcp.close()

  # Get orthogonal group
  print('Restoring orthogonal group ...')
  if orthog_grp_data is not None:
    OQ = restore_matrices(FF,orthog_grp_data)
  else:
    OQ = orthogonal_group(Q1,projective=True)  
  print('Done!')
  
  # Set up list of processed quadrics with orbit of Q1
  quads_seen = [0]*num_pols
  Q1_ind = poly_index(Q1,quads,projective=True)
  quads_seen[Q1_ind] = 1
  num_allowed_seen = 1 # Keep track of how many allowed bits are set

  # find quadric ideal generator representatives modulo OQ
  prog = progress.Progress(num_allowed,steps_until_print=len(OQ))
  Q2s = []
  print('Computing ideal generator representatives among {} quadratic forms'.format(num_pols))
  field_elts = list(field_elements(FF))
  for i,Q2 in enumerate(poly_iterator(quads,projective=True)):
    if num_allowed_seen == num_allowed: break
    if Q2 == Q1: continue
    seen_Q2 = False
    good_translates = [0]*len(field_elts)
    # Test quadratic elements of the ideal (Q1,Q2)
    for j,a in enumerate(field_elts):
      Q = a*Q1 + Q2
      Q_ind = poly_index(Q,quads,projective=True)
      if quads_seen[Q_ind]:
        seen_Q2 = True
        break
      if allowed_polys[Q_ind]:
        good_translates[j] = 1
    if seen_Q2: continue # We have seen this ideal before.
    if good_translates.count(1) == 0: continue # No good invariants here
    # Record entire orbit of Q = a*Q1 + Q2 for each good a
    for g in OQ:
      pol = matrix_act_on_poly(g,Q2)
      for good,a in zip(good_translates,field_elts):
        if not good: continue
        Q = a*Q1 + pol      
        Q_ind = poly_index(Q,quads,projective=True)
        # We may have already seen this one: nontrivial stabilizers
        if not quads_seen[Q_ind]:
          quads_seen[Q_ind] = 1
          num_allowed_seen += 1
          prog()
    if good_translates.count(1) == len(field_elts):
      # Every translate has the right invariants
      Q2s.append(Q2)        
  prog.finalize()
  print('Found {} quadratic forms for Q2\n'.format(len(Q2s)))

  if sage_outfile is not None:
    for Q2 in Q2s:
      fp.write('{}\n'.format(Q2))
    fp.close()
  if sing_outfile is not None:
    for i, Q2 in enumerate(Q2s,1):
      gp.write('q2s[{}] = {};\n'.format(i,Q2))
    gp.close()
  return Q2s

################################################################################

def genus5_nontrigonal_curve_search(polring, q1_file, q2_file,
                                       allowed_invariants_file, outfile,
                                       quadric_sing_data,
                                       quadric_point_count_data,
                                       num_jobs=1, job=0,
                                       **kwargs):
  r"""
  Search for genus-5 nontrigonal curves in a particular data set

  INPUT:

  - ``polring`` -- a polynomial ring GF(q)[v,w,x,y,z]

  - ``q1_file`` -- a filename containing the string representation of
    a single quadratic form ``Q1``, in the variables given by polring

  - ``q2_file`` -- filename containing quadratic forms, one on each line

  - ``allowed_invariants_file`` -- filename containing the allowed invariant
    pairs for any quadratic form in the span of Q1,Q2,Q3; each line of the file
    should be of the form 'a, b\n', where `a` is the singular locus dimension,
    and `b` is the number of rational points.

  - ``outfile`` -- file for writing output of this function

  - ``quadric_sing_data`` --  filename giving classification of quadric
    singularities as written by quadric_singularity_dimension

  - ``quadric_point_count_data`` --  filename giving the number of
    points on each quadric hypersurface as written by quadric_point_count

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - Reamining arguments are passed to progress.Progress class

  EFFECT:

  - For each triple ``(Q1,Q2,Q3)`` such that 

    * Q2 is among the forms in ``q2_file``

    * V(Q1,Q2,Q3) is a smooth curve of genus-5

    * Every nonzero member of the linear span of Q1,Q2,Q3 has singularity
      dimension and point-count given by one of the lines in allowed_invariants_file

    a line is written to ``outfile`` in the form 'Q2, Q3\n'

  NOTE:

  - One use of this function is to search for all nontrigonal curves of genus
    5. By disallowing the invariants for the quadric surfaces vw + x^2 and vw +
    xy, we restrict our attention to curves of gonality at least 5. 

    
  """
  R = polring
  if len(R.gens()) != 5:
    raise ValueError('Polynomialring does not have 5 generators')
  v,w,x,y,z = R.gens()
  FF = R.base_ring()
  if not FF.is_finite():
    raise ValueError('Base ring is not finite')

  # open output file
  gp = open(outfile,'w')  
  
  # get input from files
  fp = open(q1_file,'r')
  Q1 = R(fp.readline())
  fp.close()

  Q2s = []
  fp = open(q2_file,'r')
  for Q in fp:
    Q2s.append(R(Q))
  fp.close()
  print('Computing nontrigonal genus-5 curves lying on {} = 0'.format(Q1))
  print('Using {} forms for Q2'.format(len(Q2s)))

  allowed_invariants = set()
  fp = open(allowed_invariants_file,'r')
  for line in fp:
    a,b = [int(t) for t in line.strip().split(',')]
    allowed_invariants.add((a,b))
  fp.close()
  print('Allowing invariants {} for quadratic_forms.'.format(list(allowed_invariants)))
  
  
  # Get other invariant data
  sdp = open(quadric_sing_data,'r')
  pcp = open(quadric_point_count_data,'r')
    
  # Construct lookup table for polynomial invariants
  allowed_polys = set()
  q = FF.cardinality()
  quads = homogeneous_monomials(R,2)
  num_pols = (q**len(quads) - 1)//(q-1)  
  print('Constructing set of allowed quadrics.')
  prog = progress.Progress(num_pols,**kwargs)
  for i,Q in enumerate(poly_iterator(quads,projective=True)):
    prog()
    a,b = int(sdp.readline()),int(pcp.readline())
    if (a,b) in allowed_invariants:
      allowed_polys.add(Q)
  prog.finalize()
  from sys import getsizeof
  num_megs = getsizeof(allowed_polys)*1.0 / 2**20
  msg = 'Found {} allowed conics ({:.2f}MB)'
  print(msg.format(len(allowed_polys),num_megs))

  # Clean up some memory (I hate doing this, but my jobs keep crashing)
  del allowed_invariants
  sdp.close()
  cpc.close()

  start,stop = start_and_stop_work(num_pols, num_jobs, job)
  if num_jobs > 1:
    print('Start index = {}, Stop index = {}'.format(start,stop))  
  
  num_pairs = len(Q2s) * (stop-start)
  print('Searching {} forms for Q3'.format(stop-start))
  print('Total pairs to search: {}\n'.format(num_pairs))

  # Normalize a conic so it matches the output of poly_iterator(...,projective=True)
  quad_coefs = [mon.dict().keys()[0] for mon in quads]
  def normalize_form(Q):
    D = Q.dict()
    for coefs in quad_coefs:
      if coefs in D:
        c = D[coefs]
        return Q / c
    raise RuntimeError('Q = {} could not be normalized'.format(Q))

  def batch_write(fp,L,num_to_write):
    msg = ''
    for j,(A,B) in zip(range(num_to_write),L):
      msg += '{}, {}\n'.format(A,B)
    fp.write(msg)
  
  # Loop over Q2,Q3 pairs and test
  PP2_pts = list(proj_space_point_iterator(FF,2))
  prog = progress.Progress(num_pairs,**kwargs)
  curves = [] # Only keep 512 before dumping to file
  for Q2 in Q2s:
    for Q3 in poly_iterator(quads,projective=True,start=start,stop=stop):
      prog()
      Qs = [Q1,Q2,Q3]
      good_triple = True
      for aa in PP2_pts:
        Q = sum(a*pol for a,pol in zip(aa,Qs))
        if Q == 0: continue
        Q = normalize_form(Q)
        if Q not in allowed_polys:
          good_triple = False
          break
      if not good_triple: continue
      # Now the invariants for all Q's in the span are ok.

      # Test irreducibility and smoothness
      I = R.ideal(Qs)
      if ideal_dimension(I) != 2: continue

      # check for a singularity other than the affine cone point
      inds = itertools.product(range(3),range(5))
      D = {(i,j): Qs[i].derivative(R.gens()[j]) for i,j in inds}
      J = matrix(R,3,5,D)
      sing_ideal = R.ideal(Qs+J.minors(3))
      if ideal_dimension(sing_ideal) != 0: continue
      # Now V(I) is a smooth curve, given as the intersection of 3 quadrics.
      # The only thing left that could fail is that C is not irreducible.
      if not ideal_is_prime(I): continue
      curves.append((Q2,Q3))
      if len(curves) == 512:
        batch_write(gp,curves,512)
        curves = []
  batch_write(gp,curves,len(curves))
  prog.finalize()
  gp.close()
  print('Done!')
  return curves

################################################################################

def count_genus5_points(q1, filename, outprefix=None):
  r"""
  INPUT:

  - ``q1`` -- a quadratic form in GF(q)[v,w,x,y,z]

  - ``filename`` -- string describing name of a file containing pairs of
    quadratic forms in 5 variables, one pair per line, separated by a comma

  - ``outprefix`` -- optional string

  OUTPUT:

  - a dict whose keys are nonnegative integers and whose i-th value is the list
    of all triples (q1,q2,q3) such that the number of points in PP^4(GF(q)) satisfying 
    `q1 = q2 = q3 = 0` is i.

  - a list of all bad lines in the input file. 

  EFFECT:

  - If outprefix is not None, then for each i, a file will be written with name
    'outprefix.i' that contains all pairs "q2, q3", one per line, such that 
    `q1 = q2 = q3 = 0` has i points on it.

  """
  R = q1.parent()
  FF = R.base_ring()
  q = FF.cardinality()
  if len(R.gens()) != 5:
    raise ValueError
  
  q1_pts = proj_point_search(q1)
  print('Searching for points on q1 = {}'.format(q1))
  print('Found {} points on q1 = 0'.format(len(q1_pts)))

  fp = open(filename,'r')
  L = fp.readlines()
  fp.close()

  # Arrange to have about 10 lines of printing  
  D = {}
  prog = progress.Progress(len(L),steps_until_print=len(L)/100)
  bad_lines = []
  for line in L:
    prog()
    try:
      q2,q3 = [R(t) for t in line.strip().split(',')]
    except:
      bad_lines.append(line)
      continue
    num_pts = len(proj_point_search([q2,q3],q1_pts))
    if D.get(num_pts) is None:
      D[num_pts] = []
    D[num_pts].append((q1,q2,q3))
  prog.finalize()

  if outprefix is not None:
    for i in D:
      fp = open('{}.{}'.format(outprefix,i),'w')
      for _,q2,q3 in D[i]:
        fp.write('{}, {}\n'.format(q2,q3))
      fp.close()
  return D, bad_lines

################################################################################

def genus5_g14_invariants(polring):
  r"""
  Compute the singular dimension and number of rational points of vw + x^2 and vw + xy

  INPUT:

  - ``polring`` -- a polynomial ring over a finite field in 5 variables v,w,x,y,z

  OUTPUT:

  - A list of ordered pairs `[(a0,b0),(a1,b1)]` such that

    * `a0` is the dimension of the singular locus of `vw + x^2 = 0` and `b0` is
      the number of rational points on this quadric hypersurface;

    * `a1` is the dimension of the singular locus of `vw + xy = 0` and `b1` is
    the number of rational points on this quadric hypersurface.

  NOTE:

  - A non-trigonal canonically embedded genus-5 curve admits a g^1_4 if and only
    if it lies on one of these two quadratic hypersurfaces.

  - The point counts for easily computed by elementary means, and are given in
    the docstring for quadric_point_count

  """  
  XX = polring.gens()
  if len(XX) != 5:
    raise ValueError('Argument is not a polynomial ring in five variables')
  FF = polring.base_ring()
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()

  N = q**3 + q**2 + q + 1
  return [(1,N),(0,N + q**2)]

################################################################################

def has_gonality_four(pols, g14_invariants=None):
  r"""
  Test if a smooth intersection of 3 quadrics has a g^1_4

  INPUT:

  - ``pols`` -- a list of three quadratic forms over a finite field whose
    intersection defines a smooth curve `C` of genus 5 in PP^4 = Proj GF(q)[v,w,x,y,z]

  - ``g14_invariants`` -- optional list of pairs `[(a0,b0),(a1,b1)]` as in the
    output to genus5_g14_invariants

  OUTPUT:

  - True if `C` has gonality 4 and False otherwise. 

  NOTE: 

  - Provided `C` is non-trigonal, it has gonality 4 if and only if lies on one
    of `vw + x^2 = 0` or `vw + xy = 0`.

  - The argument ``g14_invariants`` is there for when one wants to test if
    many curves have a g^1_4. These invariants can be computed with the function
    genus5_g14_invariants

"""
  pols = tuple(pols)
  if len(pols) != 3:
    raise ValueError('Wrong number of quadrics')
  R = pols[0].parent()
  FF = R.base_ring()
  
  if g14_invariants is None:
    g14_invariants = genus5_g14_invariants(R)

  for aa in proj_space_point_iterator(FF,2):
    Q = sum(a*pol for a,pol in zip(aa,pols))
    dim = quadric_singularity_dimension(Q)
    num_pts = quadric_point_count(Q)
    if (dim,num_pts) in g14_invariants:
      return True
  return False

################################################################################

def test_curves_for_gonality_five(q1, filename, outfileprefix=None, **kwargs):
  r"""
  Test a file's worth of non-trigonal, non 4-gonal genus-5 curves for gonality 4

  INPUT:

  - ``q1`` -- a quadratic form to be used for the first quadric

  - ``filename`` -- string; name of the file containing information for q2, q3,
    one pair per line in the form "q2, q3"

  - ``outfileprefix`` -- string; used for dumping results

  - Remaining keyword arguments are passed to the progress module

  OUTPUT:

  - A list of tuples `(q1,q2,q3)` giving curves with gonality 5

  - A list of tuples `(q1,q2,q3)` giving curves with gonality 6

  EFFECT:

  - If ``outfileprefix`` is specified, then files will be written as follows:

    * ``outfileprefix_gonality5.data`` -- for each triple (q1,q2,q3) for which
      it has been determined that the corresponding curve has gonality 5, a line
      in the output file will be written in the form "q2, q3\n"

    * ``outfileprefix_gonality6.data`` -- for each triple (q1,q2,q3) for which
      no g^1_5 has been located, a line is written in the form "q2, q3\n".

  NOTE:

  - A curve as given has gonality 5 if and only if it possesses a cubic point

  """
  print('Restoring curves from disk ...')
  curves = restore_complete_intersections(filename,q1)
  print('Done!')

  R = q1.parent()
  FF = R.base_ring()
  q = FF.cardinality()
  FF3 = GF(q**3)

  print('Computing cubic points on quadric hypersurface {} = 0'.format(q1))
  cubic_pts_all = proj_point_search(q1.change_ring(FF3))
  cubic_pts_reps = set()
  cubic_pts_seen = set()
  for pt in cubic_pts_all:
    if pt in cubic_pts_seen: continue
    cubic_pts_reps.add(pt)
    for i in range(3):
      cubic_pts_seen.add(tuple(u**(q**i) for u in pt))
  print('Done!')

  print('Checking gonality of {} curves.'.format(len(curves)))
  prog = progress.Progress(len(curves),**kwargs)
  fives = []
  sixes = []
  for curve in curves:
    pts = proj_point_search(curve,points=cubic_pts_reps)
    if len(pts) > 0:
      fives.append(curve)
    else:
      sixes.append(curve)
    prog()
  prog.finalize()

  if outfileprefix is not None:
    fivefile = outfileprefix + '_gonality5.data'
    sixfile = outfileprefix + '_gonality6.data'

    fp = open(fivefile,'w')
    for _,q2,q3 in fives:
      fp.write('{}, {}\n'.format(q2,q3))
    fp.close()

    fp = open(sixfile,'w')
    for _,q2,q3 in sixes:
      fp.write('{}, {}\n'.format(q2,q3))
    fp.close()
  return fives, sixes

################################################################################

def genus5_trigonal_curves(field, filename=None, **kwargs):
  r"""
  Return all trigonal genus-5 irreducible plane quintics over ``field``, up to isomorphism

  INPUT:

  - ``field`` -- a finite field `F`

  - ``filename`` -- string giving file in which to dump output

  - Remaining keyword arguments are passed to Progress.progress

  OUTPUT:

  - A list of pairs `(pol,c)`, where `pol` is an irreducible quintic form over
    `F` that has a unique quadratic singularity at (0,0) and `c` is the number
    of linear automorphisms preserving that form. Every trigonal genus-5 curve
    is isomorphic to one and only one plane curve with a polynomial as returned
    by this function.

  EFFECT:

  - If ``filename`` is given, then the output polynomials are written to disk,
    one per line. Each line will also contain the number of automorphisms, 
    separated by a comma.

  NOTE:

  - Any trigonal genus-5 curve may be realized in P^2 as a plane quintic with a
    unique `F`-rational quadratic singularity. This representation is unique of
    to `PGL_3(F)` transformation because the `g^1_3` on such a curve is
    unique. We may assume our curve passes through (0,0,1). Up to equivalence,
    the quadratic forms vanishing at (0,0,1) are : `xy`, `x^2`, and `N(x,y)`,
    where `N` is a norm form for the unique quadratic extension of `F`. If `Q`
    is one of these quadratic forms, we will only consider polynomials of the
    form

      ..math::

        P = Q z^3 + g(x,y,z),
 
    where `g` is a quintic form that vanishes to order at least 3 at (0,0,1). If
    (0,0,1) is the unique singularity of the corresponding plane curve, and if
    the `y^3z^2`-term has nonzero coefficient when `Q = x^2`, then `P` defines a
    trigonal curve of genus 5.  If `O(Q)` is the orthogonal group inside
    `GL_2(F)` of one such form, then any linear transformation that preserves
    this type of equation has the form

       ..math::

           (  cA | 0 )
           (   B | 1 )

    where `A \in O(Q)`, `B` is an arbitrary row vector of length 2, and `c` is a
    nonzero element of the field.

  """
  FF = field
  if not FF.is_finite():
    raise ValueError('{} is not a finite field'.format(FF))
  q = FF.cardinality()

  if filename is not None:
    fp = open(filename,'w')

  R = PolynomialRing(FF,names=('x','y','z'))
  x,y,z = R.gens()
  monomials = homogeneous_monomials(R,5)
  monomials.remove(z**5) # Must pass through (0,0,1)
  monomials.remove(x*z**4) # Remove linear terms in x,y
  monomials.remove(y*z**4)
  monomials.remove(x*y*z**3) # Remove quadratic terms in x,y
  monomials.remove(x**2*z**3)
  monomials.remove(y**2*z**3)
  assert len(monomials) == 15

  # List of quadratic forms that determine the singularity type:
  # split node, cusp, nonsplit node
  Qs = [x*y*z**3, x**2 * z**3]
  S = PolynomialRing(FF,names=('T',))
  T = S.gen()
  for a in FF:
    if q % 2 == 0 and (T**2 + a*T + 1).is_irreducible():
      Qs.append((x**2 + a*x*y + y**2)*z**3)
      break
    elif q % 2 == 1 and (T**2 + a).is_irreducible():
      Qs.append((x**2 + a*y**2)*z**3)
      break

  # Construct subgroups of PGL3 that preserve the given singularity types
  # Any such element element has the form
  print('Computing linear groups preserving the singularity types ...')
  R2 = PolynomialRing(FF,names=('X','Y'))
  X,Y = R2.gens()
  Gs = []
  nonzero = [a for a in FF if a != 0]
  for Q in Qs:
    G = []
    # Use projective representatives of OQ so that there is no duplicate
    # among the values of cA for c \in F^* and A \in OQ
    POQ = naive_orthogonal_group(Q(X,Y,1),projective=True) 
    for A in POQ:
      for bb in itertools.product(FF,FF):
        for c in nonzero:
          B = matrix(FF,1,2,bb)
          g = block_matrix(FF,2,2,[[c*A,0],[B,1]])
          G.append(g)
    Gs.append(G)

  # Now we do 3 separate searches: one for the cuspidal case,
  # one for each of the split/nonsplit nodal cases
  print()
  msg = 'Searching for curves with a {} at (0,0,1) ...'
  sing_types = ['split node','cusp','nonsplit node']
  quintics = []
  n = len(monomials)
  for sing_type,Q,G in zip(sing_types,Qs,Gs):
    print(msg.format(sing_type))
    total_work = q**n
    prog = progress.Progress(total_work,**kwargs)
    print('Search 2^{:.2f} quintics'.format(float(log(total_work,2))))
    P2 = ProjectiveSpace(R)
    seen = set()
    for cc in affine_space_point_iterator(FF,n):
      prog()
      pol = Q + sum(c*term for c,term in zip(cc,monomials))
      if pol in seen: continue
      auts = 0
      for g in G:
        newpol = matrix_act_on_poly(g,pol)
        assert newpol.monomial_coefficient(Q) == 1
        seen.add(newpol)
        if newpol == pol: auts += 1
      if sing_type == 'cusp' and pol.monomial_coefficient(y**3*z**2) == 0: continue
      if not ideal_is_prime(R.ideal(pol)): continue
      sing_ideal = R.ideal([pol.derivative(u) for u in R.gens()] + [pol])
      Z = P2.subscheme(sing_ideal)
      if len(Z.irreducible_components()) > 1: continue
      quintics.append((pol,auts))
      if filename is not None:
        fp.write('{}, {}\n'.format(pol,auts))
    prog.finalize()
  if filename is not None:
    fp.close()
  return(quintics)

################################################################################

class IntersectionOfThreeQuadrics():
  r"""
  Class for working with canonically embedded genus-5 curves

  INPUT:

  - ``forms`` -- a list of three quadric forms in `GF(q)[v,w,x,y,z]`

  - ``check`` -- boolean (default: True); if True, test if the given forms cut
    out a smooth, irreducible curve; if so, it will have genus-5 by the
    adjunction formula

  """
  def __init__(self, forms, check=True):
    if len(forms) != 3:
      raise ValueError('Wrong number of forms!')
    R = forms[0].parent()
    FF = R.base_ring()
    if not FF.is_finite():
      raise ValueError('{} is not a finite field!'.format(FF))
    I = R.ideal(forms)
    if check:
      if ideal_dimension(I) != 2:
        raise ValueError('Given forms do not cut out a curve')
      if not ideal_is_prime(I):
        raise ValueError('Given forms do not cut out an irreducible variety')
      inds = itertools.product(range(3),range(5))
      D = {(i,j): forms[i].derivative(R.gens()[j]) for i,j in inds}
      J = matrix(R,3,5,D)
      sing_ideal = R.ideal(list(forms)+list(J.minors(3)))
      if ideal_dimension(sing_ideal) != 0:
        raise ValueError('Given forms do not cut out a smooth variety')
    self._polring = R
    self._base_ring = FF
    self._forms = tuple(forms)
    self._ideal = I

  def __repr__(self):
    msg = 'Genus-5 curve cut out by quadratic forms in {}'.format(self._polring)
    return msg

  def __str__(self):
    msg = 'Genus-5 curve over {} cut out by\n'.format(self._base_ring)
    for form in self._forms:
      msg += '  {}\n'.format(form)
    return msg

  def polring(self):
    return self._polring

  def base_ring(self):
    return self._base_ring

  def forms(self):
    return self._forms

  def ideal(self):
    return self._ideal

  def has_a_g14(self):
    g14_invariants = genus5_g14_invariants(self._polring)
    FF = self._base_ring
    for aa in proj_space_point_iterator(FF,2):
      Q = sum(a*pol for a,pol in zip(aa,self._forms))
    dim = quadric_singularity_dimension(Q)
    num_pts = quadric_point_count(Q)
    if (dim,num_pts) in g14_invariants:
      return True
    return False

  def has_a_g15(self):
    FF = self._base_ring
    q = FF.cardinality()
    S = PolynomialRing(FF,names=('T',))
    while True:
      minpol = S.random_element(3)
      if minpol.is_irreducible(): break
    FF3 = FF.extension(minpol,name='T')
    forms = [form.change_ring(FF3) for form in self._forms]
    pts = proj_point_search(forms)
    return len(pts) > 0
    
  def gonality(self):
    if hasattr(self,'_gonality'):
      return self._gonality
    if self.has_a_g14():
      self._gonality = ZZ(4)
    elif self.has_a_g15():
      self._gonality = ZZ(5)
    else:
      self._gonality = ZZ(6)
    return self._gonality

  def span_invariants(self):
    r"""
    Return a list of invariant types for the nonzero quadrics containing self
    """
    if hasattr(self,'_span_invariants'):
      return self._span_invariants
    FF = self._base_ring
    invs = []
    for aa in proj_space_point_iterator(FF,2):
      Q = sum(a*pol for a,pol in zip(aa,self._forms))
      dim = quadric_singularity_dimension(Q)
      num_pts = quadric_point_count(Q)
      invs.append((dim,num_pts))
    invs.sort()
    self._span_invariants = invs
    return invs

  def rational_points(self):
    if hasattr(self,'_rational_points'):
      return self._rational_points
    self._rational_points = proj_point_search(self._forms)
    return self._rational_points

  def number_of_rational_points(self):
    return len(self.rational_points())

  def form_with_invariants(self, sing_dim, num_pts):
    r"""
    Return a quadratic form in span of self._forms with given invariants
    """
    FF = self._base_ring
    for aa in proj_space_point_iterator(FF,2):
      Q = sum(a*pol for a,pol in zip(aa,self._forms))
      dim = quadric_singularity_dimension(Q)
      if dim != sing_dim: continue
      num = quadric_point_count(Q)
      if num != num_pts: continue
      return Q
    return None

  def automorphism_group(self):
    r"""
    Return the automorphism group of self in PGL_5
    """
    raise NotImplementedError('This implementation is terrible!')
    FF = self._base_ring
    I = self._ideal
    forms = self._forms
    G = []
    for aa in itertools.product(FF,repeat=25):
      g = matrix(FF,5,5,aa)
      if not g.is_invertible(): continue
      is_good = True
      for Q in forms:
        if not matrix_act_on_poly(g,Q) in I:
          is_good = False
          break
      if is_good:
        G.append(g)
    return G

################################################################################
############################  HYPERELLIPTIC CURVES  ############################
################################################################################

def odd_characteristic_hyperelliptic_search(FF, g, filename=None, **kwargs):
  r"""
  Search for hyperelliptic curves over FF

  INPUT:

  - ``FF`` -- a finite field of odd characteristic

  - ``g`` -- an integer > 1

  - ``filename`` -- string giving file in which to dump output

  - Remaining keywords are passed to the progress.Progress class init

  OUTPUT:

  - A list of pairs `(P(x),n)` such that `y^2 = P(x)` describes a hyperelliptic
    curve of genus ``g`` with `n` automorphisms, and such that any hyperelliptic
    curve of genus ``g`` over ``FF`` is given by exactly one such a polynomial.

  EFFECT:

  - If ``filename`` is given, then the output polynomials are written to disk,
    one per line. Each line will also contain the number of automorphisms, 
    separated by a comma.

  NOTE:

  - This is a naive search through all possible hyperelliptic equations.

  - The automorphism group is a subgroup of 

      .. math::

         (GL_2(FF) \times FF^*) / ( (a,b) : a^{g+1} = e ).

    The subgroup we're quotienting by has order (q-1) / gcd(g+1,q-1).

  """
  q = FF.cardinality()
  if q % 2 != 1:
    raise ValueError('Field cardinality must be odd!')
  
  GL2 = []
  for aa in affine_space_point_iterator(FF,4):
    M = matrix(FF,2,2,aa)
    if M.is_invertible():
      GL2.append(aa)
  R = PolynomialRing(FF,'x')
  x = R.gen()

  nonzero = [a for a in FF if a != 0]
  pairs = itertools.product(nonzero,nonzero)
  numtriv = ZZ((q-1) / gcd(q-1,g+1))
  
  if filename is not None:
    fp = open(filename,'w')
    
  seen = set()
  good = []
  prog = progress.Progress(q**(2*g+3),**kwargs)
  for aa in affine_space_point_iterator(FF,2*g+3):
    prog()
    P = R(aa)
    if P in seen: continue
    if P.degree() < 2*g+1: continue
    if gcd(P,P.derivative()) != 1: continue
    auts = 0
    for e in nonzero:
      for a,b,c,d in GL2:
        newP = R(e**(-2) * (c*x+d)**(2*g+2) * P( (a*x+b)/(c*x+d)))
        seen.add(newP)
        if P == newP: auts += 1
    assert auts % numtriv == 0, 'auts = {}, numtriv = {}'.format(auts,numtriv)
    auts = ZZ(auts/numtriv)
    if filename is not None:
      fp.write('{}, {}\n'.format(P,auts))
    good.append((P,auts))
  prog.finalize()

  if filename is not None:
    fp.close()
  return good

################################################################################

def even_characteristic_hyperelliptic_search(FF, g, filename=None, **kwargs):
  r"""
  Search for hyperelliptic curves over FF

  INPUT:

  - ``FF`` -- a finite field of even characteristic

  - ``g`` -- an integer > 1

  - ``filename`` -- string giving file in which to dump output

  - Remaining keywords are passed to the progress.Progress class init

  OUTPUT:

  - A list of triples `(P(x),Q(x),n)` where `y^2 + Q(x)*y = P(x)` describes a
    hyperelliptic curve of genus ``g`` with `n` automorphisms, and such that any
    hyperelliptic curve of genus ``g`` over ``FF`` is given by exactly one such
    equation.

  EFFECT:

  - If ``filename`` is given, then the output pairs are written to disk, one per
    line. Each line will also contain the number of automorphisms, separated by
    a comma.

  NOTE:

  - This is a naive search through all possible hyperelliptic equations.

  - The elements of the automorphism group that act trivially have the form

      .. math::

         (x,y) \mapsto (g(x), (ey + H) / (cx+d)^{g+1} )

    with `H = 0`, `g = \lambda * I`. It follows that `e = \lambda^{g+1}`, so the
    subgroup we need to quotient by has order (q-1) / gcd(g+1,q-1). 

  """
  q = FF.cardinality()
  if q % 2 != 0:
    raise ValueError('Field cardinality must be even!')
  GL2 = []
  for aa in affine_space_point_iterator(FF,4):
    M = matrix(FF,2,2,aa)
    if M.is_invertible():
      GL2.append(aa)
  R = PolynomialRing(FF,'x')
  x = R.gen()

  nonzero = [a for a in FF if a != 0]
  pairs = itertools.product(nonzero,nonzero)
  numtriv = ZZ((q-1) / gcd(q-1,g+1))

  if filename is not None:
    fp = open(filename,'w')
    
  seen = set()
  good = []
  prog = progress.Progress(q**(2*g+3 + g+2),**kwargs)
  for bb in affine_space_point_iterator(FF,g+2):
    Q = R(bb)
    Qprime = Q.derivative()
    for aa in affine_space_point_iterator(FF,2*g+3):
      prog()
      P = R(aa)
      if P.degree() < 2*g+1 and Q.degree() < g: continue
      if (P,Q) in seen: continue
      Pprime = P.derivative()
      # Check smoothness
      if gcd(Q,Qprime**2 * P + Pprime**2) != 1: continue
      bgplus1 = Q.monomial_coefficient(x**(g+1))
      if bgplus1 == 0:
        bg = Q.monomial_coefficient(x**g)
        atop = P.monomial_coefficient(x**(2*g+2))
        anext = P.monomial_coefficient(x**(2*g+1))
        if bg**2 * atop + anext**2 == 0: continue
      auts = 0
      for a,b,c,d in GL2:
        A = a*x + b
        B = c*x + d
        Q1 = R( B**(g+1)*Q(A/B) )
        P1 = R( B**(2*g+2)*P(A/B) )
        for HH in affine_space_point_iterator(FF,g+2):
          H = R(HH)
          for e in nonzero:
            Qnew = Q1 / e
            Pnew = (1/e)**2 * (H**2  + P1 + H*Q1)
            seen.add((Pnew,Qnew))
            if (P,Q) == (Pnew,Qnew): auts += 1
      assert auts % numtriv == 0, 'auts = {}, numtriv = {}'.format(auts,numtriv)
      auts = ZZ(auts/numtriv)
      good.append((P,Q,auts))
      if filename is not None:
        fp.write('{}, {}, {}\n'.format(P,Q,auts))
  prog.finalize()

  if filename is not None:
    fp.close()
  return good

################################################################################

def hyperelliptic_search(FF, g, filename=None, **kwargs):
  r"""
  Search for hyperelliptic curves over FF

  INPUT:

  - ``FF`` -- a finite field

  - ``g`` -- an integer > 1

  - ``filename`` -- string giving file in which to dump output

  - Remaining keywords are passed to the progress.Progress class init

  OUTPUT:

  - A list of pairs of polynomials `(P(x),Q(x))` such that `y^2 + Q(x)*y = P(x)`
    describes a hyperelliptic curve of genus ``g``, and such that any
    hyperelliptic curve of genus ``g`` over ``FF`` is given by exactly one such
    equation. 

  EFFECT:

  - If ``filename`` is given, then the output pairs are written to disk,
    one per line. 

  """
  q = FF.cardinality()
  if q == Infinity:
    raise ValueError('Field {} is not finite'.format(FF))
  if q % 2 == 0:
    return even_characteristic_hyperelliptic_search(FF,g,filename,**kwargs)
  elif q % 2 == 1:
    return odd_characteristic_hyperelliptic_search(FF,g,filename,**kwargs)
    
