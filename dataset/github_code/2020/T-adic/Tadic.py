r""" 

Functionality for exploring sharpness of lower bounds for totally T-adic
functions over GF(q). These are used to implement the algorithms in 

[FP] Faber, Xander and Petsche, Clay. Totally T-adic functions of small height.
     Preprint, arXiv:2003.05205 [math.NT].  (2020)

UTILITIES
- T_degree
- height
- reduce_T_power
- genus_bounds
- PGL2
- poly_transform
- affine_space_point_iterator_GF2
- start_and_stop_work

TESTS
- is_totally_T_adic
- special_is_totally_T_adic
- quick_tests

POLYNOMIAL GENERATION
- poly_generator
- poly_generator_GF2
- random_poly

POLYNOMIAL SEARCH
- naive_small_totally_T_adic_polys_GF2
- small_totally_T_adic_polys_GF2
- small_totally_T_adic_polys 

"""

from sage.all import *

from builtins import range
import itertools
import maclane
import math
import progress
import sys

################################################################################
#################################  UTILITIES  ##################################
################################################################################
    
def T_degree(f):
  r"""
  Return the T-degree of f

  INPUT:

  - ``f`` -- an element of GF(q)[T][x]

  OUTPUT:

  - The T-degree as a polynomial in GF(q)[T,x]

  """
  if f == 0:
    return -Infinity
  R = f.parent()
  K = R.base_ring()
  coefs = [K(c).numerator() for c in f.list()]
  return max(len(t.list())-1 for t in coefs)

################################################################################

def height(f):
  r"""
  Return the logarithmic height of a polynomial in GF(q)[T][x]

  INPUT:

  - ``f`` -- an element of GF(q)[T][x]

  OUTPUT:

  - The logarithmic height of `f`

  """
  return T_degree(f) / f.degree()

################################################################################

def reduce_T_power(f, n):
  r"""
  Reduce f modulo T^n for f in GF(q)[T][x]
  """
  R = f.parent()
  K = R.base_ring()
  T = K.gen()
  coefs = [K(c).numerator().list()[:n] for c in f.list()]
  coefs = [sum(c*T**i for c,i in zip(cc,range(n))) for cc in coefs]
  return R(coefs)

################################################################################

def genus_bounds(q, n):
  r"""
  Return the lower and upper genus bounds on a minimum-height element

  INPUT:

  - ``q`` -- a prime power

  - ``n`` -- a positive integer (the gonality)

  OUTPUT:

  - The ceiling of the quantity `\frac{(n-1)(q+1)}{2\sqrt{q}}`

  - The floor of the quantity `\frac{1}{2}(q+1)(n-1)^2 + \frac{1}{2}(q-1)(n-1)`

  """
  lower_bound = (n-1)*(q+1) / (2*sqrt(q))
  upper_bound = 0.5*(q+1)*(n-1)**2 + 0.5*(q-1)*(n-1)
  return ceil(lower_bound), floor(upper_bound)

################################################################################

def PGL2(field):
  r"""
  An iterator over elements of PGL_2(field)

  INPUT:

  - ``field`` -- a finite field

  OUTPUT:

  - Yield lists [a,b,c,d] such that the associated matrix
    matrix(2,2,[a,b,c,d]) varies over the `q(q+1)(q-1)` elements 
    of `PGL_2(field)`

  """
  if not field.is_finite():
    raise ValueError('{} is not a finite field'.format(field))
  # affine transformations [a,b,0,1] first
  FFq0 = field(0)
  FFq1 = field(1)
  for a in field:
    if a == 0: continue
    for b in field:
      yield [a,b,FFq0,FFq1]

  # matrices [a,b,1,d]
  for a in field:
    for d in field:
      t = a*d
      for b in field:
        if b == t: continue
        yield [a,b,FFq1,d]
        
  return

################################################################################

def poly_transform(f, M):
  r"""
  Transform the polynomial `f` by the PGL2(GF(q)) transformation given by M

  INPUT:

  - ``f`` -- a univariate polynomial `f` over GF(q)

  - ``M`` -- a list `[a,b,c,d]` of elements of GF(q) such that
    matrix(2,2,M) determines an element of `PGL_2(GF(q))`

  OUTPUT:

  - The polynomial ` (cx+d)^{\deg(f)} f((ax + b)/(cx+d))`

  """
  a,b,c,d = M
  R = f.parent()
  x = R.gen()
  # print(f,(a*x+b)/(c*x+d),f((a*x+b)/(c*x+d)))
  pol = (c*x+d)**f.degree() * f((a*x+b)/(c*x+d))
  return R(pol)

################################################################################

def affine_space_point_iterator_GF2(n, start=None, stop=None):
  r"""
  An iterator over points of `A^n(GF(2))`

  INPUT:

  - ``n`` -- a positive integer

  - ``start`` -- an optional nonnegative integer

  - ``stop`` -- an optional nonnegative integer

  OUTPUT:

  - iterator over  the `2^n` points  of `A^n(field), interpreted as  n-tuples of
    elements of `GF(2)`; if ``start`` or ``stop`` is specified, then they will
    be used as indices to determine where to start or stop iteration

  NOTE:

  - The order is lexicographic, using the ordering `0 < 1` on `GF(2)`. This
    agrees with the ordering in itertools.product.

  EXAMPLES::

    sage: L = list(affine_space_point_iterator_GF2(2))
    sage: M = list(itertools.product(GF(2),repeat=2))
    sage: L == M
    True

    sage: for a in range(5):
    ....:   for b in range(a,5):
    ....:     L = list(affine_space_point_iterator_GF2(2,start=a,stop=b))
    ....:     assert L == M[a:b]

  """
  N = 2**n
  FF = GF(2)

  if start is None or start < 0:
    start = 0
  if stop is None or stop > N:
    stop = N

  remaining_yield = stop - start
  if remaining_yield <= 0: return  

  # Easy case
  if start == 0:
    for aa in itertools.product(GF(2),repeat=n):
      if remaining_yield == 0: return
      yield aa
      remaining_yield -= 1
    return

  # Now start > 0, so we have to iterate manually.
  coefs = [0]*n
  i = -1
  while start > 0:
    rem = start % 2
    coefs[i] = rem
    start = (start - rem)//2
    i -= 1

  FF_list = [FF(0),FF(1)]
  while remaining_yield > 0:
    yield tuple(FF_list[c] for c in coefs)
    remaining_yield -= 1
    j = n - 1
    while j >= 0:
      if coefs[j] == 0:
        coefs = coefs[:j] + [coefs[j]+1] + [0]*(n-j-1)
        j = -1
        break
      else:
        j -= 1
  return

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
  if num_jobs < 1:
    raise ValueError('num_jobs = {} is invalid'.format(num_jobs))
  if job < 0 or job >= num_jobs:
    raise ValueError('job = {} is invalid'.format(job))

  q = int(total_work / num_jobs)
  r = total_work - q*num_jobs
  if job < r:
    start = (q+1)*job
    stop = (q+1)*(job+1)
  else:
    start = (q+1)*r + q*(job-r)
    stop = (q+1)*r + q*(job-r+1)
  return start, stop

################################################################################
###################################  TESTS  ####################################
################################################################################

def is_totally_T_adic(f):
  r"""
  Determine if f splits completely over GF(q)[[T]]

  INPUT:

  - ``f`` -- an element of GF(q)(T)[x]

  OUTPUT:

  - True if `f` splits completely over GF(q)[[T]]

  """
  R = f.parent()
  K = R.base_ring()
  T = K.gen()
  VV = maclane.function_field_decomposition(T,f)
  return len(VV.extvals()) == f.degree()


################################################################################

def special_is_totally_T_adic(n, v0, f):
  r"""
  Determine if f splits completely over GF(q)[[T]]

  INPUT:

  - ``n`` -- a positive integer

  - ``v0`` -- StageZeroValuation object extending the T-adic valuation

  - ``f`` -- an element of GF(q)(T)[x] of degree `(q+1)*n` with
    well-distributed roots as expected of a polynomial that is totally T-adic
    of height 1/(q+1)

  OUTPUT:

  - True if `f` splits completely over GF(q)[[T]]

  NOTE:

  - This performs an abortive version of MacLane's algorithm. The idea is that
    we expect most choices of `f` to fail to split completely.

  """
  R = f.parent()
  x = R.gen()
  K = R.base_ring()
  T = K.gen()
  F = K.constant_field()
  # points with valuation -1 first
  V = maclane.StageZeroValuation(x,-1,indval=v0)
  if len(V.decomposition(f)) < n: return False
  for u in F:
    V = maclane.StageZeroValuation(x+u,1,indval=v0) # well-distributed roots!
    if len(V.decomposition(f)) < n: return False
  return True  

  
################################################################################

_quick_tests_cache = {}

def quick_tests(f):
  r"""
  Quick tests to rule out that `f` is not totally T-adic of height `1/(q+1)`

  INPUT:

  - ``f`` -- a polynomial in GF(q)[T][x]

  OUTPUT:

  - Return True if there is `n>0` such that the leading coefficient of `f` is
    `T^n`, if the constant coefficient is `cT^n` for some `c` in `GF(q)`, if
    `\deg_T(f) = n`, and if the Newton polygon for `ord_T` of `f(x+u)` is the
    lower convex hull of the points
  
      ..math:: `(0,n), (n,0), (q*n,0), ((q+1)*n,n)`
 
    for each nonzero `u \in GF(q)`. Return False otherwise.

  NOTE:

  - The slopes are cached to make this faster. 

  """
  R = f.parent()
  x = R.gen()
  K = R.base_ring()
  T = K.gen()
  F = K.constant_field()
  q = F.cardinality()

  # find n first
  if f(0) == 0: return False
  n,s = f.degree().quo_rem(q+1)
  if s != 0: return False

  if f.leading_coefficient() != T**n: return False
  if f.list()[0] / T**n not in F: return False
  if T_degree(f) != n: return False
  
  right_slopes = _quick_tests_cache.get((q,n))
  if right_slopes is None:
    right_slopes = [ZZ(1)]*n + [ZZ(0)]*n*(q-1) + [ZZ(-1)]*n
    _quick_tests_cache[(q,n)] = right_slopes
  for u in F:
    g = f(x+u)
    if g(0) == 0: return False
    if g.newton_slopes(T) != right_slopes: return False

  return True


################################################################################
###########################  POLYNOMIAL GENERATION  ############################
################################################################################

def poly_generator(funfld, Tdegree, const_coef_nonzero=False):
  r"""
  Generate elements of GF(q)[T] of at most given degree

  INPUT:

  - ``funfld`` -- a univariate rational function field over GF(q)

  - ``Tdegree`` -- a positive integer

  - ``const_coef_nonzero`` -- boolean (default: False); whether to only
    generate polynomials with nonzero constant term

  """
  K = funfld
  F = K.constant_field()
  T = K.gen()

  if const_coef_nonzero:
    cc = (K(c) for c in F if c != 0)
  else:
    cc = (K(c) for c in F)
  for a in cc:
    for b in itertools.product(F, repeat=Tdegree):
      yield a + T*K(b)

################################################################################

def poly_generator_GF2(funfld, Tdegree, const_coef_nonzero=False):
  r"""
  Generate elements of GF(2)[T] of at most given degree

  INPUT:

  - ``funfld`` -- a univariate rational function field over GF(2)

  - ``Tdegree`` -- a positive integer

  - ``const_coef_nonzero`` -- boolean (default: False); whether to only
    generate polynomials with nonzero constant term

  """
  K = funfld
  F = K.constant_field()
  T = K.gen()
  
  if const_coef_nonzero:
    cc = [K(1)]
  else:
    cc = [K(0),K(1)]
  for a in cc:
    for b in itertools.product(F, repeat=Tdegree):
      yield a + T*K(b)
      
################################################################################

def random_poly(funfld, degree, const_coef_nonzero=False):
  r"""
  Generate a random element of GF(q)[T] of given degree

  INPUT:

  - ``funfld`` -- a univariate rational function field over GF(q)

  - ``degree`` -- a positive integer

  - ``const_coef_nonzero`` -- boolean (default: False); whether to only
    generate polynomials with nonzero constant term

  OUTPUT:

  - A random element of GF(q)[T] with given degree and restriction on the
    constant coefficient

  """
  K = funfld
  F = K.constant_field()
  T = K.gen()

  cc = F.random_element()  
  if const_coef_nonzero:
    while cc == 0:
      cc = F.random_element()  
  coefs = [cc] + [F.random_element() for i in range(degree)]
  return K(coefs)

################################################################################
#############################  POLYNOMIAL SEARCH  ##############################
################################################################################


def naive_small_totally_T_adic_polys_GF2(n):
  r"""
  Naive search for all totally T-adic polynomials over GF(2)[T] with small height

  INPUT:

  - ``n`` -- a positive integer (the gonality)

  OUTPUT:

  - set of all irreducible totally T-adic polynomial over GF(2)[T] of x-degree
    `3n` and height `1/3`

  NOTES:

  - The search space has size `f(n)` bits, where

      ..math:: f(n) = 2n^2 + 3n - 3

  - This function is only used for validating the output of the fancier version,
    called small_totally_T_adic_polys_GF2.

  """
  FF = GF(2)
  K = FunctionField(FF,names=('T',))
  R = PolynomialRing(K,names=('x',))
  T = K.gen()
  x = R.gen()
  v0 = maclane.function_field_decomposition(T,x).extvals()[0]
  
  xdegree = n*3
  poly_bits = 2*n**2 + 3*n - 3

  def GF2_quick_test(f,v0,r):
    if f(1) == 0: return False
    if v0(f(1/T)) < -r: return False
    if v0(f(T)) < 2*r: return False
    if v0(f(T+1)) < 2*r: return False
    return True


  coefs = []
  coefs.append((K(1),)) # constant coefficient  
  for i in range(1,n):
    coefs.append(poly_generator_GF2(K,i)) # negative slope terms
  coefs.append(poly_generator_GF2(K,n,True)) # vertex term
  for i in range(n+1,2*n):
    coefs.append(poly_generator_GF2(K,n)) # slope-0 terms
  coefs.append(poly_generator_GF2(K,n,True)) # vertex term
  for i in range(1,n):
    coefs.append(poly_generator_GF2(K,n-i)) # positives slope terms
  coefs.append((K(1),)) # leading coefficient
  print('Checking 2^{} polynomials\n'.format(poly_bits))
  
  prog = progress.Progress(2**poly_bits)
  D = {'quick test':0,'irreducible':0}
  S = set()
  for aa in itertools.product(*coefs):
    prog()    
    f = sum(T**(n-i) * aa[i] * x**i for i in range(n+1))
    f += sum(aa[i] * x**i for i in range(n+1,2*n+1))
    f += sum(T**i * aa[2*n+i] * x**(2*n+i) for i in range(1,n+1))
    if not GF2_quick_test(f,v0,n):
      continue
    D['quick test'] += 1
    # print(f)
    if not f.is_irreducible():
      continue
    D['irreducible'] += 1
    if not is_totally_T_adic(f):
      continue
    S.add(f)
  prog.finalize()
  print('{:5d} pass the quick tests'.format(D['quick test']))
  print('{:5d} are irreducible'.format(D['irreducible']))
  print('{:5d} are totally T-adic'.format(len(S)))
  return S


################################################################################

def small_totally_T_adic_polys_GF2(n, num_jobs=1, job=0, **kwargs):
  r"""
  Search for all totally T-adic polynomials over GF(2)[T] with small height

  INPUT:

  - ``n`` -- a positive integer (the gonality)

  - ``num_jobs`` -- number of jobs into which this computation will be broken (default: 1)

  - ``job`` -- index of this job in the full computation (default: 0)

  - Additional keyword arguments are passed to progress.Progress

  OUTPUT:

  - set of all irreducible totally T-adic polynomial over GF(2)[T] of x-degree
    `3n` and height `1/3`

  NOTES:

  - The naive search space has size `f(n)` bits, where

      ..math:: f(n) = 2n^2 + 3n - 3

    This uses a linear algebra approach to cut the naive search space down by
    about `4n` bits. 

  - This function is an implementation of Algorithm 2 in [FP], and is used
    for the cases n = 1, 2, 3

  """
  xdegree = 3*n
  poly_bits = 2*n**2 + 3*n - 3

  def make_poly(r, Tring, xring, coefs):
    r"""
    Construct a polynomial with special Newton poly from a list of coefficients
    """
    aa = []
    xx = coefs
    xx_ptr = 0
    T = Tring.gen()
    x = xring.gen()
    aa.append(Tring(1)) # constant coefficient  
    for i in range(1,r):
      # negative slope terms
      aa.append(sum(T**k * xx[xx_ptr+k] for k in range(i+1)))
      xx_ptr += i+1
    aa.append(1 + sum(T**k * xx[xx_ptr+k-1] for k in range(1,r+1))) # vertex term
    xx_ptr += r
    for i in range(r+1,2*r):
      aa.append(sum(T**k * xx[xx_ptr+k] for k in range(r+1))) # slope-0 terms
      xx_ptr += r+1
    aa.append(1 + sum(T**k * xx[xx_ptr+k-1] for k in range(1,r+1))) # vertex term
    xx_ptr += r    
    for i in range(1,r):
      aa.append(sum(T**k * xx[xx_ptr+k] for k in range(r-i+1)))
      xx_ptr += r-i+1
    aa.append(Tring(1)) # leading coefficient
    f = sum(T**(r-i) * aa[i] * x**i for i in range(r+1))
    f += sum(aa[i] * x**i for i in range(r+1,2*r+1))
    f += sum(T**i * aa[2*r+i] * x**(2*r+i) for i in range(1,r+1))  
    return f
  
  # Construct linear equations using a polynomial in several variables
  
  FF = GF(2)
  xx = ['x{}'.format(i) for i in range(poly_bits)]
  A = PolynomialRing(FF,names=xx)
  xx = A.gens()
  AT = PolynomialRing(A,names=('T',))
  RT = PolynomialRing(AT,names=('x',))
  T = AT.gen()
  x = RT.gen()
  f = make_poly(n,AT,RT,xx)

  eqns = []
  # Get info from f(T); note first r coefficients vanish
  coefs = f(T).dict()
  for i in range(n,2*n):
    if i not in coefs: continue
    eqns.append(coefs[i])
  # get info from f(T+1)
  coefs = f(T+1).dict()
  for i in range(2*n):
    if i not in coefs: continue
    eqns.append(coefs[i])
  # get info from f(1/T); use f^rev; note first n coefficients vanish
  frev = RT(x**(3*n) * f(1/x))
  coefs = frev(T).dict()
  for i in range(n,2*n):
    if i not in coefs: continue
    eqns.append(coefs[i])

  # Turn equations into a linear system and solve
  zero_vec = tuple([0]*(len(xx)))
  Arows = []
  b = []
  for eqn in eqns:
    vecs = set(tuple(term) for term in eqn.dict())
    if zero_vec in vecs:
      b.append(FF(1))
    else:
      b.append(FF(0))
    w = sum(vector(FF,v) for v in vecs)
    Arows.append(w)

  A = matrix(FF,Arows)
  sol = A.solve_right(vector(FF,b))
  ker_matrix = A.right_kernel().matrix()
  ker_matrix = ker_matrix.echelon_form() # For safety; solve_right should already do this.
  ker = ker_matrix.rows()

  # Construct solution polynomials and perform
  # irreducibility check and totally T-adic check
  K = FunctionField(FF,names=('T',))
  R = PolynomialRing(K,names=('x',))
  T = K.gen()
  x = R.gen()
  v0 = maclane.function_field_inductive_valuation(T,key_polynomial=x)

  start, stop = start_and_stop_work(2**len(ker),num_jobs,job)

  work = stop - start
  print('Checking 2^{:.1f} polynomials\n'.format(math.log(work,2)))
  if num_jobs != 1:
    print('  start = {}, stop = {}'.format(start,stop))
    sys.stdout.flush()

  prog = progress.Progress(work, **kwargs)
  irreducible_test = 0
  S = set()
  for coefs in affine_space_point_iterator_GF2(len(ker),start=start,stop=stop):
    prog()      
    cc = sol + sum(c*v for c,v in zip(coefs,ker))
    cc = tuple(K(c) for c in cc)
    f = make_poly(n,K,R,cc)
    prog.profile('make_poly')
    if f(1) == 0: continue
    # print(f)
    f_is_irreducible = f.is_irreducible()
    prog.profile('irreducible')
    if not f_is_irreducible: continue
    irreducible_test += 1
    f_is_totally_T_adic = special_is_totally_T_adic(n,v0,f)
    prog.profile('totally_T_adic')
    if not f_is_totally_T_adic: continue
    S.add(f)
  prog.finalize()
  print('{:5d} are irreducible'.format(irreducible_test))
  print('{:5d} are totally T-adic'.format(len(S)))
  return S

################################################################################f

def small_totally_T_adic_polys(q, n, trials=None, term_bound=None, **kwargs):
  r"""
  Find one or all totally T-adic polynomial over GF(q)[T] with small height

  INPUT:

  - ``q`` -- a rational prime power

  - ``n`` -- a positive integer (the gonality)

  - ``trials`` -- optional positive integer

  - ``term_bound`` -- an integer bound on the number of terms in the polynomial over GF(q)[T,x]

  - Additional keyword arguments are passed to progress.Progress

  OUTPUT:

  - If trials is None, return all irreducible totally T-adic polynomial over
    GF(q)[T] of x-degree n(q+1) and height 1/(q+1).

  - If trials is not None, return one such if it can be located in `trials`
    random tries, else None.

  NOTE:

  - This function is an implementation of Algorithm 1 in [FP].

  """
  FF = GF(q)
  K = FunctionField(FF,names=('T',))
  R = PolynomialRing(K,names=('x',))
  T = K.gen()
  x = R.gen()
  
  xdegree = n*(q+1)
  v0 = maclane.function_field_inductive_valuation(T,key_polynomial=x)
  w0 = maclane.function_field_decomposition(T,x).extvals()[0]
  def GF2_quick_test(f,r):
    if f(1) == 0: return False
    if w0(f(1/T)) < -r: return False
    if w0(f(T)) < 2*r: return False
    if w0(f(T+1)) < 2*r: return False
    return True    
  
  tests = []
  if term_bound is not None:
    num_terms = lambda f : sum(len(val.numerator().dict()) for val in f.dict().values())
    tests.append(lambda f: num_terms(f) <= term_bound)
  if q == 2:
    tests.append(lambda f : GF2_quick_test(f,n))
  else:
    tests.append(lambda f : quick_tests(f))
  tests.append(lambda f : f.is_irreducible())
  tests.append(lambda f : special_is_totally_T_adic(n,v0,f))

  if trials is not None:
    prog = progress.Progress(trials, **kwargs)
    for i in range(trials):
      f = R(1)
      for i in range(n):
        f *= T*x - random_poly(K,n,True)
      for u in FF:
        for i in range(n):        
          f *= x - u - T*random_poly(K,n-1,True)
      f = reduce_T_power(f,n+1)
      if all(test(f) for test in tests):
        prog.finalize()
        return f
      prog()        
    prog.finalize()
    return None

  coefs = []
  for i in range(n):
    coefs.append(poly_generator(K,n,True))
  for u in FF:
    for i in range(n):
      coefs.append(poly_generator(K,n-1,True))
  num_pols = (q-1)**(n + q*n) * q**(n*n + q*n*(n-1))
  poly_bits = log(num_pols,2)
  print('Checking 2^({:.1f}) polynomials\n'.format(float(poly_bits)))

  prog = progress.Progress(num_pols, **kwargs)
  S = set()
  for aa in itertools.product(*coefs):
    f = R(1)
    f *= prod(T*x - aa[i] for i in range(n))
    for i,u in enumerate(FF,1):
      f *= prod(x - u - T*aa[i*n+j] for j in range(n))
    f = reduce_T_power(f,n+1)
    if all(test(f) for test in tests):
      S.add(f)
    prog()
  prog.finalize()
  return S
