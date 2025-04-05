r""".

A version of Serre's explicit method, following Lauter [3] and Voight [5].
These computations are used for the results in [1]. 

TABLE OF CONTENTS:

- UTILITIES
  - real_roots_check
  - evaluate_pol
  - real_weil_pol_to_a_values
  - real_weil_pol
  - excessive_conditions
  - class_number_from_real_weil_pol
  - resultant_decomp_test
  - convert_weil_pol_to_real_weil_pol

- EXPLICIT METHOD
  - cvals
  - psi
  - N1_bound
  - random_ad_bound
  - a_bounds
  - find_real_weil_pols


REFERENCES:

  [1] Faber, Xander and Grantham, Jon and Howe, Everett. "On the Maximum
      Gonality of a Curve over a Finite Field". Preprint, 2022. 

  [2] Lauter, Kristin. "Non-Existence of a Curve over F_3 of Genus 5 with 14
      Rational Points. Proc. of the AMS., vol. 128, 1999, pp.369-374.

  [3] Lauter, Kristin. "Zeta functions of curves over finite fields with many
      rational points". Coding theory, cryptography and related areas, 2000,
      pp. 167-174.

  [4] Tsfasman, Michael and Vl\u{a}du\c{t}, Serge and Nogin, Dmitry.  "Algebraic
      geometric codes: basic notions". Mathematical Surveys and Monographs,
      vol. 139, Am. Math. Soc., 2007.

  [5] Voight, John. "Curves over finite fields with many points: an
      introduction".  Computational aspects of algebraic curves, Lecture Notes
      Ser. Comput. 13, 2005, pp. 124-144.

"""

from sage.all import *
from math import acos,cos,sqrt
from collections import defaultdict
import itertools
import random

################################################################################
#################################  UTILITIES  ##################################
################################################################################

def real_roots_check(f, q):
  r"""
  Determine if all roots of f are real and in range [-2sqrt(q),2sqrt(q)]

  INPUT:

  - ``f`` -- a polynomial with integer or real coefficients

  - ``q`` -- a positive integer

  OUTPUT:

  - True if `f` has all of its roots real and in the interval [-2sqrt(q),2sqrt(q)]

  NOTE:

  - Using Pari's polsturm function is about 10x faster faster than computing the
    roots of ``f`` with f.roots(ring=RR) and verifying that all of them lie in
    the appropriate interval.

  EXAMPLES::

    sage: R.<x> = ZZ[]
    sage: real_roots_check(x^2-2,1) # All roots in [-2,2]
    True
    sage: real_roots_check(x^2-2,0.25) # No root in [-1,1]
    False
    sage: real_roots_check(x^2+1,1) # No real root
    False

  """
  # Pari's polsturm counts *distinct* real roots, so we need f squarefree
  while f.discriminant() == 0:
    f,_ = f.quo_rem(gcd(f,f.derivative()))
  return pari(f).polsturm([-2*sqrt(q),2*sqrt(q)]) == f.degree()

################################################################################

def evaluate_pol(pol, pt):
  r"""
  Evaluate ``pol`` at ``pt``

  INPUT:

  - ``pol`` -- an element of a Sage PolynomialRing object `R[x_1,...,x_n]`

  - ``pt`` -- a list or tuple of length `n`

  OUTPUT:

  - The same output as pol(pt), which is the same as pol.__call__(pt), but much
    faster because it doesn't call Singular in the case of multivariate
    polynomials.

  EXAMPLES::

    sage: R.<x,y> = ZZ[]
    sage: f = x + y
    sage: pt = (1,2)
    sage: f(pt) == evaluate_pol(f,pt)
    True

  """
  R = pol.parent()
  if len(R.gens()) == 1:
    return pol(pt)  
  val = R.base_ring()(0)
  for term, c in pol.dict().items():
    val += c * prod(a**i for a,i in zip(pt,term) if i != 0)
  return val

################################################################################

def real_weil_pol_to_a_values(F, q, num_values):
  r"""
  Construct the a-values of a curve from its (hypothetical) real Weil polynomial

  INPUT:

  - ``F`` -- a monic polynomial of degree `g` with integer coefficients

  - ``q`` -- a prime power

  - ``num_values`` -- the number of values of `a_d` to return

  OUTPUT:

  - A dict whose keys are integers in range(1,num_vales+1) and whose value at
    `d` is `a_d`, as determine by the assumption that ``F`` is a real Weil
    polynomial

  NOTE:

  - We use two facts for this computation. First, the zeta function `Z(T)`
    satisfies
 
      ..math:

        P(T) / (1-T)(1-qT) = \prod_{d \ge 1} (1-T^d)^{-a_d},

    where `P` is a polynomial of degree `2g`. Second, the real Weil polynomial
    `F` satisfies 
  
      ..math::

        P(T) = T^g F(qT + 1/T).

    Substituting the second into the first, taking the logarithmic derivative,
    and simplifying gives

      ..math::

        d/dT log ( T^g F(qT + 1/T) / (1-T)(1-qT) ) = \sum_{n \ge 0} N_{n+1} T^n,

    where `N_n` is the number of `GF(q^n)`-rational points on the hypothetical
    curve.

  EXAMPLES::

    sage: R.<T> = ZZ[]
    sage: q = 8
    sage: F = T^2 + 7*T + 8 # Real Weil polynomial for y^2 + xy = x^5 + x
    sage: real_weil_pol_to_a_values(F,q,4)
    {1: 16, 2: 24, 3: 168, 4: 968}

  """  
  if not F.is_monic():
    raise ValueError('F = {} is not monic'.format(F))
  R = PowerSeriesRing(QQ,names=('T',),default_prec=num_values+1)
  T = R.gen()
  x = F.parent().gen()
  g = F.degree()
  
  # Log derivative of the zeta function and point counts over extensions
  log_deriv = (g/x) + (q-x**(-2))/F(q*x + 1/x) * F.derivative()(q*x + 1/x)
  log_deriv += 1/(1-x) + q / (1-q*x)
  log_deriv_series = log_deriv(T) 
  N_vals = dict((n,val) for n,val in enumerate(log_deriv_series.list(),1))
  
  a_vals = {}
  for n in range(1,num_values+1):
    a_val = (ZZ(1)/ZZ(n))*(N_vals.get(n,0) - sum(d*a_vals[d] for d in range(1,n) if n % d == 0))
    a_vals[n] = a_val
  return a_vals

################################################################################

def real_weil_pol(g, q, ads=None):
  r"""
  Construct a univariate polynomial F = T^g + b_1 T^{g-1} + ... from a_d's

  INPUT:

  - ``g`` -- a nonnegative integer

  - ``q`` -- a prime power

  - ``ads`` -- optional list or tuple `(a_1, ..., a_g)` of nonnegative integers

  OUTPUT:

  - Write the real Weil polynomial of a curve as

    ..math:: 

      F = \sum_{i=0}^g b_i T^{g-i}

    where the `b_i` coefficients lie in the polynomial ring 
    ZZ[a_1,...,a_g][T]. If the variable ``ads`` is unspecified, then 
    the `a_d`'s will lie in the polynomial ring ZZ[a_1,...,a_g].

  NOTE:

  - Let `u_1, ..., u_g` be the (real) roots of F. Define the power sum of the
    roots to be `s_n = \sum_{i=1}^g u_i^n`. Since the coefficients`b_i` of the
    real Weil polynomial are symmetric functions in the roots, Newton's
    identities tell us that

      ..math::

        0 = s_n + b_1 s_{n-1} + b_2 s_{n-2} + \cdots + b_{n-1} s_1 + n b_n. 

    The relationship between the value `N_n = #C(F_{q^n})` and the sum of powers
    of the roots of the zeta function allows us to recover the `s_i` from the
    `N_i`, and from there we can recover the `b_i`. See Lauter's paper [3] for
    more details.

  EXAMPLES::
  
    sage: real_weil_pol(2,8)
    T^2 + (a1 - 9)*T + 1/2*a1^2 - 17/2*a1 + a2 - 8

    sage: real_weil_pol(2,8,[16,24]) # a_d values for y^2 + xy = x^5 + x over GF(8)
    T^2 + 7*T + 8

  """
  if ads is not None:
    if len(ads) != g:
      raise ValueError('tuple {} does not have length g = {}'.format(ads,g))
    aa = (1,) + tuple(ads)
    A = QQ
    R = PolynomialRing(A,names=('T',))
  else:
    names = []
    for i in range(1,g+1):
      names.append('a{}'.format(i))
    A = PolynomialRing(QQ,names=tuple(names))
    R = PolynomialRing(A,names=('T',))
    aa = (1,) + A.gens() # Force aa[1] = a1, etc. 
  T = R.gen()

  def N(n):
    r""" Point count over GF(q^n) """
    return sum(d*aa[d] for d in ZZ(n).divisors())
  def s(n):
    r""" Power polynomials in the roots of the real Weil polynomial """
    rv = A(0)
    if n % 2 == 0:
      rv += binomial(n,n//2) * g * q**(n//2)
    limit = int(floor((n-1)/2))
    for j in range(limit+1):
      rv += binomial(n,j) * q**j * (q**(n-2*j) + 1 - N(n-2*j))
    return rv
  def b(n):
    r""" Coefficients of the real Weil polynomial """
    if n == 1:
      return -s(1)
    return (-ZZ(1)/ZZ(n))*(s(n) + sum(b(i)*s(n-i) for i in range(1,n)))
  
  return T**g + sum(b(i) * T**(g-i) for i in range(1,g+1))

################################################################################

def excessive_conditions(g):
  r"""
  Compute conditions on the a_d's that are necessary for an excessive curve

  INPUT:

  - ``g`` -- an integer > 1

  OUTPUT:

  - A minimal list of tuples of the form `(i_1, ..., i_m)` such that if `C` is a
    curve of genus g and gonality g+1 over GF(q), then `a_{i_j} = 0` for some 
    `j = 1, ..., m`. Minimal means that no tuple in the list is properly 
    contained in any other.

  NOTES:

  - We know that for g > 1, `C` admits an effective divisor `D` of degree `g`
    with `dim |D| > 0` if and only if it admits an effective divisor of degree
    `g-2`. For `g = 3`, this means we must have `a_1 = 0` in order to have
    gonality 4, while for `g = 5` it means we must have `a_1 = a_3 = 0` in order
    to have gonality 6. For `g = 7`, it means we must have `a_1 = a_5 = 0`, and
    also either `a_2 = 0` or `a_3 = 0`. Similar results hold for other genera. 

  - We compute this quantity by looking at integer partitions of `g-2`, as these
    give the degrees of the irreducible components of a hypothetical effective
    divisor of degree `g-2`. 

  EXAMPLES::

    sage: excessive_conditions(2)
    []
    sage: excessive_conditions(3)
    [(1,)]
    sage: excessive_conditions(4)
    [(1,), (2,)]
    sage: excessive_conditions(5)
    [(1,), (3,)]
    sage: excessive_conditions(7)
    [(1,), (2, 3), (5,)]   
    sage: excessive_conditions(9)
    [(1,), (2, 3), (3, 4), (2, 5), (7,)]

  """
  if g < 2:
    raise ValueError('Notion of excessive curve not defined for g < 2')
  if g == 2: return []

  conds = []
  for P in Partitions(g-2).__reversed__():
    Pset = set(P)
    # If Pset contains a conditions we've already seen
    if any(Pset.issuperset(set(x)) for x in conds):
      continue
    Plist = sorted(list(Pset))
    conds.append(tuple(Plist))
  return conds

################################################################################

def class_number_from_real_weil_pol(F, q, d=1):
  r"""
  Return the order of the class group of the curve over GF(q^d) with real Weil polynomial F

  INPUT:

  - ``F`` -- a monic polynomial of degree `g` with integer coefficients

  - ``q`` -- a prime power

  - ``d`` -- optional positive integer (default: 1)

  OUTPUT:

  - The quantity `F_d(q + 1)`, where `F` is the real Weil polynomial of a curve
    `C` and `F_d` is the real Weil polynomial of `C \otimes GF(q^d)`

  NOTE:

  - If `C` is a curve over GF(q) with Zeta function P(T) / (1-T)(1-qT), the real
    Weil polynomial for `C` is the unique polynomial `F` satisfying 
    
      ..math::

        P(T) = T^g F(qT + 1/T).

    The class number (i.e., number of rational points on the Jacobian) is given
    by `P(1) = F(q+1)`. See, e.g., the book [4], Proposition 3.1.16. 

  - If `d > 1`, then we first compute the numerator `P_d` of the Zeta function
    for `C \otimes GF(q^d)`. We use the formula in [4], Proposition 3.1.11. 

  EXAMPLES::

    sage: R.<T> = ZZ[]
    sage: q = 8
    sage: F = T^2 + 7*T + 8 # Real Weil polynomial for y^2 + xy = x^5 + x
    sage: class_number_from_real_weil_pol(F,q,1)
    152
    sage: class_number_from_real_weil_pol(F,q,3)
    265544

  """
  if d == 1:
    return F(q+1)

  R = F.parent()
  T = R.gen()
  g = F.degree()
  P = R(T**g * F(q*T + 1/T)) # numerator of the Zeta function for C
  
  K = CyclotomicField(d,'z')
  z = K.gen()
  Q = prod(P(z**i * T) for i in range(d)) # P_d(T^d) = \prod_{i=1}^d P(z^i * T)
  return ZZ(Q(1))

################################################################################

def resultant_decomp_test(F):
  r"""
  Determine if F = A*B with A,B nontrivial and res(A,B) = \pm 1

  INPUT:

  - ``F`` -- a monic integer polynomial in one variable
  
  OUTPUT:

  - True if `F = A*B` for some nontrivial `A,B` with `res(A,B) = \pm 1`, and
    False otherwise

  - The pair `(A,B)` if the first output is True, and None otherwise.

  NOTE:

  - The first example below comes from Lauter's paper [2]. 

  EXAMPLES::

    sage: R.<T> = ZZ[]
    sage: F = T^5 + 10*T^4 + 37*T^3 + 62*T^2 + 46*T + 12
    sage: resultant_decomp_test(F)
    (True, (T^2 + 4*T + 3, T^3 + 6*T^2 + 10*T + 4))

    sage: G = (T+2)*(T+4)
    sage: resultant_decomp_test(G)
    (False, None)

  """
  Ffacs = list(F.factor())
  n = len(Ffacs)
  if n == 1: return False, None
  # Choose up to half of the factors for A and test
  for i in range(1,1+n//2):
    for ff in itertools.combinations(Ffacs,i):
      A = prod(f**e for f,e in ff)
      B,_ = F.quo_rem(A)
      if A.resultant(B) in [1,-1]:
        return True, (A,B)
  return False, None

################################################################################

def convert_weil_pol_to_real_weil_pol(pol):
  r"""
  Return the real Weil polynomial associated to pol

  INPUT:

  - ``pol`` -- a monic polynomial of degree 2g with integer coefficients whose
    roots are complex conjugate q-weil numbers for some q

  OUTPUT:

  - the polynomial `h` such that `T^g h(T + q/T) = pol`

  EXAMPLES::

    sage: R.<T> = ZZ[]
    sage: pol = T^4 + 7*T^3 + 24*T^2 + 56*T + 64 # Weil polynomial for y^2 + xy = x^5 + x over GF(8)
    sage: convert_weil_pol_to_real_weil_pol(pol)
    T^2 + 7*T + 8

  """
  R = pol.parent()
  T = R.gen()
  x = T.change_ring(RR)
  roots = []
  for f,e in pol.change_ring(RR).factor():
    roots.append((-f.monomial_coefficient(x),e))
  h = prod((x-a)**e for a,e in roots)
  return R([a.round() for a in h.list()])

################################################################################
##############################  EXPLICIT METHOD  ###############################
################################################################################

def cvals(*us):
  r"""
  Return the c-coefficients in the explicit method from the u's

  INPUT:

  - ``u1, u2, ...`` -- the u-coefficients

  OUTPUT:

  - A list of the nonzero coefficients c1, c2, ... given by the formula

    ..math:: (1 + 2\sum_{r \geq 1} u_r \cos(r*t))^2 / (1 + 2\sum_{r \ge 1} u_r^2)
  
             = 1 + 2c_1 \cos(t) + 2c_2 \cos(2t) + 2c_3 \cos(3t) + \cdots.

  NOTE:

  - See Voight's article [5], Example 2.13. The given u-values do not produce
    the c-values he claims. We have corrected this mistake in the example below.

  - We implement the easily verified formula 

      ..math:: c_n =  \left(1 + \sum_{r \ge 1} u_r^2\right)^{-1} 
                      \left(u_{n/2}^2 \delta_{n \equiv 0 \pmod 2} 
                          + 2\sum_{\substack{i+j = n \\ i \ne j}} u_i u_j\right),

    where we set `u_0 = 1` and `u_{-r} = u_r` for all `r \ge 1`. 

  EXAMPLES::

    sage: cvals(1,3/4,1/4)
    (31/34, 12/17, 8/17, 1/4, 3/34, 1/68)
    
  """
  den = 1 + 2*sum(u**2 for u in us)
  n = len(us)
  cs = [1] # start with 1 so the numbering of the c_i's matches formula
  u = lambda r: 1 if r == 0 else us[abs(r)-1] # Extend u-sequence to negative indices
  for r in range(1,2*n + 1): # c_r = 0 for all r >= 2n+1
    num = 0
    s = (r+1)//2 # Starting index for one factor in sum defining c_r
    if r % 2 == 0:
      num += u(r//2)**2
      s += 1
    while s <= min(n,r+n):
      num += 2 * u(s) * u(r-s)
      s += 1
    cs.append(num / den)
  return tuple(cs[1:])

################################################################################

def psi(t, d, cs):
  r"""
  Evaluate the function Psi_d as in Serre's explicit method

  INPUT:

  - ``t`` -- a positive real number

  - ``d`` -- a positive integer

  - ``cs`` -- the list of c-coefficients as in Serre's explicit method

  OUTPUT:

  - The value of the function 

      ..math::

        Psi_d(t) = \sum_{r \ge 1} c_{rd} t^{rd}

    as in Serre's explicit method. See Voight's article [5], for example. 

  EXAMPLES::

    sage: us = [1, 3/4, 1/4]
    sage: cs = cvals(*us)
    sage: [psi(1,d,cs) for d in range(1,8)]
    [83/34, 33/34, 33/68, 1/4, 3/34, 1/68, 0]

  """
  Cs = [1] + list(cs) # Just to make the indexing as in the formula
  i = d
  val = 0
  while i <= len(cs):
    val += Cs[i]*t**i
    i += d
  return val

################################################################################

def N1_bound(q, *us):
  r"""
  Serre's explicit method coefficients for N_q(g)

  INPUT:

  - ``q`` -- a prime power

  - ``us`` -- the u-coefficients as explained by Serre

  OUTPUT:

  - ``a, b`` -- real numbers such that `N_q(g) \le a*g + b` as given by Serre

  NOTE:

  - Serre obtained the bound

      ..math:: N_q(g) = g / \Psi(1/\sqrt{q}) + 1 + \Psi(\sqrt{q}) / \Psi(1/\sqrt{q}).

    See [5], Corollary 2.12. Note that Voight's example computation is incorrect.

  EXAMPLES::

    sage: us = [1, 3/4, 1/4]
    sage: N1_bound(2,*us)
    (0.8038776304384084, 5.541105140493499)
    sage: a,b = N1_bound(2,*us)
    sage: floor(a*3 + b) # Bound for number of points on genus-3 curve over GF(2)
    7
    sage: floor(a*4+b) # Bound for number of points on genus-4 curve over GF(2)
    8
    sage: floor(a*5+b) # Bound for number of points on genus-5 curve over GF(2)
    9

  """
  cs = cvals(*us)
  a = 1 / psi(1/sqrt(q),1,cs)
  b = 1 +  psi(sqrt(q),1,cs)*a
  return a,b

################################################################################

def random_ad_bound(g, q, a_vector, u_bound=1, num_tries=256):
  r"""
  Return a bound for a_d given the values of a_r for r < d

  INPUT:

  - ``g`` -- a nonnegative integer

  - ``q`` -- a prime power

  - ``a_vector`` -- a tuple or list of consecute `a_d`'s of the form `(a_1, ..., a_r)`

  - ``u_bound`` -- bound for the interval in which to sample for u-values (default: 1)

  - ``num_tries`` -- number of random trials to execute (default: 256)

  OUTPUT:

  - A simple bound for `a_{r+1}` using Serre's explicit method with random
    choices of `u_i`, using the method of [5], Proposition 2.11.

  NOTE:

  - Based on empirical observation, we take the number of nonzero `u_i` to be
    between 1 and `r+2`, inclusive, split equally among the number of
    trials. This seems to give reasonably good results.

  EXAMPLES::

    sage: random_ad_bound(5,2,[])
    9
    sage: random_ad_bound(5,2,[4])
    7
    sage: random_ad_bound(5,2,[5])
    6

  """
  best_bound = Infinity
  r = len(a_vector)
  a_vector = (0,) + tuple(a_vector) # to get the indexing right
  for num_us in range(1,r+3): # This is more art than science
    tries = int(num_tries / (r+2))
    for _ in range(tries):
      us = [random.uniform(0,u_bound) for _ in range(num_us)]
      cs = cvals(*us)
      rhs = g + psi(sqrt(q),1,cs) + psi(1/sqrt(q),1,cs)
      lhs = sum(d*a_vector[d]*psi(1/sqrt(q),d,cs) for d in range(1,r+1))
      denom = (r+1)*psi(1/sqrt(q),r+1,cs)
      if denom == 0: continue
      bound = int((rhs - lhs)*1.0 / denom )
      best_bound = min(bound,best_bound)
  return best_bound

################################################################################

def a_bounds(g, q, N1, ads=None, us=None, u_bound=1, num_tries=256):
  r"""
  Explicit method bounds for the coefficients a_d, 1 < d \le g

  INPUT:

  - ``g`` -- a positive integer (genus)

  - ``q`` -- a prime power

  - ``N1`` -- a nonnegative integer

  - ``ads`` -- optional list of which a_d-values to return; if unspecified, all
    in the range (2,g+1) will be used

  - ``us`` -- optional u-coefficients as explained by Serre; if not specified, 
    then they will be generated uniformly at random from the interval [0,u_bound]

  - ``u_bound`` -- positive bound for the interval in which to sample for
    u-values (default: 1); this may also be given as a list, in which case the
    randomizer will run ``num_tries`` times for each value given

  - ``num_tries`` -- number of random trials to execute (default: 256)

  OUTPUT:

  - a dict whose keys are integers d with 1 < d \le g as specified by ``ads``,
    and whose value is the upper bound for a_d given by the explicit method, or
    Infinity if this choice of ``us`` gives rise to a zero value of
    Psi_d(q^{-1/2}). This bound may be sharp in some cases.

  - a dict whose keys are integers d with 1 < d \le g as specified by ``ads``,
    and whose value is the choice of ``us`` that yielded the given bound for a_d
    in the first output

  NOTE:

  - To compute a bound for `a_n`, we use the formula in Proposition 2.11 of [5],
    keeping only the terms on the left side with `d = 1` and `d = n`.

  EXAMPLES::

    sage: a_bounds(5, 2, 4, us=[1,3/4,1/4])
    {2: 8, 3: 13, 4: 27, 5: 88}

    sage: a_bounds(7, 2, 0, ads=[2,3]) # RANDOM
    ({2: 13, 3: 17},
    {2: [0.04170630273544018,
      0.5902264335132691,
      0.12166731321888324,
      0.1978799294962016],
     3: [0.1814544441504229,
      0.19932843475569717,
      0.8721272144824845,
      0.018032720461513163]})

  """
  if ads is None:
    ads = range(2,g+1)
  if us is not None:
    cs = cvals(*us)
    bds = {}
    for d in ads:
      tmp = d*psi(1/sqrt(q),d,cs)
      if tmp == 0:
        bds[d] = Infinity
        continue
      a = 1.0 / tmp
      b = ( psi(sqrt(q),1,cs) - (N1-1)* psi(1/sqrt(q),1,cs) ) / tmp
      bds[d] = int(a*g + b)
    return bds

  if isinstance(u_bound,(list,tuple,set)):
    u_bounds = list(u_bound)
  else:
    u_bounds = [u_bound]
  best_bds = defaultdict(lambda : Infinity)
  best_us = {}
  for num_us in range(1,g+3): # This is more art than science
    tries = int(num_tries / (g+2))
    for _ in range(tries):
      for umax in u_bounds:
        us = [random.uniform(0,umax) for _ in range(num_us)]
        bds = a_bounds(g,q,N1,ads=ads,us=us)
        for d,bd in bds.items():
          if bd < best_bds[d]:
            best_bds[d] = bd
            best_us[d] = us
  return dict(best_bds), best_us

################################################################################

def find_real_weil_pols(g, q, N1, known_a_vals=None, excessive=False,
                         initial_tries=1024, filename=None, verbose=True,
                        **kwargs):
  r"""
  Return the list of real Weil polynomialswith given ``g, q, N1``, etc.

  INPUT:

  - ``g`` -- a nonnegative integer

  - ``q`` -- a prime power

  - ``N1`` -- a nonnegative integer giving the number of rational points on the
    curve

  - ``known_a_vals`` -- an optional dict whose keys are positive integers d in
    the range 1 \le d \le g and whose values are nonnegative integers that
    specify values of a_d to be used in the calculation. If `a_1` is specified,
    it must agree with N1.

  - ``excessive`` -- use pointless properties of excessive curves to determine
    polynomials

  - ``initial_tries`` -- number of tries to use for setting initial bounds
    (default: 1024)

  - ``filename`` -- optional path to use for writing results to file, one 
    polynomial per line

  - ``verbose`` -- bool (default: True); print status of the computation

  - Remaining keyword arguments are passed to a_bound and random_ad_bound

  OUTPUT:

  - A complete list of real Weil polynomials that correspond to hypothetical
    curves of genus `g` over `GF(q)` with the given values of `a_d`. If
    ``excessive`` is True, then additional values of `a_d` will
    be screened using this condition.

  NOTE:

  - We use Lauter's refinement of Serre's explicit method. 

  EXAMPLES::

    sage: Fs = find_real_weil_pols(5,2,0,excessive=True,verbose=False)
    sage: len(Fs)
    45

    sage: find_real_weil_pols(3,5,0,known_a_vals={2:8},verbose=False)
    [T^3 - 6*T^2 - 2*T + 40]

    # The hyperelliptic genus-3 curve over GF(11) given by 
    # y^2 = 2*x^8 + 4*x^6 - 2*x^4 + 4*x^2 + 2 is pointless and has a_2 = 70.
    sage: find_real_weil_pols(3,11,0,known_a_vals={2:70},verbose=False)
    [T^3 - 12*T^2 + 48*T - 64]

  """
  # Open output file early in case of problem
  if filename is not None:
    fp = open(filename,'w')
  else:
    fp = None
  
  if known_a_vals is None:
    known_a_vals = {}

  if 1 in known_a_vals and N1 != known_a_vals[1]:
    msg = 'N1 = {} does not agree with a_1 = {}'
    raise ValueError(msg.format(N1,known_a_vals[1]))

  a_vals = {1:N1}
  for d,val in known_a_vals.items():
    a_vals[d] = val

  # Get pointless conditions for an excessive curve and set appropriate a_vals
  if not excessive:
    conds = []
  else:
    conds = excessive_conditions(g)
    for d in range(1,g-1):
      if (d,) not in conds: continue
      if d in known_a_vals and known_a_vals[d] != 0:
        msg = 'Specified value a_{} = {} conflicts with excessive curve conditions'
        raise ValueError(msg.format(d,known_a_vals[d]))
      a_vals[d] = 0
      conds.remove((d,))

  # Set up checking for excessive curve pointless properties
  # A condition (i1, ..., ir) only needs to be checked for a_vector
  # if the max_index equals ir (since the condition is not enforceable if
  # ir > max_index and it is already forced to hold if ir < max_index).
  if len(conds) == 0:
    seems_excessive = lambda a_vector,max_index : True
  else:
    def seems_excessive(a_vector, max_index):
      for cond in conds:
        if max_index != max(cond): continue
        # a_vector has 0-based indexing
        if all(a_vector[d-1] != 0 for d in cond): return False
      return True

  # Print the requirements we've place on the a_d's
  if verbose:
    amsg = ''
    num_vals = len(a_vals)
    for i,(key,val) in enumerate(a_vals.items(),1):
      amsg += 'a_{} = {}'.format(key,val)
      if i < num_vals - 1:
        amsg += ', '
      elif i == num_vals - 1:
        if num_vals == 1:
          amsg += '.'
        elif num_vals == 2:
          amsg += ' and '
        else:
          amsg += ', and '
      else:
        amsg += '.'
    msg =  '\nComputing real Weil polynomials for g = {}, q = {},'
    msg += '\n  and a-invariants {}'
    print(msg.format(g,q,amsg))
    if len(conds) > 0:
      msg = 'Also require the following conditions:\n'
      for cond in conds:
        msg += '  at least one of ('
        for i, y in enumerate(cond):
          msg += 'a_{}'.format(y)
          if i < len(cond) - 1:
            msg += ', '
          else:
            msg += ') vanishes\n'
      print(msg)
    print('')

  # Construct some bounds to start; work hard here
  
  initial_bounds,_ = a_bounds(g,q,N1,num_tries=initial_tries)

  # We know next a_d lies in [lower,upper], but we refine upper bound
  # further using previous a_i's
  def better_upper_bound(lower, upper, g, q, aa, **kwargs):
    if lower == upper:
      return upper
    new_upper = random_ad_bound(g,q,aa,**kwargs)
    return min(upper,new_upper)

  # Determine if progress module is present, for printing
  progress_bar = False
  if verbose:
    try:
      import progress
      progress_bar = True
    except ModuleNotFoundError:
      pass
  
  # Rule out various choices of a-values using derivative information
  F = real_weil_pol(g,q) # Indeterminate coefficients
  real_pols = PolynomialRing(QQ,names=('T',))
  good_as = [()]
  msg = 'Determining information about a_{}:\n'
  msg += '  Using range [{}, {})\n'
  msg += '  At most {} polynomials to check'
  for j in range(g-1,-1,-1):
    Fj = F.derivative(j)
    Fjcoefs = Fj.list()
    next_good_as = []
    index = g - j # Determining a_{g-j}
    # Set upper and lower bounds
    if index in a_vals:
      lower = a_vals[index]
      upper = a_vals[index]
    else:
      lower = 0
      upper = initial_bounds[index]
    work = len(good_as)*(upper-lower+1)
    if verbose: print(msg.format(index,lower,upper+1,work))
    if progress_bar: prog = progress.Progress(len(good_as))
    for aa in good_as:
      # Try to improve the bound for the next a_d
      new_upper = better_upper_bound(lower,upper,g,q,aa,**kwargs)
      next_coef_range = (lower,new_upper+1)
      for b in range(*next_coef_range): 
        subs = aa + (b,) + (0,)*(j) # pad to get length g
        if not seems_excessive(subs,g-j): continue
        # Specialize F_j using these choices of a_d
        Fj_spec_coefs = [evaluate_pol(c,subs) for c in Fjcoefs]
        Fj_spec = real_pols(Fj_spec_coefs)
        if real_roots_check(Fj_spec,q):
          next_good_as.append(aa + (b,))
      if progress_bar: prog()
    good_as = next_good_as    
    if progress_bar: prog.finalize()
    
  Fcoefs = F.list()
  good_pols = []
  for aa in good_as:
    Fspec_coefs = [evaluate_pol(c,aa) for c in Fcoefs] 
    Fspec = real_pols(Fspec_coefs)
    good_pols.append(Fspec)

  if fp is not None:
    for pol in good_pols:
      fp.write('{}\n'.format(pol))
    fp.close()
  return good_pols
    

