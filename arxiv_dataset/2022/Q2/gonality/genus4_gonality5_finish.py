
r""" 

Given the output of the C-program genus4_search, we want to test
all of the resulting curves to see which are smooth and geometrically
irreducible. These will give genus-4 gonality-5 curves.

This is described as Algorithm 3 in the paper 

  Faber, Xander and Grantham, Jon. "Ternary and Quaternary Curves of small
  fixed genus and gonality with many rational points." Exp. Math. (2021).

"""

import itertools
import progress
import search_tools
import sys

from sage.all import *

################################################################################

class SpecialGenusFourCurve(object):
  r"""
  Simple class for manipulating pointless canonical genus-4 curves over GF(q)
  that lie on a non-split quadratic surface in P^3

  INPUT:

  - ``Q`` -- a quadratic form in GF(q)[x,y,z,w] of the shape x*y + z^2 + a*z*w + w^2
    if q is even or x*y + z^2 + b*w^2 if q is odd

  - ``C`` -- a cubic form in GF(q)[x,y,z,w] with x^3-coefficient equal to 1, 
    nonzer y^3-coefficient, and zero x^2*y-, x*y^2-, z^3-, and w^3-coefficient

  - ``autVQ`` -- optional list of matrices giving the automorphism group of the 
    quadric surface `V(Q)`

  """
  def __init__(self, Q, C, autVQ=None):
    assert Q.parent() == C.parent()
    self._polring = Q.parent()
    self._all_cubics = None
    self._best_cubic = None
    
    FF = Q.parent().base_ring()
    q = FF.cardinality()
    x,y,z,w = self._polring.gens()
    
    # Check that Q is normalized correctly
    zero_monomials = [x**2, x*z, x*w, y**2, y*z, y*w]
    if any(Q.monomial_coefficient(u) != 0 for u in zero_monomials):
      raise ValueError('Q = {} is not of the expected shape'.format(Q))
    if any(Q.monomial_coefficient(u) != 1 for u in [x*y,z**2]):
      raise ValueError('Q = {} is not of the expected shape'.format(Q))
    a = Q.monomial_coefficient(z*w)
    b = Q.monomial_coefficient(w**2)
    if q % 2 == 0 and b != 1:
      raise ValueError('Q = {} is not of the expected shape'.format(Q))
    elif q % 2 == 1 and a != 0:
      raise ValueError('Q = {} is not of the expected shape'.format(Q))      
    self._Q = Q

    # Check that C is normalized correctly
    zero_monomials = [x**2*y,x*y**2,z**3,w**3]
    if any(C.monomial_coefficient(u) != 0 for u in zero_monomials):
      raise ValueError('C = {} is not of the expected shape'.format(C))
    if C.monomial_coefficient(x**3) != 1:
      raise ValueError('C = {} is not of the expected shape'.format(C))
    if C.monomial_coefficient(y**3) == 0:
      raise ValueError('C = {} is not of the expected shape'.format(C))
    self._C = C

    # Construct orthogonal group if necessary
    if autVQ is None and q % 2 == 0:
      autVQ = search_tools.char2_dim4_nonsplit_quadric_automorphism_group(FF,a)
    if autVQ is None and q % 2 == 1:
      autVQ = search_tools.odd_char_dim4_nonsplit_quadric_automorphism_group(FF,b)
    self._autVQ = autVQ

  def __repr__(self):
    msg = 'Interesection of Q = {} and C = {} in projective 3-space'
    return msg.format(self._Q,self._C)

  def C(self):
    return self._C

  def is_smooth(self):
    Q = self._Q
    C = self._C
    polring = self._polring
    xx = polring.gens()
    first_row =  [Q.derivative(u) for u in xx]
    second_row = [C.derivative(u) for u in xx]
    J = matrix(polring,2,4,[first_row,second_row])
    sing_ideal = polring.ideal([C,Q] + J.minors(2))
    return search_tools.ideal_dimension(sing_ideal) == 0

  def all_cubics(self):
    r""" 
    Store and return the set of all normalized cubics in the orbit of Aut(V(Q))
    """
    if self._all_cubics is not None:
      return self._all_cubics
    autVQ = self._autVQ
    Q = self._Q
    C = self._C
    x,y,z,w = self._polring.gens()
    all_cubics = set()
    for g in autVQ:
      new_C = search_tools.matrix_act_on_poly(g,C)
      new_C *= (1/new_C.monomial_coefficient(x**3))
      new_C -= new_C.monomial_coefficient(x**2*y)*x*Q
      new_C -= new_C.monomial_coefficient(x*y**2)*y*Q
      new_C -= new_C.monomial_coefficient(z**3)*z*Q
      new_C -= new_C.monomial_coefficient(w**3)*w*Q
      all_cubics.add(new_C)
    self._all_cubics = all_cubics
    return all_cubics

  def best_cubic(self):
    r"""
    Return the lex-minimal cubic form with the fewest monomials in self._all_cubics
    """
    if self._best_cubic is not None:
      return self._best_cubic
    all_cubics = self.all_cubics()
    min_terms = len(self._C.dict())
    min_cubics = []
    for C in all_cubics:
      num_terms = len(C.dict())
      if num_terms > min_terms:
        continue
      elif num_terms == min_terms:
        min_cubics.append(C)
      else:
        min_cubics = [C]
        min_terms = num_terms
    min_cubics.sort()
    self._best_cubic = min_cubics[0]
    return self._best_cubic

  def is_equivalent_to(self, other):
    r""" Return True if other curve is projectively equivalent to self """
    if other._Q != self._Q: return False
    return other._C in self.all_cubics()

################################################################################  

if __name__ == '__main__':
  
  if len(sys.argv) != 3:
    raise IOError('Usage: sage genus4_gonality5_finish.py [q] [filename]')
  q = ZZ(sys.argv[1])
  filename = sys.argv[2]
  fp = open(filename,'r')

  # Set q and coefficient a in xy + z^2 + azw + b*w^2
  if q == 2:
    FF = GF(2) 
    a,b = FF(1), FF(1)
    swap = lambda u : u  # don't change the string reps in the data file
  elif q == 3:
    FF = GF(3)
    a,b = FF(0), FF(1)
    swap = lambda u : u  # don't change the string reps in the data file
  elif q == 4:
    FF = GF(4,'t')
    t = FF.gen()
    a,b = FF(t),FF(1)
    swap = lambda u : u.replace('T^2+T','t') # T^2 + T = t in GF(16), per flint
  else:
    raise NotImplementedError
  
  R = PolynomialRing(FF,names=('x','y','z','w'))
  x,y,z,w = R.gens()
  Q = x*y + z**2 + a*z*w + b*w**2
    
################################################################################

  print('Computing automorphism group of V(Q) ...')
  if q in [2,4]:
    autVQ = search_tools.char2_dim4_nonsplit_quadric_automorphism_group(FF,a)
  else:
    autVQ = search_tools.odd_char_dim4_nonsplit_quadric_automorphism_group(FF,b)
  print('Done!\n')

  print('Loading cubics from file ...')
  cubic_pols = []
  fp = open(filename,'r')
  for line in fp.readlines():
    s = swap(line)
    C = R(s)
    cubic_pols.append(C)
  print('Found {} cubic polynomials\n'.format(len(cubic_pols)))
  fp.close()

  smooth_cubics = []
  nonsmooth_cubics = []
  prog = progress.Progress(len(cubic_pols))
  print('Identifying normalized cubic generators of ideals ...')
  for C in cubic_pols:
    prog()
    X = SpecialGenusFourCurve(Q,C,autVQ)
    if any(Y.is_equivalent_to(X) for Y in nonsmooth_cubics): continue
    if any(Y.is_equivalent_to(X) for Y in smooth_cubics): continue
    if X.is_smooth():
      smooth_cubics.append(X)
    else:
      nonsmooth_cubics.append(X)
  prog.finalize()
  msg = 'Found {} isomorphism classes of smooth genus-4 gonality-5 curves over GF({}).'
  print(msg.format(len(smooth_cubics),q))
  msg = 'Found {} isomorphism classes of non-smooth quadric/cubic intersections over GF({}).'
  print(msg.format(len(nonsmooth_cubics),q))

  fp = open(filename+'.smooth','w')
  for X in smooth_cubics:
    fp.write(str(X.C()) + '\n')
  fp.close()
