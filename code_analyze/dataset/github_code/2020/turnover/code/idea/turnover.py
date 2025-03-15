import numpy as np
import json
from scipy.optimize import minimize

def zetafun(ri,ro,xi,xo,bounds=None,bprime=None,Aprime=None):
  r"""Compute the turnover matrix to yield a steady-state population distribution

  Args:
    ri: entry rate
    ro: exit rate
    xi: entry population distrubution
    xo: target population distribution
    bprime: additional constraints LHS
    Aprime: additional rows of A (RHS)

  Returns:
    a matrix $\\zeta$
  """
  N = len(xo)
  # define the LHS
  b = ri*(xo-xi).T
  # vectors of indices for convenience
  iz = [(i,j) for i in range(N) for j in range(N) if i is not j]
  iv = [i*N+j for i in range(N) for j in range(N) if i is not j]
  # define the RHS system matrix
  A = np.array(\
    [ [ xo[zi[1]] if zi[0] == i else
       -xo[zi[1]] if zi[1] == i else 0
        for zi in iz
      ] for i in range(N)
    ])
  # append any additional constraints
  if (bprime is not None) and (Aprime is not None):
    b = np.concatenate(( b, np.atleast_1d(bprime) ), axis=0)
    A = np.concatenate(( A, np.atleast_2d(Aprime) ), axis=0)
  # solve the system
  jfun = lambda z: np.linalg.norm((np.dot(A,z)-b),2)
  z0   = np.dot(np.linalg.pinv(A),b)
  out  = minimize(jfun, z0, bounds = [bounds for zi in z0], method = 'L-BFGS-B')
  z    = out['x']
  if not out['success']:
    print('Warning: turnoverfun did not converge.')
  # return zeta as a matrix
  zeta = np.array(\
    [ [ z[iv.index(i*N+j)] if i*N+j in iv else 0
        for i in range(N)
      ] for j in range(N)
    ])
  return zeta

if __name__ == '__main__':
  nu =  0.05;  # births
  mu =  0.04;  # deaths
  xi = np.array([0.15, 0.20, 0.65]); # entry distribution
  xo = np.array([0.05, 0.10, 0.85]); # target distribution
  N  = len(xi)
  Z12 = 1.0
  A1  = [[1 for i in range(N**2-N)]]
  zeta = zetafun(nu,mu,xi,xo,(0.01,0.5),Z12,A1)
  print zeta
