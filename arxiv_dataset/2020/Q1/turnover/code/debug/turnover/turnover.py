import numpy as np
from scipy.optimize import nnls

def turnover(nu,mu,px,pe=None,zeta=None,dur=None):
  def ziter(Z):
    for i,zi in enumerate(Z):
      for j,zij in enumerate(zi):
        if (i!=j):
          yield zij
  def append(b,A,_b,_A):
    return ( \
      np.concatenate((b,_b),axis=0),
      np.concatenate((A,_A),axis=0)
    )
  def hot(i,j):
    return 1 if i == j else 0
  
  G = len(px)
  R = G*(G-1)
  # define the LHS
  b = nu*px.T
  # vectors of indices for convenience
  zidx  = [(i,j) for i in range(G) for j in range(G) if i is not j]
  zivec = [i*G+j for i in range(G) for j in range(G) if i is not j]
  # define the RHS system matrix (A)
  Ae = np.float(nu) * np.eye(G)
  Az = np.array(\
    [
      [
        -px[zi[0]] if zi[0] == i else
        +px[zi[0]] if zi[1] == i else 0
      for zi in zidx ]
    for i in range(G) ]
  )
  A = np.concatenate((Ae,Az),axis=1)
  # append known pe values
  if pe is not None:
    for i,pei in enumerate(pe):
      if np.isfinite(pei):
        b,A = append(b,A,
          [pei],
          [[hot(gi,i) for gi in range(G)] + [0]*R]
        )
  # append known zeta values
  if zeta is not None:
    for i,zi in enumerate(ziter(zeta)):
      if np.isfinite(zi):
        b,A = append(b,A,
          [zi],
          [[0]*G + [hot(zi,i) for zi in range(R)]]
        )
  # append duration-based constraints
  if dur is not None:
    for i,di in enumerate(dur):
      if np.isfinite(di):
        b,A = append(b,A,
          [(di**-1) - mu],
          [[0]*G + [hot(zi[0],i) for zi in zidx]]
        )
  # solve the system
  theta,err = nnls(A,b)
  # parse the solution vector
  pe,z = theta[:G],theta[G:]
  zeta = np.array(\
    [
      [
        z[zivec.index(i*G+j)] if i*G+j in zivec else 0
      for j in range(G) ]
    for i in range(G) ]
  )
  dur = np.power(mu+np.sum(zeta,axis=1),-1)
  # debug printouts
  # print('b    = {}'.format(np.around(b,3)))
  # print('A    = \n{}'.format(np.around(A,3)))
  # print('J    = {}'.format(np.around(err,6)))
  print('pe   = {}'.format(np.around(pe,3)))
  print('dur  = {}'.format(np.around(dur,3)))
  print('zeta = \n{}'.format(np.around(zeta,3)))
  print('len(t): {}, rank(A): {}'.format(len(theta),np.linalg.matrix_rank(A)))
  print('err  = {}'.format(np.around(err,9)))
  # return np.linalg.matrix_rank(A)


G = 3
if __name__ == '__main__':
  # # check Rank(A) for combinations of specified pe & zeta
  # for e1 in [np.nan,0.2]:
  #   for e2 in [np.nan,0.2]:
  #     for e3 in [np.nan,0.2]:
  #       print('-'*30)
  #       for i1 in range(3):
  #         for j1 in range(3):
  #           if i1 != j1:
  #             zeta = np.nan*np.ones((3,3))
  #             zeta[i1,j1] = 0.001
  #             R = turnover(
  #               nu = 0.05,
  #               mu = 0.03,
  #               px = np.array([0.05,0.15,0.40]),
  #               pe = np.array([e1,e2,e3]),
  #               zeta = zeta,
  #               dur = np.array([5.,15.,25.]),
  #             )
  #             print('({},{},{}) + ({},{}) -> {}'.format(e1,e2,e3,i1,j1,R))

  # # check Rank(A) for combinations of two specified zeta elements
  # for i1 in range(G):
  #   for j1 in range(G):
  #     if i1 != j1:
  #       print('-'*30)
  #       for i2 in range(G):
  #         for j2 in range(G):
  #           if i2 != j2:
  #             zeta = np.nan*np.ones((G,G))
  #             zeta[i1,j1] = 0.001
  #             zeta[i2,j2] = 0.002
  #             R = turnover(
  #               nu = 0.05,
  #               mu = 0.03,
  #               px = np.array([0.02,0.10,0.38]),
  #               # pe = np.array([0.02,0.10,0.38]),
  #               zeta = zeta,
  #               dur = np.array([5.,15.,25.]),
  #             )
  #             print('({},{}) + ({},{}) -> {}'.format(i1,j1,i2,j2,R))

  from numpy import nan
  # # one-off with G = 3
  x = nan
  turnover(
    nu   = 0.05,
    mu   = 0.03,
    px   = np.array([0.02,0.10,0.38]),
    pe   = np.array([0.02,0.10,0.38]),
    dur  = np.array([  5, 15, 25.]),
    zeta = np.array([[ x ,nan,0.1],[nan, x ,nan],[nan,nan, x ]]),
  )

# Different set of variants (using pe, not splitting evenly)

# (6)
# pe   = np.array([0.02,0.10,nan]),
# dur  = np.array([  5, 15,25.]),
# zeta = np.array([[ x ,nan,0.1],[nan, x ,nan],[nan,nan, x ]]),
# (5)
# pe   = np.array([nan,0.10,nan]),
# dur  = np.array([  5, 15,25.]),
# zeta = np.array([[ x ,nan,0.1],[  0, x ,nan],[nan,nan, x ]]),
# (4)
# pe   = np.array([nan,nan,nan]),
# dur  = np.array([  5, 15,25.]),
# zeta = np.array([[ x ,nan,0.1],[  0, x ,nan],[nan,  0, x ]]),
# (3)
# pe   = np.array([nan,nan,nan]),
# dur  = np.array([  5, 15,nan]),
# zeta = np.array([[ x ,nan,0.1],[  0, x ,nan],[  0,  0, x ]]),
# (2)
# pe   = np.array([nan,nan,nan]),
# dur  = np.array([  5,nan,nan]),
# zeta = np.array([[ x ,nan,0.1],[  0, x ,  0],[  0,  0, x ]]),
# (1)
# pe   = np.array([nan,nan,nan]),
# dur  = np.array([nan,nan,nan]),
# zeta = np.array([[ x ,  0,0.1],[  0, x ,  0],[  0,  0, x ]]),
# (0)
# pe   = np.array([nan,nan,nan]),
# dur  = np.array([nan,nan,nan]),
# zeta = np.array([[ x ,  0,  0],[  0, x ,  0],[  0,  0, x ]]),

  # # G = 4
  # turnover(
  #   nu = 0.05,
  #   mu = 0.03,
  #   px = np.array([0.01,0.05,0.10,0.34]),
  #   pe = np.array([0.01,0.05,0.10,0.34]),
  #   zeta = np.array([[nan,0.01,0.01,nan],[0.01,nan,0.01,0.01],[0.01,nan,nan,nan],[nan,nan,nan,nan]]),
  #   dur = np.array([5.,10.,15.,25.]),
  # )