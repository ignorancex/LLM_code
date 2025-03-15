import os,sys; sys.path.append(os.path.join((lambda r,f:f[0:f.index(r)+len(r)])('code',os.path.abspath(__file__)),'config')); import config
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import gif

def solve(z0,xi,xo,ri,ro,dur,Ap,bp):
  
  G = len(xo)
  # define the LHS
  b = ri*(xo - xi).T
  # vectors of indices for convenience
  iz = [(i,j) for i in range(G) for j in range(G) if i is not j]
  iv = [i*G+j for i in range(G) for j in range(G) if i is not j]
  # define the RHS system matrix
  A = np.array(\
    [ [-xo[zi[0]] if zi[0] == i else 
        xo[zi[0]] if zi[1] == i else 0
        for zi in iz
      ] for i in range(G)
    ])
  # append duration-based constraints
  if dur is not None:
    for id,di in enumerate(dur):
      if np.isfinite(di):
        b = np.concatenate((b, [(di**(-1)) - ro]))
        A = np.concatenate((A, [[1 if (i == id) else 0
                                for i in range(G)
                                for j in range(G) if i is not j]]),axis=0)
  # append any additional constraints
  if (bp is not None) and (Ap is not None):
    b = np.concatenate(( b, np.atleast_1d(bp) ), axis=0)
    A = np.concatenate(( A, np.atleast_2d(Ap) ), axis=0)
  # solve the system
  jfun = lambda z: np.linalg.norm((np.dot(A,z)-b),2)
  zo = z0*np.ones((G*G-G,1))
  out  = minimize(jfun, zo, bounds = [(0.00,0.50) for i in zo],
                            method = 'L-BFGS-B',
                            options = {'ftol': 1e-18,
                                       'gtol': 1e-18,
                                       'eps':  1e-8} )
  z = out['x']
  # return zeta as a matrix
  zeta = np.array(\
    [ [ z[iv.index(i*G+j)] if i*G+j in iv else 0
        for j in range(G)
      ] for i in range(G)
    ])
  # print(np.linalg.matrix_rank(A))

  fig = plt.figure(figsize=(6,6))
  N = len(z)
  for ij in combinations(range(N),2):
    zmax = 0.1 # max(max(z[ij[0]],z[ij[1]]),1e-6)
    zi,dzi = np.linspace(0,2*zmax,50,retstep=True)
    zj,dzj = np.linspace(0,2*zmax,50,retstep=True)
    J = np.array([[ jfun([ i if k == ij[1] else  j if k == ij[0] else zk \
      for k,zk in enumerate(z)]) \
        for i in zi]
          for j in zj])

    ax = plt.subplot(N-1,N-1,ij[0]*(N-1)+ij[1])
    ax.imshow((J+1e-9)**0.5,cmap='inferno',interpolation='bilinear')
    zix,zjx = z[ij[1]], z[ij[0]]
    plt.plot(zix/dzi, zjx/dzj, 'x', markersize=10, color='#00ff00')
    plt.ylabel('z{}{}'.format(*[e+1 for e in iz[ij[0]]]),fontsize=8)
    plt.xlabel('z{}{}'.format(*[e+1 for e in iz[ij[1]]]),fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
  efmt = lambda xi: '{:.03f}'.format(xi).ljust(6)
  vfmt = lambda x: '[{}]'.format(' '.join([efmt(xi) for xi in x]))
  fig.text(0.1,0.1, '\n'.join([
    'z0  = {}'.format(efmt(z0)),
    'xi  = {}'.format(vfmt(xi)),
    'xo  = {}'.format(vfmt(xo)),
    'ri  = {}'.format(efmt(ri)),
    'ro  = {}'.format(efmt(ro)),
    'dur = {}'.format(vfmt(dur)),
  ]),fontfamily='monospace')
  plt.tight_layout()

save = os.path.join(config.path['root'],'outputs','figs','surface-gifs',
                    'surface-05-na-na-vary-z0.gif')
@gif.gif(save=save,time=1,
         z0 = np.linspace(0,0.05,8),
         d1=5.0,
         # d2=10.0,
         # d3=30.0,
         # d1=np.linspace(4,10,8),
         # d2=np.linspace(5,20,8),
         # d3=np.linspace(15,30,8),
         # ri=0.05
         # ro=np.linspace(0.01,0.08,8)
         )
def main(z0=0.0,
         e1=0.10,e2=0.20,e3=0.70,
         x1=0.05,x2=0.20,x3=0.75,
         ri=0.05,
         ro=0.03,
         d1=np.nan,d2=np.nan,d3=np.nan):
  solve(\
    z0,
    np.array([e1,e2,e3]),
    np.array([x1,x2,x3]),
    ri,
    ro,
    np.array([d1,d2,d3]),
    None,
    None)

if __name__ == '__main__':
  main
