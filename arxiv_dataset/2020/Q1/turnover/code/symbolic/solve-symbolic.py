# README
# Goal: Solve for the values of the turnover matrix (z11), z12, z13, ...
# in terms of arbitrary distribution x1, x2, x3, ... and rates of change b1, b2, b3, ...
# Status: should be possible but sympy gives no solution (confirmed bug)
# Author: Jesse Knight, Oct 2018

import sympy as sp

G = 2
x = sp.Matrix([sp.Symbol('x'+str(i+1)) for i in range(G)])
z = [sp.Symbol('z'+str(i+1)+str(j+1)) for i in range(G) for j in range(G) if not i==j]
b = [sp.Symbol('b'+str(i+1)) for i in range(G)]

iz = [(i,j) for i in range(G) for j in range(G) if i is not j]
iv = [i*G+j for i in range(G) for j in range(G) if i is not j]

A = [ [-x[zi[0]] if zi[0] == i else 
        x[zi[0]] if zi[1] == i else 0
        for zi in iz
      ] for i in range(G)
    ]
    
Az  = sp.Matrix([[ai*zi for ai,zi in zip(row,z)] for row in A])
bAz = sp.Matrix([list(Az[ri,:])+[bi] for ri,bi in zip(range(G),b)])
print('SYSTEM:')
print('\n'.join(['0 = '+' + '.join([str(bAz[i,j]) for j in range(G*(G-1)+1)])  for i in range(G)]))
print('RANK:')
print(sp.Matrix(A).rank())
print('SOLUTION: [sympy.solve_linear_system]:')
print(sp.solve_linear_system(bAz,*z))

# Solved G = 2 by hand:
# when isolating for z12 using z21, both cancel out, yielding "b1 = -b2".
# Fine, but since Rank(A) = G-1, we should be able to
# choose z12 or z21 (free parameter) and solve for the other.
# Also tried b = [-0.1,+0.1] and solved, yielding the desired solution...!
# Conclude: bug in sympy when RHS is symbolic - i.e. b
# Reported: https://github.com/sympy/sympy/issues/15463
