import sys
sys.path.append('../lib/')
from sympy import *
from sympy.plotting import plot
"""
This script computes the error bound for the interpolation error of the shifted POD operators.
Please compare the results to the computation of the numerical errors in the scrit sPOD_interp_err.py
"""
# %%
x = symbols("x")
a = symbols("sigma")

w = (x+2)*(x+1)*(x)*(x-1)*(x-2)*(x-3) # h^6 order
#w = (x+1)*(x)*(x-1)*(x-2) # h^4 order
#w = (x)*(x-1)   # h^2 order
dw = diff(w,x)
ddw = diff(dw,x)

dRoots = solveset(dw,x)

for xstar in dRoots:

    print("xstar:", xstar, "w(xstar):", simplify(w.subs(x,xstar)), "dw(xstar):", simplify(dw.subs(x, xstar)), "ddw(xstar):",simplify(ddw.subs(x,xstar)))

p1=plot(w,show=False,xlim=[-1,2],ylim=[-1,1])
p1.show()


#%%
inicond = exp(-x**2/a**2)
d2inicond =diff(diff(inicond,x),x)
d4inicond = diff(diff(diff(diff(inicond,x),x),x),x)
d6inicond = diff(diff(d4inicond,x),x)
print("inicond: ", simplify(inicond))
print("d/dx^6 inicond", d6inicond)
print("d/dx^4 inicond", d4inicond)
print("d/dx^2 inicond", d2inicond)
