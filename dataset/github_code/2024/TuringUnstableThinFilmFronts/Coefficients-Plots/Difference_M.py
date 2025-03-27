################ define function, which returns a python expression evaluating M_m^*-M_o* #######################
from sympy import *
import numpy as np

def Mdiff():
    ######### calculate contour area of monotonic instability as reference ##########
    b = Symbol("beta")
    g = Symbol("g")
    M = Symbol("M")
    k = Symbol("k")
    lam = Symbol("lambda")
    P = Symbol("P")
    a0, a1 = symbols("a_0 a_1")
    L = Symbol("L")
    Mo, Mm  = symbols("M_o M_o")
    kmstar, kostar = symbols("k_m^* k_o^*")
    Mostar, Mmstar = symbols("M_o^* M_m^*")
    s = Symbol("s")

    L = Matrix([[-(k**4)/3-(g/3-M/2)*k**2, -(M/2)*k**2],[-Rational(1,8)*k**4-(g/8-M/6)*k**2+b,-(1+M/6)*k**2-b]])
    P = Poly((lam*eye(2)-L).det(),lam)

    # extract linear and absolute coefficient
    a1 = collect(P.nth(1),k)
    a0 = simplify(collect(P.nth(0),k))

    # define M_m(k) and M_o(k)
    Mm = simplify(solveset(a0,M).args[0])
    Mo = simplify(solveset(a1,M).args[0])

    # find critical k_m^* and k_o^*
    # pprint(simplify(solveset(Mo.diff(k),k,S.Reals)))
    kostar = (3*b)**(Rational(1,4))
    # pprint(simplify(solveset(simplify((Mm.subs(k**2,s)).diff(s)),s,S.Reals)))
    kmstar = simplify(sqrt(b*g/(72 - b) + 6*sqrt(2)*sqrt(-b*g*(b - g - 72))/(72 - b)))

    # evaluate M_m(k) and M_o(k) at critical wave numbers
    Mmstar = simplify(Mm.subs(k,kmstar))
    Mostar = simplify(Mo.subs(k,kostar))

    differenceMmMo = Mostar-Mmstar
    # convert to python function for faster evaluation
    differenceMmMo_fastEval = lambdify([b,g],differenceMmMo)

    return differenceMmMo_fastEval