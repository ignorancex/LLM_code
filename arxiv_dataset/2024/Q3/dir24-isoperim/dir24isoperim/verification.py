# (c) 2024 Joris Roos <jroos.math@gmail.com>
# pylint: disable=too-many-arguments,invalid-name,no-name-in-module

''' Verification of numerical inequalities '''

from flint import arb

from .general import b0, b1, c0, Jconst, L, Q, DQ, J, DJ, alpha0, alpha1, \
                    part_rect, part_intvl, min_val_intvl, min_val_rect

from .util import Log, log, FMT_FAIL, FMT_PASS, err, warn, Output

from .labels import lbl_dict

#
#  Define functions for verification
#

def Jm(xm: arb, xM: arb) -> arb:
    '''Lower bound for J'''
    return arb.min(J(xm), J(xM))

def JM(xm: arb, xM: arb) -> arb:
    '''Upper bound for J'''
    if xM < Jconst.x0:
        return J(xM)
    if xm > Jconst.x0:
        return J(xm)
    return J(Jconst.x0)

def absDJm(xm: arb, xM: arb) -> arb:
    '''Lower bound for |J'|'''
    if xM < Jconst.x0:
        return DJ(xM)
    if xm > Jconst.x0:
        return -DJ(xm)
    return arb(0)

def absDJM(xm: arb, xM: arb) -> arb:
    '''Upper bound for |J'|'''
    return arb.max(abs(DJ(xm)), abs(DJ(xM)))

def g_J_1(xm: arb, xM: arb, hm: arb, hM: arb, b: arb, c: arb) -> arb:
    '''Case J'''
    rv = b*c**(1-1/b)*JM(xm+hm, xM+hM)**(1-1/b)
    rv -= .5*b*(1-b)*c**(1-2/b)*Jm(xm+hm, xM+hM)**(1-2/b)*hM**(1/b)
    rv -= c/2*Jm(xm, xM)**(-1)*hM**(2-1/b)
    if xM < Jconst.x0:
        rv += c/4*DJ(xM)*J(xM)**(-2)*hm**(3-1/b)
        rv += c/32*DJ(xM)*(7+3*DJ(xM)**2)*J(xM)**(-4)*hm**(5-1/b)
    else: # *not* equivalent to xM >= x0 since x0 is not exact
        rv -= c/4*absDJM(xm, xM)*Jm(xm, xM)**(-2)*hM**(3-1/b)
        rv -= c/32*absDJM(xm, xM)*(7+3*absDJM(xm, xM)**2)*Jm(xm, xM)**(-4)*hM**(5-1/b)
    rv -= 7*c/48*(1+absDJM(xm, xM)**2)*Jm(xm, xM)**(-3)*hM**(4-1/b)
    rv -= c/90*(7+23*absDJM(xm, xM+hM)**2+6*absDJM(xm, xM+hM)**4)*Jm(xm, xM+hM)**(-5)*hM**(6-1/b)
    rv += c/2880*(7+23*absDJm(xm, xM+hM/2)**2
            +6*absDJm(xm, xM+hM/2)**4)*JM(xm, xM+hM/2)**(-5)*hm**(6-1/b)
    return rv

def g_J_2(xm: arb, xM: arb, ym: arb, yM: arb):
    '''Case J'''
    return (ym-xM)**2 + J(yM)**2 - (2*J((xm+ym)/2)-J(xm))**2

def g_Q_1(hm: arb, hM: arb, ym: arb, yM: arb, b: arb) -> arb:
    '''Case Q'''
    rv = b*Q(yM,b)**(1-1/b)
    rv -= (alpha0(b)- 2*alpha1(b)*hm+4*alpha1(b)*yM)*hM**(2-1/b)
    rv -= b/2*(1-b)*Q(ym,b)**(1-2/b)*hM**(1/b)
    return rv

def g_Q_1_y1_4(*p, b):
    '''Case Q'''
    return g_Q_1(arb(.25), arb(.25), *p, b)

def g_Q_2(hm: arb, hM: arb, ym: arb, yM: arb, b: arb) -> arb:
    '''Case Q'''
    rv = -12*alpha1(b)*hM**2
    rv += (2*alpha0(b)+8*alpha1(b)*ym)*hm
    rv -= 6*(3-1/b)*alpha1(b)*Q(yM, b)**(1/b)*hM**(2-1/b)
    rv += (2-1/b)*(2*alpha0(b)+8*alpha1(b)*ym)*Q(ym, b)**(1/b)*hM**(1-1/b)
    return rv

def g_LJQ_1(xm: arb, xM: arb, ym: arb, yM: arb, b: arb) -> arb:
    '''Case LJQ'''
    l = ((ym-xM)**(1/b) + Jm(ym, yM)**(1/b))**b
    r = ym - xM + (2**b-1)*Jm(ym, yM)
    rv = arb.max(l, r)
    rv += L(xm, b)
    rv -= 2*Q((xM+yM)/2, b)
    return rv

def g_LJQ_2(ym: arb, yM: arb, bm: arb, bM: arb) -> arb:
    '''Case LJQ'''
    return ym - arb(1/16) + (2**bm-1)*Jm(ym, yM)+L(arb(1/16), bm) \
            - 2*Q(yM/2+arb(1/32), bM)

def g_QJQ(xm: arb, xM: arb, ym: arb, yM: arb, b: arb) -> arb:
    '''Case QJQ'''
    return ym - xM + J(ym)*DJ(ym) - (2*Q((xm+yM)/2, b) - Q(xm, b))*DQ((xm+ym)/2, b)

def g_QJ_1(xm: arb, xM: arb, ym: arb, yM: arb, b: arb, c: arb) -> arb:
    '''Case QJ'''
    rv = (ym-xM)**(1/b-1)
    if xM < Jconst.x0:
        rv += c**(1/b)*(2*Jm((xm+ym)/2, (xM+yM)/2)-Q(xM,b))**(1/b-1)*DJ((xM+yM)/2)
    else:
        rv -= c**(1/b)*(2*JM((xm+ym)/2, (xM+yM)/2)-Q(xm,b))**(1/b-1)*absDJM((xm+ym)/2, (xM+yM)/2)
    rv -= c**(1/b)*(2*JM((xm+ym)/2, (xM+yM)/2)-Q(xm,b))**(1/b-1)*DQ(xm, b)
    return rv

def g_QJ_2(xm: arb, xM: arb, ym: arb, yM: arb) -> arb:
    '''Case QJ'''
    return ((ym - xM)**2 + J(yM)**2)**.5 + Q(xm,arb(.5)) - 2*JM((xm+ym)/2, (xM+yM)/2)

def g_P_1(x: arb, b: arb=b1) -> arb:
    '''Poincare'''
    return 2**(-2*b)*(arb.log(1/x)/arb.log(arb(2)))**.5 \
        +2**(-2*b)*2**arb(.5)*arb(.77)*arb.log(Jconst.w0/x)**.5-2

def g_P_1_at_val() -> arb:
    '''Poincare'''
    return g_P_1(arb(1/64)) > 0

b0p = .5+3*2**(-12)

def g_P_2(xm: arb, xM: arb, b: arb=b0p) -> arb:
    '''Poincare'''
    return 2**(-2*b)*(L(xm, arb(.5))+J(1-xm))-2*xM*(1-xm)

def g_P_3(xm: arb, xM: arb, b: arb=b0p) -> arb:
    '''Poincare'''
    rv = -.5*DQ(xm, arb(.5))
    if 1-xm < Jconst.x0:
        rv += 2**(-2*b)*DJ(1-xm)
    else:
        rv -= .5*absDJM(1-xM, 1-xm)
    rv += xm**(2*b-1)*(1-xM)
    rv -= xM
    rv += (1-xM)**(2*b)
    rv -= 2*b*xM
    return rv

def g_JL(xm: arb, xM: arb, b: arb) -> arb:
    '''Auxiliary'''
    return 2/JM(xm, xM)-b*arb.log(arb(2))**(-b)*(1-xM)**(-1)*(1-b+arb.log(1/(1-xM))) \
            *arb.log(1/(1-xm))**(-2+b)

# Verification

def verify(v):
    '''
    Call a generic verification routine (expected to return a boolean) and log result.
    Return result of verification.
    '''
    log(v.__name__ + ": ", end="")
    rv = v()
    log((FMT_PASS%"ok" if rv else FMT_FAIL%"fail"), indent=0)
    return rv

def verify_positive(g, x, y=None, maxDepth=12, verbose=1, tag="", **args):
    '''
    Verify that given lower bound function is positive using partitioning 
    on a given rectangle or interval and output result.
    '''
    msg = g.__name__
    if g.__name__ in lbl_dict:
        msg += f" [display ({lbl_dict[g.__name__]})] "
    if len(args) > 0:
        msg += " with "+", ".join([f"{k}={float(v)}" for k,v in args.items()])
    if verbose:
        log(msg + ": ", end="")

    def G(*p):
        return g(*p, **args)

    if y is None:
        success, part = part_intvl(G, x, maxDepth=maxDepth)
        if success:
            Output.get_instance().write_comment(msg)
            cmt = f"{len(part)-1:d} intervals, min. val = {min_val_intvl(G, part)}"
            Output.get_instance().write_part(g.__name__+tag, part, cmt)
            if verbose:
                log(FMT_PASS%"ok", indent=0)
                log(f"   {cmt}")
    else:
        success, part = part_rect(G, x, y, maxDepth=maxDepth)
        if success:
            Output.get_instance().write_comment(msg)
            cmt = f"{len(part):d} rectangles, min. val = {min_val_rect(G, part)}"
            Output.get_instance().write_part(g.__name__+tag, part, cmt)
            if verbose:
                log(FMT_PASS%"ok", indent=0)
                log(f"   {cmt}")
    if not success and verbose:
        log(FMT_FAIL%"fail", indent=0)
        log(f"   at {part}")
    return success, part

def batch_verify(label, methods, verbose=1):
    '''Run a list of verification methods.'''
    if verbose:
        log(label)
    Log.lvl = 1
    for m in methods:
        m()
    Log.lvl = 0

def verify_all(b=b0, c=c0):
    '''Verify all claims in the paper.'''
    Output.get_instance().write(f"# Partition data for beta0={repr(b)}, c0={repr(c)}\n\n")
    if not arb(.5) <= b <= 1:
        err("beta0 must lie in [0.5, 1]")
        return
    if not c <= 1 and c > 0:
        err("c0 must lie in (0,1]")
        return
    if b > b1:
        warn(f"b0>{float(b1):f}: running only case J")
    batch_verify("case J", [
        lambda: verify_positive(g_J_1, (arb(1/2), arb(5/8)), (arb(0), arb(3/16)), b=b, c=arb(1)),
        lambda: verify_positive(g_J_1, (arb(1/2), arb(5/8)), (arb(0), arb(3/16)),
                                tag="h", b=arb(.5), c=c),
        lambda: verify_positive(g_J_2, (arb(1/2), arb(9/16)), (arb(11/16), arb(1)))
    ])
    if b > b1:
        return
    batch_verify("case Q", [
        lambda: verify_positive(g_Q_1, (arb(0), arb(1/4)), (arb(1/4), arb(1/2)), b=b),
        lambda: verify_positive(g_Q_1_y1_4, (arb(1/4), arb(1/2)), b=arb(.5)),
        lambda: verify_positive(g_Q_2, (arb(1/4), arb(1/2)), (arb(1/4), arb(1/2)), b=b),
        lambda: verify_positive(g_Q_2, (arb(1/4), arb(1/2)), (arb(1/4), arb(1/2)),
                                tag="h", b=arb(.5))
    ])
    batch_verify("case LJQ", [
        lambda: verify_positive(g_LJQ_1, (arb(1/16), arb(1/4)), (arb(1/2), arb(3/4)), b=b),
        lambda: verify_positive(g_LJQ_1, (arb(1/16), arb(1/4)), (arb(1/2), arb(3/4)),
                                tag="h", b=arb(.5)),
        lambda: verify_positive(g_LJQ_2, (arb(1/2), arb(3/4)), (arb(1/2), arb(1)))
    ])
    batch_verify("case QJQ", [
        lambda: verify_positive(g_QJQ, (arb(1/4), arb(1/2)), (arb(1/2), arb(3/4)), b=b1),
        lambda: verify_positive(g_QJQ, (arb(1/4), arb(1/2)), (arb(1/2), arb(3/4)),
                                tag="h", b=arb(.5))
    ])
    batch_verify("case QJ", [
        lambda: verify_positive(g_QJ_1, (arb(1/4), arb(1/2)), (arb(1/2), arb(5/8)), b=b, c=arb(1)),
        lambda: verify_positive(g_QJ_1, (arb(1/4), arb(1/2)), (arb(1/2), arb(5/8)),
                                tag="h", b=arb(.5), c=c),
        lambda: verify_positive(g_QJ_2, (arb(1/4), arb(1/2)), (arb(5/8), arb(1)))
    ])
    if b > b0p:
        warn(f"beta0>{float(b0p):f}: skipping Poincare")
        return
    batch_verify("Poincare", [
        lambda: verify(g_P_1_at_val),
        lambda: verify_positive(g_P_2, (arb(1/64), arb(1/4))),
        lambda: verify_positive(g_P_3, (arb(1/4), arb(1/2)))
    ])
    batch_verify("Auxiliary", [
        lambda: verify_positive(g_JL, (arb(1/2), arb(2047/2048)), b=b)
    ])


if __name__ == '__main__':
    verify_all()
