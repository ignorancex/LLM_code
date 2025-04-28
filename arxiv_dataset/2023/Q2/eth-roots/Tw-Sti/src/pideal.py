# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

from sage.all import *

import fp
from ZR_mat import *        # Row-style lower triangular HNF



# --------------------------------------------------------------------------------------------------
# Prime ideals

# In Sagemaths, several ideal functions rely on their HNF on some basis order (as they should)
# It seems that Sage always try to compute the maximal order (which it shouldn't).
# This is the case for (e.g.):
#         .norm(), .basis(), .gens_two(), ...
# and this is quite annoying.
#
# I didn't find a way to tell Sage what is the maximal order, or to obtain a less poor behaviour
# for number field ideals, so these are fast (but maybe generically incorrect) methods that should
# work *in our setting*.

# Should be correct in most common situations *FOR PRIME IDEALS*
def pid_fast_gens_two(pid):
    (p, g) = pid.gens();# [:2]; # Removing [:2] should guarantee the 2-elt representation is correct  
    p = ZZ(p); g = g.polynomial();
    assert(is_prime(p)); # Keep it for now but unnecessary if the [:2] trick works 
    return (p, g);


def pid_fast_norm(pid):
    (p, g) = pid_fast_gens_two(pid);
    f      = g.degree();
    return ZZ(p**f);


def pid_fast_smallest_integer(pid):
    (p, g) = pid_fast_gens_two(pid);
    return p;


def pid_fast_residue_class_degree(pid):
    (p, g) = pid_fast_gens_two(pid);
    f      = g.degree();
    return ZZ(f);


# //-- END OF FILE


# Wrpt. equation order
# Must be a prime ideal. The second generator must be correct.
# Cyclotomics: Works only for prime conductor ? 
def pid_hnf_basis_eqn_order(pid):
    K = pid.number_field();
    assert(K.defining_polynomial().leading_coefficient() == 1); # Might have creepy things otherwise
    n = K.degree();

    (p, g) = pid_fast_gens_two(pid);
    assert(is_prime(p) and g.leading_coefficient() == 1); # Think both are required for this to work
    # "y = K.gen()" --> using equation order
    Fp     = GF(p);  Fpy = PolynomialRing(GF(p), name='y'); y = Fpy.gen();
    g_modp = Fpy(g); f   = g_modp.degree();
    # A basis is p, p*z, ..., p*z^(f-1), g, g*z, g*z^2, ..., g*z^(n-1-f)
    id_ZB = matrix(ZZ, [[0]*_f + [p] + [0]*(n-1-_f) for _f in range(f)]
                   + [ (g_modp*y**_k).list() + [0]*(n-f-(_k+1)) for _k in range(n-f) ]);
    # HNF for this basis (lower-triangular)
    id_ZB = row_lower_hermite_form(id_ZB, algorithm='pari0', transformation=False);
    # [OLD] + Beware modifications must stay mod p (but id_ZB cannot be on Fp because of p at top)
    # for _i in range(f+1, n):
    #     for _j in range(_i-1, f-1, -1): # _i down to 0
    #         id_ZB.add_multiple_of_row(_i,_j,-id_ZB[_i][_j]);

    return id_ZB;



# -------------------------------------------------------------------------------------------------
# DRAFT HNFs (for products, pow)

# /!\ Input: two prime ideals 
# Todo
# Faut voir ce qui est le plus utile.
# On a des p^k, des p^k * q^j * ...
# De maniere generale (Coh96, §4.7.1, §2.4.3) il faut essayer de se ramener a la situation:
#     1 HNF * 1 <g1, g2>  ==> HNF (n*2n)
# 

# def pid_fast_mul(pid, qid):  
#    return pid;



