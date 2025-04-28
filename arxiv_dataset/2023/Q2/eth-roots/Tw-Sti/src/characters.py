# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

import itertools;
from sage.all import *
from pideal   import *
from collections import namedtuple # Defining Dirichlet_log type


# --------------------------------------------------------------------------------------------------
# Dirichlet characters

# [Some of Sagemaths functions for Dirichlet characters are needlessly and utterly long.]
# [Ste07] William Stein, "Modular forms: a computational approach", 2007, Ch.4: Dirichlet characters.

# We choose to represent a Dirichlet character modulo m as:
#         m_qi = [ (p_i, e_i) ] where m = prod p_i^e_i
#         m_si = [ si ]         where (Z/mZ)* = prod Z/siZ.
#                                     [NB]  If 2^k || m, k > 2, this starts by [2,2^{k-2},...]
#         e    = lcm (si),      ie. e is the group exponent of (Z/mZ)*
#         gi   = generators for (Z/p_i^e_iZ)*
#         ln_vi= are such that chi(g) = z^ln_v, z primitive e-th root of unity, g in gi, ln_v in ln_vi.

# To enumerate all characters mod m:
#         Consider all possible logs ki \in prod (Z/siZ)
#         The image is then (for g^k, g of order s): k*e/s, k=0..s-1
# To enumerate all values:
#         Consider all possible ki \in prod (Z/siZ)
#         chi(g^k) = z^{k.ln_v mod e}
# Evaluation at -1:
#         The log of -1 is generically (si/2), and (-1,0) for q = 2^k, k>2
#         Hence, ln chi(-1) is something in {0, e/2}.
# Conductor:
#         Follow [Ste07, Alg.4.16].
#         Determining order for ln_v is easy: e/gcd(e,ln_v)
# Primitive character:
#         This turns out to be easy as well once the conductor is known:
#         - keep ln values that are not trivial
#         - keep generators mod f such that corresponding ln_values are not trivial.
#         - if 8 (or more) | m, call back g[0] mod f if ln_v[0] == 0 that was badly removed.

Dirichlet_log = namedtuple('Dirichlet_log',
                           ['m_qi',       # Modulus m = prod p_i^e_i, m_qi = [ (p_i, e_i), p_i|m]
                            'm_si',       # m_si = [ euler_phi(q_i) for q_i=p_i^e_i in m_qi,
                                          # Case p=2: [2] if 4 || m, [2,2^(k-2)] if 2^k || m, k>=3 
                            'e',          # lcm( _si for _si in m_si )
                            'gi',         # Each g_i generates (Z/q_iZ)* += 1 mod [q_j]
                            'ln_vi']);    # k_i = ln_vi[i] st. chi(g_i) = \zeta_e^{k_i} (= v_i)
    

# Construction / Group
# ---------------------------------------------------------------
# Return the trivial character, with base zeta_e
def set_dirichlet_log_one(e=1):
    one = Dirichlet_log(m_qi=[], m_si=[], e=e, gi=[], ln_vi=[]);
    return one;


# Construct a "log" Dirichlet character from one of the Sage homomorphism 
def set_dirichlet_log(chi):
    m     = chi.modulus(); assert(m > 1 and 2 != mod(m,4)); # Lazy

    # Mod-m group structure
    Zm    = Integers(m);
    m_qi  = list(factor(m));
    # Deal with the case p=2, that must be cut in two cyclic parts if 2^k | m, k >= 3.
    m_s0  = [] if m_qi[0][0] != 2 else [2] if m_qi[0][1] < 3 else [2, 2**(m_qi[0][1]-2)];
    m_si  = m_s0 + [ euler_phi(_f[0]**_f[1]) for _f in m_qi if _f[0] != 2 ];
    e     = Zm.unit_group_exponent();
    gi    = list(Zm.unit_gens());

    # Logs of images of chi mod z_e
    z     = UniversalCyclotomicField().zeta(e);
    chi_v = chi.values_on_gens();
    log_v = list( next(_k for _k in range(0,e,e/_si) if z**_k == _v)
                  for _si, _v in zip(m_si, chi_v) );
    # Construct log chi "object"
    chi_log = Dirichlet_log(m_qi=m_qi, m_si=m_si, e=e, gi=gi, ln_vi=log_v);

    return chi_log;
    

# Returns an iterator on the DirichletGroup of characters modulo m
# This is usable only once.
# Usage: D = dirichlet_log_group(m);
#        for _d in D : # or whatever iterates once on D
def dirichlet_log_group(m):
    # Mod-m group structure
    Zm    = Integers(m);
    m_qi  = list(factor(m));
    # Deal with the case p=2, that must be cut in two cyclic parts if 2^k | m, k >= 3.
    m_s0  = [] if m_qi[0][0] != 2 else [2] if m_qi[0][1] < 3 else [2, 2**(m_qi[0][1]-2)];
    m_si  = m_s0 + [ euler_phi(_f[0]**_f[1]) for _f in m_qi if _f[0] != 2 ];
    e     = Zm.unit_group_exponent();
    gi    = list(Zm.unit_gens());
    
    for ki in itertools.product(*[range(0,e,e/_si) for _si in m_si]):
        chi = Dirichlet_log(m_qi=m_qi, m_si=m_si, e=e, gi=gi, ln_vi=list(ki));
        yield chi;



# Parity
# ---------------------------------------------------------------
# This is one of the main bottleneck of Sagemaths implementation.
# Using the known decomposition of (-1) on the set of generators gives a very fast answer.
def dirichlet_log_eval_m1(chi):
    (m_qi, m_si, e, gi, ln_vi) = chi;

    # Construct ln(-1) over generators of (Z/mZ)*: generically, this is { phi(p_i^e_i)/2 }
    ln_m1_mod_m  = [ ZZ(_si/2) for _si in m_si];
    # Correct the non-cyclic (Z/2^kZ)*: in this case, ln(-1) = [1,0], not [1,2^(k-3)]
    if (m_qi[0][0] == 2 and m_qi[0][1] > 2):
        ln_m1_mod_m[1] = 0;

    # Compute ln(chi(-1)) in Z/eZ
    ln_chim1 = sum(ZZ(_a)*_log for _a, _log in zip(ln_m1_mod_m, ln_vi)).mod(e);
    assert(ln_chim1 in [0, e/2]);
    chi_m1   = 1 if ln_chim1 == 0 else -1;

    return chi_m1;


def dirichlet_log_is_odd(chi):
    return (True if dirichlet_log_eval_m1(chi) == -1 else False);

def dirichlet_log_is_even(chi):
    return (True if dirichlet_log_eval_m1(chi) ==  1 else False);


# Conductor / Primitive character
# ---------------------------------------------------------------
# Inspired from [Ste07, Alg.4.16]
# The treatment at p=2 is performed a bit differently but the result is totally equivalent.
def dirichlet_log_conductor(chi):
    (m_qi, m_si, e, gi, ln_vi) = chi;

    f_chi = 1;
    if (len(m_qi) == 0):
        return f_chi;
    
    # Going through all factors and determining their contributions to conductor
    m_pi  = ([2] if m_qi[0][0] == 2 and m_qi[0][1] > 2 else []) + [ _f[0] for _f in m_qi ];
    for _pi, _ln_gi in zip(m_pi, ln_vi):
        ord_i = e / gcd(e,_ln_gi); # Using gcd(x,0) == x
        fi    = 1 if ord_i == 1 else _pi**(1+ord_i.valuation(_pi));
        f_chi *= fi;    
        
    # Correction if 2^k || m, k > 2: orders (1,2^v) should give 2^{1+v} * 2
    #                                       (2,2^v) should give 2^{3+v} / 2
    #                                       (1,1) gives 1, (2,1) gives 4.
    if (m_qi[0][0] == 2 and m_qi[0][1] > 2 and ln_vi[1] != 0):
        f_chi = 2*f_chi if ln_vi[0] == 0 else ZZ(f_chi/2);
        
    return f_chi;


# Primitive character
# This one is also very slow using Sage.
# Essentially, once the conductor is known, it is just a matter of:
# - updating the modulus and unit group structure
# - keep generators *modulo f* that are still involved (on which chi is not trivial)
# - keep values where we kept generators.
# [NB] We don't reduce the "e" in order to use the same cyclotomic unit basis in sums over chis mod m
def dirichlet_log_primitive(chi):
    (m_qi, m_si, e, gi, ln_vi) = chi;

    # Find conductor
    f_chi = dirichlet_log_conductor(chi);
    f_qi  = list(factor(f_chi));
    if (f_chi == 1):
        one   = set_dirichlet_log_one();
        one   = one._replace(e=e);
        return one;
    
    # Construct phi(f) as above
    f_s0  = [] if f_qi[0][0] != 2 else [2] if f_qi[0][1] < 3 else [2, 2**(f_qi[0][1]-2)];
    f_si  = f_s0 + [ euler_phi(_f[0]**_f[1]) for _f in f_qi if _f[0] != 2 ];

    # Keep generators mod f depending on chi(g_i).
    # --> For p=2, keep both whenever g(5 mod 8) != 1, otherwise keep (-1) or no one.
    # --> This comes down to get back (-1) iff 2^k||m, k>2 and ln(-1)==0
    # Same mechanism for logs
    f_gi   = [ mod(gi[_i], f_chi) for _i in range(len(gi)) if ln_vi[_i] != 0];
    f_logs = [ ln_vi[_i]          for _i in range(len(gi)) if ln_vi[_i] != 0];
    # Correction for p=2 as described above
    if (m_qi[0][0] == 2 and m_qi[0][1] > 2 and ln_vi[0] == 0 and ln_vi[1] != 0):
        f_gi   = gi[:1] + f_gi;
        f_logs = [0]    + f_logs;

    # Construct the "primitive" (with previous e) character
    prim_chi = Dirichlet_log(m_qi=f_qi, m_si=f_si, e=e, gi=f_gi, ln_vi=f_logs);
       
    return prim_chi;



# Evaluation
# ---------------------------------------------------------------
# Enumerate all possible values (no specific order)
# Each iterator call outputs a new (a,chi(a)) as a polynomial in x mod x^e-1
# Note that not reducing mod phi_e(x) is helping for computing sums on these values
def dirichlet_log_enum_values(chi):
    (m_qi, m_si, e, gi, ln_vi) = chi;
    
    Zx = PolynomialRing(Integers(), name='x', sparse=False);  x = Zx.gen();
    m  = prod( _f[0]**_f[1] for _f in m_qi );
    
    for ki in itertools.product(*[range(0,_si) for _si in m_si]):
        a     = ZZ(mod( prod(_g**_k for _g, _k in zip(gi,ki)), m));
        chi_a = x**(ZZ(mod( sum(_ln_v*_k for _ln_v, _k in zip(ln_vi,ki)), e)));
        yield (a, chi_a);



# ----------------------------------------------------------------
# Bernouilli sums

# [Was97,p.32,before Th.4.2]:
#         B_{1,chi} = 1/f. \sum_{a=1}^{f} chi(a).a    for chi != 1 of conductor f,
#         B_{1,1}   = 1/2.
def B1_chi(chi_p):
    m_qi  = chi_p.m_qi;
    f     = prod(_f[0]**_f[1] for _f in chi_p.m_qi);
    fB1   = sum( _a*_chi_a for _a, _chi_a in dirichlet_log_enum_values(chi_p) ) if f != 1 else 1/2;
    return fB1 / f;




# --------------------------------------------------------------------------------------------------
# Generalized Dirichlet Characters

# Determine (generalized Dirichlet) character chi associated to a prime ideal pid
# /!\ We suppose K = CyclotomicField(m)
#            and pid is prime ideal (not necessarily of prime norm)
# Returns (Fq_a, m, ln_chi) st. Fq* = <Fq_a>, Fq=OK/pid
#                           and chi(Fq_a) = zeta_{m}^{ln_chi} [= Fq_a mod pid]
# This constructs the residue character (conjugate of the \omega^-d in [Was97])
def get_chi(K, m, pid):
    # Retrieve pid characteristics in a 'safe' way
    # [Sage is not handling wisely enough big degree fields]
    (p, g) = pid_fast_gens_two(pid); assert(is_prime(p));
    #p     = Integers()(pid.gens_two()[0]); assert(is_prime(p));
    Fp    = GF(p);
    Fpx   = PolynomialRing(Fp, name='xp'); xp = Fpx.gen();
    # /!\ gen[1]: Could be not "reduced" (ie., not a factor of K.def() mod p)
    # pid_x = Fpx(pid.gens_two()[1].polynomial()).gcd(Fpx(K.defining_polynomial()));
    pid_x = Fpx(g).gcd(Fpx(K.defining_polynomial()));
    if (pid_x == 0): # Deal with inert primes for consistency
        pid_x = Fpx(K.defining_polynomial());
    pid_f = pid_x.degree();
    
    # Construct Finite Field Fq, q = p^f with appropriate modulus (f = inertia degree)
    q     = p**pid_f;
    Fq    = GF(q, name='u', modulus=pid_x); u = Fq.variable_name();
    Fq_a  = Fq.multiplicative_generator();
    
    # Characters: [This is the conjugate of [Was97]]
    # - \omega_P : modulus p^f, lands in mu(p^f-1=dm) (p^f-1)/d (d = (p^f-1)/m in [Was97])
    # - \chi_pp is some \omega_P ^ (d) (=> lands in mu(m))
    # - \omega_P is useful for the proof, but we do not need it explicitly for computations 
    # Even if f=1, d is not necessarily 1 
    # [p=139, m=23 ==> d=6 ]
    d = Integer((q-1)/m); # Don't let d be in QQ for the sequel
    # Which character is corresponding to our choice pp=Op[0] ?
    # - We need omega^{d}(a) mod P = a^{d} mod P
    # - So, find which power of zeta_m is = Fq.multiplicative_generator()^{d} mod pp
    # --------------------------------------------
    # Version that only deals with polynomials mod p to obtain \chi_pp as zeta_m^{ln_chi_pp}
    res_px   = Fpx(Fq_a**d); # Want this mod pp_mod = pp.gens_two()[1]
    # Good enough to obtain all ^i mod pid_x
    ln_chi = [_i for _i in range(1,m) if power_mod(Fpx(K.gen().polynomial()),_i,pid_x) == res_px ];
    assert(len(ln_chi) == 1);
    ln_chi = ln_chi[0];

    # # [OLD VERSION]
    # # Describe orbits of (\omega_P)^d and find the correct one
    # # /!\ Expensive, does not work for f > 1
    # t = cputime();
    # Chip   = DirichletGroup(q); # NOT FOR q = p^f, only for q = p^1
    # # Without [:m], this is d orbits (repetitions) of characters from FF_p* to m-th roots of unity
    # Chip_d = [ _chi^d for _chi in Chip][:m]; # ^(-d) in [Was97]: this is not important here because we get all ^d values and then search for the right one.
    # t = cputime(t); print("Get Character group:\t{}s".format(t));
    # # Loop on all Chip_d to find the one st Chip_d(Fp.multiplicative_generator()) mod pp = Fp.mult...()^d mod pp
    # # /!\ This line is a bit too violent (>20s, on small examples)
    # # [ [ oo.small_residue(K(_chi(2))) for oo in Op ] for _chi in Chip_d]; # Look for the one that sends to 2^6 (mod Op[0]
    # t = cputime();
    # idx_chip = [ _i for _i in range(len(Chip_d)) if pid.small_residue(K(Chip_d[_i](Fq_a))) == (Integer(Fq_a^(-d))-p if Integer(Fq_a^d) > p/2 else Integer(Fq_a^d) )]; # q = p
    # assert(len(idx_chip) == 1);
    # chip     = Chip_d[idx_chip[0]];
    # t = cputime(t);  print("Get chi_p:\t{}s".format(t));
    
    return (Fq_a, m, ln_chi); # All that is needed to define chi: Fq_a -> zeta_m^{ln_chipp}


# Power of character
def chi_pow(chi, a):
    (Fq_a, m, ln_chi) = chi;
    ln_pow  = Integers()(mod(a*ln_chi, m));
    chi_pow = (Fq_a, m, ln_pow);

    return chi_pow;


# Conjugate
def chi_conj(chi):
    (Fq_a, m, ln_chi) = chi;
    ln_conj  = Integers()(mod(-ln_chi, m));
    chi_conj = (Fq_a, m, ln_conj);

    return chi_conj;


# Compute chi(-1) = chi(Fq_a)^((q-1)/2)
def chi_eval_m1(chi):
    (Fq_a, m, k) = chi; # chi(Fq_a) = \zeta_{m}^{k}
    ZZ = Integers(); Fq = Fq_a.parent(); q = Fq.cardinality();
    
    ln_chim1  = ZZ((q-1)/2*k).mod(m); assert(ln_chim1 in [0, m/2]);
    chi_m1    = 1 if ln_chim1 == 0 else -1;

    return chi_m1;



# Gauss sums
# --------------------------------------------------
# We have to compute g(chi) = - \sum_{a \in \FF_q} chi(a) \zeta_p^{Tr_{\FF_q/\FF_p} a}.
# Two things are fundamental for this to behave relatively smoothly:
#     1. Use the Universal Cyclotomic Field, which is using a sloppy reduction strategy.
#        [Don't attempt to build CyclotomicField(m*p), use UCF=CyclotomicField(); zmp=UCF.zeta(m*p).]
#     2. Never inverse elements that presumably lie in Q(\zeta_{mp}).

# It is possible to use [Note the *minus* sign for g_chi]:
#         from sage.arith.misc import gauss_sum
#         zm    = CyclotomicField().zeta(m); # Important NOT to use CyclotomicField(m).
#         g_chi = - gauss_sum(zm^_k, Fq);    # zm^_k = chi(Fq_a) where Fq* = <Fq_a>
# [See] https://doc.sagemath.org/html/en/reference/rings_standard/sage/arith/misc.html#sage.arith.misc.gauss_sum
#       This version is deals not only with *Dirichlet* characters (mod p), but also on Finite Fields.
# [Pb]  The issue here is that computations with Sage (GAP) are somehow limited to cyclotomic fields
#       of degree < 10^6, which is rapidly overflown by euler_phi(m*p).
# from sage.arith.misc import gauss_sum

# To overcome this, we use polynomials directly mod x^{mp}-1, hence (with gcd(m,p)=1):
#         x = \zeta_{mp}, \zeta_m=x^p, \zeta_p=x^m.
# A few implementation notes:
#     1. We use sparse polynomial rings that behave well wrpt to sums of monomials x^_
#     2. Monomials are directly (in their exponents) reduced mod x^{mp}-1.
#     3. The result is converted to a dense polynomial ring, well suited for mod computations.
#     4. Stickelberger's generators computations are done (densely) mod x^{mp}-1 and mapped back to
#        Q(\zeta_m) by reducing mod phi_p(x^m), then mod phi_m(x^p) in that *exact* order:
#             mod x^{mp}-1    ----> mod phi_p(x^m)    ----> mod phi_m(x^p)

# TODO: Remaining problem for q = p^f, f > 1:
#       We should try to simplify the "for _l in range(1,q)" 
def gauss_sum_x(chi):
    ZZ = Integers();
    Zy = PolynomialRing(Integers(), name='y', sparse=True);  y = Zy.gen();
    Zx = PolynomialRing(Integers(), name='x', sparse=False); x = Zx.gen();

    # Environment infos on chi
    (Fq_a, m, k) = chi;
    Fq = Fq_a.parent(); q = Fq.cardinality(); p = Fq.characteristic();
    # Gauss sum associated to chi
    g_chi = - sum( y**(ZZ(mod( p*ZZ(mod(k*_l,m)) + m*ZZ((Fq_a**_l).trace()), p*m)))
                   for _l in range(1,q)); # Trace is already mod p
    g_chi = Zx(g_chi);

    return g_chi;


def gauss_sum_chipow_x(chi, a):
    (Fq_a, m, ln_chi) = chi;
    chi_a   = chi_pow(chi,a);
    g_chi_a = gauss_sum_x(chi_a);

    return g_chi_a;


# Computes g(chibar) from g(chi) (if provided)
# [Was97, Lem.6.1] g(chibar) = chi(-1) bar(g(chi))
# g(chi) is a polynomial in \zeta_{mp}, so bar(g(chi)) is g(chi)(1/x) mod x^{mp}-1.
def gauss_sum_chiconj_x(chi, g_chi=[]):
    Zx = PolynomialRing(Integers(), name='x', sparse=False);
    p  = chi[0].parent().characteristic();
    m  = chi[1];
    
    g_chi_x   = Zx(g_chi);
    if (g_chi_x.degree() == -1):
        chi_bar  = chi_conj(chi);
        g_chibar = gauss_sum_x(chi_bar);
    else:
        bar_g_chi = g_chi.reverse(degree=m*p);
        chi_m1    = chi_eval_m1(chi);
        g_chibar  = chi_m1 * bar_g_chi;
        
    return g_chibar;



# Jacobi sums
# --------------------------------------------------
# It appears that the (short) Stickelberger relation \theta(a)+\theta(b)-\theta(-c), m|a+b+c,
# is generated by :
#         1/p . g(chi^a) . g(chi^b) . g(chi^c)
# Considering that chi^m = 1, this is 1/p.J(chi^a, chi^b) = 1/p g(chi^a) g(chi^b) / g(chi^{a+b})
# See [Was97, Lem.6.2(d)]

# It turns out that this Jacobi sum is also (by definition in [Was97, p.88]):
#         J(chi1, chi2) = - sum_{a \in FF_p} chi1(a) chi2(1-a)
# This is also much easier to compute when FF_p is not too large.

def jacobi_sum_dlog_table(Fq_a):
    Fq   = Fq_a.parent(); q = Fq.cardinality();
    assert(Fq_a.multiplicative_order() == q-1);

    # dlog[k] = (ln_u u^k, ln_u (1-u^k)) = (k, ln_u(1-u^k))
    # (k,s) tq u^s = 1 - u^k
    dlog = [0] + [ Fq_a**_k for _k in range(1,q-1) ]; # NB: summand for u^{q-1} is 0.
    dlog = [ (_k, dlog.index(1-dlog[_k])) for _k in range(1,q-1)];
    
    return dlog;


def jacobi_sum_x(chi1, chi2, dlog=[]):
    # Both must be compatible characters (Fq_a, m, ln)
    assert (chi1[:2] == chi2[:2]);

    ZZ = Integers();
    Zx = PolynomialRing(Integers(), name='x', sparse=False); x = Zx.gen();

    # Environment infos on chi
    (Fq_a, m, k1) = chi1;
    (_,    _, k2) = chi2; 
    Fq = Fq_a.parent(); q = Fq.cardinality(); p = Fq.characteristic();

    # Jacobi sum as polynomial mod (x^m - 1)
    dlog  = jacobi_sum_dlog_table(Fq_a) if len(dlog) == 0 else dlog;
    assert(Fq_a**dlog[-1][1] + Fq_a**dlog[-1][0] == 1);
    j_chi = - sum( x**(ZZ(mod( k1*_l + k2*_il, m))) for _l, _il in dlog );
    
    return j_chi;

