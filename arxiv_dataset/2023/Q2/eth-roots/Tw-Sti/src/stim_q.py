# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

from sage.all    import *
from characters  import *
from cyclotomics import * # cf_get_conductor()


# -------------------------------------------------------------------------------------------------
# KUCERA
# Adapting [Kuc92] "On bases of the Stickelberger ideal and of the group of circular units of a cyclotomic field"
# The "alpha" function corresponds to a new paper:
#          [BK21] "A short basis of the Stickelberger ideal of a cyclotomic field"


# Special sets of indices mod m
# ------------------------------------------------------------
# M- from [Kuc92, p.293]. We decompose it as:
# X (no 0) \ res_q + N- + { B_k for k in 1..g, g=#factor(m) }
def kucera_get_Mminus(m):
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q));
    # Start from X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    Mminus = [ ZZ(_a) for _a in range(1,m) if
               all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    # Remove N+
    Nminus = { _a for _a in Mminus
               if _a.divides(m) and is_even(len([_q for _q in m_q if not _q.divides(_a)])) };
    Mminus = [ _a for _a in Mminus if _a not in Nminus ];
    # Remove all a/(m,a)=-1 [q_i] for some q_i
    res_q = { _a for _a in Mminus
            if any([ _q.divides(_a+gcd(m,_a)) for _q in m_q if not _q.divides(_a)]) };
    Mminus = [ _a for _a in Mminus if _a not in res_q ];
    # Remove B_k = { q_k \not| a and {a/q_k/(a,m)} > 1/2 and for i > k, a = (m,a) mod [q_i] }
    for i in range(g):
        q_i = m_q[i];
        B_i = { _a for _a in Mminus if
                not q_i.divides(_a)
                and 2 * Integers()(mod(_a/gcd(_a,m), q_i)) > q_i
                and all( _q.divides(_a-gcd(m,_a)) for _q in m_q[i+1:] ) };
        Mminus = [ _a for _a in Mminus if _a not in B_i ];

    assert (len(Mminus) == euler_phi(m)/2);
    return Mminus;


# Implementing the 2021 version:
# M'_m = M_m \ {a st. m/q_i | a} u [[1,\phi(qi)/2]]
def kucera_get_Mp(m):
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q));

    Mm  = kucera_get_Mminus(m);
    Mp  = [ a for a in Mm if all( not ZZ(m/qi).divides(a) for qi in m_q ) ];
    ext = sum( ([ZZ(m*b/qi) for b in range(1, euler_phi(qi)/2+1)] for qi in m_q), []);
    Mp  = sorted(list(set( Mp + ext )));
    
    assert (len(Mp) == euler_phi(m)/2);
    return Mp;


# The Alpha function
# ----------------------------------------------------------
# Computes \alpha_m(b)
# Returns the ZZ triple [a,b,c] in [0,m[ st. m | a+b+c and :
#         \alpha_m(b) = om(a)+om(b)+om(c)+1/2.Nm = th(a)+th(b)-th(-c)  is *short* (all coeffs in {0,1})
# The relation is then (s_a + s_b + s_c) th(1) / p
def kucera_alpha(m, b, verbose=False):
    m_p  = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q  = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g    = len(m_q);
    Jb_c = [ i for i in range(g) if not m_q[i].divides(b) ]; assert(len(Jb_c) > 0);
    
    if len(Jb_c) == 1:
        qj = m_q[Jb_c[0]]; pj = m_p[Jb_c[0]];
        c  = ZZ(b*qj/m); assert( 0 < c and c <= euler_phi(qj)/2 );
        if c > 1:
            a_idx = [ ZZ(_i) for _i in [-b, b-m/qj, m/qj]]; # - {b} + {b-m/qj} + {m/qj}
        else: # c==1
            a_idx = [ ZZ(_i) for _i in [ m*euler_phi(qj)/2/qj ]*2 + [m/pj]]; # 2*{ m.\phi(qj)/(2.qj) } + {m/pj}
    else:
        # Solving u_b x + v_b y = -b/r_b
        ub      = m_q[min(Jb_c)];
        rb      = prod(m_q[i] for i in range(g) if i not in Jb_c);
        vb      = ZZ(m/rb/ub);
        (d,s,t) = ub.xgcd(vb); assert(d==1); assert(d==s*ub+t*vb);
        # s = s*(-b/rb); t = t*(-b/rb); 
        A     = ZZ(mod(-vb*t*b,m)); B = ZZ(mod(-ub*s*b,m));
        a_idx = [b, A, B];

    # Mapping everything in [0,m[
    a_idx = [ ZZ(mod(_i,m)) for _i in a_idx ];
    assert(0 == mod(sum(a_idx),m));
    if verbose: print("{} +{} +{}".format(*a_idx));

    return a_idx;


# All \alpha_m(b) for b in S.
# Computing \alpha is cheap so there is no trick here
def kucera_alpha_idx(m, S):
    al = [ kucera_alpha(m, _b) for _b in S ];
    return al;

# All \alpha_m(b) for b in M'm
def kucera_alpha_all(m):
    S  = kucera_get_Mp(m);
    al = kucera_alpha_idx(m, S);
    return al;


# ------------------------------------------------
# [ HELP FUNCTIONS FOR SUBSETS ]
# /!\ No particular care here except for correctness.

# By XP, not removing the Bk's is generally good to obtain a "short" generating basis om(a)-om(a+1)
def kucera_get_Mminus_keep_all_Bk(m):
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q));
    # Start from X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    Mminus = [ ZZ(_a) for _a in range(1,m) if
               all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    # Remove N+
    Nminus = { _a for _a in Mminus
               if _a.divides(m) and is_even(len([_q for _q in m_q if not _q.divides(_a)])) };
    Mminus = [ _a for _a in Mminus if _a not in Nminus ];
    # Remove all a/(m,a)=-1 [q_i] for some q_i
    res_q = { _a for _a in Mminus
            if any([ _q.divides(_a+gcd(m,_a)) for _q in m_q if not _q.divides(_a)]) };
    Mminus = [ _a for _a in Mminus if _a not in res_q ];
    return Mminus;


def kucera_get_X(m):
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q));
    # X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    X   = [ ZZ(_a) for _a in range(1,m) if
            all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    return X;


def kucera_get_Nminus(m):
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q));
    # Start from X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    X   = [ ZZ(_a) for _a in range(1,m) if
            all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    # Remove N+
    Nminus = { _a for _a in X
               if _a.divides(m) and is_even(len([_q for _q in m_q if not _q.divides(_a)])) };
    return list(Nminus);


def kucera_get_res_q(m):
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q));
    # Start from X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    X = [ ZZ(_a) for _a in range(1,m) if
          all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    # Remove N+
    Nminus = { _a for _a in X
               if _a.divides(m) and is_even(len([_q for _q in m_q if not _q.divides(_a)])) };
    Mminus = [ _a for _a in X if _a not in Nminus ];
    # Remove all a/(m,a)=-1 [q_i] for some q_i
    res_q = { _a for _a in Mminus
            if any([ _q.divides(_a+gcd(m,_a)) for _q in m_q if not _q.divides(_a)]) };
    return list(res_q);


def kucera_get_Bk(m,k): # k : factor index from 0 to g-1
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q)); assert(k<g);
    # Start from X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    Mminus = [ ZZ(_a) for _a in range(1,m) if
               all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    # Remove N+
    Nminus = { _a for _a in Mminus
               if _a.divides(m) and is_even(len([_q for _q in m_q if not _q.divides(_a)])) };
    Mminus = [ _a for _a in Mminus if _a not in Nminus ];
    # Remove all a/(m,a)=-1 [q_i] for some q_i
    res_q = { _a for _a in Mminus
            if any([ _q.divides(_a+gcd(m,_a)) for _q in m_q if not _q.divides(_a)]) };
    Mminus = [ _a for _a in Mminus if _a not in res_q ];
    # Remove B_k = { q_k \not| a and {a/q_k/(a,m)} > 1/2 and for i > k, a = (m,a) mod [q_i] }
    for i in range(k+1):
        q_i = m_q[i];
        B_i = { _a for _a in Mminus if
                not q_i.divides(_a)
                and 2 * Integers()(mod(_a/gcd(_a,m), q_i)) > q_i
                and all( _q.divides(_a-gcd(m,_a)) for _q in m_q[i+1:] ) };
        Mminus = [ _a for _a in Mminus if _a not in B_i ];

    return list(B_i);


# Mminus removing Bi up to i=k
def kucera_get_Mminus_k(m,k): # k : factor index
    # m = prod q_i, where q_i = p_i^e_i
    m_p = [ _f[0]        for _f in factor(m) ]; # Cheap anyway.
    m_q = [ _f[0]**_f[1] for _f in factor(m) ]; # Cheap anyway.
    g   = ZZ(len(m_q)); assert(k<g);
    # Start from X = all [1, m-1] prime to m, plus multiples of p_i^e_i (not of p_i, ... only)
    Mminus = [ ZZ(_a) for _a in range(1,m) if
               all( not m_p[_i].divides(_a) or m_q[_i].divides(_a) for _i in range(g)) ];
    # Remove N+
    Nminus = { _a for _a in Mminus
               if _a.divides(m) and is_even(len([_q for _q in m_q if not _q.divides(_a)])) };
    Mminus = [ _a for _a in Mminus if _a not in Nminus ];
    # Remove all a/(m,a)=-1 [q_i] for some q_i
    res_q = { _a for _a in Mminus
            if any([ _q.divides(_a+gcd(m,_a)) for _q in m_q if not _q.divides(_a)]) };
    Mminus = [ _a for _a in Mminus if _a not in res_q ];
    # Remove B_k = { q_k \not| a and {a/q_k/(a,m)} > 1/2 and for i > k, a = (m,a) mod [q_i] }
    for i in range(k+1):
        q_i = m_q[i];
        B_i = { _a for _a in Mminus if
                not q_i.divides(_a)
                and 2 * Integers()(mod(_a/gcd(_a,m), q_i)) < q_i
                and all( _q.divides(_a-gcd(m,_a)) for _q in m_q[i+1:] ) };
        Mminus = [ _a for _a in Mminus if _a not in B_i ];

    return Mminus;
# [ END HELP FUNCTIONS ] -----------------------------------------



# --------------------------------------------------------------------------------------------------
# Special Stickelberger elements in QQ[G]

# Theta computations
# ------------------------------------------
# \theta(a) = \sum_{(r,m)=1, 0<r<m} {- ar/m} \s_r^{-1}
def theta_kucera_a(m,a):
    n =  euler_phi(m);
    G         = [ ZZ(i) for i in range(m) if gcd(i,m)==1 ]; # s_a : a st (a,m)=1
    Ginv      = [ b.inverse_mod(m) for b in G ]; # Ginv[i] : s_{1/G[i] mod m}
    Gminv_idx = [ G.index(b) for b in Ginv];

    v = [0]*n;
    for i in range(len(G)):
        r      = G[i];
        idx    = Gminv_idx[i];
        v[idx] = ZZ(mod(-a*r,m))/ m;

    return vector(v);


# Computation of all thetas is quite costly (because of G/Ginv and QQ computations)
# So it is interesting to compute all of them in batch.
# - The output is st. ta[a] = \theta(a), and notably, ta[0] = 0.
# - The last element is \theta(m//2).
def theta_kucera_all(m):
    n         =  euler_phi(m);
    G         = [ ZZ(i) for i in range(m) if gcd(i,m)==1 ]; # s_a : a st (a,m)=1
    Ginv      = [ b.inverse_mod(m) for b in G ]; # Ginv[i] : s_{1/G[i] mod m}
    Gminv_idx = [ G.index(b) for b in Ginv];

    ta = [zero_vector(n)];
    for a in range(1, m//2+1):
        v = [0]*n;
        for i in range(len(G)):
            r      = G[i];
            idx    = Gminv_idx[i];
            v[idx] = ZZ(mod(-a*r,m))/ m;

        ta.append(vector(v));
        
    # [OLD] (Very) Slow code
    # ta = [zero_vector(euler_phi(m))] + [ theta_kucera_a(m,a) for a in range(1,m//2+1) ];
    return ta;


# Same for a subset S, computes \theta_m(S)
def theta_kucera_idx(m, S):
    n         =  euler_phi(m);
    G         = [ ZZ(i) for i in range(m) if gcd(i,m)==1 ]; # s_a : a st (a,m)=1
    Ginv      = [ b.inverse_mod(m) for b in G ]; # Ginv[i] : s_{1/G[i] mod m}
    Gminv_idx = [ G.index(b) for b in Ginv];

    ta = [];
    for a in S:
        v = [0]*n;
        for i in range(len(G)):
            r      = G[i];
            idx    = Gminv_idx[i];
            v[idx] = ZZ(mod(-a*r,m))/ m;

        ta.append(vector(v));

    return ta;


def theta_washington(m): # This is \theta(1) with Kucera notation
    v = theta_kucera_a(m,-1);
    return v;


# Omega computations
# ------------------------------------------
# omega* = w/2m (\theta(-1) + (m+1)\theta(1)), w=2m if m is odd, m if m is even
#        = w/2m (s(G) + m\theta(1))
def omega_kucera_star(m):
    w = 2*m if is_odd(m) else m;
    o_s = w/2/m * (theta_kucera_a(m,-1) + (m+1)*theta_kucera_a(m,1));
    return vector(ZZ,o_s);


# omega(a) = \theta(a) - a*\theta(1) + omega* - s(G)
# The plain version applies the formula above regardless of a
def omega_kucera_a_plain(m,a):
    os = omega_kucera_star(m);
    sG = ones_matrix(1, len(os))[0];
    ta = theta_kucera_a(m,a);
    t1 = theta_kucera_a(m,1);
    return vector(ZZ, ( ta -a*t1 +os -sG));


# The "normal" version implements [Kuc92]:
# - omega(0) = 0
# - omega(a) = -omega(m-a) if m/2<a<m
# - omega(a) = omega(b) whenever a=b [m]
def omega_kucera_a(m,a):
    _b = ZZ(mod(a,m));
    if (_b == 0):
        return zero_vector(ZZ, euler_phi(m));
    elif (_b <= m/2):
        return omega_kucera_a_plain(m,_b);
    else:
        return -omega_kucera_a_plain(m,m-_b);



# --------------------------------------------------------------------------------------------------
# Stickelberger valuations

# [CDW17] Computations of the v_i and w_i
# ------------------------------------------------------------
# Funny thing, there is not much to see on the vi's,
# but the wi's seem to have nice properties (symmetry wrpt conjugation, and wn is the trivial relation)

# Based on [DPW19], with correction -b --> 1/b for result idx
# (The i-th relation is for s_{1/a}, not s_{-a})
# Output: array V, st.:
#     V[0]  is the first relation (2-s_2),
#     V[i]  is relation a - s_a, a=G[i+1]  (convention G[0]=1)
#     V[-1] is relation "1 - s_1", ie. (m+1)-s_{m+1}
# Return a matrix ?

# Example (m=23):
#   K  = QQ(z23)                       -- Cyclotomic fields, conductor m=23
#   p  = 139 (=1 mod 23)               -- Split prime (smallest is 47)
#   pp = <23, z23+8>                   -- Arbitrary orbit starting point (this is get_orbit_p(K,139)[0])
#   Relations v_i: pp^{(a-\sigma_a)\theta} for a in 2,...,m+1 (a,m)==1
#      --> vs[0] is pp^{(2-\sigma_2)\theta}, etc.
#   sigma_i = z23 --> z23^i            -- Galois group elements.
#   v_i : correspond to pp^{(a-\sigma_a)\theta} 

## cdw : v_2, ... v_{m/2-1} + v_{m+1}
## omega_star - s(G) == reverse( v_{m+1} )
## ---> trouver une relation propre en termes de theta etc
def cdw17_va(m,a):
    n         = euler_phi(m);
    G         = [ ZZ(i) for i in range(m) if gcd(i,m)==1 ]; # s_a : a st (a,m)=1
    Ginv      = [ b.inverse_mod(m) for b in G ]; # Ginv[i] : s_{1/G[i] mod m}
    Gminv_idx = [ G.index(b) for b in Ginv];

    v = [0]*n;
    for i in range(len(G)):
        r      = G[i];
        idx    = Gminv_idx[i];
        v[idx] = floor(a * (r / m));

    return vector(v);


def sti_vals_cdw17_v(m):
    n = euler_phi(m);#n = K.degree(); assert(n == euler_phi(m));
    G         = [ ZZ(i) for i in range(m) if gcd(i,m)==1 ]; # s_a : a st (a,m)=1
    Ginv      = [ b.inverse_mod(m) for b in G ]; # Ginv[i] : s_{1/G[i] mod m}
    Gminv_idx = [ G.index(b) for b in Ginv];

    vs = [];
    for a in [_a for _a in range(2,m+2) if gcd(_a,m) == 1]:
        v = [0]*n;
        for i in range(len(G)):
            b      = G[i];
            idx    = Gminv_idx[i];
            v[idx] = floor(a * (b / m));
            
        assert(len(v) == n);
        vs.append(vector(v));

    assert(len(vs) == n); # Rank is n/2 + 1, but we have phi(m) relations
    vs = Matrix(ZZ, vs);
    return vs;


# For now, compute the vs and substract v_i+1 - v_i
def cdw17_wa(m,a):
    return cdw17_va(m,a) - cdw17_va(m,a-1);

def sti_vals_cdw17_w(m):
    vs = sti_vals_cdw17_v(m);
    n  = euler_phi(m);
    ws = ([ vs[-1]-vs[-2] ] # sG
          + [ vs[0] ] + [ vs[_i+1] - vs[_i] for _i in range(n//2-1) ]);
    # ws = [vs[0]] + [vs[i+1] - vs[i] for i in range(n-1)];

    assert(len(ws) == n//2+1);
    ws = Matrix(ZZ, ws);
    return ws;



# Vals of [CDW21] "Mildly short vectors etc"
# ----------------------------------------------------------
def cdw21_va(m,a):
    return a*theta_kucera_a(m,1)-theta_kucera_a(m,a);

# v_a = a*theta(1) - theta(a)
# NB: v_1 = 0, v_m = m*theta(1)
def sti_vals_cdw21_v(m):
    ta = theta_kucera_all(m); assert (len(ta) == m//2+1);
    sG = ones_matrix(1,euler_phi(m))[0];
    vs = [ _a*ta[1] - (ta[_a] if _a <= m//2 else (sG-ta[m-_a]) if _a != m else 0)
           for _a in range(2,m+1) ];
    vs = Matrix(ZZ, vs);
    return vs;

# w_a = v_a - v_{a-1}, 2 <= a <= m
# w_m = v_m - v_{m-1} = s(G)
# w_a = w_{m-a+1} so we just keep s(G) + w_a for 2 <= a <= (m-1)//2 + 1 = ceil(m/2)
#     --> v[-1] - v[-2] = sG
#     --> w_2 = v_2 - v_1 = v[0]
#     --> w_{(m-1)//2 + 1} = v_{(m-1)//2+1}-v_{(m-1)//2} = v[(m-1)//2-1] - v[(m-1)//2-2]
def sti_vals_cdw21_w(m):
    vs = sti_vals_cdw21_v(m); assert (vs.nrows() == m-1);
    ws = ([ vs[-1]-vs[-2] ] # sG
          + [ vs[0] ] + [vs[_i+1] - vs[_i] for _i in range((m-1)//2-1)]);
    # ws = [ vs[0] ] + [vs[i+1] - vs[i] for i in range(m-2)];
    ws = Matrix(ZZ, ws);
    assert(ws.nrows() == ceil(m/2));
    return ws;



# Kucera's (large) Stickelberger basis
# -------------------------------------------------------
# Following Th.4.2: {omega*} + {omega(a) for a in M-}
# We don't use the "omega_a" function to save significant computational effort
def sti_vals_kucera(m, thetas=[]):
    Mminus = kucera_get_Mminus(m);

    # Special vectors (computed once)
    os = omega_kucera_star(m);
    sG = ones_matrix(1, len(os))[0];
    C  = os - sG;
    # Theta(a) is ta[a]
    ta = theta_kucera_all(m) if len(thetas) == 0 else thetas; assert(len(ta) == m//2+1);
    MK = matrix(ZZ, [os] + [ ta[_a] - _a*ta[1] + C if _a <= m/2 else - ta[m-_a] + (m-_a)*ta[1] - C
                             for _a in Mminus ]);
    return MK;



# Short Stickelberger basis (brute-force) 
# --------------------------------------------------------
# Compute the basis {s(G)} u { omega(a)-omega(a+1) for a in indices }
# The resulting matrix has coefficients in {0,1}
# Note that "omega(m//2) - omega(m//2+1)" (quotes because using Kucera's definition this gives 0)
#      is computed as: \theta(1) + \theta(m//2) - \theta(m//2+1) = \theta(1) + 2*\theta(m//2) - s(G)
#                                                                = 2*\omega(m//2) - \omega*
def sti_vals_kucera_short_idx(m, idx_set, thetas=[]):
    # /!\ The "m//2" trick is valid only if m is even
    assert (is_odd(m) or ZZ(m//2) not in idx_set);
    # It will be way more efficient to compute directly "ta-a*t1 - tap1 + ap1*t1 = t1 + ta - tap1"
    # from all thetas(a)
    ta = theta_kucera_all(m) if len(thetas) == 0 else thetas; assert (len(ta) == m//2+1);

    # s(G): will be split as "real gens"
    sG  = ones_matrix(1,euler_phi(m))[0];
    # om(a)-om(a+1) for a in idx_set; careful for a=m//2, m odd
    Mm  = matrix(ZZ, [sG] + [ ta[1] + ta[_a] - ta[_a+1] for _a in idx_set if _a < m//2 ]
                 + ([ ta[1] + 2*ta[m//2] - sG ] if m//2 in idx_set else [])
                 + [ ta[1] - ta[m-_a] + ta[m-_a-1] for _a in idx_set if _a > m//2] );
    
    assert (Mm.height() == 1), "||M||>1";
    return Mm;


def kucera_short_Mminus_bruteforce(m, verbose=True, thetas=[]):
    hnf_met = 'default'; # No noticeable improvement between algorithms in this case.
    
    # Compute all thetas. This is quite costly but can be computed just once.
    if len(thetas) == 0:
        t = cputime(); ta = theta_kucera_all(m); t = cputime(t);
        if verbose: print ("\tAll theta(a):\tt={:.2f}".format(t));
    else:
        ta = thetas;    
    
    # Get the correct (mathematically proven) matrix from [Kuc92], and use its HNF for following tests.
    t = cputime(); H = sti_vals_kucera(m, thetas=ta).hermite_form(); t = cputime(t);
    if verbose: print ("\tKucera basis HNF:\tt={:.2f}".format(t));
    
    # Find a minimal set S st. {s(G)} {om(a)-om(a+1), a in S} (careful for a=m//2) gives the same HNF
    # NB: Mminus is (provably) correct for prime m, never the case for prime powers or composite m.
    #     keeping all Bk's works on almost all instances, except for {60, (... ?)}
    curM = kucera_get_Mminus(m) if is_prime(m) else kucera_get_Mminus_keep_all_Bk(m);
    # omega(a) - omega(a+1) = omega(m-a-1) - omega(m-a) if a > m/2
    # It seems that transforming a > m/2 into m-a-1 makes everything converge faster in most cases
    curM = list({ _a if _a < m/2 else m-_a-1 for _a in curM }); curM.sort();
    Mm = sti_vals_kucera_short_idx(m, curM, thetas=ta);
    Hm = Mm.hermite_form(include_zero_rows=False, algorithm=hnf_met);
    # If this does not generate the entire module, start from all (might be crude but this is rare).
    if (Hm != H): # Triggered for m=60, 720, ...
        if verbose: print("\t\x1b[33m[Warn]\tm={} Extending initial indices\x1b[0m".format(m));
        curM = kucera_get_X(m); # Very probably overkill
        curM = list({ _a if _a < m/2 else m-_a-1 for _a in curM }); curM.sort();
        Mm   = sti_vals_kucera_short_idx(m, curM, thetas=ta);
        Hm   = Mm.hermite_form(include_zero_rows=False, algorithm=hnf_met);
        assert(Hm == H), "\x1b[31m[Err]\tm={} HNF(Sti X(m)) != Expected\x1b[0m".format(m);
    if verbose: print ("\tSize of generating set:\t{},\texclude {}".format(len(curM), len(curM)-euler_phi(m)/2));
    
    # Now we do have a generating matrix, pop rows iteratively until
    # it is full rank with same HNF as target.
    idx   = -1;
    l_idx = []; # List of successfully removed (negative) position in curM
    l_exc = []; # List of successfully removed indices
    l_row = []; # List of successfully removed rows
    if verbose and euler_phi(m)/2 != len(curM): print("\tRemove:", end='', flush=True);
    while (Mm.nrows() != euler_phi(m)//2+1):
        # Go back in time if dead end
        # ------------------------------------------------
        while (idx < -len(curM)): # dead end
            # Restore the last _exc > m/2
            _restore_idx = len(l_exc)-1; # l_exc.index([_a for _a in l_exc if _a > m/2][-1]);
            # Restore in Mm, curM, + restore starting point for the search
            _n = len(l_exc)-_restore_idx; # Keep the machinery in case we have to shortcut the tree
            for _ in range(_n):
                idx  = l_idx.pop(); 
                Mm   = Mm.insert_row(Mm.nrows()+idx+1, l_row.pop());
                curM.insert(len(curM)+idx+1, l_exc.pop());
                if verbose: print(" \x1b[32m+{}\x1b[0m".format(curM[idx]), end='', flush=True);
            idx  = idx-1;
            
        # Find first element k from end of curM st. HNF( Mm(curM \ {k}) ) == H 
        # ------------------------------------------------
        exc   = curM[idx];
        row   = Mm[Mm.nrows()+idx];
        Mmt   = Mm.delete_rows([Mm.nrows()+idx]);
        Hmt   = Mmt.hermite_form(include_zero_rows=False, algorithm=hnf_met);

        # If the HNF is still good, save the exclusion
        # ------------------------------------------------
        if (Hmt == H):
            l_idx.append(idx); l_exc.append(exc); l_row.append(row);
            Mm = Mmt; curM.remove(exc);
            if verbose: print(" \x1b[36m{}\x1b[0m".format(exc), end='', flush=True);
        else: # Otherwise, consider one down in curM
            idx = idx-1;
            if verbose: print ("\x1b[33m*\x1b[0m", end='', flush=True);

        # Loop invariants
        # ------------------------------------------------
        assert(Mm.hermite_form(include_zero_rows=False) == H);       # At any time, Mm has correct HNF
        assert(Mm == sti_vals_kucera_short_idx(m, curM, thetas=ta)); # curM <--> Mm[1:] rows
        assert(Mm.nrows() >= euler_phi(m)//2+1);               # Didn't remove too much accidentally
    if verbose and len(l_exc) > 0: print("\n\tFound generating set", flush=True);
    
    assert (Mm == sti_vals_kucera_short_idx(m, curM, thetas=ta)), "\x1b[31m[Err] Mm and curM have been scrambled\x1b[0m";
    assert (Mm.hermite_form(include_zero_rows=False) == H), "\x1b[31m[Err] HNF(extract) != expected\x1b[0m";

    return curM;



# Short Stickelberger basis (using [BK21] alpha_m function) 
# --------------------------------------------------------
def sti_vals_alpha(m, verbose=False):
    # Get all alpha triples
    t = cputime(); alphas = kucera_alpha_all(m); t = cputime(t);
    if verbose: print("\talpha(b):\tt={:.2f} [{:.2f}/b]".format(t, t/len(alphas))) 

    # Get all thetas (restrict only to necessary ones)
    alphas_idx = list({*sum(alphas,[])}); # list of unique indices in alphas (lot smaller than m)
    t = cputime(); ta = theta_kucera_idx(m, alphas_idx); t = cputime(t);
    # Long version: all th(a), 0<a<m
    # t=cputime(); ta = theta_kucera_all_extend(m, thetas=thetas); t=cputime(t);
    if verbose: print("\tth(a):\tt={:.2f}".format(t)); assert (len(ta) == len(alphas_idx));
    
    # s(G): will be split as "real gens"
    sG  = ones_matrix(1,euler_phi(m))[0];
    Ma  = matrix(ZZ, [sG] + [ -sG + ta[alphas_idx.index(_a1)] + ta[alphas_idx.index(_a2)]
                              + ta[alphas_idx.index(_a3)] for _a1,_a2,_a3 in alphas ]);
    # MA = matrix(ZZ, [sG] + [ kucera_alpha(m,a,verbose=verbose) for a in S ]);
    
    assert (Ma.height() == 1), "||M||>1"; # Nb: This does not verify that all M_ij > 0 (ie. M_ij != -1)
    return Ma;





# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# Some check scripts
# -------------------------------------------

# ms = [ m[1] for m in sorted([ (euler_phi(m),m) for m in range(2,10000) if mod(m,4)!=2 and euler_phi(m) < 1300 and euler_phi(m) > 1000]) ]

# Checks for HNF
# for m in ms:
#     t1=cputime(); th=theta_kucera_all(m); t1=cputime(t1);
#     t2=cputime(); Ma = sti_vals_kucera_alpha(m).hermite_form(); t2=cputime(t2);
#     t3=cputime(); Mm = sti_vals_kucera(m,thetas=th).hermite_form(); t3=cputime(t3);
#     print((("\x1b[32m{}\x1b[0m" if Ma==Mm else "\x1b[31mWrong:{}\x1b[0m")+"\t[th/al/ku:{:.2f}/{:.2f}/{:.2f}]").format(m,t1,t2,t3), flush=True);


# Checks for [Am:Sm]
# for m in ms:
#     t1=cputime(); Am=sti_vals_kucera_alpha(m).submatrix(1,euler_phi(m)/2) - 1/2*ones_matrix(ZZ(euler_phi(m)/2)); t1=cputime(t1);
#     t2=cputime(); AmSm = Am.determinant().abs()*2; t2=cputime(t2);
#     t3=cputime(); hm = cf_hminus(ZZ(m)); t3=cputime(t3);
#     print((("\x1b[32m{}\x1b[0m" if AmSm==2^(2^(len(factor(m))-2)-1 if not is_prime_power(m) else 0)*hm else "\x1b[31mWrong:{}\x1b[0m")+"\t[Ma/det/hm:{:.2f}/{:.2f}/{:.2f}]").format(m,t1,t2,t3), flush=True);

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------
# Test tools for verifying the Stickelberger generators
# Computes the Stickelberger product idp^rel
def sti_id_product(K, p, rel, __idx_start=0):
    orb_p  = cf_orbit_p(K, p, __idx_start=__idx_start);
    id_sti = prod(map(pow, orb_p, rel));
    # assert (id_sti.is_principal());
    return id_sti;



# --------------------------------------------------------------------------------------------------
# Stickelberger's generators

#  TODO / Remaining problems:
#  --------------------------------------------------
#  - Say something about the height of the generators
#  - Combine relations to obtain "S-units" on some half-orbit ?
#    Or construct a viable explicit section for the projection Z[G] / <1+tau>
#  
#  >>> Other inertia degree seem not useful ??
#      eg. for inertia 2: pi pibar = 137^2, sti = -137 for all relations, give no info on S-units.
#  >>> (Probably) USE MAGMA.
#      Problem is that Sage does a very bad job handling ideals in "large" (>60) degree number fields.
#  
#  Nice properties of the Stickelberger relations: (globally, this is all well-known garbage)
#  --------------------------------------------------
#  - for N = k*m + r (r<m), P^{(N-s_N)\theta} = (P^{m\theta})^k * P^{(r-s_r)\theta},
#    hence, we can define v_{km+r} = k*v_{m+1} + v_r if useful.
#  - s_c \theta = \sum_{a mod m, (a,m)=1} { c.a / m } s_a^{-1}.
#  - s_e(P)^{(c-s_c)\theta} = P^{(ce-s_{ce})\theta} / (P^{(e-s_e)\theta)})^c,
#        ie. v_c pour s_e(P) = v_se - c*v_e,
#    so changing the starting point is useless for generators (but still gives extra short vectors)
#  - v_{m+1} - v_{m-1} is relation : <p> = prod_{P|p} P.
#  - weight (# of non zeroes) of relation w_i : n/2, except for the above trivial relation,
#  - for m prime (+powers ?), ||w_i|| = \sqrt{n}/2,
#  - [Schoof Sch10] (see Catalan's conjecture also) "w_i have +/- 1".
#  Modulo + or - 1's in the indices:
#  - w_{n/2+a} = w_{n/2-a}: so it is enough to consider only the *first* n/2-1+1 relations
#  - w_{m+1} is <p> = prod_{P|p} P, so what is w_{n/2} ? (if it has some meaning)
#  - For *all* relations w except w_{m+1}, w_{s_a} = 1-w_{conj{s_a}} = 1-w_{-a mod m} (indices: coeffs)
#    (For prime m. For m=2^t, w_{s_a}+w_{conj{s_a}} = 2 always.
#  - It seems that prod_{a st. w_{s_a} != 0} = 1 mod m.
#  - The matrix of (n) relations looks like (for prime m):
#    + big X with 01 up-left to down right and up-right to down-left
#    + big square U with left 0's / right 1's and bottom 1's
#    + middle columns of almost all 1's and almost all 0's
#    [[ for m=2^t, this is X:??, U:0's+2's+1's, M:1+1 except line m/2 ]]
#  
#        "id s_2 ...                    s_n" <---- columns correspond to ideals in the orbit
#                                                  i-th column is s_a(P), where a is the i-th elt
#          0 1 * *  ***  1 0  ***  * * 0 1   
#          0 0 1 *  ***  1 0  ***  * 0 1 1   <---- "w_{n/2 - a}""
#          0 * 0 1  ***  1 0  ***  0 1 * 1
#          0 * * 0 .     1 0     . 1 * * 1
#          .       . .   . .   . .       .
#          .         . . . . . .         .
#          .           . 1 0 1 .         .   <---  (n/2-1)-th relation
#          0 * * *  ***  0 1 1 *** * * * 1   <---- Unique w_{n/2} (?)
#          0           . 1 0 1 .         1
#          .         . . . . . .         .
#          .       . .   . .   . .       .
#          0 * * 0 .     1 0     . 1 * * 1
#          0 * 0 1  ***  1 0  ***  0 1 * 1
#          0 0 1 *  ***  1 0  ***  * 0 1 1   <---- "w_{n/2 + a}"
#          0 1 * *  ***  1 0  ***  * * 0 1   
#          1 1 1 1  ...  1 1  ...  1 1 1 1   <---- <p> = prod_{P|p} P
#          ^                             ^
#          |                             |
#          P_{id} never involved, always P_{s_-1}
#  
#  Complete matrix for m=23
#          0 1 0 0 1 0 0 0 1 0 1 0 1 0 1 1 1 0 1 1 0 1
#          0 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 1 1 1 0 1 1
#          0 1 0 1 1 0 0 0 1 1 1 0 0 0 1 1 1 0 0 1 0 1
#          0 0 0 0 1 0 1 0 0 0 1 0 1 1 1 0 1 0 1 1 1 1
#          0 1 1 0 0 1 0 0 1 0 1 0 1 0 1 1 0 1 1 0 0 1
#          0 0 0 0 1 0 1 0 1 1 1 0 0 0 1 0 1 0 1 1 1 1
#          0 1 0 1 0 0 0 1 1 0 1 0 1 0 0 1 1 1 0 1 0 1
#          0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 1 1 0 1 0 1 1
#          0 1 0 0 1 0 1 0 0 1 1 0 0 1 1 0 1 0 1 1 0 1
#          0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1
#          0 1 1 1 1 1 1 0 1 0 0 1 1 0 1 0 0 0 0 0 0 1
#          0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1
#          0 1 0 0 1 0 1 0 0 1 1 0 0 1 1 0 1 0 1 1 0 1
#          0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 1 1 0 1 0 1 1
#          0 1 0 1 0 0 0 1 1 0 1 0 1 0 0 1 1 1 0 1 0 1
#          0 0 0 0 1 0 1 0 1 1 1 0 0 0 1 0 1 0 1 1 1 1
#          0 1 1 0 0 1 0 0 1 0 1 0 1 0 1 1 0 1 1 0 0 1
#          0 0 0 0 1 0 1 0 0 0 1 0 1 1 1 0 1 0 1 1 1 1
#          0 1 0 1 1 0 0 0 1 1 1 0 0 0 1 1 1 0 0 1 0 1
#          0 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 1 1 1 0 1 1
#          0 1 0 0 1 0 0 0 1 0 1 0 1 0 1 1 1 0 1 1 0 1
#          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#
# Complete matrix for m=32
#          0 1 1 2 2 0 0 1 1 2 2 0 0 1 1 2
#          0 0 1 1 1 0 0 1 1 2 2 1 1 1 2 2
#          0 1 0 2 2 0 1 1 1 1 2 0 0 2 1 2
#          0 1 1 1 2 0 0 1 1 2 2 0 1 1 1 2
#          0 0 1 1 1 1 0 1 1 2 1 1 1 1 2 2
#          0 1 1 2 2 0 1 1 1 1 2 0 0 1 1 2
#          0 1 1 1 1 0 0 1 1 2 2 1 1 1 1 2
#          0 0 0 2 2 0 0 0 2 2 2 0 0 2 2 2
#          0 1 1 1 1 0 0 1 1 2 2 1 1 1 1 2
#          0 1 1 2 2 0 1 1 1 1 2 0 0 1 1 2
#          0 0 1 1 1 1 0 1 1 2 1 1 1 1 2 2
#          0 1 1 1 2 0 0 1 1 2 2 0 1 1 1 2
#          0 1 0 2 2 0 1 1 1 1 2 0 0 2 1 2
#          0 0 1 1 1 0 0 1 1 2 2 1 1 1 2 2
#          0 1 1 2 2 0 0 1 1 2 2 0 0 1 1 2
#          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1


# Following [Was97, §6] "i-s_i" relations
# ---------------------------------------------------------
# For relations "(i-s_i)\theta", following [Was97,§6], and especially proof of Th.6.10
# These correspond to cdw17_v

# Returns h = x^{mp}-1, phi_{p}(x^m), phi_{m}(x^p)
def __get_sti_moduli_x(m,p):
    assert (is_prime(p));
    
    Zy = PolynomialRing(ZZ, name='y', sparse=True);  y = Zy.gen();
    Zx = PolynomialRing(ZZ, name='x', sparse=False); x = Zx.gen();
    
    h      = x**(m*p)-1;
    # phip_m = Zx( cyclotomic_polynomial(p)(y^m) ); # Forever, but phi_p = sum x^i, i<p
    phip_m = Zx( { _k*m: 1 for _k in range(p) } );
    phim_p = Zx( cyclotomic_polynomial(m)(y**p) );

    return (h, phip_m, phim_p);


# Stickelberger elt is gamma^_a / \sigma_a(gamma), galois[_a]: zm -> zm^_a
# Use [Was97, pf.Lem.6.4] to obtain \sigma_a(g(chi)) = g( chi^a )
# Use [Was97, Lem.6.1]    to obtain 1/g(chi)  = bar(g(chi))/p
#                         to obtain g(chibar) = chi(-1) bar(g(chi)) # Useless
# ---> Compute for each a: g(chi)^{a-\sigma_a} = g(chi)^a * g(chibar^a) / q
# ---> chi is chi_p0 such that chi(a) = a mod p0
def sti_gens_cdw17_v(K, p0):
    m   = cf_get_conductor(K);
    chi = get_chi(K, m, p0); chi = chi_conj(chi); # [Was97] is the conjugate of the residue character 
    Zx  = PolynomialRing(ZZ, name='x');
    Fq  = chi[0].parent(); q = Fq.cardinality(); p = Fq.characteristic();
    # Compute the polynomial moduli used in the computation
    t = cputime(); (h, phip_m, phim_p) = __get_sti_moduli_x(m,p); t = cputime(t);
    print("h/phip_m/phim_p:\t{:.4f}s".format(t), flush=True);

    # g(chi) is re-used each time
    t = cputime(); g_x = gauss_sum_x(chi); t = cputime(t);
    print("g(chi):\t\t\t{:.4f}s".format(t), flush=True);
    
    print("Sti gens v_i = (i-s_i).theta(-1):", flush=True);
    stirel_idx = range(2, m//2+1); # [_a for _a in range(2,m+2) if gcd(_a,m) == 1];
    sti_gens   = [];
    __prev = 1; g_x_pow_a  = g_x;
    for _a in stirel_idx:
        t = cputime();
        # NB: power_mod(g_x, _a, h) is also a viable option, but less efficient when we want all powers
        # g_x_pow_a = power_mod(g_x, _a, h);
        g_x_pow_a = (g_x_pow_a * power_mod(g_x, _a-__prev, h)).mod(h);
        # chi^-a is conj(chi)^a
        # ga_x is hence q / g(chi)^{s_a} (. chi(-1))
        ga_x      = gauss_sum_chipow_x(chi, -_a);
        # This is g(chi)^{a-s_a}
        # NB: factor q (from ga_x) is removable only after all moduli have been applied
        sti_x     = (g_x_pow_a * ga_x).mod(h).mod(phip_m).mod(phim_p) / q;
        # All non zero monomials in sti_x are of degree p*... [ie, sti_x is in Q(\zeta_m) = Q(x^p)]
        sti_gen   = K(Zx( {_key/p: _val for (_key,_val) in sti_x.dict().items()} ));
        t = cputime(t);
        print("\t{}-s_{}\tt={:.4f}s".format(_a,_a,t), flush=True);
        sti_gens.append(sti_gen);
        __prev = _a;

    return sti_gens;


# Computing directly w_i = v_i / v_{i-1} to avoid integers overflows
# [This way, g(w_i)*bar(g(w_i)) = q , vs "q^i" for v_i]
# Using g(chi)^{d-c} / s_d * s_c = g(chi^c) * g(chi^-d) / q
def sti_gens_cdw17_w(K, p0):
    m   = cf_get_conductor(K);
    n   = euler_phi(m);
    chi = get_chi(K, m, p0); chi = chi_conj(chi); # [Was97] is the conjugate of the residue character 
    Zx  = PolynomialRing(Integers(), name='x');
    Fq  = chi[0].parent(); q = Fq.cardinality(); p = Fq.characteristic();
    # Compute the polynomial moduli used in the computation
    t = cputime(); (h, phip_m, phim_p) = __get_sti_moduli_x(m,p); t = cputime(t);
    print("\th/phip_m/phim_p:\tt={:.4f}s".format(t), flush=True);

    # g(chi) is re-used each time
    t = cputime(); g_x = gauss_sum_x(chi); t = cputime(t);
    print("\tg(chi):\t\t\tt={:.4f}s".format(t), flush=True);

    print("\tSti gens w_i = v_i - v_{i-1}:", flush=True);
    # stirel_idx = range(2, m//2+2); # Gives "all" generators, but not the CDW17 one (which can "jump")
    stirel_idx = [_a for _a in range(2,m+2) if gcd(_a,m) == 1][:n//2];
    sti_gens_w = [K(q)]; # sti_gens_w[i] will be conjugate of gen of omega(i)-omega(i+1), [0] is "m+1"
    __prev = 0; ga_prev_x = Zx(1);
    for _a in stirel_idx:
        # See comments of get_sti_gens_vx()
        t = cputime();
        g_x_pow_diff = power_mod(g_x, _a-__prev, h);
        # Compute g(chi^a)
        chi_pow_a = chi_pow(chi, _a);
        ga_x      = gauss_sum_x(chi_pow_a);
         # g(chi^(-a)) = q / s_a(g(chi))
        gca_x     = gauss_sum_chiconj_x(chi_pow_a, g_chi=ga_x);
        # This is g(chi)^{s_prev-s_a} * q
        g_sig_x   = (ga_prev_x*gca_x).mod(h);
        # g(chi)^{a-pr} / s_a(g(chi)) * s_pr(g(chi)) = g(chi)^{a-pr} * g(chi^pr) * g(chi^(-a)) / q
        # NB: factor q (from gca_x) is removable only after all moduli have been applied
        sti_x     = (g_x_pow_diff * g_sig_x).mod(h).mod(phip_m).mod(phim_p) / q;
        # All non zero monomials in sti_x are of degree p*... [ie, sti_x is in Q(\zeta_m) = Q(x^p)]
        sti_gen   = K(Zx( {_key/p: _val for (_key,_val) in sti_x.dict().items()} ));
        t = cputime(t);
        print("\t1-s_{}+s_{}\tt={:.4f}s".format(_a,_a-1,t), flush=True);
        sti_gens_w.append(sti_gen);
        __prev = _a; ga_prev_x = ga_x;

    return sti_gens_w;



# Code from ideas of [BK21] that write nicely as Jacobi sums
# ---------------------------------------------------------

# Returns generators for set of { [a,b,c] } indexes that correspond to the Stickelberger elt:
#         \th(a) + \th(b) + \th(c) - sG , where m | a+b+c
# Using Jacobi sums
# S is a list of [a_i,b_i,c_i] that verify (a_i+b_i+c_i)=0[m]
def sti_gens_alpha_idx(K, p0, S):
    m = cf_get_conductor(K);

    # Residue norm character
    t = cputime(); res_chi = get_chi(K, m, p0); t = cputime(t);
    Fq_a = res_chi[0]; q = Fq_a.parent().cardinality();
    print("\tchi(p0)\tt={:.4f}s".format(t));
    # Computing DLog tables for Jacobi sums
    t = cputime(); dl = jacobi_sum_dlog_table(Fq_a); t = cputime(t);
    print("\tdlog [{}]\tt={:.4f}s".format(q, t));
    # Generators as Jacobi sums J(i[0],i[1])
    sti_gens_al = [K(q)];
    for _a,_b,_c in S:
        assert(mod(_a+_b+_c,m) == 0);
        _chi_a = chi_pow(res_chi, _a);
        _chi_b = chi_pow(res_chi, _b);

        t = cputime(); _j_ab  = jacobi_sum_x(_chi_a, _chi_b, dlog=dl); t = cputime(t);
        _j_ab  = K(_j_ab);
        print("\t[{}+{}+{}]\tt={:.4f}s".format(_a,_b,_c,t), flush=True);
        sti_gens_al.append(_j_ab);
        
    return sti_gens_al;


# For the whole basis
def sti_gens_alpha(K, p0):
    m  = cf_get_conductor(K);
    S  = kucera_alpha_all(m);
    sk = sti_gens_alpha_idx(K, p0, S);
    return sk;
    

# This induces a simplified computation for the cdw17/cdw21_w elements
# Compute the [CDW21] generators (actually their conjugate version as in [CDW17]), but using Jacobi sums
def sti_gens_cdw21_w(K, p0):
    m       = cf_get_conductor(K);
    jac_idx = [ [1,_a-1,m-_a] for _a in range(2,(m-1)//2+2) ];
    sw      = sti_gens_alpha_idx(K, p0, jac_idx);
    #t = cputime(); swc = [ cf_complex_conjugate(_sw) for _sw in sw ]; t = cputime(t);
    #print("\tconjugate\tt={:.4f}s [{:.4f}/g]".format(t, t/len(swc)));
    return sw;

# //-- END OF FILE



# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# __Trash
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# __old, for XP (recompute the thetas each time)
def __unused_kucera_alpha_vec(m, b, verbose=False):
    [a1,a2,a3] = kucera_alpha(m, b, verbose=verbose);
    am         = theta_kucera_a(m, a1) + theta_kucera_a(m, a2) - theta_kucera_a(m, m-a3);
    return am;

def __unused_omega_kucera_21(m,a):
    sG = QQ(1/2)*ones_matrix(1, euler_phi(m))[0];
    return theta_kucera_a(m,a) - sG;

# Compute all thetas(b) for 1 <= b < m
# Using th(a) + th(m-a) = s(G)
def __unused_theta_kucera_all_extend(m, thetas=[]):
    ta = theta_kucera_all(m) if len(thetas) == 0 else thetas; assert(len(ta) == m//2+1);
    # Extend using th(a) + th(m-a) = s(G)
    sG = ones_matrix(1, euler_phi(m))[0];
    ta = ta + [sG - _ta for _ta in ta[ZZ(ceil(m/2)-1):0:-1]];  assert(len(ta) == m);
    return ta;

## Seems that
## omega(i) - omega(i+1)   + ((i+1-s_{i+1}) - (i-s_{i}))   = s(G)
## omega(1) - omega(2)     + (2-s_2) theta(-1)             = s(G)
## omega(m-1) = s(G) # /!\ for the "natural" continuation of omega beyond m/2

# tests
def __old_kucera_short_Mminus_no_folding(m, verbose=True):
    hnf_met = 'default';
    
    # Compute all thetas. This is quite costly but can be computed just once.
    t = cputime(); ta = theta_kucera_all(m); t = cputime(t);
    if verbose: print ("Compute all t(a): t={:.2f}".format(t));
    
    # Get the correct (mathematically proven) matrix from [Kuc92], and use its HNF for following tests.
    t = cputime(); H = sti_vals_kucera(m, thetas=ta).hermite_form(); t = cputime(t);
    if verbose: print ("Kucera basis HNF: t={:.2f}".format(t));
    
    # Find a minimal set S st. {s(G)} {om(a)-om(a+1), a in S} (careful for a=m//2) gives the same HNF
    # NB: Mminus is (provably) correct for prime m, never the case for prime powers or composite m.
    #     keeping all Bk's works on almost all instances, except for {60, (... ?)}, 
    curM = kucera_get_Mminus(m) if is_prime(m) else kucera_get_Mminus_keep_all_Bk(m);
    Mm = sti_vals_short_from_index(m, curM, thetas=ta);
    Hm = Mm.hermite_form(include_zero_rows=False, algorithm=hnf_met);
    # If this is does not generate the entire module, start from all (might be crude but this is rare).
    if (Hm != H): # Triggered for m=60, 720, ...
        if verbose: print("\x1b[33m[Warn]\tm={} Extending initial indices\x1b[0m".format(m));
        curM = kucera_get_X(m);
        Mm   = sti_vals_short_from_index(m, curM, thetas=ta);
        Hm   = Mm.hermite_form(include_zero_rows=False, algorithm=hnf_met);
        assert(Hm == H), "\x1b[31m[Err]\tm={} HNF(Sti X(m)) != Expected\x1b[0m".format(m);

    # Now we do have a generating matrix, pop rows iteratively until
    # it is full rank with same HNF as target.
    idx   = -1; # Where to point in curM
    l_idx = []; # List of successfully removed (negative) position in curM
    l_exc = []; # List of successfully removed indices
    l_row = []; # List of successfully removed rows
    if verbose: print("Remove: ", end='', flush=True);

    while (Mm.nrows() != euler_phi(m)//2+1):
        # Go back in time if dead end
        # ------------------------------------------------
        while (idx < -len(curM)): # dead end
            # Restore the last _exc > m/2
            _restore_idx = len(l_exc)-1; # l_exc.index([_a for _a in l_exc if _a > m/2][-1]);
            # Restore in Mm, curM, + restore starting point for the search
            _n = len(l_exc)-_restore_idx;
            for _ in range(_n):
                idx  = l_idx.pop(); 
                Mm   = Mm.insert_row(Mm.nrows()+idx+1, l_row.pop());
                curM.insert(len(curM)+idx+1, l_exc.pop());
                if verbose: print("\x1b[32m+{}\x1b[0m ".format(curM[idx]), end='', flush=True);
            idx  = idx-1;
            
        # Find first element k from end of curM st. HNF( Mm(curM \ {k}) ) == H 
        # ------------------------------------------------
        _idx  = idx;
        _exc  = curM[_idx];
        _row  = Mm[Mm.nrows()+_idx];
        Mmt   = Mm.delete_rows([Mm.nrows()+_idx]);
        Hmt   = Mmt.hermite_form(include_zero_rows=False, algorithm=hnf_met);

        # If it does not work, a seemingly good strategy is to try for m-_exc.
        # ------------------------------------------------
        if (Hmt != H) and (_exc > m/2) and (m-_exc in curM):
            _idx = -len(curM) + curM.index(m-_exc);
            _exc = curM[_idx];
            _row = Mm[Mm.nrows()+_idx];
            Mmt = Mm.delete_rows([Mm.nrows()+_idx]);
            Hmt = Mmt.hermite_form(include_zero_rows=False, algorithm=hnf_met);
            # We should have here a "idx = idx-1", unless Hmt != H.

        # If the HNF is still good, save the exclusion
        # ------------------------------------------------
        if (Hmt == H):
            l_idx.append(_idx); l_exc.append(_exc); l_row.append(_row);
            Mm = Mmt; curM.remove(_exc);
            if verbose: print("\x1b[{}m{}\x1b[0m ".format("36" if (_exc > m/2) or (_idx == idx) else "31", _exc), end='', flush=True); # Print in red if it has been "flipped"
        else: # Otherwise, consider one down in curM
            idx = idx-1;
            if verbose: print ("\x1b[33m*\x1b[0m", end='', flush=True);

        # Loop invariants
        # ------------------------------------------------
        assert(Mm.hermite_form(include_zero_rows=False) == H);       # At any time, Mm has correct HNF
        assert(Mm == sti_vals_short_from_index(m, curM, thetas=ta)); # curM <--> Mm[1:] rows
        assert(Mm.nrows() >= euler_phi(m)//2+1);               # Didn't remove too much accidentally
    if verbose: print("");

    # So, why did we quit the while loop ?
    #if (idx < -len(curM)): # By construction, this means Mm.hermite_form() != H (ie, with zero rows)
    #print("\x1b[31m*** ERROR *** No generating set extracted (m={})\x1b[0m".format(m));
    if verbose: print("\x1b[32mFound generating set\x1b[0m");
    
    assert (Mm == sti_vals_short_from_index(m, curM, thetas=ta));
    assert (Mm.hermite_form(include_zero_rows=False) == H), "\x1b[31m[Err] HNF(extract) != expected\x1b[0m";
    assert (Mm.nrows() == euler_phi(m)/2+1); # By construction, ok if _i >= -len(curM)

    retM = [ _a if _a < m/2 else m-_a-1 for _a in curM];
    retM.sort();
    assert(sti_vals_short_from_index(m, retM, thetas=ta).hermite_form() == H);

    return retM;


def __old_kucera_short_depth1_fails(m, verbose=True):
    hnf_met = 'default';
    
    # Compute all thetas. This is quite costly but can be computed just once.
    t = cputime(); ta = theta_kucera_all(m); t = cputime(t);
    if verbose: print ("Compute all t(a): t={:.2f}".format(t));
    
    # Get the correct (mathematically proven) matrix from [Kuc92], and use its HNF for following tests.
    t = cputime(); H = sti_vals_kucera(m, thetas=ta).hermite_form(); t = cputime(t);
    if verbose: print ("Kucera basis HNF: t={:.2f}".format(t));
    
    # Find a minimal set S st. {s(G)} {om(a)-om(a+1), a in S} (careful for a=m//2) gives the same HNF
    # NB: Mminus is (provably) correct for prime m, never the case for prime powers or composite m.
    #     keeping all Bk's works on almost all instances, except for {60, (... ?)}, 
    curM = kucera_get_Mminus(m) if is_prime(m) else kucera_get_Mminus_keep_Bk(m);
    Mm = sti_vals_short_from_index(m, curM, thetas=ta);
    Hm = Mm.hermite_form(include_zero_rows=False, algorithm=hnf_met);
    # If this is does not generate the entire module, start from all (might be crude but this is rare).
    if (Hm != H): # Triggered for m=60, 720, ...
        if verbose: print("\x1b[33m[Warn]\tm={} Extending initial indices\x1b[0m".format(m));
        curM = kucera_get_X(m);
        Mm   = sti_vals_short_from_index(m, curM, thetas=ta);
        Hm   = Mm.hermite_form(include_zero_rows=False, algorithm=hnf_met);
        assert(Hm == H), "\x1b[31m[Err]\tm={} HNF(Sti X(m)) != Expected\x1b[0m".format(m);

    # Now we do have a generating matrix, pop rows iteratively until
    # it is full rank with same HNF as target.
    idx   = -1; # Where to point in curM
    l_idx = []; # List of successfully removed (negative) position in curM
    l_exc = []; # List of successfully removed indices
    l_row = []; # List of successfully removed rows
    last_restore_idx = -1; last_restore = 0;
    if verbose: print("Remove: ", end='', flush=True);
    while (Mm.nrows() != euler_phi(m)//2+1) and idx >= -len(curM):
        # Find first element k from end of curM st. HNF( Mm(curM \ {k}) ) == H 
        _idx  = idx;
        _exc  = curM[_idx];
        _row  = Mm[Mm.nrows()+_idx];
        Mmt   = Mm.delete_rows([Mm.nrows()+_idx]);
        Hmt   = Mmt.hermite_form(include_zero_rows=False, algorithm=hnf_met);

        # If it does not work, a good strategy is to try for m-_exc.
        if (Hmt != H) and (_exc > m/2) and (m-_exc in curM):
            _idx = -len(curM) + curM.index(m-_exc);
            _exc = curM[_idx]; # = "m-_exc"
            _row = Mm[Mm.nrows()+_idx];
            Mmt = Mm.delete_rows([Mm.nrows()+_idx]);
            Hmt = Mmt.hermite_form(include_zero_rows=False, algorithm=hnf_met);
            # NB: if this works, it seems to be *always* followed by a "i = i-1" step.
            
        if (Hmt == H):
            l_idx.append(_idx); l_exc.append(_exc); l_row.append(_row);
            Mm = Mmt; curM.remove(_exc);
            if verbose: print("\x1b[{}m{}\x1b[0m ".format("36" if (_exc > m/2) or (_idx == idx) else "31", _exc), end='', flush=True); # Print in red if it has been "flipped"
            continue;
        
        if verbose: print ("\x1b[33m*\x1b[0m", end='', flush=True);
        # + TEST LAST_IDX != 0
        if (idx == -len(curM)): # dead end
            # print("\ncurM before restore {}".format(str(curM).replace(' ','')));
            # Restore the last _exc > last restore
            #print("\ncurM before restore {}".format(str(curM).replace(' ','')));
            #_cands   = l_exc if last_restore_idx == -1 else l_exc[:last_restore_idx]; # TEST LAST = 0?
            _restore_idx = len(l_exc)-1 if last_restore_idx == -1 else last_restore_idx-1;
            _restore     = l_exc[_restore_idx];

            # pop:restore x times, x=len(l_exc)-_restore ? l=[0,1,2,3,4], len=5, restore=2 => [0,1], 3 pops
            _n = len(l_exc)-_restore_idx;
            for _ in range(_n):
                # Restore in Mm, curM, + restore starting point for the search
                idx  = l_idx.pop(); # print("last working: {}".format(idx));
                Mm   = Mm.insert_row(Mm.nrows()+idx+1, l_row.pop());
                curM.insert(len(curM)+idx+1, l_exc.pop());
                if verbose: print("\x1b[32m+{}\x1b[0m ".format(curM[idx]), end='', flush=True);

            # Take out again the last checkpoint
            if last_restore != 0:
                _idx = -len(curM) + curM.index(last_restore);
                Mm = Mm.delete_rows([Mm.nrows()+_idx]);
                curM.remove(last_restore);
                if verbose: print("\x1b[31m/!\\{}\x1b[0m ".format(last_restore), end='', flush=True);
            last_restore_idx = _restore_idx;
            last_restore     = _restore;
            #print("\ncurM after restore {}".format(str(curM).replace(' ','')));
            assert(len(l_exc) == len(l_idx) == last_restore_idx);
            
        idx = idx-1;
        # print("(i={})".format(idx), end='', flush=True);
        
        assert(Mm.hermite_form(include_zero_rows=False) == H);
        assert(Mm.nrows() >= euler_phi(m)//2+1);

    if verbose: print("");
    #Mm = kucw_matrix_M(m, curM);
    #print ("#curM={}, i={}, {} [LAST]".format(len(curM), _i, str(curM).replace(' ','')));
    if (idx < -len(curM)): # By construction, this means Mm.hermite_form() != H
    # if Mm.hermite_form(algorithm=hnf_met) != H:
        print("\x1b[31m*** ERROR *** No generating set extracted (m={})\x1b[0m".format(m));
    elif verbose:
        print("\x1b[32mAll good\x1b[0m");


    assert (Mm == sti_vals_short_from_index(m, curM, thetas=ta));
    # assert (Mm.hermite_form(include_zero_rows=False) == H), "HNF(extract) != expected";
    # assert (Mm.nrows() == euler_phi(m)/2+1); # By construction, ok if _i >= -len(curM)

    return curM;




# Extended Stickelberger matrix (with the pi*conj(pi) lines) here for m=23
# M = get_ws(K,m)[:11] + [[0]*_i + [1] + [0]*(K.degree()-2*_i-2) + [1]+[0]*_i for _i in range(11)]

# With Real generators
# pp   = get_orbit_p(K,m,p);
# chi  = get_chi(K,m,pp[0]);
# Wext = get_sti_gens_wx(K,m,chi)[:11] + [ (pp[_i]*pp[21-_i]).gens_reduced()[0] for _i in range(11) ]; 



# ?
def __old_sti_gens_kucera_short(K, p0):
    m       = cf_get_conductor(K);
    jac_idx = [ [1,_a-1,m-_a] for _a in range(2,m//2+2) ];
    sw      = sti_gens_kucera_alpha(K, p0, jac_idx);
    return sw;


# ???
def __old_sti_gens_kucera_short_idx(K, p0, idx_set):
    m       = cf_get_conductor(K);
    jac_idx = [ [1,]];
    sw      = sti_gens_kucera_short(K, p0);
    sw_gens = [ sw[_a] if _a < m/2 else sw[m-_a-1] for _a in idx_set ];
    return sw_gens;



# ---------------------------------------------------------------------------------------------------
# OLD CODE FOR REFERENCE

# # [OLD VERSION]
# # Blocks when m*p > 10^6
# def __old_get_sti_gens_v(K,m,exp_chipp,Fq,q):
#     set_idx = [_a for _a in range(2,m+2) if gcd(_a,m) == 1];
#     QB = CyclotomicField(); # It seems that **not** creating "CyclotomicField(m*p)" is important
#     zm = QB.zeta(m);
#     t = cputime(); wgam = - gauss_sum(zm**exp_chipp, Fq); t = cputime(t);
#     print("g(chi):\t\t{}s".format(t));
    
#     # for each a in set_idx, we want g(chip)^{a-\sigma_a}
#     sti_elts_v = [];
#     for _a in set_idx:
#         # Stickelberger elt is gamma^_a / galois[_a](gamma), galois[_a]: zm -> zm^_a
#         # Use [Was97, pf.Lem.6.4] to obtain \sigma_a(g(chi)) = g( chi^a )
#         # Use [Was97, Lem.6.1]    to obtain 1/g(chi)  = g(chibar)/p
#         #                         to obtain g(chibar) = chi(-1) bar(g(chi)) 
#         t = cputime();
#         sti_elt = K( -wgam**_a * gauss_sum( zm**(- exp_cchipp*_a), Fq) / q ); # YES !
#         t = cputime(t);
#         print("[Time sti_elt {}-s_{}]\t{:.4f}s".format(_a,_a,t));
#         sti_elts_v.append(sti_elt);

#     return sti_elts_v;

# # [OLD VERSION]
# # Same, for the "w" relations. Mainly compute "v" elements and divide. 
# def __old_get_sti_gens_w(K,m,exp_chipp,Fq,q):
#     # First is g(chi)^(2-s_2), next ones are "^(i+1-s_i+1 / i-s_i)" (eg. for m prime)
#     sti_elts = get_sti_elt_v(K,m,exp_chipp,Fq,q);
#     sti_elts_w = [sti_elts[0]]+[ sti_elts[i]/sti_elts[i-1] for i in range(1, len(sti_elts))];
#     return sti_elts_w;











