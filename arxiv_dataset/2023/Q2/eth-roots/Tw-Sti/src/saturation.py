from sage.all import *
from nf import *
from cyclotomics import *


# --------------------------------------------------------------------------------------------------
# e-Saturation of independent multiplicative sets
#
# From V = v_1, ..., v_k, compute W = w_1, ..., w_k such that:
#         \forall u \in V,     u \in K^e \cap V ==> u \in W.
# It was impossible to get the code from [BFHP20, Alg.4.9] (or compile code of [BBLVV17, §4] for e=2.)
# So, we might again reinvent the wheel here.
#
# 1. Choice of characters
#    - for e = 2:   \chi_pid (v) = v^{(Norm(pid)-1)/2} [pid] =   1 if (v mod pid) \in pid^2,
#                                                               -1 if v mod pid \notin pid^2,
#                                                                0 if v \in pid.
#    - for e \ne 2: \chi_pid(v)  = \log_a (v mod pid)  [e], with F_q* = <a>, q = Norm(pid) = 1 [e].
#      [More precisely, this is modulo gcd(d,q-1)].
#
# 2. Norm characters choices:
#    For a collection of pid, we can have that u \in { ker chi_pid }, but u \notin W^e.
#    - The probability of success for uniform characters is roughly 1 - 1/2^{#pid - dimV}.
#    - Heuristically, it holds if we choose all pid's of different (prime) norm
#      (See [BBLVV17] or [NFS93])
#    - for all v_i, all pid, we MUST ensure v_i \notin pid (if this happens, pid must be discarded).
#    - we gain no information if gcd(q-1,e) = 1 (everyone is a e-th power in this case).
# 
#    We choose to:
#    - take our chance to use half-orbits (sets will be stable by "conjugation")
#      to avoid using ideals of norms growing too quickly
#      /!\ Beware that the above heuristic success probability does NOT hold anymore.
#    - of cardinal slightly greater than the (algebraic) dimension of V (say, 5 or 10):
#    - use split primes of prime norm p = 1 [e].
#
# 3. Units, Torsion units.
#    For all this to work, units and torsion units (for d=2 or d=m) should not be forgotten in V.
#
# 4. d-th root computations.
#    //////////   TBD, see [Thomé]   //////////
#
# --------------------------------------------------------------------------------------------------


_PREC = 3000
from lattice import *
#from twphs_algo import *
def logU_reduce_idgen(idgen, Bu, u, p_inf):
    lgen = log_embedding(idgen, p_inf, b_prec=_PREC);
    assert(fp.fp_assert_equal("sum(log_t)-ln|N(g)|", [sum(lgen)], [idgen.norm().abs().log(prec=_PREC)],
                              target=_PREC));
    t = proj_H(lgen, Bu.ncols());
    v = cvp_babai_NP(Bu, t);
    s = twphs_build_solution(idgen, v, Bu, u);
    return s;


def logU_reduce(V, units):
    K = V[0].parent();
    phi = get_inf_places(K, b_prec=_PREC);
    Bu = matrix([log_embedding(_u, phi, b_prec=_PREC) for _u in units]);
    pH = get_projection_H(Bu.ncols(), Bu.ncols(), b_prec=_PREC);
    logv = [log_embedding(_v, phi, b_prec=_PREC) for _v in V];
    targets = [ _lv*pH for _lv in logv ];
    cvp = [ cvp_babai_NP(Bu, _t) for _t in targets ];
    sols = [ twphs_build_solution(_g, _cvp, Bu, units) for _g,_cvp in zip(V,cvp) ];
    return sols;
    

    
# --------------------------------------------------------------------------------------------------
# Maps K --> F_q* / (F_q*)^d (= Z/dZ) where q = Norm(pid)
# For d = 2, the fastest is to test whether or not it is a square mod pid.
def log_chip_quadratic(pid, a):
    # O_K / pid O_K = F_q
    (p, g) = pid_fast_gens_two(pid); f = g.degree();
    q      = ZZ(p**f); Fq = GF(q);
    # a mod pid: using 'pid.small_residue(a)' is several times slower
    Fqy = PolynomialRing(Fq, name='y'); # y = Fqy.gen();
    om  = Fq( Fqy(a.polynomial()).mod(Fqy(g)) );
    # If om \in pid, return +Infinity (character must be discarded)
    if (om == 0):
       print ("/!\\ Warning: {} = 0 mod {}, must be discarded".format(a, pid_fast_gens_two()));
       return +Infinity;

    # Quadratic character mod pid (log_1)
    log_chip = 0 if om.is_square() else 1;

    # # ---- [OLD] log(a mod pid) [2]: 30% slower
    # g  = Fq.multiplicative_generator();
    # log_chip = om.log(g).mod(2);
    # # ---- [OLD] power_mod --> +/-1: 30% slower
    # log_chip = power_mod(om, (q-1)/ZZ(2)), q);
    # log_chip = 0 if log_chip == 1 else 1;
    # # ---- [OLD] power in Fq: 5% slower
    # log_chip = om**((q-1)/ZZ(2));
    # log_chip = 0 if log_chip == 1 else 1;

    return log_chip;


# To obtain the class of (a mod pid) modulo pid^d, we solve a DL mod norm(pid) and return log mod d
# NB: We require that d divides (Norm(pid)-1); could be extended by returning:
#         (q-1)/gcd(q-1,d) . (log mod gcd(q-1,d)) mod d
def log_chip_dth_power(pid, a, d=2):
    # O_K / pid O_K = F_q
    (p, g) = pid_fast_gens_two(pid); f = g.degree();
    q      = ZZ(p**f); Fp = GF(p, proof=False); Fq = GF(q, proof=False);
    # assert(is_prime(d) and mod(q-1, d) == 0);
    assert(mod(q-1,d)==0);
    # a mod pid: using 'pid.small_residue(a)' is several times slower
    Fpy = PolynomialRing(Fp, name='y'); y = Fpy.gen();
    om = Fq( Fpy(a.polynomial()).mod(Fpy(g)) ); # Conversion to Fq for f>1 only works if polynomials are over Fp
    if (om == 0):
        print ("/!\\ Warning: {} = 0 mod {}, must be discarded".format(a, pid_fast_gens_two()));
        return +infinity;
    
    Fq_g       = (Fq.multiplicative_generator());
    log_chip_d = om.log(Fq_g).mod(d);
    
    return log_chip_d;


def log_chip_dth_power_all(pid, la, d=2):
    # O_K / pid O_K = F_q
    (p, g) = pid_fast_gens_two(pid); f = g.degree();
    q      = ZZ(p**f); Fp = GF(p, proof=False); Fq = GF(q, proof=False);
    assert(is_prime(d) and mod(q-1, d) == 0);
    # a mod pid: using 'pid.small_residue(a)' is several times slower
    Fpy    = PolynomialRing(Fp, name='y'); y = Fpy.gen();
    Fq_g   = (Fq.multiplicative_generator());

    log_chip_d = [];
    for _a in la:
        _res_a = Fq( Fpy(_a.polynomial()).mod(Fpy(g)) );
        if (_res_a == 0):
            print ("/!\\ Warning: {} = 0 mod {}, must be discarded".format(_res_a, pid_fast_gens_two()));
            log_chip_d.append(+infinity);
        else:
            log_chip_d.append(_res_a.log(Fq_g).mod(d));
    
    return log_chip_d;

def log_chip_dth_power_all_ze(pid, la, d=2):
    # O_K / pid O_K = F_q
    (p, g) = pid_fast_gens_two(pid); f = g.degree();
    q      = ZZ(p**f); Fq = GF(q);
    assert(is_prime(d) and mod(q-1, d) == 0);
    # a mod pid: using 'pid.small_residue(a)' is several times slower
    Fqy    = PolynomialRing(Fq, name='y'); y = Fqy.gen();
    Fq_g   = (Fq.multiplicative_generator());
    zeta_d = Fq_g**((q-1)/d);
    
    log_chip_d = [];
    for _a in la:
        _res_a = Fq( Fqy(_a.polynomial()).mod(Fqy(g)) );
        if (_res_a == 0):
            print ("/!\\ Warning: {} = 0 mod {}, must be discarded".format(_res_a, pid_fast_gens_two()));
            log_chip_d.append(+infinity);
        else:
            _res_zd = _res_a**((q-1)/d);
            log_chip_d.append(_res_zd.log(zeta_d));
    
    return log_chip_d;


def log_chip_dth_power_all_bsgs(pid, la, d=2):
    # O_K / pid O_K = F_q
    (p, g) = pid_fast_gens_two(pid); f = g.degree();
    q      = ZZ(p**f); Fq = GF(q);
    assert(is_prime(d) and mod(q-1, d) == 0);
    # a mod pid: using 'pid.small_residue(a)' is several times slower
    Fqy    = PolynomialRing(Fq, name='y'); y = Fqy.gen();
    Fq_g   = (Fq.multiplicative_generator());
    log_ord = ZZ((q-1)/d);
    zeta_d  = Fq_g**log_ord;
    
    # Baby steps
    bsgs_k    = ZZ(ceil(sqrt(d)));
    t = cputime(); bsgs_logs = { zeta_d**_i: _i for _i in range(bsgs_k) }; t = cputime(t);
    print("Time BSGS: {:.2f}s".format(t)); #print(bsgs_logs);
    inv_zeta_k = zeta_d**(-bsgs_k);
    z_dk       = [ inv_zeta_k**_j for _j in range(bsgs_k) ];
    
    # Loop on all giant steps
    log_chip_d = [];
    for _a in la:
        _res_a = Fq( Fqy(_a.polynomial()).mod(Fqy(g)) );
        if (_res_a == 0):
            print ("/!\\ Warning: {} = 0 mod {}, must be discarded".format(_res_a, pid_fast_gens_two()));
            log_chip_d.append(+infinity);
        else:
            _res_zd = _res_a**(log_ord);
            for _j in range(bsgs_k):
                _y = _res_zd * z_dk[_j];
                _log_y = bsgs_logs.get(_y);
                if _log_y != None:
                    log_chip_d.append(bsgs_k*_j + _log_y);
                    break;
    
    return log_chip_d;




# --------------------------------------------------------------------------------------------------
# Choice of suitable prime ideals (see 2. above)
__SATURATION_OVERHEAD = 10;

# Returns the next prime q > p such that q = 1 mod div_pm1
def __next_suitable_prime(p, div_pm1):
    # Start from _p0 > p st. _p0 == 1 [md]
    p0 = p + div_pm1 - p.mod(div_pm1) + 1;
    while not is_prime(p0):
        p0 += div_pm1;
    return p0;


# Returns max { p | norm(v), v in V }
def __smooth_V(V):
    smooth_V = max(sum( ([ _f[0] for _f in factor(_v.norm()) ] for _v in V), []));
    return smooth_V;


# Returns a collection of sufficiently many split prime ideals of (prime) norm Np = 1 [d] > smooth_V
def sat_get_suitable_chip(V, d=2, smooth_V=0, __per_orbit=1, __overhead=__SATURATION_OVERHEAD):
    nb_V = len(V);
    assert(nb_V > 0);
    K    = V[0].parent();
    m    = cf_get_conductor(K); # Assumes K is cyclotomic !!
    n    = K.degree();
    print("test: finished small computation");
    # Nb characters to use (driven by the predefined _SATURATION_HEURISTIC, see comments above)
    nb_pid   = nb_V + __overhead;#__SATURATION_OVERHEAD;
    print("\ttargetting nb_pid={} characters".format(nb_pid));
    # p_max(V)
    smooth_V = __smooth_V(V) if smooth_V == 0 else smooth_V;
    print("\tusing smooth_V={}".format(smooth_V));
    
    # List of enough suitable primes p = 1 [m] (split) and 1 [d] (pid / pid^d is not trivial)
    div_pm1  = lcm(m,d); # e.g. if m=2^k, d=2, any prime p=1[m] is 1[d];
                         #      if m=q,   d=3, q!=d, we need p=1[3m] (so p=1[6m] but we don't use it).
    l_pZZ    = [__next_suitable_prime(smooth_V, div_pm1)];
    while (__per_orbit*len(l_pZZ) < nb_pid): # each prime in l_pZZ yields n/2 ("indepndt") prime ideals
        l_pZZ.append(__next_suitable_prime(l_pZZ[-1], div_pm1));
    print("\tusing prime norms={}".format(l_pZZ));
    # This is the costly part !
    l_pid    = sum((cf_orbit_p(K, _p)[:__per_orbit] for _p in l_pZZ), [])[:nb_pid];

    return l_pid;



# --------------------------------------------------------------------------------------------------
# d-Saturation (works only in cyclotomic fields)

# Returns the dth-root of elt in K
# - for d=2, this uses the built-in sqrt()
# - 
def dth_root(elt, d=2):
    if d == 2:
        r0 = elt.sqrt(); # According to Sagemaths doc, this is factoring y^2-elt using PARI.
    else:
        # print("/!\\ d != 2 could take some time");
        Ky = PolynomialRing(elt.parent(), name='y'); y = Ky.gen();
        r  = (y**d-elt).roots(); assert(len(r) == 1 and r[0][1] == 1);
        r0 = r[0][0];
    
    return r0;


def saturate_S_units(V, units=[], S=[], d=2, exp_rank=0):
    assert(len(V) > 0);
    assert(len(units) == 0 or V[0].parent() == units[0].parent());
    assert(len(S) == 0 or V[0].parent() == S[0].number_field());
    # assert(is_prime(d)); # Will see if it works
    if d!=2:
        print("/!\\ d={} not functional yet.".format(d));
    K     = V[0].parent(); # Assume it is cyclotomic !
    m     = cf_get_conductor(K);
    n     = K.degree();

    print(("t2-norms: "+"{:.2f} "*len(V)).format( *[float(t2_norm(_r)) for _r in V ]));
    t = cputime();
    V = logU_reduce(V, units);
    t = cputime(t);
    print ("Reduced log_V = matrix(GF(2), [[log_chip_quadratic(_pid, _v) for _pid in l_pid] for _v in ext_V]);{:.2f}s".format(t));
    print(("t2-norms: "+"{:.2f} "*len(V)).format( *[float(t2_norm(_r)) for _r in V ]));



    
    # 0. Complete V by units and possibly torsion (whenever gcd(#mu, d) > 1)
    mu_K  = cf_get_torsion_units(K,m);
    ext_V = (mu_K if gcd(mu_K[0].multiplicative_order(),d)>1 else []) + units + V;
    #    ... and find sufficiently many characters to sort out d-th powers
    t = cputime(); l_pid = sat_get_suitable_chip(ext_V, d=d); t = cputime(t);
    print("Finding suitable prime ideals [Time:{:.2f}s]".format(t));

    # 1. Find product of powers in V that are d-th powers
    t = cputime();
    if d == 2:
        log_V = matrix(GF(2), [[log_chip_quadratic(_pid, _v) for _pid in l_pid] for _v in ext_V]);
    else:
        log_V = matrix(GF(d), [[log_chip_dth_power(_pid, _v, d=d) for _pid in l_pid] for _v in ext_V]);
    t = cputime(t);
    print("Computing V -> V/V^{} [Time:{:.2f}s]".format(d,t));
    t = cputime(); ker_log_V = log_V.left_kernel().basis_matrix(); t = cputime(t);
    print("Left kernel of rank {} {} [Time:{:.2f}s]".format(ker_log_V.nrows(), "" if exp_rank==0 else "(exp:"+str(exp_rank)+")", t));

    # 2. Compute d-th powers /-> d-th roots
    t = cputime();
    Rd = [ prod(ext_V[j]**ker_log_V[i][j] for j in range(len(ext_V))) for i in range(ker_log_V.nrows()) ];
    t = cputime(t);
    print("Build {}-th powers [Time:{:.2f}s]".format(d,t));
    ### Reduce ?? /!\ keep the mod d

    t = cputime();
    R = [ dth_root(_rd, d=d) for _rd in Rd ];
    # r = [ _r2.sqrt() for _r2 in r2 ];
    t = cputime(t);
    print("Computing square roots [Time:{:.2f}s]".format(t));
    print(("t2-norms: "+"{:.2f} "*len(R)).format( *[float(t2_norm(_r)) for _r in R ]));

    ### Reduce ??
    t = cputime();
    R = logU_reduce(R, units);
    t = cputime(t);
    print ("Reduced {:.2f}s".format(t));
    print(("t2-norms: "+"{:.2f} "*len(R)).format( *[float(t2_norm(_r)) for _r in R ]));
    
    # 3. Extract (HNF) a generating set of V+r
    RV = R+V; # In that specific order (otherwise we have HUGE hnf coefficients)
    t = cputime(); RV_val = matrix( [[ _su.valuation(_pid) for _pid in S] for _su in RV ] ); t = cputime(t);
    print("p-Adic Valuations on R+V [Time:{:.2f}s]".format(t));
    t = cputime(); H, hu  = RV_val.hermite_form(transformation=True, include_zero_rows=False); t = cputime(t);
    print("Hermite Normal Form [Time:{:.2f}s]".format(t));
    print("Target volume: {}".format((hu*RV_val).determinant()));

    ### Reduce ??
    
    
    # 4. Compute W
    t = cputime();
    W = [ prod(RV[j]**hu[i][j] for j in range(len(RV))) for i in range(hu.nrows()) ];
    t = cputime(t);
    print("Build {}-saturated generating set [Time:{:.2f}s]".format(d,t));
    print(("t2-norms: "+"{:.2f} "*len(W)).format( *[float(t2_norm(_r)) for _r in W ]));
    t = cputime();
    W = logU_reduce(W, units);
    t = cputime(t);
    print ("Reduced {:.2f}s".format(t));
    print(("t2-norms: "+"{:.2f} "*len(W)).format( *[float(t2_norm(_r)) for _r in W ]));

    ### Reduce
    W_val = matrix( [[ _su.valuation(_pid) for _pid in S] for _su in W ] );
    print("Vol(W): {}".format(W_val.determinant()));
    ### Valuations mod p ???
    ### 

    
    return W;



# logU = matrix([ log_embedding(_u, p_inf=phi, b_prec=500) for _u in U ])
# pH = get_projection_H(logU.ncols(), logU.ncols(), b_prec=500)
# GlogU, _ = gram_schmidt_ortho(logU, normalize=False)
# Gred = [ twphs_build_solution(_v, cvp_babai_NP(logU, log_embedding(_v, p_inf=phi, fb=fb, b_prec=500)*pH, G=GlogU), logU, U) for _v in G ] # > 54s
# time Gtst = twphs_build_solution_batch(G, matrix([ cvp_babai_NP(logU, log_embedding(_v, p_inf=phi, b_prec=500)*pH, G=GlogU) for _v in G ]), logU, U) # < 12s




# # Looong, reduce modulo previous ones, but result is the same as above
# sage: GT = [];
# ....: _logS = matrix([log_embedding(_u, p_inf=phi, fb=fb, b_prec=500) for _u in U ]);
# ....: for _v in G:
# ....:     _lg = log_embedding(_v, p_inf=phi, fb=fb, b_prec=500);
# ....:     _pH = get_projection_H(_logS.ncols(), _logS.ncols(), b_prec=500);
# ....:     _g = twphs_build_solution(_v, cvp_babai_NP(_logS, _lg*_pH), _logS, U+GT);
# ....:     GT.append(_g);
# ....:     _logS = matrix(_logS.rows() + [log_embedding(_g, p_inf=phi, fb=fb, b_prec=500)]);
