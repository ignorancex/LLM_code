# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

import collections

from sage.all import *
import fp
from pideal import *


# Hopefully, always use the (e)GRH hypothesis and useful unproven conjectures for number fields
# We will still try to specify it each time it is needed, for debug and checks
proof.number_field(False);


# -------------------------------------------------------------------------------------------
# Infinite places

# There seemed to exist a precision drift somewhere (?).
# At the time, doubling precision on infinite places corrected the phenomenon.
NF_INFP_BPREC_SCALE = 2.0;


# Returns the number of infinite places of K (r1+r2)
def get_nb_inf_places(K):
    return sum(K.signature());


# Returns infinite places of suitable precision to reach b_prec. [In practice, 2*b_prec]
# Rk: At some point, there was some precision issues with complex embeddings (?).
def get_inf_places(K, b_prec=fp.BIT_PREC_DEFAULT):
    # NB: K.complex_embeddings() is much *MUCH* faster; the ordering is as follows: (real to verify)
    #         complex embeddings: _[0] == _[-1].conjugate(), etc
    #                             decreasing real part, positive imaginary part from _[0] to _[n/2-1]
    #         real embeddings     increasing real part
    # For K.places(), increasing real part, positive imaginary
    # --> all (log( abs(places[i](K.gen())-cembed[n/2-i-1](K.gen())),2) < -prec for i in range(n/2))

    # Attempt...
    # Problem is, complex_embeddings() are not ordered the same way for cyclotomic and generic fields.
    _r1, _r2 = K.signature(); 
    _p_inf   = K.complex_embeddings(prec=NF_INFP_BPREC_SCALE*b_prec);
    assert(len(_p_inf) == K.degree());
    _p_inf   = _p_inf[2*_r2:] + _p_inf[_r2-1::-1];
    assert(len(_p_inf) == _r1+_r2);

    # _p_inf = K.places(prec=NF_INFP_BPREC_SCALE*b_prec); 
    return _p_inf;


# Extends / Reduces the precision of infinite places
# This is done by recomputing new places, and reordering according to input.
def adjust_inf_places(K, p_inf, to_prec=fp.BIT_PREC_DEFAULT):
    assert(len(p_inf) == get_nb_inf_places(K));
    # Compute new places (almost free)
    to_p_inf  = get_inf_places(K, b_prec=to_prec); # K.complex_embeddings(prec=to_prec*NF_INFP_BPREC_SCALE); 
    # Reorder to_p_inf to match p_inf
    from_prec = p_inf[0].codomain().precision();
    ctrl_prec = min(from_prec, to_prec*NF_INFP_BPREC_SCALE); # Control precision
    new_p_inf = sum( ([_phi for _phi in to_p_inf
                       if fp.fp_is_zero(_phi(K.gen())-_old(K.gen()), target=ctrl_prec)]
                      for _old in p_inf), []); # //!\ Algo in "n^2"
    # Sanity check (incomplete, we could have several/no matches above in bad cases ?)
    assert(len(new_p_inf) == len(p_inf));
    assert(fp.fp_assert_equal("phi++==phi", [_phi(K.gen()) for _phi in new_p_inf], [_phi(K.gen()) for _phi in p_inf],
                              target=ctrl_prec));
    return new_p_inf;


# Extends the precision of infinite places
def extend_inf_places(K, p_inf, to_prec=fp.BIT_PREC_DEFAULT):
    if (to_prec*NF_INFP_BPREC_SCALE <= p_inf[0].codomain().precision()):
        return p_inf;
    return adjust_inf_places(K, p_inf, to_prec=to_prec);

# Reduces the precision of infinite places
def narrow_inf_places(K, p_inf, to_prec=fp.BIT_PREC_DEFAULT):
    if (to_prec*NF_INFP_BPREC_SCALE >= p_inf[0].codomain().precision()):
        return p_inf;
    return adjust_inf_places(K, p_inf, to_prec=to_prec);


# lp-norm (p=1,2,...,infinity) in K \otimes RR
# [Todo] lp_norm with precomputed p_inf ?
def lp_norm(alpha, p=2, b_prec=fp.BIT_PREC_DEFAULT):
    return vector(ComplexField(b_prec), alpha.complex_embeddings(prec=NF_INFP_BPREC_SCALE*b_prec)).norm(p);

# NB: Unlike Magma, we don't return the square of T2-norm, though it would be an integer (or rational)
def t2_norm(alpha, b_prec=fp.BIT_PREC_DEFAULT):
    return lp_norm(alpha, p=2, b_prec=b_prec);



# -------------------------------------------------------------------------------------------
# Handling Minkowski's embedding vs infinite places embeddings

# For phi : cplx -> Mink_K, where sig(K)=(r1,r2)
# and v in Mink_K
# returns P st v*P = phi^(-1)(v).
def minkowski_to_complex_matrix(r1, r2, b_prec=fp.BIT_PREC_DEFAULT):
    _CC       = ComplexField(prec=b_prec);
    _i        = _CC.gen();
    _bck_cplx = 1/sqrt(_CC(2))*matrix(_CC, [[1, 1], [_i, -_i]]);
    _P        = block_diagonal_matrix(identity_matrix(_CC, r1), *[_bck_cplx for _k in range(r2)],
                                      subdivide=False);

    assert (_P.base_ring().precision() >= b_prec);
    return _P;


def minkowski_to_complex(v, r1, r2):
    _P  = minkowski_to_complex_matrix(r1, r2, b_prec=v.base_ring().precision());
    return v*_P;



# ------------------------------------------------------------------------------------------
# Units and S-units (generic case)

# Beyond this, factorback functions are going to be too costly
# This has been designed for cyclotomics, but of course it depends on the (root) discriminant.
__FACTORBACK_MAX_DEGREE = 40;


# Obtain the rank of the (fundamental) unit group
def get_rank_units(K):
    return sum(K.signature())-1;


# S-units computation provide also fundamental units. We squeeze out the torsion part and separate
# the fundamental part (of norm 1) from the S-part.
# O is an order of some number field, fb is a list of prime ideals of O
def get_S_units(K, fb=[]):
    # In some versions of Sage, S_units() is taking forever,
    # while S_unit_group() then gens_values() works fine. Hence the workaround.
    _sU_gp  = K.S_unit_group(S=fb, proof=False); 
    _sU     = _sU_gp.gens_values();

    # Since they are computed anyway, the second return value are fundamental units.
    _s_un   = [ _su for _su in _sU if (_su.norm().abs() != 1) ];
    _un     = [ _su for _su in _sU if (_su.norm().abs() == 1) and (_su.multiplicative_order() == +Infinity) ];
    assert ((len(_un) == get_rank_units(K)) and (len(_s_un) == len(fb)));
    
    return _sU_gp, (_un, _s_un);


# ------------------------------------------------------
# S-units in compact forms. 1. if fb in fb_Cl, 2. Generic S, 3. Wrapper assigning the previous two.

# This is specifically designed for factor bases that are *included* in the one used for Buchmann
# In other words, we must have map(pari_prime, fb) in K_bnf[4].
def __get_ClS_units_raw(K, K_bnf, fb_pari=[], verbose=0):
    # ----------------------------------------------------------
    # Get factor base used to compute ClK. Might be huge.
    fb_Cl = list(K_bnf[4]);
    if verbose:     print("fb:\t{} elts\nfb_Cl:\t{} elts".format(len(fb_pari),len(fb_Cl)), flush=True);
    if verbose > 1: print("norms=\n\t{}".format([K_bnf.idealnorm(_pid) for _pid in fb_Cl]), flush=True);

    # Extract the indices f(i) such that fb[i] == fb_Cl[f(i)]
    t = cputime();
    fb_idx  = [ fb_Cl.index(_pid)               for _pid in fb_pari ];  # allows fb_pari[i] == fb_Cl[fb_idx[i]]
    nfb_idx = [ _i  for _i in range(len(fb_Cl)) if  _i not in fb_idx ];
    t = cputime(t);
    assert (len(fb_Cl) == len(fb_pari) + len(nfb_idx)), \
        "\x1b[31m[Err]\x1b[0m '__get_ClS_units_raw()' only works if fb subset fb_Cl";
    if verbose: print("Extract fb_Cl \\ fb:\tt={:.2f}".format(t), flush=True);

    # ----------------------------------------------------------
    # All rels / gens 
    # Using notations of [Coh93, §6.5.5]:
    #     H = HNF(rels) = block_matrix( [[W, 0], [B, identity]], with W = transpose(bnf[0]), B = transpose(bnf[1])
    W = matrix(K_bnf[0]).transpose();
    B = matrix(K_bnf[1]).transpose();

    # Transition matrices and the compact form basis are in bnf[7][2]
    c_form = K_bnf[7][2]; 
    # B_su = c_form[0], yu = transpose(c_form[1]), ysu = transpose(c_form[2])
    B_su_pari = c_form[0]; 
    y_u  = matrix(c_form[1]).transpose();
    y_su = matrix(c_form[2]).transpose();

    # ----------------------------------------------------------
    # We need relations for fb = fb_Cl[fb_idx] the relation matrix is now:
    #     let H'   = H[:, fb_idx + nfb_idx]     <--- permute columns
    #         N, U = HNF( H' )
    #         u    = U[:#S,:]                   <--- keep only first square block,
    #                                                actually this describes ker(H[:,not in fb_idx])
    # The new transition matrix is P = (u * ysu)
    t = cputime();
    H  = block_matrix( [[W, Integer(Integer(0))], [B, Integer(Integer(1))]], subdivide=False);
    # Permute columns so that the ones corresponding to fb come first
    Hp = block_matrix( [[H[:,fb_idx], H[:,nfb_idx]]], subdivide=False );
    # Compute new HNF
    # [Todo] Actually in most cases this should just be a permutation of rows.
    _, U = row_lower_hermite_form(Hp, transformation=True);
    # [Test] HNF pari4 ---> Seems much slower, does not improve norm of y_su.
    # tt = cputime(); Hp_pari = pari(Hp.transpose()); tt=cputime(tt); print("to pari {:.2f}".format(tt));
    # tt = cputime(); _, U = Hp_pari.mathnf(flag=4);  tt = cputime(tt); print("mathnf {:.2f}".format(tt));
    # tt = cputime(); U = matrix(U).transpose(); tt=cputime(tt); print("to Sage {:.2f}".format(tt));
    # //--- [END Test]
    N = U[:len(fb_pari),:]; # Extract kernel over pid's not in fb_pari
    assert (N.nrows() == len(fb_pari) and N.ncols() == y_su.nrows());
    t = cputime(t);
    if verbose: print("Computing kernel:\tt={:.2f}".format(t), flush=True);
    
    # ----------------------------------------------------------
    # Cleaning relations after kernel
    # We can clean: nz_idx = [ i for i in range(N.ncols()) if N[:,i] != 0 ] <-- Can be much smaller than N.ncols() ?
    #     P = ([u]) * N[:,nz_idx] * ysu[nz_idx,:]
    t = cputime();
    nz_idx = [ _i for _i in range(N.ncols()) if N[:,_i] != 0 ];
    # Adapt y_su
    y_su = N[:,nz_idx]*y_su[nz_idx,:];
    t = cputime(t);
    if verbose: print("Clean relations:\tt={:.2f}".format(t), flush=True);
    
    # ----------------------------------------------------------
    # Cleaning Compact form Basis
    ns_idx = [ _j for _j in range(len(B_su_pari)) if y_u[:,_j] != 0 or y_su[:,_j] != 0 ];
    if verbose:
        print("Remove {}/{} useless fb_Cl-units".format(len(B_su_pari)-len(ns_idx),len(B_su_pari)), flush=True);
    B_su_pari = [ B_su_pari[_j] for _j in ns_idx ];
    y_u    = y_u [:,ns_idx];
    y_su   = y_su[:,ns_idx];
    
    # ----------------------------------------------------------
    # p-adic Valuations are missing: actually we need them to obtain fundamental S-units (if bnf.gen not in S)
    t = cputime(); B_vp = [[ZZ(K_bnf.nfeltval(_su,_pid)) for _pid in fb_pari ] for _su in B_su_pari]; t = cputime(t);
    if verbose: print("Valuations of gi's:\tt={:.2f} [{:.2f}/gi]".format(t,t/len(B_su_pari)), flush=True);
    # Rq: if V = matrix([[ _s.valuation(_pid) for _pid in fb_Cl] for _s in B_su]),
    #     then H = ysu * V

    # ----------------------------------------------------------
    # Converting into Sage format
    B_su = [ K(_su) for _su in B_su_pari ];
    y_u  = y_u.rows();
    y_su = y_su.rows();
    
    return (y_u, y_su), B_su, B_vp;
    

# Works for any fb, but can be very costly even if fb \subset fb_ClK (the one used to compute Cl_K)
def __get_anyS_units_raw(K, K_bnf, fb_pari=[], verbose=0):
    # ------------------------------------------------------
    # S-units computation
    t = cputime(); su = K_bnf.bnfunits(fb_pari); t = cputime(t); # Don't use bnfsunit() that expands S-units
    if verbose: print("S-units (#fb:{}):\tt={:.2f}".format(len(fb_pari),t), flush=True);
    
    # Non trivial S-units are given first, then units, then torsion
    # Each S-unit is given as [0]: list of g_i, [1]: list of e_i st. su = prod g_i^e_i
    su = list(su[0][:-1]); # Remove torsion (last)
    su = su[len(fb_pari):] + su[:len(fb_pari)]; # Put units first, keep S-units with valuations in row/lower-triangular HNF
    # There are a lot (!) of duplicates in the g_i so we parse the structure once to get a dictionary
    # memorizing also all (r,e_i) where e_i = valuation(g_i, su[r]).
    t = cputime();
    B_su_dic = collections.defaultdict(lambda: collections.defaultdict(int)); # int defaults to 0, could also use Counter()
    for _r, (_g, _e) in enumerate(su):
        # _g is a list of gi's, _e is a list of exponents
        for _gi, _ei in zip(_g,_e):
            _gi_K  = K(_gi);
            if (_gi_K != 1):
                B_su_dic[ _gi_K ][_r] += ZZ(_ei);
    t = cputime(t);
    if verbose: print("Dict from Su:\t\tt={:.2f}".format(t), flush=True);
    
    # ------------------------------------------------------
    # We need to recondition the g_i for later use:
    # 1. use inv(g_i) which is often smaller: criterion used for now is size of denominator.
    # 2. clear denominators by including their factors in dictionary
    t   = cputime();
    lgi = list(B_su_dic.keys()); # Copy keys: did not see how not to copy these data
    for _gi in lgi:
        _gi_inv = 1/_gi;
        # If denom(g) > denom(1/g): use opposite exponents
        # Either case: replace key by key*denom(key) and include factors of denom() to dict
        _di     = _gi.denominator();
        _di_inv = _gi_inv.denominator();

        # [Todo] Change criterion to use the linf norm of coefficients ? 
        #        As we want small polynomials, the coefficients criterion is good.
        #        Oh, but this takes significant time, and using dens is already quite good already.
        if verbose > 1:
            print("\t\x1b[{}mgi: 2^{:<12.2f}\x1b[0m\tinv_gi: 2^{:<12.2f}".format
                  ("31" if _di_inv < _di else "32",
                   max([ RealField()(log(_c.abs(),2,prec=10000)) for _c in (_gi*_di).polynomial()]),
                   max([ RealField()(log(_c.abs(),2,prec=10000)) for _c in (_gi_inv*_di_inv).polynomial()])),
                  flush=True);
        if (_di_inv < _di):
            _g = _gi_inv;
            _d = _di_inv;
            for _r,_e in B_su_dic[_gi].items():
                B_su_dic[_gi][_r] = -_e;
        else:
            _g = _gi;
            _d = _di;
        _g = _g*_d;
        # Update key
        B_su_dic[_g] = B_su_dic.pop(_gi);
        # Deal with denominator if non-trivial
        _d_fact      = _d.factor(limit=2**20, proof=False); # Arbitrary limit 2^20, should be easy enough
        if (len(_d_fact) != 0): # _d != \pm 1
            for _df,_de in _d_fact:
                for _r,_e in B_su_dic[_g].items():
                    B_su_dic[K(_df)][_r] -= _de*_e; # //!\ Do not forget the inital valuations _e
    # //-- End loop on gi's.
    t = cputime(t);
    if verbose: print("Retreating gi's:\tt={:.2f}".format(t), flush=True);
    B_su = list(B_su_dic.keys());
    
    # ------------------------------------------------------
    # Now, obtain vectors y_u, y_su and B_su_vp
    t = cputime(); B_vp = [ [ZZ(K_bnf.nfeltval(_su,_pid)) for _pid in fb_pari ] for _su in B_su ]; t = cputime(t);
    if verbose: print("Valuations of gi's:\tt={:.2f} [{:.2f}/gi]".format(t,t/len(B_su)), flush=True);
    
    # ------------------------------------------------------
    # y_u, y_su: Y[r][k] = e iff B_su_dic.keys()[k][r] = e
    Y = [[0]*len(B_su) for _ in range(len(su))];
    for _k,_g in enumerate(B_su):
        for _r,_e in B_su_dic[_g].items():
            Y[_r][_k] = _e;
    y_u  = Y[:-len(fb_pari)]; # Note we have reversed the output of Pari.
    y_su = Y[-len(fb_pari):];
    
    # Control the conversion in small dimensions
    if (K.degree() < __FACTORBACK_MAX_DEGREE):
        t = cputime();
        assert( all( K(K_bnf.nffactorback(su[_r])) == prod( _s**_e[_r] for _s,   _e  in B_su_dic.items() )
                     for _r in range(len(su))));
        assert( all( K_bnf.nffactorback(su[_r]) == K_bnf.nffactorback(B_su, _y)
                     for _r,_y in enumerate(y_u+y_su) ));
        t = cputime(t);
        if verbose: print("Assert Pari->Sage:\tt={:.2f}".format(t), flush=True);
    
    return (y_u, y_su), B_su, B_vp;


# Generic function assigning previous ones. Call:
#     __get_ClS_units_raw()     if S is in the factor base fb_Cl used in pari(K).bnfinit()
#     __get_S_units_raw()       for general S, which calls K_bnf.bnfunits() and can be very costly.
# Rq: if K_bnf is given, it SHALL be K_bnf == pari(K).bnf_init(flag=1) <--- using flag=1
def get_S_units_raw(K, fb=[], K_bnf=None, fb_pari=None, verbose=0):
    if verbose: print("S-units using Pari\n"+"-"*18, flush=True);
    if len(fb) > 0: assert (fb[0].number_field() == K), "Ideals of fb are not in K";
    
    # ----------------------------------------------------------
    # Obtain K_bnf and fb_pari if they are not provided
    if (K_bnf == None):
        # flag=1 is crucial to avoid costly expansion computations
        t = cputime(); K_bnf = pari(K).bnfinit(flag=1); t = cputime(t); 
        if verbose: print("bnfinit(1):\t\tt={:.2f}".format(t), flush=True);
    else: # Check they match (basic: check definition polynomials)
        assert (list(K_bnf.getattr("nf").getattr("pol")) == list(K.defining_polynomial())
                and len(K_bnf[7][2]) != 1), \
            "\x1b[31m[Err]\x1b[0m K and K_bnf do not match, or flag was not set to 1 in 'bnfinit()'.";
    if (fb_pari == None):
        t = cputime(); fb_pari = [ _pid.pari_prime() for _pid in fb ]; t = cputime(t);
        if verbose:
            print("Convert fb to primedec:\tt={:.2f} [{:.2f}/pid]".format(t,t/len(fb) if fb!=[] else 0), flush=True);
    else: # Check they match
        assert(True);
        assert (len(fb_pari) == len(fb)
                and all(idealnorm(K_bnf, _pdec) == pid_fast_norm(_pid) for (_pdec, _pid) in zip(fb_pari, fb))), \
            "\x1b[31m[Err]\x1b[0m fb and fb_pari do not match.";
    
    # ----------------------------------------------------------
    # Determining whether fb in fb_Cl or not ---> Two S-unit computation strategies
    fb_Cl    = K_bnf[4];
    in_fb_Cl = all( _pid in fb_Cl for _pid in fb_pari );
    t = cputime();
    if (in_fb_Cl == True):
        print("\x1b[32m[Ok]  \x1b[0m fb in fb_Cl: Calling ClS-units", flush=True);
        (y_u, y_su), B_su, B_vp = __get_ClS_units_raw(K, K_bnf, fb_pari=fb_pari, verbose=verbose);
    else:
        print("\x1b[31m[Warn]\x1b[0m fb not in fb_Cl: Calling generic 'bnfunits()', can be very long.", flush=True);
        (y_u, y_su), B_su, B_vp = __get_anyS_units_raw(K, K_bnf, fb_pari=fb_pari, verbose=verbose);
    t = cputime(t);
    if verbose: print("\x1b[35m[Total]\x1b[0m S-units:\tt={:.2f}".format(t), flush=True);
    
    # ----------------------------------------------------------
    # [END] Control in small dimensions
    if (K.degree() < __FACTORBACK_MAX_DEGREE):
        t = cputime();
        # Check prime valuations of units are all 0
        assert(matrix(y_u)*matrix(B_vp) == 0);
        # Check norm(units) == 1;
        norm_Bsu = [ RealField(fp.BIT_PREC_DEFAULT)(_raw_su.norm()) for _raw_su in B_su ];
        assert(fp.fp_assert_zero("N(u)=1", [sum(_y*log(_n_su) for _y,_n_su in zip(_yu, norm_Bsu)) for _yu in y_u],
                                 target=fp.BIT_PREC_DEFAULT, sloppy=True));
        # Verify fb^rel[] == <elt[]>: do it in Pari so that it is still reasonable
        rels      = matrix(y_su)*matrix(B_vp);
        assert(all([ K_bnf.idealhnf(K_bnf.nffactorback(B_su, _exp))
                     == K_bnf.idealhnf(K_bnf.idealfactorback(fb_pari, _rel)) for (_exp, _rel) in zip(y_su, rels)]));
        t = cputime(t);
        if verbose: print("Final checks:\t\tt={:.2f}".format(t));
    
    return (y_u, y_su), B_su, B_vp;



# -------------------------------------------------------------------------------------------
# Class Number Formula
# /!\ This code apparently only works with cyclotomic fields.
#     Sthg wrong is happening for other fields.
# /!\ Precision issues: this works as is, but experience driven...
#     Nevertheless, it seems sensible that zeta_K should be computed at precision following ln |DK|
#     and that the computation (s-1)*zeta(s) is close enough to the pole for s=1+1/sqrt(|DK|)
def analytic_hR(K):
    dK     = K.discriminant().abs();
    r1, r2 = K.signature();
    # Take precision = ln(dK), distance to 1 eps = 1/sqrt(dK)
    b_prec = ceil(RealField()(log(dK, 2)));# + RealField()(log(exp(150), 2)));
    Re     = RealField(b_prec); # print("zeta:{} {} {}".format(b_prec, b_prec/2, b_prec*0.5));
    zs     = K.zeta_function(prec=b_prec);
    s      = Re(1+2**Re(-0.5*b_prec));# Re(1+exp(-b_prec*0.35));
    res_s  = Re(zs(s)); 
    # Apply Class Number Formula
    hr     = (s-1)*res_s*K.number_of_roots_of_unity()*sqrt(Re(dK))/2**Re(r1)/(2*Re(pi))**Re(r2);

    return hr;



# -------------------------------------------------------------------------------------------
# Number field elements representation.

# Log-Arg representation
# --------------------------------------------------
# Log-Arg representation: keep ln |s_i(.)| for each infinite place s_i.
#                              arg s_i(.)  same, as elements in [-1/2,1/2] (implicitly mult. by 2*Pi).
#                         (?)  v_p (.)     for p in some FB.

# Log-Arg functions take, whenever needed, the list of infinite/finite (FB) places to consider.
# Going to LogArg is easy, pulling it back is done via polynomial interpolation (to be used wisely).

logarg = collections.namedtuple('logarg',
                                ['inf',  # ln | s_i() |
                                 'args', # arg[ s_i() ] / Pi \in [-1,1]
                                 'vp',
                                 ]);

# K <----> LogArg
# ---------------
def logarg_set(elt, p_inf, fb=[], vp=[]): #, to_prec=0):
    K = elt.parent(); assert(len(p_inf) == get_nb_inf_places(K));
    b_prec = p_inf[0].codomain().precision(); # if to_prec == 0 else to_prec;
    Re     = RealField(b_prec);
    
    inf_embeddings = [ _phi(elt) for _phi in p_inf ];
    la_inf  = vector(Re, [ _si_elt.abs().log()      for _si_elt in inf_embeddings]);
    la_args = vector(Re, [ _si_elt.arg() / Re(2*pi) for _si_elt in inf_embeddings]);
    # If vp's are given, use them (independently of fb).
    # If factor base is empty as well as vp's, vp field stays empty.
    la_vp   = vector(ZZ, [ elt.valuation(_pid) for _pid in fb ]) if len(vp) == 0 else vector(ZZ, vp);
    la_elt  = logarg(inf  = la_inf, args = la_args, vp = la_vp);
    
    return la_elt;


# compute t2 norm of alpha defined by logarg la
# WARN: Works **only** with cyclotomic fields at this point
def logarg_t2_norm_cf(la):
    Re     = la.inf[0].parent();
    t2_sq  = Re(2) * sum( exp(Re(2)*_la_inf) for _la_inf in la.inf );
    t2_n   = sqrt(t2_sq);
    return t2_n;


# return ln N(b) from logarg of <a> = b . prod_{p in FB} p^vp
# WARN: Works **only** with cyclotomic fields at this point
#       Suppose b is coprime with FB.
def logarg_lnSnorm_cf(la, fb=[]):
    assert (len(fb) == len(la.vp));
    Re     = la.inf[0].parent();
    b_prec = Re.precision()/4;
    ln_Nfb = [Re(pid_fast_norm(_pid)).log() for _pid in fb];
    ln_Na  = Re(2)*sum(_la for _la in la.inf) - sum(_vp*_ln_Nfb for _vp, _ln_Nfb in zip(la.vp,ln_Nfb));
    # Na     = exp(ln_Na);
    # assert(fp.fp_assert_equal("exp(ln Nb) in ZZ", [Na], [round(Na)], target=b_prec, sloppy=True));
    return ln_Na;


# /!\ WARN: lifts only in the equation order ZZ[a] (K = Q(a))
def logarg_lift_eqn_order(la, p_inf):
    # Number Field
    K      = p_inf[0].domain(); assert(len(p_inf) == get_nb_inf_places(K));
    r1, r2 = K.signature();
    z      = K.gen();
    # Complex domain: asserting needed precision
    w_prec = p_inf[0].codomain().precision();
    Ce     = ComplexField(w_prec);
    Ce_x   = PolynomialRing(Ce, name='C');
    # 1.443 ~ 1/ln(2), so bit prec is height of la*1.443 + size of the discriminant
    # This should be very safe.
    assert(ceil(1.443*max(map(abs,la.inf)) + RR(K.discriminant().abs().log(2)))  < w_prec), "Precision of infinite places ({}) not sufficient (expected:{})".format(w_prec, ceil(1.443*max(map(abs,la.inf)) + RR(K.discriminant()).abs().log(2)));

    # Evaluation points: phis(z)
    lpts  = [ _phi(z) for _phi in p_inf ] + [ _phi(z).conjugate() for _phi in p_inf[r1:]];
    # Values of the polynomial, computed from the logarg representation
    lvals = (  [ exp(_a+Ce(2*pi*I)*_th) for _a, _th in zip(la.inf[:r1], la.args[:r1]) ]   # Real places
             + [ exp(_a+Ce(2*pi*I)*_th) for _a, _th in zip(la.inf[r1:], la.args[r1:]) ]   # Im   places
             + [ exp(_a-Ce(2*pi*I)*_th) for _a, _th in zip(la.inf[r1:], la.args[r1:]) ]); # Conjugates
    # Interpolation 
    gx   = Ce_x.lagrange_polynomial([(_pt,_v) for _pt, _v in zip(lpts,lvals)]);
    assert(fp.fp_assert_equal("gx(pts) == vals", map(gx, lpts), lvals,
                              target=w_prec, sloppy=True)); # Interpolation is quite stable, but still
    gx   = list(gx);
    
    # Sanity checks and map to ZZ[x]--> O_K
    assert(fp.fp_assert_zero("gx in RR[x]", [ _gx_C.imag_part() for _gx_C in gx],
                            target=w_prec, sloppy=True)); # Idem
    gx   = [ _gx_C.real_part() for _gx_C in gx ];
    gx_Z = [ _gx_R.round()     for _gx_R in gx ];
    assert(fp.fp_assert_equal("gx in ZZ[x]", gx, gx_Z,
                              target=w_prec, sloppy=True));
    # Little 'Sagerie': omitting padding only works for cyclotomic fields 
    g    = K(gx_Z + [0]*(K.degree()-len(gx_Z)));
    
    return g;



# Arithmetic in Log-Arg representation
# ------------------------------------
def logarg_reduce_args(c_args):
    # Apply mod [-Pi,Pi] / 2Pi : [n,n+1] --> round [-1/2,1/2]
    return vector(map(lambda _th: _th-_th.round(), c_args));


# Test equality at precision min(la1.prec, la2.prec)
# argmod: test arguments equality modulo (2pi / argmod)
def logarg_is_equal(la1, la2, sloppy=False, argmod=1):
    w_prec = min([_la.inf[0].parent().precision() for _la in [la1,la2]]);
    inf_eq = fp.fp_all_equal(la1.inf, la2.inf, target=w_prec, sloppy=sloppy); # "ln|s1|-ln|s2|"
    arg_eq = fp.fp_all_zero([min(_th.abs(), 1-_th.abs()) for _th in logarg_reduce_args(argmod*la1.args) - logarg_reduce_args(argmod*la2.args)], target=w_prec, sloppy=sloppy); # "theta_1-theta_2 mod 1"
    vp_eq  = (la1.vp == la2.vp);
    
    return (inf_eq and arg_eq and vp_eq);


def logarg_mul2(la1, la2):
    mul_inf  = la1.inf  + la2.inf;
    mul_args = logarg_reduce_args(la1.args + la2.args);
    mul_vp   = la1.vp   + la2.vp;
    
    la_mul   = logarg(inf=mul_inf, args=mul_args, vp=mul_vp);
    return la_mul;


# Naive accumulation.
# Use a "product(sum)" tree if there is too much precision drift, but tests seem very fine as is.
def logarg_mul(l_las):
    assert(len(l_las) > 0); # Too lazy to define "1", and what would be the size of the "vp" part ?
    
    la_mul = l_las[0];
    for _la in l_las[1:]:
        la_mul = logarg_mul2(la_mul, _la);

    return la_mul;


def logarg_pow(la, k):
    pow_inf  = k*la.inf;
    pow_args = logarg_reduce_args(k*la.args);
    pow_vp   = k*la.vp;
    la_pow   = logarg(inf=pow_inf, args=pow_args, vp=pow_vp);

    return la_pow;


def logarg_mpow(l_las, l_exps):
    assert(len(l_las) == len(l_exps));
    
    l_mpow  = [ logarg_pow(_la, _exp) for _la, _exp in zip(l_las, l_exps) ];
    la_mpow = logarg_mul(l_mpow);

    return la_mpow;


# -------------------------------------------------------------------------------------------
# Logarithmic embedding.
# When conjugates are present:
#     the order of infinite embeddings is as follows : 1...r_1,  ..., j, \conj(j), ...
#     the order of finite embeddings follows the same principle: -vp ln p, ..(\times f=[OK/P:Z/p]), -vq etc
# For finite/infinite parts, there are 3 options: TWISTED, EXPANDED and FLAT: (dimensions given if applied on both parts)
#     TWISTED:     [Ks:Qs] ln |a|_s                                      Dim: r_1+r_2  / k
#     FLAT:        ln |a|_s on infinite places, -vp(a) on finite places  Dim: r_1+r_2  / k
#     EXPANDED:    ln |a|_s, [Ks:Qs] times                               Dim: r_1+2r_2 / sum([OK/p:Z/p])
# For ideals, we add the following on infinite places:
#     NONE:        Put 0 everywhere.
# Note the "FLAT" version is designed to fit more or less PHS's choice on finite valuations;
# we would naturally expect instead ln |a|_s on all places (= vp ln(p) for finite places).
# /!\ p_inf[].codomain().precision() should be large enough to handle coefficients of elt.
NF_LOG_TYPE = ['TWISTED', 'FLAT', 'EXPANDED', 'NONE'];


# This returns the chosen log embedding from a logarg representation 
# We need the factor base norms for fb_type='TWISTED' or 'EXPANDED'. The simpler is probably to input the factor base directly.
# /!\ We NEED pid ideals <p,g(x)> in FB.
# 
# News: compute directly at the precision of p_inf.
def logarg_log_embedding(la, p_inf, fb=[], inf_type='TWISTED', fb_type='TWISTED'):
    K      = p_inf[0].domain();
    w_prec = p_inf[0].codomain().precision();
    Re     = RealField(w_prec);
    assert(len(la.inf) == len(p_inf) and len(p_inf) == get_nb_inf_places(K));
    assert(len(la.vp)  >= len(fb));
    assert((inf_type in NF_LOG_TYPE) and (fb_type in NF_LOG_TYPE));
    
    r1, r2 = K.signature();
    # Several remarks:
    # 1. abs_val return |s(eta)|^[Ks:R] on infinite places, N(p)^(-v_p(eta)) on finite places
    # 2. the log function has some weird precision issues (?)
    # 3. going through finite places with abs_val() takes too much time in Sage;
    #    computing it from scratch (valuation, norm) can be up to 3 times faster
    ln_inf_vals = list(la.inf);
    if (inf_type == 'FLAT'):
        ln_inf = ln_inf_vals;
    elif (inf_type == 'TWISTED'):
        ln_inf = ln_inf_vals[:r1] + [ 2*_ln_s for _ln_s in ln_inf_vals[r1:] ];
    elif (inf_type == 'EXPANDED'):
        ln_inf = ln_inf_vals[:r1] + sum((2*[_ln_s] for _ln_s in ln_inf_vals[r1:]), []);

    padic_vals = list(la.vp);
    if (fb_type == 'FLAT'): # PHS's version, but logically this would be -vp(eta) ln(p) (vs. -vp(eta) ln(norm(pid)) for TWISTED)
        ln_fb  = [ - _vp for _vp in padic_vals ];
    elif (fb_type == 'TWISTED'):
        ln_fb  = [ - _vp * pid_fast_norm(_pid).log(prec=w_prec) for _vp, _pid in zip(padic_vals, fb)];
    elif (fb_type == 'EXPANDED'):
        ln_fb  = sum((pid_fast_residue_class_degree(_pid) * [- _vp*pid_fast_smallest_integer(_pid).log(prec=w_prec)] for _vp, _pid in zip(padic_vals,fb)), []);
        
    ln_embedding = vector(ln_inf + ln_fb);
    assert (ln_embedding.base_ring().precision() >= w_prec); # //NF_INFP_BPREC_SCALE), "w_prec {} vs {}".format(w_prec,ln_embedding.base_ring().precision());
    return ln_embedding;


# //-- END OF FILE



# ?? Not always defined + touchy for v_p's (if applicable).
# def nfc_conjugate(a1):


# Compact representation
# --------------------------------------------------
# Basis of elements: g_1, ..., g_s  plus exponents e_1, ..., e_s st. elt = prod g_i^e_i

# ...... Seems this does not need too much extra code.

# To obtain the log embedding of a compact representation (gi, ai) --> prod gi^ai, do:
# - lg = [ logarg_set(gi, fb=..., vp=...) for gi in g ];
# - logarg_log_embedding(lgi) for lgi in lg ;
# - then the (g1^a1, ..., gn^an) have log embeddings: \sum ai lgi ie. vect(a).mat(lgi)



# -------------------------------------------------------------------------------------------
# [OBSOLETE] Historic code to erase
def log_embedding(elt, p_inf, fb=[], inf_type='TWISTED', fb_type='TWISTED', b_prec=fp.BIT_PREC_DEFAULT):
    assert(len(p_inf) > 0);
    _K          = p_inf[0].domain();
    _p_inf_prec = p_inf[0].codomain().precision();
    assert(len(p_inf) == get_nb_inf_places(_K));
    assert((inf_type in NF_LOG_TYPE) and (fb_type in NF_LOG_TYPE));
    assert(_p_inf_prec >= b_prec*NF_INFP_BPREC_SCALE), "Precision of infinite places ({}) not sufficient (expected:{})".format(_p_inf_prec, b_prec*NF_INFP_BPREC_SCALE);
    # Precision of p_inf should be able to deal with max coeffs of elt
    # This might not be sufficient: we need to account for ln disc(K).
    assert (ceil(max([ max(_c.numer().abs(), _c.denom()).log(2) for _c in list(elt) ])) < _p_inf_prec), "Precision of infinite places ({}) not sufficient (expected:{})".format(_p_inf_prec, ceil(max([ max(_c.numer().abs(), _c.denom()).log(2) for _c in list(elt) ])));

    _r1, _r2 = _K.signature();
    # Several remarks:
    # 1. abs_val return |s(eta)|^[Ks:R] on infinite places, N(p)^(-v_p(eta)) on finite places
    # 2. the log function has some weird precision issues (?)
    # 3. going through finite places with abs_val() takes too much time in Sage;
    #    computing it from scratch (valuation, norm) can be up to 3 times faster
    _ln_inf_vals = [ RealField(b_prec) (_phi(elt).abs().log()) for _phi in p_inf ];
    if (inf_type == 'FLAT'):
        _ln_inf = _ln_inf_vals; #[ RealField(b_prec)(_phi(elt).abs().log()) for _phi in p_inf ];
    elif (inf_type == 'TWISTED'):
        _ln_inf = _ln_inf_vals[:_r1] + [ 2*_ln_s for _ln_s in _ln_inf_vals[_r1:] ];
    elif (inf_type == 'EXPANDED'):
        _ln_inf = _ln_inf_vals[:_r1] + sum((2*[_ln_s] for _ln_s in _ln_inf_vals[_r1:]), []);
    elif (inf_type == 'NONE'): # /!\ WARN: what if r1+r2 ? ('FLAT' on other elements)
        _ln_inf = [ RealField(b_prec)(0) for _i in range(_K.degree())];
    
    _padic_vals = [ elt.valuation(_pid) for _pid in fb ]; # /!\ Very costly in Sage (computes OK)..
    if (fb_type == 'FLAT'): # PHS's version, but logically this would be -vp(eta) ln(p) (vs. -vp(eta) ln(norm(pid)) for TWISTED)
        _ln_fb  = [ - _vp for _vp in _padic_vals ];
    elif (fb_type == 'TWISTED'):
        _ln_fb  = [ - _vp * pid_fast_norm(_pid).log(prec=b_prec) for _vp, _pid in zip(_padic_vals, fb)];
        # _ln_fb = [ - elt.valuation(_pid) * _pid.norm().log(prec=b_prec) for _pid in fb ];
        # _ln_fb = [ _K.abs_val(_pid, elt, prec=b_prec).log() for _pid in fb ];
    elif (fb_type == 'EXPANDED'):
        _ln_fb  = sum((pid_fast_residue_class_degree(_pid) * [- _vp*pid_fast_smallest_integer(_pid).log(prec=b_prec)] for _vp, _pid in zip(_padic_vals,fb)), []);
        # _ln_fb = sum((_pid.residue_class_degree() * [-elt.valuation(_pid)*_pid.smallest_integer().log(prec=b_prec)] for _pid in fb), []);

    _log_embedding = vector(_ln_inf + _ln_fb);
    assert (_log_embedding.base_ring().precision() >= b_prec);
    return _log_embedding;

# [OBSOLETE] Code mapping twisted/etc
def logs_tw(elt, p_inf, fb=[], b_prec=fp.BIT_PREC_DEFAULT):
    return log_embedding(elt, p_inf, fb=fb, inf_type='TWISTED', fb_type='TWISTED', b_prec=b_prec);

def logs_val(elt, p_inf, fb=[], b_prec=fp.BIT_PREC_DEFAULT):
    return log_embedding(elt, p_inf, fb=fb, inf_type='FLAT', fb_type='FLAT', b_prec=b_prec);

def logs(elt, b_prec=fp.BIT_PREC_DEFAULT): # Mimics Magma "Logs()"
    _p_inf = get_inf_places(elt.parent(), b_prec=b_prec);
    return log_embedding(elt, _p_inf, fb=[], inf_type='FLAT', fb_type='FLAT', b_prec=b_prec);

# [OBSOLETE] Same for ideals (TWISTED version)
def logs_tw_ideal(a, fb=[], b_prec=fp.BIT_PREC_DEFAULT):
    p_inf = get_inf_places(a.number_field(), b_prec=b_prec);
    return log_embedding(a, p_inf, fb=fb, inf_type='NONE', fb_type='TWISTED', b_prec=b_prec);

# [OBSOLETE] Same for ideals (FLAT version)
def logs_val_ideal(a, fb=[], b_prec=fp.BIT_PREC_DEFAULT):
    p_inf = get_inf_places(a.number_field(), b_prec=b_prec);
    return log_embedding(a, p_inf, fb=fb, inf_type='NONE', fb_type='FLAT', b_prec=b_prec);

