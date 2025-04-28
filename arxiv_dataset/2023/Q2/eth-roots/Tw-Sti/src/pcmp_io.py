# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard, Andrea Lesavourey
# GPL-3.0-only (see LICENSE file)

from sage.all import *

import fp      # Floating point prec
import pideal  # fast_gens_two
import nf      # get_rank_units


# ------------------------------------------------------------------------------------------
# Input/Output (to files) functions
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# Lattices (aka real matrices)

# Format of the matrix is defined as follows:
# First line is: "# lattice: nrows={} ncols={} prec={}"
# A row is "[<c1>,<c2>,...]\n", where <ci> is in base 10, ie. [-]d(...)[.f(...)][Ee(...)]


# Return round(_c) if abs(_c - ZZ) <~ 2^-prec, _c otherwise.
def __out_compressed_coeff(_c, b_prec):
    if (b_prec > 0): # Ring is not exact
        _c = ZZ(round(_c)) if (fp.fp_is_integer(_c, target=b_prec, sloppy=False)) else _c;
    return _c;


def __lattice_row_out_stream(stream, row, b_prec):
    _row_str = ("["
                +",".join(map(lambda _c: __out_compressed_coeff(_c, b_prec).str(base=10).replace('e','E'), row))
                +"]\n");
    stream.write(_row_str);
    return;


def lattice_out_stream(stream, L):
    b_prec = 0 if L.base_ring().is_exact() else L.base_ring().precision();
    stream.write("# lattice: nrows={} ncols={} prec={}\n".format(L.nrows(), L.ncols(), b_prec));
    
    for _row in L:
        __lattice_row_out_stream(stream, _row, b_prec);
    stream.flush();

    return;


def __lattice_row_in_stream(stream):
    _row = eval(preparse(stream.readline().rstrip()));
    return _row;


def lattice_in_stream(stream, to_b_prec=0):
    # Control first line, extract nrows/ncols and prec
    _lat_head = stream.readline().rstrip(); assert (_lat_head.startswith("# lattice: nrows="));
    _lat_head = _lat_head[len("# lattice: "):];
    _kv       = dict(_s.split('=',1) for _s in _lat_head.split(' ')); assert (len(_kv) == 3);
    _nrows    = sage_eval(_kv.get("nrows"));
    _ncols    = sage_eval(_kv.get("ncols"));
    _in_prec  = sage_eval(_kv.get("prec"));
    # Obtain (a consistent) target prec
    _prec     = _in_prec if (to_b_prec == 0) else to_b_prec if (_in_prec == 0) else min(to_b_prec, _in_prec);

    # Read line by line:
    _L_ij = [];
    for _i in range(_nrows):
        _L_ij.append(__lattice_row_in_stream(stream));
        assert(len(_L_ij[-1]) == _ncols);

    _R = ZZ if (_prec == 0) else RealField(_prec);
    _L  = matrix(_R, _L_ij);
    return _L;
    


# ------------------------------------------------------------------------------------------
# Number field 

# Some labelling
#     Cyclotomic fields of conductor m: 'z<m>' (ex.: 'z23') [deg: 22]
#     NTRU Prime fields x^p-x-1:        'n<p>' (ex.: 'n23') [deg: 23]
#     (Default):                        'g<d>' (ex.: 'g23') [deg: 23]
def nf_get_tag(K):
    _Zx  = PolynomialRing(Integers(), names=('x',));
    (x,) = _Zx._first_ngens(1);

    if (K.defining_polynomial().is_cyclotomic() == True): # Cyclotomic
        _tag = "z{}".format(K.gen().multiplicative_order());
    elif (K.defining_polynomial() == (x**K.degree()-x-1)): # NTRU Prime
        _tag = "n{}".format(K.degree());
    else: # Generic
        _tag = "g{}".format(K.degree());

    return _tag;


def nf_set_tag(tag, eq=[]):
    _typ = tag[0];
    _val = sage_eval(tag[1:]);

    if (_typ == "z"):
        _K      = CyclotomicField(_val);
    elif (_typ == "n"):
        _Zx  = PolynomialRing(Integers(), names=('x',));
        (x,) = _Zx._first_ngens(1);
        _K   = NumberField(x**_val - x - 1, name='zp');
    else:
        _K = NumberField(PolynomialRing(Integers(), 'x')(eq), name='a');
    
    return _K;


def nf_out_stream(stream, K):
    stream.write("# number_field: tag={} eq={}\n".format(nf_get_tag(K), str(K.defining_polynomial().list()).replace(' ', '')));
    return;


def nf_in_stream(stream):
    # Control first line, extract r1+r2, precision
    _nf_head = stream.readline().rstrip(); assert (_nf_head.startswith("# number_field: tag="));
    _nf_head = _nf_head[len("# number_field: "):];
    _kv   = dict(_s.split('=',1) for _s in _nf_head.split(' ')); assert (len(_kv) == 2);
    _tag  = _kv.get("tag");
    _eq   = sage_eval(_kv.get("eq"));

    _K    = nf_set_tag(_tag, _eq);
    return _K;



# ------------------------------------------------------------------------------------------
# Infinite places

# Format of ( phi : K.gen() -> a) is (base 10):
#     a\n           for real places
#     Re(a) Im(a)\n for complex places
def __inf_place_out_stream(stream, K, phi):
    assert (phi.domain() == K);
    _z = phi(K.gen());

    if (_z.is_real() == True):
        stream.write("{}\n".format(_z.str(base=10)));
    else:
        stream.write("{} {}\n".format(_z.real_part().str(base=10), _z.imag_part().str(base=10)));

    return;


def __inf_place_in_stream(stream, K, to_prec=0):
    _z_reim  = [sage_eval(_s) for _s in stream.readline().rstrip().split(' ')];

    # Determine precision
    _in_prec = min(_z_reim_p.parent().precision() for _z_reim_p in _z_reim);
    assert (to_prec <= _in_prec);
    _prec   = _in_prec if (to_prec == 0) else to_prec;
    _RC     = RealField(_prec) if (len(_z_reim) == 1) else ComplexField(_prec);

    # Map input strings into RR or CC, verify it is indeed a root
    _z      = _RC(*_z_reim);
    assert (fp.fp_assert_zero("K.eq(z)", [K.gen().minpoly()(_z)], target=_RC.precision()));
    
    return K.hom(_z, codomain=_RC, check=False); # Rk: Same as code of K.places()


# Inf places complete file
# The first line must be: "# inf_places: nb=<n> prec=<bp>\n"
def inf_places_out_stream(stream, K, p_inf):
    stream.write("# inf_places: nb={} prec={}\n".format(len(p_inf), p_inf[0].codomain().precision()));

    for _phi in p_inf:
        __inf_place_out_stream(stream, K, _phi);
    stream.flush();

    return;


def inf_places_in_stream(stream, K):
    # Control first line, extract r1+r2, precision
    _inf_head = stream.readline().rstrip(); assert (_inf_head.startswith("# inf_places: nb="));
    _inf_head = _inf_head[len("# inf_places: "):];
    _kv   = dict(_s.split('=',1) for _s in _inf_head.split(' ')); assert (len(_kv) == 2);
    _nb   = sage_eval(_kv.get("nb")); assert (_nb == nf.get_nb_inf_places(K));
    _prec = sage_eval(_kv.get("prec"));

    # Read _nb lines
    _p_inf = [];
    for _i in range(_nb):
        _p_inf.append(__inf_place_in_stream(stream, K, to_prec=_prec));

    assert (len(_p_inf) == _nb);
    return _p_inf;



# ------------------------------------------------------------------------------------------
# Finite places (factor base)

# Format ideal <p, g> is, for x=K.gen(), g=a_0 + a_1 x + ... + a_d x^d:
#     p [a_0,a_1,...,a_d]\n (no padding with trailing zeroes from d to deg(K))
def __pid_out_stream(stream, K, pid):
    _g1, _g2 = pideal.pid_fast_gens_two(pid);
    stream.write("{} {}\n".format(_g1, str(_g2.list()).replace(' ', '')));
    
    return;


def __pid_in_stream(stream, K):
    _g1, _g2 = [sage_eval(_s) for _s in stream.readline().rstrip().split(' ')];
    # Little 'Sagerie', it works for cyclotomic fields, but generic fields need padding with [0]'s.
    _g2  = _g2 + [0]*(K.degree()-len(_g2));
    _pid = K.ideal(K(_g1),K(_g2));

    # assert (_pid.is_prime());
    return _pid;


# Factor base
# ------------------------------------------------------
# The first line must be: # fb_places: k=<n>\n
def fb_out_stream(stream, K, fb):
    stream.write("# fb_places: k={}\n".format(len(fb)));

    for _pid in fb:
        __pid_out_stream(stream, K, _pid);
    stream.flush();

    return;


def fb_in_stream(stream, K):
    # Control first line, extract r1+r2, precision
    _fb_head = stream.readline().rstrip(); assert (_fb_head.startswith("# fb_places: k="));
    _fb_head = _fb_head[len("# fb_places: "):];
    _kv   = dict(_s.split('=',1) for _s in _fb_head.split(' ')); assert (len(_kv) == 1); # Overkill
    _k   = sage_eval(_kv.get("k"));

    # Read _nb lines
    _fb = [];
    for _i in range(_k):
        _fb.append(__pid_in_stream(stream, K));

    assert (len(_fb) == _k);
    return _fb;


def __valp_out_stream(stream, vp):
    stream.write("{}\n".format(str(list(vp)).replace(' ', '')));
    stream.flush();
    return;


def __valp_in_stream(stream):
    _line  = stream.readline().rstrip();
    _vp    = sage_eval(_line);
    _vp    = vector(_vp);
    return _vp;


# Challenges
# ------------------------------------------------------
# Same functions to output/input challenges
def chal_out_stream(stream, K, chals, bsz):
    stream.write("# chals: k={} bsz={}\n".format(len(chals), bsz));

    for _chal in chals:
        __pid_out_stream(stream, K, _chal);
    stream.flush();
    
    return;


def chal_in_stream(stream, K):
    # Control first line, extract r1+r2, precision
    _chal_head = stream.readline().rstrip(); assert (_chal_head.startswith("# chals: k="));
    _chal_head = _chal_head[len("# chals: "):];
    _kv   = dict(_s.split('=',1) for _s in _chal_head.split(' ')); assert (len(_kv) == 2); # Overkill
    _k    = sage_eval(_kv.get("k")); # ignore key "bsz"

    # Read _nb lines
    _chal = [];
    for _i in range(_k):
        _chal.append(__pid_in_stream(stream, K));
    
    assert (len(_chal) == _k);
    return _chal;



# ------------------------------------------------------------------------------------------
# Log args representations

# Format of logarg type = (.inf, .args, .vp), with:
#    inf  vector in Re(prec),
#    args vector in [-1/2,1/2] in Re(prec),
#    vp   vector in ZZ.
# [inf[0],inf[1],...] [vp[0],vp[1],...]
def __logarg_out_stream(stream, K, la):
    _l_inf = list(la.inf);
    _vp    = list(la.vp);

    stream.write("{} {}\n".format(str(_l_inf).replace(' ', ''),
                                  str(_vp).replace(' ', '')));
    
    return;


def __logarg_in_stream(stream, K, b_prec):
    _l_inf, _vp = [sage_eval(_s) for _s in stream.readline().rstrip().split(' ')];
    
    # Determine precision
    _in_b_prec  = max([_l_inf_i.parent().precision() for _l_inf_i in _l_inf]);
    assert((_in_b_prec - b_prec).abs() < 10);
    _Re = RealField(b_prec);
    # Set logarg
    _la = nf.logarg(inf  = vector(_Re, _l_inf),
                    args = vector(_Re, [0]*len(_l_inf)),
                    vp   = vector(ZZ, _vp));
    
    return _la;


def logargs_out_stream(stream, K, la_list):
    n_inf  = nf.get_nb_inf_places(K);
    k      = len(la_list[0].vp) if len(la_list) > 0 else 0;
    b_prec = la_list[0].inf.base_ring().precision();
    assert(all(len(_la.inf)  == n_inf for _la in la_list));
    assert(all(_la_i.parent().precision() == b_prec
               for _la_i in sum((list(_la.inf) for _la in la_list), [])));
    assert(all(len(_la.args) == n_inf for _la in la_list));
    assert(all(len(_la.vp)   == k     for _la in la_list));
    stream.write("# logargs(fb): nb={} prec={} inf={} k={}\n".format(len(la_list), b_prec, n_inf, k));

    for _la in la_list:
        __logarg_out_stream(stream, K, _la);
    stream.flush();

    return;


def logargs_in_stream(stream, K):
    _la_head = stream.readline().rstrip(); assert (_la_head.startswith("# logargs(fb): nb="));
    _la_head = _la_head[len("# logargs(fb): "):];
    _kv    = dict(_s.split('=',1) for _s in _la_head.split(' ')); assert (len(_kv) == 4);
    _nb    = sage_eval(_kv.get("nb"));
    _prec  = sage_eval(_kv.get("prec"));
    _n_inf = sage_eval(_kv.get("inf"));  assert(_n_inf == nf.get_nb_inf_places(K));
    _k     = sage_eval(_kv.get("k"));
    
    # Read _nb lines
    _la_list = [];
    for _i in range(_nb):
        _la_list.append(__logarg_in_stream(stream, K, _prec));
    
    assert (len(_la_list) == _nb);
    assert (all(len(_la.inf)  == _n_inf for _la in _la_list));
    assert (all(len(_la.args) == _n_inf for _la in _la_list));
    assert (all(len(_la.vp)   == _k     for _la in _la_list));
    return _la_list;



# ------------------------------------------------------------------------------------------
# Number field elements

# Format for x=K.gen(), su=a_0 + a_1 x + ... + a_d x^d:
#     [a_0,a_1,...,a_d]\n (no padding with trailing zeroes from d to deg(K))
def __nfelt_out_stream(stream, K, nfelt):
    stream.write("{}\n".format(str(nfelt.polynomial().list()).replace(' ', '')));
    stream.flush();
    return;


def __nfelt_in_stream(stream, K):
    _line = stream.readline();
    if not _line:
        return K(0); # for incomplete cldl files
    # The following line does not work anymore with very large integers :( :@
    # _nfelt = sage_eval(_line.rstrip());
    # Alternative (verbose) version
    _line      = _line.rstrip();   assert(_line[0] == "[" and _line[-1] == "]"); _line   = _line[1:-1];
    _nfelt_str = _line.split(','); assert(len(_nfelt_str) > 0);
    _nfelt     = [];
    for _nfelt_c_str in _nfelt_str:
        _nfelt_c_str = _nfelt_c_str.split("/"); assert(len(_nfelt_c_str) in [1,2]);
        _nfelt_c     = ZZ(_nfelt_c_str[0]);
        if (len(_nfelt_c_str) > 1):
            _nfelt_c = _nfelt_c / ZZ(_nfelt_c_str[1]);
        _nfelt.append(_nfelt_c);
    # Little 'Sagerie', it works for cyclotomic fields, but generic fields need padding.
    _nfelt = K(_nfelt + [0]*(K.degree()-len(_nfelt)));
    return _nfelt;


# S-Units (corresponding to previous places)
# ---------------------------------------------
def sunits_out_stream(stream, K, u, su):
    assert (len(u) == nf.get_rank_units(K));
    stream.write("# sunits(fb): nu={} k={}\n".format(len(u), len(su)));
    
    for _su in u+su:
        __nfelt_out_stream(stream, K, _su);
    stream.flush();

    return;


def sunits_in_stream(stream, K):
    # Control first line, extract r1+r2, precision
    _su_head = stream.readline().rstrip(); assert (_su_head.startswith("# sunits(fb): nu="));
    _su_head = _su_head[len("# sunits(fb): "):];
    _kv   = dict(_s.split('=',1) for _s in _su_head.split(' ')); assert (len(_kv) == 2);
    _nu   = sage_eval(_kv.get("nu")); assert (_nu == nf.get_rank_units(K));
    _k    = sage_eval(_kv.get("k"));

    # Read units
    _u = [];
    for _i in range(_nu):
        _u.append(__nfelt_in_stream(stream, K));
        assert (_u[-1].norm().abs() == 1);
    # Read S-units
    _su = [];
    for _i in range(_k):
        _su.append(__nfelt_in_stream(stream, K));

    return _u, _su;


# Raw format
# ---------------------------------------------
# Raw format: output y_u, y_su=[(y_1,...,y_N), ...] (k elts) and B_usu=[g_1,...,g_N] \in K^N
# st. (s)u[j] = B_usu^y_(s)u[j] = prod g_i^y_i
# NB: Functions are called "S-units" but they might be applied to a lower index generating set.
def __raw_power_out_stream(stream, y_elt):
    stream.write("{}\n".format(str(list(y_elt)).replace(' ', '')));
    stream.flush();
    return;


def __raw_power_in_stream(stream):
    _line  = stream.readline().rstrip();
    _y_elt = sage_eval(_line);
    _y_elt = vector(_y_elt);
    return _y_elt;


# len(y_u) = nu, len(y_su) = k, len(B_usu) = N, len(B_vp) = N
# len(y_u|su[i]) = N, len(B_usu[i]) = K.degree() = n, len(B_vp[i]) = k
def sunits_raw_out_stream(stream, K, y_u, y_su, B_su, B_vp):
    # Test if everything is consistent
    assert (len(y_u)  == nf.get_rank_units(K));
    assert (len(B_su) == len(B_vp));
    assert (all(len(_yi) == len(B_su)  for _yi in y_u+y_su));
    assert (all(len(_vp) == len(y_su)  for _vp in B_vp)); # Suppose we have an algebraically independent family
    stream.write("# raw sunits(fb): nu={} k={} N={}\n".format(len(y_u), len(y_su), len(B_su)));

    for _yi in y_u + y_su:
        __raw_power_out_stream(stream, _yi);
    for _su in B_su:
        __nfelt_out_stream(stream, K, _su);
    for _vp in B_vp:
        __valp_out_stream(stream, _vp);
    stream.flush();

    return;


def sunits_raw_in_stream(stream, K):
    # Control first line, extract r1+r2, precision
    _su_head = stream.readline().rstrip(); assert (_su_head.startswith("# raw sunits(fb): nu="));
    _su_head = _su_head[len("# raw sunits(fb): "):];
    _kv   = dict(_s.split('=',1) for _s in _su_head.split(' ')); assert (len(_kv) == 3);
    _nu   = sage_eval(_kv.get("nu")); assert (_nu == nf.get_rank_units(K));
    _k    = sage_eval(_kv.get("k"));
    _N    = sage_eval(_kv.get("N"));
    
    # Read units raw powers
    _y_u  = [];
    for _i in range(_nu):
        _y_u.append(__raw_power_in_stream(stream));
        assert (len(_y_u[-1]) == _N);
    # Read units / S-units raw powers
    _y_su = [];
    for _i in range(_k):
        _y_su.append(__raw_power_in_stream(stream));
        assert (len(_y_su[-1]) == _N);
    # Read S-units basis
    _B_su = [];
    for _j in range(_N):
        _B_su.append(__nfelt_in_stream(stream, K));
    # Read S-units valuations on FB (that Sage cannot compute itself in reasonable time)
    _B_vp = [];
    for _j in range(_N):
        _B_vp.append(__valp_in_stream(stream));
        assert (len(_B_vp[-1]) == _k);

    assert (matrix(_y_u)*matrix(_B_vp) == 0);
    _norm_Bsu = [ RealField(fp.BIT_PREC_DEFAULT)(_raw_su.norm()) for _raw_su in _B_su ];
    assert (fp.fp_assert_zero("N(u)=1", [sum(_y*log(_n_su) for _y, _n_su in zip(_yu,_norm_Bsu)) for _yu in _y_u], target=fp.BIT_PREC_DEFAULT, sloppy=True) );
    
    return (_y_u, _y_su), _B_su, _B_vp;



# Reading ClDL solution works the same way as reading S-units
def cldl_in_stream(stream, K):
    _cldl_head = stream.readline().rstrip(); assert (_cldl_head.startswith("# cldl(../data/"));
    _cldl_head = _cldl_head[len("# cldl("):];
    _fchal     = _cldl_head.split("): ")[0];
    _kv   = dict(_s.split('=',1) for _s in _cldl_head.split(' ')[1:]); assert (len(_kv) == 3);
    _k    = sage_eval(_kv.get("k")); # Ignore bsz
    _typ  = _kv.get("typ");
    
    # Read cldl elts
    _cldl = [];
    for _i in range(_k):
        _c = __nfelt_in_stream(stream, K);
        if (_c == K(0)):
            break;
        _cldl.append(_c);
    
    return _cldl, _fchal, _typ;



# ------------------------------------------------------------------------------------------
# Precomputation files
# (Reading/Writing data files, including headers)
__END_TAG = "# --- END ---";


# Lattices / Matrix files
# ------------------------------------------------
def lattice_write_data(filename, L):
    _f_out = open(filename, "w");

    lattice_out_stream(_f_out, L);

    _f_out.write("{}\n".format(__END_TAG)); _f_out.close();
    return;
    

def lattice_read_data(filename, to_b_prec=0):
    _f_in = open(filename, "r");

    _L    = lattice_in_stream(_f_in, to_b_prec=to_b_prec);
    
    _last_line = _f_in.readline().rstrip(); assert (_last_line == __END_TAG);
    _f_in.close();
    return _L;


# Infinite places file
# ------------------------------------------------
# nf / p_inf (head+list)
def inf_places_write_data(filename, K, p_inf):
    _f_out = open(filename, "w");

    nf_out_stream(_f_out, K);
    inf_places_out_stream(_f_out, K, p_inf);

    _f_out.write("{}\n".format(__END_TAG)); _f_out.close();
    return;


def inf_places_read_data(filename, K):
    _f_in = open(filename, "r");

    _K    = nf_in_stream(_f_in); assert (_K == K);
    _p_inf = inf_places_in_stream(_f_in, K);
    assert (len(_p_inf) == nf.get_nb_inf_places(K));

    _last_line = _f_in.readline().rstrip(); assert (_last_line == __END_TAG);
    _f_in.close();
    return _p_inf;


# Finite places (FB) file
# ------------------------------------------------
#  nf / fb (head + list)
def fb_write_data(filename, K, fb):
    _f_out = open(filename, "w");

    nf_out_stream(_f_out, K);
    fb_out_stream(_f_out, K, fb);

    _f_out.write("{}\n".format(__END_TAG)); _f_out.close();
    return;


def fb_read_data(filename, K):
    _f_in = open(filename, "r");

    _K    = nf_in_stream(_f_in); assert (_K == K);
    _fb   = fb_in_stream(_f_in, K);
    
    _last_line = _f_in.readline().rstrip(); assert (_last_line == __END_TAG);
    _f_in.close();
    return _fb;


# S-Units file
# ------------------------------------------------
# nf / su (head + list)
def sunits_write_data(filename, K, u, su):
    _f_out = open(filename, "w");

    nf_out_stream(_f_out, K);
    sunits_out_stream(_f_out, K, u, su);

    _f_out.write("{}\n".format(__END_TAG)); _f_out.close();
    return;


def sunits_read_data(filename, K):
    _f_in   = open(filename, "r");
    _K      = nf_in_stream(_f_in); assert (_K == K);
    _u, _su = sunits_in_stream(_f_in, K);

    _last_line = _f_in.readline().rstrip(); assert (_last_line == __END_TAG);
    _f_in.close();
    return (_u, _su);


def sunits_raw_write_data(filename, K, y_u, y_su, B_su, B_vp):
    _f_out = open(filename, "w");

    nf_out_stream(_f_out, K);
    sunits_raw_out_stream(_f_out, K, y_u, y_su, B_su, B_vp);

    _f_out.write("{}\n".format(__END_TAG)); _f_out.close();
    return;


def sunits_raw_read_data(filename, K):
    _f_in  = open(filename, "r");
    _K     = nf_in_stream(_f_in); assert (_K == K);
    (_y_u, _y_su), _B_su, _B_vp = sunits_raw_in_stream(_f_in, K);

    _last_line = _f_in.readline().rstrip(); assert (_last_line == __END_TAG);
    _f_in.close();
    return (_y_u, _y_su), _B_su, _B_vp;


# Logargs file
# ------------------------------------------------
def logarg_write_data(filename, K, la_list):
    _f_out = open(filename, "w");

    nf_out_stream(_f_out, K);
    logargs_out_stream(_f_out, K, la_list);
    
    _f_out.write("{}\n".format(__END_TAG)); _f_out.close();
    return;


def logarg_read_data(filename, K):
    _f_in = open(filename, "r");
    _K    = nf_in_stream(_f_in); assert (_K == K);
    _la   = logargs_in_stream(_f_in, K);
    
    _last_line = _f_in.readline().rstrip(); assert (_last_line == __END_TAG);
    _f_in.close();
    return _la;



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# TODO


# challenges only
# nf / chals (head + list)
def chal_read_data(filename, K):
    _f_in  = open(filename, "r");
    _K     = nf_in_stream(_f_in); assert (_K == K);
    _chals = chal_in_stream(_f_in, K);

    _f_in.close();
    return _chals;


# ClDL solutions
# nf / cldls (head+list)
def cldl_read_data(filename, K):
    _f_in  = open(filename, "r");
    _K     = nf_in_stream(_f_in); assert (_K == K);
    _cldls, _chname, _FBtyp = cldl_in_stream(_f_in, K);

    _f_in.close();
    return _cldls, _chname, _FBtyp;







# -------------------------------------------------------------------------------------------
# [OBSOLETE]
def __old_pcmp_out_data(filename, K, p_inf, fb, u, su):
    _f_out = open(filename, "w");
    nf_out_stream(_f_out, K);

    # Pcmp: places (inf+fb), sunits, in whatever order
    fb_out_stream(_f_out, K, fb);
    sunits_out_stream(_f_out, K, u, su);
    inf_places_out_stream(_f_out, K, p_inf); # At the end because it has no interest and is very verbose

    _f_out.close();
    return;


def __old_pcmp_read_data(filename):
    _f_in = open(filename, "r");

    _K      = nf_in_stream(_f_in);
    _fb     = fb_in_stream(_f_in, _K);
    _u, _su = sunits_in_stream(_f_in, _K);
    _p_inf  = inf_places_in_stream(_f_in, _K);

    _f_in.close();
    return _K, _p_inf, _fb, (_u, _su);


# Format of the matrix output is defined as follows (to be compatible with Magma functions)
# [\n
# \t[<b_0>],\n
# ... (one line per b_i)
# \t[<b_n>],\n
# ]\n
# where each b_i is a comma separated list of elements ' c0, c1, ..., ck ',
# and each c0 is [-]d(ddd...).f(fff...)[Ee(eee...)]
def old_lattice_write_data(filename, L):
    _f_out = open(filename, "w");
    # Handle the case L is of type lattice instead of matrix ?
    # It was a problem in Magma, not sure it is in Sage
    # L = matrix(..., L);

    _L_str = ("[\n\t"
              + ",\n\t".join(map(str, [list(_row) for _row in L.rows()]))
              + ",\n]\n").replace('e','E');
    _f_out.write(_L_str);
    _f_out.close();

    return;


def old_lattice_read_data(filename, to_b_prec=0):
    _f_in = open(filename, "r");
    _M_ij = eval(preparse(_f_in.read())); # contains list of list of coeffs
    _f_in.close();

    # Get input precision
    _in_b_prec = max([_Mi[_j].parent().precision() for _Mi in _M_ij for _j in range(len(_Mi))]);
    # input precision might be way too high for both practical computations and usefulness
    # so a shrinking mechanism is provided (no control on _to_prec < _in_prec, though)
    _b_prec    = _in_b_prec if ((to_b_prec == 0) or (_in_b_prec < to_b_prec)) else to_b_prec;
     
    # Construct matrix
    _R = RealField(_b_prec);
    _M = matrix(_R, _M_ij);
    
    return _M;


def old_lattice_ZZ_read_data(filename):
    _f_in = open(filename, "r");
    _M_ij = eval(preparse(_f_in.read())); # contains list of list of coeffs
    _f_in.close();

    # Construct matrix
    _M = matrix(ZZ, _M_ij);

    return _M;
