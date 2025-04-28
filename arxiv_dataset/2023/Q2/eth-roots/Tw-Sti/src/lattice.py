# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

from sage.all import *
import fp
from ZR_mat import *


# ----------------------------------------------------------------------------------------------
# Lattice constants (Rankin-d: hermite factor/orthogonality defect, etc)

# Orthogonality defect
def lnrankin(M,d,ln_V=0):
    assert (d <= M.nrows());
    _ln_rk_d   = sum([ln(M[_k].norm()) for _k in range(d)]);
    _ln_V      = lnvol(M) if (ln_V == 0) else ln_V;
    _ln_rk_d   = _ln_rk_d - _ln_V*M.base_ring()(d/M.nrows());
    return _ln_rk_d;

def rankin(M,d,ln_V=0):
    return exp(lnrankin(M,d,ln_V=ln_V));

def lnrankin_reduced(M,d,ln_V=0):
    return (1/M.base_ring()(M.nrows()))*lnrankin(M,d,ln_V=ln_V);

def rankin_reduced(M,d,ln_V=0):
    return exp(lnrankin_reduced(M,d,ln_V=ln_V));


# [OLD] Non numerically stable versions
# def old_rankin(M, d): # This version is not numerically stable
#     assert (d <= M.nrows());
#     _red_vol = vol_reduced(M);
#     _rk_d    = mul([ M[_k].norm()/_red_vol for _k in range(d)]);
#     return _rk_d;

# def old_rankin_reduced(M, d): # This version is not numerically stable
#     return rankin(M,d)**(1/M.base_ring()(M.nrows()));


# Minimum angle
def min_angle(M):
    _pi     = M.base_ring()(pi);
    _min_th = _pi/M.base_ring()(2);
    for _i in range(M.nrows()):
        _th_i = _pi/M.base_ring()(2);
        for _j in range(_i+1, M.nrows()):
            _theta  = arccos( (M[_i]*M[_j])/M[_i].norm()/M[_j].norm() );
            _th_ij  = min(_theta, _pi-_theta);
            _th_i   = min(_th_i, _th_ij);
        _min_th = min(_min_th, _th_i);

    return _min_th;

def mean_angle(M):
    _pi     = M.base_ring()(pi);
    _s      = 0;
    _N      = M.base_ring()(M.nrows()*(M.nrows()-1))/M.base_ring()(2);
    for _i in range(M.nrows()):
        for _j in range(_i+1, M.nrows()):
            _theta  = arccos( (M[_i]*M[_j])/M[_i].norm()/M[_j].norm() );
            _th_ij  = min(_theta, _pi-_theta);
            _s      = _s + _th_ij;
    return (_s/_N);



# ------------------------------------------------------------------------------------------
# FpLLL helpers
# The Sage version of fplll has some hard-to-track floating-point issues btw versions,
# so the best is probably to call directly fplll via system calls.
import subprocess

# Path details
__FPLLL_PATH = "/usr/local/bin/";

def fplll_get_path():
    return __FPLLL_PATH;

def fplll_change_path(new_path):
    global __FPLLL_PATH;
    __FPLLL_PATH = new_path;


# System machinery for external fplll calls
# Consistency is not verified [eg. asking for BKZ with a non empty vector is going to fail] 
# Final parsing is correct under the condition that matrices are output in a compatible Sage format (-of uk, -of bk, -of vk)
def __fplll_call(act, B, t=[], opts=""):
    _t_fname = tmp_filename();
    _t_file  = open(_t_fname, 'w');
    _t_file.write("[{0}]\n".format("\n".join([str(list(i)).replace(",","") for i in B.rows()])));
    if (len(t) != 0):
        _t_file.write("{0}\n".format(str(list(t)).replace(",", "")));
    _t_file.close(); # Automatically erased when Sage exits

    # Requires Python >= 3.7
    #print ("cmd: env -i {0}./fplll -a {1} {2} {3}".format(__FPLLL_PATH, act, opts, _t_fname));
    proc = subprocess.run(["env -i {0}./fplll -a {1} {2} {3}".format(__FPLLL_PATH, act, opts, _t_fname)],
                          shell=True, stdout=subprocess.PIPE, text=True);
    # Valid if output is with commas (Sage compliant), otherwise matrices will be torn apart.
    out  = proc.stdout.rstrip().replace(",\n", ",").split("\n");
    
    assert (out[0] != '[]'); # Empty output can hide a precision problem.
    return out;


def __fplll_call_cvp(B, t):
    # Weird bug? Output '[]' with -of c, and some vector without it. Doc says "-of c (default if -a cvp)".
    [_v_str] = __fplll_call("cvp", B, t=t, opts="");
    _v       = sage_eval("vector({0})".format(_v_str.replace(" ", ",")));
    return _v;


def __fplll_call_svp(B):
    [_s_str] = __fplll_call("svp", B, opts="-of s");
    _s       = sage_eval("vector({0})".format(_s_str.replace(" ", ",")));
    return _s;


# Official LLL options
# ??
def __fplll_call_lll(B, **kwargs):
    _opt_str  = "";

    for _key, _value in kwargs.items():
        if _value == True:
            _opt_str += " -"+_key;
        else:
            _opt_str += " -"+_key+" "+str(_value);
    # Output format
    _opt_str += " -of bkuk";
    
    # LLL call, _u is st. _u*B = _bkz
    _lll_str, _u_str = __fplll_call("lll", B, opts=_opt_str);
    _lll     = sage_eval("matrix({0})".format(_lll_str));
    _u       = sage_eval("matrix({0})".format(_u_str)); # _u * B = _lll

    return _lll, _u;


# Official BKZ options
# -b: block_size -f: float_type (eg. mpfr) -p: precision
# -bkzmaxloops: int -bkzmaxtime: sec -bkzautoabort
# -s: strategy file -bkzghbound:... -bkzboundedlll -bkzdumgso filename
def __fplll_call_bkz(B, block_size=0, **kwargs):
    _bck_size = B.nrows() if (block_size == 0) else block_size;
    _opt_str  = "-b {0}".format(_bck_size);
    # Flexibility to add any arguments that are legitimate for fplll -a bkz or fplll -a hkz
    for _key, _value in kwargs.items():
        if _value == True:
            _opt_str += " -"+_key;
        else:
            _opt_str += " -"+_key+" "+str(_value);
    # Output format
    _opt_str += " -of bkuk";

    # BKZ call, _u is st. _u*B = _bkz
    _bkz_str, _u_str = __fplll_call("bkz", B, opts=_opt_str);
    _bkz     = sage_eval("matrix({0})".format(_bkz_str));
    _u       = sage_eval("matrix({0})".format(_u_str)); # _u * B = _bkz

    return _bkz, _u;



# -------------------------------------------------------------------------------------------
# LLL Reduction : return always B, U st. B=U*M, B=LLL(M)
def lll_ZZ(MZ, **kwargs):
    return __fplll_call_lll(MZ, **kwargs);

# Returns BKZ reduction of M
# Options: see __fplll_call_bkz() doc
def lll(M, work_prec=0, **kwargs):
    # Wrap
    _MZ, _l = matvec_real_to_ZZ(M, work_prec=work_prec);
    # Reduce
    _lll_Z, _uZ = __fplll_call_lll(_MZ, **kwargs);
    # Unwrap
    _lll    = matvec_ZZ_to_real(_lll_Z, _l);

    return _lll, _uZ;


# -------------------------------------------------------------------------------------------
# BKZ Reduction : return always B, U st. B=U*M, B=BKZ-k(M)
def bkz_ZZ(MZ, block_size=0, **kwargs):
    return __fplll_call_bkz(MZ, block_size=block_size, **kwargs);

# Returns BKZ reduction of M
# Options: see __fplll_call_bkz() doc
def bkz(M, work_prec=0, block_size=0, **kwargs):
    # Wrap
    _MZ, _l = matvec_real_to_ZZ(M, work_prec=work_prec);
    # Reduce
    _bkz_Z, _uZ = __fplll_call_bkz(_MZ, block_size=block_size, **kwargs);
    # Unwrap
    _bkz    = matvec_ZZ_to_real(_bkz_Z, _l);

    return _bkz, _uZ;


# ------------------------------------------------------------------------------------------
# Solving SVP/CVP

# Exact SVP
def svp_exact_ZZ(MZ):
    return __fplll_call_svp(MZ);

def svp_exact(M, work_prec=0):
    # Wrap
    _MZ, _l = matvec_real_to_ZZ(M, work_prec=work_prec);
    # Call FPLLL SVP
    _svpZ = __fplll_call_svp(_MZ); 
    # Unwrap
    _svp    = matvec_ZZ_to_real(_svpZ, _l);

    return _svp;


# Rounding
# NB: does not work if M is on ZZ.
def cvp_rounding(M, t):
    _iM = rc_mat_inverse(M);
    _y  = vector(ZZ, map(round, (t*_iM).dense_coefficient_list()));
    
    return _y*M, _y;


# Babai
# NB: This follows [Gal, \S18.1, Alg.26], but significantly differs from eg. the code of [DPW19].
def cvp_babai_NP(M, t, G=0):
    t_prec = ceil(RealField()(2*log(len(t), 2) +
                              (log(t.norm(infinity)) if not t.base_ring().is_exact() else
                               log(max(map(lambda _c: max(_c.numerator(), _c.denominator()), t.list())), 2))));
    if (G == 0):
        # GSO should handle precision correctly when M is on ZZ/QQ, here we take care of what is needed for the target
        _G, _ = gram_schmidt_ortho(M, normalize=False, b_prec=t_prec);
    else:
        assert (G.base_ring().is_exact() or G.base_ring().precision() >= t_prec), "[CVP-NP] GSO given at prec '{}' but target has size '{}'.".format(G.base_ring().precision(), t_prec);
        _G = G;
    
    _w = t;
    # Babai's Nearest Plane, see book [Gal, \S18.1, Alg.26]
    _n = M.nrows();
    _v = zero_vector(M.ncols());
    _y = [0]*M.nrows();
    for _i in range(_n-1, -1, -1):
        _z     = (_w*_G[_i])/_G[_i].norm()**2;
        _y[_i] = round(_z);
        _vi    = _y[_i]*M[_i];
        _v     = _v + _vi; # (Permutation needed here if GSO permutes rows)
        _w = _w - (_z - _y[_i])*_G[_i] - _vi; # (and here)
    
    _y = vector(ZZ, _y);
    assert (_v.base_ring() ==  M.base_ring());
    return _v, _y;


# Exact CVP
# The non-ZZ routine does take t back towards 0 using Babai NP before solving CVP(L,t),
def cvp_exact_ZZ(MZ, tZ):
    # Babai NP ?
    return __fplll_call_cvp(MZ, tZ);

def cvp_exact(M, t, work_prec=0):
    # Clean target
    _v, _ = cvp_babai_NP(M, t);
    _t    = t - _v;
    # Wrap
    _MZ, _l = matvec_real_to_ZZ(M,  work_prec=work_prec);
    _tZ, _  = matvec_real_to_ZZ(_t, work_prec=_l);
    # Call FPLLL CVP
    _cvpZ   = __fplll_call_cvp(_MZ, _tZ);
    # Unwrap (ZZ -> RR + Babai's NP)
    _cvp    = matvec_ZZ_to_real(_cvpZ, _l);
    _cvp    = _cvp + _v;
    
    return _cvp;



