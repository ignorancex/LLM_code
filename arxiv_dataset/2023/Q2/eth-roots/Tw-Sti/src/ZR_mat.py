# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

from sage.all import *

import fp


# ----------------------------------------------------------------------------------------------
# General "matrix on real" caveats handling
def rc_mat_inverse(M):
    _R      = M.base_ring(); # Real or Complex
    _b_prec = _R.precision();
    _RR     = _R.to_prec(4*_b_prec); # Should be enough, 1 is not sufficient, 2 or 3 might work

    _iM     = ((M.change_ring(_RR)).inverse()).change_ring(_R);

    # Check there is no precision issue (NB: compute the product in high precision)
    _chk_M  = (M.change_ring(_RR)*_iM.change_ring(_RR)).change_ring(_R);
    assert (fp.fp_assert_equal("M*iM=In", _chk_M.list(), identity_matrix(M.nrows()).list(),
                               target=_b_prec, sloppy=True)); # /!\ Sloppy
    return _iM;


# Conversions RR <--> ZZ
# Scaling factors are given in log_2 to match "precision" of RR

# Get integer equivalent of (M) [ returns floor(2^prec*M) on R ]
# Returns also the log_2 of the applied scaling factor.
def matvec_real_to_ZZ(Mv, work_prec=0):
    _R         = Mv.base_ring().fraction_field();
    _log_scale = _R.precision() if (work_prec == 0) else work_prec;
    _scale     = Integer(2)**(_log_scale);
    _MZ        = Mv.apply_map(lambda _mij:Integer(floor(_scale*_mij)));

    return _MZ, _log_scale;


# Returns to real numbers
def matvec_ZZ_to_real(Mv, log_scale, to_prec=0):
    assert ((to_prec == 0) or log_scale <= to_prec);

    _prec = log_scale if (to_prec == 0) else to_prec;
    _R    = RealField(_prec);
    _MR   = Mv.apply_map(lambda _mij:_R(_mij)/_R(2)**(log_scale));
    
    assert (_MR.base_ring().precision() >= _prec);
    return _MR;



# ----------------------------------------------------------------------------------------------
# Gram-Schmidt Orthogonalization
# Input: M is m*n matrix on any Ring
# Return G,P: G is GSO of M (normalized if normalize=True), and permutation matrix (of the rows)
#             P is the transition matrix such that M = P*G
# NB: annoyingly, sage does not implement this outside of ZZ/QQ, RDF and CDF.
# NB: b_prec is effective iff M has exact ring AND exact=False
def gram_schmidt_ortho(M, normalize=False, exact=False, b_prec=fp.BIT_PREC_DEFAULT):
    _R     = M.base_ring();
    if (_R.is_exact() and exact==True):
        _R = _R.fraction_field();
    elif (_R.is_exact()):
        b_prec = 2 * max(b_prec,
                     ceil(RealField()(log(max(map(lambda _c: max(_c.numerator(), _c.denominator()), M.list())), 2)
                                      + 2*log(max(M.ncols(),M.nrows()), 2))) );
        _R = RealField(b_prec);
        print ("[GSO] **Warn** Coeffs of M in exact ring, moved to '{}'.".format(_R));
    elif (_R.precision() < b_prec):
        print ("[GSO] **Warn** Asked for precision '{}' but input has precision '{}'.".format(b_prec, _R.precision()));
        
    _n = M.nrows();
    _G = M.change_ring(_R);
    _P = identity_matrix(_R, _n);
    
    # Main loop
    # NB: Exchanging _i and _j loop would lead to somewhat "order-independent" algorithm,
    #     allowing to choose the smallest length projection for the next step
    #    for _i in range(1,_n):
    #        _G[_i] = _G[_i] - sum( (_G[_i]*_G[_j])/_G[_j].norm()**2 * _G[_j] for _j in range(_i));
    _G_norm = [_G[0].norm()]*_n;
    for _i in range(1,_n):
        t= cputime();
        # _mu_i       = [(_G[_i]*_G[_j])/_G[_j].norm()**2 for _j in range(_i)] + [1] + [0]*(_n-_i-1);
        _mu_i       = [(_G[_i]*_G[_j])/_G_norm[_j]**2 for _j in range(_i)] + [1] + [0]*(_n-_i-1);
        t1 = cputime(t);
        _G[_i]      = _G[_i] - sum(_mu_i[_j] * _G[_j] for _j in range(_i));
        _G_norm[_i] = _G[_i].norm();
        t2 = cputime(t);
        _P[_i] = vector(_R, _mu_i);#[_mu_i[_j] for _j in range(_i)] + [_R(1)] + [_R(0)]*(_n-_i-1));
        t3 = cputime(t);
        # print ("{}: {:.2f} {:.2f} {:.2f}".format(_i, t1, t2-t1, t3-t2));
        
    # Orthonormalization (not by default)
    if (normalize == True):
        for _i in range(_n):
            _norm_i   = _G_norm[_i];
            _P[:,_i] *= _norm_i;
            _G[_i]   /= _norm_i;
            
    # **Warn** These assertions are not costless
    if (exact == False) and (_n < 100):
        assert (fp.fp_assert_equal("M=PG", M.list(), (_P*_G).list(), target=_R.precision()//3, sloppy=True));
    elif (_n < 100):
        assert (M-_P*_G == 0);
    return _G, _P;



# ----------------------------------------------------------------------------------------------
# Volumes
# Return Vol M:
# - if M is on an exact ring (ZZ, Fp, etc), volume is best computed as sqrt |det M*Mt|
# - if M is on a field "with precision" (RR, CC), the above is not numerically stable, so it is
#   better advised to compute it as exp( sum ln ||gi*|| for gi* in M*:=gram_schmidt_ortho(M))
def lnvol(M, gso=0):
    assert (M.nrows() <= M.ncols());
    _gM = gram_schmidt_ortho(M)[0] if (gso == 0) else gso; assert(_gM.nrows() == M.nrows());
    return sum([ ln(_gM[_k].norm()) for _k in range(M.nrows())]);

def vol_exact(M):
    assert(M.base_ring().is_exact());
    if M.is_square():
        _vol = M.determinant().abs();
    else:
        assert (M.nrows() <= M.ncols());        
        _gram_M = M*M.transpose();
        _vol    = _gram_M.determinant().abs().sqrt();
    return _vol;

def vol(M, gso=0):
    if (M.base_ring().is_exact()):
        return vol_exact(M);
    else:
        return exp(lnvol(M, gso=gso));



# Return |Vol M|^(1/n), where n is the matrix number of rows
def lnvol_reduced(M, gso=0):
    return (1/M.base_ring()(M.nrows()))*lnvol(M, gso=gso);

def vol_reduced(M, gso=0):
    if (M.base_ring().is_exact()):
        return vol_exact(M)**(1/M.nrows());
    else:
        return exp(lnvol_reduced(M, gso=gso));


# ----------------------------------------------------------------------------------------------
# Hermite Normal Forms

# Always a pain to retro-engineer HNF conventions. There are 2 parameters to consider:
# - row or column vectors (rows: H = UA, cols: H = AU, U \in GL(ZZ))
# - lower- or upper-triangular HNF
# Digging into conventions:
# - Cohen [Coh96, Def.2.4.2] is      column style upper triangular HNF
# - Sage  *.hermite_form() outputs a row    style upper triangular HNF
# - "Wikipedia row-style"    is      row    style upper triangular HNF
# - "Wikipedia column-style" is      column style lower triangular HNF
# - Magma                  outputs a row    style lower triangular HNF (To Be Confirmed)
# - Pari                   outputs a row    style lower triangular HNF
# For example, S-units in Sage (which map to Pari) are given with their valuations in row/lower triangular HNF.

# It is easy to navigate through them as shown by the following diagram:
#         Column / Upper (Cohen)   <--------------------------->  Row / Lower (Pari)
#                  ^                        (Transpose)                 ^
#                  |                                                    |
#                  | (Reverse rows / cols)        (Reverse rows / cols) |
#                  |                                                    |
#                  V                        (Transpose)                 V
#         Column / Lower           <--------------------------->  Row / Upper (Sage)

# Using matrices with row vectors here, we privilege the Row / Lower that most naturally
# transposes Cohen's HNF.


# Default algorithm for HNFs is "flint", which is the most inefficient possible choice.
# For a prime ideal in Q(z521), we have: pari0 ~> pari1 (1s) > ntl (2s) > flint (4s) >> pari4 (9s)
# + If the coefficients are small, 'pari0' is a very good choice;
# + if the covolume is very large, 'padic' is very efficient.
def row_upper_hermite_form(MZ, **kwargs):
    return MZ.hermite_form(**kwargs);


def row_lower_hermite_form(MZ, **kwargs):
    # Change convention
    MZ.reverse_rows_and_columns(); # /!\ This actually modifies the matrix
    # The copy is here because *.hermite_form() returns immutable matrices, and we need to reverse everythg
    # If transformation = True is passed through, then we have 2 output values
    if (kwargs.get('transformation', False) == True):
        H, U = MZ.hermite_form(**kwargs);
        # H0 = H1*f(MZ) --> f(H0) = f(H1)*MZ
        # [actually, f(M) = Ir.M.Ic, where Ik denotes the counter-diagonal exchange matrix of dimension k]
        H  = copy(H); H.reverse_rows_and_columns();
        U  = copy(U); U.reverse_rows_and_columns(); 
        HU = (H,U);
    else:
        H  = copy(MZ.hermite_form(**kwargs));
        H.reverse_rows_and_columns();
        HU = H;
    # Restore MZ as it was
    MZ.reverse_rows_and_columns(); # /!\ A priori we weren't supposed to modify MZ in this procedure
    return HU;
