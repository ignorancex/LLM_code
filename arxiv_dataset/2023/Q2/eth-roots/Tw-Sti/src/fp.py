# Code in support of ePrint:2021/1384
# Copyright 2021, Olivier Bernard
# GPL-3.0-only (see LICENSE file)

# Floating Point arithmetic handling
from sage.all import *


# Default precision (arbitrary)
BIT_PREC_DEFAULT = 500;

# -------------------------------------------------------------------------------------------
# Test at zero
# 1. It is almost impossible to guarantee when working at precision N that the precision drift
#    will be contained at +20 (even if in most cases it is the case, some functions are not
#    numerically stable, e.g. matrix inversion is particularly difficult to this respect.
# 2. Hence, it is a problem to decide which is an acceptable drift, and what is not.
#    We choose to accept values under lN + O, and suggest l=__BIT_PREC_SLOPPY and O = __BIT_PREC_SHIFT for sloppy results.
#    The offset/shift is always tolerated, even in the "exact" case.
# 3. The provided interface allows to relax the requirements locally.
# 4. There is a non-modifiable maximum gap: if the result is True, it ensures val < 2^{__BIT_PREC_MAX_ACCEPT}.

__BIT_PREC_EXACT      = 0.99; # 1.0 is too demanding for very high precisions
__BIT_PREC_SLOPPY     = 0.25; # Seems impossible to guarantee M*M^(-1) = 1 at prec(M)
__BIT_PREC_SHIFT      =  30;  # 4 or 5 digits are generally sufficient (somewhere btw 10^4 and 10^5).
__BIT_PREC_MAX_ACCEPT = -32;

# By default, no sloppyness is applied
SLOPPY_DEFAULT        = False;


def __fp_get_threshold(target, sloppy, __sloppyness, __shift):
    # Determine the target precision
    __sloppyness = __BIT_PREC_EXACT if (sloppy == False) else __sloppyness;
    _threshold   = min(__BIT_PREC_MAX_ACCEPT, -target*__sloppyness+__shift);
    return _threshold;


# ------------------------------------------------------------------------------------------------------------
# Checks against zero 
# If sloppy = False (default), __sloppyness is automatically set to __BIT_PREC_EXACT.
# Csqc: if you want to use another sloppyness factor, you must pass sloppy=True.
def fp_is_zero(val, target=BIT_PREC_DEFAULT, sloppy=SLOPPY_DEFAULT,
               __sloppyness=__BIT_PREC_SLOPPY, __shift=__BIT_PREC_SHIFT):
    _threshold = __fp_get_threshold(target, sloppy, __sloppyness, __shift);
    # Test against the target
    _l2_infty  = log(abs(val), 2);
    _is_zero   = True if (_l2_infty < _threshold) else False;

    return _is_zero;


# Returns true iff val is "precision"-close to ZZ
def fp_is_integer(val, target=BIT_PREC_DEFAULT, sloppy=SLOPPY_DEFAULT,
                  __sloppyness=__BIT_PREC_SLOPPY, __shift=__BIT_PREC_SHIFT):
    val_ZZ = round(val);
    return fp_is_zero(val-val_ZZ, target=target, sloppy=sloppy, __sloppyness=__sloppyness, __shift=__shift);


# Check for list of values
def fp_all_zero(lval, target=BIT_PREC_DEFAULT, sloppy=SLOPPY_DEFAULT,
                __sloppyness=__BIT_PREC_SLOPPY, __shift=__BIT_PREC_SHIFT):
    _threshold = __fp_get_threshold(target, sloppy, __sloppyness, __shift);
    # Get the max log over all values and test over the target
    _l2_infty    = -Infinity if (len(lval) == 0) else max([log(abs(_val), 2) for _val in lval]);
    _all_zero     = True if (_l2_infty < _threshold) else False;

    return _all_zero;


def fp_all_equal(lval, rval, target=BIT_PREC_DEFAULT, sloppy=SLOPPY_DEFAULT,
                 __sloppyness=__BIT_PREC_SLOPPY, __shift=__BIT_PREC_SHIFT):
    _Re   = RealField(target);
    # Avoid weird behaviour when prec of one of lval, rval is very small.
    _dval = [ _Re(_lval) - _Re(_rval) for _lval, _rval in zip(lval, rval) ];
    return fp_all_zero(_dval, target=target, sloppy=sloppy, __sloppyness=__sloppyness, __shift=__shift);



# ------------------------------------------------------------------------------------------------------------
# For assertions (list of values)
# This function displays the max log of the absolute value if all are not close enough to zero.
def fp_assert_zero(description, lval, target=BIT_PREC_DEFAULT, sloppy=SLOPPY_DEFAULT,
                  __sloppyness=__BIT_PREC_SLOPPY, __shift=__BIT_PREC_SHIFT):
    # Copy fp_all_zero here, we need _l2_infty for the error message
    _threshold = __fp_get_threshold(target, sloppy, __sloppyness, __shift);
    # Get the max log over all values and test over the target
    _l2_infty    = -Infinity if (len(lval) == 0) else max([log(abs(_val), 2) for _val in lval]);
    _all_zero     = True if (_l2_infty < _threshold) else False;

    # Comprehensive error message
    if (_all_zero == False):
        print ("\x1b[31m[Err]\x1b[0m log_2 infty-norm of '{}' is {:.2f} [thr:{:.2f}/exp:{:.2f}]".format(description,float(_l2_infty),float(_threshold),float(-target)));
    
    return _all_zero;


def fp_assert_equal(description, lval, rval, target=BIT_PREC_DEFAULT, sloppy=SLOPPY_DEFAULT,
                    __sloppyness=__BIT_PREC_SLOPPY, __shift=__BIT_PREC_SHIFT):
    _Ce   = ComplexField(target);
    # Avoid weird behaviour when prec of one of lval, rval is very small.
    _dval = [ _Ce(_lval) - _Ce(_rval) for _lval, _rval in zip(lval, rval) ];
    return fp_assert_zero(description, _dval, target=target, sloppy=sloppy,
                          __sloppyness=__sloppyness, __shift=__shift);
