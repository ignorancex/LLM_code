import os
import ctypes as ct
import numpy as np
from gsl_wrappers import GSLSpline, GSLSpline2d, BICUBIC, BILINEAR
import sys

c_dbl_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
c_int_array = np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS")

if sys.version_info[0] < 3:
    ascii_string = ct.c_char_p
else:
    class ascii_string(object):
        @classmethod
        def from_param(cls, value):
            if isinstance(value, bytes):
                return value
            else:
                return value.encode('ascii')


class c_limber_config(ct.Structure):
    _fields_ = [
        ("xlog", ct.c_bool),
        ("ylog", ct.c_bool),
        ("n_ell", ct.c_int),
        ("ell", ct.POINTER(ct.c_double)),
        ("prefactor", ct.c_double),
        ("status", ct.c_int),
       #("absolute_tolerance", ct.c_double),
        #("relative_tolerance", ct.c_double),
]


LIMBER_STATUS_OK =  0
LIMBER_STATUS_ZERO =  1
LIMBER_STATUS_NEGATIVE =  2
LIMBER_STATUS_ERROR =  3

c_gsl_spline = ct.c_void_p

dirname = os.path.split(__file__)[0]
#lib = ct.cdll.LoadLibrary(os.path.join(dirname, "../../shear/spectra/interface.so"))
lib = ct.cdll.LoadLibrary(os.path.join(dirname, "src/spec_tools.so"))
lib.get_named_w_spline.restype = c_gsl_spline
lib.get_named_w_spline.argtypes = [ct.c_size_t, ascii_string, ct.c_int, c_dbl_array, ct.c_double, ct.c_void_p]

lib.get_named_w2_spline.restype = c_gsl_spline #Assign a ctypes type to specify the result type of the foreign function. Use None for void, a function not returning anything.
lib.get_named_w2_spline.argtypes = [ct.c_size_t, ascii_string, ct.c_int, c_dbl_array, ct.c_double, ct.c_void_p] #Assign a tuple of ctypes types to specify the argument types that the function accepts

lib.get_named_w3_spline.restype = c_gsl_spline #Assign a ctypes type to specify the result type of the foreign function. Use None for void, a function not returning anything.
lib.get_named_w3_spline.argtypes = [ct.c_size_t, ascii_string, ct.c_int, c_dbl_array, ct.c_double, ct.c_void_p] #Assign a tuple of ctypes types to specify the argument types that the function accepts

lib.get_named_nchi_spline.restype = c_gsl_spline
lib.get_named_nchi_spline.argtypes = [ct.c_size_t, ascii_string, ct.c_int, c_dbl_array, ct.c_void_p, ct.c_void_p]

#lib.get_named_Dchi_spline.restype = c_gsl_spline
#lib.get_named_Dchi_spline.argtypes = [ct.c_size_t, ascii_string, ct.c_int, c_dbl_array, ct.c_void_p, ct.c_void_p]

#lib.get_reduced_kernel.restype = c_gsl_spline
#lib.get_reduced_kernel.argtypes = [ct.c_void_p, ct.c_void_p]

lib.cmb_wl_kappa_kernel.restype = c_gsl_spline
lib.cmb_wl_kappa_kernel.argtypes = [ct.c_double, ct.c_double, c_gsl_spline]

lib.get_kernel_peak.restype = ct.c_double
lib.get_kernel_peak.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]

#lib.limber_integral.restype = ct.c_int
#lib.limber_integral.argtypes = [ct.POINTER(c_limber_config), ct.c_void_p, ct.c_void_p, 
#                                ct.c_void_p, ct.c_void_p, ct.c_int, c_dbl_array]

lib.limber_integral.restype = c_gsl_spline
#lib.limber_integral.restype = ct.c_int
#for f(k,z)
#lib.limber_integral.argtypes = [ct.POINTER(c_limber_config), ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p] 
# for D(k,z) and f(k,z) 
#lib.limber_integral.argtypes = [ct.POINTER(c_limber_config), ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p] 
# for D(k,z) and f(k,z) and b(k,z)
#lib.limber_integral.argtypes = [ct.POINTER(c_limber_config), ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p]
lib.limber_integral.argtypes = [ct.POINTER(c_limber_config), ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p] 

#lib.limber_integral_rsd.restype = ct.c_int
#lib.limber_integral_rsd.argtypes = [ct.POINTER(c_limber_config), ct.c_void_p, ct.c_void_p, 
#                                    ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, 
#                                    ct.c_int, ct.c_int, c_dbl_array]

lib.load_interpolator_chi.restype = ct.c_void_p
lib.load_interpolator_chi.argtypes = [ct.c_size_t, ct.c_void_p, ascii_string, ascii_string, ascii_string, ascii_string]

#for f(k,z)

lib.load_interpolator_chi2.restype = ct.c_void_p
lib.load_interpolator_chi2.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p, ascii_string, ascii_string, ascii_string, ascii_string]

#for D(k,z)

lib.load_interpolator_chi3.restype = ct.c_void_p
lib.load_interpolator_chi3.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p, ascii_string, ascii_string, ascii_string, ascii_string]


#for b(k,z)

lib.load_interpolator_chi4.restype = ct.c_void_p
lib.load_interpolator_chi4.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p, ascii_string, ascii_string, ascii_string, ascii_string]


c_power_scaling_function = ct.CFUNCTYPE(ct.c_double, ct.c_double, 
    ct.c_double, ct.c_double, ct.c_voidp)

lib.load_interpolator_chi_function.restype = ct.c_void_p
lib.load_interpolator_chi_function.argtypes = [ct.c_size_t, ct.c_void_p, 
ascii_string, ascii_string, ascii_string, ascii_string, c_power_scaling_function, ct.c_void_p]

#for f(k,z)

lib.load_interpolator_chi_function2.restype = ct.c_void_p
lib.load_interpolator_chi_function2.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p, 
ascii_string, ascii_string, ascii_string, ascii_string, c_power_scaling_function, ct.c_void_p]


#for D(k,z)

lib.load_interpolator_chi_function3.restype = ct.c_void_p
lib.load_interpolator_chi_function3.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p, 
ascii_string, ascii_string, ascii_string, ascii_string, c_power_scaling_function, ct.c_void_p]

#for b(k,z)

lib.load_interpolator_chi_function4.restype = ct.c_void_p
lib.load_interpolator_chi_function4.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p, 
ascii_string, ascii_string, ascii_string, ascii_string, c_power_scaling_function, ct.c_void_p]


lib.interp_2d.restype = ct.c_double
lib.interp_2d.argtypes = [ct.c_double, ct.c_double, ct.c_void_p]

lib.destroy_interp_2d.restype = None
lib.destroy_interp_2d.argtypes = [ct.c_void_p]


def get_cmb_kappa_spline(chi_max, chi_star, a_of_chi):
    "Compute the CMB WL kernel W_cmb(chi) spline"
    return GSLSpline(lib.cmb_wl_kappa_kernel(chi_max, chi_star, a_of_chi))

def evaluate_power(power, k, z):
    return lib.interp_2d(k,z,power)

def free_power(power):
    try:
        lib.destroy_interp_2d(power)
    except ct.ArgumentError as e:
        power.__del__()


def get_named_nchi_spline(block, section, nbin, z, a_of_chi, chi_of_z):
    return GSLSpline(lib.get_named_nchi_spline(block._ptr, section, nbin, z, a_of_chi, chi_of_z))

#def get_named_Dchi_spline(block, section, nbin, z, a_of_chi, chi_of_z):
#    return GSLSpline(lib.get_named_Dchi_spline(block._ptr, section, nbin, z, a_of_chi, chi_of_z))


def get_named_w_spline(block, section, bin, z, chi_max, a_of_chi):
    "Compute a galcl kernel W(chi) spline"
    return GSLSpline(lib.get_named_w_spline(block._ptr, section, bin, z, chi_max, a_of_chi))

def get_named_w2_spline(block, section, bin, z, chi_max, a_of_chi):
    "Compute a galcl kernel W2(chi) spline"
    return GSLSpline(lib.get_named_w2_spline(block._ptr, section, bin, z, chi_max, a_of_chi))

def get_named_w3_spline(block, section, bin, z, chi_max, a_of_chi):
    "Compute a galcl kernel W3(chi) spline"
    return GSLSpline(lib.get_named_w3_spline(block._ptr, section, bin, z, chi_max, a_of_chi))



def load_power_chi(block, chi_of_z, section, k_name, z_name, p_name):
    "Load P(k,z) and convert z -> chi"
    r = lib.load_interpolator_chi(block._ptr, chi_of_z, section, k_name, z_name, p_name)
    if not r:
        raise ValueError("Could not load power spectrum from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, p_name))
    return r


def load_power_chi2(block, chi_of_z, z_of_chi, section, k_name, z_name, t_name):
    "Load T(k,z) and convert z -> chi -> z(chi)"
    rr = lib.load_interpolator_chi2(block._ptr, chi_of_z, z_of_chi, section, k_name, z_name, t_name)
    if not rr:
        raise ValueError("Could not load growth rate f from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, t_name))
    return rr

def load_power_chi3(block, chi_of_z, z_of_chi, section, k_name, z_name, p_name):
    "Load p(k,z) and convert z -> chi -> z(chi)"
    rrr = lib.load_interpolator_chi3(block._ptr, chi_of_z, z_of_chi, section, k_name, z_name, p_name)
    if not rrr:
        raise ValueError("Could not load growth factor D from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, p_name))
    return rrr


def load_power_chi4(block, chi_of_z, z_of_chi, section, k_name, z_name, p_name):
    "Load p(k,z) and convert z -> chi -> z(chi)"
    rrrr = lib.load_interpolator_chi4(block._ptr, chi_of_z, z_of_chi, section, k_name, z_name, p_name)
    if not rrrr:
        raise ValueError("Could not load galaxy bias b from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, p_name))
    return rrrr


def load_power_chi_function(block, chi_of_z, section, k_name, z_name, p_name, function, args):
    "Load P(k,z) and convert z -> chi and scale P->f(k,z)*P"
    r = lib.load_interpolator_chi_function(block._ptr, chi_of_z, section, k_name, z_name, p_name, function, args)
    if not r:
        raise ValueError("Could not load scaled power spectrum from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, p_name))
    return r

def load_power_chi_function2(block, chi_of_z, z_of_chi, section, k_name, z_name, t_name, function, args):
    "Load T(k,z) and convert z -> chi and scale T->f(k,z)*T"
    rr = lib.load_interpolator_chi_function2(block._ptr, chi_of_z, z_of_chi, section, k_name, z_name, t_name, function, args)
    if not rr:
        raise ValueError("Could not load scaled growth rate f from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, t_name))
    return rr

def load_power_chi_function3(block, chi_of_z, z_of_chi, section, k_name, z_name, p_name, function, args):
    "Load pnl(k,z) and convert z -> chi and scale pnl->f(k,z)*pnl"
    rrr = lib.load_interpolator_chi_function3(block._ptr, chi_of_z, z_of_chi, section, k_name, z_name, p_name, function, args)
    if not rrr:
        raise ValueError("Could not load scaled growth factor D from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, p_name))
    return rrr


def load_power_chi_function4(block, chi_of_z, z_of_chi, section, k_name, z_name, p_name, function, args):
    "Load pnl(k,z) and convert z -> chi and scale pnl->f(k,z)*pnl"
    rrrr = lib.load_interpolator_chi_function4(block._ptr, chi_of_z, z_of_chi, section, k_name, z_name, p_name, function, args)
    if not rrrr:
        raise ValueError("Could not load scaled galaxy bias b from section {0} (k:{1} z:{2} p:{3})".format(section, k_name, z_name, p_name))
    return rrrr



def get_kernel_peak(WbX, WbY, nchi=500): #changed but not used
    "Get chi of maximum of kernel"
    return lib.get_kernel_peak(WbX, WbY, nchi)


def limber(WbX, WfX, WmX, WbY, WfY, WmY, P, fk, Dk, Bk, xlog, ylog, ell, prefactor):
    config = c_limber_config()
    config.xlog = xlog
    config.ylog = ylog
    config.n_ell = len(ell)
    config.ell = np.ctypeslib.as_ctypes(ell)
    config.prefactor = prefactor
    config.status = 0
    spline_ptr = lib.limber_integral(ct.byref(config), WbX, WfX, WmX, WbY, WfY, WmY, P, fk, Dk, Bk)
    if config.status == LIMBER_STATUS_ZERO:
        ylog = False
    elif config.status == LIMBER_STATUS_NEGATIVE:
        ylog = False
    spline = GSLSpline(spline_ptr, xlog=xlog, ylog=ylog)
    return spline
