"""
propagation.py

Written by Jake Rogers
Last modified: 03/10/2024

This file contains the core code to perform propagation via phase space. The 
function run_psprop(...) is the intended interface for users who have already
acquired a Mutual Optical Intensity (MOI) function. 

The algorithm proceeds as follows:
1. Rewrite the MOI using the Wigner basis
2. Calculate the Wigner Distribution Function, or Phase Space Density (PSD) 
   in the present context, by a 1D Fourier transform.
3. Propagate the PSD via a shearing transformation.
4. Calculate the propagated MOI (still using the Wigner basis). 

Extraction of the wave function from the propagated MOI occurs by taking a 
particular cross section of the propagated MOI under the Wigner basis. See 
extract_wf(...) for more details.

The term Wigner distribution is here used to refer to any distribution which 
consists of one Fourier component and one real space component. The term 
phase space density is a kind of Wigner distribution specific to the optical 
context, which has the unique property of free space propagation via shearing.
"""
import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.ndimage import affine_transform

def run_psprop(
    moi: np.ndarray,
    x: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    pixel_size: float,
    wavelength: float,
    prop_dist:  float,
    ) -> np.ndarray:
    """The main interface for performing phase space propagation.

    Args:
        moi (np.ndarray): The mutual optical intensity (MOI) at the sample or 
        unpropagated plane. 
        x (np.ndarray): The 1D array of points used to define the meshgrid
        for X1 and X2.
        X1 (np.ndarray): A 2D array which forms one of the bases for the 
        initial MOI. It is paired with X2.
        X2 (np.ndarray): A 2D array which forms the other basis for the 
        initial MOI. It is paired with X1. 
        wavelength (float): The average wavelength for the MOI.
        prop_dist (float): The desired propagation distance.

    Returns:
        np.ndarray: The MOI at the propagated plane under the Wigner basis.
    """
    n_points = moi.shape[0]
    
    moi_wig = wigner_basis(moi, x, X1, X2)
    psd = wigner_dist(moi_wig)
    
    psd_z = psd_prop(psd, n_points, pixel_size, wavelength, prop_dist)
    
    moi_wig_z = inv_wigner_dist(psd_z)
    
    return moi_wig_z

def wigner_basis(
    moi: np.ndarray,
    x: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    interp_method: str = "linear"
    ) -> np.ndarray:
    """Rewrites the basis for the mutual optical intensity (MOI) in terms of 
    averaged position and the difference in position. This is a requirement 
    for calculating the Wigner distribution function.
    
    The initial MOI can be considered mathematically as: 
        J(x1, x2)
    So that this function represents a mapping:
        J(x1, x2) --> J'(X, D)
    where X = (x1 + x2) / 2 is the averaged position and D = x2 - x1 is the 
    difference, with the prime symbol denoting the modified MOI. 
    This is represented programmatically by retaining the initial basis 
    (X1, X2) but adjusting the MOI distribution. 

    Args:
        moi (np.ndarray): The mutual optical intensity (MOI) at the sample or 
        unpropagated plane.
        x (np.ndarray): The 1D array of points used to define the meshgrid
        for X1 and X2.
        X1 (np.ndarray): A 2D array which forms one of the bases for the 
        initial MOI. It is paired with X2.
        X2 (np.ndarray): A 2D array which forms the other basis for the 
        initial MOI. It is paired with X1. 
        interp_method (str, optional): The method to use for transforming the
        MOI by interpolation. Defaults to "linear".

    Returns:
        np.ndarray: The modified MOI now under the Wigner basis.
    """
    
    moi_re = interpn((x,x), np.real(moi), ( X1+X2/2, X1-X2/2 ), 
                     method=interp_method, bounds_error=False, fill_value=0)
    moi_im = interpn((x,x), np.imag(moi), ( X1+X2/2, X1-X2/2 ), 
                     method=interp_method, bounds_error=False, fill_value=0)
    
    return moi_re + 1j * moi_im
    
def wigner_dist(moi_wig: np.ndarray) -> np.ndarray:
    """Calculates the Wigner distribution function or the phase space density
    by Fourier transform. The zeroth axis of the MOI under the Wigner basis 
    is assumed to represent the difference in position. This is handled 
    automatically if using run_psprop(...)

    Args:
        moi_wig (np.ndarray): The mutual optical intensity under the 
        Wigner basis. 

    Returns:
        np.ndarray: The Wigner distribution function also referred to as the 
        phase space density function.
    """
    wdf = ifftshift(fft(fftshift(moi_wig, axes=0), axis=0, 
                        norm='ortho'), axes=0)
    return wdf

def inv_wigner_dist(psd_z: np.ndarray) -> np.ndarray:
    """Calculates the mutual optical intensity under the Wigner basis. This
    inverts the transformation by the wigner_dist(...) function. 

    Args:
        psd_z (np.ndarray): The phase space density at the propagation or 'z' 
        position.

    Returns:
        np.ndarray: The mutual optical intensity under the Wigner basis.
    """
    moi_wig = fftshift(ifft(ifftshift(psd_z, axes=0), axis=0,
                            norm='ortho'),axes=0)
    return moi_wig

def psd_prop(
    psd: np.ndarray,
    n_points: int,
    pixel_size: float, 
    wavelength: float, 
    prop_dist:  float,
    interp_order: int = 3,
    ) -> np.ndarray:
    """Propagates the phase space density through a shearing transformation.

    Args:
        psd (np.ndarray): The phase space density at the unpropagated plane.
        n_points (int): The number of points along one dimension of the PSD.
        pixel_size (float): The size of a pixel in the real domain, i.e., the
        size of a pixel used to define the initial MOI.
        wavelength (float): The average wavelength for the MOI.
        prop_dist (float): The desired propagation distance.
        interp_order (int, optional): The order to use for interpolation by 
        the affine_transform(...) function. Defaults to 3 for a balance 
        between precision and computation time. 
        
    Returns:
        np.ndarray: The propagated PSD.
    """
    scaled_dist  = prop_dist * wavelength / (n_points * pixel_size**2)
    
    center = np.array([n_points//2, n_points//2])
    
    # A transformation matrix which models a geometric shearing of the PSD
    # This encapsulates the free space propagation for PSD distributions
    shear_mat = np.array([[1, 0], [-scaled_dist, 1]])
    
    transformed_center = shear_mat.dot(center)
    # Needed to recenter the distribution after shearing
    offset = center - transformed_center

    return affine_transform(psd, shear_mat, order=interp_order, offset=offset)

def wf_extract(moi_wig: np.ndarray, x: np.ndarray, 
               interp_method='linear') -> np.ndarray:
    """Extracts the wavefunction from a mutual optical intensity under the 
    Wigner basis. This is done through interpolating along a particular
    cross section of the function. 
    Mathematically, for MOI J'(x, d), this cross section corresponds to 
    J'(x, d = 2x) where d is the positional difference in the Wigner basis.

    Args:
        moi_wig (np.ndarray): The mutual optical intensity under the 
        Wigner basis. 
        x (np.ndarray): The 1D array of points used to define the meshgrid
        for X1 and X2.
        interp_method (str, optional): The method to use for transforming the
        MOI by interpolation. Defaults to "linear".

    Returns:
        np.ndarray: The wave function at the propagated plane
    """
    interp_re  = RegularGridInterpolator((x, x), 
                    np.real(moi_wig), method=interp_method)
    
    interp_im  = RegularGridInterpolator((x, x),
                    np.imag(moi_wig), method=interp_method)
    
    points = np.column_stack((x, x / 2))

    cross_re = interp_re(points) 
    cross_im = interp_im(points)
    cross = cross_re + 1j * cross_im
    
    return cross