# ===================================================================================
# Copyright 2024 Liam McAllister, Jakob Moritz, Richard Nally, and Andreas Schachner
#
#   This script validates the results from complex structure moduli stabilization in 
#   the various de Sitter and anti-de Sitter vacua obtained in ArXiv:2406.13751.
#
#   In the event of bugs or other issues, please reach out via as3475@cornell.edu
#   or a.schachner@lmu.de.
#
# ===================================================================================
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------------


# Srandard imports
import warnings, os, sys, time

# Import numpy
import numpy as np

# Import scipy libraries
from scipy.special import zeta

# CYTools imports
from cytools import Polytope

# custom imports
from polylogs import *


def W_flux(z,tau,Mvec,Kvec,intersection_numbers,a_matrix,n_cf,gvs, wconst = 0.):
    """
    **Summary:**
    Computes the flux superpotential for PFVs in the presence of conifold curves.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
        tau (np.complex128): Value of the axiodilaton.
        Mvec (np.ndarray): M-vector defining the PFV.
        Kvec (np.ndarray): K-vector defining the PFV.
        intersection_numbers (np.ndarray): Triple intersection numbers of the mirror CY threefold in the conifold basis.
        a_matrix (np.ndarray): A-matrix defining the quadratic piece in the prepotential for the complex structure moduli.
        n_cf (int): Number of conifolds.
        gvs (np.ndarray): GV invariants for the mirror CY threefold to be used in defining the racetrack potential.
        wconst (np.complex128,optional): Shift in the superpotential. Defaults to 0.

    Returns:
        np.complex128: Value of the flux superpotential.
    
    """

    polylog2_vals = polylog2(gvs[:,:-1]@z)

    MFa_inst = -1/(2*np.pi*1j)**2 * np.sum(gvs[:,-1]*(gvs[:,:-1]@Mvec)*polylog2_vals)

    w_value = -tau*Kvec@z+np.einsum("ijk,i,j,k",intersection_numbers,z,z,Mvec)/2-MFa_inst
    
    w_value += n_cf*Mvec[0]/24

    return wconst+w_value*np.sqrt(2/np.pi)


def W_flux_gradient(z,tau,Mvec,Kvec,intersection_numbers,a_matrix,n_cf,gvs):
    """
    **Summary:**
    Returns the gradient of the flux superpotential for PFVs in the presence of conifold curves.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
        tau (np.complex128): Value of the axiodilaton.
        Mvec (np.ndarray): M-vector defining the PFV.
        Kvec (np.ndarray): K-vector defining the PFV.
        intersection_numbers (np.ndarray): Triple intersection numbers of the mirror CY threefold in the conifold basis.
        a_matrix (np.ndarray): A-matrix defining the quadratic piece in the prepotential for the complex structure moduli.
        n_cf (int): Number of conifolds.
        gvs (np.ndarray): GV invariants for the mirror CY threefold to be used in defining the racetrack potential.
    
    Returns:
        np.ndarray: Gradient of the flux superpotential with the axiodilaton appearing at the end of the output.
    
    """

    # Derivative wrt tau
    dw_tau = -Kvec@z

    # Derivative of the contribution from worldsheet instantons wrt z
    dw_inst = -1/(2*np.pi*1j) * (np.sum( (gvs[:,-1]*(gvs[:,:-1]@Mvec)*polylog1(gvs[:,:-1]@z))[None,:]*gvs[:,:-1].T,axis=1))
    
    # Total derivative wrt z
    dw_z = -tau*Kvec + np.einsum("ijk,j,k",intersection_numbers,Mvec,z) - dw_inst

    # Combine derivatives
    dw_value = np.append(dw_z,[dw_tau])

    # Return value with correct normalisation
    return dw_value*np.sqrt(2/np.pi)

def mirror_volume(z,tau,intersection_numbers,gvs,h11,h21):
    """
    **Summary:**
    Computes volume of the mirror CY including corrections from (\alpha')^3 and worldsheet instantons.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
        tau (np.complex128): Value of the axiodilaton.
        intersection_numbers (np.ndarray): Triple intersection numbers of the mirror CY threefold in the conifold basis.
        gvs (np.ndarray): GV invariants for the mirror CY threefold to be used in defining the racetrack potential.
        h11 (int): h11 of the CY threefold (not the mirror!).
        h21 (int): h21 of the CY threefold (not the mirror!).
    
    Returns:
        np.ndarray: Value of corrected mirror volume.
    
    """

    # Define conjugate
    z_bar = np.conj(z)

    # Volumes of mirror curves
    chargz = gvs[:,:-1]@z

    # Evaluate polylog-2 for these curve volumes
    polylog2_vals = polylog2(chargz)
    
    # Compute worldsheet instanton piece
    wsi_piece = -1j/(2*np.pi*1j)**3* 2*(2*gvs[:,-1]@polylog3(chargz)-2*np.pi*1j*(gvs[:,:-1]@(z-z_bar))@(gvs[:,-1]*polylog2_vals))
    
    # Write down xi for the mirror 1-loop correction
    xi = 1/2*zeta(3)/(2*np.pi)**3*2*(h21-h11)

    # Define corrected mirror dual CY volume
    return (1j/6.*np.einsum("ijk,i,j,k",intersection_numbers,z-z_bar,z-z_bar,z-z_bar)+4*1j*xi+wsi_piece)/8

def K_cs_gradient(z,tau,intersection_numbers,gvs,h11,h21):
    """
    **Summary:**
    Computes the gradient of the Kähler potential with respect to the complex structure moduli and the axiodilaton.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
        tau (np.complex128): Value of the axiodilaton.
        intersection_numbers (np.ndarray): Triple intersection numbers of the mirror CY threefold in the conifold basis.
        gvs (np.ndarray): GV invariants for the mirror CY threefold to be used in defining the racetrack potential.
        h11 (int): h11 of the CY threefold (not the mirror!).
        h21 (int): h21 of the CY threefold (not the mirror!).
    
    Returns:
        np.ndarray: Gradient of the Kähler potential with the axiodilaton appearing at the end of the output.
    
    """
    
    # Define conjugate
    z_bar = np.conj(z)

    # Volumes of mirror curves
    chargz = gvs[:,:-1]@z

    # Define corrected mirror dual CY volume
    vtilde = mirror_volume(z,tau,intersection_numbers,gvs,h11,h21)

    # Derivative of Kähler potential with respect to tau
    dk_tau = 1j/2./np.imag(tau)

    # Gradient of WSI contributions
    wsi_piece_gradient = -1j/(2*np.pi*1j)**2 * (-np.sum((gvs[:,-1]*2*np.pi*1j*(gvs[:,:-1]@(z-z_bar))*polylog1(chargz))[None,:]*gvs[:,:-1].T,axis=1))

    # Gradient of K wrt to z
    dk_z = -1j*np.imag((1j/2.*np.einsum("ijk,j,k",intersection_numbers,z-z_bar,z-z_bar)+ wsi_piece_gradient)/8/vtilde)

    # Combine derivatives
    dk_value = np.append(dk_z,[dk_tau])

    return dk_value

def fterms_flux(z,tau,Mvec,Kvec,pvec,intersection_numbers,a_matrix,gvs,h11,h21, wconst = 0.):
    """
    **Summary:**
    Returns the F-terms for the complex structure moduli and the axiodilaton for PFVs in the presence of conifold curves.

    Args:
        z (np.ndarray): Values of the complex structure moduli.
        tau (np.complex128): Value of the axiodilaton.
        Mvec (np.ndarray): M-vector defining the PFV.
        Kvec (np.ndarray): K-vector defining the PFV.
        pvec (TYPE): Description
        intersection_numbers (np.ndarray): Triple intersection numbers of the mirror CY threefold in the conifold basis.
        a_matrix (np.ndarray): A-matrix defining the quadratic piece in the prepotential for the complex structure moduli.
        gvs (np.ndarray): GV invariants for the mirror CY threefold to be used in defining the racetrack potential.
        h11 (int): h11 of the CY threefold (not the mirror!).
        h21 (int): h21 of the CY threefold (not the mirror!).
        wconst (np.complex128,optional): Shift in the superpotential. Defaults to 0.
    
    Returns:
        np.ndarray: F-terms for the complex structure moduli and the axiodilaton with the latter appearing at the end of the array.
    
    """

    # Get curve charges
    GV_charges = gvs[:,:-1]

    # Get GV invariants
    GVinvariants = gvs[:,-1]

    # Find conifold index in the array of curve charges
    coni_curve_index = np.where(np.abs(GV_charges@pvec)<1e-8)[0][0]

    # Determine the numer of conifolds
    n_cf = GVinvariants[coni_curve_index]
    
    # Compute gradient of flux superpotential
    dW = W_flux_gradient(z,tau,Mvec,Kvec,intersection_numbers,a_matrix,n_cf,gvs)
    
    # Compute gradient of Kähler potential
    dK = K_cs_gradient(z,tau,intersection_numbers,gvs,h11,h21)
    
    # Compute value of the flux superpotential at the minimum
    W0 = W_flux(z,tau,Mvec,Kvec,intersection_numbers,a_matrix,n_cf,gvs,wconst = 0)

    # Add contribution from non-perturbative superpotential from the Kähler moduli
    W = wconst+W0

    # Combine all terms to give the F-terms for the complex structure moduli and the axiodilaton
    DW = dW+dK*W
    
    return DW,dW,dK,W0































