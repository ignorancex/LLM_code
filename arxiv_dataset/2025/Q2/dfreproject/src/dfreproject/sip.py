from typing import Tuple

import astropy
import torch


def get_sip_coeffs(
    wcs: astropy.wcs.WCS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract SIP polynomial coefficients from a WCS object.

    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object potentially containing SIP distortion.

    Returns:
    --------
    dict:
        Dictionary containing SIP coefficient matrices A, B, AP, BP and orders.
    """
    sip_coeffs = {}

    # Check if SIP distortion is present
    sip = getattr(wcs, "sip", None)
    if sip is None:
        return None

    # Extract the SIP matrices
    sip_coeffs["a_order"] = sip.a_order
    sip_coeffs["b_order"] = sip.b_order
    sip_coeffs["a"] = sip.a
    sip_coeffs["b"] = sip.b

    # Check for inverse coefficients
    if hasattr(sip, "ap_order") and sip.ap_order > 0:
        sip_coeffs["ap_order"] = sip.ap_order
        sip_coeffs["ap"] = sip.ap
    else:
        sip_coeffs["ap_order"] = 0

    if hasattr(sip, "bp_order") and sip.bp_order > 0:
        sip_coeffs["bp_order"] = sip.bp_order
        sip_coeffs["bp"] = sip.bp
    else:
        sip_coeffs["bp_order"] = 0

    return sip_coeffs


def apply_sip_distortion(
    u: torch.Tensor, v: torch.Tensor, sip_coeffs: Tuple, device: str = "cpu"
):
    """
    Apply SIP distortion to intermediate pixel coordinates.

    Parameters:
    -----------
    u, v : torch.Tensor
        Intermediate pixel coordinates (before distortion).
    sip_coeffs : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        SIP coefficient matrices.
    device : torch.device, optional
        Device to place tensors on.

    Returns:
    --------
    tuple:
        (u', v') distorted coordinates.


    # Convert to tensors if needed

    """
    if sip_coeffs is None:
        return u, v

    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)


    a_order = sip_coeffs["a_order"]
    a_matrix = torch.tensor(sip_coeffs["a"], dtype=torch.float64, device=device)
    b_matrix = torch.tensor(sip_coeffs["b"], dtype=torch.float64, device=device)

    # Precompute powers of u and v
    # Save memory by computing and using only needed powers
    u_powers = [torch.ones_like(u)]
    v_powers = [torch.ones_like(v)]

    for n in range(1, a_order + 1):
        u_powers.append(u_powers[-1] * u)
        v_powers.append(v_powers[-1] * v)

    # Apply polynomial distortion
    f_u = torch.zeros_like(u)
    f_v = torch.zeros_like(v)

    for i in range(a_order + 1):
        for j in range(a_order + 1 - i):
            if i == 0 and j == 0:
                continue  # Skip the (0,0) term

            term = u_powers[i] * v_powers[j]
            f_u = f_u + a_matrix[i, j] * term
            f_v = f_v + b_matrix[i, j] * term

    # Return corrected coordinates
    return u + f_u, v + f_v


def apply_inverse_sip_distortion(
    u: torch.Tensor, v: torch.Tensor, sip_coeffs: Tuple, device: str = "cpu"
):
    """
    Apply inverse SIP distortion to go from distorted to intermediate coordinates.

    Parameters:
    -----------
    u, v : torch.Tensor
        Distorted coordinates.
    sip_coeffs : Tuple
        SIP coefficient matrices.
    device : torch.device, optional
        Device to place tensors on.

    Returns:
    --------
    tuple:
        (u', v') undistorted coordinates.
    """
    if sip_coeffs is None:
        return u, v
    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)


        # Use iterative method if inverse SIP coefficients are not defined
    if sip_coeffs["ap_order"] == 0 or sip_coeffs["bp_order"] == 0:
        return iterative_inverse_sip(u, v, sip_coeffs, device)

        # Ensure inputs are tensors on correct device and dtype
    u = u.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)

    ap_order = sip_coeffs["ap_order"]

    ap_matrix = torch.tensor(sip_coeffs["ap"], dtype=torch.float32, device=device)
    bp_matrix = torch.tensor(sip_coeffs["bp"], dtype=torch.float32, device=device)
    del sip_coeffs
    # Precompute powers of u and v to avoid repeated allocation
    u_powers = [torch.ones_like(u)]
    v_powers = [torch.ones_like(v)]
    for n in range(1, ap_order + 1):
        u_powers.append(u_powers[-1] * u)
        v_powers.append(v_powers[-1] * v)

    # Initialize correction terms
    f_u = torch.zeros_like(u)
    f_v = torch.zeros_like(v)

    for i in range(ap_order + 1):
        for j in range(ap_order + 1 - i):
            if i == 0 and j == 0:
                continue
            term = u_powers[i] * v_powers[j]
            f_u = f_u + ap_matrix[i, j] * term
            f_v = f_v + bp_matrix[i, j] * term
    del u_powers, v_powers, ap_matrix, bp_matrix
    return u + f_u, v + f_v


def iterative_inverse_sip(
    u: torch.Tensor,
    v: torch.Tensor,
    sip_coeffs: Tuple,
    device: str = "cpu",
    max_iter: int = 20,
    tol: float = 1e-8,
):
    """
    Iteratively solve for undistorted coordinates when inverse SIP coefficients
    are not available.

    Parameters:
    -----------
    u, v : torch.Tensor
        Distorted coordinates.
    sip_coeffs : Tuple
        SIP coefficient matrices.
    device : torch.device, optional
        Device to place tensors on.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance.

    Returns:
    --------
    tuple:
        (u', v') undistorted coordinates.
    """
    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Initial guess: undistorted = distorted
    u_undist = u.clone()
    v_undist = v.clone()

    for i in range(max_iter):
        # Apply forward SIP to get predicted distorted coordinates
        u_pred, v_pred = apply_sip_distortion(u_undist, v_undist, sip_coeffs, device)

        # Compute error
        u_error = u - u_pred
        v_error = v - v_pred

        # Check convergence
        max_error = torch.max(
            torch.abs(torch.cat([u_error.flatten(), v_error.flatten()]))
        )
        if max_error < tol:
            break

        # Update undistorted coordinates
        u_undist = u_undist + u_error
        v_undist = v_undist + v_error

    return u_undist, v_undist
