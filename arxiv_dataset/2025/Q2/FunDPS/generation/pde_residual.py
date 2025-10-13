import torch
import torch.nn.functional as F


def get_darcy_residual(x_pred):
    """Calculate the PDE loss for Darcy flow equation.

    The Darcy flow equation is: -∇·(a∇u) = f
    where a is the permeability field and u is the pressure field.

    Args:
        x_pred (torch.Tensor): Predicted data containing both fields
            First channel ([:,0:1]) is permeability field a
            Second channel ([:,1:2]) is pressure field u

    Returns:
        torch.Tensor: PDE residual
    """
    device = x_pred.device
    a_pred = x_pred[:, 0:1]
    u_pred = x_pred[:, 1:2]

    # Initialize derivative kernels for finite difference
    length = a_pred.shape[-1]
    dx = 1 / (length - 1)
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * dx)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * dx)

    # Calculate gradients using finite difference
    grad_x = F.conv2d(u_pred, deriv_x, padding=(0, 1))
    grad_y = F.conv2d(u_pred, deriv_y, padding=(1, 0))

    # Multiply by permeability field
    grad_x = a_pred * grad_x
    grad_y = a_pred * grad_y

    # Calculate divergence
    div_x = F.conv2d(grad_x, deriv_x, padding=(0, 1))
    div_y = F.conv2d(grad_y, deriv_y, padding=(1, 0))

    # Full PDE residual (-∇·(a∇u) = 1)
    pde_residual = div_x + div_y + 1

    # Remove boundary points
    pde_residual = pde_residual[..., 2:-2, 2:-2]
    return pde_residual


def get_poisson_residual(x_pred):
    """Calculate the PDE loss for Poisson equation.

    The Poisson equation is: ∇²u = a
    where a is the source term and u is the solution field.

    Args:
        x_pred (torch.Tensor): Predicted data containing both fields
            First channel ([:,0:1]) is source term a
            Second channel ([:,1:2]) is solution field u

    Returns:
        torch.Tensor: PDE residual
    """
    a_pred = x_pred[:, 0:1]
    u_pred = x_pred[:, 1:2]

    # Calculate grid spacing
    length = a_pred.shape[-1]
    h = 1 / (length - 1)

    # Pad u for finite difference calculation
    u_padded = F.pad(u_pred, (1, 1, 1, 1), mode="constant", value=0)

    # Calculate Laplacian using second order finite difference
    laplacian = (u_padded[..., :-2, 1:-1] + u_padded[..., 2:, 1:-1] + u_padded[..., 1:-1, :-2] + u_padded[..., 1:-1, 2:] - 4 * u_pred) / h**2

    # PDE residual (∇²u - a = 0)
    pde_residual = laplacian - a_pred

    # Remove boundary points
    pde_residual = pde_residual[..., 1:-1, 1:-1]
    return pde_residual


def get_ns_nonbounded_residual(x_pred):
    """Calculate the PDE residual for non-bounded Navier-Stokes equation."""
    device = x_pred.device
    vorticity = x_pred[:, 1:2]

    # dx = 1 / (vorticity.shape[-1] - 1)
    dx = 1.0
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * dx)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * dx)

    # Calculate divergence of vorticity
    div_vort_x = F.conv2d(vorticity, deriv_x, padding=(0, 1))
    div_vort_y = F.conv2d(vorticity, deriv_y, padding=(1, 0))
    pde_residual = div_vort_x + div_vort_y

    # Remove boundary points
    pde_residual = pde_residual[:, :, 1:-1, 1:-1]
    return pde_residual


def get_helmholtz_residual(x_pred):
    """Calculate the PDE residual for Helmholtz equation: -∇²u + u = a

    The Helmholtz equation in this case relates a source term 'a' to a field 'u'
    through the Helmholtz operator: -∇²u + u = a

    Args:
        x_pred (torch.Tensor): Predicted data containing both fields
            First channel ([:,0:1]) is source field a
            Second channel ([:,1:2]) is solution field u

    Returns:
        torch.Tensor: PDE residual (-∇²u + u - a)
    """
    a_pred = x_pred[:, 0:1]  # Source term
    u_pred = x_pred[:, 1:2]  # Solution field

    # Get spatial resolution and grid spacing
    length = u_pred.shape[-1]
    h = 1.0 / (length - 1)

    # Compute Laplacian using 5-point stencil finite difference
    u_padded = F.pad(u_pred, (1, 1, 1, 1), mode="constant", value=0)
    laplacian = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] + u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u_pred) / h**2

    # PDE residual: ∇²u + u = a
    pde_residual = laplacian + u_pred - a_pred

    # Remove boundary points
    pde_residual = pde_residual[..., 1:-1, 1:-1]
    return pde_residual


def get_burgers_residual(x_pred):
    """Calculate the PDE residual for Burgers' equation.

    The viscous Burgers equation is: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    where ν is the viscosity coefficient (0.01 in this case)

    Args:
        x_pred (torch.Tensor): Predicted solution tensor
            Should contain single channel for u field

    Returns:
        torch.Tensor: PDE residual
    """
    device = x_pred.device
    u_pred = x_pred[:, 1:2]

    # Initialize derivative kernels for finite difference
    # dx = 1.0 / (u_pred.shape[-1] - 1)
    dx = 1.0
    deriv_t = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * dx)
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * dx)

    # Calculate temporal derivative ∂u/∂t
    u_t = F.conv2d(u_pred, deriv_t, padding=(1, 0))

    # Calculate spatial derivative ∂u/∂x
    u_x = F.conv2d(u_pred, deriv_x, padding=(0, 1))

    # Calculate second spatial derivative ∂²u/∂x²
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))

    # Viscosity coefficient
    nu = 0.01

    # Full PDE residual: ∂u/∂t + u∂u/∂x - ν∂²u/∂x²
    pde_residual = u_t + u_pred * u_x - nu * u_xx

    # Remove boundary points
    pde_residual = pde_residual[..., 2:-2, 2:-2]
    return pde_residual


def get_ns_bounded_residual(x_pred):
    device = x_pred.device
    u_pred = x_pred[:, 1:2]
    dx = 1.0
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * dx)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * dx)
    grad_x = F.conv2d(u_pred, deriv_x, padding=(0, 1))
    grad_y = F.conv2d(u_pred, deriv_y, padding=(1, 0))
    pde_residual = grad_x + grad_y
    pde_residual = pde_residual[..., 1:-1, 1:-1]
    return pde_residual


def get_pde_residual(dataset_name):
    if dataset_name == "darcy":
        return get_darcy_residual
    elif dataset_name == "poisson":
        return get_poisson_residual
    elif dataset_name == "ns-nonbounded":
        return get_ns_nonbounded_residual
    elif dataset_name == "helmholtz":
        return get_helmholtz_residual
    elif dataset_name == "burgers":
        return get_burgers_residual
    elif dataset_name == "ns-bounded":
        return get_ns_bounded_residual
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")
