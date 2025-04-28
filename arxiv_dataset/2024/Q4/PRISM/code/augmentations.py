### ====== PRISM MRI AUGMENTATION FUNCTIONS ======

import numpy as np

# 1. Gamma Correction
def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an image.

    Args:
        image (np.ndarray): The input image to which gamma correction will be applied.
        gamma (float): The gamma value for the correction.

    
    Returns:
        np.ndarray: Image with gamma correction applied.
    """
    
    # Apply mask to remove background noise
    mask = image > 15
    image = image * mask

    image = image / 255.0  # Normalize pixel values (0-255 to 0-1)
    
    gamma_corrected = np.power(image, gamma)  # Apply gamma correction
    gamma_corrected = np.uint8(gamma_corrected * 255) # Convert back to 0-255 range

    # Prevent pixels from exceeding 255
    gamma_corrected = np.clip(gamma_corrected, 0, 255)
    
    return gamma_corrected * mask # Apply mask to the corrected image


# 2. Bias Fields
def apply_bias_fields(image: np.ndarray, order: int = 3, coeff: float = 0.2, mod: str = 'T2') -> np.ndarray:
    """
    Apply MRI bias field to an image using polynomial basis functions.

    Args:
        image (np.ndarray): The input image to which the bias field will be added.
        order (int): The order of the polynomial basis functions.
        coeff (float): Magnitude of the polynomial coefficients.
    
    Returns:
        np.ndarray: Image with bias field applied.
    """


    if not isinstance(order, int) or order < 0:
        raise ValueError('Order must be a positive integer.')
    
    if mod == 'PD':
        coeff /= 2
    
    # image = image / 255.0  # Normalize pixel values (0-255 to 0-1)
   
    # Apply mask to remove background noise
    mask = image > 15
    image = image * mask
    
    # Get the shape of the image
    shape = np.array(image.shape)
    half_shape = shape / 2

    # Create ranges for the bias field
    ranges = [np.arange(-n, n) + 0.5 for n in half_shape]
   
    # Initialize the bias field map
    bias_field = np.zeros(shape)
    
    # Create meshgrid
    meshes = np.asarray(np.meshgrid(*ranges, indexing='ij'))

    # Normalize mesh values
    for mesh in meshes:
        mesh_max = np.max(np.abs(mesh))
        if mesh_max > 0:
            mesh /= mesh_max
    
    x_mesh, y_mesh = meshes[:2]  # Assuming 2D image

    # Add polynomial terms to the bias field
    i = 0
    for x_order in range(order + 1):
        for y_order in range(order + 1 - x_order):
            coefficient = coeff * np.random.randn()  # Coefficients sampled from normal distribution
            # coefficient = coeff
            new_map = coefficient * (x_mesh ** x_order) * (y_mesh ** y_order)
            bias_field += new_map
            i += 1

    # Apply the bias field
    bias_field = np.exp(bias_field).astype(np.float32)
    biased_image = image * bias_field  # Apply the bias field to the image

    # biased_image = np.uint8(biased_image * 255) # Convert back to 0-255 range    
    
    # Prevent pixels from exceeding 255
    biased_image = np.clip(biased_image, 0, 255)
    
    return biased_image * mask


# 3. Gaussian Noise
def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 1) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image (np.ndarray): The input image to which the noise will be added.
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
    
    Returns:
        np.ndarray: Image with Gaussian noise added.
    """
    # image = image / 255.0  # Normalize pixel values (0-255 to 0-1)
    
    # Apply mask to remove background noise
    mask = image > 15
    image = image * mask
    
    noise = np.random.normal(mean, std, image.shape)  # Generate Gaussian noise
    noisy_image = image + noise  # Add the noise to the image
    
    # noisy_image = np.uint8(noisy_image * 255) # Convert back to 0-255 range    
   
    # Prevent pixels from exceeding 255
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image * mask