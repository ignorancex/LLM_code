import numpy as np

def rotation_matrix(angular_momentum_vector, reference_axis=[0.0, 1.0, 0.0]):
    """
    Calculate the rotation matrix to align an angular momentum vector with a reference axis.

    This function computes the rotation matrix that aligns the given angular momentum vector with the specified reference axis. The rotation matrix is computed using Rodrigues' rotation formula.

    Parameters:
    -----------
    angular_momentum_vector : array_like
        The angular momentum vector that needs to be aligned. It should be a 3-element array or list.
    reference_axis : array_like, optional
        The axis to align with. Default is [0.0, 1.0, 0.0].

    Returns:
    --------
    numpy.ndarray
        A 3x3 rotation matrix that aligns the `angular_momentum_vector` with the specified `reference_axis`.

    Notes:
    ------
    The function normalizes the input angular momentum vector and computes the rotation axis and angle required to align the vector with the reference axis. The rotation matrix is then constructed using Rodrigues' rotation formula, which is efficient and leverages matrix operations for performance.

    Rodrigues' rotation formula is used to compute the rotation matrix \( R \) as follows:
    \[
    R = I \cdot \cos(\theta) + \mathbf{k} \cdot \mathbf{k}^T \cdot (1 - \cos(\theta)) + [\mathbf{k}]_{\times} \cdot \sin(\theta)
    \]
    where:
    - \( I \) is the identity matrix.
    - \( \mathbf{k} \) is the unit vector along the rotation axis.
    - \( [\mathbf{k}]_{\times} \) is the skew-symmetric matrix of \( \mathbf{k} \).
    - \( \theta \) is the rotation angle.

    For more information on Rodrigues' rotation formula, see:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Documentation generated with assistance from ChatGPT.


    Examples:
    ---------
    >>> angular_momentum_vector = [1.0, 2.0, 3.0]
    >>> calc_faceon_matrix(angular_momentum_vector)
    array([[ 0.96035675, -0.07928651,  0.26726124],
           [-0.07928651,  0.84142698,  0.53452248],
           [-0.26726124, -0.53452248,  0.80178373]])
    """
    # Convert inputs to numpy arrays and normalize the angular momentum vector
    v = np.asarray(angular_momentum_vector)
    v /= np.linalg.norm(v)
    
    # Calculate the rotation axis (perpendicular vector) and angle
    k = np.cross(reference_axis, v)
    k /= np.linalg.norm(k)
    theta = np.arccos(np.dot(reference_axis, v))
    
    # Sine and cosine of the rotation angle
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Construct the rotation matrix using Rodrigues' rotation formula
    skew_symmetric = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    
    R = np.eye(3) * cos_theta + np.outer(k, k) * (1 - cos_theta) + skew_symmetric * sin_theta

    return R


def compute_density_grid(x_coords, y_coords, z_coords, masses, bin_edges, num_bins, scale_height):
    """
    Compute the volumetric density grid for stars based on their spatial and mass distributions.

    This function calculates a 2D grid density representation from star positions and masses. It bins stars into a 2D grid defined by `bin_edges` and computes the density based on their distance from a reference point and a scale parameter `scale_height`.

    Parameters:
    -----------
    x_coords : array_like
        Array of x-coordinates of the stars.
    y_coords : array_like
        Array of y-coordinates of the stars.
    z_coords : array_like
        Array of z-coordinates of the stars.
    masses : array_like
        Array of masses of the stars.
    bin_edges : array_like
        Array defining the bin edges for the x and y coordinates. The grid will be defined in the range [bin_edges[0], bin_edges[-1]].
    num_bins : int
        Number of bins (grid dimensions) in each direction. The grid will be (num_bins-1) x (num_bins-1) in size.
    scale_height : float
        Scale parameter used to limit the height from the center to consider for density calculations.

    Returns:
    --------
    numpy.ndarray
        A 2D array representing the density grid. Each element in the grid corresponds to the density of stars in that bin.

    Notes:
    ------
    - The function uses `np.digitize` to assign stars to columns based on their x and y coordinates.
    - For each grid cell, it calculates the median z-coordinate (`median_z`) and center the z-coordinates relative to this median.
    - The function then considers only those stars whose centered z-coordinate is within `scale_height` to compute the density.
    - The density for each grid cell is computed as the sum of masses of stars within the scale `scale_height`, divided by `scale_height`.


    Documentation generated with assistance from ChatGPT.

    Example:
    ---------
    >>> x_coords = [0.1, 0.5, 0.8, 1.0]
    >>> y_coords = [0.2, 0.4, 0.6, 0.9]
    >>> z_coords = [0.1, 0.3, 0.7, 0.6]
    >>> masses = [1.0, 1.5, 2.0, 0.5]
    >>> bin_edges = [0.0, 0.5, 1.0]
    >>> num_bins = 3
    >>> scale_height = 0.5
    >>> compute_density_grid(x_coords, y_coords, z_coords, masses, bin_edges, num_bins, scale_height)
    array([[2., 0.],
           [3., 4.]])
    """
    # Compute bin indices
    x_bin_indices = np.digitize(x_coords, bin_edges) - 1
    y_bin_indices = np.digitize(y_coords, bin_edges) - 1

    # Initialize density grid
    density_grid = np.zeros((num_bins - 1, num_bins - 1))
    # Calculate density grid
    for bin_x in range(num_bins - 1):
        for bin_y in range(num_bins - 1):
            cell_indices = np.where((x_bin_indices == bin_x) & (y_bin_indices == bin_y))[0]
            if cell_indices.size == 0:
                continue

            cell_z_coords = z_coords[cell_indices]
            cell_masses = masses[cell_indices]
            median_z = np.median(cell_z_coords)
            centered_z_coords = np.abs(cell_z_coords - median_z)
            within_scale_indices = centered_z_coords < scale_height
            density_grid[bin_x, bin_y] = np.sum(cell_masses[within_scale_indices]) / scale_height

    return density_grid


def apply_boundary_conditions(x, gal_x, boxsize, flag):
    half_boxsize = boxsize / 2.
    diff = x.max(axis=0) - x.min(axis=0)
    mask = diff > half_boxsize  # mask to find direction/dimension with difference larger than half of the box e.g. x,y,z=0,1,2                                                                                     

    x_condition = x > half_boxsize # mask to find particle coordinates with difference larger than half of the box                                                                                                  
    adjustment = np.where(x_condition & mask, boxsize, 0) # only adjust (e.g. subtract the boxsize) for particular  direction and coordinates                                                                       
    x -= adjustment
    if flag == 0: # correct galaxy center coordinates only once! We should not center twice for gas and star analysis blocks                                                                                        
        gal_x -= np.where(mask, boxsize, 0)
    return x
