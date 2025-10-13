import numpy as np
import torch
import healpy as hp
import astropy.units as u

def apply_stdnorm(maps):
    """
    Apply standard normalization per channel: (x - mean) / std, channel-wise.

    Parameters:
    - maps: np.ndarray, shape (..., C) where C is the number of channels

    Returns:
    - np.ndarray with same shape, normalized per channel
    """
    maps = maps.copy()
    for c in range(maps.shape[-1]):
        channel = maps[..., c]
        maps[..., c] = (channel - np.mean(channel)) / np.std(channel)
    return maps
    
def stats(maps):
    return np.min(maps), np.max(maps), np.mean(maps), np.std(maps)
    
def renormalize_dm_maps(dm_maps, train_maps, variance_scaling=True):
    """
    Renormalizes diffusion model maps to match the mean and variance of training maps.

    Parameters:
    - dm_maps: np.ndarray, shape (N, C, H, W)
    - train_maps: np.ndarray, shape (N, H, W, C)
    - variance_scaling: bool, if True, match variance in addition to mean

    Returns:
    - np.ndarray of shape (N, C, H, W), renormalized
    """
    dm_maps = np.transpose(dm_maps, (0, 2, 3, 1)).copy()
    num_channels = train_maps.shape[-1]

    for i in range(num_channels):
        tr_min = np.min(train_maps[:, :, :, i])
        tr_max = np.max(train_maps[:, :, :, i])

        dm_maps[:,:,:, i] = dm_maps[:,:,:, i] * (tr_max - tr_min) + tr_min

        if variance_scaling:
            dm_mean = np.mean(dm_maps[:,:,:, i])
            dm_std = np.std(dm_maps[:,:,:, i])
            tr_mean = np.mean(train_maps[:, :, :, i])
            tr_std = np.std(train_maps[:, :, :, i])
            dm_maps[:,:,:, i] = (dm_maps[:,:,:, i] - dm_mean) * (tr_std / dm_std) + tr_mean

    return dm_maps

@u.quantity_input
def get_patch_centers(gal_cut: u.deg, step_size: u.deg):
    """ Function to get the centers of the various patches to be cut out.

    Parameters
    ----------
    gal_cut: float
        We will miss out the region +/- `gal_cut` in Galactic latitude, measured
        in degrees.
    step_size: float
        Stepping distance in Galactic longitude, measured in degrees, between 
        patches.

    Returns
    -------
    list(tuple(float))
        List of two-element tuples containing the longitude and latitude.
    """
    gal_cut = gal_cut.to(u.deg)
    step_size = step_size.to(u.deg)
    assert gal_cut.unit == u.deg
    assert step_size.unit == u.deg
    southern_lat_range = np.arange(-90, (-gal_cut-step_size).value, step_size.value) * u.deg
    northern_lat_range = np.arange((gal_cut + step_size).value, 90, step_size.value) * u.deg
    lat_range = np.concatenate((southern_lat_range, northern_lat_range))

    centers = []
    for t in lat_range:
        step = step_size.value / np.cos(t.to(u.rad).value)
        for i in np.arange(0, 360, step):
            centers.append((i * u.deg, t))
    return centers

class FlatCutter(object):
    """ Object to control the extraction of flat patches from a given HEALPix 
    map.

    Object is initialized with parameters defining the geometry of the patch:
    its length in degrees, and the number of pixels in each direction. 
    The `rotate_and_interpolate` method defines a grid centered at (0, 0)
    of dimensions corresponding to `xlen`, `ylen`, and rotates it to the 
    point (lon, lat). The value of the map at the resulting grid of longitudes 
    and latitudes is then determined by interpolation. 
    """
    @u.quantity_input
    def __init__(self, ang_x: u.deg, ang_y: u.deg, xres, yres):
        assert type(xres) is int
        assert type(yres) is int
        self.xres = xres
        self.yres = yres

        self.ang_x = ang_x
        self.ang_y = ang_y
        
        # get grid of unit vectors corresponding to flat patch around
        # pole (z = 1). For this we use ang_x in radians, as is appropriate
        # for the implicit small-angle approximation 
        self.xarr = np.linspace(- self.ang_x.to(u.rad).value / 2., 
                                self.ang_x.to(u.rad).value / 2., xres)
        self.yarr = np.linspace(- self.ang_y.to(u.rad).value / 2., 
                                self.ang_y.to(u.rad).value / 2., yres)

        xgrid, ygrid = np.meshgrid(self.xarr, self.yarr)
        xgrid = xgrid.ravel()[None, :]
        ygrid = ygrid.ravel()[None, :]
        zgrid = np.ones_like(ygrid)
        
        # vectors corresponding to cartesian grid around poll
        self.vecs = np.concatenate((xgrid, ygrid, zgrid)).T

        # get the latitude (*not colatitude*) and longitude in degrees
        # of the cartesian grid points around the pole. 
        self.lons, self.lats = hp.vec2ang(self.vecs, lonlat=True)
        self.lats *= u.deg
        self.lons *= u.deg
        return
    
    @u.quantity_input
    def rotate_to_pole_and_interpolate(self, lon: u.deg, lat: u.deg, ma):
        """ Method to rotate the grid at (0, 0) to `rot=(lon, lat)`, and sample
        the map at the grid points by interpolation.

        Parameters
        ----------
        lat, lon: float
            Latitude (*not* colatitude) and longitude of point to be rotated
            to the North pole, in degrees.
        ma: ndarray
            Healpix map from which the interpolation is to be made.
        """
        if hp.pixelfunc.maptype(ma) == 0:  # a single map is converted to a list
            ma = [ma]
        # define a rotation object in terms of the theta_rot and phi_rot angles.
        # This returns a rotator object that can be applied to rotate a given
        # vector by this angle. Since we are interested in rotating some patch
        # to the pole, we actually want to apply the *inverse* rotation operator
        # to the vectors self.co_lats, self.lons.
        lon = lon.to(u.deg)
        lat = lat.to(u.deg)
        rotator = hp.Rotator(rot=[lon.value, lat.value - 90.], deg=True)
        self.inv_lon_grid, self.inv_lat_grid = rotator.I(self.lons.to(u.deg).value, self.lats.to(u.deg).value, lonlat=True)
        # Interpolate the original map to the pixels centers in the new ref frame
        m_rot = [hp.get_interp_val(each, self.inv_lon_grid, self.inv_lat_grid, lonlat=True) for each in ma]

        # Rotate polarization
        if len(m_rot) > 1:
            # Create a complex map from QU  and apply the rotation in psi due to the rotation
            # Slice from the end of the array so that it works both for QU and IQU
            m_rot[-2], m_rot[-1] = spin2rot(m_rot[-2], m_rot[-1], rotator.angle_ref(self.inv_lon_grid, self.inv_lat_grid, lonlat=True))
            m_rot[-2], m_rot[-1] = spin2rot(m_rot[-2], m_rot[-1], self.lons.to(u.rad).value)
        else:
            m_rot = m_rot[0]
        return np.moveaxis(np.array(m_rot).reshape(-1, self.xres, self.yres), 0, -1)
    

def replace_zeros_with_neighbor_avg(hp_map):
    """
    Replace zero values in a HEALPix map with the average of their non-zero neighbors.

    This function scans through the provided HEALPix map and replaces each zero-value pixel
    with the average value of its neighboring pixels that are non-zero. If no valid non-zero
    neighbors are found, the zero value may be replaced with a predefined value or left unchanged,
    depending on the desired behavior for such cases.

    Parameters:
    hp_map (np.ndarray): A 1D numpy array representing a HEALPix map with NSIDE resolution.

    Returns:
    np.ndarray: The modified HEALPix map with zero values replaced by the average of their non-zero neighbors.
    """
    nside = hp.get_nside(hp_map)
    zeros_indices = np.where(hp_map == 0)[0]
    
    for idx in zeros_indices:
        neighbors = hp.get_all_neighbours(nside, idx)
        # Filter out neighbors with invalid indices (-1) and zero values in the map
        valid_neighbors = neighbors[(neighbors >= 0) & (hp_map[neighbors] != 0)]
        
        if len(valid_neighbors) > 0:
            # Calculate the average of valid, non-zero neighbors
            average_value = np.mean(hp_map[valid_neighbors])
            hp_map[idx] = average_value
        else:
            # If no valid neighbors are found, handle this edge case appropriately
            # Here, you might set it to a default value or leave it unchanged
            hp_map[idx] = 0 # Or any other appropriate value
            
    return hp_map

def split_data_to_tensors(data, train_size=0.7, val_size=0.15, test_size=0.15,seed=42):
    """
    Splits the data into training, validation, and test datasets based on provided sizes,
    rearranges the data for PyTorch, and converts them into tensors.
    
    Parameters:
        data (np.array): The input data with shape (b, h, w, c).
        train_size (float): Fraction of the data to be used for training.
        val_size (float): Fraction of the data to be used for validation.
        test_size (float): Fraction of the data to be used for testing.
    
    Returns:
        tuple: A tuple containing training, validation, and test datasets as PyTorch tensors.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("The sum of train_size, val_size, and test_size should be 1.")
    rng = np.random.default_rng(seed)
    
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)

    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_set = torch.tensor(data[train_indices].transpose(0, 3, 1, 2), dtype=torch.float32)
    val_set = torch.tensor(data[val_indices].transpose(0, 3, 1, 2), dtype=torch.float32)
    test_set = torch.tensor(data[test_indices].transpose(0, 3, 1, 2), dtype=torch.float32)

    return train_set, val_set, test_set