import numpy as np

def define_rectangular_grid_from_arrays(x, y, z, q,num_points1,num_points2,num_points3,max1=None,max2=None,max3=None,min1=None,min2=None,min3=None):
    """
    Define a rectangular grid based on the extremities of separate coordinate arrays in 3D.
    
    Parameters:
        x (array-like): Array of x coordinates.
        y (array-like): Array of y coordinates.
        z (array-like): Array of z coordinates.
        num_points1/2/3 (int): Number of points in each dimension of the grid.
        max1/2/3: if given use this instead of percentile to get maximum value of the dimension grid

    Returns:
        tuple: A tuple containing three 2D arrays representing the grid coordinates (X, Y, Z).
    """
    # Extract min and max values for each dimension
    min_x, max_x = np.percentile(x,100-q),np.percentile(x,q)
    min_y, max_y = np.percentile(y,100-q),np.percentile(y,q)
    min_z, max_z =np.percentile(z,100-q),np.percentile(z,q)

    
 # Override with explicit min and max values if provided
    min_x = min1 if min1 is not None else min_x
    max_x = max1 if max1 is not None else max_x
    min_y = min2 if min2 is not None else min_y
    max_y = max2 if max2 is not None else max_y
    min_z = min3 if min3 is not None else min_z
    max_z = max3 if max3 is not None else max_z

    
    # Create linear spaces for each dimension
    x_space = np.linspace(min_x, max_x, num=num_points1)

    y_space = np.linspace(min_y, max_y, num=num_points2)

    z_space = np.linspace(min_z, max_z, num=num_points3)

    # Create a meshgrid from the linear spaces
    X, Y, Z = np.meshgrid(x_space, y_space, z_space, indexing='ij')
        
    return X,Y,Z


#function for reducing the resolution of the fit_grid in whatever dimesion
def reduce_res(fit_grid, fit_pot, column, new_res, grid_R_shape,grid_z_shape):
    '''Takes average while reducing resolution
    Put column=0 for R dimension resolution decrease and column=1 for z dimension res decrease. 
    grid_R_shape and grid_z_shape is used to get earlier number of grid points in that dimension. 
    returns fit_grid_new, fit_grid_pot, new resolution in R and in z'''
    dim_max = fit_grid[-1, column]
    dim_min = fit_grid[0, column]
    
    if column == 0:
        # reduction in R resolution
        c = int(grid_R_shape / ((dim_max - dim_min) / new_res))
        z_shape = grid_z_shape
        R_shape = grid_R_shape // c + (grid_R_shape % c > 0)
        
        fit_grid_new = np.zeros((z_shape * R_shape, 2))
        
        # Averaging over the R dimension
        fit_grid_reshaped = fit_grid[:, 0].reshape(z_shape, grid_R_shape)
        fit_grid_new[:, 0] = np.median(fit_grid_reshaped[:, :c * R_shape].reshape(z_shape, R_shape, c), axis=2).flatten()
        
        fit_grid_new[:, 1] = fit_grid[:, 1].reshape(z_shape, grid_R_shape).T[:R_shape].T.flatten()
        fit_pot_new = np.median(fit_pot.reshape(z_shape, grid_R_shape)[:, :c * R_shape].reshape(z_shape, R_shape, c), axis=2).flatten()
        
    elif column == 1:
        # reduction in z resolution
        c = int(grid_z_shape / ((dim_max - dim_min) / new_res))
        
        R_shape = grid_R_shape
        z_shape = grid_z_shape // c + (grid_z_shape % c > 0)
        
        fit_grid_new = np.zeros((R_shape * z_shape, 2))
        
        # Averaging over the z dimension
        fit_grid_reshaped = fit_grid[:, 1].reshape(grid_z_shape, grid_R_shape)
        fit_grid_new[:, 1] = np.median(fit_grid_reshaped[:c * z_shape].reshape(z_shape, c, R_shape), axis=1).flatten()
        
        fit_grid_new[:, 0] = fit_grid[:, 0].reshape(grid_z_shape, grid_R_shape)[:z_shape].flatten()
        fit_pot_new = np.median(fit_pot.reshape(grid_z_shape, grid_R_shape)[:c * z_shape].reshape(z_shape, c, R_shape), axis=1).flatten()
        
    else:
        raise ValueError("column can only be 0 or 1")
        
    z_res = fit_grid_new[:, 1].reshape((z_shape, R_shape))[:, 1][1] - fit_grid_new[:, 1].reshape((z_shape, R_shape))[:, 1][0]
    R_res = fit_grid_new[:, 0].reshape((z_shape, R_shape))[0][1] - fit_grid_new[:, 0].reshape((z_shape, R_shape))[0][0]
    
    return fit_grid_new, fit_pot_new, R_res, z_res



