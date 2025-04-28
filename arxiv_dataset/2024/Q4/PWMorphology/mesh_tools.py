import numpy as np
from skimage import measure
from scipy.interpolate import RegularGridInterpolator

def get_interp_function(d,var,method='linear'):
    """
    MM: Use RegularGridInterpolator to pass data to interpolating function for a given variable
    Parameters
    -----------
    d : dict
       athena data dict from read_data
    var: str
       name of variable to be interpolated
       
    Returns
    --------
    var_interp: an interpolating function that can be called with a tuple (phi,theta,r)
    """
    dph = np.gradient(d['x3v'])[0]
    two_pi = ( (d['x3v'][-1]-d['x3v'][0]+dph) /(2*np.pi) > 0.99 ) # boolean to determine if spans 2pi in phi
    
    if two_pi:
        x3v = np.append(d['x3v'][0]-dph,d['x3v'])
        x3v = np.append(x3v,x3v[-1]+dph)
        var_data = np.append([d[var][-1]],d[var],axis=0)
        var_data = np.append(var_data,[var_data[0]],axis=0)
    else:
        x3v = d['x3v']
        var_data = d[var]
        
    var_interp = RegularGridInterpolator((x3v,d['x2v'],d['x1v']),var_data,bounds_error=False,method=method)
    return var_interp

def cart_to_polar(x,y,z):
    """cartesian->polar conversion (matches 0<phi<2pi convention of Athena++)
    Parameters
    x, y, z
    Returns
    r, th, phi
    """
    r = np.sqrt(x**2 + y**2 +z**2)
    th = np.arccos(z/r)
    phi = np.arctan2(y,x)
    phi = np.where(phi>=0,phi,phi+2*np.pi)
    return r,th,phi


def get_uniform_meshgrid(lim,npoints=50,center=[0,0,0]):
    """
    MM: define a uniform spacing meshgrid of cartesian points around some center
    Parameters
    ------------
    lim: float
        limit of the box, such that the cartesian box extends +/- lim relative to the center
    center: (3,) array/list of floats
        x,y,z coordinates of the center of the grid
        
    Returns
    -----------
    xx,yy,zz,rr,tt,pp: (npoints,npoints,npoints) arrays of x,y,z and r,theta,phi coordinates of the points
    
    """
    x = np.linspace(-lim,lim,npoints)+center[0]
    y = np.linspace(-lim,lim,npoints)+center[1]
    z = np.linspace(-lim,lim,npoints)+center[2]

    yy,xx,zz = np.meshgrid(y,x,z)
    r,th,ph = cart_to_polar(xx,yy,zz)

    return xx,yy,zz,r,th,ph

def get_marching_cubes_mesh(xx,yy,zz,data,level,step_size=1):
    """ MM: call marching cubes algorithm to get a surface 
    Parameters
    ----------
    xx,yy,zz: (N,N,N) arrays of x,y,z coordinates
    data: (N,N,N) array of floats in which to find surface
    level: float value to construct surface at
    step_size: down sampling of 3D data
    dx: x,y,z spacing of data
    
    Returns
    -------
    verts : (V, 3) array
        Spatial coordinates for V unique mesh vertices. Coordinate order
        matches input `volume` (M, N, P). If ``allow_degenerate`` is set to
        True, then the presence of degenerate triangles in the mesh can make
        this array have duplicate vertices.
    faces : (F, 3) array
        Define triangular faces via referencing vertex indices from ``verts``.
        This algorithm specifically outputs triangles, so each face has
        exactly three indices.
    centroids : (F,3) array
        Centroids of triangular faces
    areas : (F,) array
        areas of triangular faces
    normals : (F,3) array
        face-centered normals 
        
        
    NOTE: 
    assumes uniform spacing in x,y,z
    
    """
    center = np.mean(xx[:,0,0]), np.mean(yy[0,:,0]), np.mean(zz[0,0,:])
    lim = xx[-1,0,0]
    dx = xx[1,0,0]-xx[0,0,0]
    
    verts, faces,_,_ =  measure.marching_cubes(data,
                                                       level=level,
                                                       step_size=step_size,
                                                       spacing=[dx,dx,dx])
    verts = verts-lim+center  # recenter the verticies
    
    centroids = mesh_get_centroids(verts,faces)
    areas     = mesh_get_areas(verts,faces)
    normals   = mesh_get_centroid_normals(verts,faces)
    
    return verts,faces, centroids, areas, normals

def mesh_get_areas(verts, faces):
    """
    MM: based on skiimage routine "measure.mesh_surface_area()"
    Compute surface area, given vertices & triangular faces
    Parameters
    ----------
    verts : (V, 3) array of floats
        Array containing (x, y, z) coordinates for V unique mesh vertices.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in `verts`
    Returns
    -------
    areas : (F,3) array of floats
        Surface areas of mesh triangles. Units now [coordinate units] ** 2.
    Notes
    -----
    The arguments expected by this function are the first two outputs from
    `skimage.measure.marching_cubes`. For unit correct output, ensure correct
    `spacing` was passed to `skimage.measure.marching_cubes`.
    This algorithm works properly only if the ``faces`` provided are all
    triangles.
    See Also
    --------
    skimage.measure.marching_cubes
    skimage.measure.marching_cubes_classic
    """
    # Fancy indexing to define two vector arrays from triangle vertices
    actual_verts = verts[faces]
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    del actual_verts

    # Area of triangle in 3D = 1/2 * Euclidean norm of cross product
    #return ((np.cross(a, b) ** 2).sum(axis=1) ** 0.5).sum() / 2.
    return ((np.cross(a, b) ** 2).sum(axis=1) ** 0.5) / 2.

def mesh_get_centroids(verts,faces):
    """
    MM: from verts, faces, return the coordinates of the centroids
    
    Parameters
    ----------
    verts : (V, 3) array of floats
        Array containing (x, y, z) coordinates for V unique mesh vertices.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in `verts`
    Returns
    -------
    centroids: (F, 3) array of floats
        array containing (x,y,z) coordinates of triangle centroids    
    """
    return verts[faces].sum(axis=1)/3.


def mesh_get_centroid_normals(verts,faces):
    """
    MM: from verts, faces, return the normals at the centroids
    
    Parameters
    ----------
    verts : (V, 3) array of floats
        Array containing (x, y, z) coordinates for V unique mesh vertices.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in `verts`
    Returns
    -------
    normals: (F, 3) array of floats
        array containing (x,y,z) components of normal to the plane defined by the vertices of the triangle.     
    """
       # Fancy indexing to define two vector arrays from triangle vertices
    actual_verts = verts[faces]
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    del actual_verts
    
    # cross product is perpendicular to two vectors connecting vertecies
    cp = np.cross(b, a)
    norms = (cp.T/np.linalg.norm(cp,axis=1)).T
    
    return (cp.T/np.linalg.norm(cp,axis=1)).T

def mesh_interpolate_at_xyzpoints(d,var,points):
    """
    MM: convience function to interpolate a variable to mesh points
    Parameters
    -----------
    d: athena++ data dict
    var: str variable name in, e.g. "rho"
    points: array of cartesian positions (eg vertices or centroids) (N,3) floats N x (x,y,z)
    """
    var_interp = get_interp_function(d,var)
    rp,thp,php = cart_to_polar(points[:,0],points[:,1],points[:,2])
    return var_interp( (php,thp,rp) )
