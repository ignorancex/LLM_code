import numpy as np
from galpy.util import bovy_coords, _rotate_to_arbitrary_vector
from scipy.optimize import newton
from scipy.special import erfinv

_R0 = 8.125 # Gravity Collab.
_z0 = 0.02 #bennett and bovy

#utilities for transformations etc

def transform_aby(xyz,alpha,beta,gamma):
    """
    Transform xyz coordinates by rotation around x-axis (alpha), transformed y-axis (beta) and twice transformed z-axis (gamma)
    """
    Rx = np.zeros([3,3])
    Ry = np.zeros([3,3])
    Rz = np.zeros([3,3])
    Rx[0,0] = 1
    Rx[1] = [0, np.cos(alpha), -np.sin(alpha)]
    Rx[2] = [0, np.sin(alpha), np.cos(alpha)]
    Ry[0] = [np.cos(beta), 0, np.sin(beta)]
    Ry[1,1] = 1
    Ry[2] = [-np.sin(beta), 0, np.cos(beta)]
    Rz[0] = [np.cos(gamma), -np.sin(gamma), 0]
    Rz[1] = [np.sin(gamma), np.cos(gamma), 0]
    Rz[2,2] = 1
    R = np.matmul(Rx,np.matmul(Ry,Rz))
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(R, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', R, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

def transform_zvecpa(xyz,zvec,pa):
    """
    transform coordinates using the axis-angle method
    """
    pa_rot= np.array([[np.cos(pa),np.sin(pa),0.],
                         [-np.sin(pa),np.cos(pa),0.],
                         [0.,0.,1.]])

    zvec/= np.sqrt(np.sum(zvec**2.))
    zvec_rot= _rotate_to_arbitrary_vector(np.array([[0.,0.,1.]]),zvec,inv=True)[0]
    trot= np.dot(pa_rot,zvec_rot)
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(trot, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', trot, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

#density models

def spherical(R,phi,z,params=[2.5,]):
    """
    general spherical power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,]
    OUTPUT
        density at R, phi, z (normalised to 1 at R=8 kpc and z=0.02)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2+z**2)**(-params[0])
    dens = dens/(np.sqrt(_R0**2+_z0**2)**(-params[0]))
    if grid:
        dens = dens.reshape(dim)
    return dens

def axisymmetric(R,phi,z,params=[2.5,1.]):
    """
    general axisymmetric power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,c]
    OUTPUT
        density at R, phi, z (normalised to 1 at R=8 kpc and z=0.02)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2+z**2/params[1]**2)**-params[0]
    dens = dens/np.sqrt(_R0**2+_z0**2/params[1]**2)**-params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_norot(R,phi,z,params=[2.5,1.,1.]):
    """
    general triaxial power-law density model (no rotation)
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,b,c]
    OUTPUT
        density at R, phi, z
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)**-params[0]
    dens = dens/np.sqrt(_R0**2+_z0**2/params[2]**2)**-params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_single_angle_aby(R,phi,z,params=[2.,0.5,0.5,0.5,0.5,0.5]):
    """
    triaxial power-law density model, rotated using alpha,beta,gamma scheme
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,p,q,A,beta,gamma]
    OUTPUT
        density at R, phi, z
    """
    grid = False
    alpha = 0.9*np.pi*params[3]+0.05*np.pi-np.pi/2.
    beta = 0.9*np.pi*params[4]+0.05*np.pi-np.pi/2.
    gamma = 0.9*np.pi*params[5]+0.05*np.pi-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_aby(np.dstack([x,y,z])[0], alpha,beta,gamma)
    xsun, ysun, zsun = transform_aby([_R0, 0., _z0],alpha,beta,gamma)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_single_angle_zvecpa(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.]):
    """
    triaxial power-law density model rotated using zvec,pa scheme
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,p,q,theta,phi,pa]
    OUTPUT
        density at R, phi, z
    """
    grid = False
    theta = params[3]*2*np.pi
    tz = (params[4]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[5]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def justexpdisk(R,phi,z,params=[1/1.8,1/0.8]):
    """
    vanilla exponential disk
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [1/hr, 1/hz]
    OUTPUT
        density at R, phi, z
    """
    hr = params[0]
    hz = params[1]
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    return diskdens/diskdens_sun


def triaxial_single_angle_zvecpa_plusexpdisk(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.,0.01],split=False):
    """
    triaxial power law, with zvec,pa rotation and expdisk contaminant
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,p,q,theta,phi,pa,fdisc]
    OUTPUT
        density at R, phi, z
    """
    original_z = np.copy(z)
    grid = False
    theta = params[3]*2*np.pi
    tz = (params[4]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[5]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[6])*dens/sundens, (params[6])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[6])*dens/sundens+(params[6]*diskdens/diskdens_sun)
        #dens = ((1-params[6])*dens+params[6]*diskdens)/((1-params[6])*sundens+params[6]*diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens

def triaxial_single_cutoff_zvecpa_plusexpdisk(R,phi,z,params=[2.,1.,0.5,0.5,0.,0.,0.,0.01],split=False):
    """
    triaxial power law, with zvec,pa rotation and expdisk contaminant
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,beta,p,q,theta,phi,pa,fdisc]
    OUTPUT
        density at R, phi, z
    """
    original_z = np.copy(z)
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    dens = (r_e)**(-params[0])*np.exp(-params[1]*r_e)
    sundens = (r_e_sun)**(-params[0])*np.exp(-params[1]*r_e_sun)
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens

def triaxial_broken_angle_zvecpa_plusexpdisk(R,phi,z,params=[2.,3.,5.,0.5,0.5,0.,0.,0.,0.01],split=False):
    """
    triaxial broken power law, with zvec,pa rotation and expdisk contaminant
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha_in,alpha_out,beta,p,q,theta,phi,pa,fdisc]
    OUTPUT
        density at R, phi, z
    """
    original_z = np.copy(z)
    grid = False
    theta = params[5]*2*np.pi
    tz = (params[6]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[7]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[3]**2+zsun**2/params[4]**2)
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    dens = np.zeros(len(r_e))
    dens[r_e < params[2]] = (r_e[r_e < params[2]])**(-params[0])
    dens[r_e > params[2]] = (params[2])**(params[1]-params[0])*(r_e[r_e > params[2]])**(-params[1])
    if params[2] < r_e_sun:
        sundens = (params[2])**(params[1]-params[0])*(r_e_sun)**(-params[1])
    else:
        sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[8])*dens/sundens, (params[8])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[8])*dens/sundens+(params[8]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens


def triaxial_einasto_zvecpa(R,phi,z,params=[10.,3.,0.8,0.8,0.,0.99,0.]):
    """
    triaxial einasto profile, with zvec,pa rotation
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [n,r_eb,p,q,theta,phi,pa]
    OUTPUT
        density at R, phi, z
    """
    grid = False
    r_eb = params[0]
    n = params[1]
    p = params[2]
    q = params[3]
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
    dn = 3*n - 1./3. + 0.0079/n
    dens = np.exp(-dn*((r_e/r_eb)**(1/n)-1))
    sundens = np.exp(-dn*((r_e_sun/r_eb)**(1/n)-1))
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_einasto_zvecpa_plusexpdisk(R,phi,z,params=[10.,3.,0.8,0.8,0.,0.99,0.,0.], split=False):
    """
    triaxial einasto profile, with zvec,pa rotation plus expdisk contaminant
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [n,r_eb,p,q,theta,phi,pa,fdisc]
    OUTPUT
        density at R, phi, z
    """
    grid = False
    original_z = np.copy(z)
    r_eb = params[0]
    n = params[1]
    p = params[2]
    q = params[3]
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
    dn = 3.*n - 1./3. + 0.0079/n
    dens = np.exp(-dn*((r_e/r_eb)**(1/n)-1))
    sundens = np.exp(-dn*((r_e_sun/r_eb)**(1/n)-1))
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens
    return dens
