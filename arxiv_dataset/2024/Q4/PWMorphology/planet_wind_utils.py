import numpy as np
from astropy.io import ascii
import athena_read as ar

def read_trackfile(fn,m1=0,m2=0):
    orb=ascii.read(fn)
    print ("reading orbit file for planet wind simulation...")
    if m1==0:
        m1 = orb['m1']
    if m2==0:
        m2 = orb['m2']

    orb['sep'] = np.sqrt(orb['x']**2 + orb['y']**2 + orb['z']**2)

    orb['r'] = np.array([orb['x'],orb['y'],orb['z']]).T
    orb['rhat'] = np.array([orb['x']/orb['sep'],orb['y']/orb['sep'],orb['z']/orb['sep']]).T

    orb['v'] = np.array([orb['vx'],orb['vy'],orb['vz']]).T
    orb['vmag'] = np.linalg.norm(orb['v'],axis=1)
    orb['vhat'] = np.array([orb['vx']/orb['vmag'],orb['vy']/orb['vmag'],orb['vz']/orb['vmag']]).T

    orb['xcom'] = m2*orb['x']/(m1+m2)
    orb['ycom'] = m2*orb['y']/(m1+m2)
    orb['zcom'] = m2*orb['z']/(m1+m2)
    
    orb['vxcom'] = m2*orb['vx']/(m1+m2)
    orb['vycom'] = m2*orb['vy']/(m1+m2)
    orb['vzcom'] = m2*orb['vz']/(m1+m2)
    
    orb['rcom'] = np.array([orb['xcom'],orb['ycom'],orb['zcom']]).T
    orb['vcom'] = np.array([orb['vxcom'],orb['vycom'],orb['vzcom']]).T
    
    return orb


def read_data(fn,orb,
              m1=0,m2=0,rsoft2=0.1,level=0,
              get_cartesian=True,get_cartesian_vel=True,
             x1_min=None,x1_max=None,
             x2_min=None,x2_max=None,
             x3_min=None,x3_max=None,
              gamma=5./3.,
              pole_dir=2):
    """ Read spherical data and reconstruct cartesian mesh for analysis/plotting """
    
    print ("read_data...reading file",fn)
    
    
    d = ar.athdf(fn,level=level,subsample=True,
                 x1_min=x1_min,x1_max=x1_max,
                 x2_min=x2_min,x2_max=x2_max,
                 x3_min=x3_min,x3_max=x3_max,
                     return_levels=True) # approximate arrays by subsampling if level < max
    print (" ...file read, constructing arrays")
    print (" ...gamma=",gamma)
    
    # current time
    t = d['Time']
    # get properties of orbit
    rcom,vcom = rcom_vcom(orb,t)

    if m1==0:
        m1 = np.interp(t,orb['time'],orb['m1'])
    if m2==0:
        m2 = np.interp(t,orb['time'],orb['m2'])

    data_shape = (len(d['x3v']),len(d['x2v']),len(d['x1v']))
   
    d['gx1v']=np.broadcast_to(d['x1v'],(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    d['gx2v']=np.swapaxes(np.broadcast_to(d['x2v'],(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    d['gx3v']=np.swapaxes(np.broadcast_to(d['x3v'],(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    ####
    # GET THE VOLUME 
    ####
    
    ## dr, dth, dph
    d1 = d['x1f'][1:] - d['x1f'][:-1]
    d2 = d['x2f'][1:] - d['x2f'][:-1]
    d3 = d['x3f'][1:] - d['x3f'][:-1]
    
    gd1=np.broadcast_to(d1,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    gd2=np.swapaxes(np.broadcast_to(d2,(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    gd3=np.swapaxes(np.broadcast_to(d3,(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    
    # AREA / VOLUME 
    sin_th = np.sin(d['gx2v'])
    d['dA'] = d['gx1v']**2 * sin_th * gd2*gd3
    d['dvol'] = d['dA'] * gd1
    
    # free up d1,d2,d3
    del d1,d2,d3
    del gd1,gd2,gd3
    
    
    ### 
    # CARTESIAN VALUES
    ###
    if(get_cartesian):
        print ("...getting cartesian arrays...")
        # angles
        cos_th = np.cos(d['gx2v'])
        sin_ph = np.sin(d['gx3v'])
        cos_ph = np.cos(d['gx3v']) 
        
        # cartesian coordinates
        if(pole_dir==2):
            d['x'] = d['gx1v'] * sin_th * cos_ph 
            d['y'] = d['gx1v'] * sin_th * sin_ph 
            d['z'] = d['gx1v'] * cos_th
        if(pole_dir==0):
            d['y'] = d['gx1v'] * sin_th * cos_ph 
            d['z'] = d['gx1v'] * sin_th * sin_ph 
            d['x'] = d['gx1v'] * cos_th

        if(get_cartesian_vel):
            # cartesian velocities
            if(pole_dir==2):
                d['vx'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
                d['vy'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
                d['vz'] = cos_th*d['vel1'] - sin_th*d['vel2']  
            if(pole_dir==0):
                d['vy'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
                d['vz'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
                d['vx'] = cos_th*d['vel1'] - sin_th*d['vel2'] 
            
        del cos_th, sin_th, cos_ph, sin_ph
    
    return d



def get_midplane_theta(myfile,level=0):
    dblank=ar.athdf(myfile,level=level,quantities=[],subsample=True)

    # get closest to midplane value
    return dblank['x2v'][ np.argmin(np.abs(dblank['x2v']-np.pi/2.) ) ]


def get_plot_array_midplane(arr):
    return np.append(arr,[arr[0]],axis=0)


def rcom_vcom(orb,t):
    """pass a pm_trackfile.dat that has been read, time t"""
    rcom =  np.array([np.interp(t,orb['time'],orb['rcom'][:,0]),
                  np.interp(t,orb['time'],orb['rcom'][:,1]),
                  np.interp(t,orb['time'],orb['rcom'][:,2])])
    vcom =  np.array([np.interp(t,orb['time'],orb['vcom'][:,0]),
                  np.interp(t,orb['time'],orb['vcom'][:,1]),
                  np.interp(t,orb['time'],orb['vcom'][:,2])])
    
    return rcom,vcom

def pos_secondary(orb,t):
    x2 = np.interp(t,orb['time'],orb['x'])
    y2 = np.interp(t,orb['time'],orb['y'])
    z2 = np.interp(t,orb['time'],orb['z'])
    return x2,y2,z2


### Ray tracing
from scipy.interpolate import RegularGridInterpolator

def get_interp_function(d,var):
    dph = np.gradient(d['x3v'])[0]
    x3v = np.append(d['x3v'][0]-dph,d['x3v'])
    x3v = np.append(x3v,x3v[-1]+dph)
    
    var_data = np.append([d[var][-1]],d[var],axis=0)
    var_data = np.append(var_data,[var_data[0]],axis=0)
    
    var_interp = RegularGridInterpolator((x3v,d['x2v'],d['x1v']),var_data,bounds_error=False)
    return var_interp

def cart_to_polar(x,y,z):
    r = np.sqrt(x**2 + y**2 +z**2)
    th = np.arccos(z/r)
    phi = np.arctan2(y,x)
    return phi,th,r

def get_ray(YZoffset=(0,0),star_pos=(-1.e11,0,0),length=1.e11,phase=0.0,inclination=0.0,rstar=7.e10,npoints=100):
    """ returns a ray in the intrinsic x,y,z coordinates of the mesh.
    After rotations, The viewer is along the -X axis, and the stellar surface is projected on the Y-Z plane"""
    xstar,ystar,zstar=star_pos
    Ystar,Zstar = YZoffset
    Xstar = -np.sqrt(rstar**2-Ystar**2-Zstar**2)
    
    # define dir vector (has length 1.0)
    dz = np.sin(inclination)
    dy = -np.cos(inclination)*np.sin(phase)
    dx = -np.cos(inclination)*np.cos(phase)
    dir_vec = np.array([dx,dy,dz])
    
    
    # define origin
    # rotate around Y then z to go from XYZ to xyz
    rotz = np.array([[np.cos(-phase),-np.sin(-phase),0.0],
                     [np.sin(-phase),np.cos(-phase),0.0],
                     [0.0,0.0,1.0]])
    rotY = np.array([[np.cos(-inclination),0.0,np.sin(-inclination)],
                     [0.0,1.0,0.0],
                     [-np.sin(-inclination),0.0,np.cos(-inclination)]])
    origin = np.matmul( np.matmul(np.array([Xstar,Ystar,Zstar]),rotY), rotz) + np.array([xstar,ystar,zstar])
    print (origin)
    
    
    # ray 
    ray={}
    ray['l'] = np.linspace(0,length,npoints)
    ray['x'] = np.flipud(ray['l']*dx) + origin[0] 
    ray['y'] = np.flipud(ray['l']*dy) + origin[1]
    ray['z'] = np.flipud(ray['l']*dz) + origin[2]
    
    # spherical polar
    ray['phi'],ray['theta'],ray['r'] = cart_to_polar(ray['x'],ray['y'],ray['z'])
    
    return ray
