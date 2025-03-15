###break it down into multiple functions files later (categorically)
import sys
sys.path.append('/apps/python3/3.12.1/lib/python3.12/site-packages')
sys.path.append('g/data/jh2/ax8338/testvenv/bin/')
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import os
import glob
import time
import csv
import yt
import yt.units as u
import h5py
import matplotlib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interpolate
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from scipy.spatial import KDTree
from tqdm import tqdm
hydrogenMassFraction=0.76

#FUNCTIONS

#cartesian to cylindrical. TAKES ARRAY. gives out array phi,R,z. same for v.
def cart_to_cylind(c_array):
    return np.array((np.arctan2(c_array[:,1],c_array[:,0]),np.sqrt(c_array[:,0]**2+c_array[:,1]**2),c_array[:,2])).T

def cart_to_cylind_v(v_array,c_array):
    return np.array(((c_array[:,0]*v_array[:,1]-c_array[:,1]*v_array[:,0])/(np.sqrt(c_array[:,0]**2+c_array[:,1]**2)),
                     (c_array[:,0]*v_array[:,0]+c_array[:,1]*v_array[:,1])/(np.sqrt(c_array[:,0]**2+c_array[:,1]**2)),
                     v_array[:,2])).T

# takes array cylindrical coordinates in order phi,R,z
def cylind_to_cart(c_array):
    x=c_array[:,1]*np.cos(c_array[:,0])
    y=c_array[:,1]*np.sin(c_array[:,0])
    z=c_array[:,2]
    return np.array((x,y,z)).T

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


#function to read hdf5 files and give coordinates, ID, masses, velocities, ages and potential arrays for star particles

def read_stars(fname):
    os.chdir(fname)
    coord_array=vel_array=np.zeros((1,3))
    ID_array=m_array=age_array=pot_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        coord_array = np.append(coord_array,np.array(file0['PartType4']['Coordinates'])*1000 ,axis=0) #pc
        ID_array = np.append(ID_array, np.array(file0['PartType4']['ParticleIDs']),axis=0)
        m_array = np.append(m_array, np.array(file0['PartType4']['Masses'])*1e10 ,axis=0) #M_sun
        vel_array = np.append(vel_array, np.array(file0['PartType4']['Velocities']), axis=0) #km/s
        age_array = np.append(age_array, (curTime-np.array(file0['PartType4']['StellarFormationTime']))*1000, axis =0)    #Myr
        pot_array =np.append(pot_array,np.array(file0['PartType4']['Potential']),axis=0)  #(km/s)^2
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:], m_array[1:], vel_array[1:], age_array[1:], pot_array[1:]

#function to read hdf5 files and give coordinates and potential arrays for gas particles

def read_gas(fname):
    os.chdir(fname)
    coord_array=V_array=np.zeros((1,3))
    ID_array= pot_array=int_energy=m_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        coord_array = np.append(coord_array,np.array(file0['PartType0']['Coordinates'])*1000 ,axis=0) #pc
        ID_array = np.append(ID_array, np.array(file0['PartType0']['ParticleIDs']),axis=0)
        V_array = np.append(V_array, np.array(file0['PartType0']['Velocities']),axis=0)
        m_array = np.append(m_array, np.array(file0['PartType0']['Masses'])*1e10 ,axis=0) #M_sun
        pot_array =np.append(pot_array,np.array(file0['PartType0']['Potential']),axis=0)  #(km/s)^2
        int_energy = np.append(int_energy, np.array(file0['PartType0']['InternalEnergy']),axis=0)
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:],V_array[1:], m_array[1:],pot_array[1:],int_energy[1:]

####function to read initial stars
def read_oldstars(fname):
    os.chdir(fname)
    coord_array=V_array=np.zeros((1,3))
    ID_array= pot_array=m_array=age_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        age_array = np.append(age_array, (curTime-np.array(file0['PartType2']['StellarFormationTime']))*1000, axis =0)    #Myr
        # age_array = np.append(age_array, (curTime-np.array(file0['PartType3']['StellarFormationTime']))*1000, axis =0)
        
        coord_array = np.append(coord_array,np.array(file0['PartType2']['Coordinates'])*1000 ,axis=0) #pc
        # coord_array = np.append(coord_array,np.array(file0['PartType3']['Coordinates'])*1000 ,axis=0) #pc

        ID_array = np.append(ID_array, np.array(file0['PartType2']['ParticleIDs']),axis=0)
        # ID_array = np.append(ID_array, np.array(file0['PartType3']['ParticleIDs']),axis=0)

        V_array = np.append(V_array, np.array(file0['PartType2']['Velocities']),axis=0)
        # V_array = np.append(V_array, np.array(file0['PartType3']['Velocities']),axis=0)

        m_array = np.append(m_array, np.array(file0['PartType2']['Masses'])*1e10 ,axis=0) #M_sun
        # m_array = np.append(m_array, np.array(file0['PartType3']['Masses'])*1e10 ,axis=0) #M_sun

        pot_array =np.append(pot_array,np.array(file0['PartType2']['Potential']),axis=0)  #(km/s)^2
        # pot_array =np.append(pot_array,np.array(file0['PartType3']['Potential']),axis=0)  #(km/s)^2
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:],m_array[1:],V_array[1:],age_array[1:], pot_array[1:]

def read_darkmatter(fname):
    os.chdir(fname)
    coord_array=V_array=np.zeros((1,3))
    ID_array= pot_array=m_array=age_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        # age_array = np.append(age_array, (curTime-np.array(file0['PartType2']['StellarFormationTime']))*1000, axis =0)    #Myr
        # age_array = np.append(age_array, (curTime-np.array(file0['PartType3']['StellarFormationTime']))*1000, axis =0)
        
        coord_array = np.append(coord_array,np.array(file0['PartType1']['Coordinates'])*1000 ,axis=0) #pc
        # coord_array = np.append(coord_array,np.array(file0['PartType3']['Coordinates'])*1000 ,axis=0) #pc

        ID_array = np.append(ID_array, np.array(file0['PartType1']['ParticleIDs']),axis=0)
        # ID_array = np.append(ID_array, np.array(file0['PartType3']['ParticleIDs']),axis=0)

        V_array = np.append(V_array, np.array(file0['PartType1']['Velocities']),axis=0)
        # V_array = np.append(V_array, np.array(file0['PartType3']['Velocities']),axis=0)

        m_array = np.append(m_array, np.array(file0['PartType1']['Masses'])*1e10 ,axis=0) #M_sun
        # m_array = np.append(m_array, np.array(file0['PartType3']['Masses'])*1e10 ,axis=0) #M_sun

        pot_array =np.append(pot_array,np.array(file0['PartType1']['Potential']),axis=0)  #(km/s)^2
        # pot_array =np.append(pot_array,np.array(file0['PartType3']['Potential']),axis=0)  #(km/s)^2
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:],V_array[1:], m_array[1:],pot_array[1:]
    
def read_fitgridhdf5(snapshot_number,filepath):
    '''
    function to read the the hdf5 file that contains the fit_grid and fit_pot for all snapshots.
    takes in number of the snapshot and returns the fit_grid and fit_pot arrays
    '''
    snapshot_name = f'snapshot_{snapshot_number}'
    
    try:
        with h5py.File(filepath, 'r') as hdf:
            if f'{snapshot_name}' in hdf:
                grp = hdf[f'{snapshot_name}']
                try:
                # Extracting relevant data from the snapshot
                    fit_grid = grp['fit_grid'][:]
                    fit_pot = grp['fit_pot'][:]

                except Exception as e:
                    if "Unable to synchronously open object (object 'fit_grid' doesn't exist)" in str(e):
                        fit_grid = grp['grid'][:]
                        fit_pot = grp['potential'][:]
                
                print(f"Successfully read data for {snapshot_name}")
                return fit_grid,fit_pot
            else:
                raise ValueError(f"Snapshot {snapshot_name} does not exist in the HDF5 file.")
    
    except Exception as e:
        print(f"Error reading snapshot {snapshot_number}: {e}")
        return None

##derivative function- gives first and second derivative?
def der(y,x):
    h = np.diff(x)
    return (-y[4:]+8*y[3:-1]-8*y[1:-3]+y[:-4])/(12*h[1:-2]),(-y[4:]+16*y[3:-1]-30*y[2:-2]+16*y[1:-3]-y[:-4])/(12*h[1:-2]**2)

def der_diff(y,x):
    return np.diff(y)/np.diff(x),np.diff(np.diff(y)/np.diff(x))/np.diff(x)[1:]**2

def fourth_order_derivatives(x, y):
    n = len(y)
    if n < 5:
        raise ValueError("Array must have at least 5 elements for 4th order accuracy.")
    
    dy_dx = np.zeros(n)
    d2y_dx2 = np.zeros(n)

    # For internal points using 4th order central differences
    for i in range(2, n-2):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        h3 = x[i+2] - x[i+1]

        dy_dx[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (6 * (h2 + h1))

        h1 = x[i-1] - x[i-2]
        h2 = x[i] - x[i-1]
        h3 = x[i+1] - x[i]
        h4 = x[i+2] - x[i+1]

        term1 = -y[i+2] / (h3 * h4 * (h3 + h4))
        term2 = 16 * y[i+1] / (h2 * h3 * (h2 + h3))
        term3 = -30 * y[i] / (h1 * h2 * (h1 + h2))
        term4 = 16 * y[i-1] / (h1 * h2 * (h1 + h2))
        term5 = -y[i-2] / (h1 * h2 * (h1 + h2))

        d2y_dx2[i] = (term1 + term2 + term3 + term4 + term5) / (12)

    # For boundary points using lower order differences
    dy_dx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy_dx[1] = (y[2] - y[0]) / (x[2] - x[0])
    dy_dx[-2] = (y[-1] - y[-3]) / (x[-1] - x[-3])
    dy_dx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    d2y_dx2[0] = (y[2] - 2*y[1] + y[0]) / (x[2] - x[0])**2
    d2y_dx2[1] = (y[3] - 2*y[2] + y[1]) / (x[3] - x[1])**2
    d2y_dx2[-2] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-3])**2
    d2y_dx2[-1] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-2])**2

    return dy_dx, d2y_dx2

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

def weighted_median(data, weights):
    sorted_data = np.argsort(data)
    sorted_weights = weights[sorted_data]
    sorted_data = data[sorted_data]
    
    cumulative_weight = np.cumsum(sorted_weights)
    cutoff = cumulative_weight[-1] / 2.0
    
    return sorted_data[np.where(cumulative_weight >= cutoff)[0][0]]

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def _HII(field, data):
    T = data["PartType0","temperature"].in_units('K')
    HII = data["PartType0","GrackleHII"]
    HII[(T > 10766.8) & (T<10766.9)] = 0.745
    return HII

yt.add_field(
    name=("PartType0","HII"),
    function=_HII,
    sampling_type = 'local',
    units="dimensionless"
)

def meanWeight(field, data):
    mw = 1/(hydrogenMassFraction + data["PartType0","HII"]+(1-hydrogenMassFraction)/4 + data["PartType0","GrackleHeII"]/4 + data["PartType0","GrackleHeIII"]/2)
    return mw
yt.add_field(name=("PartType0","meanWeight"), function = meanWeight, sampling_type = 'local', units="dimensionless", force_override=True)
def Tg(field, data):
    return(data["PartType0","InternalEnergy"]*data["PartType0","meanWeight"]*u.mh*2/3/u.kb)
yt.add_field(name=("PartType0","Tg"), function=Tg, sampling_type = 'local', units="K", force_override=True)

##function to get CoM of cold gas given the filename
def CoM(fileName,mean_median=None):
    """takes filename then reads it using yt, finds the temperature and then takes gas particles colder than 10^4 K and finds their CoM.
    If mean_median=0 or None, it uses mean to find centre of mass. If it is 1, weighted median is used """
    hydrogenMassFraction=0.76
    bbox_lim = 10000 #kpc
    bbox = [[-bbox_lim,bbox_lim],
            [-bbox_lim,bbox_lim],
            [-bbox_lim,bbox_lim]]
    
    unit_base = {'UnitLength_in_cm'         : 3.08568e+21, #1    kpc
                 'UnitMass_in_g'            :   1.989e+43, #1e10 solarmass
                 'UnitVelocity_in_cm_per_s' :      100000} #1    km/s
    
    ds2 = yt.load(fileName,unit_base=unit_base, bounding_box=bbox)
    ad=ds2.all_data()
    T_g = np.array(ad['PartType0','Tg'])
    C_g = np.array(ad['PartType0','Coordinates'])*1000 #pc
    M_g = np.array(ad['PartType0','Masses'])*1e10 #M_sun
    if mean_median==None or mean_median==0:
        return np.average(C_g[T_g<1e4],axis=0,weights=M_g[T_g<1e4])
    if mean_median==1:
        return np.array((weighted_median(C_g[T_g<1e4][:,0],weights=M_g[T_g<1e4]),weighted_median(C_g[T_g<1e4][:,1],weights=M_g[T_g<1e4]),weighted_median(C_g[T_g<1e4][:,2],weights=M_g[T_g<1e4])))
    
# Define the function to read shift values from a CSV file
def read_shift_values_from_csv(file_path):
    shifts = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            shifts.append([float(x) for x in row])
    return shifts 
