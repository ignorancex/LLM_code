import numpy as np

#cartesian to cylindrical. TAKES ARRAY. gives out array phi,R,z. 
def cart_to_cylind(c_array):
    return np.array((np.arctan2(c_array[:,1],c_array[:,0]), np.sqrt(c_array[:,0]**2 + c_array[:,1]**2), c_array[:,2])).T

#same as coordinate but for velocity
def cart_to_cylind_v(v_array, c_array):
    return np.array(((c_array[:,0] * v_array[:,1] - c_array[:,1] * v_array[:,0]) / np.sqrt(c_array[:,0]**2 + c_array[:,1]**2),
                     (c_array[:,0] * v_array[:,0] + c_array[:,1] * v_array[:,1]) / np.sqrt(c_array[:,0]**2 + c_array[:,1]**2),
                     v_array[:,2])).T

# takes array cylindrical coordinates in order phi,R,z and returns x,y,z
def cylind_to_cart(c_array):
    x = c_array[:,1] * np.cos(c_array[:,0])
    y = c_array[:,1] * np.sin(c_array[:,0])
    z = c_array[:,2]
    return np.array((x, y, z)).T
