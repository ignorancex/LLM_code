import numpy as np

def Hessian_approx(positions, gradients, H=None, mode='bfgs'):
    if mode == 'bfgs':
        return BFGS_hessian_approx(positions, gradients, H)
    elif mode == 'fisher':
        raise NotImplementedError
    elif mode == 'gsm':
        raise NotImplementedError


def BFGS_hessian_approx(positions, gradients, H=None):
    '''Returns Hessian approximation (B matrix) & number of points used to estimate it'''
    d = positions.shape[1]
    npos = positions.shape[0]
    #if H is None: 
    #    H = np.eye(d) # initial hessian. Moved to initilization below which is better
   
    nabla = gradients[0]
    x = positions[0]
    
    it = 0 
    points_used = 0 
    for i in range(npos-1):
        it += 1
        x_new = positions[i+1]
        nabla_new = gradients[i+1]
        s = x_new - x
        y = nabla_new - nabla
        r = 1/(np.dot(y,s))
        if r < 0:               #Curvature condition to ensure positive definite
            continue
        if (H is None) : #initialize based on, but before first update. Taken from Nocedal
            #H = np.eye(x.size) * np.dot(y,s)/np.dot(y, y) #gets confusing if we multiply or divide
            H = np.eye(x.size) / np.dot(y,s) *np.dot(y, y) 
        points_used +=1
        z = np.dot(H, s)
        update = np.outer(y, y) / np.dot(s, y) - np.outer(z, z) / np.dot(s, z)
        H += update
        nabla = nabla_new[:].flatten()
        x = x_new[:].flatten()
    return H, points_used 

    

def BFGS_inverse_Hessian_approx(positions, gradients, H=None): # needs to be cleaned based on Hessian_approx code
    '''Returns Inverse Hessian approximation (H matrix) & number of points used to estimate it'''
    d = positions.shape[1]
    npos = positions.shape[0]
    if H is None:
        H = np.eye(d) # initial hessian
    
    nabla = gradients[0]
    x = positions[0]
    
    it = 0 
    points_used = 0 
    for i in range(npos-1):
        it += 1
        x_new = positions[i+1]
        nabla_new = gradients[i+1]
        s = x_new - x
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1)) 
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        if r < 0:               # Curvature condition to ensure positive definite
            continue
        points_used += 1
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update

        nabla = nabla_new[:].flatten()
        x = x_new[:].flatten()
    
    return H, points_used

