import numpy as np
import string
import multiprocess as mp
from scipy.spatial.transform import Rotation as R
import random
global THE_ZERO
global tolerance

THE_ZERO = 1e-8
tolerance = 1e-8

def read_line(txt):
    out_array  = []
    string = txt[1:-2].replace(' ','').replace('[','').split('],')
    for el in string[:-1]:
        aux = []
        for component in el.split(','):
            aux.append(float(component))
        out_array.append(np.array(aux[1:]))
    return np.array(out_array)    


def load_knot(knot_type = '+3_1', length = 100, index = 0):
    txt = open('generated_knots/length{}/knots_{}_{}.txt'.format(length, length, knot_type),'r')
    lines = txt.readlines()
    knot = read_line(lines[2*index])
    knot = np.append(knot,[knot[0]], axis = 0)
    return knot


def knot_intensity(core, l):
    
    """
    Inputs:

    - core, a dictionary.
    
    - l: an integer, indicates the length of the curve
  
    Outputs:
        an array of length l giving knot intensity distribution. 


    """ 
    order = 'intrinsic'
            
    color = l*[0]


    if order == 'intrinsic':
        for op in range(l):
            if core[op][1] < core[op][2]: 
                if core[op][0] != '0_1':
                    for a in range(core[op][1], core[op][2]):
                        color[a] = color[a] + 1
            else:
                if core[op][0] != '0_1':
                    for a in range(core[op][1],l):
                        color[a] = color[a] + 1
                    for a in range(0,core[op][2]+1):
                        color[a] = color[a] + 1    
        
    return np.round(np.array([el/l for el in color]),2)
        

def fingerprint(c_dict, l):
    """
    Inputs:

    - c_dict, a dictionary.
    - l: an integer, indicates the length of the curve
      
    Outputs:
        an array of length l giving the fingerprint function. 


    """ 
    
    f = knot_intensity(c_dict, l)	

    g = [(1/100)*(np.asarray(f) > i/100).sum()/len(f) for i in range(100)]
    for i in range(1,100):
        g[i] = g[i] + g[i-1]
    return np.array(g)



def knot_core(curve):
    '''
    Computes the Knot Core of a curve. 
    '''    
    if curve[0][0] == curve[-1][0] and curve[0][1] == curve[-1][1] and curve[0][2] == curve[-1][2]:
        curve = curve[:-1]
    to_knotoID= '\n'.join(['%s %s %s'%tuple(row) for row in curve])

    stdout = get_ipython().getoutput('echo "$to_knotoID"| /Users/agnesebarbensi/Downloads/Knoto-ID-1.3.0-Darwin/bin/knotted_core stdin  --closure-method=rays --nb-projections=50 --timeout=0.5 --names-db=internal --output stdout ' ) 
    
    stdout = [x.split("\t")[:-1] for x in stdout]
    core = [int(stdout[-1:][0][0]),int(stdout[-1:][0][1])]
    ktype = stdout[-1:][0][4]
    return(ktype,core)



def make_openings(curve):
    open_curves = []
    for ii in range(0,len(curve)):
        open_curve = [el for el in curve[ii:]] 
        for kk in range(ii):
            open_curve.append(curve[kk])
        open_curves.append([np.array(open_curve),ii])
    return open_curves



def makeComps(openinings_core,curve,b = 1):
    a = knot_core(curve[0])
    core = [a[0], curve[0][a[1][0]],curve[0][a[1][1]],curve[1]]
    openinings_core.append(core)
    



def Global_knot_core(curve):
    manager = mp.Manager()
    openinings_core = manager.list()
    PROCESSES = mp.cpu_count()
    dic = {}
    with mp.Pool(PROCESSES) as pool:
        params = make_openings(curve)
        results = [pool.apply_async(makeComps, args=[openinings_core,p,1]) for p in params]
        for r in results:
            r.get()
    for el in openinings_core:
        dic[el[3]] = [el[0],np.where(curve == el[1])[0][0],np.where(curve == el[2])[0][0]]
    return dic    





def intersection_routine(Q1,Q2,Q3,j1,j2,tolerance):
    """
    Checks if the the j1-j2 segment intersects the triangle Q1-Q2-Q3 
    
    Params:
        Q1,Q2,Q3 -- 3 (3,) arrays  (points bounding the triangle)
        j1,j2    -- 2 (3,) arrays  (points bounding the segment)
    Returns:
        Boolean

    """
    V1 = Q2 - Q1
    V2 = Q3 - Q1
    s = j2 - j1
    n = np.cross(V1,V2)
    norme_n = np.linalg.norm(n)
    if norme_n <THE_ZERO:
        return False
    if np.linalg.norm(s) < THE_ZERO:
        return False
    ns = np.dot(n,s)
    if abs(ns) < THE_ZERO:
        return False
    d = Q1 - j1
    t = np.dot(n,d)/ns
    if (t < (-tolerance)) or (t > (1+tolerance)):
        return False
    r = t * s + j1 - Q1
    u =(np.dot(n,np.cross(r,V2)))/(norme_n*norme_n)
    v = -(np.dot(n,np.cross(r,V1)))/(norme_n*norme_n)
    if (u >= -tolerance) and (v >= -tolerance) and (u+v <= 1 + tolerance):
        return True
    return False


def rotation(P1,P2,P3,theta):
    """
    Rotates P3 of an angle theta along the P1-P2 axis 

    Params:
        P1,P2,P3 -- 3 (3,) arrays
        theta    -- float 
    Returns:
        (3,) array

    """  
    p2 = [P2[i] - P1[i] for i in range(3)]
    p3 = [P3[i] - P1[i] for i in range(3)]
    rotation_axis = np.array(p2)
    rotation_vector = theta * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_p3 = rotation.apply(p3)
    rotated_P3 = [rotated_p3[i] + P1[i] for i in range(3)]
    
    return(rotated_P3)


def random_unit():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Returns:
        (3,) array
    
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    unit = np.array([x,y,z])
    return (unit)



def CS_move(Curve,i,theta, self_avoiding = True):
    """
    Performs a crank shaft move on PL curve at index i of angle theta. 
    This means rotating the point Curve[i] along the line Curve[i-1]-Curve[i+1]. 
    If self_avoiding is True, it returns an index where that indicates
    where the strand passage happens. 
    To check that a CS move is non-phantom, we check that no segment in 
    Curve intersects the revolution surface defined by moving Curve[i]. 
    
    Params:
        curve         -- (3,n) array
        i             -- integer 
        theta         -- float.
        self_avoiding -- boolean
        
    Returns:
        (3,n) array   

    """
    
    traj = Curve.copy()
    n = len(traj)
    delta = 2*tolerance

    change = False
    
    
    new_point = rotation(traj[(i-1)%n],traj[(i+1)%n],traj[i%n],theta)  
        
    segments = np.array([x for x in range(n) if x not in [(i-2)%n,(i-1)%n,i%n,(i+1)%n] ])

        
    if self_avoiding:
        """
        Check if non-phantom by looking that no arc in the curve intersects the 
        triangles traj[i+1],traj[i],new_point and traj[i+1],traj[i+2],new_point
        """         
        sh1 = (1-delta)* traj[(i-1)%n] + delta*traj[(i-2)%n]
        sh2 = (1-delta)* traj[(i+1)%n] + delta*traj[(i+2)%n]
        if intersection_routine(traj[i%n],traj[(i-1)%n],new_point,traj[(i-2)%n],sh1,tolerance) or intersection_routine(traj[i%n],traj[(i+1)%n],new_point,traj[(i-2)%n],traj[(i-1)%n],tolerance):
            change = True
            where =(i-2)%n
        if intersection_routine(traj[i%n],traj[(i-1)%n],new_point,traj[(i+1)%n],traj[(i+2)%n],tolerance) or intersection_routine(traj[i%n],traj[(i+1)%n],new_point,sh2,traj[(i+2)%n],tolerance):
            change = True
            where = (i+1)%n

        for a in segments:
            if intersection_routine(traj[i%n],traj[(i-1)%n],new_point,traj[a%n],traj[(a+1)%n],tolerance) or intersection_routine(traj[i%n],traj[(i+1)%n],new_point,traj[a%n],traj[(a+1)%n],tolerance):
                change = True
                where = a


     
        traj[i%n] = new_point
        if change == False:
            return(traj,change,False)
        return(traj, change, where)
    else:
        traj[i%n] = new_point
    
        return(traj)

