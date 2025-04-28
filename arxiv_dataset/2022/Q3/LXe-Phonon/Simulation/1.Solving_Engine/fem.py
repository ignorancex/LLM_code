#########################################################################
# Finite Element Method Implementation for Solving the Strongly Damped  #
# Wave Equation!                                                        #
#                                                                       #
# The Entire code needed to build a fully functioning simulation is     #
# right here                                                            #
#                                                                       #
#                                                                       #
# Panos Oikonomou (po524) | NYUAD, Arneodo Lab | 2021                   #
#########################################################################

# Import the relevant modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.linalg import solve
from scipy.spatial import Delaunay

#########################################################################
## Helper Functions
#########################################################################

# Get a property file and return a dictionary with the info
def get_properties(fname):
    labels  = np.loadtxt(fname,usecols=0,dtype=str,delimiter='(')
    units   = np.loadtxt(fname,usecols=0,dtype=str,delimiter=':')
    values  = np.loadtxt(fname,usecols=1)

    return dict(zip(labels,values))         # Create a dictionary with the properties of the liquid



#########################################################################
## Space Discretisation
#########################################################################

# Some helper functions

def merge(A,B):
    '''Merges two lists performing a set product A x B'''
    L = []
    for a in A:
        for b in B:
            L.append([a,b])
    return L

def center(box):
    '''Returns the central point of a box'''
    x = 0 
    y = 0
    for p in box:
        x+=p[0]
        y+=p[1]

    return (x/4,y/4)

def create_box(point,box,points):
    '''Creates a new box on the grid'''
    new_box = [point]
    for p in box:
        if p == point: continue

        new_p = ((p[0] + point[0])/2,(p[1] + point[1])/2) # Create new point

        if new_p not in points:     # For if the point is new add it
            points.append(new_p)

        new_box.append(new_p)
    
    c = center(new_box)
    if c not in points: points.append(c)

    return new_box

def box_needed(point,box,F,h=5e-2):
    '''Given a particular Function F, and a box, it returns weather or 
    not there needs to be a finer asjustment in the mesh at said point'''
    needed = False
    for p in box:
        if p == point: continue

        if abs(F(*p)-F(*point)) > h: return True

    return needed

def reorder(points):
    scores = np.array([len(points)*point[1]+point[0] for point in points])
    indx   = np.arange(0,len(points))

    sorted_pairs = sorted(zip(scores,indx))

    tuples = zip(*sorted_pairs)
    scores, indx = [ list(tuple) for tuple in  tuples]

    return points[indx]


# Now for the actual discretisation function in triangles
def get_mesh(h=5e-2, F = lambda r,z: abs(r),bounds=[(0,1),(0,1)],):
    '''Returns a triangle mesh and a set of points for an arbitrary function'''

    # Set the bounds
    r_bounds = bounds[0]
    z_bounds = bounds[1]

    # Create the first 4 points.
    points = merge(r_bounds,z_bounds)
    ground_box = points.copy()
    points.append(center(ground_box))   # Add the central point of the box (treated differently)

    # Now we recursively generate the rest of the points.
    box_q = [ground_box]            # We start with only one box on the queue

    # While there are boxes left to check
    while len(box_q) > 0:
        box = box_q[0]                  # Get the current box

        # For all the points in the box
        # If we need to add a new box add it
        for point in box:
            if box_needed(point,box,F): 
                new_box = create_box(point,box,points)      # Create a box
                box_q.append(new_box)                       # Add the new box to the box queue

        # Remove the current box from the boxes 
        box_q.pop(0)

    points = np.array(points)

    # This step is important to get prettier sparse matrices
    points = reorder(points)

    # We can generate a mesh using Delaunay Triangulization
    mesh = Delaunay(points)

    # We return the mesh and the points
    return points,mesh

# Generate arectangular grid of points, jsut in case we want a quick and dirty solution
def get_mesh_grid(h=5e-2,bounds=[(0,1),(0,1)]):
    # Create axes
    r = np.arange(bounds[0][0],bounds[0][1]+h,h)
    z = np.arange(bounds[1][0],bounds[1][1]+h,h)

    # Generate a list of points
    points = np.array(np.meshgrid(r,z)).T
    points = points.reshape(points.shape[0]**2,2)

    # This step is important to get prettier sparse matrices
    points = reorder(points)

    # Generate the mesh
    mesh = Delaunay(points)

    # We return the mesh and the points
    return points,mesh

# Generate a mesh using a random scoring function
def get_mesh_random(h_min=2e-2, F = lambda r,z: np.exp(-(r**2)/0.5**2), bounds=[(0,1),(0,1)]):

    # Now we will genrate a set of points with this separation
    R = np.arange(bounds[0][0],bounds[0][1]+h_min,h_min)
    Z = np.arange(bounds[1][0],bounds[1][1]+h_min,h_min)
    points = merge(R,Z)

    # Now filter the mesh
    i=0
    while i < len(points):
        if np.random.rand() >= F(*points[i]):
            points.remove(points[i])
            i-=1
        i+=1

    # Add the 4 bounday points bback to the mesh
    for p in merge(*bounds):
        if p not in points:
            points.append(p)

    points = np.array(points)

    # This step is important to get prettier sparse matrices
    points = reorder(points)

    # We can generate a mesh using Delaunay Triangulization
    mesh = Delaunay(points)

    # We return the mesh and the points
    return points,mesh


# One more function, to print the mesh in a pretty way
def plot_mesh(points,mesh,F,bounds=[(0,1),(0,1)],plot=False):
    
    # Define a figure
    fig = plt.figure(figsize=(5,5),dpi=150)
    ax  = fig.add_subplot(111)
    ax.set_title('Space Discretisation')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$z$')

    # Plot the discretisation function as the background
    Npts = 50
    r = np.linspace(*bounds[0],Npts)
    z = np.linspace(*bounds[1],Npts)
    extent = np.min(r), np.max(r), np.min(z), np.max(z)

    R,Z = np.meshgrid(r,z)
    FF = np.flip(F(R,Z),axis=0)
    ax.imshow(FF,cmap=cm.Blues,interpolation='bilinear',extent=extent)

    # Scatter the points
    ax.scatter(*np.array(points).T,s=1,marker='o',c='k')

    # Plot the triangles on top of that
    ax.triplot(points[:,0], points[:,1], mesh.simplices,lw=0.1,c='k');

    # If we should open the plot window
    if plot: plt.show()

    # Return the plot
    return fig,ax


#########################################################################
## Building the S and T Matrices for linear tringly things
#########################################################################

# Some helper functions
def get_simplices(tri, point):
    '''Find all simplices this point belongs to'''

    visited = set()
    queue = [tri.vertex_to_simplex[point]]
    while queue:
        simplex = queue.pop()
        for i, s in enumerate(tri.neighbors[simplex]):
            if tri.simplices[simplex][i] != point and s != -1 and s not in visited:
                queue.append(s)
        visited.add(simplex)
    return np.array(list(visited))

def jacobian(p):
    '''Calculates the jacobian of the triangle transformation given the 3 points'''
    return (p[1][0]-p[0][0])*(p[2][1]-p[1][1])-(p[2][0]-p[0][0])*(p[1][1]-p[0][1])


## The actual thing
def get_TS(points,mesh):
    N_points = len(points)

    # We start with empty matrices
    T = np.zeros((N_points,N_points))
    S = np.zeros((N_points,N_points))

    # The matrices with the integrals for T and S
    A_T = np.pi/60 * np.array([[2,2,1],  [2,6,2], [1,2,2] ])
    A_S = np.pi/3  * np.array([[2,-1,-1],[-1,1,0],[-1,0,1]])

    # For all the points
    for i in range(len(points)):
        # Find the neighboring triangles to x_i
        nei_tri = get_simplices(mesh,i)
        nei_pts = np.unique(mesh.simplices[nei_tri].reshape(-1))
        
        # For all the neighboring triangles
        for triangle in nei_tri:
            p_indx = mesh.simplices[triangle]   # Get the points for set triangle
            pts    = points[p_indx]             # Get the number of points
            J      = jacobian(pts)              # Calculate the jacobian for this triangle

            # the index (1,2,3) of the ith point is:
            ki = p_indx.tolist().index(i)

            for kj in range(len(p_indx)):        # For each of those points
                # We need to unpack if this point is 1,2, or 3.
                # To do that we use the index of the triangle list
                j = p_indx[kj]
                T[i][j] += A_T[ki][kj]*J
                S[i][j] += A_S[ki][kj]*J

    
    return T,S


# And a method to visualize both of the sparse matrices
def plot_TS(S,T,plot='True'):
    # Create Figure
    fig = plt.figure(figsize=(10,5),dpi=200)
    fig.suptitle('Block Tridiagonal S and T matrices visualized')

    ax_T = fig.add_subplot(121)
    ax_S = fig.add_subplot(122)
    ax_T.set_title(r'$T_{ij}$');
    ax_S.set_title(r'$S_{ij}$');
    
    # Plot the non-zero cells of the matrices in black
    ax_T.imshow(T!=0,cmap=cm.binary)
    ax_S.imshow(S!=0,cmap=cm.binary)

    if plot: plt.show()

    return fig,ax_T,ax_S


#########################################################################
## Handle Boundary Conditions
#########################################################################

def get_boundary(points,bounds = [(0,2),(0,1)]):
    '''Returns a list of boundary point indexes'''

    pts = points.tolist()
    bd_points = merge(*bounds)
    # bd_lower = np.arange(0,pts.index(bd_points[2]))
    # bd_upper = np.arange(pts.index(bd_points[1]),len(pts))
    bd_lower = [pts.index(p) for p in pts if p[1]==bounds[1][0]]
    bd_upper = [pts.index(p) for p in pts if p[1]==bounds[1][1]]
    bd_left  = [pts.index(p) for p in pts if p[0]==bounds[0][0]]
    bd_right = [pts.index(p) for p in pts if p[0]==bounds[0][1]]

    return bd_lower,bd_upper,bd_left,bd_right

def plot_boundary(boundary,points,plot=True):
    '''Plots the boundary in A 2D view to make sure there is no funny bussiness'''

    bd_lower,bd_upper,bd_left,bd_right = boundary

    # Create a figure
    fig = plt.figure(figsize=(4,4),dpi=150)
    ax  = fig.add_subplot(111)
    ax.set_title('Boundary')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$z$')

    # Scatter the points
    ax.scatter(*points[bd_lower].T,label='Lower Boundary',marker='.',s=1)
    ax.scatter(*points[bd_upper].T,label='Upper Boundary',marker='.',s=1)
    ax.scatter(*points[bd_left].T, label='Left Boundary', marker='.',s=1)
    ax.scatter(*points[bd_right].T,label='Right Boundary',marker='.',s=1)
    
    plt.legend(frameon=False)

    if plot: plt.show()

    return fig,ax

def set_bc_rhs(boundary,B):
    '''Sets the boundary conditions on the left hand side'''
    bd_lower,bd_upper,bd_left,bd_right = boundary

    # I know this can look prettier... it's 3:00 right now
    B[bd_lower] = 0
    B[bd_upper] = 0
    B[bd_right] = 0
    # B[bd_left]  = 0

def set_bc_lhs(boundary,A,points,mesh):
    '''Sets the boundary conditions on the right hand side'''
    # The conditions are as follows:
    # Upper: Dirichlet  |   U = 0
    # Lower: Dirichlet  |   U = 0
    # Left : Neuman     |   dU/dr = 0
    # Right: Dirichlet  |   U = 0

    # Unpack the boundary
    bd_lower,bd_upper,bd_left,bd_right = boundary

    # Get the point indices
    pts = points.tolist()

    # Dirichlet
    # Lower
    for p in bd_lower:
        for i in range(len(pts)): A[p][i] = 0
        A[p][p] = 1

    # Upper
    for p in bd_upper:
        for i in range(len(pts)): A[p][i] = 0
        A[p][p] = 1

    # Right
    for p in bd_right:
        for i in range(len(pts)): A[p][i] = 0
        A[p][p] = 1

    # # Neumann
    # # Left
    # for p in bd_left:
    #     for i in range(len(pts)): A[p][i] = 0
    #     A[p][p] = 1

    #     nei_pts = [ pt for pt in np.unique(mesh.simplices[get_simplices(mesh,p)].reshape(-1)) if points[p][0]!= 0]
    #     for i in nei_pts:A[p][i] = -1/len(nei_pts)

    # Convert the matrix in linear form and try again
    
    return csc_matrix(A,dtype=float)

def plot_A(A,plot=True):
    '''Plots the Scheme matrix'''
    # Create a figure
    fig = plt.figure(figsize=(5,5),dpi=200)
    ax = fig.add_subplot(111)
    ax.set_title('Scheme Matrix')

    # Plot the matrix
    ax.imshow(A!=0,cmap=cm.binary)

    if plot: plt.show()

    return fig,ax




#########################################################################
## Numerical Scheme and Simulation Commands
#########################################################################

def get_scheme_matrix(dt,T,S,tau=1,lam=1):
    '''Returns the scheme Matrix for a particular setup'''
    return T*(lam**2/dt) + (1/tau+dt)*S
    # return T/dt

def get_rhs(dt,S,T,U_curr,U_prev,F,tau=1,lam=1,boundary=None):
    '''Returns the vector for the left hand side of the equation'''

    BB =  np.matmul(S/tau + (2*lam**2/dt) * T, U_curr) - np.matmul(T,U_prev/dt*lam**2) - dt*np.matmul(T,F)
    if type(boundary) != type(None):
        bd_lower,bd_upper,bd_left,bd_right = boundary
        B_bound = np.matmul(S/tau + (2*lam**2/dt) * T, U_curr) - np.matmul(T,U_prev/dt*lam**2)
        
        for p in bd_lower: BB[p] = B_bound[p]
        for p in bd_upper: BB[p] = B_bound[p]
        for p in bd_right: BB[p] = B_bound[p]
        # for p in bd_left:BB[p] = B_bound[p]

    return BB
    # return np.matmul(-S*(1+dt) + (2/dt) * T,U_curr) + np.matmul(S-T/dt,U_prev) - dt*np.matmul(T,F)


def get_F(f,t,points):
    '''Returns the vector F'''
    return np.array([f(*point,t) for point in points])

def step(dt,A,S,T,U_curr,U_prev,F,tau=1,lam=1,boundary=None):
    '''Performs one simulation step'''
    
    # Calculate the left hand side
    B = get_rhs(dt,S,T,U_curr,U_prev,F,tau=tau,lam=lam)
    if (type(boundary)!=type(None)): set_bc_rhs(boundary,B)

    # move one step
    U_prev = U_curr.copy()
    U_curr = spsolve(A,B)

    return U_curr,U_prev

def run(t,dt,A,S,T,U_curr,U_prev,f,points,tau=1,lam=1,boundary=None):
    '''Run the simulation continiously for time t'''

    # Get the number of iterations
    N = int(t//dt)

    # Display a progress bar for fun
    for i in range(N):
        F = get_F(f,i*dt,points)
        U_curr, U_prev = step(dt,A,S,T,U_curr,U_prev,F,tau=tau,lam=lam)

    return U_curr

def plot_U(points,mesh,U):
    '''Plots the solution in the grid'''
    # Create the figure
    fig = plt.figure(figsize=(6,5),dpi=200)
    ax  = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(r'Wave in $\phi$-slice')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$z$')

    # Plot the grid and the wave on top of it.
    ax.triplot(points[:,0], points[:,1], mesh.simplices,lw=0.1,c='grey')
    clev = np.linspace(U.min(),U.max(),49)
    cf = ax.tricontourf(points[:,0], points[:,1], mesh.simplices,U,levels=clev,cmap=cm.coolwarm)

    fig.colorbar(cf)