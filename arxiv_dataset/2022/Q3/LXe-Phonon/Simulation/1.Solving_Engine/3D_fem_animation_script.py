# Import the necessary libraries
from fem import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from scipy import interpolate
import sys
import time


# Control variables
# These are the variables that control our simulation.
bounds  = [(0,1),(0,1)]             # Domain bounds
h       = 1e-2                      # Mesh Fineness parameter
FF      = lambda r,z: abs(1.3*r)    # Discretisation function
dt      = 1e-2                      # Simulation time step  

# Define a Source function
f0      = lambda r,z,t: np.exp(-((z-v*t)**2 + r**2)/(2*a**2)) # Gaussian
# f       = lambda r,z,t: a*f0(r,z,t)* ( (2*v/a*(z-t*v))**2 - (2*v*2/a))/10 * (1/(1+np.exp(-20*(t-0.2))))
f       = lambda r,z,t: v**2 * (z+10*a-v*t)/((2*np.pi)**(3/2) * a**5) * f0(r,z+10*a,t)

v       = 1e-0                  # Source wave speed
a       = 2e-2                  # Source std

# Now we will get the mesh and points
points,mesh = get_mesh(h,FF,bounds)
boundary    = get_boundary(points)

# Buld T, S matrices
T,S = get_TS(points,mesh)

# Construct the scheme matrix and apply boundary conditions
Ap= get_scheme_matrix(dt,T,S)
A = set_bc_lhs(boundary,Ap,points,mesh)

# Define the initial condition vectors
# Goal is to solve for U_next, using U_curr and U_prev
U_curr = np.zeros(len(points))
U_prev = np.zeros(len(points))
F      = np.array([f(*point,0) for point in points])


# Let's create an animation
z = 0.5             # We are animating for z = 0.5
Npts = 200
RR  = np.linspace(bounds[0][0],bounds[0][1],70)

# Create the figure
fig = plt.figure(figsize=(10,10),dpi=200)
ax  = fig.add_subplot(111,projection='3d')
ax.set_title(r'Wave in $\phi$-slice t=0')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$Pressure @ z=%.2e [Au]$'%z)
ax.w_xaxis.set_pane_color ((0., 0., 0., 0.))
ax.w_yaxis.set_pane_color ((0., 0., 0., 0.))
ax.w_zaxis.set_pane_color ((0., 0., 0., 0.))

ax.set_zlim(-100,100)

# Get the Pressure for one axis
U = np.array([np.mean(U_curr[mesh.simplices[mesh.find_simplex([r,z]) ]]) for r in RR])
interp = interpolate.make_interp_spline(RR,U)
U = interp(RR)

Ff = lambda x,y: interp((x**2+y**2)**0.5)
R = np.linspace(bounds[0][0],bounds[0][1],Npts)
theta = np.linspace(0,3/2*np.pi,30)

R,theta = np.meshgrid(R,theta)

X = R*np.cos(theta)
Y = R*np.sin(theta)
Z = Ff(X,Y)

# Plot the pressure in 3D
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='coolwarm', edgecolor='none')
# cb = fig.colorbar(surf)

barWidth = 40


class ProgressBar():
    '''
    Provides a simple progress bar class
    '''
    def __init__(self, nsteps, width=barWidth):
        self._start = self._stop = time.time()
        self._nsteps = nsteps
        self._width = width
        self._status = ""

    def update(self, step):
        '''
        This function produces and updates a progress bar.
        It was stolen and modified from
        http://stackoverflow.com/a/15860757/1552338
        '''
        self._start = self._stop
        self._stop = time.time()
        self._status = self._stop - self._start

        progress = float(step)/float(self._nsteps - 1)
        if progress >= 1:
            progress = 1
            self._status = "Complete...\r\n"
        block = int(round(self._width * progress))
        text = "\rProcessing: [{}] {:.1%} {:.3}".format("#" * block+ "-" * (self._width- block),progress,self._status)
        sys.stdout.write(text)
        sys.stdout.flush()

N_steps = 5000
draw_every = 2

progress1 = ProgressBar(N_steps)  # initialize the progress bar
progress2 = ProgressBar(N_steps)  # initialize the progress bar

def animate(i):
    global U_prev,U_curr,F,surf,ax

    for j in range(i*draw_every,(i+1)*draw_every):
        F = get_F(f,j*dt,points)
        # Calculate the left hand side
        B = get_rhs(dt,S,T,U_curr,U_prev,F)

        # move one step
        U_prev = U_curr.copy()
        U_curr = spsolve(A,B)

    # Update the plot
    # Remove elements
    surf.remove()
    # cb.remove()
    
    # # Get the new colormap levels
    # mid = 0 #(U_curr.min() + U_curr.max())/2
    # m = mid - max(abs(U_curr.min() - mid),abs(U_curr.max() - mid)) 
    # M = mid + max(abs(U_curr.min() - mid),abs(U_curr.max() - mid)) 
    # clev = np.linspace(m,M,49)

    # Get the values along the z-plane
    U = np.array([np.mean(U_curr[mesh.simplices[mesh.find_simplex([r,z]) ]]) for r in RR])
    interp = interpolate.make_interp_spline(RR,U)
    U = interp(RR)

    Ff = lambda x,y: interp((x**2+y**2)**0.5)
    Z = Ff(X,Y)

    # Plot the 3D version
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cm.coolwarm,norm=colors.CenteredNorm(), edgecolor='none')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cm.coolwarm, edgecolor='none')
    # cb = fig.colorbar(surf)

    ax.set_title(r'Wave in $\phi$-slice t=%.2e'%(i*dt))

    progress1.update(i)

    return surf

# print("Started Animation Making")
anim = animation.FuncAnimation(fig, animate, frames=range(N_steps), repeat=False)
# print("Started Rendering")
anim.save('3D_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'],progress_callback=lambda i, n: progress2.update(i))
print('\n')