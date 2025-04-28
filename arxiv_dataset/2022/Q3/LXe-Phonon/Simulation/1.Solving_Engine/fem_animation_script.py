# Import the necessary libraries
from fem import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
U_curr = F.copy()
U_prev = np.array([f(*point,-dt) for point in points])


# Let's create an animation
# Create the figure
fig = plt.figure(figsize=(6,5),dpi=200)
ax  = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_title(r'Wave in $\phi$-slice t=0')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$z$')

# Plot the grid and the wave on top of it.
ax.triplot(points[:,0], points[:,1], mesh.simplices,lw=0.1,c='grey')
# clev = np.linspace(U_curr.min(),U_curr.max(),49)
cf = ax.tricontourf(points[:,0], points[:,1], mesh.simplices,U_curr,cmap=cm.coolwarm)
cb = fig.colorbar(cf)

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

N_steps = 2000
draw_every = 2

progress1 = ProgressBar(N_steps)  # initialize the progress bar
progress2 = ProgressBar(N_steps)  # initialize the progress bar

def animate(i):
    global U_prev,U_curr,F,cf,ax,cb

    for j in range(i*draw_every,(i+1)*draw_every):
        F = get_F(f,j*dt,points)
        # Calculate the left hand side
        B = get_rhs(dt,S,T,U_curr,U_prev,F)

        # move one step
        U_prev = U_curr.copy()
        U_curr = spsolve(A,B)

    # Update the plot
    for c in cf.collections:
        c.remove()
    
    mid = 0 #(U_curr.min() + U_curr.max())/2
    m = mid - max(abs(U_curr.min() - mid),abs(U_curr.max() - mid)) 
    M = mid + max(abs(U_curr.min() - mid),abs(U_curr.max() - mid)) 
    clev = np.linspace(m,M,49)
    cf = ax.tricontourf(points[:,0], points[:,1], mesh.simplices,U_curr,levels=clev,cmap=cm.coolwarm)
    cb.remove()
    cb = fig.colorbar(cf)

    ax.set_title(r'Wave in $\phi$-slice t=%.2e'%(i*dt))

    progress1.update(i)

    return cf

# print("Started Animation Making")
anim = animation.FuncAnimation(fig, animate, frames=range(N_steps), repeat=False)
# print("Started Rendering")
anim.save('animation4.mp4', fps=60, extra_args=['-vcodec', 'libx264'],progress_callback=lambda i, n: progress2.update(i))

# import matplotlib as mpl 
# mpl.rcParams['animation.ffmpeg_path'] = '/home/po524/Desktop/ffmpeg-4.3.2/ffmpeg'
# anim.save('animation4.mp4',writer=animation.FFMpegWriter(fps=2) ,progress_callback=lambda i, n: progress2.update(i))