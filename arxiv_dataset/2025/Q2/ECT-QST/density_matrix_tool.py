#!/usr/bin/env python3

#
# routines for general density matrices
# 1. generation
# 2. distances
# 3. characteristics (purity, entropy, etc.)
# 4. 2D and 3D plotting
#
# (c) 2024 Giovanni Garberoglio (garberoglio@ectstar.eu), Daniele Binosi (binosi@ectstar.eu)

import numpy as np

import scipy as sp
from scipy.stats import unitary_group
from scipy.stats import dirichlet

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Pauli matrices, which are always useful
sigmax = np.array([[0,1],  [1,0]],  dtype=np.complex128)
sigmay = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sigmaz = np.array([[1,0],  [0,-1]], dtype=np.complex128)

def rho_from_state(v):
    rho = np.outer(v, v.conj())
    return(rho)

def random_pure_state(N,ndiag=None, complex_amplitude=True):
    """
    A random pure state.
    Options:
    
    ndiag = integer -> the density operator has ndiag elements only
    complex_amplitude = False -> only real amplitudes
    """
    while True:
        v = np.random.uniform(-1.0,1.0,size=N)
        if complex_amplitude:
            v = v + 1j * np.random.uniform(-1.0,1.0,size=N)
        
        if ndiag is not None:
            assert ndiag <= N
            idx = np.random.choice(N,size=N-ndiag, replace=False)
            v[idx] = 0.0

        norm = np.real(np.sum(np.conj(v) * v))
        if norm < N:
            break
    v = v / np.sqrt(norm)
    return( v )

        
def random_pure_state_real(N):
    """
    Creates a random state with real amplitudes
    """
    return random_pure_state(N,complex_amplitude=False)

######################################################################

# Special states (GHZ, W, etc.)

def normalize_psi(psi):
    ret = psi / np.sqrt( np.real(np.dot(np.conj(psi), psi) ))
    return(ret)

def GHZ_psi(nqubit):
    """
    returns the |GHZ> state of n qubits
    |GHZ> = (|0...0> + |1...1>) / sqrt(2)
    https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state
    """
    N = 2**nqubit
    v = np.zeros(N, dtype=np.complex128)
    v[0]  = 1/np.sqrt(2) # |0...0>
    v[-1] = 1/np.sqrt(2) # |1...1>
    
    return (v)


def GHZ(nqubit):
    """
    returns the density matrix of the |GHZ> state of n qubits
    """
    v = GHZ_psi(nqubit);
    return np.outer(v, np.conj(v))

def W_psi(nqubit):
    """
    returns the density matrix of the |W> state of n qubits
    |W> = (|0...1> + |0...10> + |0...100> + |10...>) / sqrt(n)
    https://en.wikipedia.org/wiki/W_state
    """
    N = 2**nqubit
    v = np.zeros(N, dtype=np.complex128)
    idx = np.power(2,range(nqubit))
    v[idx] = 1.0/np.sqrt(nqubit)
    
    return (v)

def W(nqubit):
    """
    returns the density matrix of the |W> state of n qubits
    |W> = (|0...1> + |0...10> + |0...100> + |10...>) / sqrt(n)
    https://en.wikipedia.org/wiki/W_state
    """
    v = W_psi(nqubit)
    return np.outer(v, np.conj(v))

def color_code_7qubit_0_psi():
    psi = np.zeros(2**7, dtype=np.complex128)
    
    # [::-1] is the difference between the real and fake world :-)
    psi[int('1010101'[::-1],2)] = 1
    psi[int('1100011'[::-1],2)] = 1
    psi[int('0101101'[::-1],2)] = 1
    psi[int('0011011'[::-1],2)] = 1
    psi[int('1001110'[::-1],2)] = 1
    psi[int('0110110'[::-1],2)] = 1
    psi[int('1111000'[::-1],2)] = 1
    psi[int('0000000'[::-1],2)] = 1
    
    return(normalize_psi(psi))

def color_code_7qubit_1_psi():
    psi = np.zeros(2**7, dtype=np.complex128)
    
    psi[int('0101010'[::-1],2)] = 1
    psi[int('1010010'[::-1],2)] = 1
    psi[int('0011100'[::-1],2)] = 1
    psi[int('1100100'[::-1],2)] = 1
    psi[int('0110001'[::-1],2)] = 1
    psi[int('1001001'[::-1],2)] = 1
    psi[int('0000111'[::-1],2)] = 1
    psi[int('1111111'[::-1],2)] = 1
    
    return(normalize_psi(psi))


def color_code_7qubit_superposition():
    zero = color_code_7qubit_0_psi()
    one  = color_code_7qubit_1_psi()
    psi = zero + one
    return(normalize_psi(psi))

######################################################################

def random_density_matrix(N,n=0):
    """
    Creates a random statistical operator in a NxN Hilbert space 
    by inchoerent superposition of n random states with probabilities
    having progressively smaller random uniform values.

    The n random states are taken from a random unitary matrix

    Default is to consider all the states.
    """
    U = unitary_group.rvs(N)
    # create random probabilities and normalize them
    D = np.zeros(N)
    
    n = N if n==0 else n # the number of nonzero probabilities
    D[:n] = np.random.rand(n)
    #D = D/np.sum(D)
    D[:n] = dirichlet.rvs(alpha=[1]*n,size=1) # better distribution of probabilities

    # here it is, a random matrix in the Hilbert space
    rho = U @ (D.reshape(-1,1) * U.conj().T)

    return(rho)

def random_density_matrix_w_zeros(N,nz=0):
    """
    Creates a random statistical operator in a NxN Hilbert space 
    by inchoerent superposition of n random states with probabilities
    having progressively smaller random uniform values.

    The sets to zero nz elements on the diagonal
    """
    rho = random_density_matrix(N)
    
    idx = np.random.choice(N,size=nz,replace=False) # elements to zero out
    rho[idx,:] = 0 # zero out rows
    rho[:,idx] = 0 # zero out cols

    return rho / np.trace(rho)


def add_noise(rho,N=1e2,n=1):
    """                                                                                                  
    Add noise to the matrix so that the trace is of the order of N                                       
    and the elements that are zero are of the order of n,                                                
    where n is taken from an exponential distribution                                                    
    """
    rho_small = np.random.exponential(scale=n, size=rho.shape)

    rho_big = rho * N / np.max(np.real(rho.diagonal()))
    idx = np.nonzero(rho_big)
    for i,j in zip(*idx):
        modulus = np.abs(rho_big[i,j])
        phase = rho_big[i,j]/modulus
        rho_big[i,j] = np.random.poisson(modulus) * phase

    ret = rho_big + rho_small
    return ret / np.trace(ret)
    
    
######################################################################

def Frobenius_norm(A):
    """
    Frobenius norm of a matrix, that is the sum of the squares of the
    absolute values of A.
    """
    ret = np.real(np.sum(A*np.conj(A)))
    return(ret)

def Frobenius_distance(A,B):
    return( np.sqrt(Frobenius_norm(A-B)) )

def p_norm(A, p=2):
    """
    p norm of a matrix A.
    """
    val, vec = np.linalg.eigh(A)
    ret = np.sum(np.abs(val)**p)**(1/p)

    return(ret)

def holevo_helstrom_probability(A,B):
    
    onenrm =  p_norm(A-B, p=1)
    ret = 1/2 *(1 + (1/2)*onenrm )

    return(ret)

def purity(A):
    """
    Purity of a matrix A, that is tr(A^2).
    This is the same as the norm of the matrix.
    """
    return(Frobenius_norm(A))

def matrix_sqrt(A):
    """
    Try to use scipy, and revert to hand-made program if it fails.
    Because it fails with some matrices on "old" x86-64 systems.
    """
    ret = sp.linalg.sqrtm(A).astype(np.complex128)

    if np.all(np.isfinite(ret)) == False:
        val, vec = np.linalg.eigh(A)
        ALMOST_ZERO = 1e-6
        val = np.abs(val)
        val[val < ALMOST_ZERO] = 0.0
        val = np.sqrt(val)
        ret = vec @ (val.reshape(-1,1) *  np.conj(vec.T))

    assert np.all(np.isfinite(ret))
    return(ret)
        
def fidelity(A,B):
    """
    Returns tr( sqrt( sqrt(A) B sqrt(A) )
    If B is none, then we take the density matrix of this system
    """
    # sqrt_A = sp.linalg.sqrtm(A) 
    # sqrt_M = sp.linalg.sqrtm(sqrt_A @ (B @ sqrt_A))
    sqrt_A = matrix_sqrt(A) 
    sqrt_M = matrix_sqrt(sqrt_A @ (B @ sqrt_A))
    # fidelity should be real
    fi = np.real(np.trace(sqrt_M))
    fi = fi if 0<= fi <= 1 else 1
    
    return( fi )

def Bures_distance(A,B):
    
    fi = fidelity(A,B)
    ret = np.sqrt(2*np.abs((1-np.sqrt(fi))))
    
    return (ret)

def Bures_angle(A,B):
    """
    Measures the statistical distance between quantum states
    """
    fi = fidelity(A,B)
    ret = np.arccos(np.sqrt(fi))
    
    return (ret)


def concurrence(A):
    """
    Calculates the concurrence of the density matrix A
    #At the moment implemented only for a two qubit system
    """

    if A.size != 16:
        #print ("concurrence: wrong shape of input; use generalized_concurrence instead")
        return np.real(np.sqrt(2*(1-np.trace(A @ A))))

    sigmay = np.array([[0,-1j],[1j,0]])
    sigmay2 = np.kron(sigmay,sigmay)

    #sqrt_A = sp.linalg.sqrtm(A)
    sqrt_A = matrix_sqrt(A)
    A_tilde = sigmay2 @ (np.conj(A) @ sigmay2)
    R2 = sqrt_A @ (A_tilde @ sqrt_A)

    val, vec = np.linalg.eigh(R2)
    val[val<0.0] = 0.0
    val = np.sort(np.sqrt(val))
    Cval = 2*val[-1] - np.sum(val)
    
    return( max(0,Cval) )

def entanglement_of_formation(A):
    """
    Calculates the entanglement of formation
    for a two qubit system
    """
    if A.size != 16:
        print ("entanglement of formation: wrong shape of input")
        return 0

    C = concurrence(A)
    x = (1+np.sqrt(1-C*C))/2

    if x>0 and x<1:
        eof =  -x * np.log2(x) - (1-x)*np.log2(1-x)
    else:
        eof = 0 # x is either 0 or 1. In this case the limit is 0
        
    return( eof )

def entropy(A):
    """
    Calculates the Von Neumann entropy of a matrix, 
    defined as the sum of the entropies opf the eigenvalues
    """

    val, vec = np.linalg.eigh(A)
    val = np.ma.masked_inside(val,-10**-10,10**-10).filled(0)

    with np.errstate(divide='ignore'):
        lg = np.log2(val)
    
    lg[np.isneginf(lg)]=0
    vne = np.ma.masked_inside(-val*lg,-10**-10,10**-10).filled(0)

    return ( np.sum(vne) )

def KL_divergence(P,Q):
    """
    The Kullback-Leibler divergence between P and Q
    KL(A,B) = tr(P log(P) - log(Q))
    
    If P and Q are statistical operators
    This measures the "distance" between the distribution Q and P.
    """
    valP, vecP = np.linalg.eigh(P)
    idx = (valP <= 0.0)
    log_valP = np.log2(np.abs(valP))
    log_valP[idx] = 0.0
    
    valQ, vecQ = np.linalg.eigh(Q)
    idx = (valQ <= 0.0)
    log_valQ = np.log2(np.abs(valQ))
    log_valQ[idx] = 0.0

    lP = vecP @ (log_valP.reshape(-1,1) * np.conj(vecP.T))
    lQ = vecQ @ (log_valQ.reshape(-1,1) * np.conj(vecQ.T))    

    return np.real(np.trace(P @ (lP - lQ)))

def KL_symm_divergence(A,B,w=0.5):
    """
    The symmetric Kullback-Leibler divergence between A and B
    KL(A,B) = w * (KL(A,B) + KL(B,A))
    defaults to w = 0.5
    """
    KLAB = KL_divergence(A,B)
    KLBA = KL_divergence(B,A)
    ret = w * (KLAB + KLBA)
    return(ret)

def gini_index(vector):
    """
    Returns the (rescaled) Gini index of the diagonal of the density matrix.
    """
    sorted_vector = np.sort(vector)
    N = len(sorted_vector)

    norm = np.linalg.norm(sorted_vector, 1)

    norm_vector = sorted_vector / norm   # Normalize by L1 norm

    # Compute the weighted sum using NumPy vectorized operations
    weight = (N - np.arange(1, N+1) + 0.5) / N
    gini_sum = np.dot(norm_vector, weight)

    gini = (1 - 2 * gini_sum) / (1 - 1/N)

    return norm*gini / N

from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import ticker

def addCircle(ax,x,y):
    x0=x+0.5 ; y0=y+0.5; r=0.4
    c = Circle((x0, y0), r,color='darkred')
    ax.add_patch(c)
    art3d.pathpatch_2d_to_3d(c, z=0, zdir="z")

def plot_density_matrix_3D(rho, title='Modulus of rho, colored by phase', elev=40,azim=30,
                           cmap_name='twilight_shifted', trans=None, ticks_reduction_factor=1,
                           z_axis_res=3, colBar = True, filename = None, circles=None):
    
    nofqubits = int(np.log2(rho.shape[0]))
    
    _x = np.arange(rho.shape[0])
    _y = np.arange(rho.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    y, x = _xx.ravel(), _yy.ravel()
    
    # we plot the absolute value
    top = np.abs(rho).flatten()
    bottom = 0.0 #np.zeros_like(top).flatten()
    w=0.8
    d=0.8
    
    mpl_cyclic_names = ['twilight','twilight_shifted','hsv']
    oth_cyclic_names = ['cmocean_phase','hue_L60','erdc_iceFire','nic_Edge','colorwheel','cyclic_mrybm','cyclic_mygbm']

    if cmap_name in mpl_cyclic_names:
       cm_map = plt.colormaps[cmap_name]
    elif cmap_name in oth_cyclic_names:
       # load the requested color map; we need to change this to avoid hard coded paths
       script_dir = Path(__file__).resolve().parent
       path = script_dir.parent / 'colormaps' / 'cyclic'
       cmap_name = cmap_name + ".txt"
       cm_data = np.loadtxt(path / cmap_name)
       cm_map = LinearSegmentedColormap.from_list(cmap_name, cm_data)
    else:
       cm_map = plt.colormaps['twilight_shifted']

    # the quantity used for the colormap
    phase = np.angle(rho).flatten() * 180.0/np.pi
    norm = plt.Normalize(-180, 180)              
    cols = cm_map(norm(phase))     
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #ax.xaxis.set_major_formatter(StrMethodFormatter(bin_strf))
    ax.xaxis.set_ticks(np.arange(0, 2**nofqubits+1, ticks_reduction_factor))
    x_lbls = ax.get_xticks().tolist()
    x_lbls[-1]=x_lbls[-1]-1
    for i in range(len(x_lbls)):
       x_lbls[i] = format(x_lbls[i],'0'+str(nofqubits)+'b')
    ax.set_xticklabels(x_lbls)
    ax.tick_params(axis='x', labelrotation = -25)

    #ax.yaxis.set_major_formatter(StrMethodFormatter(bin_strf))
    ax.yaxis.set_ticks(np.arange(0, 2**nofqubits+1, ticks_reduction_factor))
    y_lbls = ax.get_yticks().tolist()
    y_lbls[-1]=y_lbls[-1]-1
    for i in range(len(y_lbls)):
       y_lbls[i] = format(y_lbls[i],'0'+str(nofqubits)+'b')
    ax.set_yticklabels(y_lbls)
    ax.tick_params(axis='y', labelrotation = 47.5)

    #mask = top.nonzero()
    mask = np.ones_like(x,dtype=bool) # plot everything
    p = ax.bar3d(x[mask],y[mask],bottom,w,d,top[mask], color=cols[mask], alpha=trans, cmap=cm_map)
    ax.view_init(elev=elev,azim=azim)
    ax.set_title(title)
    # plot circles where measures have to be done
    if circles is not None:
        for c in circles:
            addCircle(ax,c[0],c[1])

    # plot the colorbar
    if colBar == True: 
        cbar = fig.colorbar(p, cmap=cm_map, pad=0.2, ticks=[0,1/8,1/4,3/8,1/2,5/8,3/4,7/8,1])
        cbar.ax.set_yticklabels(['-$\pi$', '-3$\pi$/4', '-$\pi$/2 ', '-$\pi$/4', '0', '$\pi$/4', '$\pi$/2', '3$\pi$/4', '$\pi$',])

    z_lbls = ax.get_zticks().tolist()
    max_val = float(z_lbls[-1])
    ax.zaxis.set_major_locator(ticker.FixedLocator(np.linspace(0,1.01*max_val,5)))

    z_lbls = ax.get_zticks().tolist()
    for i in range(len(z_lbls)):
       z_lbls[i] = format(z_lbls[i],'.'+str(z_axis_res)+'f')
    z_lbls[0] = ''
    print(z_lbls)
    ax.set_zticklabels(z_lbls)
    ax.tick_params(axis='z', labelrotation = 45)
    
    plt.xticks(va='center', ha='left')
    plt.yticks(va='center', ha='right')
            
    
    if filename:
        plt.savefig(filename)
    
    plt.show()

def plot_density_matrix_3D_real(rho,
                           title='Real part of rho',
                           elev=40,azim=30, cmap_name='Blues', trans=None):
    _x = np.arange(rho.shape[0])
    _y = np.arange(rho.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    y, x = _xx.ravel(), _yy.ravel()
    
    # we plot the real part of rho
    top = np.real(rho).flatten()
    bottom = np.zeros_like(top).flatten()
    w=0.8
    d=0.8
    
    # the quantity used for the colormap
    #phase = np.angle(rho).flatten() * 180.0/np.pi
    # a colormap
    norm = plt.Normalize(-1, 1)        
    cmap = plt.colormaps[cmap_name]
    cols = cmap(norm(top))         
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)
    ax.bar3d(x,y,bottom,w,d,top, color=cols, alpha=trans)
    ax.view_init(elev=elev,azim=azim)
    ax.set_title(title)
        
    # plot the colorbar
    colorMap = plt.cm.ScalarMappable(norm, cmap=cmap_name)
    #colorMap.set_array(phase)        
    colBar = plt.colorbar(colorMap,pad=0.2)
        
    plt.show()

def plot_density_matrix_3D_imag(rho,
                           title='Imaginary part of rho',
                           elev=40,azim=30, cmap_name='Greens', trans=None):
    _x = np.arange(rho.shape[0])
    _y = np.arange(rho.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    y, x = _xx.ravel(), _yy.ravel()
    
    # we plot the imaginary part of rho
    top = np.imag(rho).flatten()
    bottom = np.zeros_like(top).flatten()
    w=0.8
    d=0.8
    
    # the quantity used for the colormap
    #phase = np.angle(rho).flatten() * 180.0/np.pi
    # a colormap
    norm = plt.Normalize(-1, 1)        
    cmap = plt.colormaps[cmap_name]
    cols = cmap(norm(top))         
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)
    ax.bar3d(x,y,bottom,w,d,top, color=cols, alpha=trans)
    ax.view_init(elev=elev,azim=azim)
    ax.set_title(title)
        
    # plot the colorbar
    colorMap = plt.cm.ScalarMappable(norm, cmap=cmap_name)
    #colorMap.set_array(phase)        
    colBar = plt.colorbar(colorMap,pad=0.2)
        
    plt.show()

def plot_density_matrix_2D(rho):

    fig, ax = plt.subplots(1,2)
    mod   = np.abs(rho)
    phase = np.angle(rho) * 180.0/np.pi
    
    im1 = ax[0].imshow(mod)
    ax[0].set_title("modulus")
    fig.colorbar(im1,ax=ax[0],shrink=0.7)

    cmap_name='twilight_shifted'
    #norm = plt.Normalize(-180, 180)
    #print("min/max",np.min(phase),np.max(phase))
    #cmap = plt.colormaps[cmap_name]
    #cols = cmap(norm(phase))
    #cols = cmap(phase)              
    im2 = ax[1].imshow(phase,cmap=cmap_name, vmin=-180, vmax=180)    
    ax[1].set_title("phase (deg)")

    fig.colorbar(im2,ax=ax[1],shrink=0.7)
    
    plt.tight_layout()
    plt.show()

