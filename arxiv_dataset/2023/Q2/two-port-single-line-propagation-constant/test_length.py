import numpy as np
import matplotlib.pyplot as plt

def compute_lambd(ereff, f, lengths):
    # normalized eigenvalue without the factor kappa (the network ratio).
    c0 = 299792458
    gamma = 2*np.pi*f/c0*np.sqrt(-ereff+1j*np.finfo(complex).eps)    
    inx = np.tril_indices(len(lengths), k=-1)  # get indecies of unique pairs 
    # num_pairs = N*(N-2)*(N**2-1)/8
    lambd = []
    for g in gamma:
        vij   = np.exp(-g*(lengths[inx[0]]-lengths[inx[1]])) - np.exp(g*(lengths[inx[0]]-lengths[inx[1]]))
        z = vij*np.exp(-g*(lengths[inx[0]]+lengths[inx[1]]))
        y = vij*np.exp(g*(lengths[inx[0]]+lengths[inx[1]]))
        W = (np.outer(y,z) - np.outer(z,y)).conj()
        lambd.append(np.sum(abs(W)**2)/2)
    lambd = np.array(lambd)
    return lambd/max(lambd)

if __name__=='__main__':
    c0 = 299792458
    ereff = 1 - 0.00001j
    f = np.linspace(1, 22, 512)*1e9
    gamma = 2*np.pi*f/c0*np.sqrt(-ereff+1j*np.finfo(complex).eps)
    
    lengths = np.array([0, 7, 22, 27, 28, 31, 39, 41, 57, 64])*3e-3

    plt.figure(dpi=150)
    plt.plot(f*1e-9, compute_lambd(ereff, f, lengths), lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized eigenvalue')
    plt.ylim(0,1.2)
    plt.title('Normalized eigenvalue without considering the factor kappa')
    
    plt.show()