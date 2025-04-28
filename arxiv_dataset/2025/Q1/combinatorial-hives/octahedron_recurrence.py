import argparse
import numpy as np
from scipy.stats import unitary_group
import pickle
from tqdm import tqdm
import plotly.graph_objects as go

def generate_GUE_matrix(size):
    # Real and imaginary parts of the off-diagonal elements
    real_part = np.random.normal(0, np.sqrt(1/2), (size, size))
    imag_part = np.random.normal(0, np.sqrt(1/2), (size, size))
    # Initialize the complex matrix
    matrix = real_part + 1j * imag_part
    # Adjust diagonal elements: real with variance 1
    for i in range(size):
        matrix[i, i] = np.random.normal(0, 1)
    # Make the matrix Hermitian
    gaussian_unitary_ensemble = (matrix + matrix.conj().T) / 2
    return gaussian_unitary_ensemble

def generate_Random_Projection(size):
    # Create diagonal matrix with half entries 1's and other half 0's
    D = np.diag(np.concatenate((np.ones(size//2),np.zeros(size-size//2))))
    # Create random unitary matrix
    Q = unitary_group.rvs(size)
    # Generate random projection matrix
    random_projection = Q @ D @ Q.conj().T
    return random_projection

def generate_Sequential_Spectrum_Matrix(size):
    # Create diagonal matrix with entries 0,1,2,...,size-1
    D = np.diag(np.arange(size))
    # Create random unitary matrix
    Q = unitary_group.rvs(size)
    # Generate sequential spectrum matrix
    sequential_spectrum_matrix = Q @ D @ Q.conj().T
    return sequential_spectrum_matrix

def generate_Alternating_Sequential_Spectrum_Matrix(size):
    # Create diagonal matrix with entries size,size-2,size-4,...,2,0,-2,-4,...,-(size-2)
    D = np.diag(np.arange(size,-size,-2))
    # Create random unitary matrix
    Q = unitary_group.rvs(size)
    # Generate sequential spectrum matrix
    alternating_sequential_spectrum_matrix = Q @ D @ Q.conj().T
    return alternating_sequential_spectrum_matrix

def octahedron_recurrence(X,Y,LA_diff=1000):
    # Generate the Gelfand-Tsetlin patterns
    GT_mu = {}
    GT_nu = {}
    GT_la = {}
    for i in range(N):
        Z = X[:N-i,:N-i]
        eigvals = np.linalg.eigvalsh(Z)[::-1]
        for j in range(N-i):
            GT_mu[(i,j)] = eigvals[j]
    for i in range(N):
        Z = Y[:N-i,:N-i]
        eigvals = np.linalg.eigvalsh(Z)[::-1]
        for j in range(N-i):
            GT_nu[(i,j)] = eigvals[j]
    for i in range(1):
        for j in range(N-i):
            GT_la[(i,j)] = LA_diff*(N-j)
    # Fill up the values in the double hive H using GT patterns
    H = {}
    H[(0,N)] = 0
    for i in range(1,N+1):
        H[(i,N)] = H[i-1,N] + GT_la[(0,i-1)]
    for i in range(N):
        for j in range(N-i):
            H[(i,N-j-1)] = H[(i,N-j)] + GT_mu[(i,j)]
    for j in range(N):
        for i in range(j,N):
            H[(i+1,j)] = H[(i,j)] + GT_nu[(j,i-j)]
    # Fill up the upper double hive on the tetrahedron T
    T = {}
    for key in H.keys():
        i = key[0]
        j = key[1]
        if i >= j:
            T[(0,i-j,j,N-i)] = H[key]
        else:
            T[(j-i,0,i,N-j)] = H[key]
    # Perform octahedron recurrence
    for i in range(N-1):
        for j in range(i+1):
            for k in range(N-i-1):
                x,y,z,w = (i-j,j,k,N-i-k-2)
                a = T[(x+1,y,z+1,w)]
                b = T[(x+1,y,z,w+1)]
                c = T[(x,y+1,z,w+1)]
                d = T[(x,y+1,z+1,w)]
                e = T[(x,y,z+1,w+1)]
                f = max(a + c, b + d) - e
                T[(x+1,y+1,z,w)] = f
    # Obtain the double hive on the bottom of the tetrahedron
    H_oct = {}
    for i in range(N+1):
        for j in range(N+1):
            if i+j <= N:
                H_oct[(i,j)] = T[(j,i,0,N-i-j)]
            else:
                H_oct[(i,j)] = T[(N-i,N-j,i+j-N,0)]
    # Obtain height map of double hive (top (S) and bottom (S_oct))
    S = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            S[i,j] = H[(i,j)]
    S_oct = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            S_oct[i,j] = H_oct[(i,j)]
    # Visualize only one triangle (set other triangle values to NAN)
    # for i in range(N+1):
    #     for j in range(N+1):
    #         if i+j > N:
    #             S_oct[i,j] = np.nan
    return H, H_oct, T, S, S_oct


if __name__ == '__main__':

    # Create argparse for the parameters
    parser = argparse.ArgumentParser(description='Use octahedron recurrence for sampling hives')
    parser.add_argument('--matrix_type',choices=['GUE','RP','SSM','ASSM'],default='GUE',help='Type of matrix to use for minor process')
    parser.add_argument('--M', type=int, default=1000, help='Number of matrices')
    parser.add_argument('--N', type=int, default=250, help='Size of the matrix')
    parser.add_argument('--sig_X', type=float, default=1, help='Scaling factor for X')
    parser.add_argument('--sig_Y', type=float, default=1, help='Scaling factor for Y')
    parser.add_argument('--LA_diff', type=float, default=1000, help='Difference factor in large lambda')
    args = parser.parse_args()

    matrix_type = args.matrix_type
    M = args.M
    N = args.N
    sig_X = args.sig_X
    sig_Y = args.sig_Y
    LA_diff = args.LA_diff

    generate_matrix_dict = {'GUE':generate_GUE_matrix, 
                            'RP':generate_Random_Projection, 
                            'SSM':generate_Sequential_Spectrum_Matrix,
                            'ASSM':generate_Alternating_Sequential_Spectrum_Matrix}
    generate_matrix_func = generate_matrix_dict[matrix_type]

    stack_H = []
    stack_H_oct = []
    stack_S = []
    stack_S_oct = []
    mean_T = {}
    for i in tqdm(range(M),leave=False):
        # Generate the matrices X and Y
        X = sig_X * generate_matrix_func(N)
        Y = sig_Y * generate_matrix_func(N)
        H, H_oct, T, S, S_oct = octahedron_recurrence(X,Y,LA_diff)
        stack_H.append(H)
        stack_H_oct.append(H_oct)
        stack_S.append(S)
        stack_S_oct.append(S_oct)
        if i==0:
            for key in T.keys():
                mean_T[key] = 0
        for key in T.keys():
            mean_T[key] += T[key]
    for key in mean_T.keys():
        mean_T[key] /= M
    with open('{}_mean_T_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.pickle'.format(matrix_type,M,sig_X,sig_Y), 'wb') as handle:
        pickle.dump(mean_T, handle, protocol=pickle.HIGHEST_PROTOCOL)
    S = np.array(stack_S)
    S_oct = np.array(stack_S_oct)
    mean_S = np.mean(stack_S,axis=0)
    mean_S_oct = np.mean(stack_S_oct,axis=0)
    var_S = np.var(stack_S,axis=0)
    var_S_oct = np.var(stack_S_oct,axis=0)

    np.save('{}_S_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format(matrix_type,M,sig_X,sig_Y),S)
    np.save('{}_S_oct_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format(matrix_type,M,sig_X,sig_Y),S_oct)
    np.save('{}_mean_S_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format(matrix_type,M,sig_X,sig_Y),mean_S)
    np.save('{}_mean_S_oct_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format(matrix_type,M,sig_X,sig_Y),mean_S_oct)
    np.save('{}_var_S_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format(matrix_type,M,sig_X,sig_Y),var_S)
    np.save('{}_var_S_oct_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format(matrix_type,M,sig_X,sig_Y),var_S_oct)

    for i in range(N+1):
        for j in range(N+1):
            if i+j > N:
                mean_S_oct[i,j] = np.nan
                var_S_oct[i,j] = np.nan

    # Plot the mean of height map of the bottom double hive
    fig = go.Figure(data=[go.Surface(z=mean_S_oct, x=np.arange(0,N+1), y=np.arange(0,N+1))])
    fig.update_layout(title='Mean value of Bottom Hive of the Tetrahedron ({} Matrices)'.format(matrix_type),
                        scene = dict(
                            xaxis_title='i',
                            yaxis_title='j',
                            zaxis_title='Mean of H_oct(i,j)'),
                            autosize=False,
                            width=800, height=800,
                            margin=dict(l=65, r=50, b=65, t=90))
    # fig.show()
    fig.write_html("{}_mean_oct_recc_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.html".format(matrix_type,M,sig_X,sig_Y))

    # Plot the variance of height map of the bottom double hive
    fig = go.Figure(data=[go.Surface(z=var_S_oct, x=np.arange(0,N+1), y=np.arange(0,N+1))])
    fig.update_layout(title='Variance value of Bottom Hive of the Tetrahedron ({} Matrices)'.format(matrix_type),
                        scene = dict(
                            xaxis_title='i',
                            yaxis_title='j',
                            zaxis_title='Variance of H_oct(i,j)'),
                            autosize=False,
                            width=800, height=800,
                            margin=dict(l=65, r=50, b=65, t=90))
    # fig.show()
    fig.write_html("{}_var_oct_recc_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.html".format(matrix_type,M,sig_X,sig_Y))

