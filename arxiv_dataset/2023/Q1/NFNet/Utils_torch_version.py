# Author: Bill Pengyuan Zhai. Harvard University. Yelin Group. Oct 2022

import numpy as np
import torch
import torch.nn as nn
import math
from scipy import sparse

# For loading MNIST
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def mix_rbf_kernel(x, y, sigma_list): # Kernel matrix
    ndim = x.ndim
    if ndim == 1:
        exponent = (x[:, None] - y[None, :])**2 # Get pair-wise differences organized in a matrix
        print('exponent', exponent)
    elif ndim == 2:
        exponent = ((x[:, None, :] - y[None, :, :])**2).sum(axis=2) #(512, 1, 6) (1, 512, 6)-->(512, 512, 6)
    else:
        raise
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + np.exp(-gamma * exponent) # kernel matrix exponentiate each element
    return K

# Related to calculating the MMD gradient
def kernel_expect(K, px, py):
    '''
    expectation value of kernel function.
    
    Args:
        px (1darray): the first PDF.
        py (1darray): the second PDF.
        
    Returns:
        float: kernel expectation.
    '''
    #print('debug px', px)
    #print('debug py', py)
    #print('kernel', K)
    return px@K@py.T

#===============================Util functions=====================================

def get_nn_pairs(geometry):
    '''define pairs that cnot gates will apply.'''
    num_bit = np.prod(geometry)
    if len(geometry) == 2:
        nrow, ncol = geometry
        res = []
        for ij in range(num_bit):
            i, j = ij // ncol, ij % ncol
            res.extend([(ij, i_ * ncol + j_)
                        for i_, j_ in [((i + 1) % nrow, j), (i, (j + 1) % ncol)]])
        return res
    elif len(geometry) == 1:
        res = []
        for inth in range(2):
            for i in range(inth, num_bit, 2):
                res = res + [(i, i_ % num_bit) for i_ in range(i + 1, i + 2)]
        return res
    else:
        raise NotImplementedError('')

def binary_basis(geometry): # complete probability space
    num_bit = np.prod(geometry)
    M = 2**num_bit
    x = np.arange(M)
    return unpacknbits(x[:,None], num_bit).reshape((-1,)+geometry)

def unpacknbits(arr, nbit, axis=-1):
    '''unpack numbers to bitstrings.'''
    nd = np.ndim(arr)
    if axis < 0:
        axis = nd + axis
    return (((arr & (1 << np.arange(nbit - 1, -1, -1)).reshape([-1] + [1] * (nd - axis - 1)))) > 0).astype('int8')

#===========================BAS dataset related==================================

def thin_barstripe_pdf(geometry, n_fermions):
    '''get bar and stripes PDF'''
    x = binary_basis(geometry) # All 3 by 3 image patterns
    pl = is_thin_bs(x, n_fermions)
    return pl/pl.sum()

# Create the bars and stripe pdf as data probability distribution
def barstripe_pdf(geometry):
    '''get bar and stripes PDF'''
    x = binary_basis(geometry) # All 3 by 3 image patterns
    pl = is_bs(x)
    return pl/pl.sum()

def barstripe_counts(geometry):
    '''get bar and stripes PDF'''
    x = binary_basis(geometry) # All 3 by 3 image patterns
    pl = is_bs(x)
    return pl

def is_thin_bs(samples, n_fermions):
    return ( (np.abs(np.diff(samples,axis=-1)).sum(axis=(1,2))==0)\
           & (samples.sum(axis=(1,2))==n_fermions) )|( (np.abs(np.diff(samples,axis=1)).sum(axis=(1,2))==0)\
           & (samples.sum(axis=(1,2))==n_fermions) ) # the or operator!

def is_bs(samples):
    # the input samples has a shape (2^n_qubits, height, width)
    '''a sample is a bar or a stripe.'''
    return (np.abs(np.diff(samples,axis=-1)).sum(axis=(1,2))==0\
           )|((np.abs(np.diff(samples, axis=1)).sum(axis=(1,2)))==0) # the or operator!


def make_bas_thin(geometry): # assume geometry is square
    img = np.zeros(geometry)[None, :]
    n_b_or_s = geometry[0]
    bar_samples = img.repeat(n_b_or_s, axis=0)
    # First make bar images these index lists are i and j indices respectively
    bar_samples[[i for i in range(n_b_or_s)], [i for i in range(n_b_or_s)]]=1
    stripe_samples = np.transpose(bar_samples, axes=[0,2,1])
    return bar_samples, stripe_samples

def make_bas(geometry): # make the regular BAS images, aka, all valid patterns extracted from the basis
    p_bas = barstripe_pdf(geometry)
    print('debug np.nonzer', np.nonzero(p_bas))
    valid_ind = np.nonzero(p_bas)[0].squeeze() # returns the indices where the probability is non-zero
    x_basis_m_n = binary_basis(geometry) # numpy array
    return x_basis_m_n[valid_ind]

#===============================MNIST PDF==========================================
def mnist_single_pdf(digit, ind, size):
    #MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    label_inds = mnist.targets == digit
    data_digit = mnist.data[label_inds][ind]

    # Crop and downscale
    trans0 = transforms.CenterCrop(22)
    trans1 = transforms.Resize(size)
    trans = transforms.Compose([trans0, trans1])
    data_digit_sub = trans(data_digit.unsqueeze(0)) # need to add one channel for MNIST in order to use the transformation
    freq_data = data_digit_sub[0] # Then get rid of the channel dimension
    #freq_data = np.heaviside(freq_data-freq_data.mean(),freq_data) # Do some filtering to make the image more clear

    #freq_data += np.random.rand(size*size) #1
    q_data = freq_data/freq_data.sum()
    return q_data, data_digit

def mnist_single_binary(digit, ind, size):
    # test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #                    transforms.ToTensor(),
    #                    transforms.Normalize((0.1307,), (0.3081,)),
    #                    lambda x: x>0,
    #                    lambda x: x.float(),
    #                ])),
    # batch_size=1, shuffle=True)

    #MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    label_inds = mnist.targets == digit
    data_digit = mnist.data[label_inds][ind]

    # Crop and downscale
    trans0 = transforms.CenterCrop(22)
    trans1 = transforms.Resize(size)
    trans2 = transforms.Lambda(lambda x: x>0)
    trans3 = transforms.Lambda(lambda x: x.float())
    trans = transforms.Compose([trans0, trans1, trans2, trans3])

    data_digit_sub = trans(data_digit.unsqueeze(0)) # need to add one channel for MNIST in order to use the transformation
    freq_data = data_digit_sub[0] # Then get rid of the channel dimension
    #freq_data = np.heaviside(freq_data-freq_data.mean(),freq_data) # Do some filtering to make the image more clear

    #freq_data += np.random.rand(size*size) #1
    q_data = freq_data/freq_data.sum()
    return q_data, data_digit

    # x, _ = test_loader.dataset[7777]
    # return x
#===============================Object classes=====================================
jc = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

class V_l(nn.Module):
    def __init__(self, conn, N):
        super().__init__()
        self.bii = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.bjj = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.bij_real = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.bij_img = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))

        self.conn, self.N = conn, N 

    # In a variational setting the values of the b_ij's should be conveniently resettable
    def forward(self):
        B_l = torch.zeros((self.N, self.N), dtype=torch.cfloat)
        i,j = self.conn
        # This
        B_l[i,i], B_l[j,j], B_l[i,j], B_l[j,i]= self.bii, self.bjj, self.bij_real+jc*self.bij_img, self.bij_real-jc*self.bij_img

        V_l = torch.matrix_exp(jc*B_l) # This might potentially be a wrong scipy result
        return V_l

# A network that is composed of an array of V_l objects
class Network(nn.Module):
    # Given a list of i-j connections, initialize each layer's V_l object (Note, their parameters values are not set yet!)
    def __init__(self, conn_list, N):
        super().__init__()

        self.N = N
        self.L = len(conn_list)
        # print('debug self.L', self.L)
        # Make sure that the parameters in V_l_list and the V_tot are always updated together! 
        # Currently, they are empty
        V_l_list = [] # A list of modules
        for l in range(self.L):
            V_l_list.append(V_l(conn_list[l], N))

        # Convert to modulelist
        self.V_l_module_list = nn.ModuleList(V_l_list)
        self.V_tot = torch.eye(self.N, dtype=torch.cfloat)

    # Assign parameter values into each V_l block
    # params_m is a numpy nd array of shape (L, 4), where L is the total number of layers
    # After setting the parameters, update the new total V_tot matrix
    # Supposingly this function is called whenever the parameters are updated at each iteration
    # This is reserved for the case when we don't use torch.autograd at the optimization.
    def manual_set_params(self, params_m): # Assume params_m has shape (L, 4)
        l = 0
        # print('debug self.V_l_module_list len', len(self.V_l_module_list))
        # print('debug self.V_l_module_list.parameters()', self.V_l_module_list.parameters())
        # for param in self.V_l_module_list.parameters(): # this loops through L*4 number of parameters
            # print('param', param)

        for param in self.V_l_module_list.parameters(): # each param is a 4-element real tensor
            param.data = params_m[l//4, l%4]
            l = l + 1
        return 
         
    def get_V_tild(self, y, x):
        # First get the corresponding rows to be considered
        #print('y, x', (y,x))
        assert len(x) == self.N and len(y) == self.N, 'Wrong input or output length'
        assert self.V_tot is not None, 'self.V_tot is None'
        rows = [i for i in range(self.N) if x[i] == 1]
        cols = [i for i in range(self.N) if y[i] == 1]
        #print('rows, cols', (rows, cols))
        V_tild = self.V_tot[rows, :][:, cols] # This is the fastest way to find the submatrix given a csr matrix
        #print('debug V_tild', V_tild)
        return V_tild    

    def forward(self, y, x): # y is a batch of output states on all qubit measurements, and x is a batch of input states
        self.V_tot = torch.eye(self.N, dtype=torch.cfloat) # Note: creating a sparse csr matrix is more expensive than creating numpy arrays
        for l in range(self.L): 
            #print('before multiply', self.V_l_list[l].get_V_l())
            V_l = self.V_l_module_list[l]() # call forward() on a subgate module returns the actual V_l matrix
            self.V_tot = self.V_tot@(V_l)
            #print('after multiply, V_tot', self.V_tot)
        
        b_size = y.shape[0]

        V_tild_list = []
        for b in range(b_size):
            V_tild = self.get_V_tild(y[b], x[b])
            V_tild_list.append(V_tild[None, :])

        # print('debug V_tild_list', V_tild_list)
        V_tild_3d = torch.concat(V_tild_list, axis=0)
        amps = torch.det(V_tild_3d) # amplitudes (batch,)
        return amps  

    def forward_partial_observation(self, y, x, subsystem_mask): # subsystem mask is which qubits to measure y from
        self.V_tot = torch.eye(self.N, dtype=torch.cfloat) # Note: creating a sparse csr matrix is more expensive than creating numpy arrays
        for l in range(self.L): 
            #print('before multiply', self.V_l_list[l].get_V_l())
            V_l = self.V_l_module_list[l]() # call forward() on a subgate module returns the actual V_l matrix
            self.V_tot = self.V_tot@(V_l)
            #print('after multiply, V_tot', self.V_tot)
        
        # print('debug self.V_tot', self.V_tot)
        b_size = y.shape[0]

        V_tild_list = []
        for b in range(b_size): # Use a loop for now until I found a parallel solution to this
            V_tild = self.get_V_tild(subsystem_mask[b], x[b]) # The inputs follow col, row indexing
            V_tild_list.append(V_tild[None, :])

        V_tild_3d = torch.concat(V_tild_list, axis=0) # has shape(batch, l, k)
        
        V_tild_dagger_3d = torch.transpose(torch.conj(V_tild_3d), dim0=1, dim1=2)

        M_batch = []
        for b in range(b_size):
            L = torch.sum(x[b]) # L is the total number of fermions in the input string x
            K = y[b].shape[0] # K is the number of subsystem qubits to measure from 
            M = torch.zeros((2*(L+K), 2*(L+K)), dtype=torch.cfloat)
            temp = torch.arange(K)*2

            #reset!
            Mi_indices = L+temp + y[b]
            Mj_indices = L+2*K+torch.arange(L)
            #M[Mi_indices, :][:, Mj_indices] = torch.transpose(V_tild_3d[b], dim0=0, dim1=1)
            # print('y[b]', y[b])
            # print('debug V_tild_3d[b]', V_tild_3d[b])
            # print('debug Mi_indices', Mi_indices)
            # print('debug Mj_indices', Mj_indices)
            M[ Mi_indices[:,None], Mj_indices ] = torch.transpose(V_tild_3d[b], dim0=0, dim1=1)
            # print('flag0 M', M)

            #reset!
            Mi_indices = torch.arange(L-1, -1, -1)
            Mj_indices = L+temp+torch.ones(K, dtype=torch.int)-y[b]
            # print('debug Mi_indices, Mi_indices')
            # print(Mi_indices)
            # print(Mj_indices)
            #M[Mi_indices, :][:, Mj_indices] = torch.transpose(V_tild_dagger_3d[b], dim0=0, dim1=1)
            M[ Mi_indices[:,None], Mj_indices ] = torch.transpose(V_tild_dagger_3d[b], dim0=0, dim1=1)
            # print('flag1', M)

            #reset!
            Mi_indices = torch.arange(L-1, -1, -1)
            Mj_indices = L+2*K+torch.arange(L)
            #M[Mi_indices, :][:, Mj_indices] = torch.eye(L, dtype=torch.cfloat)
            M[ Mi_indices[:,None], Mj_indices ] = torch.eye(L, dtype=torch.cfloat)
            # print('flag2', M)

            #reset! Note some cases have i > j which shall be excluded
            yb_flip_bit_mask = (torch.ones(K, dtype=torch.int)-y[b]).bool() # an int tensor 1, 0, 1... converted to booleans
            
            select_i_indices = L + temp.masked_select(yb_flip_bit_mask) # such as L+ 0, 4, 8
            # print('debug  select_i_indices', select_i_indices)
            # maket it 1, 5, 9
            select_j_indices = select_i_indices+1
            # print('debug  select_j_indices', select_j_indices)

            M[select_i_indices, select_j_indices] = 1.0


            # Mi_indices = L+temp + y[b]
            # Mj_indices = L+temp + torch.ones(K, dtype=torch.int)-y[b]
            # #M[Mi_indices, :][:, Mj_indices] = torch.eye(K, dtype=torch.cfloat)
            # M[ Mi_indices[:,None], Mj_indices ] = torch.eye(K, dtype=torch.cfloat)
            # print('flag3 Mi_indices', Mi_indices)
            # print('flag3 Mj_indices', Mj_indices)
            # print('flag3 M', M)

            # Now the complete M should be anti-symmetric
            M = M - torch.transpose(M, dim0=0, dim1=1)
            # print('flag4 M', M)

            M_batch.append(M[None,:])

        # Calculate the determinants in batch
        # print('debug M_batch', M_batch[0].shape)
        M_batch = torch.concat(M_batch, axis=0)
        # print('debug torch.det(M_batch)', torch.det(M_batch))
        probs = torch.sqrt( torch.det(M_batch) ) # The Pfaffian is directly the probability
        return probs

# For the Fermion-preserving evolution
def initialize_sparse(N, conn_i, conn_j, a, b, c, d): # 
    # print('debug a, b, c, d)',(a, b, c, d))
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sy = sparse.csr_matrix(np.array([[0 , -1j], [1j , 0]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    # A list of single x, y, z operator applied on the site i_site
    sx_list = []
    sy_list = []
    sz_list = []
    for i_site in range(N):
        x_ops = [id] * N
        y_ops = [id] * N
        z_ops = [id] * N
        x_ops[i_site] = sx
        y_ops[i_site] = sy
        z_ops[i_site] = sz
        X = x_ops[0]
        Y = y_ops[0]
        Z = z_ops[0]
        for j in range(1, N):
            X = sparse.kron(X, x_ops[j], 'csr')
            Y = sparse.kron(Y, y_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sy_list.append(Y)
        sz_list.append(Z)
    # H_zz = sparse.csr_matrix((2**N, 2**N))
    # H_xx = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_yy = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_xy = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_yx = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_z = sparse.csr_matrix(np.zeros((2**N, 2**N)),dtype=np.complex128)
    #H_zzzi = sparse.csr_matrix((2**N, 2**N),dtype=np.complex128)

    # H_zz = H_zz + sz_list[i] * sz_list[j]
    H_xx = sx_list[conn_i] @ sx_list[conn_j]
    H_yy = sy_list[conn_i] @ sy_list[conn_j]
    H_xy = sx_list[conn_i] @ sy_list[conn_j]
    H_yx = sy_list[conn_i] @ sx_list[conn_j]

    H_zzz_id = sparse.csr_matrix(np.eye(2**N), dtype=np.complex128) # This is a diagonal of real numbers
    for i in range(conn_i, conn_j):
        H_zzz_id = sz_list[i] @  H_zzz_id
    
    H_z = - (a/2) * sz_list[conn_i] - (b/2) * sz_list[conn_j]

    H = H_z + (  (1.0j*c/2) * H_xy - (1.0j*c/2) * H_yx + (d*(1.0j)/2) * H_xx + (d*(1.0j)/2) * H_yy  ) @ H_zzz_id
    
    # print('H_z', H_z)
    # print('H_xy',H_xy)
    # print('H_yx',H_yx)
    # print('H_xx',H_xx)
    # print('H_yy',H_yy)

    # print('debug H', H) 
    # print('debug iH', 1j* H)
    return H.todense()

