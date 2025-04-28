# Author: Bill Pengyuan Zhai. Harvard University. Yelin Group. Oct 2022

import numpy as np
import torch
import torch.nn as nn
import math


#===============================Object classes=====================================
jc = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

class R_l(nn.Module):
    def __init__(self, conn, N):
        self.a0 = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.a1 = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.a2 = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.a3 = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.a4 = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.a5 = nn.Parameter(torch.tensor(math.pi) * torch.rand(1))
        self.conn, self.N = conn, N 

    def forward(self):
        A_l = torch.zeros((2 * self.N, 2 * self.N), dtype=torch.cfloat)
        A_l[]

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
        print('debug self.L', self.L)
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
        print('debug self.V_l_module_list len', len(self.V_l_module_list))
        print('debug self.V_l_module_list.parameters()', self.V_l_module_list.parameters())
        for param in self.V_l_module_list.parameters(): # this loops through L*4 number of parameters
            print('param', param)

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



class Network_general(nn.Module):
